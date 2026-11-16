from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid

VENDOR=Path(__file__).resolve().parents[1]/'vendor'
sys.path.insert(0,str(VENDOR/'src'))
from jove_visualizer.script_parser import parse_steps_from_text
from jove_visualizer.config import load_settings
from jove_visualizer.retrieval import CorpusIndex
from jove_visualizer.pipeline import match_steps
from jove_visualizer.reporting import results_to_json,results_to_excel
from .storage import DriveStore
from .library import Library
from .providers import ModelGateway,generate_step_audio
from .selection import Selector
from .timing import pacing_outliers
from .render import render

def write_json(path,value):
    path=Path(path);temp=path.with_suffix('.temporary')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)

class Jobs:
    """Single-worker pilot. Job access is checked by authenticated owner on every read."""
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.executor=ThreadPoolExecutor(max_workers=1);self.futures={};self.lock=threading.Lock()

    def submit(self,owner,text,config,reference=None):
        if not text.strip():raise ValueError('No script supplied')
        if len(text)>50000:raise ValueError('Pilot script limit is 50,000 characters')
        with self.lock:
            if any(not f.done() for f in self.futures.values()):raise ValueError('The pilot is processing a job. Wait until it finishes before starting another.')
            ident=uuid.uuid4().hex;folder=self.root/ident;folder.mkdir()
            (folder/'input.txt').write_text(text,encoding='utf-8')
            if reference:(folder/'reference.xlsx').write_bytes(reference)
            write_json(folder/'state.json',{'id':ident,'owner':owner,'status':'queued','stage':'queued','created':time.time()})
            self.futures[ident]=self.executor.submit(run,folder,owner,text,config)
        return ident

    def state(self,ident,owner):
        if not re.fullmatch('[a-f0-9]{32}',ident):raise ValueError('Invalid job')
        state=json.loads((self.root/ident/'state.json').read_text())
        if state['owner']!=owner:raise PermissionError('This project belongs to another user')
        return state

    def file(self,ident,owner,name):
        self.state(ident,owner)
        if name not in {'preview.mp4','editable-package.zip','report.json','matches.xlsx','matches.json','state.json','audio-adjustments.json'}:
            raise PermissionError('Unsupported download')
        return self.root/ident/name

def run(folder,owner,text,config):
    state=json.loads((folder/'state.json').read_text());store=None;remote=None;index=None
    def checkpoint(stage,status='running',**details):
        state.update(stage=stage,status=status,updated=time.time(),**details);write_json(folder/'state.json',state)
        print(f"[job {state['id'][:8]}] {stage}: {details.get('detail',status)}",flush=True)
        if store and remote:store.upload(folder/'state.json',remote)
    try:
        store=DriveStore(config['google_service_account'])
        remote=store.folder(config['OUTPUT_FOLDER_ID'],'job-'+state['id'])
        store.upload(folder/'input.txt',remote)
        checkpoint('library',output_folder_id=remote)
        library=Library.from_folder(store,config['LIBRARY_FOLDER_ID'],folder.parent/'library-cache')
        settings=load_settings(VENDOR/'config.json')
        settings.openai_model=config.get('TEXT_MODEL','gpt-5-mini')
        # The pilot never deserializes a remotely supplied trained Python object.
        settings.learned_ranker_enabled=False;settings.reviewer_feedback_enabled=False
        settings.openai_max_concurrency=2
        settings.openai_timeout_seconds=60
        settings.openai_max_retries=2
        checkpoint('model_setup',detail='Loading the caption model and prepared title cache')
        index=CorpusIndex(library.index(),settings)
        steps=parse_steps_from_text(text)
        if not steps:raise ValueError('No numbered script steps found')
        if len(steps)>20:raise ValueError('Pilot limit: 20 script steps')
        reference=[]
        if (folder/'reference.xlsx').exists():
            from .sheets import read_candidates
            reference=read_candidates((folder/'reference.xlsx').read_bytes())
            def normalized(s):return re.sub(r'\W+',' ',s).strip().lower()
            stepmap={s.number:normalized(s.text) for s in steps}
            if any(c['step'] not in stepmap or normalized(c['text'])!=stepmap[c['step']] for c in reference):
                raise ValueError('Reference workbook steps do not match this script. Use the correct workbook or leave the optional reference empty.')
        checkpoint('matching',step_count=len(steps),media_count=len(library.videos))
        def matching_progress(done,total,message):
            checkpoint('matching',matching_completed=done,matching_total=total,detail=message)
        results=match_steps(index,steps,settings,api_key=config['OPENAI_API_KEY'],progress=matching_progress)
        results_to_json(results,folder/'matches.json');results_to_excel(results,folder/'matches.xlsx')
        store.upload(folder/'matches.json',remote);store.upload(folder/'matches.xlsx',remote)
        checkpoint('narration')
        voice=config['voice'];audios=[]
        for i,step in enumerate(steps):
            path=folder/f'narration-{i+1:03}.wav'
            info=generate_step_audio(step.text,step.previous_text,step.next_text,voice['id'],float(voice['speed']),
                                     config['ELEVENLABS_API_KEY'],path)
            audios.append({**info,'path':str(path)})
            store.upload(path,remote);store.upload(path.with_suffix('.json'),remote)
            checkpoint('narration',completed_audio_steps=i+1)
        checkpoint('visual_verification')
        gateway=ModelGateway(config['OPENAI_API_KEY'],config.get('VISION_MODEL','gpt-4.1-mini'),int(config.get('MAX_VISION_CALLS',80)))
        matcher=Selector(index,library,gateway,folder/'evidence')
        matches=json.loads((folder/'matches.json').read_text());shots=[];issues=[]
        for i,(step,row,audio) in enumerate(zip(steps,matches,audios)):
            checkpoint('visual_verification',completed_visual_steps=i,total_visual_steps=len(steps),detail=f'Reviewing video frames for step {i+1} of {len(steps)}')
            options=[row.get('best_match'),*row.get('additional_timestamps_same_id',[]),*row.get('alternatives',[])]
            from .sheets import seconds
            suggested=[{'video_id':str(c['journal_id']),'start':seconds(c['start_timestamp']),
                        'end':seconds(c['end_timestamp']),'caption':c['caption_passage']} for c in options if c]
            refs=[c for c in reference if c['step']==step.number]
            rejected=[c for c in refs if re.search(r'not accurate|inaccurate|incorrect|reject|wrong',c['feedback'],re.I)]
            preferred=[c for c in refs if c not in rejected]
            suggested=preferred+[c for c in suggested if not any(c['video_id']==r['video_id'] and c['start']<r['end'] and c['end']>r['start'] for r in rejected)]
            chosen=matcher.select(step,suggested,audio['duration'])
            if not chosen:issues.append({'step':step.number,'reason':'No complete action verified in available footage.'})
            else:
                chosen.update(audio_path=audio['path'],audio_duration=audio['duration']);shots.append(chosen)
                if chosen['needs_timing_review']:issues.append({'step':step.number,'reason':'Narration adjustment is outside the pilot automatic range.','tempo':chosen['tempo']})
            write_json(folder/'report.json',{'issues':issues,'selected':shots,'attempts':matcher.attempts,'model_requests':gateway.calls,
                'review_method':'Sampled-frame AI review; no human visual approval is implied.'})
            checkpoint('visual_verification',completed_visual_steps=i+1)
        tempos=[s['tempo'] for s in shots]
        for i in pacing_outliers(tempos):issues.append({'step':shots[i]['step'],'reason':'Abrupt adjacent narration tempo change; select a different range or review.'})
        write_json(folder/'report.json',{'issues':issues,'selected':shots,'attempts':matcher.attempts,'model_requests':gateway.calls,
            'trained_ranking_enabled':False,'reviewer_memory_enabled':False,
            'review_method':'Sampled-frame AI review; complete playback review remains required.'})
        write_json(folder/'audio-adjustments.json',[{'step':s['step'],'original_audio_seconds':s['audio_duration'],
            'selected_video_seconds':s['end']-s['start'],'tempo':s['tempo'],'review':s['needs_timing_review']} for s in shots])
        for name in ['report.json','audio-adjustments.json']:store.upload(folder/name,remote)
        if issues:
            checkpoint('review','needs_review',issues=issues);return
        checkpoint('assembly')
        verification=render(shots,folder)
        checkpoint('delivery',verification=verification)
        for name in ['preview.mp4','editable-package.zip','edit_plan.json']:store.upload(folder/name,remote)
        checkpoint('complete','complete',verification=verification,
            note='Draft generated. Review scientific fit, visual continuity and narration playback before production use.')
    except Exception as e:
        # Avoid exposing provider messages, request bodies, or credential-bearing URLs.
        message=str(e) if isinstance(e,(ValueError,PermissionError)) else f'{type(e).__name__}: processing stopped; inspect the private server log.'
        state.update(status='failed',error=message,updated=time.time());write_json(folder/'state.json',state)
        if store and remote:
            try:store.upload(folder/'state.json',remote)
            except Exception:pass
    finally:
        if index:index.connection.close()

