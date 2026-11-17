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
import io
import zipfile
import traceback

VENDOR=Path(__file__).resolve().parents[1]/'vendor'
sys.path.insert(0,str(VENDOR/'src'))
from jove_visualizer.script_parser import parse_steps_from_text
from jove_visualizer.config import load_settings
from jove_visualizer.retrieval import CorpusIndex
from jove_visualizer.pipeline import match_steps
from jove_visualizer.reporting import results_to_json,results_to_excel
from .storage import DriveStore
from .library import Library
from .providers import ModelGateway,generate_step_audio,VisionBudgetExceeded
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

    def recent(self,owner):
        projects=[]
        for path in self.root.glob('*/state.json'):
            if not re.fullmatch('[a-f0-9]{32}',path.parent.name):continue
            try:
                state=json.loads(path.read_text())
                if state.get('owner')==owner:projects.append(state)
            except (OSError,ValueError):continue
        return sorted(projects,key=lambda s:s.get('created',0),reverse=True)

    def resume(self,ident,owner,config):
        with self.lock:
            state=self.state(ident,owner)
            if state['status'] not in {'failed','needs_review'}:raise ValueError('This project is not paused or failed.')
            if any(not f.done() for f in self.futures.values()):raise ValueError('Wait for the current project to finish.')
            folder=self.root/ident
            text=(folder/'input.txt').read_text(encoding='utf-8')
            state.update(status='queued',stage='recovery',error=None,issues=[],updated=time.time())
            write_json(folder/'state.json',state)
            self.futures[ident]=self.executor.submit(run,folder,owner,text,config)
        return ident

    def resolve_timing(self,ident,owner,config):
        with self.lock:
            state=self.state(ident,owner)
            if state['status']!='needs_review' or state.get('completed_visual_steps')!=state.get('step_count'):
                raise ValueError('Finish visual review before resolving timing.')
            if state.get('timing_attempted'):raise ValueError('The timing pass is complete. Unresolved footage requires review.')
            if any(not f.done() for f in self.futures.values()):raise ValueError('Wait for the current project to finish.')
            folder=self.root/ident
            state.update(status='queued',stage='timing',error=None,updated=time.time())
            write_json(folder/'state.json',state)
            self.futures[ident]=self.executor.submit(run_timing,folder,owner,config)
        return ident

    def diagnostics(self,ident,owner):
        self.state(ident,owner)
        folder=self.root/ident
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
            for name in ['state.json','report.json','failure.json','model-requests.json']:
                path=folder/name
                if path.is_file():archive.write(path,name)
            # Only local media-processing logs, never settings or credentials.
            logs=sorted(folder.rglob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:20]
            for path in logs:
                archive.writestr(str(path.relative_to(folder)),path.read_bytes()[-65536:])
        return output.getvalue()

def run(folder,owner,text,config):
    state=json.loads((folder/'state.json').read_text());store=None;remote=None;index=None
    def checkpoint(stage,status='running',**details):
        state.update(stage=stage,status=status,updated=time.time(),**details);write_json(folder/'state.json',state)
        print(f"[job {state['id'][:8]}] {stage}: {details.get('detail',status)}",flush=True)
        if store and remote:store.upload(folder/'state.json',remote)
    try:
        store=DriveStore(config['google_service_account'])
        remote=state.get('output_folder_id') or store.folder(config['OUTPUT_FOLDER_ID'],'job-'+state['id'])
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
        if (folder/'matches.json').exists():
            saved=json.loads((folder/'matches.json').read_text())
            if len(saved)!=len(steps) or any(row.get('script_step')!=step.text or row.get('step_number')!=step.number for row,step in zip(saved,steps)):
                raise ValueError('Saved matches do not match this script; start a new project.')
            checkpoint('matching',detail='Reusing saved timestamp matches')
        else:
            results=match_steps(index,steps,settings,api_key=config['OPENAI_API_KEY'],progress=matching_progress)
            results_to_json(results,folder/'matches.json');results_to_excel(results,folder/'matches.xlsx')
            store.upload(folder/'matches.json',remote);store.upload(folder/'matches.xlsx',remote)
        checkpoint('narration')
        voice=config['voice'];audios=[]
        for i,step in enumerate(steps):
            path=folder/f'narration-{i+1:03}.wav'
            if path.exists() and path.with_suffix('.json').exists():
                info=json.loads(path.with_suffix('.json').read_text())
                from .tts_preprocessor import prepare_script_for_tts
                spoken='\n\n'.join(prepare_script_for_tts(step.text)[0])
                if info.get('spoken')!=spoken or info.get('voice_id')!=voice['id'] or info.get('generation_speed')!=float(voice['speed']):
                    raise ValueError('Saved narration uses a different script or voice profile; choose its original profile or start a new project.')
            else:
                info=generate_step_audio(step.text,step.previous_text,step.next_text,voice['id'],float(voice['speed']),
                                         config['ELEVENLABS_API_KEY'],path)
                store.upload(path,remote);store.upload(path.with_suffix('.json'),remote)
            audios.append({**info,'path':str(path)})
            checkpoint('narration',completed_audio_steps=i+1)
        checkpoint('visual_verification')
        def vision_progress(done,limit):
            checkpoint('visual_verification',vision_requests=done,vision_request_limit=limit)
        gateway=ModelGateway(config['OPENAI_API_KEY'],config.get('VISION_MODEL','gpt-4.1-mini'),int(config.get('MAX_VISION_CALLS',80)),
                             cache_dir=folder/'review-cache',progress=vision_progress)
        matcher=Selector(index,library,gateway,folder/'evidence')
        matches=json.loads((folder/'matches.json').read_text());shots=[];issues=[]
        completed=min(len(steps),int(state.get('completed_visual_steps',0)))
        if completed and (folder/'report.json').exists():
            report=json.loads((folder/'report.json').read_text())
            if not gateway.prior_calls:gateway.prior_calls=report.get('model_requests',[])
            prior={s.number:s.text for s in steps[:completed]}
            shots=report.get('selected',[])
            if any(s.get('step') not in prior or s.get('text')!=prior[s['step']] for s in shots):
                raise ValueError('Saved visual selections do not match this script.')
            issues=[x for x in report.get('issues',[]) if x.get('step') in prior]
            matcher.attempts=report.get('attempts',[])
        for i,(step,row,audio) in enumerate(zip(steps,matches,audios)):
            if i<completed:continue
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
            write_json(folder/'report.json',{'issues':issues,'selected':shots,'attempts':matcher.attempts,'model_requests':gateway.prior_calls+gateway.calls,
                'review_method':'Sampled-frame AI review; no human visual approval is implied.'})
            store.upload(folder/'report.json',remote)
            checkpoint('visual_verification',completed_visual_steps=i+1)
        tempos=[s['tempo'] for s in shots]
        for i in pacing_outliers(tempos):issues.append({'step':shots[i]['step'],'reason':'Abrupt adjacent narration tempo change; select a different range or review.'})
        write_json(folder/'report.json',{'issues':issues,'selected':shots,'attempts':matcher.attempts,'model_requests':gateway.prior_calls+gateway.calls,
            'trained_ranking_enabled':False,'reviewer_memory_enabled':False,
            'review_method':'Sampled-frame AI review; complete playback review remains required.'})
        write_json(folder/'audio-adjustments.json',[{'step':s['step'],'original_audio_seconds':s['audio_duration'],
            'selected_video_seconds':s['end']-s['start'],'tempo':s['tempo'],'review':s['needs_timing_review']} for s in shots])
        for name in ['report.json','audio-adjustments.json']:store.upload(folder/name,remote)
        if issues:
            checkpoint('review','needs_review',issues=issues,detail='Visual review is complete. Resolve the flagged timing or footage issues before rendering.');return
        # Reconstruct local cache paths rather than trusting stale absolute paths.
        for shot in shots:
            shot['path']=str(library.video(shot['video_id']))
            shot['audio_path']=audios[next(i for i,s in enumerate(steps) if s.number==shot['step'])]['path']
        checkpoint('assembly')
        verification=render(shots,folder)
        checkpoint('delivery',verification=verification)
        for name in ['preview.mp4','editable-package.zip','edit_plan.json']:store.upload(folder/name,remote)
        checkpoint('complete','complete',verification=verification,
            note='Draft generated. Review scientific fit, visual continuity and narration playback before production use.')
    except VisionBudgetExceeded:
        issues.append({'step':step.number,'reason':'Visual review request limit reached. Further steps are unverified. Continue the saved project or review its candidates.'})
        write_json(folder/'report.json',{'issues':issues,'selected':shots,'attempts':matcher.attempts,'model_requests':gateway.prior_calls+gateway.calls,
            'review_method':'Incomplete sampled-frame review; no complete edit or scientific approval is implied.'})
        store.upload(folder/'report.json',remote)
        checkpoint('visual_verification','needs_review',issues=issues,error=None,
                   detail='Visual review paused at its request limit. Matching and narration are saved.')
    except Exception as e:
        # Avoid exposing provider messages, request bodies, or credential-bearing URLs.
        message=str(e) if isinstance(e,(ValueError,PermissionError)) else f'{type(e).__name__}: processing stopped; inspect the private server log.'
        state.update(status='failed',error=message,updated=time.time());write_json(folder/'state.json',state)
        # Record locations without provider response bodies or credential-bearing URLs.
        trace=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}
               for f in traceback.extract_tb(e.__traceback__)]
        failure={'exception':type(e).__name__,'stage':state.get('stage'),'trace':trace}
        if 'gateway' in locals():failure['vision_requests']=len(gateway.calls)
        write_json(folder/'failure.json',failure)
        print(f"[job {state['id'][:8]}] failed: {json.dumps(failure)}",flush=True)
        if store and remote:
            try:
                store.upload(folder/'state.json',remote)
                store.upload(folder/'failure.json',remote)
                if (folder/'report.json').exists():store.upload(folder/'report.json',remote)
            except Exception:pass
    finally:
        if index:index.connection.close()


def run_timing(folder,owner,config):
    """Reconcile only flagged durations, retaining narration and earlier matching work."""
    from .reconciliation import reconcile_step
    from .timing import fit_audio
    state=json.loads((folder/'state.json').read_text());store=None;remote=state['output_folder_id']
    if state['owner']!=owner:raise PermissionError('This project belongs to another user')
    report=json.loads((folder/'report.json').read_text())
    backup=folder/'report-before-timing.json'
    if not backup.exists():write_json(backup,report)
    def checkpoint(detail,status='running',stage='timing',**details):
        state.update(status=status,stage=stage,detail=detail,updated=time.time(),**details)
        write_json(folder/'state.json',state)
        print(f"[job {state['id'][:8]}] {stage}: {detail}",flush=True)
        if store:store.upload(folder/'state.json',remote)
    def save_report():
        write_json(folder/'report.json',report)
        if store:store.upload(folder/'report.json',remote)
    try:
        store=DriveStore(config['google_service_account'])
        checkpoint('Loading saved selections and narration. No narration is being regenerated.')
        library=Library.from_folder(store,config['LIBRARY_FOLDER_ID'],folder.parent/'library-cache')
        steps=parse_steps_from_text((folder/'input.txt').read_text(encoding='utf-8'))
        stepmap={s.number:s for s in steps};shots=report['selected']
        if len(shots)!=len(steps) or len({s['step'] for s in shots})!=len(steps):
            raise ValueError('Some steps have no verified footage; resolve them before the timing pass.')
        if any(s['step'] not in stepmap or s['text']!=stepmap[s['step']].text for s in shots):
            raise ValueError('Saved selections do not match the script.')
        targets=[i for i,s in enumerate(shots) if fit_audio(s['audio_duration'],s['end']-s['start']).needs_review]
        checkpoint(f'Rechecking {len(targets)} duration mismatches against the existing narration.',timing_total=len(targets),timing_completed=0)
        gateway=ModelGateway(config['OPENAI_API_KEY'],config.get('VISION_MODEL','gpt-4.1-mini'),int(config.get('MAX_VISION_CALLS',80)),
                             cache_dir=folder/'timing-review'/'cache',
                             progress=lambda done,limit:checkpoint(state['detail'],vision_requests=done,vision_request_limit=limit))
        report.setdefault('timing_attempts',[])
        for done,i in enumerate(targets):
            shot=shots[i]
            chosen,attempts=reconcile_step(stepmap[shot['step']],shot,report.get('attempts',[]),shot['audio_duration'],
                library,gateway,folder/'timing-evidence',progress=checkpoint)
            report['timing_attempts'].extend(attempts)
            if chosen:shots[i]=chosen
            report['timing_model_requests']=gateway.prior_calls+gateway.calls
            save_report()
            checkpoint(f"Timing checked for step {shot['step']}: "+('compatible range found.' if chosen else 'no compatible complete action verified.'),timing_completed=done+1)
        # Replace timing flags only; preserve any unrelated scientific or missing-footage issue.
        issues=[x for x in report.get('issues',[]) if not x.get('reason','').startswith(('Narration adjustment','Abrupt adjacent narration tempo change','No duration-compatible'))]
        for shot in shots:
            decision=fit_audio(shot['audio_duration'],shot['end']-shot['start'])
            shot['tempo']=decision.tempo;shot['needs_timing_review']=decision.needs_review
            if decision.needs_review:
                issues.append({'step':shot['step'],'reason':'No duration-compatible complete action was verified. Review footage or revise the narration.',
                               'audio_seconds':round(shot['audio_duration'],2),'video_seconds':round(shot['end']-shot['start'],2),'tempo':round(shot['tempo'],4)})
        for i in pacing_outliers([s['tempo'] for s in shots]):
            issues.append({'step':shots[i]['step'],'reason':'Abrupt adjacent narration tempo change; select a different range or review.'})
        report['issues']=issues;report['timing_pass_complete']=True
        save_report()
        adjustments=[{'step':s['step'],'original_audio_seconds':s['audio_duration'],'selected_video_seconds':s['end']-s['start'],
                      'tempo':s['tempo'],'review':s['needs_timing_review']} for s in shots]
        write_json(folder/'audio-adjustments.json',adjustments);store.upload(folder/'audio-adjustments.json',remote)
        state['timing_attempted']=True
        if issues:
            checkpoint('Timing reconciliation finished. The remaining steps need footage or narration review.',status='needs_review',stage='review',issues=issues);return
        checkpoint('Timing is compatible. Rendering the video draft.',stage='assembly',issues=[])
        for i,shot in enumerate(shots):
            shot['path']=str(library.video(shot['video_id']))
            shot['audio_path']=str(folder/f'narration-{next(j for j,s in enumerate(steps) if s.number==shot["step"])+1:03}.wav')
        verification=render(shots,folder)
        for name in ['preview.mp4','editable-package.zip','edit_plan.json']:store.upload(folder/name,remote)
        checkpoint('Draft rendered. Review scientific fit and complete playback before production use.',status='complete',stage='complete',verification=verification)
    except Exception as exc:
        trace=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
        failure={'exception':type(exc).__name__,'stage':state.get('stage'),'trace':trace}
        write_json(folder/'failure.json',failure)
        print(f"[job {state['id'][:8]}] failed: {json.dumps(failure)}",flush=True)
        state.update(status='failed',error=f'{type(exc).__name__}: timing processing stopped. Download private diagnostics.',updated=time.time())
        write_json(folder/'state.json',state)
        if store:
            try:store.upload(folder/'state.json',remote);store.upload(folder/'failure.json',remote)
            except Exception:pass

