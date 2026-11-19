from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor,as_completed
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
    def __init__(self,root,worker_config=None,external_worker=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.executor=ThreadPoolExecutor(max_workers=1);self.futures={};self.lock=threading.Lock()
        from .job_queue import JobQueue
        self.queue=JobQueue(self.root/'jobs.sqlite');self.worker_config=worker_config
        self.external_worker=external_worker;self.stop_event=threading.Event()
        if worker_config is not None:self.queue.activate_build(str(worker_config.get('ENGINE_BUILD','')))
        if worker_config is not None and not external_worker:
            self.worker_thread=threading.Thread(target=self.work_forever,daemon=True,name='studio-job-worker')
            self.worker_thread.start()

    def work_forever(self):
        worker=uuid.uuid4().hex;build=str((self.worker_config or {}).get('ENGINE_BUILD',''))
        while not self.stop_event.is_set():
            try:request=self.queue.claim(worker,build)
            except Exception:request=None
            if request is None:self.stop_event.wait(2);continue
            ident=request['id'];folder=self.root/ident;heartbeat_stop=threading.Event()
            def heartbeat():
                while not heartbeat_stop.wait(10):self.queue.heartbeat(ident,worker)
            beat=threading.Thread(target=heartbeat,daemon=True);beat.start()
            try:
                config={**(self.worker_config or {}),**request['config'],'ENGINE_BUILD':build}
                if request['kind']=='catalog':run_catalog(folder,request['owner'],config)
                else:run(folder,request['owner'],(folder/'input.txt').read_text(encoding='utf-8'),config)
                state=self.state(ident,request['owner'])
                self.queue.finish(ident,worker,state['status'])
            except Exception as exc:
                try:
                    state=json.loads((folder/'state.json').read_text())
                    state.update(status='failed',error='Processing stopped: '+type(exc).__name__,updated=time.time());write_json(folder/'state.json',state)
                finally:self.queue.finish(ident,worker,'failed')
            finally:heartbeat_stop.set();beat.join(timeout=1)

    def prepare_catalog(self,owner,config):
        if self.worker_config is None:raise ValueError('Library preparation requires the configured job worker.')
        ident=uuid.uuid4().hex;folder=self.root/ident;folder.mkdir()
        write_json(folder/'state.json',{'id':ident,'owner':owner,'kind':'catalog','status':'queued','stage':'library_preparation','created':time.time()})
        queued=self.queue.enqueue(ident,owner,'library-actions:'+config['LIBRARY_FOLDER_ID'],config,kind='catalog')
        if queued!=ident:(folder/'state.json').unlink();folder.rmdir()
        return queued

    def submit(self,owner,text,config,reference=None):
        if not text.strip():raise ValueError('No script supplied')
        if len(text)>50000:raise ValueError('Pilot script limit is 50,000 characters')
        with self.lock:
            if self.worker_config is None and any(not f.done() for f in self.futures.values()):raise ValueError('The pilot is processing a job. Wait until it finishes before starting another.')
            ident=uuid.uuid4().hex;folder=self.root/ident;folder.mkdir()
            (folder/'input.txt').write_text(text,encoding='utf-8')
            if reference:(folder/'reference.xlsx').write_bytes(reference)
            write_json(folder/'state.json',{'id':ident,'owner':owner,'status':'queued','stage':'queued','created':time.time()})
            if self.worker_config is not None:
                queued=self.queue.enqueue(ident,owner,text,config)
                if queued!=ident:
                    for path in folder.iterdir():path.unlink()
                    folder.rmdir();return queued
            else:self.futures[ident]=self.executor.submit(run,folder,owner,text,config)
        return ident

    def state(self,ident,owner):
        if not re.fullmatch('[a-f0-9]{32}',ident):raise ValueError('Invalid job')
        state=json.loads((self.root/ident/'state.json').read_text())
        if state['owner']!=owner:raise PermissionError('This project belongs to another user')
        return state

    def file(self,ident,owner,name):
        self.state(ident,owner)
        if name not in {'preview.mp4','editable-package.zip','report.json','matches.xlsx','matches.json','state.json','audio-adjustments.json','coverage.json','quality.json','edit_plan.json'}:
            raise PermissionError('Unsupported download')
        path=self.root/ident/name
        plan=path.parent/'edit_plan.json'
        if name=='audio-adjustments.json' and plan.exists() and (not path.exists() or path.stat().st_mtime<plan.stat().st_mtime):
            from .render import audio_adjustments_from_plan
            saved=json.loads(plan.read_text())
            write_json(path,saved['audio_adjustments'] if 'audio_adjustments' in saved else audio_adjustments_from_plan(saved))
        return path

    def recent(self,owner):
        projects=[]
        for path in self.root.glob('*/state.json'):
            if not re.fullmatch('[a-f0-9]{32}',path.parent.name):continue
            try:
                state=json.loads(path.read_text())
                if state.get('owner')==owner:projects.append(state)
            except (OSError,ValueError):continue
        return sorted(projects,key=lambda s:s.get('created',0),reverse=True)

    def review_step(self,ident,owner,step,config):
        state=self.state(ident,owner)
        if state['status'] not in {'complete','needs_review'}:raise ValueError('Wait for processing to finish before reviewing footage.')
        folder=self.root/ident;plan=json.loads((folder/'edit_plan.json').read_text())
        shots=[s for s in plan['shots'] if s['step']==step]
        if not shots:raise ValueError('No rendered footage exists for this step.')
        path=folder/'review-exclusions.json';excluded=json.loads(path.read_text()) if path.exists() else []
        for shot in shots:
            row={k:shot[k] for k in ['step','video_id','start','end']}
            if row not in excluded:excluded.append(row)
        write_json(path,excluded)
        if state['status']=='complete':
            state['status']='needs_review';write_json(folder/'state.json',state)
        return self.resume(ident,owner,config)

    def shot_preview(self,ident,owner,index):
        self.state(ident,owner)
        plan=json.loads((self.root/ident/'edit_plan.json').read_text())
        if not isinstance(index,int) or not 0<=index<len(plan['shots']):raise ValueError('Invalid shot')
        return self.root/ident/'media'/f'shot-{index+1:03}.mp4'

    def restore_latest(self,owner,config):
        """Restore only signed checkpoints for this invited owner after host loss."""
        from .checkpoints import valid
        store=DriveStore(config['google_service_account'])
        folders=[f for f in store.children(config['OUTPUT_FOLDER_ID']) if re.fullmatch(r'job-[a-f0-9]{32}',f['name'])]
        for remote in sorted(folders,key=lambda f:f.get('createdTime',''),reverse=True)[:20]:
            files=store.children(remote['id']);states=[f for f in files if f['name']=='state.json']
            if not states:continue
            item=max(states,key=lambda f:f.get('modifiedTime',f.get('createdTime','')))
            try:
                state=store.request('GET','https://www.googleapis.com/drive/v3/files/'+item['id'],params={'alt':'media','supportsAllDrives':'true'}).json()
                if state.get('owner')!=owner or not re.fullmatch('[a-f0-9]{32}',state.get('id','')) or remote['name']!='job-'+state['id'] or not valid(state,config):continue
                folder=self.root/state['id'];folder.mkdir(exist_ok=True)
                newest={}
                for f in sorted(files,key=lambda f:f.get('modifiedTime',f.get('createdTime',''))):newest[f['name']]=f
                if 'input.txt' not in newest:continue
                store.download(newest['input.txt']['id'],folder/'input.txt')
                if hashlib.sha256((folder/'input.txt').read_bytes()).hexdigest()!=state['input_sha256']:continue
                for name,f in newest.items():
                    if name in {'matches.json','matches.xlsx','report.json','quality.json','review-exclusions.json','preview.mp4','editable-package.zip','audio-adjustments.json','edit_plan.json'} or re.fullmatch(r'narration-\d{3}\.(?:wav|json)',name):
                        store.download(f['id'],folder/name)
                    elif name=='selections.json':
                        (folder/'automatic').mkdir(exist_ok=True);store.download(f['id'],folder/'automatic'/name)
                if state['status'] in {'running','queued'}:
                    state.update(status='needs_review',detail='The host restarted. Saved matching, narration and selections are restored; rebuild to continue.')
                state['restored_from_cloud']=True;write_json(folder/'state.json',state)
                return state['id']
            except (ValueError,KeyError,OSError):continue
        return None

    def resume(self,ident,owner,config):
        with self.lock:
            state=self.state(ident,owner)
            queued=self.queue.get(ident,owner)
            if queued and queued['status']=='running' and queued['lease_until']>time.time():raise ValueError('This project is still processing. Wait for the worker to finish.')
            if state['status'] not in {'failed','needs_review'}:raise ValueError('This project is not paused or failed.')
            if any(not f.done() for f in self.futures.values()):raise ValueError('Wait for the current project to finish.')
            folder=self.root/ident
            text=(folder/'input.txt').read_text(encoding='utf-8') if state.get('kind')!='catalog' else 'library-actions:'+config['LIBRARY_FOLDER_ID']
            state.update(status='queued',stage='recovery',error=None,issues=[],verification=None,
                         detail='Rebuilding from the saved script, matches and narration.',updated=time.time())
            write_json(folder/'state.json',state)
            if self.worker_config is not None:self.queue.enqueue(ident,owner,text,config,kind=state.get('kind','video'))
            else:self.futures[ident]=self.executor.submit(run,folder,owner,text,config)
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

    def full_subset(self,ident,owner,config):
        with self.lock:
            state=self.state(ident,owner)
            if state['status'] not in {'needs_review','failed'}:raise ValueError('Wait for the current review to finish.')
            if any(not f.done() for f in self.futures.values()):raise ValueError('A project is already processing.')
            folder=self.root/ident
            state.update(status='queued',stage='subset_search',error=None,full_subset_finished=False,verification=None,updated=time.time())
            write_json(folder/'state.json',state)
            self.futures[ident]=self.executor.submit(run_full_subset,folder,owner,config)
        return ident

    def requires_subset_recheck(self,ident,owner):
        self.state(ident,owner)
        from .reconciliation import VERIFICATION_VERSION
        from .coverage import VERSION
        try:
            saved=json.loads((self.root/ident/'coverage.json').read_text())
            return saved.get('version')!=VERSION or saved.get('verification_version')!=VERIFICATION_VERSION
        except (OSError,ValueError):return True

def run_catalog(folder,owner,config):
    """One-time library preparation, isolated from user script generation."""
    from .action_catalog import open_catalog
    state=json.loads((folder/'state.json').read_text());started=time.monotonic()
    if state['owner']!=owner:raise PermissionError('This project belongs to another user')
    def progress(done,total):
        state.update(status='running',stage='library_preparation',updated=time.time(),catalog_completed=done,catalog_total=total,
                     detail=f'Prepared reusable visual discovery for {done} of {total} videos. This is not a script match or scientific approval.')
        write_json(folder/'state.json',state)
    progress(0,0)
    try:
        store=DriveStore(config['google_service_account'])
        library=Library.from_folder(store,config['LIBRARY_FOLDER_ID'],folder.parent/'library-cache')
        catalog=open_catalog(library,store,config['OUTPUT_FOLDER_ID'])
        gateway=ModelGateway(config['OPENAI_API_KEY'],config.get('VISION_MODEL','gpt-4.1-mini'),160,
                             cache_dir=library.cache/'action-preparation-cache')
        gateway.deadline=time.monotonic()+1800
        catalog.build(library,gateway,progress=progress,persist=lambda p:store.upload(p,config['OUTPUT_FOLDER_ID']))
        state.update(status='complete',stage='library_preparation',updated=time.time(),catalog_completed=len(library.videos),
                     catalog_total=len(library.videos),detail='Visual action catalogue prepared. New scripts can retrieve these candidates without re-screening the whole library.',
                     processing_seconds=round(time.monotonic()-started,2));write_json(folder/'state.json',state)
    except Exception as exc:
        state.update(status='failed',stage='library_preparation',updated=time.time(),error=type(exc).__name__+': library preparation stopped. Completed videos are retained.',
                     detail='Retry library preparation to continue from completed videos.');write_json(folder/'state.json',state)
        raise


def prepare_narration(folder,step,position,voice,api_key):
    path=folder/f'narration-{position+1:03}.wav';generated=False
    if path.exists() and path.with_suffix('.json').exists():
        info=json.loads(path.with_suffix('.json').read_text())
        from .tts_preprocessor import prepare_script_for_tts
        spoken='\n\n'.join(prepare_script_for_tts(step.text)[0])
        if info.get('spoken')!=spoken or info.get('voice_id')!=voice['id'] or info.get('generation_speed')!=float(voice['speed']):
            raise ValueError('Saved narration uses a different script or voice profile; choose its original profile or start a new project.')
    else:
        info=generate_step_audio(step.text,step.previous_text,step.next_text,voice['id'],float(voice['speed']),api_key,path)
        generated=True
    return position,path,info,generated


def run(folder,owner,text,config):
    config={**config,'_PIPELINE_STARTED':time.monotonic(),'_REUSED_STAGES':[],'_STAGE_SECONDS':{}}
    state=json.loads((folder/'state.json').read_text());store=None;remote=None;index=None
    if state['owner']!=owner:raise PermissionError('This project belongs to another user')
    from .checkpoints import signature
    if not (folder/'input.txt').exists():(folder/'input.txt').write_text(text,encoding='utf-8')
    state['input_sha256']=hashlib.sha256((folder/'input.txt').read_bytes()).hexdigest()
    state['checkpoint_signature']=signature(state,config)
    last_flush=0.;last_remote_stage=None;stage_started=time.monotonic();timed_stage=None;narration_pool=None
    def checkpoint(stage,status='running',**details):
        nonlocal last_flush,last_remote_stage,stage_started,timed_stage
        if stage!=timed_stage:
            if timed_stage:config['_STAGE_SECONDS'][timed_stage]=round(config['_STAGE_SECONDS'].get(timed_stage,0)+time.monotonic()-stage_started,3)
            timed_stage=stage;stage_started=time.monotonic()
        state.update(stage=stage,status=status,updated=time.time(),stage_seconds=config['_STAGE_SECONDS'],**details);write_json(folder/'state.json',state)
        print(f"[job {state['id'][:8]}] {stage}: {details.get('detail',status)}",flush=True)
        if store and remote and (stage!=last_remote_stage or status!='running' or time.monotonic()-last_flush>=15):
            store.upload(folder/'state.json',remote);last_flush=time.monotonic();last_remote_stage=stage
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
        settings.openai_timeout_seconds=25
        settings.openai_max_retries=1
        checkpoint('model_setup',detail='Loading the caption model and prepared title cache')
        index=CorpusIndex(library.index(),settings)
        from .library_retrieval import restrict_index
        index=restrict_index(index,library.videos)
        steps=parse_steps_from_text(text)
        if not steps:raise ValueError('No numbered script steps found')
        if len(steps)>20:raise ValueError('Pilot limit: 20 script steps')
        if len({s.number for s in steps})!=len(steps):raise ValueError('Script step numbers must be unique.')
        reference=[]
        if (folder/'reference.xlsx').exists():
            from .sheets import read_candidates
            reference=read_candidates((folder/'reference.xlsx').read_bytes())
            def normalized(s):return re.sub(r'\W+',' ',s).strip().lower()
            stepmap={s.number:normalized(s.text) for s in steps}
            if any(c['step'] not in stepmap or normalized(c['text'])!=stepmap[c['step']] for c in reference):
                raise ValueError('Reference workbook steps do not match this script. Use the correct workbook or leave the optional reference empty.')
        # Narration and retrieval depend on the script, not on each other's
        # results. Generate speech alongside matching instead of adding latency.
        voice=config['voice'];audios=[None]*len(steps)
        narration_pool=ThreadPoolExecutor(max_workers=3)
        narration_futures=[narration_pool.submit(prepare_narration,folder,step,i,voice,config['ELEVENLABS_API_KEY']) for i,step in enumerate(steps)]
        checkpoint('matching',step_count=len(steps),media_count=len(library.videos),narration_started_in_parallel=True)
        def matching_progress(done,total,message):
            checkpoint('matching',matching_completed=done,matching_total=total,detail=message)
        if (folder/'matches.json').exists():
            saved=json.loads((folder/'matches.json').read_text())
            if len(saved)!=len(steps) or any(row.get('script_step')!=step.text or row.get('step_number')!=step.number for row,step in zip(saved,steps)):
                raise ValueError('Saved matches do not match this script; start a new project.')
            checkpoint('matching',detail='Reusing saved timestamp matches')
            config['_REUSED_STAGES'].append('matching')
        else:
            results=match_steps(index,steps,settings,api_key=config['OPENAI_API_KEY'],progress=matching_progress)
            results_to_json(results,folder/'matches.json');results_to_excel(results,folder/'matches.xlsx')
            store.upload(folder/'matches.json',remote);store.upload(folder/'matches.xlsx',remote)
        checkpoint('narration')
        for done,future in enumerate(as_completed(narration_futures),1):
            i,path,info,generated=future.result()
            if generated:store.upload(path,remote);store.upload(path.with_suffix('.json'),remote)
            else:config['_REUSED_STAGES'].append('narration-'+str(steps[i].number))
            audios[i]={**info,'path':str(path)}
            checkpoint('narration',completed_audio_steps=done,detail=f'Narration ready for {done} of {len(steps)} steps.')
        if config.get('SHORTLIST_PLUGIN'):
            from .plugin_project import package_project
            checkpoint('plugin_project', detail='Preparing shortlisted footage for the Premiere plugin.')
            bundle = package_project(folder, steps, audios, library)
            store.upload(bundle, remote)
            checkpoint('plugin_ready', 'complete', plugin_project_id=store.upload_ids[(remote, bundle.name)],
                       detail='The shortlisted project is ready for the desktop plugin.')
            return
        # Whole-library screening is an explicit recovery path. New scripts must
        # finish suggested timestamps and suggested videos before widening.
        if config.get('AUTOMATIC_EDITOR',True):
            run_automatic_edit(folder,steps,audios,index,library,config,checkpoint,store,remote)
            return
        if config.get('FULL_VISUAL_PIPELINE',False):
            checkpoint('subset_search',detail='Narration is saved. Starting the complete supplied-video search and action checks.')
            index.connection.close();index=None
            run_full_subset(folder,owner,config)
            return
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
        if narration_pool:narration_pool.shutdown(wait=True,cancel_futures=True)
        if index:index.connection.close()


def run_automatic_edit(folder,steps,audios,index,library,config,checkpoint,store,remote):
    from .automatic_edit import AutomaticEditor,VERSION
    from .action_render import render_actions, cut_schedule, TimingCoverageError
    from .action_catalog import open_catalog
    from .editorial import quality_report
    started=time.monotonic()
    checkpoint('visual_verification',detail='Checking suggested timestamps first. Suitable footage is retained while timing is solved separately.',
               editor_version=VERSION,engine_build=config.get('ENGINE_BUILD'),verification=None,total_visual_steps=len(steps),completed_visual_steps=0)
    gateway=ModelGateway(config['OPENAI_API_KEY'],config.get('VISION_MODEL','gpt-4.1-mini'),int(config.get('MAX_VISION_CALLS',80)),
                         cache_dir=folder/'automatic-cache')
    audio_map={step.number:audio for step,audio in zip(steps,audios)}
    def progress(message,**kw):
        checkpoint('visual_verification',detail=message,vision_requests=len(gateway.calls),**kw)
        store.upload(folder/'automatic'/'selections.json',remote)
        if (folder/'review-exclusions.json').exists():store.upload(folder/'review-exclusions.json',remote)
    catalog=None
    try:catalog=open_catalog(library,store,config['OUTPUT_FOLDER_ID'],index.embedder)
    except Exception as exc:checkpoint('visual_verification',detail='Prepared action catalogue unavailable; using caption retrieval and source storyboards.',catalog_error=type(exc).__name__)
    # Reserve two minutes for rendering/upload instead of spending the entire
    # ten-minute target on discovery. Actual wall time is measured in the report.
    remaining=480-(time.monotonic()-config.get('_PIPELINE_STARTED',started))
    editor=AutomaticEditor(index,library,gateway,folder/'automatic',steps,audio_map,progress=progress,
        seconds=max(1,min(int(config.get('VISUAL_REVIEW_SECONDS',300)),remaining)),workers=int(config.get('VISUAL_WORKERS',3)),action_catalog=catalog)
    if config.get('HOSTED_SHORTLIST'):
        from .sheets import read_candidates
        rows = read_candidates((folder/'matches.xlsx').read_bytes())
        plans = editor.run_shortlist(rows)
    else:
        plans=editor.run(json.loads((folder/'matches.json').read_text()))
    missing=[s.number for s in steps if s.number not in {p['step'] for p in plans}]
    issues=[{'step':p['step'],'reason':'; '.join(p['missing_actions']+p['unobservable_details']) or 'Illustrative footage requires review.'}
            for p in plans if p['needs_review']]
    issues.extend({'step':number,'reason':'No usable related footage was retained. This step is omitted from the partial review edit.'} for number in missing)
    report={'editor_version':VERSION,'selected':plans,'issues':issues,'omitted_steps':missing,
            'catalog_ready':bool(catalog is not None and catalog.ready),
            'catalog_sources':len(catalog.data['videos']) if catalog is not None else 0,
            'reused_stages':config.get('_REUSED_STAGES',[]),
            'stage_seconds':dict(config.get('_STAGE_SECONDS',{})),
            'attempts':editor.attempts,'model_requests':gateway.prior_calls+gateway.calls,
            'visual_seconds':round(time.monotonic()-started,2),'input_contract':'Plain script only. No reference timestamps or benchmark selections were supplied to this editor.',
            'review_method':('Sheet-only visual review with verified live action and narration-aligned cuts.'
                             if config.get('HOSTED_SHORTLIST') else 'Suggested ranges followed by wider retrieval and visual discovery.')}
    write_json(folder/'report.json',report);store.upload(folder/'report.json',remote)
    for plan in plans:
        if (not plan.get('scientific_audit_checked') or not plan.get('dense_checked')
                or any(shot.get('medium') != 'experimental_footage' or shot.get('evidence') != 'demonstrated'
                       for shot in plan['shots'])):
            issues.append({'step':plan['step'],'reason':'Live experimental footage demonstrating the action still needs verification.'})
    if plans and not issues:
        try:
            scheduled, _, _ = cut_schedule(plans, audio_map, plugin_cuts=bool(config.get('HOSTED_SHORTLIST')))
            for number in sorted({shot['step'] for shot in scheduled if shot['review']}):
                issues.append({'step':number,'reason':'The final cut exceeds the allowed playback speed or narration pause.'})
        except TimingCoverageError as exc:
            issues.append({'reason':str(exc)})
    if not plans or issues:
        report['issues'] = issues
        write_json(folder/'report.json',report);store.upload(folder/'report.json',remote)
        quality=quality_report(steps,plans,[],time.monotonic()-config.get('_PIPELINE_STARTED',started))
        write_json(folder/'quality.json',quality);store.upload(folder/'quality.json',remote)
        checkpoint('review','needs_review',issues=issues,verification=None,
                   detail='The edit needs verified live footage and natural timing for every step. See the saved review reasons.')
        return
    checkpoint('assembly',detail=f'Rendering {len(plans)} script steps using real source footage. Narration is retained; review flags stay visible.',issues=issues)
    result=render_actions(plans,audio_map,library,folder,plugin_cuts=bool(config.get('HOSTED_SHORTLIST')))
    for row in json.loads((folder/'audio-adjustments.json').read_text()):
        if not .85<=row['video_speed']<=1.151:
            issues.append({'step':row['step'],'reason':f"Footage plays at {row['video_speed']:.2f}x to retain the complete selected action and natural narration. Review pacing."})
        if row.get('additional_viewing_seconds',0)>.3:
            issues.append({'step':row['step'],'reason':f"The selected action requires {row['additional_viewing_seconds']:.1f} extra seconds of viewing after narration; review pacing or select a more concise action."})
    result['review_only']=bool(issues);result['omitted_steps']=missing
    elapsed=round(time.monotonic()-config.get('_PIPELINE_STARTED',started),2)
    quality=quality_report(steps,plans,json.loads((folder/'audio-adjustments.json').read_text()),elapsed)
    result['review_only']=bool(issues) or not quality['automated_checks_passed']
    write_json(folder/'quality.json',quality)
    report.update(issues=issues,verification=result,quality=quality,processing_seconds=elapsed,draft_contains_placeholders=False)
    write_json(folder/'report.json',report)
    for name in ['report.json','preview.mp4','editable-package.zip','edit_plan.json','audio-adjustments.json','quality.json']:
        store.upload(folder/name,remote)
    checkpoint('review' if result['review_only'] else 'complete','needs_review' if result['review_only'] else 'complete',
               detail=f"MP4 and editable timeline rendered with real footage for {len(plans)} of {len(steps)} steps. No placeholder cards. Review the listed uncertainties before production use.",
               issues=issues,verification=result,timing_attempted=True,full_subset_finished=True,completed_visual_steps=len(steps))


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


def run_full_subset(folder,owner,config):
    """Search every video's full timeline, including videos absent from the caption index."""
    from .coverage import screen_video,suggested_candidates,VERSION
    from .reconciliation import reconcile_step,VERIFICATION_VERSION
    state=json.loads((folder/'state.json').read_text())
    if state['owner']!=owner:raise PermissionError('This project belongs to another user')
    store=None;remote=state['output_folder_id'];coverage=None;gateway=None
    def checkpoint(detail,status='running',stage='subset_search',**extra):
        state.update(status=status,stage=stage,detail=detail,updated=time.time(),**extra)
        write_json(folder/'state.json',state)
        print(f"[job {state['id'][:8]}] {stage}: {detail}",flush=True)
        if store:store.upload(folder/'state.json',remote)
    def save_coverage():
        write_json(folder/'coverage.json',coverage)
        if store:store.upload(folder/'coverage.json',remote)
    try:
        store=DriveStore(config['google_service_account'])
        checkpoint('Loading every supplied video. Captions are not required for this search.')
        library=Library.from_folder(store,config['LIBRARY_FOLDER_ID'],folder.parent/'library-cache')
        if not library.videos:raise ValueError('The media library is empty')
        library.restore_overviews()
        text=(folder/'input.txt').read_text(encoding='utf-8')
        steps=parse_steps_from_text(text);signature=hashlib.sha256(text.encode()).hexdigest()
        if not steps:raise ValueError('The saved script has no numbered steps')
        if (folder/'coverage.json').exists():coverage=json.loads((folder/'coverage.json').read_text())
        if not coverage or coverage.get('script_sha256')!=signature or coverage.get('version')!=VERSION:
            coverage={'version':VERSION,'script_sha256':signature,'videos':{},'verified_steps':{},'complete':False,
                      'method':'Full-duration storyboard screening every three seconds, then dense verification of candidate actions. This is not frame-by-frame playback.'}
        expected={str(vid):item['sha256'] for vid,item in library.videos.items()}
        # A changed library invalidates coverage and any selections derived from it.
        if coverage.get('source_hashes')!=expected:
            coverage.update(source_hashes=expected,videos={},verified_steps={},complete=False)
        if coverage.get('verification_version')!=VERIFICATION_VERSION:
            if coverage.get('verified_steps'):write_json(folder/f'coverage-before-verification-v{VERIFICATION_VERSION}.json',coverage)
            coverage.update(verification_version=VERIFICATION_VERSION,verified_steps={})
        save_coverage()
        backup=folder/'report-before-full-subset.json'
        if not backup.exists() and (folder/'report.json').exists():backup.write_bytes((folder/'report.json').read_bytes())
        gateway=ModelGateway(config['OPENAI_API_KEY'],config.get('VISION_MODEL','gpt-4.1-mini'),int(config.get('MAX_VISION_CALLS',80)),
                             cache_dir=folder/'subset-review'/'cache',progress=lambda done,limit:checkpoint(state['detail'],vision_requests=done,vision_request_limit=limit))
        count=len(library.videos)
        for number,(vid,item) in enumerate(sorted(library.videos.items()),1):
            old=coverage['videos'].get(str(vid))
            if old and old.get('source_sha256')==item['sha256'] and old.get('status')=='screened_by_storyboards':continue
            checkpoint(f'Preparing full-timeline visual evidence for video {number} of {count}.',coverage_completed=len(coverage['videos']),coverage_total=count)
            path=None if library.has_prepared_overview(vid) else library.video(vid)
            overview=screen_video(vid,path,item['sha256'],steps,gateway,folder.parent/'library-cache'/'full-overviews',progress=checkpoint)
            coverage['videos'][str(vid)]=overview;save_coverage()
            checkpoint(f'Full-timeline storyboard screening completed for {len(coverage["videos"])} of {count} videos.',coverage_completed=len(coverage['videos']),coverage_total=count)
        if set(coverage['videos'])!=set(expected):raise ValueError('Incomplete media coverage; final selections are blocked')
        coverage['complete']=True;save_coverage()
        checkpoint(f'All {count} videos screened across their full timelines. Verifying precise candidate actions.',coverage_complete=True,coverage_completed=count,coverage_total=count)
        candidates=[c for v in coverage['videos'].values() for c in v['candidates']]
        rejected=[]
        if (folder/'reference.xlsx').exists():
            from .sheets import read_candidates
            refs=read_candidates((folder/'reference.xlsx').read_bytes())
            stepmap={s.number:re.sub(r'\W+',' ',s.text).strip().lower() for s in steps}
            if any(r['step'] not in stepmap or re.sub(r'\W+',' ',r['text']).strip().lower()!=stepmap[r['step']] for r in refs):
                raise ValueError('Reviewed timestamps do not match this script.')
            rejected=[c for c in refs if re.search(r'not accurate|inaccurate|incorrect|reject|wrong',c['feedback'],re.I)]
            for c in refs:
                if c not in rejected and c['video_id'] in library.videos:candidates.append({**c,'tier':'reviewed_timestamp','confidence':1.,'critical_uncertainties':[]})
        protocol=[{'step':s.number,'text':s.text} for s in steps]
        preferred={}
        if (folder/'matches.json').exists():
            from .sheets import seconds
            original_rows=json.loads((folder/'matches.json').read_text())
            candidates.extend(suggested_candidates(original_rows,set(library.videos)))
            for row in original_rows:
                options=[row.get('best_match'),*row.get('additional_timestamps_same_id',[]),*row.get('alternatives',[])]
                preferred[row['step_number']]=[{'video_id':str(c['journal_id']),'start':seconds(c['start_timestamp']),'end':seconds(c['end_timestamp'])} for c in options if c]
        candidates=[c for c in candidates if not any(c['step']==r['step'] and c['video_id']==r['video_id'] and c['start']<r['end'] and c['end']>r['start'] for r in rejected)]
        def priority(candidate):
            if candidate.get('tier')=='reviewed_timestamp':return -2
            if candidate.get('tier')=='suggested_timestamp':return -1
            options=preferred.get(candidate['step'],[])
            if any(candidate['video_id']==c['video_id'] and candidate['start']<c['end'] and candidate['end']>c['start'] for c in options):return 0
            if any(candidate['video_id']==c['video_id'] for c in options):return 1
            return 2
        issues=[];shots=[];attempts=[]
        for i,step in enumerate(steps):
            audio_path=folder/f'narration-{i+1:03}.wav';info_path=audio_path.with_suffix('.json')
            if not audio_path.exists() or not info_path.exists():raise ValueError('Saved narration is missing; finish narration before full-subset recovery.')
            audio=json.loads(info_path.read_text())
            from .tts_preprocessor import prepare_script_for_tts
            if audio.get('spoken')!='\n\n'.join(prepare_script_for_tts(step.text)[0]):raise ValueError('Saved narration does not match this script')
            saved=coverage['verified_steps'].get(str(step.number))
            if saved is None:
                ranked=sorted([c for c in candidates if c['step']==step.number],key=lambda c:(priority(c),bool(c.get('critical_uncertainties')),-c['confidence']))
                # Avoid repeated near-identical windows; no fixed per-video shortlist cap.
                options=[]
                for c in ranked:
                    if any(c['video_id']==o['video_id'] and abs(c['start']-o['start'])<3 for o in options):continue
                    options.append(c)
                chosen=None
                for option in options:
                    base={**option,'step':step.number,'text':step.text,'audio_path':str(audio_path),'audio_duration':audio['duration']}
                    chosen,observed=reconcile_step(step,base,[],audio['duration'],library,gateway,folder/'subset-evidence',progress=checkpoint,protocol=protocol,caption_lookup=library.caption_context)
                    attempts.extend(observed)
                    if chosen:break
                saved={'selection':chosen,'attempts':[a for a in attempts if a.get('step')==step.number]}
                coverage['verified_steps'][str(step.number)]=saved;save_coverage()
            if saved['selection']:shots.append(saved['selection'])
            else:issues.append({'step':step.number,'reason':f'No scientifically compatible complete action with natural narration timing was verified after screening all {count} supplied videos.'})
            report={'issues':issues,'selected':shots,'attempts':[a for v in coverage['verified_steps'].values() for a in v['attempts']],
                    'model_requests':gateway.prior_calls+gateway.calls,'coverage_complete':True,'coverage_total':count,
                    'review_method':'Full-duration sparse storyboard screening, then dense sampled-frame verification. Final scientific and playback review remains required.'}
            write_json(folder/'report.json',report);store.upload(folder/'report.json',remote)
            checkpoint(f'Precise action verification completed for {i+1} of {len(steps)} steps.',subset_steps_completed=i+1,subset_steps_total=len(steps))
        for i in pacing_outliers([s['tempo'] for s in shots]):issues.append({'step':shots[i]['step'],'reason':'Adjacent narration pacing requires review.'})
        report['issues']=issues;write_json(folder/'report.json',report);store.upload(folder/'report.json',remote)
        if issues:
            checkpoint(f'All {count} videos screened. Rendering a review draft with clearly marked cards for unresolved steps.',stage='assembly',issues=issues)
            for shot in shots:shot['path']=str(library.video(shot['video_id']))
            audios=[]
            for i,step in enumerate(steps):
                wav=folder/f'narration-{i+1:03}.wav';info=json.loads(wav.with_suffix('.json').read_text())
                audios.append({'path':str(wav),'duration':info['duration']})
            from .render import render_review_draft
            verification=render_review_draft(shots,steps,audios,folder)
            from .render import audio_adjustments_from_plan
            write_json(folder/'audio-adjustments.json',audio_adjustments_from_plan(json.loads((folder/'edit_plan.json').read_text())))
            report['draft_contains_placeholders']=True;report['placeholder_steps']=verification['placeholder_steps']
            write_json(folder/'report.json',report);store.upload(folder/'report.json',remote)
            for name in ['preview.mp4','editable-package.zip','edit_plan.json','audio-adjustments.json']:store.upload(folder/name,remote)
            checkpoint(f'Review draft rendered: {len(shots)} of {len(steps)} steps use verified selections. Unresolved steps are visibly marked; this is not a finished production edit.',status='needs_review',stage='review',issues=issues,full_subset_finished=True,timing_attempted=True,verification=verification);return
        for shot in shots:
            shot['path']=str(library.video(shot['video_id']))
            shot['audio_path']=str(folder/f'narration-{next(i for i,s in enumerate(steps) if s.number==shot["step"])+1:03}.wav')
        checkpoint('All steps have compatible selections. Rendering the preview and editable timeline.',stage='assembly',issues=[])
        verification=render(shots,folder)
        from .render import audio_adjustments_from_plan
        write_json(folder/'audio-adjustments.json',audio_adjustments_from_plan(json.loads((folder/'edit_plan.json').read_text())))
        for name in ['preview.mp4','editable-package.zip','edit_plan.json','audio-adjustments.json']:store.upload(folder/name,remote)
        checkpoint('Draft rendered. Review scientific fit and complete playback before production use.',status='complete',stage='complete',verification=verification,full_subset_finished=True)
    except VisionBudgetExceeded:
        if coverage:save_coverage()
        checkpoint('Full-subset review paused at its request limit. Completed videos and assessments are saved.',status='needs_review',stage='subset_search')
    except Exception as exc:
        trace=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
        failure={'exception':type(exc).__name__,'stage':state.get('stage'),'trace':trace}
        write_json(folder/'failure.json',failure);print(f"[job {state['id'][:8]}] failed: {json.dumps(failure)}",flush=True)
        state.update(status='failed',error=f'{type(exc).__name__}: full-subset processing stopped. Download private diagnostics.',updated=time.time());write_json(folder/'state.json',state)
        if store:
            try:store.upload(folder/'state.json',remote);store.upload(folder/'failure.json',remote)
            except Exception:pass

