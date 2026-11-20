"""Invite-only hosted pilot. All source media comes from private cloud storage."""
from pathlib import Path
import os
import time
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_name,'2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import json
import re
import tempfile
import streamlit as st

st.set_page_config(page_title='JoVE Visualize Studio',page_icon='🎬',layout='wide')
st.title('JoVE Visualize Studio')
st.caption('Script → candidate footage → narration → visual review → timed edit')

def configuration():
    required=['OPENAI_API_KEY','ELEVENLABS_API_KEY','google_service_account','LIBRARY_FOLDER_ID','OUTPUT_FOLDER_ID','allowed_emails','auth']
    return [key for key in required if key not in st.secrets]

try:missing=configuration()
except FileNotFoundError:missing=['secure application settings']
if missing:
    st.info('Deployment setup is required before processing company scripts or media.')
    st.write('Missing settings: '+', '.join(missing))
    st.write('The app is closed until sign-in, an invitation list, provider credentials and private library access are configured.')
    st.stop()

if not st.user.is_logged_in:
    st.write('Sign in with an invited account to create a video draft.')
    if st.button('Sign in',type='primary'):st.login()
    st.stop()

email=str(st.user.get('email','')).strip().lower()
allowed={str(x).strip().lower() for x in st.secrets['allowed_emails']}
from pipeline.access import invited
if not invited(email,st.user.get('email_verified'),allowed):
    st.error('This account is not on the verified invitation list.')
    if st.button('Sign out'):st.logout()
    st.stop()

from runtime_loader import load_runtime
runtime=load_runtime()
Jobs=runtime.Jobs
DriveStore=runtime.DriveStore

@st.cache_resource
def manager_for_build(recovery_version):
    root=Path(os.environ.get('STUDIO_JOBS_ROOT',str(Path(tempfile.gettempdir())/'jove-studio-jobs')))
    return Jobs(root,worker_config=settings(),external_worker=os.environ.get('STUDIO_EXTERNAL_WORKER')=='1')

def manager():
    return manager_for_build(runtime.revision)

def settings():
    keys=['OPENAI_API_KEY','ELEVENLABS_API_KEY','google_service_account','LIBRARY_FOLDER_ID','OUTPUT_FOLDER_ID']
    cfg={k:dict(st.secrets[k]) if k=='google_service_account' else st.secrets[k] for k in keys}
    cfg.update(TEXT_MODEL=st.secrets.get('TEXT_MODEL','gpt-5-mini'),VISION_MODEL=st.secrets.get('VISION_MODEL','gpt-4.1-mini'),
               MAX_VISION_CALLS=int(st.secrets.get('MAX_VISION_CALLS',80)),ENGINE_BUILD=runtime.revision)
    cfg['HOSTED_SHORTLIST'] = True
    return cfg

with st.sidebar:
    st.write(email)
    st.caption('Private pilot · one processing job at a time')
    st.write('Sources: private Drive library. Narration: ElevenLabs. Text and frame analysis: OpenAI.')
    if st.button('Sign out'):st.logout()

tabs=st.tabs(['Create video','Project status','Workflow','Library readiness'])
with tabs[0]:
    st.subheader('Start with your script')
    mode=st.radio('Script source',['Google Doc link','Word or text upload'],horizontal=True)
    script_url=None;upload=None
    if mode=='Google Doc link':script_url=st.text_input('Google Doc or Drive document link')
    else:upload=st.file_uploader('Numbered script',type=['docx','txt'])
    profiles=st.secrets.get('voice_profiles',{})
    if not profiles:
        st.warning('Configure at least one approved voice profile in secure settings.');st.stop()
    profile=st.selectbox('Approved voice profile',list(profiles))
    allow_illustrations=st.checkbox(
        'Allow AI-generated scientific illustrations ONLY if suitable JoVE footage is unavailable',
        value=False,key='allow_illustrations',
        help='OFF by default. JoVE videos are always searched first. Generated illustrations are explicitly labelled and never treated as experimental footage or scientific proof.')
    if allow_illustrations:st.caption('Opt-in experimental fallback: generated illustrations are review-only and incur additional image-generation API usage. No fabricated experimental recordings.')
    else:st.caption('JoVE footage only. No AI-generated images or animations will be used.')
    st.caption('Only the script is required. The app finds footage, generates narration and produces a review edit. Any uncertain visuals or substantial timing changes are flagged.')
    consent=st.checkbox('Generate using the configured existing credits and approved private processing services.')
    if st.button('Generate video draft',type='primary',disabled=not consent):
        try:
            config=settings();config['voice']=dict(profiles[profile]);config['ALLOW_AI_ILLUSTRATIONS']=bool(allow_illustrations)
            if script_url:
                match=re.fullmatch(r'https://(?:docs\.google\.com/document/d/|drive\.google\.com/file/d/)([A-Za-z0-9_-]+)(?:/[^\s]*)?',script_url.strip())
                if not match:raise ValueError('Use a Google Docs or Drive file link.')
                text=DriveStore(config['google_service_account']).text_document(match.group(1))
            elif upload:
                if upload.name.lower().endswith('.docx'):
                    from pipeline.documents import word_script_text
                    text=word_script_text(upload.getvalue())
                else:text=upload.getvalue().decode('utf-8-sig')
            else:raise ValueError('Provide a script first.')
            ident=manager().submit(email,text,config)
            st.session_state['job_id']=ident;st.success('Project started. Open Project status to follow progress.')
        except (ValueError,PermissionError) as e:st.error(str(e))
        except Exception:st.error('Unable to start. Check document access and secure settings in the private server logs.')

with tabs[1]:
    st.subheader('Your current project')
    ident=st.session_state.get('job_id')
    recent=manager().recent(email)
    if not recent and not st.session_state.get('restore_attempted'):
        st.session_state['restore_attempted']=True
        try:
            restored=manager().restore_latest(email,settings())
            if restored:ident=restored;st.session_state['job_id']=restored;recent=manager().recent(email)
        except Exception:st.info('Saved cloud projects could not be restored automatically. The private Drive checkpoints are retained.')
    if not ident and recent:
        ident=recent[0]['id'];st.session_state['job_id']=ident
    if recent:
        options=[p['id'] for p in recent];labels={p['id']:('Library preparation' if p.get('kind')=='catalog' else 'Video project')+' · '+time.strftime('%d %b %H:%M',time.localtime(p.get('created',0)))+' · '+p['status'].replace('_',' ') for p in recent}
        ident=st.selectbox('Project',options,index=options.index(ident) if ident in options else 0,
                           format_func=lambda value:labels[value],key='projects-'+','.join(options))
        st.session_state['job_id']=ident
    if not ident:st.write('Start a project from Create video.')
    else:
        @st.fragment(run_every='8s')
        def show_job():
            try:state=manager().state(ident,email)
            except Exception:
                st.warning('This runtime no longer has the project. Its saved checkpoints remain in the configured output folder.');return
            st.write('Status: **'+state['status'].replace('_',' ')+'**')
            st.write('Stage: '+state['stage'].replace('_',' '))
            if state['stage']=='review' and not state.get('verification'):st.write('Visual review is complete. Resolve the flagged steps before rendering.')
            elif state.get('detail'):st.write(state['detail'])
            st.caption('Last reported activity: '+str(round(time.time()-state.get('updated',state.get('created',0))))+' seconds ago')
            if state['stage']=='plugin_ready':
                st.link_button('Download shortlisted Premiere project',
                               'https://drive.google.com/file/d/'+state['plugin_project_id']+'/view')
                st.write('Extract the project and open it with Script to Premiere Shortlist for footage review and editing.')
                return
            if state.get('kind')=='catalog':
                if state.get('catalog_total'):st.progress(min(1.,state.get('catalog_completed',0)/state['catalog_total']))
                if state.get('error'):st.error(state['error'])
                if state['status']=='failed' and st.button('Continue library preparation'):
                    manager().resume(ident,email,settings());st.rerun()
                return
            if state['status']=='queued':st.caption('Waiting for the processing worker. Other queued or active work finishes first.')
            if state['status']=='running' and time.time()-state.get('updated',0)>90:
                st.warning('No recent stage update. Saved work is retained; do not submit a duplicate project.')
            if state['stage']=='matching' and state.get('matching_total'):
                st.progress(min(1.,state.get('matching_completed',0)/state['matching_total']),text=f"{state.get('matching_completed',0)} of {state['matching_total']} steps in this matching phase")
            if state['stage']=='visual_verification' and state.get('total_visual_steps'):
                st.progress(min(1.,state.get('completed_visual_steps',0)/state['total_visual_steps']),text=f"{state.get('completed_visual_steps',0)} of {state['total_visual_steps']} steps reviewed")
            if state['stage']=='timing' and state.get('timing_total'):
                st.progress(min(1.,state.get('timing_completed',0)/state['timing_total']),text=f"{state.get('timing_completed',0)} of {state['timing_total']} timing mismatches checked")
            if state['stage']=='subset_search' and state.get('coverage_total'):
                st.progress(min(1.,state.get('coverage_completed',0)/state['coverage_total']),text=f"{state.get('coverage_completed',0)} of {state['coverage_total']} videos screened across their full timelines")
                st.caption('Discovery frames are sampled every three seconds. Candidate actions require a separate dense check.')
            if state.get('error'):st.error(state['error'])
            if (state.get('verification') or {}).get('review_only'):
                if (state.get('verification') or {}).get('placeholder_steps'):
                    st.warning('This older draft contains placeholder cards. Rebuild it with the updated editor below.')
                else:st.info('A real-footage review MP4 and editable timeline are available below. Review labels identify uncertain visuals or timing changes.')
            if state['status'] in {'failed','needs_review'}:
                st.download_button('Download private diagnostics',manager().diagnostics(ident,email),
                    file_name='project-diagnostics.zip',key='download-diagnostics',mime='application/zip')
                if st.button('Rebuild edit from saved script and narration',key='rebuild-'+ident):
                    try:
                        cfg=settings();cfg['voice']=dict(profiles[profile])
                        manager().resume(ident,email,cfg);st.rerun()
                    except (ValueError,PermissionError) as exc:st.error(str(exc))
            if state.get('issues') and state['status'] in {'needs_review','failed'}:
                st.warning('Review the listed visual and timing limitations before production use.')
                st.dataframe(state['issues'],hide_index=True,use_container_width=True)
            plan_path=manager().file(ident,email,'edit_plan.json')
            if state['status'] in {'needs_review','complete'} and plan_path.is_file():
                with st.expander('Review footage by script step'):
                    plan=json.loads(plan_path.read_text());numbers=list(dict.fromkeys(s['step'] for s in plan['shots']))
                    number=st.selectbox('Script step',numbers,key='review-step-'+ident)
                    selected=[(i,s) for i,s in enumerate(plan['shots']) if s['step']==number]
                    for i,shot in selected:
                        st.write(shot.get('text',''));st.caption(f"Source {shot['video_id']} · {shot['start']:.2f}–{shot['end']:.2f} seconds · {shot.get('evidence','uncertain')} · playback {shot.get('video_speed',1.):.2f}×")
                        preview=manager().shot_preview(ident,email,i)
                        if preview.is_file():st.video(str(preview))
                    retry_consent=st.checkbox('Find different footage for this step using existing credits.',key='retry-consent-'+ident)
                    if st.button('Reject these shots and find alternatives',disabled=not retry_consent,key='retry-'+ident):
                        try:
                            cfg=settings();cfg['voice']=dict(profiles[profile]);manager().review_step(ident,email,number,cfg);st.rerun()
                        except (ValueError,PermissionError) as exc:st.error(str(exc))
            for name in ['matches.xlsx','report.json','quality.json','coverage.json','audio-adjustments.json','preview.mp4','editable-package.zip']:
                if state['status'] in {'running','queued'} and name in {'preview.mp4','editable-package.zip','audio-adjustments.json'}:continue
                if state.get('editor_version') and not state.get('verification') and name in {'preview.mp4','editable-package.zip','audio-adjustments.json'}:continue
                path=manager().file(ident,email,name)
                if path.is_file():
                    if name=='preview.mp4':st.video(str(path))
                    if path.stat().st_size<150*1024**2:
                        st.download_button('Download '+name,path.read_bytes(),file_name=name,key='download-'+name)
            if state['status']=='complete':st.caption('Draft complete. Scientific and playback review remain required before production approval.')
        show_job()

with tabs[2]:
    st.subheader('One connected workflow')
    for n,label in enumerate(['Read the numbered script and production notes','Run fresh Visualizer matching across the caption index',
        'Generate narration and timing with the approved voice profile','Check Visualizer suggested timestamps first',
        'Search suggested videos next, then the wider library only for unresolved footage',
        'Keep visually suitable selections and resolve narration timing separately','Flag unresolved footage and timing for review',
        'Render an MP4 review draft and a portable editable timeline package'],1):st.write(f'{n}. {label}')
    st.caption('The pilot uses the original Visualizer engine with trained-model deserialization disabled. It does not imply that unavailable videos or sampled-frame decisions have received human approval.')

with tabs[3]:
    st.subheader('Prepare the media library once')
    st.write('Create a reusable catalogue of visible actions from the approved videos. New scripts still undergo close visual matching; catalogue descriptions are search candidates, not verified matches.')
    st.write('This preparation uses existing OpenAI credits. It is separate from the 5–10 minute script-processing target, and completed videos are reused on retry.')
    prepare_consent=st.checkbox('Prepare the approved library using existing credits. No purchases.',key='prepare-consent')
    if st.button('Prepare or update action catalogue',disabled=not prepare_consent):
        try:
            ident=manager().prepare_catalog(email,settings());st.session_state['job_id']=ident
            st.success('Preparation queued. Open Project status to follow progress.')
        except (ValueError,PermissionError) as exc:st.error(str(exc))

