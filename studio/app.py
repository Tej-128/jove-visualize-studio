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

from pipeline.runner import Jobs
from pipeline.storage import DriveStore

@st.cache_resource
def manager():return Jobs(Path(tempfile.gettempdir())/'jove-studio-jobs')

def settings():
    keys=['OPENAI_API_KEY','ELEVENLABS_API_KEY','google_service_account','LIBRARY_FOLDER_ID','OUTPUT_FOLDER_ID']
    cfg={k:dict(st.secrets[k]) if k=='google_service_account' else st.secrets[k] for k in keys}
    cfg.update(TEXT_MODEL=st.secrets.get('TEXT_MODEL','gpt-5-mini'),VISION_MODEL=st.secrets.get('VISION_MODEL','gpt-4.1-mini'),
               MAX_VISION_CALLS=int(st.secrets.get('MAX_VISION_CALLS',80)))
    return cfg

with st.sidebar:
    st.write(email)
    st.caption('Private pilot · one processing job at a time')
    st.write('Sources: private Drive library. Narration: ElevenLabs. Text and frame analysis: OpenAI.')
    if st.button('Sign out'):st.logout()

tabs=st.tabs(['Create video','Project status','Workflow'])
with tabs[0]:
    st.subheader('Start with your script')
    mode=st.radio('Script source',['Google Doc link','Word or text upload'],horizontal=True)
    script_url=None;upload=None
    if mode=='Google Doc link':script_url=st.text_input('Google Doc or Drive document link')
    else:upload=st.file_uploader('Numbered script',type=['docx','txt'])
    reference=st.file_uploader('Optional reviewed timestamp workbook for this exact script',type=['xlsx'])
    profiles=st.secrets.get('voice_profiles',{})
    if not profiles:
        st.warning('Configure at least one approved voice profile in secure settings.');st.stop()
    profile=st.selectbox('Approved voice profile',list(profiles))
    st.caption('The approved generation profile is retained. Step audio is adjusted only within the pilot pacing limits; other cases stop for review.')
    consent=st.checkbox('Generate using the configured existing credits and approved private processing services.')
    if st.button('Generate video draft',type='primary',disabled=not consent):
        try:
            config=settings();config['voice']=dict(profiles[profile])
            if script_url:
                match=re.fullmatch(r'https://(?:docs\.google\.com/document/d/|drive\.google\.com/file/d/)([A-Za-z0-9_-]+)(?:/[^\s]*)?',script_url.strip())
                if not match:raise ValueError('Use a Google Docs or Drive file link.')
                text=DriveStore(config['google_service_account']).text_document(match.group(1))
            elif upload:
                if upload.name.lower().endswith('.docx'):
                    from docx import Document
                    from io import BytesIO
                    d=Document(BytesIO(upload.getvalue()));text='\n'.join(p.text for p in d.paragraphs)
                    text+='\n'+'\n'.join(' | '.join(c.text for c in row.cells) for t in d.tables for row in t.rows)
                else:text=upload.getvalue().decode('utf-8-sig')
            else:raise ValueError('Provide a script first.')
            ident=manager().submit(email,text,config,reference.getvalue() if reference else None)
            st.session_state['job_id']=ident;st.success('Project started. Open Project status to follow progress.')
        except (ValueError,PermissionError) as e:st.error(str(e))
        except Exception:st.error('Unable to start. Check document access and secure settings in the private server logs.')

with tabs[1]:
    st.subheader('Your current project')
    ident=st.session_state.get('job_id')
    recent=manager().recent(email)
    if not ident and recent:
        ident=recent[0]['id'];st.session_state['job_id']=ident
    if not ident:st.write('Start a project from Create video.')
    else:
        @st.fragment(run_every='8s')
        def show_job():
            try:state=manager().state(ident,email)
            except Exception:
                st.warning('This runtime no longer has the project. Its saved checkpoints remain in the configured output folder.');return
            st.write('Status: **'+state['status'].replace('_',' ')+'**')
            st.write('Stage: '+state['stage'].replace('_',' '))
            if state.get('detail'):st.write(state['detail'])
            st.caption('Last reported activity: '+str(round(time.time()-state.get('updated',state.get('created',0))))+' seconds ago')
            if state['stage']=='matching' and state.get('matching_total'):
                st.progress(min(1.,state.get('matching_completed',0)/state['matching_total']),text=f"{state.get('matching_completed',0)} of {state['matching_total']} steps in this matching phase")
            if state['stage']=='visual_verification' and state.get('total_visual_steps'):
                st.progress(min(1.,state.get('completed_visual_steps',0)/state['total_visual_steps']),text=f"{state.get('completed_visual_steps',0)} of {state['total_visual_steps']} steps reviewed")
            if state.get('error'):st.error(state['error'])
            if state['status'] in {'failed','needs_review'}:
                st.download_button('Download private diagnostics',manager().diagnostics(ident,email),
                    file_name='project-diagnostics.zip',key='download-diagnostics',mime='application/zip')
            if state.get('issues'):
                st.warning('Review is required before a complete edit can be rendered.')
                st.dataframe(state['issues'],hide_index=True,use_container_width=True)
            for name in ['matches.xlsx','report.json','audio-adjustments.json','preview.mp4','editable-package.zip']:
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
        'Generate narration and timing with the approved voice profile','Inspect every suggested timestamp option for each step',
        'If needed, search the full suggested videos, then the wider accessible library',
        'Select complete actions with compatible narration pacing','Resolve flagged steps before assembly',
        'Render MP4 and a portable editable timeline package'],1):st.write(f'{n}. {label}')
    st.caption('The pilot uses the original Visualizer engine with trained-model deserialization disabled. It does not imply that unavailable videos or sampled-frame decisions have received human approval.')

