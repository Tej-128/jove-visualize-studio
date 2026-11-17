import base64
import io
import json
import math
import wave
import hashlib
import time
import threading
from pathlib import Path
import requests
from openai import OpenAI
from .tts_preprocessor import prepare_script_for_tts

class VisionBudgetExceeded(RuntimeError):
    pass

class ModelGateway:
    def __init__(self,api_key,model,max_calls=80,cache_dir=None,progress=None):
        self.client=OpenAI(api_key=api_key,timeout=90,max_retries=0)
        self.model=model;self.max_calls=max_calls;self.calls=[]
        self.cache_dir=Path(cache_dir) if cache_dir else None
        self.progress=progress;self.cache_hits=0
        self.lock=threading.RLock()
        self.deadline=None
        self.prior_calls=[]
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True,exist_ok=True)
            journal=self.cache_dir.parent/'model-requests.json'
            if journal.exists():self.prior_calls=json.loads(journal.read_text(encoding='utf-8'))

    def ensure_capacity(self):
        if self.deadline is not None and time.monotonic()>=self.deadline:
            raise VisionBudgetExceeded('Visual review reached its time budget. Rendering retained candidates for review.')
        if len(self.calls)>=self.max_calls:
            raise VisionBudgetExceeded('The visual review request limit was reached. Saved work is retained; review or continue the saved project.')

    def ask(self,instructions,payload,images=()):
        fingerprint=hashlib.sha256(json.dumps([self.model,instructions,payload,list(images)],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        cached=self.cache_dir/(fingerprint+'.json') if self.cache_dir else None
        if cached and cached.exists():
            self.cache_hits+=1
            return json.loads(cached.read_text(encoding='utf-8'))
        with self.lock:
            self.ensure_capacity()
            record={'status':'started','request_hash':fingerprint}
            self.calls.append(record)
            self._save_calls()
        contents=[{'type':'text','text':json.dumps(payload,ensure_ascii=False)}]
        for im in images:
            contents.extend([{'type':'text','text':f'{im.get("label", "Source")} time {im["time"]} seconds'},
                             {'type':'image_url','image_url':{'url':im['data'],'detail':im.get('detail','low')}}])
        started=time.monotonic()
        try:
            client=self.client
            if self.deadline is not None:
                client=client.with_options(timeout=max(1,min(45,self.deadline-time.monotonic())))
            r=client.chat.completions.create(model=self.model,messages=[{'role':'system','content':instructions},
                {'role':'user','content':contents}],response_format={'type':'json_object'})
            answer=json.loads(r.choices[0].message.content)
            if not isinstance(answer,dict):raise ValueError('Visual review returned an invalid response shape')
            record.update(status='complete',id=r.id,tokens=r.usage.total_tokens if r.usage else None)
            if cached:
                with self.lock:
                    temp=cached.with_suffix('.temporary');temp.write_text(json.dumps(answer),encoding='utf-8');temp.replace(cached)
            return answer
        except Exception as exc:
            record.update(status='failed',exception=type(exc).__name__)
            raise
        finally:
            record['elapsed_seconds']=round(time.monotonic()-started,2)
            self._save_calls()
            if self.progress:self.progress(len(self.calls),self.max_calls)

    def _save_calls(self):
        with self.lock:
            if self.cache_dir:
                dest=self.cache_dir.parent/'model-requests.json'
                temp=dest.with_suffix('.temporary');temp.write_text(json.dumps(self.prior_calls+self.calls),encoding='utf-8');temp.replace(dest)

def generate_step_audio(text,previous,next_text,voice_id,speed,api_key,dest,output_format='pcm_44100'):
    chunks,_=prepare_script_for_tts(text)
    spoken='\n\n'.join(chunks)
    if not spoken.strip():raise ValueError('No spoken script after preparation')
    if len(spoken)>9000:raise ValueError('This step is too long; split it into coherent narration steps.')
    if not .7<=speed<=1.2:raise ValueError('Unsupported generation speed')
    r=requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps',
        params={'output_format':output_format},headers={'xi-api-key':api_key,'Content-Type':'application/json'},
        json={'text':spoken,'model_id':'eleven_multilingual_v2','voice_settings':{
            'stability':.95,'similarity_boost':.85,'style':0.,'use_speaker_boost':True,'speed':speed},
            'previous_text':previous,'next_text':next_text},timeout=(10,45))
    if r.status_code!=200:raise RuntimeError(f'ElevenLabs generation failed ({r.status_code}); check quota and permitted audio formats in secure settings.')
    body=r.json();pcm=base64.b64decode(body['audio_base64'])
    if output_format!='pcm_44100':raise ValueError('Only the approved PCM format is implemented.')
    if len(pcm)%2:raise ValueError('Unexpected PCM response')
    with wave.open(str(dest),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100);w.writeframes(pcm)
    alignment=body.get('alignment')
    if not alignment:raise ValueError('Narration has no timing data; alignment is required.')
    info={'spoken':spoken,'duration':len(pcm)/88200,'alignment':alignment,'voice_id':voice_id,'generation_speed':speed}
    Path(dest).with_suffix('.json').write_text(json.dumps(info),encoding='utf-8')
    return info

