import base64
import io
import json
import math
import wave
from pathlib import Path
import requests
from openai import OpenAI
from .tts_preprocessor import prepare_script_for_tts

class ModelGateway:
    def __init__(self,api_key,model,max_calls=80):
        self.client=OpenAI(api_key=api_key,timeout=120,max_retries=1)
        self.model=model;self.max_calls=max_calls;self.calls=[]

    def ask(self,instructions,payload,images=()):
        if len(self.calls)>=self.max_calls:raise RuntimeError('Job AI request limit reached; review before continuing.')
        self.calls.append({'status':'started'})
        contents=[{'type':'text','text':json.dumps(payload,ensure_ascii=False)}]
        for im in images:
            contents.extend([{'type':'text','text':f'Source time {im["time"]} seconds'},
                             {'type':'image_url','image_url':{'url':im['data'],'detail':'low'}}])
        r=self.client.chat.completions.create(model=self.model,messages=[{'role':'system','content':instructions},
            {'role':'user','content':contents}],response_format={'type':'json_object'})
        self.calls[-1]={'status':'complete','id':r.id,'tokens':r.usage.total_tokens if r.usage else None}
        return json.loads(r.choices[0].message.content)

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
            'previous_text':previous,'next_text':next_text},timeout=180)
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

