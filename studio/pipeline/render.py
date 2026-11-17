from pathlib import Path
from fractions import Fraction
import json
import math
import shutil
import wave
import xml.etree.ElementTree as ET
import zipfile
from .media import execute,retime_audio
from .timing import allocate_frames

def p(parent,key,value):ET.SubElement(parent,'property',name=key).text=str(value)


def audio_adjustments_from_plan(plan):
    rate=float(Fraction(plan['fps']));rows=[]
    for shot in plan['shots']:
        seconds=(shot['timeline_end_frame']-shot['timeline_start_frame'])/rate
        if seconds<=0:raise ValueError('Invalid timeline duration')
        placeholder=shot.get('kind')=='footage_not_verified'
        rows.append({'step':shot['step'],'original_audio_seconds':shot['audio_duration'],'timeline_seconds':seconds,
                     'selected_video_seconds':None if placeholder else shot['end']-shot['start'],
                     'tempo':shot['audio_duration']/seconds,'review':placeholder,'video_id':shot['video_id'],
                     'kind':'review_placeholder' if placeholder else 'verified_selection'})
    return rows

def build_mlt(folder,shots,total,fps,width,height):
    root=ET.Element('mlt',producer='tractor',LC_NUMERIC='C')
    ET.SubElement(root,'profile',description='JoVE Pipeline',width=str(width),height=str(height),
        frame_rate_num=str(fps.numerator),frame_rate_den=str(fps.denominator),progressive='1',
        sample_aspect_num='1',sample_aspect_den='1',display_aspect_num='16',display_aspect_den='9',colorspace='709')
    black=ET.SubElement(root,'producer',id='black',**{'in':'0','out':str(total-1)})
    p(black,'resource','0');p(black,'mlt_service','color');p(black,'length',total)
    bg=ET.SubElement(root,'playlist',id='background');ET.SubElement(bg,'entry',producer='black',**{'in':'0','out':str(total-1)})
    for i,s in enumerate(shots):
        pr=ET.SubElement(root,'chain',id=f'clip{i}',**{'in':'0','out':str(s['prepared_frames']-1)})
        for k,v in {'resource':f'media/shot-{i+1:03}.mp4','mlt_service':'avformat','audio_index':-1,
                    'set.test_audio':1,'shotcut:producer':1,'shotcut:caption':f'{i+1:02} {s["text"]}'}.items():p(pr,k,v)
    au=ET.SubElement(root,'chain',id='narration')
    p(au,'resource','narration.wav');p(au,'mlt_service','avformat');p(au,'video_index',-1)
    mainbin=ET.SubElement(root,'playlist',id='main_bin');p(mainbin,'xml_retain',1)
    for i,s in enumerate(shots):ET.SubElement(mainbin,'entry',producer=f'clip{i}')
    for i,s in enumerate(shots[:-1]):
        t=ET.SubElement(root,'tractor',id=f'dissolve{i}',**{'in':'0','out':'7'})
        p(t,'shotcut:transition',1)
        ET.SubElement(t,'track',producer=f'clip{i}',**{'in':str(s['prepared_frames']-8),'out':str(s['prepared_frames']-1)})
        ET.SubElement(t,'track',producer=f'clip{i+1}',**{'in':'0','out':'7'})
        tr=ET.SubElement(t,'transition',**{'in':'0','out':'7'})
        for k,v in {'mlt_service':'luma','a_track':0,'b_track':1,'alpha_over':1,'fix_background_alpha':1}.items():p(tr,k,v)
    audio=ET.SubElement(root,'playlist',id='audio');p(audio,'shotcut:name','A1 - Narration');p(audio,'shotcut:trackAudio',1)
    ET.SubElement(audio,'entry',producer='narration',**{'in':'0','out':str(total-1)})
    video=ET.SubElement(root,'playlist',id='video');p(video,'shotcut:name','V1 - Verified selections')
    for i,s in enumerate(shots):
        ET.SubElement(video,'entry',producer=f'clip{i}',**{'in':str(8 if i else 0),
            'out':str(s['prepared_frames']-1-(8 if i<len(shots)-1 else 0))})
        if i<len(shots)-1:ET.SubElement(video,'entry',producer=f'dissolve{i}',**{'in':'0','out':'7'})
    main=ET.SubElement(root,'tractor',id='tractor',shotcut='1',**{'in':'0','out':str(total-1)})
    p(main,'shotcut:name','JoVE script-to-video review draft')
    ET.SubElement(main,'track',producer='background');ET.SubElement(main,'track',producer='audio',hide='video');ET.SubElement(main,'track',producer='video',hide='audio')
    for service,b in [('mix',1),('frei0r.cairoblend',2)]:
        tr=ET.SubElement(main,'transition',**{'in':'0','out':str(total-1)})
        for k,v in {'mlt_service':service,'a_track':0,'b_track':b,'always_active':1,'sum':1}.items():p(tr,k,v)
    ET.indent(root);ET.ElementTree(root).write(folder/'editable.mlt',encoding='utf-8',xml_declaration=True)

def render(shots,folder,fps='30000/1001',width=1280,height=720):
    folder=Path(folder);media=folder/'media';media.mkdir(parents=True,exist_ok=True)
    rate=Fraction(fps);cuts=allocate_frames([s['end']-s['start'] for s in shots],fps)
    voiced=[]
    for i,(s,(a,b)) in enumerate(zip(shots,cuts)):
        core=b-a
        if core<=8:raise ValueError('Action too short for centered section transitions')
        left=int(i>0);right=int(i<len(shots)-1)
        n=core+4*left+4*right;s['prepared_frames']=n;s['timeline_start_frame']=a;s['timeline_end_frame']=b
        start=s['start']-left*4/float(rate)
        if start<0:raise ValueError('Missing incoming transition handle')
        execute(['-threads','1','-ss',f'{start:.12f}','-i',s['path'],'-an','-vf',
            f'fps={rate},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1',
            '-frames:v',str(n),'-r',str(rate),'-c:v','libx264','-threads','1','-preset','fast','-crf','20',
            media/f'shot-{i+1:03}.mp4'],media/f'shot-{i+1:03}.log')
        wav=media/f'vo-{i+1:03}.wav'
        retime_audio(s['audio_path'],wav,s['audio_duration']/(core/float(rate)),core,rate);voiced.append(wav)
    with wave.open(str(folder/'narration.wav'),'wb') as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(48000)
        for wav in voiced:
            with wave.open(str(wav),'rb') as inp:output.writeframes(inp.readframes(inp.getnframes()))
    args=['-filter_complex_threads','1']
    for i in range(len(shots)):args+=['-threads','1','-i',media/f'shot-{i+1:03}.mp4']
    args+=['-i',folder/'narration.wav']
    filters=[f'[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={rate}[v{i}]' for i in range(len(shots))]
    previous='v0';length=shots[0]['prepared_frames']
    for i in range(1,len(shots)):
        label=f'join{i}'
        filters.append(f'[{previous}][v{i}]xfade=transition=fade:duration={8/float(rate):.12f}:offset={(length-8)/float(rate):.12f}[{label}]')
        length+=shots[i]['prepared_frames']-8;previous=label
    assert length==cuts[-1][1]
    args+=['-filter_complex',';'.join(filters),'-map',f'[{previous}]','-map',f'{len(shots)}:a:0',
           '-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-r',str(rate),
           '-frames:v',str(length),'-c:a','aac','-b:a','192k','-movflags','+faststart',folder/'preview.mp4']
    execute(args,folder/'render.log')
    build_mlt(folder,shots,length,rate,width,height)
    provenance=[{k:v for k,v in s.items() if k not in {'path','audio_path'}} for s in shots]
    (folder/'edit_plan.json').write_text(json.dumps({'fps':fps,'frames':length,'shots':provenance},indent=2),encoding='utf-8')
    with zipfile.ZipFile(folder/'editable-package.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in [folder/'editable.mlt',folder/'edit_plan.json',folder/'narration.wav',*media.glob('shot-*.mp4')]:archive.write(p,p.relative_to(folder).as_posix())
    return {'frames':length,'fps':fps,'duration':length/float(rate),'dissolves':len(shots)-1,'dissolve_frames':8}


def render_review_draft(verified,steps,audios,folder,width=1280,height=720):
    """Deliver an editable review artifact with explicit cards for unsupported steps."""
    from PIL import Image,ImageDraw,ImageFont
    import textwrap
    folder=Path(folder);cards=folder/'review-cards';cards.mkdir(exist_ok=True)
    by_step={s['step']:s for s in verified};timeline=[];missing=[]
    for step,audio in zip(steps,audios):
        if step.number in by_step:
            shot=dict(by_step[step.number]);shot['audio_path']=audio['path'];timeline.append(shot);continue
        missing.append(step.number)
        image=Image.new('RGB',(width,height),'#102536');draw=ImageDraw.Draw(image)
        header=ImageFont.load_default(size=34);body=ImageFont.load_default(size=28)
        draw.text((60,60),'REVIEW DRAFT — FOOTAGE NOT VERIFIED',font=header,fill='#ffc857')
        draw.text((60,145),f'Step {step.number}',font=header,fill='white')
        lines=textwrap.wrap(step.text,width=65)
        draw.multiline_text((60,225),'\n'.join(lines),font=body,fill='white',spacing=14)
        draw.text((60,height-90),'Placeholder for review. This card does not depict the procedure.',font=ImageFont.load_default(size=22),fill='#c5d2dc')
        still=cards/f'step-{step.number:03}.png';image.save(still)
        source=still.with_suffix('.mp4')
        execute(['-loop','1','-i',still,'-t',str(audio['duration']+.5),'-an','-r','30000/1001',
                 '-c:v','libx264','-threads','1','-pix_fmt','yuv420p',source],source.with_suffix('.log'))
        timeline.append({'step':step.number,'text':step.text,'video_id':'review-placeholder','path':str(source),
                         'start':.2,'end':.2+audio['duration'],'audio_path':audio['path'],'audio_duration':audio['duration'],
                         'tempo':1.,'confidence':0.,'tier':'clearly_marked_review_placeholder','kind':'footage_not_verified',
                         'reason':'No scientifically compatible complete action with natural narration timing has been verified.'})
    result=render(timeline,folder,width=width,height=height)
    return {**result,'review_only':True,'placeholder_steps':missing,'verified_steps':len(verified)}

