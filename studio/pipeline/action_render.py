"""Render every script step on the narration clock, with explicit review labels.

The narration is the clock: each step lasts exactly as long as its narration.
Too much selected footage is trimmed around the essential action; too little plays
at native speed and its last frame is held. A step without footage gets a labelled
card. Every adjustment is reported; none is presented as verification.
"""
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import math
import wave
import textwrap
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
from .media import execute,retime_audio
from .timing import allocate_frames

RENDER_VERSION=2

class TimingCoverageError(ValueError):
    """The narration clock or the selected footage is invalid."""


def _fit_shots(shots,seconds):
    """Return (shots, playback speed, trimmed source seconds) for one step."""
    lengths=[s['end']-s['start'] for s in shots];total=sum(lengths)
    if total>seconds*1.15:
        keep=seconds/total;fitted=[]
        for s,length in zip(shots,lengths):
            want=min(length,max(.3,length*keep))
            core_a=s.get('action_start',s['start']);core_b=s.get('action_end',s['end'])
            center=(core_a+core_b)/2
            a=max(s['start'],min(center-want/2,s['end']-want));b=min(s['end'],a+want)
            fitted.append({**s,'start':a,'end':b,'source_range':[s['start'],s['end']]})
        kept=sum(s['end']-s['start'] for s in fitted)
        return fitted,min(1.15,max(.85,kept/seconds)),total-kept
    if total<seconds*.85:
        return [dict(s) for s in shots],1.,0.
    return [dict(s) for s in shots],total/seconds,0.


def cut_schedule(plans,audios,fps='30000/1001',plugin_cuts=False):
    """Schedule every step for exactly its narration duration. plugin_cuts is accepted for compatibility."""
    rate=float(Fraction(fps))
    if not plans:raise TimingCoverageError('No script steps were supplied.')
    for plan in plans:
        voice=audios[plan['step']]['duration']
        if not math.isfinite(voice) or voice<=0:raise TimingCoverageError('Invalid narration duration.')
        if any(not math.isfinite(s['end']-s['start']) or s['end']-s['start']<=0 for s in plan['shots']):
            raise TimingCoverageError('Invalid selected footage duration.')
    boundaries=allocate_frames([audios[p['step']]['duration'] for p in plans],fps);result=[];adjustments=[]
    for plan,(begin,end) in zip(plans,boundaries):
        voice=audios[plan['step']]['duration'];step_frames=end-begin;step_seconds=step_frames/rate
        tempo=max(.95,min(1.05,voice/step_seconds))
        common={'step':plan['step'],'text':plan.get('text',''),'status':plan.get('status'),'on_screen_text':plan.get('on_screen_text',[]),
                'missing_actions':plan.get('missing_actions',[]),'unobservable_details':plan.get('unobservable_details',[]),
                'additional_viewing_seconds':0.,'review':True}
        if not plan['shots']:
            result.append({**common,'kind':'slate','video_id':None,'start':0.,'end':step_seconds,'frames':step_frames,
                'timeline_start_frame':begin,'timeline_end_frame':end,'video_speed':1.,'hold_seconds':0.,
                'reason':'No suitable footage. Missing: '+('; '.join(plan.get('missing_actions',[])) or 'see the review report')})
            adjustments.append({'step':plan['step'],'original_audio_seconds':voice,'timeline_seconds':step_seconds,'tempo':tempo,
                'additional_viewing_seconds':0.,'source_video_seconds':0.,'video_speed':1.,'hold_seconds':0.,
                'trimmed_seconds':0.,'slate':True,'review':True})
            continue
        shots,speed,trimmed=_fit_shots(plan['shots'],step_seconds)
        lengths=[s['end']-s['start'] for s in shots]
        used=0;elapsed=0.
        for index,(s,length) in enumerate(zip(shots,lengths)):
            elapsed+=length/speed
            last=index==len(shots)-1
            cut=step_frames if last else min(step_frames-(len(shots)-1-index),max(used+1,round(elapsed*rate)))
            frames=cut-used
            if frames<1:raise TimingCoverageError(f"Step {plan['step']} has more shots than narration frames.")
            result.append({**s,**common,'kind':'footage','frames':frames,'timeline_start_frame':begin+used,
                'timeline_end_frame':begin+cut,'video_speed':speed,'hold_seconds':max(0.,frames/rate-length/speed)})
            used=cut
        adjustments.append({'step':plan['step'],'original_audio_seconds':voice,'timeline_seconds':step_seconds,'tempo':tempo,
            'additional_viewing_seconds':0.,'source_video_seconds':sum(lengths),'video_speed':speed,
            'hold_seconds':max(0.,step_seconds-sum(lengths)/speed),'trimmed_seconds':trimmed,'review':True})
    if len(result)>1:
        for i,shot in enumerate(result):
            shot.update(left_handle=4*(i>0),right_handle=4*(i+1<len(result)))
            length=shot['frames']+shot['left_handle']+shot['right_handle']
            shot['encoded_frames']=length
            if shot['kind']=='footage':
                # Spread the source across the dissolve handles while it stays near natural
                # speed; otherwise keep the speed and hold the last frame for the difference.
                source=shot['end']-shot['start'];spread=source*rate/length
                shot['video_speed']=spread if .85<=spread<=shot['video_speed'] else max(.85,min(1.,shot['video_speed']))
                shot['hold_seconds']=max(0.,length/rate-source/shot['video_speed'])
    return result,adjustments,boundaries[-1][1]


def _font(size):
    return ImageFont.load_default(size=size)


def slate_card(shot,width,height):
    """A plainly labelled gap card: the narration continues and the editor sources footage."""
    card=Image.new('RGB',(width,height),'#13232a');draw=ImageDraw.Draw(card)
    big=_font(max(16,int(width/30)));body=_font(max(12,int(width/50)))
    x=int(width*.07);y=int(height*.11);chars=max(24,int((width-2*x)/max(1,body.getlength('n'))))
    draw.text((x,y),f"STEP {shot['step']} · NO SUITABLE FOOTAGE FOUND",font=big,fill='#f2b35b');y+=int(big.size*1.9)
    for line in textwrap.wrap(shot.get('text',''),chars)[:5]:
        draw.text((x,y),line,font=body,fill='#ffffff');y+=int(body.size*1.45)
    missing=shot.get('missing_actions') or []
    if missing:
        y+=int(body.size*.8)
        for line in textwrap.wrap('Missing footage: '+'; '.join(missing),chars)[:4]:
            draw.text((x,y),line,font=body,fill='#9fd8c9');y+=int(body.size*1.45)
    draw.text((x,height-int(height*.12)),'REVIEW DRAFT · narration continues · editor to source footage',font=body,fill='#8fa7b0')
    return card


def review_lines(shot,chars):
    level='action directly visible' if shot.get('evidence_level')=='directly_visible' else 'representative footage'
    lines=[f"REVIEW DRAFT · Step {shot['step']} · {level}"]
    unverified=shot.get('unverified_details') or shot.get('unobservable_details') or []
    if unverified:
        lines.extend(textwrap.wrap('Not verified: '+', '.join(map(str,unverified)),chars)[:2])
    if shot.get('hold_seconds',0)>.3:
        lines.append(f"Footage ends; last frame held {shot['hold_seconds']:.1f} s")
    if not .85<=shot['video_speed']<=1.151:
        lines.append(f"Footage speed {shot['video_speed']:.2f}x")
    for instruction in shot.get('on_screen_text') or []:
        lines.extend(textwrap.wrap(instruction,chars))
    return lines


def build_cut_timeline(folder,shots,frames,fps,width,height):
    root=ET.Element('mlt',producer='tractor',LC_NUMERIC='C')
    ET.SubElement(root,'profile',description='JoVE narration-aligned review',width=str(width),height=str(height),
        frame_rate_num=str(fps.numerator),frame_rate_den=str(fps.denominator),progressive='1',
        sample_aspect_num='1',sample_aspect_den='1',display_aspect_num='16',display_aspect_den='9',colorspace='709')
    def prop(element,name,value):ET.SubElement(element,'property',name=name).text=str(value)
    for i,shot in enumerate(shots):
        producer=ET.SubElement(root,'chain',id=f'clip{i}')
        status=(shot.get('status') or 'review').replace('_',' ')
        for k,v in {'resource':f'media/shot-{i+1:03}.mp4','mlt_service':'avformat','audio_index':-1,
                    'shotcut:producer':1,'shotcut:caption':f"Step {shot['step']} [{status}]: {shot.get('reason','')}"}.items():prop(producer,k,v)
    audio=ET.SubElement(root,'chain',id='narration');prop(audio,'resource','narration.wav');prop(audio,'mlt_service','avformat');prop(audio,'video_index',-1)
    binlist=ET.SubElement(root,'playlist',id='main_bin');prop(binlist,'xml_retain',1)
    for i in range(len(shots)):ET.SubElement(binlist,'entry',producer=f'clip{i}')
    voice=ET.SubElement(root,'playlist',id='audio');prop(voice,'shotcut:name','A1 Narration');prop(voice,'shotcut:trackAudio',1)
    ET.SubElement(voice,'entry',producer='narration',**{'in':'0','out':str(frames-1)})
    video=ET.SubElement(root,'playlist',id='video');prop(video,'shotcut:name','V1 Footage')
    for i,shot in enumerate(shots):
        end = shot.get('encoded_frames',shot['frames'])
        overlap = 2*shot.get('right_handle',0)
        ET.SubElement(video,'entry',producer=f'clip{i}',**{'in':str(2*shot.get('left_handle',0)),'out':str(end-overlap-1)})
        if overlap:
            dissolve=ET.Element('tractor',id=f'dissolve{i}',out='7')
            root.insert(list(root).index(video),dissolve)
            ET.SubElement(dissolve,'track',producer=f'clip{i}',**{'in':str(end-8),'out':str(end-1)})
            ET.SubElement(dissolve,'track',producer=f'clip{i+1}',**{'in':'0','out':'7'})
            transition=ET.SubElement(dissolve,'transition',out='7')
            for k,v in {'mlt_service':'luma','a_track':0,'b_track':1}.items():prop(transition,k,v)
            ET.SubElement(video,'entry',producer=f'dissolve{i}',**{'in':'0','out':'7'})
    tractor=ET.SubElement(root,'tractor',id='tractor',shotcut='1',**{'in':'0','out':str(frames-1)})
    ET.SubElement(tractor,'track',producer='video',hide='audio');ET.SubElement(tractor,'track',producer='audio',hide='video')
    mix=ET.SubElement(tractor,'transition',**{'in':'0','out':str(frames-1)})
    for k,v in {'mlt_service':'mix','a_track':0,'b_track':1,'always_active':1,'sum':1}.items():prop(mix,k,v)
    ET.indent(root);ET.ElementTree(root).write(folder/'editable.mlt',encoding='utf-8',xml_declaration=True)

def render_actions(plans,audios,library,folder,fps='30000/1001',width=1280,height=720,workers=2,plugin_cuts=False):
    folder=Path(folder);media=folder/'media';media.mkdir(parents=True,exist_ok=True)
    if not plans:raise ValueError('No script steps were supplied.')
    shots,adjustments,frames=cut_schedule(plans,audios,fps,plugin_cuts);rate=Fraction(fps)
    # Resolve and validate source paths before concurrent encoders read them.
    paths={s['video_id']:library.video(s['video_id']) for s in shots if s['kind']=='footage'}
    def prepare(pair):
        i,shot=pair
        shot={**shot,'frames':shot.get('encoded_frames',shot['frames'])}
        source=paths.get(shot['video_id']) if shot['kind']=='footage' else None
        output=media/f'shot-{i+1:03}.mp4';marker=output.with_suffix('.sha256')
        signature=hashlib.sha256(json.dumps([RENDER_VERSION,str(source),shot,str(rate),width,height],sort_keys=True,default=str).encode()).hexdigest()
        if output.exists() and marker.exists() and marker.read_text()==signature:
            import imageio_ffmpeg
            if imageio_ffmpeg.count_frames_and_secs(str(output))[0]==shot['frames']:return
        if shot['kind']=='slate':
            card=media/f'card-{i+1:03}.png';slate_card(shot,width,height).save(card)
            args=['-loop','1','-framerate',str(rate),'-i',card,'-filter_threads','1','-vf',f'fps={rate},format=yuv420p,setsar=1']
        else:
            args=['-threads','1','-ss',f"{shot['start']:.9f}",'-t',f"{shot['end']-shot['start']:.9f}",'-i',source]
            base=(f"setpts=(PTS-STARTPTS)/{shot['video_speed']:.12f},fps={rate},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                  f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,tpad=stop_mode=clone:stop_duration={shot['hold_seconds']+.2:.3f},"
                  f"trim=end_frame={shot['frames']}")
            font=_font(max(12,int(width/55)))
            lines=review_lines(shot,max(10,int((width-32)/font.getlength('W'))))
            line_height=font.size+6
            label=Image.new('RGBA',(width,24+line_height*len(lines)),(0,0,0,205))
            ImageDraw.Draw(label).multiline_text((16,10),'\n'.join(lines),font=font,spacing=6,fill='#ffffff')
            overlay=media/f'review-{i:03}.png';label.save(overlay)
            args+=['-i',overlay,'-filter_complex_threads','1','-filter_complex',f'[0:v]{base}[v];[v][1:v]overlay=0:H-h:format=auto[out]','-map','[out]']
        # Output duration is governed by frame count AFTER playback adjustment.
        args+=['-an','-frames:v',str(shot['frames']),'-r',str(rate),'-c:v','libx264','-threads','1',
               '-preset','veryfast','-crf','21','-pix_fmt','yuv420p',output]
        execute(args,media/f'shot-{i+1:03}.log')
        marker.write_text(signature)
    with ThreadPoolExecutor(max_workers=max(1,min(2,workers))) as pool:list(pool.map(prepare,enumerate(shots)))
    voice_paths=[]
    for plan in plans:
        number=plan['step'];row=next(a for a in adjustments if a['step']==number)
        count=sum(s['frames'] for s in shots if s['step']==number)
        dest=media/f'vo-{number:03}.wav'
        retime_audio(audios[number]['path'],dest,row['tempo'],count,rate);voice_paths.append(dest)
    with wave.open(str(folder/'narration.wav'),'wb') as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(48000)
        for path in voice_paths:
            with wave.open(str(path),'rb') as voice:output.writeframes(voice.readframes(voice.getnframes()))
    concat=folder/'segments.txt';concat.write_text(''.join(f"file 'media/shot-{i+1:03}.mp4'\n" for i in range(len(shots))))
    if any(s.get('left_handle') for s in shots):
        args=['-filter_complex_threads','1']
        filters=[]
        for i in range(len(shots)):
            args+=['-threads','1','-i',media/f'shot-{i+1:03}.mp4']
            filters.append(f'[{i}:v]setpts=PTS-STARTPTS,fps={rate}[v{i}]')
        previous='v0'
        for i,s in enumerate(shots[1:],1):
            operation=(f"xfade=duration={8/float(rate):.12f}:offset={(s['timeline_start_frame']-4)/float(rate):.12f}"
                       if s['left_handle'] else 'concat=n=2:v=1:a=0')
            expected_end = s['timeline_end_frame'] + s['right_handle']
            filters.append(f'[{previous}][v{i}]{operation},fps={rate},tpad=stop_mode=clone:stop=2,trim=end_frame={expected_end}[j{i}]')
            previous=f'j{i}'
        execute(args+['-i',folder/'narration.wav','-filter_complex',';'.join(filters+[f'[{previous}]tpad=stop_mode=clone:stop=1[final]']),'-map','[final]',
            '-map',f'{len(shots)}:a','-r',str(rate),'-frames:v',str(frames),'-c:v','libx264','-threads','2',
            '-pix_fmt','yuv420p','-crf','18','-c:a','aac','-movflags','+faststart',folder/'preview.mp4'],folder/'render.log')
    else:execute(['-f','concat','-safe','1','-i',concat,'-i',folder/'narration.wav','-map','0:v:0','-map','1:a:0',
             '-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',folder/'preview.mp4'],folder/'render.log')
    # Check the actual artifact, not just the intended filter graph.
    import imageio_ffmpeg
    actual,_=imageio_ffmpeg.count_frames_and_secs(str(folder/'preview.mp4'))
    if actual!=frames:raise ValueError(f'Rendered frame count mismatch: expected {frames}, received {actual}')
    build_cut_timeline(folder,shots,frames,rate,width,height)
    plan={'version':2,'fps':fps,'frames':frames,'shots':shots,'steps':plans,'audio_adjustments':adjustments,
          'transition_policy':'Centered eight-frame dissolve at every join',
          'timing_policy':'Each step lasts exactly as long as its narration; holds, trims and cards are reported.'}
    (folder/'edit_plan.json').write_text(json.dumps(plan,indent=2,default=str),encoding='utf-8')
    (folder/'audio-adjustments.json').write_text(json.dumps(adjustments,indent=2),encoding='utf-8')
    with zipfile.ZipFile(folder/'editable-package.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for path in [folder/'editable.mlt',folder/'edit_plan.json',folder/'narration.wav',*[media/f'shot-{i+1:03}.mp4' for i in range(len(shots))]]:
            archive.write(path,path.relative_to(folder).as_posix())
    slates=sorted({s['step'] for s in shots if s['kind']=='slate'})
    return {'frames':frames,'fps':fps,'duration':frames/float(rate),'shots':len(shots),'steps':len(plans),
            'review_only':True,'placeholder_steps':slates,'footage_steps':len(plans)-len(slates),
            'selected_steps':sum(p.get('status')=='selected' for p in plans),'verified_steps':0,
            'publication_approved':False,'dissolves':sum(bool(s.get('left_handle')) for s in shots)}
