"""Render multi-shot steps on the narration clock, with explicit footage review labels."""
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

class TimingCoverageError(ValueError):
    """Selection must find enough relevant context before rendering can start."""


def cut_schedule(plans,audios,fps='30000/1001',plugin_cuts=False):
    rate=float(Fraction(fps))
    if not plans:raise TimingCoverageError('No usable footage was selected.')
    for plan in plans:
        voice=audios[plan['step']]['duration']
        if not math.isfinite(voice) or voice<=0:raise TimingCoverageError('Invalid narration duration.')
        lengths=[s['end']-s['start'] for s in plan['shots']]
        if not lengths or any(not math.isfinite(n) or n<=0 for n in lengths):raise TimingCoverageError('Invalid selected footage duration.')
        if sum(lengths)/voice<.85:
            raise TimingCoverageError(f"Step {plan['step']} needs more relevant footage or a different selection. Rendering it would create excessive slow motion.")
    # Never compress a long selected procedure into extreme fast-forward.
    # Natural speech is preserved; any additional viewing time is reported.
    durations=[max(audios[p['step']]['duration'],sum(s['end']-s['start'] for s in p['shots'])/1.15) for p in plans]
    if plugin_cuts:
        durations = [audios[p['step']]['duration'] for p in plans]
    boundaries=allocate_frames(durations,fps);result=[];adjustments=[]
    for plan,(begin,end) in zip(plans,boundaries):
        total=sum(s['end']-s['start'] for s in plan['shots'])
        if not math.isfinite(total) or total<=0:raise ValueError('Invalid selected footage duration')
        step_seconds=(end-begin)/rate
        audio_tempo=max(.95,min(1.05,audios[plan['step']]['duration']/step_seconds))
        silence=max(0.,step_seconds-audios[plan['step']]['duration']/audio_tempo)
        used=0;elapsed=0.
        for index,s in enumerate(plan['shots']):
            elapsed+=s['end']-s['start']
            cut=end-begin if index==len(plan['shots'])-1 else round((end-begin)*elapsed/total)
            frames=cut-used
            if frames<1:raise ValueError('Shot cannot fit the narration frame clock')
            speed=(s['end']-s['start'])/(frames/rate)
            review=bool(plan['needs_review'] or not .85<=speed<=1.151 or silence>.3)
            result.append({**s,'step':plan['step'],'text':plan['text'],'frames':frames,
                'timeline_start_frame':begin+used,'timeline_end_frame':begin+cut,'video_speed':speed,
                'review':review,'on_screen_text':plan.get('on_screen_text', []),'additional_viewing_seconds':silence,'missing_actions':plan['missing_actions'],'unobservable_details':plan['unobservable_details']})
            used=cut
        adjustments.append({'step':plan['step'],'original_audio_seconds':audios[plan['step']]['duration'],
            'timeline_seconds':(end-begin)/rate,'tempo':audio_tempo,'additional_viewing_seconds':silence,
            'source_video_seconds':total,'video_speed':total/((end-begin)/rate),
            'review':any(s['review'] for s in result if s['step']==plan['step'])})
    if plugin_cuts:
        for i, shot in enumerate(result):
            section = shot.get('section', shot['step'])
            left = i > 0 and section != result[i-1].get('section',result[i-1]['step'])
            right = i+1 < len(result) and section != result[i+1].get('section',result[i+1]['step'])
            shot.update(left_handle=4*left, right_handle=4*right)
            length = shot['frames'] + shot['left_handle'] + shot['right_handle']
            shot['encoded_frames'] = length
            shot['video_speed'] = (shot['end']-shot['start'])*rate/length
            if shot['frames'] <= shot['left_handle']+shot['right_handle'] or not .85 <= shot['video_speed'] <= 1.15:
                raise TimingCoverageError(f"Step {shot['step']} needs a different verified range for the transition.")
    return result,adjustments,boundaries[-1][1]

def build_cut_timeline(folder,shots,frames,fps,width,height):
    root=ET.Element('mlt',producer='tractor',LC_NUMERIC='C')
    ET.SubElement(root,'profile',description='JoVE narration-aligned review',width=str(width),height=str(height),
        frame_rate_num=str(fps.numerator),frame_rate_den=str(fps.denominator),progressive='1',
        sample_aspect_num='1',sample_aspect_den='1',display_aspect_num='16',display_aspect_den='9',colorspace='709')
    def prop(element,name,value):ET.SubElement(element,'property',name=name).text=str(value)
    for i,shot in enumerate(shots):
        producer=ET.SubElement(root,'chain',id=f'clip{i}')
        for k,v in {'resource':f'media/shot-{i+1:03}.mp4','mlt_service':'avformat','audio_index':-1,
                    'shotcut:producer':1,'shotcut:caption':f"Step {shot['step']}: {shot['reason']}"}.items():prop(producer,k,v)
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
    if not plans:raise ValueError('No usable source footage is available; unrelated footage will not be invented.')
    shots,adjustments,frames=cut_schedule(plans,audios,fps,plugin_cuts);rate=Fraction(fps)
    if plugin_cuts and any(s['review'] for s in shots):
        from .editorial import metadata_only_review
        by_step = {plan['step']: plan for plan in plans}
        for shot in shots:
            if shot['review'] and (not metadata_only_review(by_step[shot['step']])
                                  or not .85 <= shot['video_speed'] <= 1.15
                                  or shot['additional_viewing_seconds'] > .3):
                raise TimingCoverageError('A selected shot still needs review.')
    # Resolve and validate source paths before concurrent encoders read them.
    paths={s['video_id']:library.video(s['video_id']) for s in shots}
    def prepare(pair):
        i,shot=pair;source=paths[shot['video_id']]
        shot = {**shot,'frames':shot.get('encoded_frames',shot['frames'])}
        output=media/f'shot-{i+1:03}.mp4';marker=output.with_suffix('.sha256')
        signature=hashlib.sha256(json.dumps([str(source),shot,str(rate),width,height],sort_keys=True).encode()).hexdigest()
        if output.exists() and marker.exists() and marker.read_text()==signature:
            import imageio_ffmpeg
            if imageio_ffmpeg.count_frames_and_secs(str(output))[0]==shot['frames']:return
        args=['-threads','1','-ss',f"{shot['start']:.9f}",'-t',f"{shot['end']-shot['start']:.9f}",'-i',source]
        base=f"setpts=(PTS-STARTPTS)/{shot['video_speed']:.12f},fps={rate},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,tpad=stop_mode=clone:stop_duration=0.1,trim=end_frame={shot['frames']}"
        if shot['review'] or shot['on_screen_text']:
            font=ImageFont.load_default(size=max(12,int(width/55)))
            lines=['REVIEW: visual details need checking'] if shot['review'] else []
            if not .85<=shot['video_speed']<=1.15:
                lines.append(f"Footage speed {shot['video_speed']:.2f}x")
            for instruction in shot['on_screen_text']:
                lines.extend(textwrap.wrap(instruction, width=max(10,int((width-32)/font.getlength('W')))))
            line_height=font.size+6
            label=Image.new('RGBA',(width,24+line_height*len(lines)),(0,0,0,205))
            draw=ImageDraw.Draw(label)
            draw.multiline_text((16,10),'\n'.join(lines),font=font,spacing=6,fill='#ffffff')
            overlay=media/f'review-{i:03}.png';label.save(overlay)
            args+=['-i',overlay,'-filter_complex_threads','1','-filter_complex',f'[0:v]{base}[v];[v][1:v]overlay=0:H-h:format=auto[out]','-map','[out]']
        else:args+=['-filter_threads','1','-vf',base]
        # Output duration is governed by frame count AFTER playback adjustment.
        args+=['-an','-frames:v',str(shot['frames']),'-r',str(rate),'-c:v','libx264','-threads','1',
               '-preset','veryfast','-crf','21','-pix_fmt','yuv420p',media/f'shot-{i+1:03}.mp4']
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
            filters.append(f'[{previous}][v{i}]{operation},fps={rate}[j{i}]')
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
    plan={'version':1,'fps':fps,'frames':frames,'shots':shots,'steps':plans,'audio_adjustments':adjustments,
          'transition_policy':'Centered eight-frame dissolve at section changes' if plugin_cuts else 'Straight cuts'}
    (folder/'edit_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (folder/'audio-adjustments.json').write_text(json.dumps(adjustments,indent=2),encoding='utf-8')
    with zipfile.ZipFile(folder/'editable-package.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for path in [folder/'editable.mlt',folder/'edit_plan.json',folder/'narration.wav',*[media/f'shot-{i+1:03}.mp4' for i in range(len(shots))]]:
            archive.write(path,path.relative_to(folder).as_posix())
    return {'frames':frames,'fps':fps,'duration':frames/float(rate),'shots':len(shots),'steps':len(plans),
            'review_only':any(s['review'] for s in shots),'placeholder_steps':[],
            'footage_steps':len(plans),'verified_steps':sum(not p['needs_review'] for p in plans),'dissolves':sum(bool(s.get('left_handle')) for s in shots)}


