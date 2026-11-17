from pathlib import Path
import json
import subprocess
import base64
from PIL import Image
import io
import imageio_ffmpeg

def execute(args,log):
    with Path(log).open('w',encoding='utf-8') as f:
        p=subprocess.run([str(imageio_ffmpeg.get_ffmpeg_exe()),'-hide_banner','-nostdin','-y',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=300)
    if p.returncode:raise RuntimeError(f'Media processing failed; see {Path(log).name}')

def sample_frames(path,start,end,folder,count=16):
    if not 0<=start<end:raise ValueError('Invalid sampling interval')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    items=[]
    for i in range(count):
        t=start+(end-start)*i/max(1,count-1)
        frame=folder/f'{i:03}.jpg'
        execute(['-ss',f'{t:.6f}','-i',path,'-frames:v','1','-vf','scale=640:-2','-q:v','3',frame],folder/f'{i:03}.log')
        if not frame.exists():continue
        raw=frame.read_bytes()
        items.append({'time':round(t,4),'data':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()})
    return items

def duration(path):
    # FFmpeg prints its probed duration even when no output is selected.
    import re
    p=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',str(path)],capture_output=True,text=True,timeout=30)
    m=re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)',p.stderr)
    if not m:raise ValueError('Unknown media duration')
    h,m,s=map(float,m.groups());return h*3600+m*60+s

def retime_audio(source,dest,tempo,target_frames,fps):
    from .timing import atempo_filters
    seconds=target_frames/float(fps)
    execute(['-i',source,'-vn','-af',atempo_filters(tempo)+f',apad,atrim=duration={seconds:.12f}',
             '-ar','48000','-ac','1','-c:a','pcm_s16le',dest],Path(dest).with_suffix('.log'))

