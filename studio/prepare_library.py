"""One-time administrator packaging; runtime uses private Drive files only."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=Path,required=True);p.add_argument('--videos',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    allowed=['manifest.json','corpus.sqlite','embeddings.npy']
    for name in allowed:
        if not (a.index/name).is_file():raise ValueError(f'Missing {name}')
    target=a.output/'index.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in allowed:z.write(a.index/name,name)
    files=[target,*[f for f in a.videos.iterdir() if f.is_file() and f.suffix.lower() in {'.mp4','.mov','.mkv','.avi','.m4v','.webm'}]]
    checks={}
    for f in files:
        with f.open('rb') as handle:checks[f.name]=hashlib.file_digest(handle,'sha256').hexdigest()
    (a.output/'sha256sums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Upload index.zip and sha256sums.json to a private Drive library folder. Upload the original Videos folder beneath it. No secrets or trained Python objects are included.')

if __name__=='__main__':main()

