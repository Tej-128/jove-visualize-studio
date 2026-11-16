"""Split large local assets for a size-limited transfer connector. Never changes originals."""
from pathlib import Path
import argparse
import hashlib
import json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=Path,required=True);parser.add_argument('--videos',type=Path,required=True)
    parser.add_argument('--checksums',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    checks=json.loads(args.checksums.read_text())
    sources=[args.index,*[p for p in args.videos.iterdir() if p.is_file() and p.stat().st_size>100*1024**2]]
    manifest={};queue=[]
    for source in sources:
        record={'sha256':checks[source.name],'parts':[]}
        digest=hashlib.sha256()
        with source.open('rb') as f:
            n=1
            while True:
                block=f.read(64*1024**2)
                if not block:break
                digest.update(block);name=f'{source.name}.part{n:03}'
                target=args.output/name;target.write_bytes(block)
                record['parts'].append({'name':name,'sha256':hashlib.sha256(block).hexdigest()})
                queue.append({'name':name,'path':str(target.resolve()),'location':'library' if source==args.index else 'videos'})
                n+=1
        if digest.hexdigest()!=checks[source.name]:raise ValueError('Original changed during packaging')
        manifest[source.name]=record
    (args.output/'asset-parts.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (args.output/'upload-queue.json').write_text(json.dumps(queue,indent=2),encoding='utf-8')
    print(f'Prepared {len(queue)} verified parts from {len(manifest)} assets; no originals were modified.')

if __name__=='__main__':main()

