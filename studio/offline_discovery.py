"""Read-only pilot discovery. No queue, provider client, remote store or renderer."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor' / 'src'))
from jove_visualizer.features import build_fts_query
from jove_visualizer.script_parser import parse_steps_from_text
from pipeline.action_catalog import ActionCatalog
from pipeline.documents import word_script_text
from pipeline.editorial import action_units


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['script', 'catalog', 'videos', 'corpus', 'output']:
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    job = uuid.uuid4().hex
    folder = args.output / job
    folder.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (folder / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git','-c','safe.directory='+repo.as_posix(),'-C',repo,'rev-parse','HEAD'],text=True).strip()
    run = dict(id=job, commit=commit, started=datetime.now(timezone.utc).isoformat(),
               allow_ai_illustrations=False, provider_requests=0, provider_cost=0,
               worker_count=1, queue=None, processing_status='running', technical_status='not_rendered',
               scientific_status='unverified', script_sha256=digest(args.script), seconds={})
    save('state.json', run)
    text = word_script_text(args.script.read_bytes())
    steps = parse_steps_from_text(text)
    save('script.json', dict(steps=[asdict(s) for s in steps], references=re.findall(r'https?://\S+', text)))
    run['seconds']['ingestion_and_step_analysis'] = time.monotonic()-start
    t = time.monotonic()
    sources = json.loads(args.catalog.read_bytes())['sources']
    verified = {}
    for path in args.videos.glob('*.mp4'):
        match = re.match(r'(\d+)(?:_|\.)', path.name)
        if match and match[1] in sources and digest(path)==sources[match[1]]:
            verified[match[1]] = path
    catalog = ActionCatalog(folder/'cache', {k:{'sha256':v} for k,v in sources.items()})
    catalog.load(args.catalog)
    run.update(corpus_sha256=digest(args.corpus), catalog_sha256=digest(args.catalog), catalogued=len(catalog.data['videos']),
               authorized_sources=len(sources), verified_media=len(verified), library_fingerprint=catalog.signature)
    save('sources.json', {k:{'sha256':sources[k],'filename':v.name} for k,v in verified.items()})
    run['seconds']['library_validation'] = time.monotonic()-t
    t = time.monotonic()
    ids = sorted(verified)
    marks = ','.join('?' for _ in ids)
    db = sqlite3.connect(args.corpus.resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    sql = f'''SELECT p.article_id, p.article_title, p.start_ms*0.001, p.end_ms*0.001, p.text
              FROM passage_fts JOIN passages p ON p.passage_id=passage_fts.rowid
              WHERE passage_fts MATCH ? AND p.article_id IN ({marks}) ORDER BY bm25(passage_fts) LIMIT 6'''
    for step in steps:
        actions = []
        for action in action_units(step.text):
            query = build_fts_query(action)
            rows = db.execute(sql, [query, *ids]).fetchall() if query and ids else []
            captions = [dict(zip(('video_id','title','start','end','caption'),r)) for r in rows]
            actions.append(dict(action=action, transcripts=captions, observations=catalog.search(action, ids)))
        save(f'step-{step.number:02}.json', dict(step=step.number, text=step.text, actions=actions,
             selected=None, verification='unverified', narration=False, included=False))
    db.close()
    run['seconds']['discovery'] = time.monotonic()-t
    run.update(processing_status='offline_discovery_complete', steps=len(steps), completed=datetime.now(timezone.utc).isoformat(),
               missing_footage=None, selected=0, verified=0, duration=time.monotonic()-start)
    save('state.json',run)
    print(folder)


if __name__ == '__main__':
    main()
