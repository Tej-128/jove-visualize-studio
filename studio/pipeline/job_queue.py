"""Persistent single-active-job queue. SQLite contains no provider credentials.

The interface and a separate worker can share this database on a persistent volume.
Free-host mode also uses it, but loss of that host's disk still requires Drive restore.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import sqlite3
import time

SAFE_CONFIG={'voice','TEXT_MODEL','VISION_MODEL','MAX_VISION_CALLS','ENGINE_BUILD',
             'VISUAL_REVIEW_SECONDS','VISUAL_WORKERS','AUTOMATIC_EDITOR','MAX_JOB_SECONDS',
             'LIBRARY_FOLDER_ID','OUTPUT_FOLDER_ID','HOSTED_SHORTLIST','ALLOW_AI_ILLUSTRATIONS'}


class JobQueue:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, fingerprint TEXT NOT NULL, kind TEXT NOT NULL, config TEXT NOT NULL, build TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, lease_until REAL NOT NULL DEFAULT 0, worker TEXT, attempts INTEGER NOT NULL DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS runtime (key TEXT PRIMARY KEY,value TEXT NOT NULL)')

    def activate_build(self,build):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("INSERT INTO runtime VALUES('build',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(build,))
            db.execute("UPDATE jobs SET build=? WHERE status='queued'",(build,))

    @contextmanager
    def connection(self):
        db=sqlite3.connect(self.path,timeout=15);db.row_factory=sqlite3.Row
        try:yield db;db.commit()
        except BaseException:db.rollback();raise
        finally:db.close()

    def enqueue(self,ident,owner,text,config,kind='video'):
        if config.get('ALLOW_AI_ILLUSTRATIONS', False) is not False:
            raise ValueError('This queue accepts JoVE-only jobs.')
        safe={k:config[k] for k in SAFE_CONFIG if k in config}
        payload=json.dumps(safe,sort_keys=True)
        fingerprint=hashlib.sha256(json.dumps([owner,text,safe,kind],sort_keys=True).encode()).hexdigest()
        now=time.time()
        build=str(config.get('ENGINE_BUILD',''))
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            active=db.execute("SELECT value FROM runtime WHERE key='build'").fetchone()
            if active is not None and active['value']!=build:raise ValueError('The editor was updated. Refresh the page before starting this job.')
            prior=db.execute("SELECT id FROM jobs WHERE fingerprint=? AND status IN ('queued','running')",(fingerprint,)).fetchone()
            if prior:return prior['id']
            row=db.execute('SELECT status,lease_until FROM jobs WHERE id=?',(ident,)).fetchone()
            if row and row['status']=='running' and row['lease_until']>now:raise ValueError('This project is already processing.')
            db.execute("INSERT INTO jobs(id,owner,fingerprint,kind,config,build,status,created) VALUES(?,?,?,?,?,?,'queued',?) ON CONFLICT(id) DO UPDATE SET config=excluded.config,build=excluded.build,status='queued',lease_until=0,worker=NULL,created=excluded.created,fingerprint=excluded.fingerprint",
                (ident,owner,fingerprint,kind,payload,str(config.get('ENGINE_BUILD','')),now))
        return ident

    def claim(self,worker,build,lease=90):
        now=time.time()
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            active=db.execute("SELECT value FROM runtime WHERE key='build'").fetchone()
            if active is not None and active['value']!=build:return None
            if db.execute("SELECT 1 FROM jobs WHERE status='running' AND lease_until>?",(now,)).fetchone():return None
            # Interrupted jobs can be recovered by the current worker build. No
            # old and new engines can claim the same live lease concurrently.
            db.execute("UPDATE jobs SET status='queued',worker=NULL WHERE status='running' AND lease_until<=?",(now,))
            row=db.execute("SELECT * FROM jobs WHERE status='queued' AND (build=? OR attempts>0) ORDER BY created LIMIT 1",(build,)).fetchone()
            if row is None:return None
            db.execute("UPDATE jobs SET status='running',worker=?,lease_until=?,attempts=attempts+1 WHERE id=?",(worker,now+lease,row['id']))
            return {**dict(row),'config':json.loads(row['config'])}

    def heartbeat(self,ident,worker,lease=90):
        with self.connection() as db:
            return db.execute("UPDATE jobs SET lease_until=? WHERE id=? AND worker=? AND status='running'",(time.time()+lease,ident,worker)).rowcount==1

    def finish(self,ident,worker,status):
        if status not in {'complete','needs_review','failed'}:status='failed'
        with self.connection() as db:
            db.execute('UPDATE jobs SET status=?,lease_until=0 WHERE id=? AND worker=?',(status,ident,worker))

    def get(self,ident,owner):
        with self.connection() as db:row=db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone()
        if row is None:return None
        if row['owner']!=owner:raise PermissionError('This project belongs to another user')
        return dict(row)
