"""Independent processing worker. Mount settings privately; never commit secrets."""
from pathlib import Path
import argparse
import os
import tomllib
from runtime_loader import load_runtime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--settings',type=Path,default=Path(os.environ.get('STUDIO_SETTINGS_FILE','/run/secrets/studio_settings')))
    parser.add_argument('--jobs',type=Path,default=Path(os.environ.get('STUDIO_JOBS_ROOT','/data/jobs')))
    args=parser.parse_args()
    config=tomllib.loads(args.settings.read_text(encoding='utf-8'))
    required=['OPENAI_API_KEY','ELEVENLABS_API_KEY','google_service_account','OUTPUT_FOLDER_ID','LIBRARY_FOLDER_ID']
    if any(k not in config for k in required):raise ValueError('Worker secure settings are incomplete.')
    runtime=load_runtime();config['ENGINE_BUILD']=runtime.revision
    jobs=runtime.Jobs(args.jobs,worker_config=config,external_worker=True)
    try:jobs.work_forever()
    except KeyboardInterrupt:jobs.stop_event.set()


if __name__=='__main__':main()
