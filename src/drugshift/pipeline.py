"""One-click orchestration with immutable configuration IDs and restart-safe jobs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from .prepare import prepare,checksum
from .experiment import compute_job,atomic_json


def stable_hash(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode('utf8')).hexdigest()

def launch(root,config,skip_download=False):
    root=Path(root).resolve()
    cfg=json.loads(Path(config).read_text(encoding='utf8'))
    assert cfg['members']>=2 and cfg['epochs']>=1 and 0<cfg['alpha']<1
    root.joinpath('logs').mkdir(exist_ok=True)
    log_path=root/'logs/pipeline.log'
    # Append, never truncate: diagnostics survive a sudden power failure.
    class Tee:
        def __init__(self,primary,log):self.primary=primary;self.log=log
        def write(self,s):self.primary.write(s);self.log.write(s);self.log.flush()
        def flush(self):self.primary.flush();self.log.flush()
    with open(log_path,'a',encoding='utf8',buffering=1) as f:
        orig_stdout,orig_stderr=sys.stdout,sys.stderr
        sys.stdout=Tee(sys.stdout,f);sys.stderr=Tee(sys.stderr,f)
        try:
            print(f'\n==== DrugShift-UQ started {datetime.now(timezone.utc).isoformat()} ====',flush=True)
            print('[system]',platform.platform(),sys.version,flush=True)
            if cfg['run_tests']:
                print('[tests] Running self-tests before processing real experiments',flush=True)
                subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=root,check=True)
            if skip_download:
                print('[dev] Using preprocessed local data; this mode is never for public-dataset claims.',flush=True)
                dpath=root/'data/processed'
                if not (dpath/'responses.csv').exists():raise ValueError('Missing local fixture')
            else:prepare(root,cfg)
            # Real dataset identity and all feature values, not just timestamps, fingerprint runs.
            data_hash={fn:checksum(root/'data/processed'/fn,'sha256') for fn in
                       ['responses.csv','cell_features.csv','drug_features.csv']}
            code_hash={p.relative_to(root).as_posix():checksum(p,'sha256') for p in
                       sorted((root/'src/drugshift').glob('*.py'))}
            core={k:v for k,v in cfg.items() if k not in ['force_recompute','run_tests']}
            run_id=stable_hash({'config':core,'data':data_hash,'code':code_hash})[:12]
            run_dir=root/'results/runs'/run_id;run_dir.mkdir(parents=True,exist_ok=True)
            atomic_json({'config':cfg,'data_sha256':data_hash,'code_sha256':code_hash,
                         'run_id':run_id,'python':sys.version,'platform':platform.platform(),
                         'started_utc':datetime.now(timezone.utc).isoformat()},run_dir/'run_manifest.json')
            jobs=[]
            for seed in cfg['seeds']:
                for regime in cfg['regimes']:jobs.append((regime,'full',seed))
            for regime in cfg['ablation_regimes']:
                for variant in cfg['ablations']:jobs.append((regime,variant,cfg['ablation_seed']))
            print(f'[schedule] {len(jobs)} jobs; resume ID {run_id}.',flush=True)
            for n,(regime,variant,seed) in enumerate(jobs,1):
                print(f'\n[progress] {n}/{len(jobs)}: {regime}/{variant}/{seed}',flush=True)
                job_hash=stable_hash({'run_id':run_id,'regime':regime,'variant':variant,'seed':seed})
                out=run_dir/'jobs'/regime/f'seed_{seed}'/variant
                if cfg['force_recompute'] and out.exists():
                    import shutil
                    print(f'[force] Deleting only this requested job folder: {out}')
                    shutil.rmtree(out)
                compute_job(root,run_id,regime,variant,seed,cfg,job_hash)
            from .reporting import generate_report
            generate_report(root,run_id)
            (root/'results'/'LATEST_RUN.txt').write_text(run_id,encoding='utf8')
            print(f'[done] results/FINAL_REPORT.html; run ID {run_id}',flush=True)
            return run_id
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout,sys.stderr=orig_stdout,orig_stderr


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--config',default='config.json')
    p.add_argument('--skip-download',action='store_true',help='Unit/smoke testing only, not final research')
    a=p.parse_args();launch(a.root,Path(a.root)/a.config,a.skip_download)
if __name__=='__main__':main()
