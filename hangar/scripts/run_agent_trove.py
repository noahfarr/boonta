"""Review or manually execute ONE native Boonta job. Never submits to Slurm."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys

from hangar.agent_trove_experiment import plan, read_preflight


def implementation_digest():
    root=Path(__file__).parents[2]; digest=hashlib.sha256()
    for directory in ('boonta','hangar'):
        for path in sorted((root/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.yaml'):
                digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--split-manifest',required=True)
    p.add_argument('--objective',choices=['clm','blm'],required=True);p.add_argument('--stage',choices=['pilot','main'],default='pilot')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--preflight',type=Path,help='Completed matching preflight run directory containing metrics.npz')
    p.add_argument('--pilot-report',type=Path);p.add_argument('--rl-readiness',type=Path)
    p.add_argument('--resume',type=Path,help='Explicit complete milestone or recovery checkpoint directory')
    p.add_argument('--engineering-only',action='store_true',help='Pilot only: acknowledge unresolved task/repository identities')
    p.add_argument('--execute',action='store_true',help='Execute this job in the current allocation; never submit a job')
    a=p.parse_args()
    cfg,r=plan(Path(a.cache).resolve(),Path(a.tokenizer).resolve(),Path(a.split_manifest).resolve(),a.objective,a.stage,a.seed)
    r.update(implementation_sha256=implementation_digest(),
        versions={name:importlib.metadata.version(name) for name in ('jax','flax','optax','numpy','tokenizers')})
    blockers=[]
    if not r['task_disjoint_certified'] and not (a.stage=='pilot' and a.engineering_only):
        blockers.append('Task/repository identity is not certified; engineering-only pilot is explicit opt-in, main is blocked')
    if a.preflight is None:blockers.append('Matching GPU preflight run directory required')
    else:
        pre=read_preflight(a.preflight)
        if not pre.get('passed') or any(pre.get(k)!=r[k] for k in ('objective','context','model_parameters','tokenizer_sha256','split_manifest_sha256')):
            blockers.append('Preflight does not match this objective/model/context/split')
        r['preflight']=pre
    if a.stage=='main':
        for path,key,label in ((a.pilot_report,'passed','paired pilot'),(a.rl_readiness,'passed','terminal RL readiness')):
            if path is None or not json.loads(path.read_text()).get(key):blockers.append(f'Passing {label} report required')
    if a.resume is not None:
        checkpoint=a.resume.resolve()
        run=next((parent for parent in [checkpoint,*checkpoint.parents] if (parent/'experiment.json').exists()),None)
        if run is None:blockers.append('Resume checkpoint lacks original experiment.json provenance')
        else:
            old=json.loads((run/'experiment.json').read_text())
            for key in ('stage','objective','seed','context','allocated_positions','pool_size','refreshes','global_batch',
                        'tokenizer_sha256','split_manifest_sha256','implementation_sha256','versions'):
                if old.get(key)!=r.get(key):blockers.append(f'Resume changes fixed experiment setting: {key}')
    overrides=r['overrides']+['artisan=agent_trove_pretraining',f'hydra.run.dir={a.run_dir.resolve()}']
    if a.resume is not None:overrides.append(f'checkpoint={a.resume.resolve()}')
    command=[sys.executable,'-m','hangar.train',*overrides]
    r.update(blockers=blockers,command=command)
    print(json.dumps(r,indent=2))
    print('\nCommand (not submitted):\n'+shlex.join(command),flush=True)
    if not a.execute:return
    if blockers:raise SystemExit('Launch blocked: '+'; '.join(blockers))
    import jax
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise SystemExit('One allocated GPU required')
    if a.run_dir.exists():raise SystemExit('Use a new run directory, including on resume')
    a.run_dir.mkdir(parents=True)
    (a.run_dir/'experiment.json').write_text(json.dumps(r,indent=2)+'\n')
    os.execv(sys.executable,command)

if __name__=='__main__':main()
