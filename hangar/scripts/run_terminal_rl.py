"""Review or manually execute one terminal GRPO run from a Boonta checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from hangar.recipes.recurrent_grpo_terminal_tools import checkpoint_params
from hangar.scripts.run_agent_trove import implementation_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--condition',choices=['clm','blm'],required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--budget',type=int,default=100000);p.add_argument('--evaluation-interval',type=int,default=5000)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.budget%a.evaluation_interval or a.evaluation_interval%40:raise ValueError('Budget and evaluation interval must align to 40 environment slots per GRPO rollout')
    overrides=['algorithm=recurrent_grpo','environment=terminal_tools','artisan=terminal_rl','scoring=final',
        f'pretraining_checkpoint={a.checkpoint.resolve()}',f'environment.kwargs.tokenizer_id={a.tokenizer.resolve()}',
        f'seed={a.seed}',f'num_steps={a.budget}',f'num_epochs={a.budget//a.evaluation_interval}',f'hydra.run.dir={a.run_dir.resolve()}']
    command=[sys.executable,'-m','hangar.train',*overrides]
    protocol=dict(condition=a.condition,seed=a.seed,budget_environment_decision_slots=a.budget,
        evaluation_interval=a.evaluation_interval,pretraining_checkpoint=str(a.checkpoint.resolve()),
        task_split='train versus evaluation namespaces; held-out numeric parameter ranges',
        implementation_sha256=implementation_digest(),command=command,
        note='One action is one assistant turn, at most one executable command. Inactive group slots count toward the fixed rollout budget; executed commands are logged separately.')
    print(json.dumps(protocol,indent=2));print(shlex.join(command),flush=True)
    if not a.execute:return
    import jax
    if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise ValueError('One allocated GPU required')
    if a.run_dir.exists():raise ValueError('Use a new isolated RL run directory')
    checkpoint_params(a.checkpoint)  # Check readability before starting the trainer.
    a.run_dir.mkdir(parents=True)
    (a.run_dir/'rl_experiment.json').write_text(json.dumps(protocol,indent=2)+'\n')
    os.execv(sys.executable,command)

if __name__=='__main__':main()
