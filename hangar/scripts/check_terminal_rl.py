"""Summarize paired terminal learning curves and test whether the probe is informative."""
import argparse
import json
from pathlib import Path

import numpy as np

from hangar.scripts.check_agent_trove_pilot import evaluations


def curve(paths):
    rows=evaluations(paths)
    steps=sorted(rows)
    if not steps or steps[0]!=0:raise ValueError('Zero-shot evaluation required')
    success=np.array([rows[s]['metrics']['evaluation/task_success'] for s in steps])
    if not np.isfinite(success).all():raise ValueError('Nonfinite success curve')
    auc=float(np.trapezoid(success,steps))
    variation=False
    for directory in paths:
        p=Path(directory)/'metrics.npz'
        if p.exists():
            with np.load(p) as data:
                if 'actor/group_return_std' in data and np.nanmax(data['actor/group_return_std'])>0:variation=True
    return dict(steps=steps,success=success.tolist(),zero_shot=float(success[0]),final_success=float(success[-1]),
                auc=auc,normalized_auc=auc/steps[-1] if steps[-1] else None,
                observed_training_reward_variation=variation,
                tasks={str(s):[t['task_id'] for t in rows[s]['tasks']] for s in steps})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clm',nargs='+',type=Path,required=True);p.add_argument('--blm',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path)
    a=p.parse_args();left,right=curve(a.clm),curve(a.blm)
    reasons=[]
    if left['steps']!=right['steps']:reasons.append('Evaluation budgets differ')
    if left['tasks']!=right['tasks']:reasons.append('Held-out task instances differ')
    for name,row in (('clm',left),('blm',right)):
        if not row['observed_training_reward_variation']:reasons.append(f'{name} probe has no observed group reward variation')
    result=dict(passed=not reasons,reasons=reasons,clm=left,blm=right,
                normalized_auc_difference=right['normalized_auc']-left['normalized_auc'],
                uncertainty='One paired run is exploratory; no uncertainty interval is inferred from one pair.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots()
        for name,row in (('CLM',left),('BLM',right)):axis.plot(row['steps'],row['success'],marker='o',label=name)
        axis.set(xlabel='Environment decision slots',ylabel='Held-out task success',ylim=(0,1));axis.legend()
        fig.tight_layout();fig.savefig(a.plot)
    print(json.dumps(result,indent=2))
    if reasons:raise SystemExit(1)

if __name__=='__main__':main()
