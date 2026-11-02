"""Compare one paired pilot; never starts another job."""
import argparse
import json
from pathlib import Path

import numpy as np


def evaluations(paths):
    result={}
    for directory in paths:
        for p in Path(directory).glob('evaluation/evaluation-*.json'):
            row=json.loads(p.read_text())
            if row['step'] in result and row!=result[row['step']]:raise ValueError('Conflicting checkpoint evaluations')
            result[row['step']]=row
    return result


def compare(clm,blm):
    left,right=evaluations(clm),evaluations(blm)
    reasons=[]
    if not left or not right:raise ValueError('Both pilot evaluation series are required')
    if sorted(left)!=sorted(right):reasons.append('Checkpoint steps differ')
    steps=sorted(set(left)&set(right))
    if 0 not in steps or 2**28 not in steps:reasons.append('Pilot must include initialization and the complete 2^28 budget')
    if len(steps)!=6:reasons.append('Six scheduled checkpoint evaluations required')
    if 0 in steps and left[0].get('initial_parameters_sha256')!=right[0].get('initial_parameters_sha256'):
        reasons.append('Initial weights differ')
    for step in steps:
        if left[step]['row_ids']!=right[step]['row_ids']:reasons.append(f'Validation histories differ at {step}')
        for row in (left[step],right[step]):
            for key in ('validation/assistant_nll','validation/command_nll','validation/observation_nll'):
                if not np.isfinite(row['metrics'][key]):reasons.append(f'Nonfinite {key} at {step}')
    for name,series in (('clm',left),('blm',right)):
        if 0 in series and 2**28 in series:
            if series[2**28]['metrics']['validation/assistant_nll']>=series[0]['metrics']['validation/assistant_nll']:
                reasons.append(f'{name} held-out action NLL did not decrease')
    plans=[]
    for paths in (clm,blm):
        values=[json.loads((Path(p)/'experiment.json').read_text()) for p in paths if (Path(p)/'experiment.json').exists()]
        if not values:raise ValueError('Experiment provenance is missing')
        plans.append(values[0])
    for key in ('seed','context','allocated_positions','global_batch','split_manifest_sha256','tokenizer_sha256','implementation_sha256','versions'):
        if plans[0].get(key)!=plans[1].get(key):reasons.append(f'Paired protocol mismatch: {key}')
    return dict(passed=not reasons,reasons=reasons,steps=steps,task_disjoint_certified=all(plans[i]['task_disjoint_certified'] for i in (0,1)),
        clm_run_dirs=list(map(str,clm)),blm_run_dirs=list(map(str,blm)),
        data={name:[dict(step=s,assistant_nll=series[s]['metrics']['validation/assistant_nll'],
                        command_nll=series[s]['metrics']['validation/command_nll'],valid_tool_call_rate=series[s]['metrics']['validation/valid_tool_call_rate']) for s in steps]
              for name,series in (('clm',left),('blm',right))})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clm',nargs='+',type=Path,required=True);p.add_argument('--blm',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path)
    a=p.parse_args();r=compare(a.clm,a.blm)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots()
        for name,rows in r['data'].items():axis.plot([x['step'] for x in rows],[x['assistant_nll'] for x in rows],marker='o',label=name.upper())
        axis.set(xlabel='Allocated positions (matched compute proxy)',ylabel='Held-out assistant-token NLL (nats)')
        axis.legend();fig.tight_layout();fig.savefig(a.plot)
    print(json.dumps(r,indent=2))
    if not r['passed']:raise SystemExit(1)

if __name__=='__main__':main()
