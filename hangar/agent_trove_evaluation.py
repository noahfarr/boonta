"""Checkpoint metrics on a fixed grouped holdout; sum NLL before dividing."""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer

from boonta.artisans import Metrics
from boonta.datasets.agent_trove import AgentTrove, windows
from boonta.datasets.agent_trove_layout import masks
from boonta.datasets.agent_trove_actions import parse_action
from boonta.networks.language_policy import generate


class AgentTroveEvaluation:
    def __init__(self,cache,tokenizer,split_manifest,context,directory,rows=256,seed=8675309,generation_rows=16,max_tokens=128):
        self.tokenizer=Tokenizer.from_file(str(tokenizer))
        self.dataset=AgentTrove(cache_dir=cache,tokenizer=self.tokenizer,split='validation',
            split_manifest=split_manifest,length=context,pool_size=1)
        self.context,self.count,self.seed=context,rows,seed
        self.directory=Path(directory);self.generation_rows=generation_rows;self.max_tokens=max_tokens
        self.selected=None;self.forward=None;self.decode=None

    def prepare(self):
        if self.selected is None:
            self.selected=[]
            for row in self.dataset.rows.sample_rows(self.context,self.count,self.seed):
                target_masks,messages,spans=masks(row['_tokens'],row['_mask'],self.tokenizer)
                transition,=windows(row,self.tokenizer,self.context,self.dataset.pad_id,False)
                self.selected.append((row['index'],transition,target_masks,messages,spans))
            if not self.selected: raise ValueError('No eligible grouped validation records')
        return self.selected

    def craft(self,key,algorithm,environment,state,logs):
        self.prepare()
        network=algorithm.network
        if self.forward is None:
            @jax.jit
            def forward(params,batch):
                t=batch.first
                _,dist=network.apply(params,t.obs[None],t.action[None],t.reward[None],t.done[None],carry=None,temperature=1.0)
                return dist.log_prob(batch.second.action[None])[0]
            self.forward=forward
            self.decode=jax.jit(lambda params,tokens,lengths,key:generate(network,params,tokens,lengths,key,
                self.max_tokens,self.dataset.pad_id,self.tokenizer.token_to_id('<|im_end|>'),0.0))
        totals={name:[0.,0] for name in ('assistant','command','observation','valid','nonassistant')}
        valid_calls=generated=0
        for ordinal,(index,batch,target_masks,messages,spans) in enumerate(self.selected):
            logp=np.asarray(self.forward(state.algorithm_state.params,batch))
            target_masks['nonassistant']=target_masks['valid'] & ~target_masks['assistant']
            for name,mask in target_masks.items():
                count=len(mask);total=-float(logp[:count][mask].sum(dtype=np.float64))
                totals[name][0]+=total;totals[name][1]+=int(mask.sum())
            if ordinal<self.generation_rows:
                prefix=next(start for role,start,stop in spans if role=='assistant')
                prompt=np.full((1,self.context),self.dataset.pad_id,np.int32)
                prompt[0,:prefix]=np.asarray(batch.first.obs)[:prefix]
                actions,active,_=self.decode(state.algorithm_state.params,jnp.asarray(prompt),jnp.array([prefix],jnp.int32),
                    jax.random.fold_in(jax.random.key(self.seed),index))
                values=np.asarray(actions)[0][np.asarray(active)[0]].tolist()
                ended=bool(values and values[-1]==self.tokenizer.token_to_id('<|im_end|>'))
                if ended: values=values[:-1]
                text=self.tokenizer.decode(values,skip_special_tokens=False)
                try:
                    parse_action(text)
                    valid_calls+=int(ended)
                except ValueError: pass
                generated+=1
        metrics={}
        for name,(total,count) in totals.items():
            metrics[f'validation/{name}_nll_sum']=total
            metrics[f'validation/{name}_targets']=count
            metrics[f'validation/{name}_nll']=total/count if count else float('nan')
        metrics.update({'validation/valid_tool_call_rate':valid_calls/generated if generated else float('nan'),
                        'validation/generations':generated,'validation/rows':len(self.selected)})
        step=int(state.algorithm_state.step)
        report=dict(step=step,seed=self.seed,row_ids=[item[0] for item in self.selected],metrics=metrics,
            task_disjoint_certified=self.dataset.rows.split_manifest['task_disjoint_certified'],
            observation_definition='user/tool bodies including their end markers; nonassistant additionally includes framing/system')
        if step==0:
            digest=hashlib.sha256()
            for leaf in jax.tree.leaves(state.algorithm_state.params):digest.update(np.asarray(leaf).tobytes())
            report['initial_parameters_sha256']=digest.hexdigest()
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/f'evaluation-{step}.json').write_text(json.dumps(report,indent=2)+'\n')
        return Metrics('heldout',metrics)
