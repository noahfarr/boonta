"""Fixed procedural evaluation and conditional policy KL for terminal GRPO."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boonta.artisans import Metrics
from boonta.environments.terminal_tools import TerminalEpisode
from boonta.datasets.agent_trove import encode_messages,serialize
from boonta.networks.language_policy import generate


def history(episode,tokenizer,context,max_tokens):
    tokens,_=serialize(encode_messages({'messages':episode.messages},tokenizer))
    header=tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False).ids
    values=np.concatenate([tokens,np.array(header,np.int32)])
    if len(values)+max_tokens>context:raise ValueError('Evaluation would truncate an RL history')
    result=np.full((1,context),tokenizer.token_to_id('<|endoftext|>'),np.int32)
    result[0,:len(values)]=values
    return jnp.asarray(result),jnp.asarray([len(values)],jnp.int32)


class TerminalEvaluation:
    def __init__(self,tokenizer,context,directory,episodes=64,seed=271828,max_tokens=128):
        from tokenizers import Tokenizer
        self.tokenizer=Tokenizer.from_file(str(tokenizer));self.context=context
        self.directory=Path(directory);self.episodes=episodes;self.seed=seed;self.max_tokens=max_tokens
        self.decode=None;self.kl=None

    def craft(self,key,algorithm,environment,state,logs):
        native=algorithm.network.network
        pad=self.tokenizer.token_to_id('<|endoftext|>');end=self.tokenizer.token_to_id('<|im_end|>')
        if self.decode is None:
            self.decode=jax.jit(lambda params,tokens,lengths,key:generate(native,params,tokens,lengths,key,self.max_tokens,pad,end,0.0))
            @jax.jit
            def kl(params,reference,tokens,lengths):
                done=jnp.zeros(tokens.shape,bool).at[:,0].set(True)
                zero=jnp.zeros_like(tokens);reward=jnp.zeros(tokens.shape,jnp.float32)
                _,p=native.apply(params,tokens,zero,reward,done,carry=None,temperature=1.0)
                _,q=native.apply(reference,tokens,zero,reward,done,carry=None,temperature=1.0)
                lp=jax.nn.log_softmax(p.at(lengths-1).logits,-1)
                lq=jax.nn.log_softmax(q.at(lengths-1).logits,-1)
                return (jnp.exp(lp)*(lp-lq)).sum(-1).mean()
            self.kl=kl
        outcomes=[];kls=[];valid=turns=commands=0
        for index in range(self.episodes):
            episode=TerminalEpisode(self.seed+index,'evaluation',5)
            try:
                tokens,lengths=history(episode,self.tokenizer,self.context,self.max_tokens)
                kls.append(float(self.kl(state.algorithm_state.params,state.algorithm_state.reference_params,tokens,lengths)))
                while not episode.done:
                    tokens,lengths=history(episode,self.tokenizer,self.context,self.max_tokens)
                    values,active,_=self.decode(state.algorithm_state.params,tokens,lengths,jax.random.key(self.seed+index))
                    ids=np.asarray(values)[0][np.asarray(active)[0]].tolist()
                    ended=bool(ids and ids[-1]==end)
                    text=self.tokenizer.decode(ids[:-1] if ended else ids,skip_special_tokens=False)
                    if not ended:text='INVALID: missing assistant handoff'
                    _,_,_,info=episode.step(text)
                    valid+=info['valid_call'];commands+=info['executed_commands'];turns+=1
                outcomes.append(dict(task_id=episode.task_id,kind=episode.kind,success=episode.success,turns=episode.turns))
            finally:episode.close()
        metrics={'evaluation/task_success':sum(x['success'] for x in outcomes)/len(outcomes),
                 'evaluation/valid_tool_call_rate':valid/max(turns,1),
                 'evaluation/fixed_history_policy_kl':float(np.mean(kls)),
                 'evaluation/executed_commands':commands,'evaluation/decision_turns':turns}
        step=int(state.algorithm_state.step)
        self.directory.mkdir(parents=True,exist_ok=True)
        (self.directory/f'evaluation-{step}.json').write_text(json.dumps(dict(step=step,seed=self.seed,metrics=metrics,tasks=outcomes,
            kl_definition='Full-vocabulary next-token KL(current || own pretrained reference) at fixed initial evaluation histories'),indent=2)+'\n')
        return Metrics('terminal',metrics)
