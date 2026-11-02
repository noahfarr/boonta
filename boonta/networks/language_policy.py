"""Autoregressive tool-action adapter around the unchanged pretrained Network.

Observations are complete token histories, not new role features. GRPO treats a
whole generated assistant turn as one composite action. Native parameter paths
and the single vocabulary projection are preserved.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp


def generate(network,params,tokens,lengths,key,max_tokens,pad_id,end_id,temperature=1.0):
    batch,context=tokens.shape
    done=jnp.zeros(tokens.shape,bool).at[:,0].set(True)
    carry,dist=network.apply(params,tokens,jnp.zeros_like(tokens),
        jnp.zeros(tokens.shape,jnp.float32),done,carry=None,temperature=temperature)
    # Prefill is padded on the right. Invalidate those KV slots and restore each
    # row's true next position before autoregressive decoding.
    carry=tuple(c.replace(position_offset=lengths[:,None]) for c in carry)
    logits=dist.at(lengths-1).logits
    def step(state,index):
        carry,logits,active,logp,key=state
        key,sample_key=jax.random.split(key)
        value=jnp.where(temperature>0,jax.random.categorical(sample_key,logits),jnp.argmax(logits,-1)).astype(jnp.int32)
        live=active & (lengths+index<context)
        token=jnp.where(live,value,pad_id)
        selected=jnp.take_along_axis(jax.nn.log_softmax(logits,-1),value[:,None],-1)[:,0]
        logp=logp+jnp.where(live,selected,0.)
        carry,next_dist=network.apply(params,token[:,None],jnp.zeros((batch,1),jnp.int32),
            jnp.zeros((batch,1),jnp.float32),jnp.zeros((batch,1),bool),carry=carry,temperature=temperature)
        logits=next_dist.at(jnp.zeros(batch,jnp.int32)).logits
        active=live & (token!=end_id)
        return (carry,logits,active,logp,key),(token,live)
    (_,_,_,logp,_),(actions,active)=jax.lax.scan(step,
        (carry,logits,jnp.ones(batch,bool),jnp.zeros(batch,jnp.float32),key),jnp.arange(max_tokens))
    return actions.T,active.T,logp


class TurnDistribution:
    def __init__(self,policy,params,obs,temperature):
        self.policy,self.params,self.obs,self.temperature=policy,params,obs,temperature
        self._entropy=None

    def sample_and_log_prob(self,seed):
        tokens=self.obs['tokens']; lengths=self.obs['length']
        shape=lengths.shape
        actions,_,logp=generate(self.policy.network,self.params,
            tokens.reshape(-1,tokens.shape[-1]),lengths.reshape(-1),seed,
            self.policy.max_tokens,self.policy.pad_id,self.policy.end_id,self.temperature)
        return actions.reshape(*shape,self.policy.max_tokens),logp.reshape(shape)

    def sample(self,seed): return self.sample_and_log_prob(seed)[0]

    def log_prob(self,actions):
        prefix=self.obs['tokens']; lengths=self.obs['length']; shape=lengths.shape
        context=prefix.shape[-1]; n=lengths.size
        tokens=prefix.reshape(n,context); lengths=lengths.reshape(-1)
        actions=actions.reshape(n,self.policy.max_tokens)
        pos=jnp.arange(self.policy.max_tokens)[None,:]
        ended=jnp.cumsum(actions==self.policy.end_id,axis=-1)-(actions==self.policy.end_id)
        active=(ended==0)&(lengths[:,None]+pos<context)
        slots=jnp.minimum(lengths[:,None]+pos,context-1)
        # Masking the tail prevents padding tokens from overwriting real prefix
        # positions when a context boundary is reached.
        def insert(i,array):
            old=array[jnp.arange(n),slots[:,i]]
            return array.at[jnp.arange(n),slots[:,i]].set(jnp.where(active[:,i],actions[:,i],old))
        tokens=jax.lax.fori_loop(0,self.policy.max_tokens,insert,tokens)
        x=tokens
        y=jnp.concatenate([tokens[:,1:],jnp.full((n,1),self.policy.pad_id,jnp.int32)],axis=1)
        done=jnp.zeros(x.shape,bool).at[:,0].set(True)
        _,dist=self.policy.network.apply(self.params,x,jnp.zeros_like(x),jnp.zeros(x.shape,jnp.float32),done,
                                         carry=None,temperature=1.0)
        logp,entropy=dist.statistics(y)
        target_positions=jnp.arange(context)[None,:]
        action_length=active.sum(-1)
        weight=(target_positions>=lengths[:,None]-1)&(target_positions<lengths[:,None]+action_length[:,None]-1)
        self._entropy=(entropy*weight).sum(-1).reshape(shape)
        return (logp*weight).sum(-1).reshape(shape)

    def entropy(self):
        if self._entropy is None:
            raise ValueError('Evaluate log_prob(actions) before conditional entropy')
        # GRPO logs this on sampled action prefixes. It is a conditional-entropy
        # diagnostic, not an extra objective or exact joint entropy computation.
        return self._entropy


@dataclass
class LanguagePolicy:
    network: object
    pad_id: int
    end_id: int
    max_tokens: int = 128

    def initialize_carry(self,key,input_shape): return None

    def init(self,key,obs,action=None,reward=None,done=None,carry=None,temperature=1.0):
        batch=obs['tokens'].shape[0]
        x=jnp.full((batch,1),self.pad_id,jnp.int32)
        return self.network.init(key,x,jnp.zeros_like(x),jnp.zeros(x.shape,jnp.float32),
                                 jnp.ones(x.shape,bool),carry=None,temperature=1.0)

    def apply(self,params,obs,action=None,reward=None,done=None,carry=None,temperature=1.0,mutable=False):
        result=(None,TurnDistribution(self,params,obs,temperature))
        return (result,{}) if mutable else result
