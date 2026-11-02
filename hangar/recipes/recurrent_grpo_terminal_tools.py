"""Native, full-parameter GRPO from a Boonta pretraining checkpoint."""
from pathlib import Path

import jax
import numpy as np
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_grpo import RecurrentGRPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks.language_policy import LanguagePolicy
from boonta.utils import load_checkpoint, resolve_checkpoint
from .recurrent_bc_agent_trove import make_network


def checkpoint_params(path):
    path=Path(resolve_checkpoint(path))
    if (path/'algorithm_state').exists():path=path/'algorithm_state'
    saved=load_checkpoint(path)
    params=saved['params'] if isinstance(saved,dict) else saved.params
    return params


def make(cfg):
    env=environments.make(**cfg.environment)
    native=make_network(cfg)
    policy=LanguagePolicy(native,env.pad,env.end,env.max_tokens)
    algorithm=RecurrentGRPO(cfg=instantiate(cfg.algorithm),network=policy,
        optimizer=optax.chain(optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),optax.adam(cfg.optimizer.lr)))
    initialize=algorithm.init
    def warmstart(key,timestep):
        state=initialize(key,timestep)
        params=checkpoint_params(cfg.pretraining_checkpoint)
        if jax.tree.structure(params)!=jax.tree.structure(state.params):
            raise ValueError('Pretraining and RL native Qwen3 parameter structures differ')
        for left,right in zip(jax.tree.leaves(params),jax.tree.leaves(state.params),strict=True):
            if left.shape!=right.shape:raise ValueError('Pretraining parameter shapes differ')
        params=jax.tree.map(lambda a,b:np.asarray(a,dtype=b.dtype),params,state.params)
        return state.replace(params=params,reference_params=params,optimizer_state=algorithm.optimizer.init(params['params']))
    algorithm.init=warmstart
    algorithm,env,pit,lap=instantiate(cfg.curriculum)(algorithm,env)
    env=RecordEpisodeStatistics(env,gamma=cfg.algorithm.gamma)
    return {'algorithm':algorithm,'environment':env,'pit':pit,'lap':lap}
