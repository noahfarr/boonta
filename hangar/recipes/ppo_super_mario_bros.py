import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from optax.contrib._muon import MuonDimensionNumbers

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.pynes import super_mario_bros
from boonta.environments.wrappers import (LogInfo, RecordEpisodeStatistics,
                                          TransformReward)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network

from . import schedules
from .recurrent_ppo_super_mario_bros import Memory, NatureCNN


def make(cfg):
    dtype = cfg.network.get("dtype")
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    senses = NatureCNN if cfg.environment.kwargs.get("pixels", True) else Memory
    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=senses(features=cfg.network.features, dtype=dtype)
        ),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions, use_bias=False, dtype=dtype)),
            critic=nn.Dense(1, use_bias=False, dtype=dtype),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.rollout.num_steps
    )

    algorithm = PPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.contrib.muon(
                learning_rate,
                beta=cfg.optimizer.beta,
                weight_decay=cfg.optimizer.weight_decay,
                muon_weight_dimension_numbers=lambda params: jax.tree.map(
                    lambda p: MuonDimensionNumbers(-2, -1) if p.ndim >= 2 else None,
                    params,
                ),
            ),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = LogInfo(env, keys=super_mario_bros.KEYS)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    env = TransformReward(
        env, fn=jnp.sign if cfg.clip_reward else lambda reward: reward / cfg.reward_scale
    )
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
