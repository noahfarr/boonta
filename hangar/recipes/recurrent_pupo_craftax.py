import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from optax.contrib._muon import MuonDimensionNumbers

from boonta import environments
from boonta.algorithms.recurrent_pupo import RecurrentPuPO
from boonta.environments.wrappers import OptimisticAutoReset, RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network

from . import schedules


def make(cfg):
    dtype = (cfg.get("network") or {}).get("dtype")
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    hidden_dim = cfg.cell.features

    env = OptimisticAutoReset(
        env, num_envs=cfg.environment.num_envs, ratio=cfg.environment.reset_ratio
    )

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    lambda obs: obs.astype(jnp.float32),
                    nn.Dense(hidden_dim, use_bias=False, dtype=dtype),
                ]
            ),
        ),
        torso=instantiate(cfg.stack),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions, use_bias=False, dtype=dtype)),
            critic=nn.Dense(1, use_bias=False, dtype=dtype),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.rollout.num_steps
    )

    algorithm = RecurrentPuPO(
        cfg=instantiate(cfg.algorithm),
        importance_exponent=instantiate(cfg.importance_exponent),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.multi_transform(
                {
                    "muon": optax.contrib.muon(
                        learning_rate,
                        muon_weight_dimension_numbers=MuonDimensionNumbers(-2, -1),
                    ),
                    "adam": optax.adam(learning_rate),
                },
                lambda params: jax.tree.map(
                    lambda p: "muon" if p.ndim >= 2 else "adam", params
                ),
            )
            if cfg.optimizer.get("name") == "muon"
            else optax.adam(learning_rate),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
