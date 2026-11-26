import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, Network
from boonta.networks.layers import Flatten

from . import schedules


def tower(outputs, dtype):
    return nn.Sequential(
        [
            lambda obs: jnp.moveaxis(obs, -3, -1).astype(dtype or jnp.float32) / 255.0,
            nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID", dtype=dtype),
            nn.relu,
            nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID", dtype=dtype),
            nn.relu,
            nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID", dtype=dtype),
            nn.relu,
            Flatten(start_dim=-3),
            nn.Dense(512, dtype=dtype),
            nn.relu,
            nn.Dense(outputs, dtype=jnp.float32),
        ]
    )


def make(cfg):
    import ale_py
    import gymnasium

    gymnasium.register_envs(ale_py)
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    dtype = (cfg.get("network") or {}).get("dtype")

    network = Network(
        head=ActorCritic(
            actor=Categorical(tower(num_actions, dtype)),
            critic=tower(1, dtype),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps
    )

    algorithm = PPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(learning_rate, eps=cfg.optimizer.get("eps", 1e-8)),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
