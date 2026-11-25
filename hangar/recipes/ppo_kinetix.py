import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, Network


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    dtype = (cfg.get("network") or {}).get("dtype")
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        head=ActorCritic(
            actor=Categorical(
                nn.Sequential(
                    [
                        nn.Dense(256, dtype=dtype),
                        nn.tanh,
                        nn.Dense(256, dtype=dtype),
                        nn.tanh,
                        nn.Dense(num_actions, dtype=jnp.float32),
                    ]
                )
            ),
            critic=nn.Sequential(
                [
                    nn.Dense(256, dtype=dtype),
                    nn.tanh,
                    nn.Dense(256, dtype=dtype),
                    nn.tanh,
                    nn.Dense(1, dtype=jnp.float32),
                ]
            ),
        ),
    )

    algorithm = PPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(cfg.optimizer.lr),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
