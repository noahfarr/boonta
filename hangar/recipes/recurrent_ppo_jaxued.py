import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network

from . import schedules


class MazeEncoder(nn.Module):
    @nn.compact
    def __call__(self, obs):
        image = nn.Conv(16, (3, 3), padding="VALID")(obs["image"].astype(jnp.float32))
        image = nn.relu(image.reshape(*image.shape[:-3], -1))
        direction = jax.nn.one_hot(obs["agent_dir"], 4)
        direction = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)))(direction)
        return jnp.concatenate([image, direction], axis=-1)


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions

    env = SameStepAutoReset(env)
    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=FeatureExtractor(observation_extractor=MazeEncoder()),
        torso=instantiate(cfg.stack),
        head=ActorCritic(
            actor=Categorical(
                nn.Sequential(
                    [
                        nn.Dense(32, kernel_init=orthogonal(2.0)),
                        nn.relu,
                        nn.Dense(num_actions, kernel_init=orthogonal(0.01)),
                    ]
                )
            ),
            critic=nn.Sequential(
                [
                    nn.Dense(32, kernel_init=orthogonal(2.0)),
                    nn.relu,
                    nn.Dense(1, kernel_init=orthogonal(1.0)),
                ]
            ),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.rollout.num_steps
    )

    algorithm = RecurrentPPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(learning_rate),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
