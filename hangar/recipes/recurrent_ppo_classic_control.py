import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.jaxued import control_generator
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import (ActorCritic, Categorical, FeatureExtractor,
                             Gaussian, Network)

from . import schedules

WIDTH = 64


class SpreadScale(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(x)
        scale = self.param(
            "scale", constant(float(np.log(np.expm1(1.0 - 1e-3)))), (self.action_dim,)
        )
        return jnp.concatenate([mean, jnp.broadcast_to(scale, mean.shape)], axis=-1)


def trunk(output: nn.Module, scale: float) -> nn.Sequential:
    return nn.Sequential(
        [
            nn.Dense(WIDTH, kernel_init=orthogonal(scale)),
            nn.relu,
            nn.Dense(WIDTH, kernel_init=orthogonal(scale)),
            nn.relu,
            output,
        ]
    )


def make(cfg):
    env = environments.make(**cfg.environment)
    space = env.action_space()

    env = SameStepAutoReset(env)
    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    if jnp.issubdtype(space.dtype, jnp.integer):
        actor = Categorical(
            trunk(nn.Dense(space.num_actions, kernel_init=orthogonal(0.01)), 2.0)
        )
    else:
        (action_dim,) = space.shape
        actor = Gaussian(trunk(SpreadScale(action_dim), 2.0))

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [nn.Dense(WIDTH, kernel_init=orthogonal(np.sqrt(2))), nn.relu]
            )
        ),
        torso=instantiate(cfg.stack),
        head=ActorCritic(
            actor=actor,
            critic=trunk(nn.Dense(1, kernel_init=orthogonal(1.0)), 2.0),
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

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(
        algorithm, env, sample=control_generator(cfg.environment.env_id)
    )
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
