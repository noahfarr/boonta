import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.wrappers import RecordMultiAgentEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, Network
from boonta.networks.layers import Flatten

from . import schedules


class ViewAndEnergy(nn.Module):
    features: int

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        view = Flatten(start_dim=-3)(obs["view"].astype(jnp.float32))
        energy = obs["energy"].astype(jnp.float32)[..., None]
        inventory = jnp.log1p(obs["inventory"].astype(jnp.float32))[..., None]
        team = obs["team"].astype(jnp.float32)[..., None]
        return nn.Sequential([nn.Dense(self.features), nn.relu])(
            jnp.concatenate([view, energy, inventory, team], axis=-1)
        )


def make(cfg):
    env = environments.make(**cfg.environment)

    num_actions = env.action_space().num_actions
    hidden_dim = cfg.cell.features
    num_agents = env.num_agents
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=ViewAndEnergy(features=hidden_dim),
        torso=instantiate(cfg.stack),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions)),
            critic=nn.Dense(1),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg,
        batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps * num_agents,
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
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
