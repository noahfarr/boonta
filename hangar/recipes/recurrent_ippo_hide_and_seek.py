import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.wrappers import LogInfo, RecordMultiAgentEpisodeStatistics
from boonta.networks import Categorical, Network

from . import schedules
from .ippo_hide_and_seek import INFO, critic, encoder


class OmniscientCritic(nn.Module):
    actor: nn.Module
    critic: nn.Module

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, carry=None, temperature=None, **kwargs):
        carry, distribution = self.actor(
            obs, action, reward, done, carry=carry, temperature=temperature, **kwargs
        )
        return carry, (distribution, self.critic(obs).astype(jnp.float32))

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.actor.initialize_carry(key, input_shape)


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    num_agents = env.num_agents
    env = instantiate(cfg.environment.auto_reset, env)

    actor = Network(
        feature_extractor=encoder("agent", cfg.cell.features, depth=1),
        torso=instantiate(cfg.stack),
        head=Categorical(nn.Dense(num_actions)),
    )
    learning_rate = schedules.learning_rate(
        cfg,
        batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps * num_agents,
    )
    algorithm = RecurrentPPO(
        cfg=instantiate(cfg.algorithm),
        network=OmniscientCritic(actor=actor, critic=critic()),
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(learning_rate),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = LogInfo(env, keys=INFO, tags=())
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
