import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network


def generator():
    from jax2d.engine import PhysicsEngine
    from kinetix.environment import (EnvParams, StaticEnvParams, UEDParams,
                                     sample_kinetix_level)

    env_params, static_env_params, ued_params = (
        EnvParams(),
        StaticEnvParams(),
        UEDParams(),
    )
    physics_engine = PhysicsEngine(static_env_params)

    def sample(key):
        return sample_kinetix_level(
            key, physics_engine, env_params, static_env_params, ued_params
        )

    return sample


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions

    env = SameStepAutoReset(env)
    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    nn.Dense(256),
                    nn.tanh,
                    nn.Dense(256),
                    nn.tanh,
                ]
            ),
        ),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions)),
            critic=nn.Dense(1),
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

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(
        algorithm, env, sample=generator()
    )
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
