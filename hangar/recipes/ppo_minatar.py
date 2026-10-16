import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.algorithms.wrappers import RND
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network
from boonta.networks.layers import Flatten


def observation_extractor():
    return nn.Sequential(
        [
            lambda obs: obs.astype(jnp.float32),
            nn.Conv(16, (3, 3), padding="VALID"),
            nn.relu,
            Flatten(start_dim=-3),
            nn.Dense(128),
            nn.relu,
        ]
    )


def curiosity_network(head):
    return Network(
        feature_extractor=FeatureExtractor(observation_extractor=observation_extractor()),
        head=head,
    )


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=observation_extractor(),
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

    if cfg.get("intrinsic_reward"):
        rnd = cfg.intrinsic_reward
        algorithm = RND(
            algorithm=algorithm,
            random_network=curiosity_network(nn.Dense(rnd.features)),
            predictor_network=curiosity_network(
                nn.Sequential(
                    [nn.Dense(128), nn.relu, nn.Dense(128), nn.relu, nn.Dense(rnd.features)]
                )
            ),
            critic_network=curiosity_network(nn.Dense(1)),
            optimizer=optax.adam(rnd.lr),
            intrinsic_coefficient=rnd.intrinsic_coefficient,
            intrinsic_gamma=rnd.intrinsic_gamma,
            intrinsic_lambda=rnd.intrinsic_lambda,
            statistics_decay=rnd.statistics_decay,
            update_proportion=rnd.update_proportion,
            update_epochs=rnd.update_epochs,
            num_minibatches=rnd.num_minibatches,
            observation_clip=rnd.observation_clip,
        )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
