import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.super_mimic import SuperMimic, WorldModel
from boonta.environments.wrappers import NormalizeObservation, RecordEpisodeStatistics
from boonta.networks import FeatureExtractor, FixedGaussian, Network


def mlp(sizes, outputs=None):
    layers = [layer for size in sizes for layer in (nn.Dense(size), nn.elu)]
    return nn.Sequential(layers + ([nn.Dense(outputs)] if outputs else []))


def make(cfg):
    env = environments.make(**cfg.environment)
    action_dim, *_ = env.action_space().shape
    obs_dim, *_ = env.observation_space().shape
    env = instantiate(cfg.environment.auto_reset, env)

    def clipped(*transforms):
        return optax.chain(optax.clip_by_global_norm(cfg.optimizer.max_grad_norm), *transforms)

    algorithm = SuperMimic(
        cfg=instantiate(cfg.algorithm),
        actor=Network(
            feature_extractor=FeatureExtractor(observation_extractor=mlp(cfg.network.actor)),
            head=FixedGaussian(nn.Sequential([nn.Dense(action_dim), nn.tanh]), scale=cfg.network.action_noise),
        ),
        critic=Network(
            feature_extractor=FeatureExtractor(observation_extractor=mlp(cfg.network.critic)),
            head=nn.Dense(1),
        ),
        world_model=WorldModel(
            dynamics=mlp(cfg.network.world_model, obs_dim),
            reward=mlp(cfg.network.heads, 1),
            observation=mlp(cfg.network.heads, obs_dim),
            termination=mlp(cfg.network.heads, 1),
        ),
        buffer=instantiate(cfg.buffer),
        actor_optimizer=clipped(
            optax.clip(cfg.optimizer.max_actor_grad), optax.adam(cfg.optimizer.lr, b1=0.9, b2=0.95)
        ),
        critic_optimizer=clipped(optax.adam(cfg.optimizer.lr)),
        world_model_optimizer=clipped(optax.adam(cfg.optimizer.lr)),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    env = NormalizeObservation(env)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
