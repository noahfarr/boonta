import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.super_mimic import SuperMimic
from boonta.environments.wrappers import RecordEpisodeStatistics, TransformObservation
from boonta.networks import FeatureExtractor, FixedGaussian, Network, Observation, WorldModel


def mlp(sizes, outputs=None):
    layers = [layer for size in sizes for layer in (nn.Dense(size), nn.elu)]
    return nn.Sequential(layers + ([nn.Dense(outputs)] if outputs else []))


def make(cfg):
    env = environments.make(**cfg.environment)
    action_dim, *_ = env.action_space().shape
    obs_dim, *_ = env.observation_space().shape
    env = instantiate(cfg.environment.auto_reset, env)
    env = TransformObservation(
        env, lambda obs: Observation(observation=obs, state=obs, context=obs[..., :0], command=obs[..., :0])
    )

    def clipped(*transforms):
        return optax.chain(optax.clip_by_global_norm(cfg.optimizer.max_grad_norm), *transforms)

    def decayed(steps_per_iteration):
        return optax.cosine_decay_schedule(
            cfg.optimizer.lr,
            decay_steps=cfg.optimizer.decay_iterations * steps_per_iteration,
            alpha=cfg.optimizer.final_scale,
        )

    algorithm_cfg = cfg.algorithm
    critic_steps = algorithm_cfg.policy_updates * algorithm_cfg.critic_epochs * algorithm_cfg.critic_minibatches

    algorithm = SuperMimic(
        cfg=instantiate(cfg.algorithm),
        actor=Network(
            feature_extractor=FeatureExtractor(observation_extractor=mlp(cfg.network.actor)),
            head=FixedGaussian(nn.Dense(action_dim), scale=cfg.network.action_noise),
        ),
        critic=Network(
            feature_extractor=FeatureExtractor(observation_extractor=mlp(cfg.network.critic)),
            head=nn.Dense(1),
        ),
        world_model=WorldModel(
            dynamics_fn=nn.Sequential(
                [
                    lambda state, action, context: jnp.concatenate([state, action, context], axis=-1),
                    mlp(cfg.network.world_model, obs_dim),
                ]
            ),
            reward_fn=nn.Sequential(
                [
                    lambda state, action, goal, **_: jnp.concatenate([state, action, goal], axis=-1),
                    mlp(cfg.network.heads, 1),
                ]
            ),
            termination_fn=nn.Sequential(
                [lambda state, goal, **_: jnp.concatenate([state, goal], axis=-1), mlp(cfg.network.heads, 1)]
            ),
            observation_fn=lambda state, **_: state,
        ),
        buffer=instantiate(cfg.buffer),
        actor_optimizer=clipped(
            optax.clip(cfg.optimizer.max_actor_grad),
            optax.adam(decayed(algorithm_cfg.policy_updates), b1=0.9, b2=0.95),
        ),
        critic_optimizer=clipped(optax.adam(decayed(critic_steps))),
        world_model_optimizer=clipped(optax.adam(decayed(algorithm_cfg.world_model_updates))),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
