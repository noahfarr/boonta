import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.super_mimic import SuperMimic
from boonta.environments.mjlab import motion_tracking
from boonta.environments.mjlab.task import NUM_JOINTS
from boonta.environments.world_model import Observation, WorldModel
from boonta.environments.wrappers import RecordEpisodeStatistics, TransformObservation
from boonta.networks import FeatureExtractor, FixedGaussian, FloatingBase, GoalConditioned, Network
from boonta.networks.layers import Normalize


def mlp(sizes, outputs=None):
    layers = [layer for size in sizes for layer in (nn.Dense(size), nn.elu)]
    return nn.Sequential(layers + ([nn.Dense(outputs)] if outputs else []))


def normalized(module):
    return nn.Sequential([Normalize(), module])


def make(cfg):
    env = environments.make(**cfg.environment)
    task = motion_tracking.describe(env)
    goal_size, *_ = jnp.shape(task.goal(jnp.int32(0)))
    anchored = cfg.network.anchored

    def adapt(obs):
        observation = obs["observation"]
        if anchored:
            observation = jnp.concatenate([observation, obs["anchor"]], axis=-1)
        return Observation(
            observation=observation,
            state=obs["state"],
            context=obs["context"],
            command=obs["frame"][..., 0].astype(jnp.int32),
        )

    env = TransformObservation(env, adapt)

    def reward(state, following, command, **_):
        return jax.vmap(task.reward)(state, following, command)

    def observe(state, context, command, **_):
        observation = jax.vmap(task.observe)(state, context)
        if anchored:
            return jnp.concatenate([observation, jax.vmap(task.anchor)(state, command)], axis=-1)
        return observation

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
            feature_extractor=FeatureExtractor(
                observation_extractor=normalized(
                    GoalConditioned(
                        encoder=mlp(cfg.network.encoder, cfg.network.latent),
                        decoder=mlp(cfg.network.decoder),
                        goal_size=goal_size,
                    )
                )
            ),
            head=FixedGaussian(nn.Dense(NUM_JOINTS), scale=0.0),
        ),
        critic=Network(
            feature_extractor=FeatureExtractor(observation_extractor=normalized(mlp(cfg.network.critic))),
            head=nn.Dense(1),
        ),
        world_model=WorldModel(
            env,
            dynamics_fn=FloatingBase(
                normalized(mlp(cfg.network.world_model, 12 + 2 * NUM_JOINTS)), num_joints=NUM_JOINTS
            ),
            reward_fn=reward,
            termination_fn=nn.Sequential(
                [
                    lambda state, goal, **_: jnp.concatenate([state, goal], axis=-1),
                    normalized(mlp(cfg.network.heads, 1)),
                ]
            ),
            observation_fn=observe,
            goal=jax.vmap(task.goal),
            advance=jax.vmap(task.advance),
        ),
        buffer=instantiate(cfg.buffer),
        actor_optimizer=clipped(
            optax.clip(cfg.optimizer.max_actor_grad),
            optax.adam(decayed(algorithm_cfg.policy_updates), b1=0.9, b2=0.95),
        ),
        critic_optimizer=clipped(optax.adam(decayed(critic_steps))),
        world_model_optimizer=clipped(optax.adam(decayed(algorithm_cfg.world_model_updates))),
        sensor_noise=task.perturb,
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
