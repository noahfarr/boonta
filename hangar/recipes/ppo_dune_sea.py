import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Gaussian, Network


def mlp(sizes, outputs):
    layers = []
    for size in sizes:
        layers += [nn.Dense(size), nn.swish]
    return nn.Sequential([*layers, nn.Dense(outputs)])


def make(cfg):
    env = environments.make(**cfg.environment)
    action_dim, *_ = env.action_space().shape
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        head=ActorCritic(
            actor=Gaussian(mlp(cfg.network.actor, 2 * action_dim)),
            critic=mlp(cfg.network.critic, 1),
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
