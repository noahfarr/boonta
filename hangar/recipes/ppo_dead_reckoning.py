import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import Network

from . import schedules
from .recurrent_ppo_dead_reckoning import features, head


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(feature_extractor=features(num_actions), head=head(num_actions))

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps
    )

    algorithm = PPO(
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
