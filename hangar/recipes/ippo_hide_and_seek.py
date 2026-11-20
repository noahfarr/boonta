import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import LogInfo, RecordMultiAgentEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network

INFO = ("hiders_seen", "preparing", "holding", "locked")


class Pick(nn.Module):
    key: str

    def __call__(self, obs):
        return obs[self.key]


def encoder(key: str, features: int, depth: int = 2) -> FeatureExtractor:
    layers = [Pick(key)]
    for _ in range(depth):
        layers += [nn.Dense(features), nn.relu]
    return FeatureExtractor(observation_extractor=nn.Sequential(layers))


def critic() -> Network:
    return Network(feature_extractor=encoder("state", 256), head=nn.Dense(1))


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    env = instantiate(cfg.environment.auto_reset, env)

    actor = Network(feature_extractor=encoder("agent", 128), head=nn.Dense(num_actions))
    algorithm = PPO(
        cfg=instantiate(cfg.algorithm),
        network=ActorCritic(actor=Categorical(actor), critic=critic()),
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(cfg.optimizer.lr),
        ),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = LogInfo(env, keys=INFO, tags=())
    env = RecordMultiAgentEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
