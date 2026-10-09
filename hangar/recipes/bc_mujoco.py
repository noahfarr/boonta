import flax.linen as nn
import optax
from hydra.utils import instantiate

from boonta import datasets
from boonta import environments
from boonta.algorithms.bc import BC
from boonta.environments.wrappers import RecordEpisodeStatistics, SameStepAutoReset, Vectorize
from boonta.networks import FeatureExtractor, Gaussian, Network


def make(cfg):
    env = environments.make(**cfg.environment)
    env = SameStepAutoReset(env)

    action_dim, *_ = env.action_space().shape

    env = Vectorize(env, num_envs=cfg.environment.num_envs)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential(
                [
                    nn.Dense(256),
                    nn.relu,
                    nn.Dense(256),
                    nn.relu,
                ]
            ),
        ),
        head=Gaussian(nn.Dense(2 * action_dim)),
    )

    algorithm = BC(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.adam(cfg.optimizer.lr),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env)
    return {
        "algorithm": algorithm,
        "environment": env,
        "dataset": datasets.make(**cfg.dataset),
        "pit": pit,
        "lap": lap,
    }
