import flax.linen as nn
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_ppo import RecurrentPPO
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network
from boonta.networks.layers import Flatten

from . import schedules

# XLand-MiniGrid observations are (height, width, 2) uint8 grids of categorical
# ids: layer 0 is the tile id (NUM_TILES) and layer 1 is the color id
# (NUM_COLORS). We embed each id rather than treating it as a float, which is
# both cheaper than a one-hot conv and avoids imposing a false ordinal scale on
# the category indices.
NUM_TILES = 13
NUM_COLORS = 12


class GridEncoder(nn.Module):
    hidden_dim: int
    embed_dim: int = 8

    @nn.compact
    def __call__(self, obs):
        tiles = nn.Embed(NUM_TILES, self.embed_dim)(obs[..., 0].astype(jnp.int32))
        colors = nn.Embed(NUM_COLORS, self.embed_dim)(obs[..., 1].astype(jnp.int32))
        grid = jnp.concatenate([tiles, colors], axis=-1)
        grid = nn.relu(nn.Conv(16, (3, 3), padding="VALID")(grid))
        grid = Flatten(start_dim=-3)(grid)
        return nn.relu(nn.Dense(self.hidden_dim)(grid))


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    hidden_dim = cfg.cell.features
    env = instantiate(cfg.environment.auto_reset, env)

    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=GridEncoder(hidden_dim=hidden_dim),
        ),
        torso=instantiate(cfg.stack),
        head=ActorCritic(
            actor=Categorical(nn.Dense(num_actions)),
            critic=nn.Dense(1),
        ),
    )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.podracer.config.num_steps
    )

    algorithm = RecurrentPPO(
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
