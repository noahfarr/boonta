import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.wrappers import RecordEpisodeStatistics, TransformReward
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Network

# The planes the Jumanji adapter builds for PacMan, in order.
FLOOR, PLAYER, GHOSTS, POWER_UPS, PELLETS, SCARED = range(6)

# Jumanji pays 10 per pellet, 50 per power-up and more per ghost eaten; a pellet is worth one after this.
REWARD_SCALE = 0.1


class Planes(nn.Module):
    """A small convolutional encoder over the PacMan planes from the Jumanji adapter."""

    features: int = 256

    @nn.compact
    def __call__(self, obs):
        hidden = obs.astype(jnp.float32)
        for channels, stride in ((32, 1), (64, 2), (64, 2)):
            hidden = nn.relu(nn.Conv(channels, (3, 3), strides=stride, padding="SAME")(hidden))
        hidden = hidden.reshape((*obs.shape[:-3], -1))
        return nn.relu(nn.Dense(self.features)(hidden))


class Flat(nn.Module):
    """An MLP over the flattened PacMan planes: much cheaper than the convolutions, which have to run
    at the full 31 by 28 resolution."""

    features: int = 256
    hidden: int = 512

    @nn.compact
    def __call__(self, obs):
        flat = obs.astype(jnp.float32).reshape((*obs.shape[:-3], -1))
        return nn.relu(nn.Dense(self.features)(nn.relu(nn.Dense(self.hidden)(flat))))


def centred(planes, radius: int):
    """The board shifted so PacMan sits in the middle, cropped to 2 * radius + 1 cells a side.

    Jumanji's PacMan wraps at the edges of the board, so shifting with wrap-around matches the game.
    """
    rows, columns, _ = planes.shape
    where = jnp.argmax(planes[..., PLAYER].reshape(-1))
    row, column = where // columns, where % columns
    shifted = jnp.roll(planes, (rows // 2 - row, columns // 2 - column), axis=(0, 1))
    return shifted[rows // 2 - radius : rows // 2 + radius + 1, columns // 2 - radius : columns // 2 + radius + 1]


class Egocentric(nn.Module):
    """Every piece on every walkable cell, plus a local view centred on PacMan.

    The maze is the same in every episode, so the walls carry nothing: the global part reads
    PacMan, the ghosts, the power-ups and the pellets on the walkable cells alone (`floor`, fixed
    when the recipe is built) and loses nothing the observation holds. The local view shows the
    floor, ghosts, power-ups and pellets within `radius` cells of PacMan, so what is near him reads
    alike wherever he stands. The frightened time left joins both.
    """

    floor: tuple
    features: int = 256
    radius: int = 7
    layer_norm: bool = False

    @nn.compact
    def __call__(self, obs):
        planes = obs.astype(jnp.float32)
        lead = planes.shape[:-3]
        flat = planes.reshape((-1, *planes.shape[-3:]))
        cells = flat.reshape((flat.shape[0], -1, flat.shape[-1]))[:, jnp.asarray(self.floor)]
        board = cells[..., [PLAYER, GHOSTS, POWER_UPS, PELLETS]].reshape((flat.shape[0], -1))
        local = jax.vmap(lambda board: centred(board, self.radius))(flat)[..., [FLOOR, GHOSTS, POWER_UPS, PELLETS]]
        scared = flat[:, 0, 0, SCARED : SCARED + 1]

        def layer(inputs):
            hidden = nn.Dense(self.features)(inputs)
            return nn.relu(nn.LayerNorm()(hidden) if self.layer_norm else hidden)

        board = layer(board)
        local = layer(local.reshape((local.shape[0], -1)))
        hidden = layer(jnp.concatenate([board, local, scared], axis=-1))
        return hidden.reshape((*lead, self.features))


def walkable(env) -> tuple:
    """The cells PacMan can stand on, as indices into the flattened board. The maze never changes."""
    _, timestep = env.init(jax.random.key(0))
    return tuple(int(cell) for cell in np.flatnonzero(np.asarray(timestep.obs)[..., FLOOR]))


def encoder(name: str, features: int, floor: tuple, layer_norm: bool = False) -> nn.Module:
    if name == "egocentric":
        return Egocentric(floor=floor, features=features, layer_norm=layer_norm)
    assert not layer_norm, "layer_norm is only wired into the egocentric encoder"
    if name == "mlp":
        return Flat(features=features)
    if name == "cnn":
        return Planes(features=features)
    raise ValueError(f"Unknown PacMan encoder: {name}")


def make(cfg):
    env = environments.make(**cfg.environment)
    num_actions = env.action_space().num_actions
    floor = walkable(env)
    env = TransformReward(env, lambda reward: REWARD_SCALE * reward)
    env = instantiate(cfg.environment.auto_reset, env)

    net_cfg = cfg.get("network") or {}
    network = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=encoder(
                net_cfg.get("encoder", "egocentric"),
                net_cfg.get("features", 256),
                floor,
                net_cfg.get("layer_norm", False),
            )
        ),
        head=ActorCritic(actor=Categorical(nn.Dense(num_actions)), critic=nn.Dense(1)),
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
