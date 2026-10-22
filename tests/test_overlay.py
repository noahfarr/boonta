from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from boonta.artisans import Image, Overlay
from boonta.artisans.overlay import NEGATIVE, POSITIVE, paint
from boonta.curricula import prd
from boonta.curricula.prd import (
    Advantage,
    CountedRelevance,
    Diagonal,
    SuccessorFootprint,
    SuccessorRepresentation,
    UniformFootprint,
    UniformGain,
    UniformRelevance,
)
from boonta.environments.dune_sea import ChainOfForks, Hotspots
from boonta.environments.dune_sea.chain_of_forks import cell as fork_cell
from boonta.environments.dune_sea.hotspots import cell as hotspot_cell
from boonta.environments.wrappers.archive_auto_reset import ArchiveAutoReset
from boonta.utils import Transition

NUM_ENVS = 8


@struct.dataclass
class FakeState:
    pit_state: object
    environment_state: object
    algorithm_state: object = None


def make_selector(**overrides):
    base = dict(
        successor=SuccessorRepresentation(gamma=0.9),
        footprint=SuccessorFootprint(),
        relevance=CountedRelevance(),
        gain=Advantage(),
        score=Diagonal(),
        allocation=prd.Marginal(),
        k=2.0,
    )
    base.update(overrides)
    return prd.Selector(**base)


def stack(timesteps):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *timesteps)


def run(game, cell_fn, selector, steps=6, updates=2):
    env = ArchiveAutoReset(game, num_envs=NUM_ENVS, cell_fn=cell_fn, archive_size=64)
    key = jax.random.key(0)
    env_state, timestep = env.init(key)
    ledger = selector.init(env.archive.archive_size, env.num_actions)
    low, high = env.action_space().low, env.action_space().high
    for _ in range(updates):
        firsts, seconds = [], []
        for _ in range(steps):
            key, act, move = jax.random.split(key, 3)
            actions = jax.random.randint(act, (NUM_ENVS,), int(low), int(high) + 1)
            env_state, after = env.step(move, env_state, actions)
            firsts.append(timestep)
            seconds.append(after)
            timestep = after
        blank = jnp.zeros((steps, NUM_ENVS))
        transitions = Transition(
            first=stack(firsts),
            second=stack(seconds),
            aux={
                "value": blank,
                "log_prob": blank,
                "logits": jnp.zeros((steps, NUM_ENVS, env.num_actions)),
            },
        )
        key, sub = jax.random.split(key)
        ledger = selector.update(ledger, sub, transitions)
    return env, env_state, ledger


def fixture(game, cell_fn, selector):
    env, env_state, ledger = run(game, cell_fn, selector)
    state = FakeState(pit_state=ledger, environment_state=env_state)
    pit = SimpleNamespace(environment=env, selector=selector)
    return state, SimpleNamespace(algorithm=None, environment=env, pit=pit)


GAMES = [
    pytest.param(ChainOfForks(depth=4), fork_cell, id="chain_of_forks"),
    pytest.param(
        Hotspots(width=9, num_hotspots=4, nearest=2, farthest=7), hotspot_cell, id="hotspots"
    ),
]
TILE = 8


def heat(game, image):
    _, num_columns = np.asarray(game.cell_grid).shape
    return image[:, num_columns * TILE + TILE // 2 :]


@pytest.mark.parametrize("game, cell_fn", GAMES)
def test_overlay_lays_the_value_beside_the_render(game, cell_fn):
    state, podracer = fixture(game, cell_fn, make_selector())
    artifact = Overlay(tile=TILE).craft(jax.random.key(1), podracer, state, {})

    assert isinstance(artifact, Image)
    image = np.asarray(artifact.data)
    num_rows, num_columns = np.asarray(game.cell_grid).shape
    assert image.dtype == np.uint8
    assert image.shape == (num_rows * TILE, 2 * num_columns * TILE + TILE // 2, 3)
    render = np.asarray(game.render(game.init(jax.random.key(1))[0], tile=TILE))
    np.testing.assert_array_equal(image[:, : num_columns * TILE], render)


@pytest.mark.parametrize("game, cell_fn", GAMES)
def test_overlay_lights_the_cells_the_archive_holds(game, cell_fn):
    state, podracer = fixture(game, cell_fn, make_selector())
    image = np.asarray(Overlay(tile=TILE).craft(jax.random.key(2), podracer, state, {}).data)

    centres = heat(game, image)[TILE // 2 :: TILE, TILE // 2 :: TILE]
    assert centres.sum() > 0


def test_overlay_follows_the_runs_restart_rule():
    game = ChainOfForks(depth=4)
    selector = make_selector(
        footprint=UniformFootprint(), relevance=UniformRelevance(), gain=UniformGain()
    )
    state, podracer = fixture(game, fork_cell, selector)
    image = np.asarray(Overlay(tile=TILE).craft(jax.random.key(3), podracer, state, {}).data)

    centres = heat(game, image)[TILE // 2 :: TILE, TILE // 2 :: TILE]
    lit = centres[centres.sum(axis=-1) > 0]
    assert len(lit) > 0
    assert len(np.unique(lit.reshape(-1, 3), axis=0)) == 1


def test_paint_shades_negative_values_in_their_own_colour():
    canvas = paint(np.array([1.0, -1.0, 0.5]), np.array([[0, 1, 2, -1]]), TILE)
    centres = canvas[TILE // 2, TILE // 2 :: TILE]

    np.testing.assert_array_equal(centres[0], POSITIVE.astype(np.uint8))
    np.testing.assert_array_equal(centres[1], NEGATIVE.astype(np.uint8))
    np.testing.assert_array_equal(centres[2], (0.5 * POSITIVE).astype(np.uint8))
    np.testing.assert_array_equal(centres[3], [0, 0, 0])
