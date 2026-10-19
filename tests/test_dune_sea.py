import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments.dune_sea import NCA, DuneSea, measure_liveliness, render_field
from boonta.environments.dune_sea.dune_sea import (BACKWARD, FLORA, FORWARD, NOOP, ROCK, SAND, STEP_LEFT,
                                                   STEP_RIGHT, STEP_X, STEP_Y, TURN_LEFT, TURN_RIGHT,
                                                   VIEW_RADIUS, flood_region)
from boonta.environments.dune_sea.nca import gather_neighbourhood
from boonta.environments.dune_sea.tileset import (load_tileset, neighbour_mask, reachable_masks,
                                                  tileset_available)


def roll_out(nca, seed=0, size=24, steps=32):
    rule_key, field_key, step_key = jax.random.split(jax.random.key(seed), 3)
    rule = nca.draw_rule(rule_key)
    field = nca.seed_field(field_key, size, size)
    return jax.jit(lambda: nca.run_rollout(rule, field, step_key, steps))()


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_a_rollout_holds_only_valid_symbols(temperature):
    trajectory = roll_out(NCA(num_symbols=5, temperature=temperature))
    assert trajectory.shape == (32, 24, 24)
    assert int(trajectory.min()) >= 0 and int(trajectory.max()) < 5


def test_one_key_gives_one_rollout():
    nca = NCA(temperature=1.0)
    np.testing.assert_array_equal(roll_out(nca, seed=3), roll_out(nca, seed=3))
    assert not np.array_equal(roll_out(nca, seed=3), roll_out(nca, seed=4))


def test_at_zero_temperature_the_rule_is_deterministic_given_the_field():
    nca = NCA(temperature=0.0)
    rule = nca.draw_rule(jax.random.key(0))
    field = nca.seed_field(jax.random.key(1), 16, 16)
    first = nca.evolve_field(rule, field, jax.random.key(2))
    second = nca.evolve_field(rule, field, jax.random.key(3))
    np.testing.assert_array_equal(first, second)


def test_each_cell_sees_its_three_by_three_neighbourhood():
    field = jnp.zeros((5, 5), jnp.int32).at[2, 3].set(1)
    senses = gather_neighbourhood(field, 2)
    assert senses.shape == (5, 5, 18)
    east_slot = (1 * 3 + 2) * 2 + 1
    assert float(senses[2, 2, east_slot]) == 1.0
    assert float(senses[2, 2].reshape(9, 2)[:, 1].sum()) == 1.0


def test_liveliness_tells_frozen_noisy_and_structured_apart():
    frozen = jnp.zeros((8, 32, 32), jnp.int32)
    noise = jax.random.randint(jax.random.key(0), (8, 32, 32), 0, 4)
    stripes = jnp.broadcast_to((jnp.arange(32) // 4) % 2, (8, 32, 32))
    stripes = stripes.at[::2].set(1 - stripes[::2])
    assert float(measure_liveliness(frozen, 4)["activity"]) == 0.0
    assert float(measure_liveliness(frozen, 4)["diversity"]) < 0.01
    assert abs(float(measure_liveliness(noise, 4)["structure"])) < 0.1
    assert float(measure_liveliness(stripes, 4)["structure"]) > 0.5


def test_render_gives_each_symbol_its_own_colour():
    field = jnp.arange(6).reshape(2, 3)
    frame = render_field(field, 6, scale=2)
    assert frame.shape == (4, 6, 3) and frame.dtype == jnp.uint8
    colours = {tuple(frame[2 * (s // 3), 2 * (s % 3)].tolist()) for s in range(6)}
    assert len(colours) == 6


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="num_symbols"):
        NCA(num_symbols=1)
    with pytest.raises(ValueError, match="temperature"):
        NCA(temperature=-1.0)


EAST, WEST = 1, 3


def small_world(**kwargs):
    return DuneSea(**{"width": 10, "height": 8, "num_agents": 3, **kwargs})


def seat_agents(state, positions, directions):
    return state.replace(
        cells=jnp.zeros_like(state.cells),
        position=jnp.asarray(positions, jnp.int32),
        direction=jnp.asarray(directions, jnp.int32),
    )


def play_actions(env, state, actions):
    return env.step(jax.random.key(1), state, jnp.asarray(actions))


def test_the_world_starts_with_agents_on_distinct_cells():
    env = small_world(num_agents=12)
    state, timestep = env.init(jax.random.key(0))
    cells = state.position[:, 1] * env.width + state.position[:, 0]
    assert len(set(cells.tolist())) == 12
    assert timestep.obs.shape == env.observation_space().shape
    assert timestep.action.shape == env.action_space().shape
    assert bool(timestep.terminated.all())


@pytest.mark.parametrize(
    "action, dx, dy, facing",
    [
        (FORWARD, 1, 0, EAST),
        (BACKWARD, -1, 0, EAST),
        (STEP_LEFT, 0, -1, EAST),
        (STEP_RIGHT, 0, 1, EAST),
        (TURN_LEFT, 0, 0, 0),
        (TURN_RIGHT, 0, 0, 2),
        (NOOP, 0, 0, EAST),
    ],
)
def test_moves_step_relative_to_the_facing_and_turns_rotate(action, dx, dy, facing):
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    state = seat_agents(state, [[5, 5], [0, 0], [9, 7]], [EAST, 0, 0])
    state, _ = play_actions(env, state, [action, NOOP, NOOP])
    assert state.position[0].tolist() == [5 + dx, 5 + dy]
    assert int(state.direction[0]) == facing


def test_the_world_wraps_around_its_edges():
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    state = seat_agents(state, [[9, 0], [4, 4], [6, 6]], [EAST, 0, 0])
    state, _ = play_actions(env, state, [FORWARD, NOOP, NOOP])
    assert state.position[0].tolist() == [0, 0]


def test_agents_aiming_at_one_cell_or_swapping_stay_put():
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    clash = seat_agents(state, [[4, 5], [6, 5], [0, 0]], [EAST, WEST, 0])
    after, _ = play_actions(env, clash, [FORWARD, FORWARD, NOOP])
    assert after.position[:2].tolist() == [[4, 5], [6, 5]]
    swap = seat_agents(state, [[4, 5], [5, 5], [0, 0]], [EAST, WEST, 0])
    after, _ = play_actions(env, swap, [FORWARD, FORWARD, NOOP])
    assert after.position[:2].tolist() == [[4, 5], [5, 5]]
    line = seat_agents(state, [[4, 5], [5, 5], [6, 5]], [EAST, EAST, EAST])
    after, _ = play_actions(env, line, [FORWARD, FORWARD, FORWARD])
    assert after.position.tolist() == [[5, 5], [6, 5], [7, 5]]
    held = seat_agents(state, [[4, 5], [5, 5], [0, 0]], [EAST, EAST, 0])
    after, _ = play_actions(env, held, [FORWARD, NOOP, NOOP])
    assert after.position[0].tolist() == [4, 5]


@pytest.mark.parametrize("facing", range(4))
def test_each_agent_sees_others_turned_so_it_faces_up(facing):
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    ahead = [5 + STEP_X[facing], 4 + STEP_Y[facing]]
    left = (facing + 3) % 4
    beside = [5 + STEP_X[left], 4 + STEP_Y[left]]
    state = seat_agents(state, [[5, 4], ahead, beside], [facing, 0, 0])
    view = env.observe(state)[0, ..., 0]
    r = VIEW_RADIUS
    assert float(view[r, r]) == 1.0
    assert float(view[r - 1, r]) == 1.0
    assert float(view[r, r - 1]) == 1.0
    assert float(view.sum()) == 3.0


def test_episodes_truncate_at_the_limit():
    env = small_world(time_limit=3)
    state, _ = env.init(jax.random.key(0))
    for index in range(3):
        state, timestep = play_actions(env, state, [NOOP, NOOP, NOOP])
        assert bool(timestep.truncated.all()) == (index == 2)


def test_render_draws_every_cell():
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    frame = env.render(state, scale=4)
    assert frame.shape == (8 * 4, 10 * 4, 3) and frame.dtype == jnp.uint8


@pytest.mark.parametrize("seed", range(6))
def test_agents_spawn_on_sand_that_is_all_one_region(seed):
    env = small_world(width=24, height=20, num_agents=8)
    state, _ = env.init(jax.random.key(seed))
    x, y = state.position[:, 0], state.position[:, 1]
    assert bool((state.cells[y, x] == SAND).all())
    sand = state.cells == SAND
    reached = flood_region(sand, y[0] * env.width + x[0])
    np.testing.assert_array_equal(reached, sand)
    assert 0.0 < float((state.cells == ROCK).mean()) < 0.8


def test_the_fill_sets_how_rocky_the_map_is():
    open_map, _ = small_world(rock_fill=0.0, flora_fill=0.0).init(jax.random.key(0))
    assert int(open_map.cells.sum()) == 0
    solid_map, _ = small_world(rock_fill=1.0, flora_fill=0.0).init(jax.random.key(0))
    assert int(solid_map.cells.sum()) == 0


def test_rock_blocks_the_way():
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    state = seat_agents(state, [[5, 5], [0, 0], [9, 7]], [EAST, 0, 0])
    state = state.replace(cells=state.cells.at[5, 6].set(ROCK))
    after, _ = play_actions(env, state, [FORWARD, NOOP, NOOP])
    assert after.position[0].tolist() == [5, 5]
    after, _ = play_actions(env, state, [STEP_LEFT, NOOP, NOOP])
    assert after.position[0].tolist() == [5, 4]


@pytest.mark.parametrize("facing", range(4))
def test_each_agent_sees_rock_turned_so_it_faces_up(facing):
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    state = seat_agents(state, [[5, 4], [0, 0], [9, 7]], [facing, 0, 0])
    state = state.replace(cells=state.cells.at[4 + STEP_Y[facing], 5 + STEP_X[facing]].set(ROCK))
    view = env.observe(state)[0, ..., 1]
    assert float(view[VIEW_RADIUS - 1, VIEW_RADIUS]) == 1.0
    assert float(view.sum()) == 1.0


def test_without_the_tileset_render_falls_back_to_flat_colours(monkeypatch):
    from boonta.environments.dune_sea import dune_sea as world

    monkeypatch.setattr(world, "tileset_available", lambda: False)
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    frame = env.render(state, scale=4)
    assert frame.shape == (8 * 4, 10 * 4, 3) and frame.dtype == jnp.uint8


@pytest.mark.skipif(not tileset_available(), reason="needs the Glionox desert tileset in dune_sea/assets")
def test_the_tileset_has_a_tile_for_every_rock_neighbourhood():
    for name in ("overhang", "patch_edges"):
        filled = np.asarray(load_tileset()[name])[..., 3].reshape(256, -1).max(axis=1) > 0
        assert all(filled[mask] for mask in reachable_masks() if mask)
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    assert env.render(state, scale=16).shape == (8 * 16, 10 * 16, 3)


def test_the_autotile_mask_marks_the_edges_and_corners_rock_overhangs():
    north, north_east, east, south_east, south, south_west, west, north_west = (1 << bit for bit in range(8))
    is_rock = jnp.zeros((5, 5), bool).at[1, 2].set(True).at[2, 3].set(True)
    assert int(neighbour_mask(is_rock)[2, 2]) == north | north_east | east | south_east | north_west
    lone_corner = jnp.zeros((5, 5), bool).at[3, 1].set(True)
    assert int(neighbour_mask(lone_corner)[2, 2]) == south_west


def test_flora_blocks_the_way_and_shows_in_its_own_channel():
    env = small_world()
    state, _ = env.init(jax.random.key(0))
    state = seat_agents(state, [[5, 5], [0, 0], [9, 7]], [EAST, 0, 0])
    state = state.replace(cells=state.cells.at[5, 6].set(FLORA))
    after, _ = play_actions(env, state, [FORWARD, NOOP, NOOP])
    assert after.position[0].tolist() == [5, 5]
    view = env.observe(state)[0]
    assert float(view[VIEW_RADIUS - 1, VIEW_RADIUS, 2]) == 1.0
    assert float(view[..., 1].sum()) == 0.0


def test_flora_grows_on_sand_only_and_can_be_switched_off():
    lush, _ = small_world(width=32, height=24, flora_fill=0.2).init(jax.random.key(1))
    assert int((lush.cells == FLORA).sum()) > 0
    bare, _ = small_world(width=32, height=24, flora_fill=0.0).init(jax.random.key(1))
    assert int((bare.cells == FLORA).sum()) == 0
