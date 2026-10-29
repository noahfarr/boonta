import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments import dune_sea
from boonta.environments.dune_sea import Agent, Jundland
from boonta.environments.dune_sea import Cell as StateCell
from boonta.environments.dune_sea.jundland import (ACTION_COST, BACKWARD, BROADCAST,
                                                   FORWARD, INTERACT, LEFT, MINE_COST,
                                                   NOOP, PLACE, RIGHT, TURN_LEFT,
                                                   TURN_RIGHT)
from boonta.environments.dune_sea.render import render
from boonta.environments.dune_sea.state import (MATURE_ENERGY, MAX_ENERGY, MAX_PLANT_ENERGY,
                                                SEED_ENERGY, STARTING_ENERGY)

SIZE = 3
EMPTY = jnp.zeros((SIZE, SIZE))
KEY = jax.random.key(0)
GROWTH = (MAX_PLANT_ENERGY - SEED_ENERGY) / 100
CALM = jnp.int32(0)
STORM = jnp.int32(120)


def make_cell(wall, energy, bush=None):
    return StateCell(wall, energy, jnp.zeros_like(wall) if bush is None else bush, jnp.zeros_like(wall))


def agents(positions, directions, energy=STARTING_ENERGY, inventory=0):
    count = len(directions)
    return Agent(
        position=jnp.array(positions),
        direction=jnp.array(directions),
        energy=jnp.broadcast_to(jnp.asarray(energy, jnp.float32), (count,)),
        broadcast=jnp.zeros(count, bool),
        inventory=jnp.broadcast_to(jnp.asarray(inventory, jnp.int32), (count,)),
    )


def world(cells, agent, **kwargs):
    count, *_ = agent.position.shape
    height, width = cells.wall.shape
    env = Jundland(
        height=height,
        width=width,
        num_agents=count,
        generator=lambda key, height, width, num_agents, num_plants: (cells, agent),
        **kwargs,
    )
    state, _ = env.init(KEY)
    return env, state.replace(agent=agent)


def step(env, state, *actions, key=KEY):
    return jax.jit(env.step)(key, state, jnp.array(actions, jnp.int32))


def plant(row, column, amount, size=SIZE):
    return jnp.zeros((size, size)).at[row, column].set(amount)


def test_interact_takes_the_energy_of_the_cell_being_faced():
    env, state = world(make_cell(EMPTY, plant(1, 2, 40.0)), agents([[1, 1]], [0]))

    state, timestep = step(env, state, INTERACT)

    assert state.cells.energy.sum() == 0
    assert state.agent.energy[0] == pytest.approx(STARTING_ENERGY + 40.0 - ACTION_COST[INTERACT])


@pytest.mark.parametrize("direction", [1, 2, 3])
def test_interact_does_nothing_unless_the_agent_faces_the_energy(direction):
    env, state = world(make_cell(EMPTY, plant(1, 2, 40.0)), agents([[1, 1]], [direction]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.energy[1, 2] == pytest.approx(40.0 + GROWTH)


def test_standing_on_energy_does_not_collect_it():
    env, state = world(make_cell(EMPTY, plant(1, 1, 40.0)), agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.energy[1, 1] == pytest.approx(40.0 + GROWTH)


def test_interact_mines_a_wall_into_the_inventory_before_collecting_and_mining_costs_more():
    env, state = world(make_cell(plant(1, 2, 1.0), plant(1, 2, 40.0)), agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.wall.sum() == 0
    assert state.agent.inventory[0] == 1
    assert state.cells.energy[1, 2] == pytest.approx(40.0 + GROWTH)
    assert state.agent.energy[0] == pytest.approx(STARTING_ENERGY - MINE_COST)


def test_two_agents_facing_one_plant_cannot_both_take_it():
    env, state = world(make_cell(EMPTY, plant(1, 1, 40.0)), agents([[1, 0], [1, 2]], [0, 2]))

    state, _ = step(env, state, INTERACT, INTERACT)

    gains = state.agent.energy - STARTING_ENERGY + ACTION_COST[INTERACT]
    assert sorted(float(gain) for gain in gains) == pytest.approx([0.0, 40.0], abs=1e-4)


def test_turning_changes_what_the_agent_can_interact_with():
    env, state = world(make_cell(EMPTY, plant(0, 1, 40.0)), agents([[1, 1]], [0]))

    state, _ = step(env, state, TURN_LEFT)
    state, _ = step(env, state, INTERACT)

    assert state.cells.energy.sum() == 0


@pytest.mark.parametrize(
    "action, cost",
    [
        (NOOP, 1.0),
        (FORWARD, 1.5),
        (BACKWARD, 1.5),
        (LEFT, 1.5),
        (RIGHT, 1.5),
        (TURN_LEFT, 1.0),
        (TURN_RIGHT, 1.0),
        (INTERACT, 1.0),
        (BROADCAST, 1.5),
        (PLACE, 2.0),
    ],
)
def test_every_action_has_its_own_cost(action, cost):
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0]))

    state, _ = step(env, state, action)

    assert state.agent.energy[0] == pytest.approx(STARTING_ENERGY - cost)


def test_a_move_into_a_wall_costs_as_much_as_a_move():
    env, state = world(make_cell(plant(1, 2, 1.0), EMPTY), agents([[1, 1]], [0]))

    state, _ = step(env, state, FORWARD)

    assert state.agent.position[0].tolist() == [1, 1]
    assert state.agent.energy[0] == pytest.approx(STARTING_ENERGY - ACTION_COST[FORWARD])


def test_noop_and_broadcast_leave_the_agent_where_it_is():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0]))

    for action in (NOOP, BROADCAST):
        after, _ = step(env, state, action)
        assert (after.agent.position == state.agent.position).all()


def test_the_reward_is_one_for_every_step_survived_and_nothing_for_the_step_that_kills():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 0], [1, 2]], [0, 0], energy=[100.0, 1.5]))

    state, timestep = step(env, state, NOOP, NOOP)
    assert timestep.reward.tolist() == [1.0, 1.0]
    assert timestep.terminated.tolist() == [False, False]

    state, timestep = step(env, state, NOOP, NOOP)
    assert timestep.reward.tolist() == [1.0, 0.0]
    assert timestep.terminated.tolist() == [False, True]

    state, timestep = step(env, state, NOOP, NOOP)
    assert timestep.reward.tolist() == [1.0, 1.0]
    assert timestep.terminated.tolist() == [False, False]


def test_energy_never_rises_above_the_maximum():
    env, state = world(make_cell(EMPTY, plant(1, 2, 100.0)), agents([[1, 1]], [0], energy=MAX_ENERGY - 10))

    state, _ = step(env, state, INTERACT)

    assert state.agent.energy[0] == MAX_ENERGY


def test_plants_grow_linearly_up_to_a_cap():
    energy = plant(0, 0, 20.0).at[2, 2].set(MAX_PLANT_ENERGY - 0.1)
    env, state = world(make_cell(EMPTY, energy), agents([[1, 1]], [0]))

    state, _ = step(env, state, NOOP)
    assert state.cells.energy[0, 0] == pytest.approx(20.0 + GROWTH)
    state, _ = step(env, state, NOOP)
    assert state.cells.energy[0, 0] == pytest.approx(20.0 + 2 * GROWTH)
    assert state.cells.energy[2, 2] == pytest.approx(MAX_PLANT_ENERGY)
    assert state.cells.energy[0, 2] == 0


FIELD = 9


def harvests(wall, energy, samples=400):
    env, state = world(make_cell(wall, energy), agents([[4, 3]], [0]))
    keys = jax.random.split(KEY, samples)
    after, _ = jax.jit(jax.vmap(lambda key: env.step(key, state, jnp.array([INTERACT]))))(keys)
    return np.asarray(after.cells.energy)


def test_a_mature_harvest_scatters_zero_to_three_seeds_in_the_five_by_five_field_around_it():
    energy = harvests(jnp.zeros((FIELD, FIELD)), plant(4, 4, 100.0, FIELD))
    seeds = energy == SEED_ENERGY
    counts = seeds.sum((1, 2))

    assert set(counts.tolist()) <= {0, 1, 2, 3} and counts.max() == 3
    assert 0.18 < (counts == 0).mean() < 0.32
    outside = seeds.copy()
    outside[:, 2:7, 2:7] = False
    assert not outside.any()
    assert (energy[seeds.any((1, 2))] >= 0).all()


def test_seeds_never_land_on_an_agent_but_can_land_where_the_plant_was():
    seeds = harvests(jnp.zeros((FIELD, FIELD)), plant(4, 4, 100.0, FIELD)) == SEED_ENERGY

    assert not seeds[:, 4, 3].any()
    assert seeds[:, 4, 4].any()


def test_seeds_never_land_on_walls_or_on_plants():
    wall = jnp.zeros((FIELD, FIELD)).at[2:7, 2:7].set(1.0).at[4, 3].set(0.0).at[4, 4].set(0.0).at[3, 4].set(0.0)
    energy = harvests(wall, plant(4, 4, 100.0, FIELD).at[3, 4].set(50.0))
    seeds = energy == SEED_ENERGY

    assert seeds[:, 4, 4].any()
    seeds[:, 4, 4] = False
    assert not seeds.any()
    assert energy[:, 3, 4] == pytest.approx(50.0 + GROWTH)


def test_an_immature_harvest_kills_the_plant_without_seeds():
    energy = harvests(jnp.zeros((FIELD, FIELD)), plant(4, 4, MATURE_ENERGY - 1.0, FIELD))

    assert (energy == 0).all()


def test_a_harvest_at_the_edge_scatters_seeds_only_onto_the_map():
    env, state = world(make_cell(jnp.zeros((FIELD, FIELD)), plant(0, 1, 100.0, FIELD)), agents([[0, 0]], [0]))
    keys = jax.random.split(KEY, 200)
    after, _ = jax.jit(jax.vmap(lambda key: env.step(key, state, jnp.array([INTERACT]))))(keys)
    seeds = np.asarray(after.cells.energy) == SEED_ENERGY

    assert seeds.any()
    assert not seeds[:, 4:, :].any() and not seeds[:, :, 4:].any()


def test_the_carrying_capacity_sets_how_many_plants_the_world_starts_with():
    env = Jundland(num_agents=8, carrying_capacity=2.0)
    state, timestep = env.init(KEY)

    assert env.num_plants == round(2.0 * 8 * 1.5 / GROWTH)
    assert int((state.cells.energy > 0).sum()) == env.num_plants
    assert float(timestep.info["plants"]) == pytest.approx(env.num_plants / 8)
    assert Jundland(num_agents=8, carrying_capacity=1.0).num_plants < env.num_plants


def test_plants_start_on_open_ground_away_from_agents_and_below_the_cap():
    env = Jundland()
    state, _ = env.init(KEY)
    energy = np.asarray(state.cells.energy)
    rows, columns = np.asarray(state.agent.position).T

    assert not (energy * np.asarray(state.cells.wall)).any()
    assert not energy[rows, columns].any()
    assert energy.max() <= MAX_PLANT_ENERGY


def test_place_puts_a_wall_on_the_faced_cell_and_spends_dirt():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0], inventory=1))

    state, _ = step(env, state, PLACE)

    assert state.cells.wall[1, 2] == 1
    assert state.agent.inventory[0] == 0
    state, _ = step(env, state, PLACE)
    assert state.cells.wall.sum() == 1


def test_place_needs_dirt_and_a_free_cell():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0], inventory=0))
    state, _ = step(env, state, PLACE)
    assert state.cells.wall.sum() == 0

    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1], [1, 2]], [0, 2], inventory=1))
    state, _ = step(env, state, PLACE, NOOP)
    assert state.cells.wall.sum() == 0
    assert state.agent.inventory.tolist() == [1, 1]


def test_placed_wall_can_be_mined_back():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0], inventory=1))

    state, _ = step(env, state, PLACE)
    state, _ = step(env, state, INTERACT)

    assert state.cells.wall.sum() == 0
    assert state.agent.inventory[0] == 1


def test_agents_start_with_the_dirt_and_energy_they_are_given():
    env = Jundland(initial_dirt=4, starting_energy=60.0)
    state, timestep = env.init(KEY)

    assert (state.agent.inventory == 4).all()
    assert (state.agent.energy == 60.0).all()
    assert float(timestep.obs["energy"][0]) == pytest.approx(60.0 / MAX_ENERGY, rel=1e-2)


def test_an_agent_without_energy_is_inert_and_neither_blocks_nor_shows():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 0], [1, 1]], [0, 0], energy=[125.0, 0.0]))

    view = env.observe(state)["view"][0]
    assert view[2, 3, 2] == 0 and view[2, 3, 3] == 0

    state, timestep = step(env, state, FORWARD, FORWARD)

    assert state.agent.position[0].tolist() == [1, 1]
    assert state.agent.position[1].tolist() != [1, 1]
    assert timestep.reward[1] == 0 and bool(timestep.terminated[1])


def test_the_timestep_reports_the_action_that_was_chosen_even_for_an_agent_that_is_dead():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[0, 0], [2, 2]], [0, 0], energy=[125.0, 0.0]))
    chosen = jnp.array([TURN_LEFT, FORWARD], jnp.int32)

    _, timestep = jax.jit(env.step)(KEY, state, chosen)

    assert timestep.action.tolist() == chosen.tolist()


def test_the_world_runs_on_without_a_time_limit():
    env = Jundland(height=12, width=12, num_agents=2)
    state, _ = env.init(KEY)
    state = state.replace(time=jnp.int32(1_000_000))

    _, timestep = jax.jit(env.step)(KEY, state, jnp.zeros(2, jnp.int32))

    assert not bool(timestep.truncated.any())
    assert env.time_limit() > 0


def test_the_episode_is_truncated_for_everyone_at_the_time_limit():
    env = Jundland(height=12, width=12, num_agents=2, time_limit=5)
    state, _ = env.init(KEY)
    idle = jnp.zeros(env.num_agents, jnp.int32)

    for _ in range(4):
        state, timestep = jax.jit(env.step)(KEY, state, idle)
        assert not bool(timestep.truncated.any())
    state, timestep = jax.jit(env.step)(KEY, state, idle)

    assert bool(timestep.truncated.all())
    assert not bool(timestep.terminated.any())


def test_a_bump_steals_unless_the_victim_is_looking_at_the_thief():
    for victim_direction, lands in ((0, True), (2, False), (1, True), (3, True)):
        env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 0], [1, 1]], [0, victim_direction]))

        state, _ = step(env, state, FORWARD, NOOP)

        assert (float(state.agent.energy[1]) < STARTING_ENERGY - 10) == lands
        assert (float(state.agent.energy[0]) > STARTING_ENERGY + 10) == lands


def test_a_theft_moves_the_whole_catch_from_the_victim_to_the_thief():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 0], [1, 1]], [0, 0], energy=[40.0, 200.0]))

    state, _ = step(env, state, FORWARD, NOOP)

    assert state.agent.position[0].tolist() == [1, 0]
    assert state.agent.energy[0] == pytest.approx(40.0 + env.catch_amount - ACTION_COST[FORWARD])
    assert state.agent.energy[1] == pytest.approx(200.0 - env.catch_amount - ACTION_COST[NOOP])


def test_a_thief_cannot_take_more_than_the_victim_has():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 0], [1, 1]], [0, 0], energy=[40.0, 30.0]))

    state, timestep = step(env, state, FORWARD, NOOP)

    assert bool(timestep.terminated[1]) and state.agent.energy[1] == STARTING_ENERGY
    assert state.agent.energy[0] == pytest.approx(40.0 + 30.0 - ACTION_COST[FORWARD])


def test_a_bump_to_the_side_steals_too():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1], [2, 1]], [0, 0], energy=[40.0, 200.0]))

    state, _ = step(env, state, RIGHT, NOOP)

    assert state.agent.energy[0] == pytest.approx(40.0 + env.catch_amount - ACTION_COST[RIGHT])


def test_a_wall_hides_what_is_behind_it():
    env, state = world(make_cell(EMPTY.at[1, 1].set(1.0), plant(1, 2, 40.0)), agents([[1, 0]], [0]))

    view = env.observe(state)["view"][0]

    assert view[2, 3, 0] == 1 and view[2, 3, 6] == 1
    assert view[1, 3, 1] == 0 and view[1, 3, 6] == 0


def test_the_view_shows_plants_as_a_fraction_of_their_cap():
    env, state = world(make_cell(EMPTY, plant(1, 2, 50.0)), agents([[1, 0]], [0]))

    view = env.observe(state)["view"][0]

    assert float(view[1, 3, 1]) == pytest.approx(50.0 / MAX_PLANT_ENERGY, abs=1e-2) and view[1, 3, 6] == 1
    assert float(view[..., 1].max()) <= 1.0


def test_the_view_separates_the_agent_from_everyone_else():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 0], [1, 1], [0, 0]], [0, 0, 0]))

    view = env.observe(state)["view"][0]

    assert view[3, 3, 2] == 1 and view[3, 3, 3] == 0
    assert view[2, 3, 3] == 1 and view[2, 3, 2] == 0
    assert view[3, 2, 3] == 1 and view[3, 2, 2] == 0


def test_the_view_shows_which_way_the_others_face_relative_to_the_viewer():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1], [1, 2], [0, 1]], [0, 0, 2]))

    view = env.observe(state)["view"][0]

    assert view.shape[-1] == 11
    assert view[2, 3, 7] == 1 and view[2, 3, 9] == 0
    assert view[3, 2, 9] == 1 and view[3, 2, 7] == 0
    assert view[3, 3, 7] == 1


def test_the_field_of_view_is_the_view_window_minus_what_walls_hide():
    env, state = world(
        make_cell(jnp.zeros((9, 9)).at[4, 4].set(1.0), jnp.zeros((9, 9))), agents([[4, 3]], [0]), view_size=5
    )

    field_of_view = env.field_of_view(state)[0]

    assert field_of_view.shape == (9, 9)
    assert bool(field_of_view[4, 3]) and bool(field_of_view[4, 4]) and bool(field_of_view[2, 1])
    assert not bool(field_of_view[4, 5])
    assert not bool(field_of_view[4, 6]) and not bool(field_of_view[4, 7])
    assert not bool(field_of_view[0, 0]) and not bool(field_of_view[8, 8])
    assert int(field_of_view.sum()) < 25


def test_an_agent_without_energy_has_no_field_of_view():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0], energy=0.0))

    assert not bool(env.field_of_view(state).any())


def test_an_agent_in_a_bush_is_hidden_from_anyone_more_than_one_cell_away():
    bush = EMPTY.at[1, 2].set(1.0)
    env, state = world(make_cell(EMPTY, EMPTY, bush), agents([[1, 0], [1, 2]], [0, 0]))

    far = env.observe(state)["view"][0]
    assert far[1, 3, 3] == 0
    assert far[1, 3, 5] == 1

    state = state.replace(agent=state.agent.replace(position=jnp.array([[1, 1], [1, 2]])))
    near = env.observe(state)["view"][0]
    assert near[2, 3, 3] == 1


def test_an_agent_always_sees_itself_even_in_a_bush():
    env, state = world(make_cell(EMPTY, EMPTY, EMPTY.at[1, 1].set(1.0)), agents([[1, 1]], [0]))

    assert env.observe(state)["view"][0][3, 3, 2] == 1


def test_interact_clears_the_bush_that_is_faced():
    env, state = world(make_cell(EMPTY, EMPTY, EMPTY.at[1, 2].set(1.0)), agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.bush.sum() == 0
    assert state.agent.inventory[0] == 0


def test_a_plant_is_harvested_before_the_bush_on_it_is_cleared():
    env, state = world(make_cell(EMPTY, plant(1, 2, 40.0), EMPTY.at[1, 2].set(1.0)), agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)
    assert state.cells.energy.sum() == 0 and state.cells.bush.sum() == 1
    state, _ = step(env, state, INTERACT)
    assert state.cells.bush.sum() == 0


def test_nothing_can_be_placed_on_a_bush():
    env, state = world(make_cell(EMPTY, EMPTY, EMPTY.at[1, 2].set(1.0)), agents([[1, 1]], [0], inventory=1))

    state, _ = step(env, state, PLACE)

    assert state.cells.wall.sum() == 0 and state.agent.inventory[0] == 1


def test_the_cave_generator_grows_bushes_on_open_ground_only():
    state, _ = Jundland(height=48, width=48).init(KEY)

    bush, wall = state.cells.bush, state.cells.wall

    assert 0.03 < float(bush.mean()) < 0.4
    assert float((bush * wall).sum()) == 0


def sheltering(wall, position):
    size, *_ = wall.shape
    env, state = world(make_cell(wall, jnp.zeros((size, size))), agents([position], [0]))
    return env, state.replace(time=STORM)


def test_a_storm_drains_an_agent_in_the_open_but_not_in_calm_weather():
    env, state = sheltering(jnp.zeros((5, 5)), [2, 2])

    calm, _ = step(env, state.replace(time=CALM), NOOP)
    stormy, _ = step(env, state, NOOP)

    assert calm.agent.energy[0] == pytest.approx(STARTING_ENERGY - 1.0)
    assert stormy.agent.energy[0] == pytest.approx(STARTING_ENERGY - 1.0 - env.storm_drain)


def test_every_wall_beside_the_agent_softens_the_storm():
    drained = []
    for count in range(5):
        wall = jnp.zeros((5, 5))
        for row, column in [(1, 2), (3, 2), (2, 1), (2, 3)][:count]:
            wall = wall.at[row, column].set(1.0)
        env, state = sheltering(wall, [2, 2])
        state, _ = step(env, state, NOOP)
        drained.append(STARTING_ENERGY - float(state.agent.energy[0]) - 1.0)

    assert drained == pytest.approx([3.0, 2.25, 1.5, 0.75, 0.0])


def test_the_edge_of_the_map_counts_as_a_wall():
    env, state = sheltering(jnp.zeros((3, 3)).at[1, 0].set(1.0).at[0, 1].set(1.0), [0, 0])

    state, timestep = step(env, state, NOOP)

    assert state.agent.energy[0] == pytest.approx(STARTING_ENERGY - 1.0)
    assert float(timestep.info["sheltered"]) == 1


def test_the_storm_is_announced_before_it_arrives():
    env = Jundland(storm_period=10, storm_length=4, storm_warning=2)
    state, _ = env.init(KEY)
    ramp = [float(env.weather(state.replace(time=jnp.int32(time)))[0]) for time in (2, 4, 5, 6, 9)]

    assert ramp == pytest.approx([0.0, 0.5, 1.0, 1.0, 0.0])
    assert float(env.observe(state.replace(time=jnp.int32(5)))["storm"][0]) == 1.0


def test_an_agent_can_wall_itself_in_with_the_dirt_it_carries():
    cells = make_cell(jnp.zeros((3, 3)).at[0, 0].set(1.0).at[2, 0].set(1.0), jnp.zeros((3, 3)))
    env, state = world(cells, agents([[1, 0]], [0], inventory=1))

    state, _ = step(env, state, PLACE)

    assert state.cells.wall[1, 1] == 1
    assert int(env.walled(state.agent.position, state.cells.wall)[0]) == 4


def test_the_first_timestep_reports_the_same_info_as_every_other():
    env = Jundland(height=12, width=12, num_agents=2)
    state, first = env.init(KEY)
    _, later = jax.jit(env.step)(KEY, state, jnp.zeros(2, jnp.int32))

    assert jax.tree.structure(first.info) == jax.tree.structure(later.info)
    assert jax.tree.structure(first.obs) == jax.tree.structure(later.obs)


def test_the_info_reports_rates_per_acting_agent_and_fractions_of_what_happened():
    energy = plant(1, 2, 100.0).at[0, 0].set(40.0)
    env, state = world(
        make_cell(EMPTY, energy), agents([[1, 1], [2, 0], [2, 1], [0, 2]], [0, 0, 0, 0], energy=[100.0, 50.0, 50.0, 0.0])
    )

    after, timestep = step(env, state, INTERACT, FORWARD, BROADCAST, NOOP)
    info = {name: float(value) for name, value in timestep.info.items()}
    living = np.asarray(after.agent.energy)[np.asarray(after.agent.energy) > 0]

    assert info["deaths"] == pytest.approx(0.5)
    assert info["energy"] == pytest.approx(living.mean())
    assert info["harvests"] == pytest.approx(1 / 3)
    assert info["mature_harvests"] == 1.0
    assert info["thefts"] == pytest.approx(1 / 3)
    assert info["broadcasts"] == pytest.approx(1 / 3)
    assert info["placements"] == 0.0
    assert np.isnan(info["sheltered"])


def test_the_info_has_nothing_to_say_about_what_did_not_happen():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0], energy=0.0))

    _, timestep = step(env, state, NOOP)
    info = {name: float(value) for name, value in timestep.info.items()}

    assert info["deaths"] == 1.0 and info["plants"] == 0.0
    assert info["energy"] == STARTING_ENERGY
    for name in ("harvests", "mature_harvests", "thefts", "broadcasts", "placements", "sheltered"):
        assert np.isnan(info[name]), name


def test_a_dead_agent_is_reborn_at_once_on_open_ground_nobody_stands_on():
    wall = jnp.zeros((FIELD, FIELD)).at[::2, ::2].set(1.0)
    alive = [[1, 1], [1, 3], [3, 1]]
    dead = [[1, 5], [5, 1], [7, 7], [3, 3]]
    energy = [100.0] * len(alive) + [0.0] * len(dead)
    env, state = world(
        make_cell(wall, jnp.zeros((FIELD, FIELD))),
        agents(alive + dead, [0] * 7, energy=energy, inventory=3),
        initial_dirt=2,
    )
    keys = jax.random.split(KEY, 200)
    after, timestep = jax.jit(jax.vmap(lambda key: env.step(key, state, jnp.zeros(7, jnp.int32))))(keys)
    positions = np.asarray(after.agent.position)

    assert np.asarray(timestep.terminated)[:, 3:].all() and not np.asarray(timestep.terminated)[:, :3].any()
    assert (np.asarray(after.agent.energy)[:, 3:] == STARTING_ENERGY).all()
    assert (np.asarray(after.agent.inventory)[:, 3:] == 2).all()
    assert (np.asarray(wall)[positions[..., 0], positions[..., 1]] == 0).all()
    assert all(len({tuple(cell) for cell in world}) == 7 for world in positions)
    assert len({tuple(cell) for cell in positions[:, 3]}) > 5


def test_rebirth_leaves_the_world_as_it_was():
    env, state = world(make_cell(EMPTY.at[0, 0].set(1.0), plant(2, 2, 40.0)), agents([[1, 1]], [0], energy=0.0))

    after, timestep = step(env, state, NOOP)

    assert (after.cells.wall == state.cells.wall).all()
    assert after.cells.energy[2, 2] == pytest.approx(40.0 + GROWTH)
    assert int(after.time) == 1 and not bool(timestep.truncated.any())


def test_a_newborn_sees_its_new_life():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0], energy=0.0))

    _, timestep = step(env, state, NOOP)

    assert float(timestep.obs["energy"][0]) == pytest.approx(STARTING_ENERGY / MAX_ENERGY, rel=1e-2)
    assert timestep.obs["view"][0][3, 3, 2] == 1


def test_an_agent_with_nowhere_to_be_reborn_waits_dead():
    env, state = world(
        make_cell(jnp.zeros((1, 2)).at[0, 1].set(1.0), jnp.zeros((1, 2))), agents([[0, 0], [0, 1]], [0, 0], energy=[100.0, 0.0])
    )

    after, timestep = step(env, state, NOOP, NOOP)

    assert after.agent.energy[1] == 0 and bool(timestep.terminated[1])
    assert after.agent.position[1].tolist() == [0, 1]


def test_scent_spreads_from_a_plant_and_fades_with_distance():
    env, state = world(make_cell(jnp.zeros((15, 15)), plant(7, 7, MAX_PLANT_ENERGY, 15)), agents([[0, 0]], [0]))
    scent = np.asarray(state.cells.scent)[7, 7:]

    assert (np.diff(scent) < 0).all()
    assert scent[6] > 0


def test_walls_block_scent_and_hold_none():
    wall = jnp.zeros((15, 15)).at[:, 9].set(1.0).at[0, 9].set(0.0)
    env, state = world(make_cell(wall, plant(7, 7, MAX_PLANT_ENERGY, 15)), agents([[0, 0]], [0]))
    scent = np.asarray(state.cells.scent)

    assert (scent[:, 9][1:] == 0).all()
    assert scent[7, 10] < scent[7, 5]


def test_ripe_plants_smell_stronger_and_eaten_plants_fade():
    ripe = world(make_cell(jnp.zeros((9, 9)), plant(4, 4, MAX_PLANT_ENERGY, 9)), agents([[0, 0]], [0]))[1]
    young = world(make_cell(jnp.zeros((9, 9)), plant(4, 4, SEED_ENERGY, 9)), agents([[0, 0]], [0]))[1]
    assert ripe.cells.scent[4, 6] > young.cells.scent[4, 6]

    env, state = world(make_cell(jnp.zeros((9, 9)), plant(4, 4, MAX_PLANT_ENERGY, 9)), agents([[4, 3]], [0]))
    before = float(state.cells.scent[4, 6])
    state, _ = step(env, state, INTERACT)
    for _ in range(20):
        state, _ = step(env, state, NOOP)
    assert float(state.cells.scent[4, 6]) < before


def test_an_agent_smells_where_it_stands_and_on_each_side_relative_to_its_facing():
    energy = plant(4, 8, MAX_PLANT_ENERGY, 9)
    east = world(make_cell(jnp.zeros((9, 9)), energy), agents([[4, 4]], [0]))
    north = world(make_cell(jnp.zeros((9, 9)), energy), agents([[4, 4]], [3]))

    here, front, right, back, left = np.asarray(east[0].observe(east[1])["scent"][0], np.float32)
    assert front > here > back
    _, front, right, back, left = np.asarray(north[0].observe(north[1])["scent"][0], np.float32)
    assert right > left


def test_make_builds_jundland():
    assert isinstance(dune_sea.make("jundland", num_agents=3), Jundland)


def test_every_render_draws_the_field_of_view():
    env = Jundland(height=10, width=10, num_agents=4)
    state, _ = env.init(KEY)
    nothing = np.zeros(env.field_of_view(state).shape, bool)

    drawn = env.render(state, 1)
    bare = render(state, 1, env.teams, nothing)

    assert drawn.shape == bare.shape
    assert bool((drawn != bare).any())


def test_a_blended_render_slides_agents_between_their_two_positions():
    env, after = world(make_cell(EMPTY, EMPTY), agents([[1, 2]], [0]))
    before = after.replace(agent=after.agent.replace(position=jnp.array([[1, 1]])))

    start = env.render(after, 1, storm=0.0, previous=before, blend=0.0)
    middle = env.render(after, 1, storm=0.0, previous=before, blend=0.5)
    end = env.render(after, 1, storm=0.0, previous=before, blend=1.0)

    assert bool((start == env.render(before, 1, storm=0.0)).all())
    assert bool((end == env.render(after, 1, storm=0.0)).all())
    assert bool((middle != start).any()) and bool((middle != end).any())


def test_an_agent_that_jumps_far_is_not_dragged_across_the_map():
    env, after = world(make_cell(jnp.zeros((8, 8)), jnp.zeros((8, 8))), agents([[1, 6]], [0]))
    before = after.replace(agent=after.agent.replace(position=jnp.array([[1, 0]])))

    drawn = env.render(after, 1, storm=0.0, previous=before, blend=0.0)

    assert bool((drawn == env.render(after, 1, storm=0.0)).all())


def test_the_sandstorm_follows_the_weather_moves_with_time_and_can_be_switched_off():
    env, state = world(make_cell(EMPTY, EMPTY), agents([[1, 1]], [0]))
    stormy, later = state.replace(time=STORM), state.replace(time=STORM + 5)

    assert bool((env.render(state, 1) == env.render(state, 1, storm=0.0)).all())
    assert bool((env.render(stormy, 1) != env.render(later, 1)).any())
    assert bool((env.render(stormy, 1, storm=0.0) == env.render(later, 1, storm=0.0)).all())
    assert bool((env.render(stormy, 1) != env.render(stormy, 1, storm=0.0)).any())
