import jax
import jax.numpy as jnp
import pytest

from boonta.environments import dune_sea
from boonta.environments.dune_sea import Agent, Cell
from boonta.environments.dune_sea.dune_sea import (BROADCAST, CATCH_AMOUNT,
                                                    ENERGY_COST, FORWARD,
                                                    INTERACT, NOOP, PLACE,
                                                    TURN_LEFT)
from boonta.environments.dune_sea.state import (MATURE_ENERGY, MAX_PLANT_ENERGY,
                                                SEED_ENERGY)

SIZE = 3
EMPTY = jnp.zeros((SIZE, SIZE))
KEY = jax.random.key(0)


def agents(positions, directions, energy=0.5, inventory=0):
    count = len(directions)
    return Agent(
        position=jnp.array(positions),
        direction=jnp.array(directions),
        energy=jnp.broadcast_to(jnp.asarray(energy, jnp.float32), (count,)),
        broadcast=jnp.zeros(count, bool),
        inventory=jnp.broadcast_to(jnp.asarray(inventory, jnp.int32), (count,)),
    )


def world(name, cells, agent, **kwargs):
    count = agent.position.shape[0]
    env = dune_sea.make(
        name,
        height=SIZE,
        width=SIZE,
        num_agents=count,
        generator=lambda key, height, width, num_agents: (cells, agent),
        **kwargs,
    )
    state, _ = env.init(KEY)
    return env, state


def step(env, state, *actions):
    return jax.jit(env.step)(KEY, state, jnp.array(actions, jnp.int32))


def plant(row, column, amount):
    return EMPTY.at[row, column].set(amount)


def test_interact_takes_the_energy_of_the_cell_being_faced():
    cells = Cell(EMPTY, EMPTY, plant(1, 2, 0.4))
    env, state = world("dune_sea", cells, agents([[1, 1]], [0]))

    state, timestep = step(env, state, INTERACT)

    assert state.cells.energy.sum() == 0
    assert state.agent.energy[0] == pytest.approx(0.5 + 0.4 - ENERGY_COST)
    assert timestep.reward[0] == pytest.approx(0.4 - ENERGY_COST)


@pytest.mark.parametrize("direction", [1, 2, 3])
def test_interact_does_nothing_unless_the_agent_faces_the_energy(direction):
    cells = Cell(EMPTY, EMPTY, plant(1, 2, 0.4))
    env, state = world("dune_sea", cells, agents([[1, 1]], [direction]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.energy[1, 2] == pytest.approx(0.4)


def test_standing_on_energy_does_not_collect_it():
    cells = Cell(EMPTY, EMPTY, plant(1, 1, 0.4))
    env, state = world("dune_sea", cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.energy[1, 1] == pytest.approx(0.4)


def test_interact_mines_a_wall_into_the_inventory_before_collecting():
    cells = Cell(plant(1, 2, 1.0), EMPTY, plant(1, 2, 0.4))
    env, state = world("dune_sea", cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.wall.sum() == 0
    assert state.agent.inventory[0] == 1
    assert state.cells.energy[1, 2] == pytest.approx(0.4)


def test_two_agents_facing_one_plant_cannot_both_take_it():
    cells = Cell(EMPTY, EMPTY, plant(1, 1, 0.4))
    env, state = world("dune_sea", cells, agents([[1, 0], [1, 2]], [0, 2]))

    state, _ = step(env, state, INTERACT, INTERACT)

    gains = state.agent.energy - 0.5 + ENERGY_COST
    assert sorted(float(gain) for gain in gains) == pytest.approx([0.0, 0.4], abs=1e-6)


def test_turning_changes_what_the_agent_can_interact_with():
    cells = Cell(EMPTY, EMPTY, plant(0, 1, 0.4))
    env, state = world("dune_sea", cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, TURN_LEFT)
    state, _ = step(env, state, INTERACT)

    assert state.cells.energy.sum() == 0


def test_noop_and_broadcast_leave_the_agent_where_it_is_and_cost_energy():
    cells = Cell(EMPTY, EMPTY, EMPTY)
    env, state = world("dune_sea", cells, agents([[1, 1]], [0]))

    for action in (NOOP, BROADCAST):
        after, _ = step(env, state, action)
        assert (after.agent.position == state.agent.position).all()
        assert after.agent.energy[0] == pytest.approx(0.5 - ENERGY_COST)


def test_commons_plants_ripen_up_to_a_cap():
    cells = Cell(EMPTY, EMPTY, plant(0, 0, 0.1).at[2, 2].set(MAX_PLANT_ENERGY - 0.001))
    env, state = world("commons", cells, agents([[1, 1]], [0]), growth_steps=100)

    state, _ = step(env, state, NOOP)

    growth = (MAX_PLANT_ENERGY - SEED_ENERGY) / 100
    assert state.cells.energy[0, 0] == pytest.approx(0.1 + growth)
    assert state.cells.energy[2, 2] == pytest.approx(MAX_PLANT_ENERGY)
    assert state.cells.energy[0, 2] == 0


def test_commons_ripe_plant_regrows_from_a_seed_after_harvest():
    cells = Cell(EMPTY, EMPTY, plant(1, 2, MAX_PLANT_ENERGY))
    env, state = world("commons", cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.agent.energy[0] == pytest.approx(0.5 + MAX_PLANT_ENERGY - ENERGY_COST)
    assert state.cells.energy[1, 2] == pytest.approx(SEED_ENERGY)


def test_commons_plant_harvested_before_it_matures_is_gone():
    cells = Cell(EMPTY, EMPTY, plant(1, 2, MATURE_ENERGY - 0.1))
    env, state = world("commons", cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.agent.energy[0] == pytest.approx(0.5 + MATURE_ENERGY - 0.1 - ENERGY_COST)
    assert state.cells.energy[1, 2] == 0
    for _ in range(3):
        state, _ = step(env, state, NOOP)
    assert state.cells.energy.sum() == 0


def test_place_puts_a_wall_on_the_faced_cell_and_spends_dirt():
    cells = Cell(EMPTY, EMPTY, EMPTY)
    env, state = world("dune_sea", cells, agents([[1, 1]], [0], inventory=1))

    state, _ = step(env, state, PLACE)

    assert state.cells.wall[1, 2] == 1
    assert state.agent.inventory[0] == 0
    state, _ = step(env, state, PLACE)
    assert state.cells.wall.sum() == 1


def test_place_needs_dirt_and_a_free_cell():
    cells = Cell(EMPTY, EMPTY, EMPTY)
    env, state = world("dune_sea", cells, agents([[1, 1]], [0], inventory=0))
    state, _ = step(env, state, PLACE)
    assert state.cells.wall.sum() == 0

    env, state = world("dune_sea", cells, agents([[1, 1], [1, 2]], [0, 2], inventory=1))
    state, _ = step(env, state, PLACE, NOOP)
    assert state.cells.wall.sum() == 0
    assert state.agent.inventory.tolist() == [1, 1]


def test_placed_wall_can_be_mined_back():
    cells = Cell(EMPTY, EMPTY, EMPTY)
    env, state = world("dune_sea", cells, agents([[1, 1]], [0], inventory=1))

    state, _ = step(env, state, PLACE)
    state, _ = step(env, state, INTERACT)

    assert state.cells.wall.sum() == 0
    assert state.agent.inventory[0] == 1


def test_an_agent_without_energy_is_inert_and_neither_blocks_nor_shows():
    cells = Cell(EMPTY, EMPTY, EMPTY)
    env, state = world("dune_sea", cells, agents([[1, 0], [1, 1]], [0, 0], energy=[0.5, 0.0]))

    assert state.agent.energy[1] == 0
    view = env.observe(state)["view"][0]
    assert view[2, 3, 3] == 0 and view[2, 3, 4] == 0

    state, timestep = step(env, state, FORWARD, FORWARD)

    assert state.agent.position[0].tolist() == [1, 1]
    assert state.agent.position[1].tolist() == [1, 1]
    assert state.agent.energy[1] == 0 and timestep.reward[1] == 0
    assert bool(timestep.terminated[1])


def test_a_thief_takes_energy_from_the_agent_it_faces():
    cells = Cell(EMPTY, EMPTY, EMPTY)
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    env, state = world(
        "dune_sea", cells, agents([[1, 0], [1, 1]], [0, 2], energy=[0.9, 0.2]), **roles
    )

    state, timestep = step(env, state, NOOP, INTERACT)

    assert state.agent.energy[0] == pytest.approx(0.9 - CATCH_AMOUNT - ENERGY_COST)
    assert state.agent.energy[1] == pytest.approx(0.2 + CATCH_AMOUNT - ENERGY_COST)
    assert timestep.reward[1] == pytest.approx(CATCH_AMOUNT - ENERGY_COST)


def test_a_thief_cannot_take_more_than_the_victim_has():
    cells = Cell(EMPTY, EMPTY, EMPTY)
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    env, state = world(
        "dune_sea", cells, agents([[1, 0], [1, 1]], [0, 2], energy=[0.3, 0.2]), **roles
    )

    state, _ = step(env, state, NOOP, INTERACT)

    assert state.agent.energy[0] == 0
    assert state.agent.energy[1] == pytest.approx(0.2 + 0.3 - ENERGY_COST)


def test_only_thieves_steal_and_only_eaters_eat():
    cells = Cell(EMPTY, EMPTY, plant(1, 2, 0.4))
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    env, state = world(
        "dune_sea", cells, agents([[1, 0], [1, 1]], [0, 0], energy=[0.5, 0.5]), **roles
    )

    state, _ = step(env, state, INTERACT, INTERACT)

    assert state.cells.energy[1, 2] == pytest.approx(0.4)
    assert state.agent.energy[0] == pytest.approx(0.5 - ENERGY_COST)
    assert state.agent.energy[1] == pytest.approx(0.5 - ENERGY_COST)


def test_a_wall_hides_what_is_behind_it():
    wall = EMPTY.at[1, 1].set(1.0)
    cells = Cell(wall, EMPTY, plant(1, 2, 0.4))
    env, state = world("dune_sea", cells, agents([[1, 0]], [0]))

    view = env.observe(state)["view"][0]

    assert view[2, 3, 0] == 1 and view[2, 3, 6] == 1
    assert view[1, 3, 2] == 0 and view[1, 3, 6] == 0


def test_without_a_wall_the_same_cell_is_visible():
    cells = Cell(EMPTY, EMPTY, plant(1, 2, 0.4))
    env, state = world("dune_sea", cells, agents([[1, 0]], [0]))

    view = env.observe(state)["view"][0]

    assert view[1, 3, 2] == pytest.approx(0.4) and view[1, 3, 6] == 1


def test_the_view_separates_teammates_from_opponents():
    cells = Cell(EMPTY, EMPTY, EMPTY)
    env, state = world(
        "dune_sea", cells, agents([[1, 0], [1, 1], [0, 0]], [0, 0, 0]), teams=[0, 1, 0]
    )

    view = env.observe(state)["view"][0]

    assert view[3, 3, 3] == 1 and view[3, 3, 4] == 0
    assert view[2, 3, 4] == 1 and view[2, 3, 3] == 0
    assert view[3, 2, 3] == 1


def test_seekers_are_frozen_while_hiders_prepare():
    env = dune_sea.make("hide_and_seek", height=12, width=12, preparation_steps=3)
    state, _ = env.init(KEY)
    start = state.agent
    seekers = env.thieves
    actions = jnp.stack([jnp.array(FORWARD), jnp.array(TURN_LEFT)])[jnp.arange(env.num_agents) % 2]

    for _ in range(3):
        state, _ = jax.jit(env.step)(KEY, state, actions)
    assert (state.agent.position[seekers] == start.position[seekers]).all()
    assert (state.agent.direction[seekers] == start.direction[seekers]).all()
    assert (state.agent.direction[~seekers] != start.direction[~seekers]).any()

    state, _ = jax.jit(env.step)(KEY, state, jnp.full(env.num_agents, TURN_LEFT, jnp.int32))
    assert (state.agent.direction[seekers] != start.direction[seekers]).all()


def test_hide_and_seek_gives_each_role_its_powers():
    env = dune_sea.make("hide_and_seek", num_hiders=2, num_seekers=3)

    assert env.num_agents == 5
    assert env.teams.tolist() == [0, 0, 1, 1, 1]
    assert env.eaters.tolist() == [True, True, False, False, False]
    assert env.thieves.tolist() == [False, False, True, True, True]


def test_the_game_ends_for_everyone_when_a_team_is_eliminated():
    env = dune_sea.make("hide_and_seek", height=12, width=12, preparation_steps=0)
    state, _ = env.init(KEY)
    energy = jnp.where(env.thieves, 0.5, 0.0)
    state = state.replace(agent=state.agent.replace(energy=energy))

    state, timestep = jax.jit(env.step)(KEY, state, jnp.zeros(env.num_agents, jnp.int32))

    assert bool(timestep.terminated.all())
    assert bool(timestep.done.all())


def test_the_game_goes_on_while_both_teams_have_someone_alive():
    env = dune_sea.make("hide_and_seek", height=12, width=12)
    state, _ = env.init(KEY)

    state, timestep = jax.jit(env.step)(KEY, state, jnp.zeros(env.num_agents, jnp.int32))

    assert not bool(timestep.terminated.any())


def test_the_episode_is_truncated_for_everyone_at_the_time_limit():
    env = dune_sea.make("hide_and_seek", height=12, width=12, time_limit=5)
    state, _ = env.init(KEY)
    step = jax.jit(env.step)
    idle = jnp.zeros(env.num_agents, jnp.int32)

    for _ in range(4):
        state, timestep = step(KEY, state, idle)
        assert not bool(timestep.truncated.any())
    state, timestep = step(KEY, state, idle)

    assert bool(timestep.truncated.all())
    assert not bool(timestep.terminated.any())
    assert bool(timestep.done.all())


def test_hiders_start_with_dirt_and_seekers_with_none():
    env = dune_sea.make("hide_and_seek", num_hiders=2, num_seekers=3, initial_dirt=4)

    state, timestep = env.init(KEY)

    assert state.agent.inventory.tolist() == [4, 4, 0, 0, 0]
    assert timestep.obs["inventory"].tolist() == [4.0, 4.0, 0.0, 0.0, 0.0]


def test_a_catch_in_hide_and_seek_removes_a_hider_in_one_interaction():
    env = dune_sea.make("hide_and_seek", height=12, width=12, num_hiders=1, num_seekers=1, preparation_steps=0)
    state, _ = env.init(KEY)
    position = jnp.array([[5, 5], [5, 6]])
    state = state.replace(
        agent=state.agent.replace(position=position, direction=jnp.array([0, 2]), energy=jnp.array([1.0, 0.4]))
    )

    state, timestep = jax.jit(env.step)(KEY, state, jnp.array([NOOP, INTERACT], jnp.int32))

    assert state.agent.energy[0] == 0
    assert bool(timestep.terminated.all())

