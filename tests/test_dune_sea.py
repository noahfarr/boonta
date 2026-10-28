import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments import dune_sea
from boonta.environments.dune_sea import Agent
from boonta.environments.dune_sea.hide_and_seek import HideAndSeek
from boonta.environments.dune_sea.render import render
from boonta.environments.dune_sea import Cell as StateCell
from boonta.environments.dune_sea.dune_sea import (DuneSea, BROADCAST, CATCH_AMOUNT,
                                                    ENERGY_COST, FORWARD,
                                                    INTERACT, NOOP, PLACE,
                                                    TURN_LEFT)
from boonta.environments.dune_sea.state import (GROWTH_STEPS, MATURE_ENERGY, MAX_PLANT_ENERGY,
                                                SEED_ENERGY)

def make_cell(wall, energy, bush=None):
    blank = jnp.zeros_like(wall)
    return StateCell(wall, energy, blank if bush is None else bush)


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


class Sandbox(DuneSea):
    def __init__(self, height, width, num_agents, generator, view_size=7, time_limit=256, teams=None, eaters=None, thieves=None, **kwargs):
        super().__init__(
            height,
            width,
            num_agents,
            view_size,
            time_limit,
            generator,
            teams=[0] * num_agents if teams is None else teams,
            eaters=[True] * num_agents if eaters is None else eaters,
            thieves=[False] * num_agents if thieves is None else thieves,
            **kwargs,
        )


def world(name, cells, agent, **kwargs):
    count = agent.position.shape[0]
    make = name if isinstance(name, type) else functools.partial(dune_sea.make, name)
    env = make(
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
    cells = make_cell(EMPTY, plant(1, 2, 0.4))
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]))

    state, timestep = step(env, state, INTERACT)

    assert state.cells.energy.sum() == 0
    assert state.agent.energy[0] == pytest.approx(0.5 + 0.4 - ENERGY_COST)
    assert timestep.reward[0] == pytest.approx(0.4 - ENERGY_COST)


@pytest.mark.parametrize("direction", [1, 2, 3])
def test_interact_does_nothing_unless_the_agent_faces_the_energy(direction):
    cells = make_cell(EMPTY, plant(1, 2, 0.4))
    env, state = world(Sandbox, cells, agents([[1, 1]], [direction]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.energy[1, 2] == pytest.approx(0.4)


def test_standing_on_energy_does_not_collect_it():
    cells = make_cell(EMPTY, plant(1, 1, 0.4))
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.energy[1, 1] == pytest.approx(0.4)


def test_interact_mines_a_wall_into_the_inventory_before_collecting():
    cells = make_cell(plant(1, 2, 1.0), plant(1, 2, 0.4))
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.wall.sum() == 0
    assert state.agent.inventory[0] == 1
    assert state.cells.energy[1, 2] == pytest.approx(0.4)


def test_two_agents_facing_one_plant_cannot_both_take_it():
    cells = make_cell(EMPTY, plant(1, 1, 0.4))
    env, state = world(Sandbox, cells, agents([[1, 0], [1, 2]], [0, 2]))

    state, _ = step(env, state, INTERACT, INTERACT)

    gains = state.agent.energy - 0.5 + ENERGY_COST
    assert sorted(float(gain) for gain in gains) == pytest.approx([0.0, 0.4], abs=1e-6)


def test_turning_changes_what_the_agent_can_interact_with():
    cells = make_cell(EMPTY, plant(0, 1, 0.4))
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, TURN_LEFT)
    state, _ = step(env, state, INTERACT)

    assert state.cells.energy.sum() == 0


def test_noop_and_broadcast_leave_the_agent_where_it_is_and_cost_energy():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]))

    for action in (NOOP, BROADCAST):
        after, _ = step(env, state, action)
        assert (after.agent.position == state.agent.position).all()
        assert after.agent.energy[0] == pytest.approx(0.5 - ENERGY_COST)


def test_plants_ripen_up_to_a_cap():
    cells = make_cell(EMPTY, plant(0, 0, 0.1).at[2, 2].set(MAX_PLANT_ENERGY - 0.001))
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]), growth_steps=100)

    state, _ = step(env, state, NOOP)

    growth = (MAX_PLANT_ENERGY - SEED_ENERGY) / 100
    assert state.cells.energy[0, 0] == pytest.approx(0.1 + growth)
    assert state.cells.energy[2, 2] == pytest.approx(MAX_PLANT_ENERGY)
    assert state.cells.energy[0, 2] == 0


def test_ripe_plant_regrows_from_a_seed_after_harvest():
    cells = make_cell(EMPTY, plant(1, 2, MAX_PLANT_ENERGY))
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]), growth_steps=GROWTH_STEPS)

    state, _ = step(env, state, INTERACT)

    assert state.agent.energy[0] == pytest.approx(0.5 + MAX_PLANT_ENERGY - ENERGY_COST)
    assert state.cells.energy[1, 2] == pytest.approx(SEED_ENERGY)


def test_plant_harvested_before_it_matures_is_gone():
    cells = make_cell(EMPTY, plant(1, 2, MATURE_ENERGY - 0.1))
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]), growth_steps=GROWTH_STEPS)

    state, _ = step(env, state, INTERACT)

    assert state.agent.energy[0] == pytest.approx(0.5 + MATURE_ENERGY - 0.1 - ENERGY_COST)
    assert state.cells.energy[1, 2] == 0
    for _ in range(3):
        state, _ = step(env, state, NOOP)
    assert state.cells.energy.sum() == 0


def test_place_puts_a_wall_on_the_faced_cell_and_spends_dirt():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(Sandbox, cells, agents([[1, 1]], [0], inventory=1))

    state, _ = step(env, state, PLACE)

    assert state.cells.wall[1, 2] == 1
    assert state.agent.inventory[0] == 0
    state, _ = step(env, state, PLACE)
    assert state.cells.wall.sum() == 1


def test_place_needs_dirt_and_a_free_cell():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(Sandbox, cells, agents([[1, 1]], [0], inventory=0))
    state, _ = step(env, state, PLACE)
    assert state.cells.wall.sum() == 0

    env, state = world(Sandbox, cells, agents([[1, 1], [1, 2]], [0, 2], inventory=1))
    state, _ = step(env, state, PLACE, NOOP)
    assert state.cells.wall.sum() == 0
    assert state.agent.inventory.tolist() == [1, 1]


def test_placed_wall_can_be_mined_back():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(Sandbox, cells, agents([[1, 1]], [0], inventory=1))

    state, _ = step(env, state, PLACE)
    state, _ = step(env, state, INTERACT)

    assert state.cells.wall.sum() == 0
    assert state.agent.inventory[0] == 1


def test_an_agent_without_energy_is_inert_and_neither_blocks_nor_shows():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(Sandbox, cells, agents([[1, 0], [1, 1]], [0, 0], energy=[0.5, 0.0]))

    assert state.agent.energy[1] == 0
    view = env.observe(state)["view"][0]
    assert view[2, 3, 2] == 0 and view[2, 3, 3] == 0

    state, timestep = step(env, state, FORWARD, FORWARD)

    assert state.agent.position[0].tolist() == [1, 1]
    assert state.agent.position[1].tolist() == [1, 1]
    assert state.agent.energy[1] == 0 and timestep.reward[1] == 0
    assert bool(timestep.terminated[1])


def test_a_thief_takes_energy_from_the_agent_it_faces():
    cells = make_cell(EMPTY, EMPTY)
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    env, state = world(
        Sandbox, cells, agents([[1, 0], [1, 1]], [0, 2], energy=[0.9, 0.2]), **roles
    )

    state, timestep = step(env, state, NOOP, INTERACT)

    assert state.agent.energy[0] == pytest.approx(0.9 - CATCH_AMOUNT - ENERGY_COST)
    assert state.agent.energy[1] == pytest.approx(0.2 + CATCH_AMOUNT - ENERGY_COST)
    assert timestep.reward[1] == pytest.approx(CATCH_AMOUNT - ENERGY_COST)


def test_a_thief_cannot_take_more_than_the_victim_has():
    cells = make_cell(EMPTY, EMPTY)
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    env, state = world(
        Sandbox, cells, agents([[1, 0], [1, 1]], [0, 2], energy=[0.3, 0.2]), **roles
    )

    state, _ = step(env, state, NOOP, INTERACT)

    assert state.agent.energy[0] == 0
    assert state.agent.energy[1] == pytest.approx(0.2 + 0.3 - ENERGY_COST)


def test_only_thieves_steal_and_only_eaters_eat():
    cells = make_cell(EMPTY, plant(1, 2, 0.4))
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    env, state = world(
        Sandbox, cells, agents([[1, 0], [1, 1]], [0, 0], energy=[0.5, 0.5]), **roles
    )

    state, _ = step(env, state, INTERACT, INTERACT)

    assert state.cells.energy[1, 2] == pytest.approx(0.4)
    assert state.agent.energy[0] == pytest.approx(0.5 - ENERGY_COST)
    assert state.agent.energy[1] == pytest.approx(0.5 - ENERGY_COST)


def test_a_wall_hides_what_is_behind_it():
    wall = EMPTY.at[1, 1].set(1.0)
    cells = make_cell(wall, plant(1, 2, 0.4))
    env, state = world(Sandbox, cells, agents([[1, 0]], [0]))

    view = env.observe(state)["view"][0]

    assert view[2, 3, 0] == 1 and view[2, 3, 6] == 1
    assert view[1, 3, 1] == 0 and view[1, 3, 6] == 0


def test_without_a_wall_the_same_cell_is_visible():
    cells = make_cell(EMPTY, plant(1, 2, 0.4))
    env, state = world(Sandbox, cells, agents([[1, 0]], [0]))

    view = env.observe(state)["view"][0]

    assert float(view[1, 3, 1]) == pytest.approx(0.4, abs=1e-2) and view[1, 3, 6] == 1


def test_the_view_separates_teammates_from_opponents():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(
        Sandbox, cells, agents([[1, 0], [1, 1], [0, 0]], [0, 0, 0]), teams=[0, 1, 0]
    )

    view = env.observe(state)["view"][0]

    assert view[3, 3, 2] == 1 and view[3, 3, 3] == 0
    assert view[2, 3, 3] == 1 and view[2, 3, 2] == 0
    assert view[3, 2, 2] == 1


def test_hide_and_seek_gives_each_role_its_powers():
    env = dune_sea.make("hide_and_seek", num_hiders=2, num_seekers=3)

    assert env.num_agents == 5
    assert env.teams.tolist() == [0, 0, 1, 1, 1]
    assert env.eaters.tolist() == [True, True, False, False, False]
    assert env.thieves.tolist() == [False, False, True, True, True]


def test_the_game_ends_for_everyone_when_a_team_is_eliminated():
    env = dune_sea.make("hide_and_seek", height=12, width=12)
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
    env = dune_sea.make("hide_and_seek", height=12, width=12, time_limit=5, sight_drain=0.0)
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
    env = dune_sea.make(
        "hide_and_seek",
        height=12,
        width=12,
        num_hiders=1,
        num_seekers=1,
        catch_amount=1.0,
        sight_drain=0.0,
    )
    state, _ = env.init(KEY)
    position = jnp.array([[5, 5], [5, 6]])
    state = state.replace(
        agent=state.agent.replace(position=position, direction=jnp.array([0, 2]), energy=jnp.array([1.0, 0.4]))
    )

    state, timestep = jax.jit(env.step)(KEY, state, jnp.array([NOOP, INTERACT], jnp.int32))

    assert state.agent.energy[0] == 0
    assert bool(timestep.terminated.all())


class Watching(Sandbox):
    def __init__(self, *args, sight_drain, **kwargs):
        super().__init__(*args, **kwargs)
        self.sight_drain = sight_drain

    drain = HideAndSeek.drain


def sighted(cells, agent, **kwargs):
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    return world(Watching, cells, agent, sight_drain=0.1, catch_amount=0.0, **roles, **kwargs)


def test_a_seeker_that_sees_a_hider_takes_energy_from_it_every_step():
    env, state = sighted(make_cell(EMPTY, EMPTY), agents([[1, 0], [1, 2]], [0, 0], energy=[0.8, 0.5]))

    state, timestep = step(env, state, NOOP, NOOP)

    assert state.agent.energy[0] == pytest.approx(0.8 - 0.1 - ENERGY_COST)
    assert state.agent.energy[1] == pytest.approx(0.5 + 0.1 - ENERGY_COST)
    assert timestep.reward[0] == pytest.approx(-0.1 - ENERGY_COST)


def test_a_wall_between_them_stops_the_drain():
    cells = make_cell(EMPTY.at[1, 1].set(1.0), EMPTY)
    env, state = sighted(cells, agents([[1, 0], [1, 2]], [0, 0], energy=[0.8, 0.5]))

    state, _ = step(env, state, NOOP, NOOP)

    assert state.agent.energy[0] == pytest.approx(0.8 - ENERGY_COST)
    assert state.agent.energy[1] == pytest.approx(0.5 - ENERGY_COST)


def test_a_hider_outside_the_view_is_not_drained():
    env = Watching(
        height=12,
        width=12,
        num_agents=2,
        view_size=5,
        teams=[0, 1],
        eaters=[True, False],
        thieves=[False, True],
        sight_drain=0.1,
        generator=lambda key, height, width, num_agents: (
            make_cell(jnp.zeros((12, 12)), jnp.zeros((12, 12))),
            agents([[1, 0], [1, 9]], [0, 0], energy=[0.8, 0.5]),
        ),
    )
    state, _ = env.init(KEY)

    state, _ = step(env, state, NOOP, NOOP)

    assert state.agent.energy[0] == pytest.approx(0.8 - ENERGY_COST)


def test_the_drain_is_limited_to_what_the_hider_has_and_split_between_watchers():
    cells = make_cell(EMPTY, EMPTY)
    roles = dict(teams=[0, 1, 1], eaters=[True, False, False], thieves=[False, True, True])
    env, state = world(
        Watching,
        cells,
        agents([[1, 1], [1, 0], [1, 2]], [0, 0, 0], energy=[0.05, 0.5, 0.5]),
        sight_drain=0.1,
        catch_amount=0.0,
        **roles,
    )

    state, _ = step(env, state, NOOP, NOOP, NOOP)

    assert state.agent.energy[0] == 0
    gains = state.agent.energy[1:] - 0.5 + ENERGY_COST
    assert float(gains.sum()) == pytest.approx(0.05, abs=1e-5)


def test_hide_and_seek_is_played_by_sight_and_not_by_touch():
    env = dune_sea.make("hide_and_seek")

    assert env.sight_drain > 0 and env.catch_amount == 0


def test_the_field_of_view_is_the_view_window_minus_what_walls_hide():
    wall = EMPTY.at[1, 1].set(1.0)
    env = Sandbox(
        height=9,
        width=9,
        num_agents=1,
        view_size=5,
        generator=lambda key, height, width, num_agents: (
            make_cell(jnp.zeros((9, 9)).at[4, 4].set(1.0), jnp.zeros((9, 9))),
            agents([[4, 3]], [0]),
        ),
    )
    state, _ = env.init(KEY)

    field_of_view = env.field_of_view(state)[0]

    assert field_of_view.shape == (9, 9)
    assert bool(field_of_view[4, 3]) and bool(field_of_view[4, 4]) and bool(field_of_view[2, 1])
    assert not bool(field_of_view[4, 5])
    assert not bool(field_of_view[4, 6]) and not bool(field_of_view[4, 7])
    assert not bool(field_of_view[0, 0]) and not bool(field_of_view[8, 8])
    assert int(field_of_view.sum()) < 25


def test_an_agent_without_energy_has_no_field_of_view():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(Sandbox, cells, agents([[1, 1]], [0], energy=0.0))

    assert not bool(env.field_of_view(state).any())


def test_every_render_draws_the_field_of_view():
    env = dune_sea.make("hide_and_seek", height=10, width=10)
    state, _ = env.init(KEY)
    nothing = np.zeros(env.field_of_view(state).shape, bool)

    drawn = env.render(state, 1)
    bare = render(state, 1, env.teams, nothing)

    assert drawn.shape == bare.shape
    assert bool((drawn != bare).any())


def test_an_agent_in_a_bush_is_hidden_from_anyone_more_than_one_cell_away():
    bush = EMPTY.at[1, 2].set(1.0)
    cells = make_cell(EMPTY, EMPTY, bush)
    env, state = world(Sandbox, cells, agents([[1, 0], [1, 2]], [0, 0]), teams=[0, 0])

    far = env.observe(state)["view"][0]
    assert far[3, 3, 2] == 1
    assert far[1, 3, 2] == 0
    assert far[1, 3, 5] == 1

    state = state.replace(agent=state.agent.replace(position=jnp.array([[1, 1], [1, 2]])))
    near = env.observe(state)["view"][0]
    assert near[2, 3, 2] == 1


def test_an_agent_always_sees_itself_even_in_a_bush():
    bush = EMPTY.at[1, 1].set(1.0)
    env, state = world(Sandbox, make_cell(EMPTY, EMPTY, bush), agents([[1, 1]], [0]))

    assert env.observe(state)["view"][0][3, 3, 2] == 1


def test_a_bush_hides_a_hider_from_the_sight_drain_until_a_seeker_is_adjacent():
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    bush = EMPTY.at[1, 2].set(1.0)
    cells = make_cell(EMPTY, EMPTY, bush)

    env, state = world(
        Watching, cells, agents([[1, 2], [1, 0]], [0, 0], energy=[0.8, 0.5]), sight_drain=0.1, catch_amount=0.0, **roles
    )
    state, _ = step(env, state, NOOP, NOOP)
    assert state.agent.energy[0] == pytest.approx(0.8 - ENERGY_COST)

    env, state = world(
        Watching, cells, agents([[1, 2], [1, 1]], [0, 0], energy=[0.8, 0.5]), sight_drain=0.1, catch_amount=0.0, **roles
    )
    state, _ = step(env, state, NOOP, NOOP)
    assert state.agent.energy[0] == pytest.approx(0.8 - 0.1 - ENERGY_COST)


def test_interact_clears_the_bush_that_is_faced():
    bush = EMPTY.at[1, 2].set(1.0)
    env, state = world(Sandbox, make_cell(EMPTY, EMPTY, bush), agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)

    assert state.cells.bush.sum() == 0
    assert state.agent.inventory[0] == 0


def test_a_plant_is_harvested_before_the_bush_on_it_is_cleared():
    bush = EMPTY.at[1, 2].set(1.0)
    cells = make_cell(EMPTY, plant(1, 2, 0.4), bush)
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]))

    state, _ = step(env, state, INTERACT)
    assert state.cells.energy.sum() == 0 and state.cells.bush.sum() == 1
    state, _ = step(env, state, INTERACT)
    assert state.cells.bush.sum() == 0


def test_a_seeker_can_clear_the_bush_a_hider_stands_in():
    roles = dict(teams=[0, 1], eaters=[True, False], thieves=[False, True])
    bush = EMPTY.at[1, 2].set(1.0)
    env, state = world(
        Sandbox,
        make_cell(EMPTY, EMPTY, bush),
        agents([[1, 2], [1, 1]], [0, 0]),
        catch_amount=0.0,
        **roles,
    )

    state, _ = step(env, state, NOOP, INTERACT)

    assert state.cells.bush.sum() == 0


def test_nothing_can_be_placed_on_a_bush():
    bush = EMPTY.at[1, 2].set(1.0)
    env, state = world(Sandbox, make_cell(EMPTY, EMPTY, bush), agents([[1, 1]], [0], inventory=1))

    state, _ = step(env, state, PLACE)

    assert state.cells.wall.sum() == 0 and state.agent.inventory[0] == 1


def test_the_cave_generator_grows_bushes_on_open_ground_only():
    env = dune_sea.make("hide_and_seek", height=48, width=48)
    state, _ = env.init(KEY)

    bush, wall = state.cells.bush, state.cells.wall

    assert 0.03 < float(bush.mean()) < 0.4
    assert float((bush * wall).sum()) == 0


def tagging(cells, agent, **kwargs):
    count = agent.position.shape[0]
    return world(
        Sandbox,
        cells,
        agent,
        thieves=[True] * count,
        catch_amount=0.3,
        tag_from_behind=True,
        **kwargs,
    )


def test_a_tag_lands_only_on_an_agent_that_is_facing_away():
    cells = make_cell(EMPTY, EMPTY)
    for victim_direction, lands in ((0, True), (2, False), (1, False), (3, False)):
        env, state = tagging(cells, agents([[1, 0], [1, 1]], [0, victim_direction], energy=[0.5, 0.5]))

        state, _ = step(env, state, INTERACT, NOOP)

        assert (float(state.agent.energy[1]) < 0.5 - ENERGY_COST - 0.1) == lands
        assert (float(state.agent.energy[0]) > 0.5 + 0.1) == lands


def test_a_tag_moves_energy_from_the_tagged_agent_to_the_tagger():
    cells = make_cell(EMPTY, EMPTY)
    env, state = tagging(cells, agents([[1, 0], [1, 1]], [0, 0], energy=[0.4, 0.8]))

    state, timestep = step(env, state, INTERACT, NOOP)

    assert state.agent.energy[0] == pytest.approx(0.4 + 0.3 - ENERGY_COST)
    assert state.agent.energy[1] == pytest.approx(0.8 - 0.3 - ENERGY_COST)
    assert timestep.reward[0] == pytest.approx(0.3 - ENERGY_COST)


def test_the_view_shows_which_way_the_others_face_relative_to_the_viewer():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(
        Sandbox, cells, agents([[1, 1], [1, 2], [0, 1]], [0, 0, 2])
    )

    view = env.observe(state)["view"][0]

    assert view.shape[-1] == 11
    assert view[2, 3, 7] == 1 and view[2, 3, 9] == 0
    assert view[3, 2, 9] == 1 and view[3, 2, 7] == 0
    assert view[3, 3, 7] == 1


def test_tag_defaults_make_every_agent_a_tagger_who_starts_half_charged():
    env = dune_sea.make("tag")
    state, timestep = env.init(KEY)

    assert env.thieves.all() and env.tag_from_behind
    assert float(state.agent.energy.max()) == pytest.approx(0.5)
    assert timestep.obs["view"].shape[-1] == 11


def test_a_tag_game_ends_when_one_agent_is_left():
    env = dune_sea.make("tag", height=16, width=16, num_agents=4)
    state, _ = env.init(KEY)
    state = state.replace(agent=state.agent.replace(energy=jnp.array([0.5, 0.0, 0.0, 0.0])))

    state, timestep = jax.jit(env.step)(KEY, state, jnp.zeros(4, jnp.int32))

    assert bool(timestep.terminated.all())


def test_a_blended_render_slides_agents_between_their_two_positions():
    cells = make_cell(EMPTY, EMPTY)
    env, after = world(Sandbox, cells, agents([[1, 2]], [0]))
    before = after.replace(agent=after.agent.replace(position=jnp.array([[1, 1]])))

    start = env.render(after, 1, storm=0.0, previous=before, blend=0.0)
    middle = env.render(after, 1, storm=0.0, previous=before, blend=0.5)
    end = env.render(after, 1, storm=0.0, previous=before, blend=1.0)

    assert bool((start == env.render(before, 1, storm=0.0)).all())
    assert bool((end == env.render(after, 1, storm=0.0)).all())
    assert bool((middle != start).any()) and bool((middle != end).any())


def test_an_agent_that_jumps_far_is_not_dragged_across_the_map():
    size = 8
    cells = make_cell(jnp.zeros((size, size)), jnp.zeros((size, size)))
    env = Sandbox(
        height=size,
        width=size,
        generator=lambda key, height, width, num_agents: (cells, agents([[1, 6]], [0])),
    )
    after, _ = env.init(KEY)
    before = after.replace(agent=after.agent.replace(position=jnp.array([[1, 0]])))

    drawn = env.render(after, 1, storm=0.0, previous=before, blend=0.0)

    assert bool((drawn == env.render(after, 1, storm=0.0)).all())


def test_the_sandstorm_moves_with_time_and_can_be_switched_off():
    cells = make_cell(EMPTY, EMPTY)
    env, state = world(Sandbox, cells, agents([[1, 1]], [0]))
    later = state.replace(time=jnp.int32(25))

    assert bool((env.render(state, 1) != env.render(later, 1)).any())
    assert bool((env.render(state, 1, storm=0.0) == env.render(later, 1, storm=0.0)).all())
    assert bool((env.render(state, 1) != env.render(state, 1, storm=0.0)).any())


def test_the_timestep_reports_the_action_that_was_chosen_even_for_an_agent_that_is_dead():
    env = dune_sea.make("tag", height=12, width=12, num_agents=2)
    state, _ = env.init(KEY)
    state = state.replace(agent=state.agent.replace(energy=jnp.array([0.5, 0.0])))
    chosen = jnp.array([TURN_LEFT, FORWARD], jnp.int32)

    after, timestep = jax.jit(env.step)(KEY, state, chosen)

    assert timestep.action.tolist() == chosen.tolist()
    assert (after.agent.position[1] == state.agent.position[1]).all()


def test_shore_plants_fade_with_the_opacity_they_are_given():
    import numpy as np

    from boonta.environments.dune_sea import render as drawing

    base = np.full((16, 16, 3), 100.0, np.float32)
    mask = np.ones((1, 1), bool)
    name = drawing.REED_SPRITES[0]
    full = drawing.paint_sprite(base.copy(), name, mask, directory=drawing.FOREST_DIRECTORY)
    half = drawing.paint_sprite(base.copy(), name, mask, directory=drawing.FOREST_DIRECTORY, opacity=np.full((1, 1), 0.5))
    none = drawing.paint_sprite(base.copy(), name, mask, directory=drawing.FOREST_DIRECTORY, opacity=np.zeros((1, 1)))

    drawn = np.abs(full - base).sum(-1) > 1.0
    assert bool(drawn.any())
    assert bool((none == base).all())
    ratio = np.abs(half - base).sum(-1)[drawn] / np.abs(full - base).sum(-1)[drawn]
    assert float(np.abs(ratio - 0.5).max()) < 0.02



def test_every_environment_shares_one_observation_and_action_layout():
    envs = [dune_sea.make(name) for name in ("tag", "hide_and_seek")]

    def layout(env):
        space = env.observation_space()
        shapes = {name: leaf.shape[1:] for name, leaf in space.items()}
        return shapes, env.action_space().num_actions

    assert len({str(layout(env)) for env in envs}) == 1
