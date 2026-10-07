from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments import peanut_gb as gb

pokemon = gb.pokemon_red

ROM = gb.ROMS / "pokemon_red.gb"
needs_rom = pytest.mark.skipif(not ROM.exists(), reason=f"no Pokemon Red ROM at {ROM}")
LEADER, STARTER = pokemon.LEADER_LEVELS, pokemon.STARTER_LEVEL


def mon(level=STARTER, experience=0, health=1, full=1):
    return level, experience, health, full


def ram(party=(), size=None, badges=0, flags=(), owned=(), place="REDS_HOUSE_2F", at=(3, 6), menu=False):
    image = np.zeros(gb.RAM_SIZE, np.uint8)

    def put(address, value, width=1):
        start = address - gb.RAM_BASE
        image[start : start + width] = list(int(value).to_bytes(width, "big"))

    def raise_bits(address, bits):
        for bit in bits:
            image[address - gb.RAM_BASE + bit // 8] |= 1 << bit % 8

    put(pokemon.PARTY_SIZE, len(party) if size is None else size)
    for slot, (level, experience, health, full) in enumerate(party):
        put(pokemon.LEVELS[slot], level)
        put(pokemon.EXPERIENCE[slot], experience, 3)
        put(pokemon.HEALTH[slot], health, 2)
        put(pokemon.MAX_HEALTH[slot], full, 2)
    put(pokemon.BADGES, badges)
    put(pokemon.MAP, pokemon.NAMES.index(place))
    put(pokemon.X, at[0])
    put(pokemon.Y, at[1])
    if menu:
        put(pokemon.TILEMAP + 10, pokemon.BORDER_CORNER)
        put(pokemon.TILEMAP + 11, pokemon.BORDER_EDGE)
    raise_bits(pokemon.EVENTS[0], flags)
    raise_bits(pokemon.OWNED[0], owned)
    return jnp.asarray(image)


@pytest.fixture
def pay(monkeypatch):
    def play(rams, **weights):
        start, *later = rams
        script = iter(later)
        frame = np.zeros((4, 4), np.uint8)
        monkeypatch.setattr(
            gb, "advancer", lambda pool: lambda emulator, action: (emulator, frame, next(script))
        )
        env = gb.GameBoy(SimpleNamespace(close=lambda: None), (np.zeros(1, np.uint8), frame, start), pokemon, **weights)
        state, _ = env.init(jax.random.key(0))
        timesteps = []
        for _ in later:
            state, timestep = env.step(jax.random.key(0), state, jnp.int32(0))
            timesteps.append(timestep)
        return timesteps

    return play


@pytest.mark.parametrize(
    "image, expected",
    [
        pytest.param(ram(), 0, id="no party"),
        pytest.param(ram([mon(5), mon(30)]), 5 + LEADER[0] - STARTER, id="capped at the first leader"),
        pytest.param(ram([mon(5), mon(30)], badges=0b11), 5 + LEADER[2] - STARTER, id="a badge lifts the cap"),
        pytest.param(ram([mon(5), mon(5)], badges=0b11), 5, id="under the cap pays as it is"),
        pytest.param(ram([mon(10), mon(3)]), 10 + 3 - STARTER, id="a catch pays its level"),
        pytest.param(ram([mon(5), mon(30), mon(99)], size=2), 5 + LEADER[0] - STARTER, id="beyond the party"),
    ],
)
def test_leveling_counts_party_levels_up_to_the_next_leaders_cap(image, expected):
    assert int(pokemon.leveling(image)) == expected


@pytest.mark.parametrize(
    "image, expected",
    [
        pytest.param(ram([mon(experience=1000), mon(experience=300)]), 1300, id="the whole party"),
        pytest.param(ram([mon(experience=1000), mon(experience=300)], size=1), 1000, id="beyond the party"),
        pytest.param(ram([mon(experience=0xFFFFFF)]), LEADER[0] ** 3, id="an impossible read clamps"),
        pytest.param(ram([mon(experience=2959)]), LEADER[0] ** 3, id="clamped without a badge"),
        pytest.param(ram([mon(experience=2959)], badges=0b1), 2959, id="a badge lifts the ceiling"),
    ],
)
def test_experience_counts_each_member_up_to_the_cube_of_the_next_leaders_level(image, expected):
    assert float(pokemon.earned(image)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "party, vitality, fainted",
    [
        pytest.param([], 0.0, False, id="no party"),
        pytest.param([mon(health=10, full=20), mon(health=5, full=20)], 15 / 40, False, id="hurt"),
        pytest.param([mon(health=10, full=20), mon(health=0, full=20)], 10 / 40, False, id="one down"),
        pytest.param([mon(health=0, full=20), mon(health=0, full=20)], 0.0, True, id="all down"),
    ],
)
def test_vitality_is_the_party_health_fraction_and_fainting_needs_every_member_down(party, vitality, fainted):
    assert float(pokemon.vitality(ram(party))) == pytest.approx(vitality)
    assert bool(pokemon.fainted(ram(party))) == fainted


@pytest.mark.parametrize(
    "party, owned, expected",
    [
        pytest.param([], (0, 2, 28), 0, id="no party"),
        pytest.param([mon()], (0, 2, 28), 3, id="a party"),
        pytest.param([mon()], (0, 2, 28, *range(144, 152)), 11, id="the last byte"),
    ],
)
def test_caught_counts_owned_species_only_once_the_party_exists(party, owned, expected):
    assert int(pokemon.caught(ram(party, owned=owned))) == expected


def test_a_place_is_the_loaded_map_alone():
    names = ("REDS_HOUSE_2F", "REDS_HOUSE_1F", "ROUTE_1")
    places = [int(pokemon.place(ram(place=name))) for name in names]
    assert len(set(places)) == len(names)
    assert all(0 <= place < pokemon.NUM_MAPS for place in places)
    assert int(pokemon.place(ram(place=names[0], badges=0b11, flags=(1, 2)))) == places[0]


@pytest.mark.parametrize(
    "place, size",
    [("PALLET_TOWN", 1.0), ("ROUTE_1", 2.0), ("VIRIDIAN_CITY", 4.0), ("REDS_HOUSE_2F", 64 / 360)],
)
def test_a_map_is_sized_against_pallet_town(place, size):
    assert float(pokemon.extent(ram(place=place))) == pytest.approx(size)


@pytest.mark.parametrize(
    "fresh, gained, accepted",
    [
        pytest.param(jnp.zeros(78, jnp.uint32).at[3].set(0b101), 2, True, id="a normal gain"),
        pytest.param(jnp.zeros(78, jnp.uint32).at[0].set((1 << gb.BURST) - 1), gb.BURST, True, id="the largest burst"),
        pytest.param(jnp.zeros(78, jnp.uint32).at[0].set((1 << (gb.BURST + 1)) - 1), 0, False, id="one bit too many"),
        pytest.param(jnp.full(78, 0xFFFFFFFF, jnp.uint32), 0, False, id="a corrupt read"),
    ],
)
def test_sifting_flags_pays_a_normal_gain_and_refuses_a_corrupt_burst(fresh, gained, accepted):
    held = jnp.zeros(78, jnp.uint32)
    kept, paid = gb.sift(held, fresh)
    assert float(paid) == gained
    np.testing.assert_array_equal(kept, fresh if accepted else held)


@pytest.mark.parametrize(
    "weights, rams, paid",
    [
        pytest.param(
            {"flag_reward": 1.0},
            [ram(), ram(flags=(0, 1)), ram(flags=(0, 1, 2)), ram(flags=(0,)), ram(flags=(0, 1))],
            [2.0, 1.0, 0.0, 0.0],
            id="flags",
        ),
        pytest.param(
            {"flag_reward": 1.0},
            [ram(), ram(flags=range(gb.BURST + 1)), ram(flags=(0,))],
            [0.0, 1.0],
            id="a corrupt burst of flags",
        ),
        pytest.param(
            {"level_reward": 1.0},
            [ram([mon(5)]), ram([mon(7)]), ram([mon(6)]), ram([mon(8)])],
            [2.0, 0.0, 1.0],
            id="levels",
        ),
        pytest.param(
            {"experience_reward": 0.01},
            [ram([mon(experience=100)]), ram([mon(experience=600)]), ram([mon(experience=400)]), ram([mon(experience=2000)])],
            [5.0, 0.0, 14.0],
            id="experience",
        ),
        pytest.param(
            {"catch_reward": 1.0},
            [ram([mon()], owned=(0,)), ram([mon()], owned=(0, 1, 2)), ram([mon()], owned=(0,))],
            [2.0, 0.0],
            id="catches",
        ),
    ],
)
def test_progress_is_paid_once_and_a_loss_is_never_refunded(pay, weights, rams, paid):
    rewards = [float(timestep.reward) for timestep in pay(rams, **weights)]
    np.testing.assert_allclose(rewards, paid, rtol=1e-5)


@pytest.mark.parametrize(
    "weights, rams, paid",
    [
        pytest.param(
            {"map_reward": 1.0},
            [ram(place="REDS_HOUSE_2F"), ram(place="ROUTE_1"), ram(place="REDS_HOUSE_2F"), ram(place="ROUTE_1")],
            [2.0, 0.0, 0.0],
            id="a place",
        ),
        pytest.param(
            {"map_reward": 1.0},
            [ram(place="REDS_HOUSE_2F"), ram(place="ROUTE_1"), ram(place="ROUTE_1", flags=(0,)), ram(place="REDS_HOUSE_2F")],
            [2.0, 0.0, 64 / 360],
            id="a place after a new flag",
        ),
        pytest.param(
            {"tile_reward": 1.0},
            [ram(at=(3, 6)), ram(at=(3, 7)), ram(at=(3, 6)), ram(at=(3, 7))],
            [1.0, 0.0, 0.0],
            id="a tile",
        ),
    ],
)
def test_a_place_or_tile_pays_its_first_visit_and_a_new_flag_starts_the_count_over(pay, weights, rams, paid):
    rewards = [float(timestep.reward) for timestep in pay(rams, **weights)]
    np.testing.assert_allclose(rewards, paid, rtol=1e-5)


@pytest.mark.parametrize(
    "rams, paid",
    [
        pytest.param(
            [ram([mon(health=10, full=20)]), ram([mon(health=0, full=20)]), ram([mon(health=0, full=20)])],
            [-1.0, 0.0],
            id="a blackout costs once",
        ),
        pytest.param([ram(), ram(menu=True), ram(menu=True), ram()], [-0.5, -0.5, 0.0], id="the menu costs each step it is open"),
        pytest.param(
            [
                ram([mon(health=10, full=20)]),
                ram([mon(health=0, full=20)]),
                ram([mon(health=20, full=20)]),
                ram([mon(health=10, full=20)]),
                ram([mon(health=20, full=20)]),
            ],
            [-1.0, 0.0, 0.0, 0.5],
            id="a revive after a blackout pays no heal",
        ),
        pytest.param(
            [ram(), ram([mon(health=20, full=20)]), ram([mon(health=12, full=20)]), ram([mon(health=20, full=20)])],
            [0.0, 0.0, 0.4],
            id="the starter is neither a heal nor a death",
        ),
        pytest.param(
            [ram([mon(health=5, full=10)]), ram([mon(health=5, full=10), mon(health=10, full=10)])],
            [0.0],
            id="a catch pays no heal",
        ),
    ],
)
def test_penalties_and_heals_follow_the_party_without_ending_the_episode(pay, rams, paid):
    timesteps = pay(rams, faint_penalty=1.0, heal_reward=1.0, menu_penalty=0.5)
    np.testing.assert_allclose([float(timestep.reward) for timestep in timesteps], paid, rtol=1e-5)
    assert not any(bool(timestep.terminated or timestep.truncated) for timestep in timesteps)


@pytest.fixture(scope="module")
def env():
    held = gb.make("pokemon_red", num_envs=1, num_threads=1, menu_penalty=0.01, tile_reward=1.0)
    yield held
    held.pool.close()


@needs_rom
def test_the_opening_state_is_the_players_room_without_events(env):
    state, timestep = env.init(jax.random.key(0))
    assert int(pokemon.map_id(state.ram)) == pokemon.NAMES.index("REDS_HOUSE_2F")
    assert set(timestep.info) == set(pokemon.KEYS)
    assert all(bool(jnp.isfinite(value)) for value in timestep.info.values())
    assert float(timestep.info["events"]) == 0.0
    assert float(timestep.info["maps"]) == 1.0 and float(timestep.info["tiles"]) == 1.0


@needs_rom
def test_a_held_state_continues_the_same_trajectory(env):
    key = jax.random.key(1)
    state, _ = env.init(key)
    held, actions = [state], jax.random.randint(key, (6,), 0, env.pool.num_actions)
    for action in actions:
        state, _ = env.step(key, state, action)
        held.append(state)
    again, _ = env.step(key, held[3], actions[3])
    np.testing.assert_array_equal(again.emulator, held[4].emulator)
    np.testing.assert_array_equal(again.ram, held[4].ram)


@needs_rom
def test_random_play_raises_no_emulator_errors(env):
    key = jax.random.key(4)
    state, _ = env.init(key)
    assert env.pool.state_size < 65536
    for _ in range(200):
        key, k = jax.random.split(key)
        state, _ = env.step(k, state, jax.random.randint(k, (), 0, env.pool.num_actions))
    assert env.pool.errors() == 0


@needs_rom
def test_the_start_button_opens_a_menu_the_penalty_sees_and_b_closes_it(env):
    key = jax.random.key(0)
    state, _ = env.init(key)
    assert int(pokemon.menu(state.ram)) == 0
    opened, timestep = env.step(key, state, jnp.int32(gb.ACTIONS.index("start")))
    assert int(pokemon.menu(opened.ram)) == 1
    assert float(timestep.reward) == pytest.approx(-0.01)
    closed, timestep = env.step(key, opened, jnp.int32(gb.ACTIONS.index("b")))
    assert int(pokemon.menu(closed.ram)) == 0
    assert float(timestep.reward) == 0.0


@needs_rom
def test_walking_one_tile_down_and_back_pays_the_new_tile_once(env):
    key = jax.random.key(0)
    state, _ = env.init(key)
    moved, first = env.step(key, state, jnp.int32(gb.ACTIONS.index("down")))
    back, second = env.step(key, moved, jnp.int32(gb.ACTIONS.index("up")))
    assert float(first.reward) == 1.0 and float(second.reward) == 0.0
    assert int(pokemon.walk(moved.ram)) != int(pokemon.walk(state.ram)) == int(pokemon.walk(back.ram))
