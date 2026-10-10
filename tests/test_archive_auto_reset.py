import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.environments.wrappers import TimeLimit
from boonta.environments.wrappers.archive_auto_reset import (
    Archive,
    ArchiveAutoReset,
    Selector,
    lru,
    share,
)
from boonta.utils import Timestep

LENGTH = 4
NUM_ENVS = 8
ARCHIVE_SIZE = 16
KEY = jax.random.key(0)


@struct.dataclass
class Trail:
    depth: jnp.ndarray
    tag: jnp.ndarray


class Ladder(Environment):
    def observe(self, state, key=None):
        return jnp.stack([state.depth, state.tag]).astype(jnp.float32)

    def frame(self, state, action):
        return Timestep(
            obs=self.observe(state),
            action=action,
            reward=state.depth.astype(jnp.float32),
            terminated=state.depth >= LENGTH,
            truncated=jnp.bool_(False),
            info={},
        )

    def init(self, key):
        state = Trail(jnp.int32(0), jnp.int32(0))
        return state, self.frame(state, jnp.int32(0))

    def step(self, key, state, action):
        action = jnp.asarray(action, jnp.int32)
        state = Trail(state.depth + 1, state.tag + action)
        return state, self.frame(state, action)

    def observation_space(self):
        return Space(shape=(2), dtype=jnp.float32, low=0.0, high=float(LENGTH))

    def action_space(self):
        return Space(shape=(), dtype=jnp.int32, low=0, high=1)

    def horizon(self):
        return LENGTH


def cell_of(state):
    return jnp.stack([state.tag], axis=-1)


def factory(fn):
    return lambda env: fn


class Echo(Selector):
    def select(self, state, wrapper, archive_auto_reset_state, key):
        return state, jnp.full(4, archive_auto_reset_state.current_slots[0], jnp.int32)


def archive(archive_size=ARCHIVE_SIZE, cell_fn=cell_of, **kwargs):
    return Archive(archive_size=archive_size, cell_fn=cell_fn, **kwargs)


def batch(tags):
    tags = jnp.asarray(tags, jnp.int32)
    return Trail(depth=jnp.zeros_like(tags), tag=tags)


def build(inner=None, cell_fn=cell_of, **kwargs):
    kwargs.setdefault("capacity", ARCHIVE_SIZE)
    return ArchiveAutoReset(
        Ladder() if inner is None else inner,
        num_envs=NUM_ENVS,
        cell_fn=factory(cell_fn),
        selection=kwargs.pop("selection", Echo()),
        **kwargs,
    )


def drive(env, steps=20, seed=0, action=0):
    key = jax.random.key(seed)
    state, timestep = env.init(key)
    for _ in range(steps):
        key, sub = jax.random.split(key)
        state, timestep = env.step(sub, state, jnp.full(NUM_ENVS, action, jnp.int32))
    return state, timestep


def test_an_empty_archive_holds_no_cells_and_zeroed_slots():
    state = archive().init(batch(jnp.zeros(NUM_ENVS)))
    assert state.keys.shape == (ARCHIVE_SIZE, 1)
    assert state.mask.shape == (ARCHIVE_SIZE,) and state.mask.dtype == jnp.bool
    assert state.cell_states.snapshot.tag.shape == (ARCHIVE_SIZE,)
    assert int(state.mask.sum()) == 0


def test_the_stored_shape_drops_the_env_axis_and_takes_the_archive_axis():
    state = archive().init(batch(jnp.zeros(NUM_ENVS)))
    assert state.cell_states.snapshot.depth.shape == (ARCHIVE_SIZE,)
    assert state.cell_states.snapshot.tag.shape == (ARCHIVE_SIZE,)


def test_adding_a_state_stores_it_at_the_slot_it_claimed():
    held = archive()
    state = held.init(batch(jnp.zeros(NUM_ENVS)))
    arriving = batch(jnp.arange(NUM_ENVS))
    state, index, _, _ = held.add(state, arriving, jnp.ones(NUM_ENVS, bool))
    assert bool(jnp.all(index >= 0))
    assert int(state.mask.sum()) == NUM_ENVS
    assert bool(jnp.all(state.cell_states.snapshot.tag[index] == arriving.tag))


def test_a_cell_seen_twice_lands_in_the_same_slot():
    held = archive()
    state = held.init(batch(jnp.zeros(NUM_ENVS)))
    arriving = batch(jnp.arange(NUM_ENVS))
    state, first, _, _ = held.add(state, arriving, jnp.ones(NUM_ENVS, bool))
    state, second, _, _ = held.add(state, arriving, jnp.ones(NUM_ENVS, bool))
    assert bool(jnp.all(first == second))
    assert int(state.mask.sum()) == NUM_ENVS


def test_revisiting_a_cell_overwrites_the_state_it_holds():
    held = archive()
    state = held.init(batch(jnp.zeros(NUM_ENVS)))
    live = jnp.ones(NUM_ENVS, bool)
    state, index, _, _ = held.add(state, Trail(jnp.full(NUM_ENVS, 1), jnp.arange(NUM_ENVS)), live)
    state, again, _, _ = held.add(state, Trail(jnp.full(NUM_ENVS, 9), jnp.arange(NUM_ENVS)), live)
    assert bool(jnp.all(index == again))
    assert bool(jnp.all(state.cell_states.snapshot.depth[index] == 9))


def test_one_writer_per_cell_and_the_eligible_one_wins():
    held = archive()
    state = held.init(batch(jnp.zeros(NUM_ENVS)))
    live = jnp.ones(NUM_ENVS, bool)
    crowd = Trail(jnp.arange(NUM_ENVS) + 1, jnp.zeros(NUM_ENVS, jnp.int32))
    eligible = jnp.arange(NUM_ENVS) == 3
    state, index, _, _ = held.add(state, crowd, live, live, eligible)
    assert int(state.mask.sum()) == 1
    assert bool(jnp.all(index == index[0]))
    slot = int(index[0])
    assert bool(state.cell_states.eligible[slot])
    assert int(state.cell_states.snapshot.depth[slot]) == 4
    rho = jnp.arange(NUM_ENVS) == 5
    state, index, _, _ = held.add(state, crowd, live, rho, jnp.zeros(NUM_ENVS, bool))
    assert bool(state.cell_states.eligible[slot])
    assert int(state.cell_states.snapshot.depth[slot]) == 4


def test_a_warm_visit_cannot_overwrite_a_cell_a_cold_one_reached():
    held = archive()
    state = held.init(batch(jnp.zeros(NUM_ENVS)))
    live = jnp.ones(NUM_ENVS, bool)
    cold = jnp.ones(NUM_ENVS, bool)
    state, index, _, _ = held.add(state, Trail(jnp.full(NUM_ENVS, 1), jnp.arange(NUM_ENVS)), live, cold)
    state, again, _, _ = held.add(state, Trail(jnp.full(NUM_ENVS, 9), jnp.arange(NUM_ENVS)), live, ~cold)
    assert bool(jnp.all(index == again))
    assert bool(jnp.all(state.cell_states.snapshot.depth[index] == 1))
    assert bool(jnp.all(state.cell_states.rho[index]))


def test_a_warm_visit_fills_a_cell_no_cold_one_has_reached_until_one_does():
    held = archive()
    state = held.init(batch(jnp.zeros(NUM_ENVS)))
    live = jnp.ones(NUM_ENVS, bool)
    warm = jnp.zeros(NUM_ENVS, bool)
    state, index, _, _ = held.add(state, Trail(jnp.full(NUM_ENVS, 3), jnp.arange(NUM_ENVS)), live, warm)
    assert bool(jnp.all(state.cell_states.snapshot.depth[index] == 3))
    assert not bool(jnp.any(state.cell_states.rho[index]))
    state, _, _, _ = held.add(state, Trail(jnp.full(NUM_ENVS, 7), jnp.arange(NUM_ENVS)), live, ~warm)
    assert bool(jnp.all(state.cell_states.snapshot.depth[index] == 7))
    assert bool(jnp.all(state.cell_states.rho[index]))


def test_a_state_that_may_not_be_added_claims_no_slot():
    held = archive()
    state = held.init(batch(jnp.zeros(NUM_ENVS)))
    state, index, _, _ = held.add(state, batch(jnp.arange(NUM_ENVS)), jnp.zeros(NUM_ENVS, bool))
    assert bool(jnp.all(index == -1))
    assert int(state.mask.sum()) == 0
    assert float(jnp.max(jnp.abs(state.cell_states.snapshot.tag))) == 0.0


def test_a_full_probe_window_refuses_the_cell_while_the_archive_has_room():
    held = archive(archive_size=64)
    state = held.init(batch(jnp.zeros(1)))
    for step in range(8):
        state, index, _, _ = held.add(
            state, batch(jnp.asarray([step * 64])), jnp.ones(1, bool)
        )
        assert int(index[0]) >= 0
    room = int(jnp.sum(~state.mask))
    _, refused, _, _ = held.add(state, batch(jnp.asarray([8 * 64])), jnp.ones(1, bool))
    assert room == 64 - 8
    assert int(refused[0]) == -1


def test_a_crowded_window_evicts_the_least_recently_visited_cell():
    held = archive(archive_size=64, eviction_fn=lru)
    state = held.init(batch(jnp.zeros(1)))
    for step in range(8):
        state, index, _, _ = held.add(
            state, batch(jnp.asarray([step * 64])), jnp.ones(1, bool)
        )
    state, again, _, _ = held.add(
        state, batch(jnp.asarray([1 * 64])), jnp.ones(1, bool)
    )
    state, index, _, evicted = held.add(
        state, batch(jnp.asarray([8 * 64])), jnp.ones(1, bool)
    )
    assert int(index[0]) >= 0
    assert int(evicted[0]) == int(index[0])
    assert int(evicted[0]) != int(again[0])
    assert int(state.keys[index[0], 0]) == 8 * 64
    assert int(state.mask.sum()) == 8


def test_eviction_stays_off_unless_it_is_asked_for():
    held = archive(archive_size=64)
    state = held.init(batch(jnp.zeros(1)))
    for step in range(8):
        state, _, _, _ = held.add(
            state, batch(jnp.asarray([step * 64])), jnp.ones(1, bool)
        )
    _, refused, _, evicted = held.add(
        state, batch(jnp.asarray([8 * 64])), jnp.ones(1, bool)
    )
    assert int(refused[0]) == -1
    assert int(jnp.max(evicted)) == -1


def test_cells_competing_for_one_home_slot_leave_a_single_winner():
    held = archive(archive_size=64)
    state = held.init(batch(jnp.zeros(8)))
    state, index, _, _ = held.add(state, batch(jnp.arange(8) * 64), jnp.ones(8, bool))
    assert int(jnp.sum(index >= 0)) == 1
    assert int(state.mask.sum()) == 1
    winner = int(jnp.max(index))
    assert int(state.keys[winner, 0]) == int(jnp.argmax(index >= 0)) * 64


def test_the_archive_hands_back_the_state_at_a_slot():
    held = archive()
    state = held.init(batch(jnp.zeros(NUM_ENVS)))
    state, index, _, _ = held.add(state, batch(jnp.arange(NUM_ENVS)), jnp.ones(NUM_ENVS, bool))
    got = held.take(state, index)
    assert bool(jnp.all(got.tag == jnp.arange(NUM_ENVS)))


def test_share_rounds_the_fraction_the_block_holds():
    assert share(2.0, NUM_ENVS) == 4
    assert share(4.0, NUM_ENVS) == 6
    assert share(1.0, NUM_ENVS) == 0
    assert share(float("inf"), NUM_ENVS) == NUM_ENVS


def test_a_running_episode_keeps_the_state_it_stepped_into():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    before = state.env_state.depth
    state, timestep = env.step(key, state, jnp.zeros(NUM_ENVS, jnp.int32))
    running = ~timestep.done
    assert bool(jnp.any(running))
    assert bool(jnp.all(state.env_state.depth[running] == (before + 1)[running]))


def test_the_archive_never_shrinks_as_it_fills():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    held = []
    for _ in range(30):
        key, sub = jax.random.split(key)
        state, _ = env.step(sub, state, jax.random.randint(sub, (NUM_ENVS,), 0, 2))
        held.append(int(state.archive_state.mask.sum()))
    assert held == sorted(held)
    assert held[-1] > held[0]


def test_every_slot_holds_the_state_its_cell_says_it_holds():
    state, _ = drive(build(), steps=30)
    archived = state.archive_state
    stored = cell_of(archived.cell_states.snapshot)
    assert bool(jnp.all(jnp.where(archived.mask[:, None], stored == archived.keys, True)))


def test_a_banked_state_carries_no_leftover_clock():
    env = build(inner=TimeLimit(Ladder(), 6), cell_fn=lambda s: cell_of(s.env_state))
    state, _ = drive(env, steps=24)
    assert int(jnp.max(state.archive_state.cell_states.snapshot.time)) == 0


def test_the_clock_would_be_inherited_without_the_reset():
    env = build(inner=TimeLimit(Ladder(), 6), cell_fn=lambda s: cell_of(s.env_state))
    key = jax.random.key(0)
    state, _ = env.init(key)
    for _ in range(3):
        key, sub = jax.random.split(key)
        state, _ = env.step(sub, state, jnp.zeros(NUM_ENVS, jnp.int32))
    assert int(jnp.max(state.env_state.time)) <= 6


def test_the_archive_holds_a_state_before_the_first_step():
    env = build()
    state, _ = env.init(jax.random.key(0))
    assert int(state.archive_state.mask.sum()) > 0


def test_a_restart_never_lands_on_an_empty_slot():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    for _ in range(20):
        key, sub = jax.random.split(key)
        state, _ = env.step(sub, state, jnp.zeros(NUM_ENVS, jnp.int32))
        archived = state.archive_state
        assert bool(jnp.all(jnp.isin(state.env_state.tag, archived.keys[archived.mask, 0])))


def test_a_finished_episode_spawns_a_fresh_world():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    for _ in range(10):
        key, sub = jax.random.split(key)
        state, timestep = env.step(sub, state, jnp.ones(NUM_ENVS, jnp.int32))
        assert bool(jnp.all(state.env_state.depth[timestep.done] == 0))


def test_occupied_counts_envs_outside_the_block_by_their_current_slot():
    env = build()
    state, _ = env.init(jax.random.key(0))
    current_slots = jnp.full(NUM_ENVS, -1, jnp.int32).at[1].set(3).at[2].set(3).at[3].set(5)
    state = state.replace(current_slots=current_slots)
    counted = env.occupied(state, placed=4)
    assert float(counted[3]) == 2.0
    assert float(counted[5]) == 1.0
    assert float(jnp.sum(counted)) == 3.0



def test_the_step_survives_jit():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    stepped = jax.jit(env.step)(key, state, jnp.zeros(NUM_ENVS, jnp.int32))
    assert stepped[1].done.shape == (NUM_ENVS,)


def test_the_clock_restarts_with_the_episode():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    for _ in range(20):
        key, sub = jax.random.split(key)
        state, timestep = env.step(sub, state, jnp.zeros(NUM_ENVS, jnp.int32))
        assert int(jnp.max(state.age)) <= LENGTH
        assert bool(jnp.all(state.age[timestep.done] == 0))


def test_every_step_reports_the_cell_the_env_is_in():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    for _ in range(12):
        key, sub = jax.random.split(key)
        state, timestep = env.step(sub, state, jnp.zeros(NUM_ENVS, jnp.int32))
        cell = timestep.info["cell"]
        assert bool(jnp.all(cell >= 0))
        assert bool(jnp.all(state.archive_state.mask[cell]))
        assert bool(jnp.all(cell == state.current_slots))


def test_without_an_update_every_episode_is_reported_as_rho():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    for _ in range(8):
        key, sub = jax.random.split(key)
        state, timestep = env.step(sub, state, jnp.ones(NUM_ENVS, jnp.int32))
        assert bool(jnp.all(timestep.info["rho"]))


def test_an_update_with_a_key_reports_the_archive():
    env = build()
    state, _ = drive(env, steps=4)
    _, logs = lox.spool(lambda state: env.place(state, jax.random.key(3), None))(state)
    assert {"archive/placed", "archive/num_cells", "archive/num_eligible"} <= set(logs)


def test_a_step_reports_the_archive_miss_and_discovery_rate():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    _, logs = lox.spool(env.step)(key, state, jnp.zeros(NUM_ENVS, jnp.int32))
    assert {
        "archive/miss",
        "archive/discovery_rate",
        "archive/rho",
        "archive/num_evictions",
    } <= set(logs)
    assert "rho/episode_return" in logs


def test_the_episode_return_is_only_reported_when_an_episode_ends():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    seen = []
    for _ in range(LENGTH):
        key, sub = jax.random.split(key)
        (state, _), logs = lox.spool(env.step)(
            sub, state, jnp.zeros(NUM_ENVS, jnp.int32)
        )
        seen.append(jnp.asarray(logs["mu/episode_return"]))
    assert bool(jnp.all(jnp.isnan(seen[0])))
    assert bool(jnp.all(~jnp.isnan(seen[-1])))


def test_without_preempt_every_return_is_a_cold_one():
    env = build()
    key = jax.random.key(0)
    state, _ = env.init(key)
    for _ in range(LENGTH):
        key, sub = jax.random.split(key)
        (state, _), logs = lox.spool(env.step)(
            sub, state, jnp.zeros(NUM_ENVS, jnp.int32)
        )
    cold = jnp.asarray(logs["rho/episode_return"])
    assert bool(jnp.all(~jnp.isnan(cold)))
    assert bool(jnp.array_equal(cold, jnp.asarray(logs["mu/episode_return"])))


def test_a_state_the_eligibility_rejects_is_stored_but_not_marked_eligible():
    env = build(eligibility_fn=factory(lambda state: state.tag < 2))
    state, timestep = drive(env, steps=3, action=1)
    held = state.archive_state
    tags = held.keys[held.mask][:, 0]
    assert bool(jnp.any(tags >= 2))
    eligible = env.archive.eligible(held)
    assert bool(jnp.all(held.keys[eligible][:, 0] < 2))


def test_an_updated_block_ends_at_the_next_step_and_restarts_from_its_cell():
    env = build()
    state, _ = drive(env, steps=2, action=1)
    cell = int(state.current_slots[0])
    state = env.place(state, jax.random.key(3), None)
    block = jnp.arange(NUM_ENVS) >= NUM_ENVS - 4
    assert bool(jnp.all(state.due_mask == block))
    assert bool(jnp.all(state.assigned_slots[block] == cell))
    assert bool(jnp.all(state.assigned_slots[~block] == -1))
    after, cut = env.step(jax.random.key(4), state, jnp.ones(NUM_ENVS, jnp.int32))
    assert bool(jnp.all(cut.truncated[block])) and not bool(jnp.any(cut.truncated[~block]))
    assert bool(jnp.all(~after.rho[block])) and bool(jnp.all(after.current_slots[block] == cell))
    assert bool(jnp.all(after.age[block] == 0)) and not bool(jnp.any(after.due_mask))
    stored = after.archive_state.cell_states.snapshot.tag[after.current_slots[block]]
    assert bool(jnp.all(after.env_state.tag[block] == stored))
    assert bool(jnp.all(after.env_state.tag[~block] == state.env_state.tag[~block] + 1))
    assert bool(jnp.all(cut.obs[block, 1] == stored))
    assert bool(jnp.all(cut.info["start"][block] == False))  # noqa: E712
