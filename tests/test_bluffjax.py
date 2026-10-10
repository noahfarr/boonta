import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("bluffjax")

from boonta import environments


def play(env, key, steps):
    state, timestep = env.init(key)
    timesteps = [timestep]
    for _ in range(steps):
        key, act_key, step_key = jax.random.split(key, 3)
        mask = env.action_mask(state)
        action = jax.random.categorical(act_key, jnp.where(mask, 0.0, -jnp.inf))
        state, timestep = env.step(step_key, state, action)
        timesteps.append(timestep)
        if bool(timestep.done.all()):
            break
    return state, timesteps


def test_only_the_seat_to_move_sees_its_cards_and_chooses():
    env = environments.make("bluffjax", "kuhn_poker")
    state, timestep = env.init(jax.random.key(0))
    mover = int(state.current_player_idx)
    waiting = 1 - mover
    assert timestep.obs["observation"][mover].any()
    assert not timestep.obs["observation"][waiting].any()
    assert timestep.obs["mask"][mover].all()
    assert timestep.obs["mask"][waiting].tolist() == [True, False]


def test_the_seat_to_move_decides_the_action():
    env = environments.make("bluffjax", "kuhn_poker")
    state, _ = env.init(jax.random.key(0))
    mover = int(state.current_player_idx)
    bet = jnp.zeros(2, jnp.int32).at[mover].set(1)
    ignored = jnp.ones(2, jnp.int32).at[mover].set(0)
    raised, _ = env.step(jax.random.key(1), state, bet)
    checked, _ = env.step(jax.random.key(1), state, ignored)
    assert int(raised.pot.sum()) == 3
    assert int(checked.pot.sum()) == 2


@pytest.mark.parametrize("seed", range(4))
def test_kuhn_poker_ends_zero_sum(seed):
    env = environments.make("bluffjax", "kuhn_poker")
    _, timesteps = play(env, jax.random.key(seed), env.time_limit())
    last = timesteps[-1]
    assert bool(last.terminated.all()) and not bool(last.truncated.any())
    total = sum(timestep.reward for timestep in timesteps)
    assert float(total.sum()) == 0.0


def test_a_game_cut_at_its_horizon_is_truncated():
    env = environments.make("bluffjax", "bluff", kwargs={"horizon": 4})
    _, timesteps = play(env, jax.random.key(0), 4)
    last = timesteps[-1]
    assert len(timesteps) == 5
    assert bool(last.truncated.all()) and not bool(last.terminated.any())


def test_simultaneous_games_take_every_seat_action():
    env = environments.make("bluffjax", "goofspiel")
    state, timestep = env.init(jax.random.key(0))
    assert timestep.obs["mask"].all()
    action = jnp.array([12, 0], jnp.int32)
    state, _ = env.step(jax.random.key(1), state, action)
    mask = env.action_mask(state)
    assert not bool(mask[0, 12]) and not bool(mask[1, 0])


def test_spaces_match_what_the_game_emits():
    for name in ("kuhn_poker", "goofspiel", "werewolf"):
        env = environments.make("bluffjax", name)
        _, timestep = env.init(jax.random.key(0))
        spaces = env.observation_space()
        assert timestep.obs["observation"].shape == spaces["observation"].shape
        assert timestep.obs["mask"].shape == spaces["mask"].shape
        assert timestep.action.shape == env.action_space().shape
