import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments import dune_sea
from boonta.environments.dune_sea import Chain
from boonta.environments.dune_sea.chain import cell


def play(env, state, policy):
    timesteps = []
    while not timesteps or not bool(timesteps[-1].terminated):
        action = jnp.int32(policy(int(state.position)))
        state, timestep = env.step(jax.random.key(len(timesteps)), state, action)
        timesteps.append(timestep)
    return timesteps


@pytest.mark.parametrize("length", [1, 4, 10])
@pytest.mark.parametrize("corridor_action", [0, 1])
def test_only_the_action_at_the_end_of_the_chain_matters(length, corridor_action):
    env = Chain(length=length)
    state, _ = env.init(jax.random.key(0))
    for answer, payout in ((env.secret, 1.0), (env.secret - 1, 0.0)):
        timesteps = play(env, state, lambda position: answer if position == length else corridor_action)
        assert len(timesteps) == length + 1 == env.time_limit()
        assert [float(timestep.reward) for timestep in timesteps] == [0.0] * length + [payout]
        assert not any(bool(timestep.truncated) for timestep in timesteps)


def test_only_the_secret_action_pays():
    env = Chain(length=3, num_actions=5)
    state, _ = env.init(jax.random.key(0))
    payouts = [float(play(env, state, lambda position: action)[-1].reward) for action in range(5)]
    assert payouts == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_a_restart_at_the_end_of_the_chain_decides_in_one_step():
    env = dune_sea.make("chain", length=10)
    state, _ = env.init(jax.random.key(0))
    assert int(cell(env)(state)) == env.start == 0
    for position in range(env.num_cells):
        placed = env.place(state, position)
        assert int(cell(env)(placed)) == position
        np.testing.assert_array_equal(env.observe(placed), jax.nn.one_hot(position, env.num_cells))
        timesteps = play(env, placed, lambda position: env.secret)
        assert len(timesteps) == env.num_cells - position
        assert float(timesteps[-1].reward) == 1.0


def test_the_tables_describe_the_game_the_agent_plays():
    env = Chain(length=6, num_actions=3)
    policy = jax.nn.softmax(jax.random.normal(jax.random.key(0), (env.num_cells, 3)))
    state, _ = env.init(jax.random.key(0))
    answered = policy[env.length, env.secret]
    assert float(env.expected_return(policy)) == float(answered)

    advantages = np.asarray(env.advantages(policy))
    np.testing.assert_array_equal(advantages[: env.length], 0.0)
    np.testing.assert_allclose(advantages[env.length], np.eye(3)[env.secret] - float(answered), rtol=1e-6)
    np.testing.assert_allclose(np.sum(policy * advantages, axis=-1), 0.0, atol=1e-6)

    transitions = np.asarray(env.transitions(policy))
    for position in range(env.num_cells):
        after, timestep = env.step(jax.random.key(0), env.place(state, position), jnp.int32(0))
        landing = env.start if bool(timestep.terminated) else int(after.position)
        np.testing.assert_array_equal(transitions[position], np.eye(env.num_cells)[landing])


def test_the_chain_renders_the_agent_red_and_the_deciding_cell_green():
    env = Chain(length=6)
    state, _ = env.init(jax.random.key(0))
    tile = 8
    image = np.asarray(env.render(state, tile=tile))
    num_rows, num_columns = np.asarray(env.cell_grid).shape
    assert (num_rows, num_columns) == (1, env.num_cells)
    assert image.shape == (num_rows * tile, num_columns * tile, 3)
    assert image.dtype == np.uint8

    def centre(position, across=0.5):
        return image[tile // 2, int((position + across) * tile)]

    np.testing.assert_array_equal(centre(env.start, across=0.4), [255, 0, 0])
    np.testing.assert_array_equal(centre(env.length), [0, 255, 0])
    for position in range(1, env.length):
        np.testing.assert_array_equal(centre(position), [100, 100, 100])
