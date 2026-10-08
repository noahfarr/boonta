from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.artisans import Gauntlet
from boonta.environments.connectx import ConnectX, ConnectXState, Negamax, uniform
from boonta.environments.wrappers import Opponent, Vectorize


def lines(board, piece, inarow):
    rows, columns = board.shape
    for row in range(rows):
        for column in range(columns):
            for down, across in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [
                    (row + distance * down, column + distance * across)
                    for distance in range(inarow)
                ]
                if all(0 <= r < rows and 0 <= c < columns and board[r, c] == piece for r, c in cells):
                    return True
    return False


def reference(board, mover, column, inarow):
    board = board.copy()
    sign = np.where(np.arange(2) == mover, 1.0, -1.0)
    if board[0, column] != 0:
        return board, -sign, True, "illegal"
    row = max(r for r in range(board.shape[0]) if board[r, column] == 0)
    board[row, column] = mover + 1
    if lines(board, mover + 1, inarow):
        return board, sign, True, "win"
    full = bool(np.all(board != 0))
    return board, np.zeros(2), full, "draw" if full else "playing"


@pytest.mark.parametrize("rows, columns, inarow", [(6, 7, 4), (5, 5, 3)])
def test_random_games_follow_the_reference_rules(rows, columns, inarow):
    environment = ConnectX(rows, columns, inarow)
    step = jax.jit(environment.step)
    generator = np.random.default_rng(0)
    endings = []

    for game in range(60):
        state, timestep = environment.init(jax.random.key(game))
        board, done = np.zeros((rows, columns), np.int8), False
        while not done:
            mover = int(state.mover)
            column = int(generator.integers(columns))
            action = jnp.array([column, columns - 1 - column], jnp.int32)
            if mover == 1:
                action = action[::-1]
            board, reward, done, outcome = reference(board, mover, column, inarow)
            state, timestep = step(jax.random.key(0), state, action)

            np.testing.assert_array_equal(state.board, board)
            np.testing.assert_array_equal(timestep.reward, reward)
            np.testing.assert_array_equal(timestep.terminated, [done, done])
        endings.append(outcome)

    assert endings.count("win") > 10


def test_an_illegal_move_loses_for_the_mover():
    environment = ConnectX()
    state = ConnectXState(board=jnp.zeros((6, 7), jnp.int8).at[:, 3].set(jnp.array([1, 2, 1, 2, 1, 2], jnp.int8)), mover=jnp.int32(1))
    state, timestep = environment.step(jax.random.key(0), state, jnp.array([0, 3]))
    np.testing.assert_array_equal(timestep.reward, [1.0, -1.0])
    assert bool(timestep.terminated.all())


def test_a_full_board_without_a_line_is_a_draw():
    pattern = np.array(
        [
            [1, 1, 2, 2, 1, 1, 0],
            [2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1],
        ],
        np.int8,
    )
    filled = pattern.copy()
    filled[0, 6] = 2
    assert not lines(filled, 1, 4) and not lines(filled, 2, 4)

    environment = ConnectX()
    state = ConnectXState(board=jnp.asarray(pattern), mover=jnp.int32(1))
    state, timestep = environment.step(jax.random.key(0), state, jnp.array([0, 6]))
    np.testing.assert_array_equal(timestep.reward, [0.0, 0.0])
    assert bool(timestep.terminated.all())


def test_either_seat_can_move_first_and_each_sees_its_own_pieces_first():
    environment = ConnectX()
    movers = {int(environment.init(jax.random.key(seed))[0].mover) for seed in range(32)}
    assert movers == {0, 1}

    state, timestep = environment.init(jax.random.key(0))
    np.testing.assert_array_equal(timestep.terminated, [True, True])
    mover = int(state.mover)
    state, timestep = environment.step(jax.random.key(1), state, jnp.array([2, 2]))
    board = np.asarray(timestep.obs["board"])
    assert board[mover, 5, 2, 0] == 1 and board[1 - mover, 5, 2, 1] == 1
    assert board[1 - mover, :, :, 2].all() and not board[mover, :, :, 2].any()


def test_the_mover_may_play_every_open_column_and_the_waiting_seat_only_passes():
    environment = ConnectX()
    board = jnp.zeros((6, 7), jnp.int8).at[:, 3].set(jnp.array([1, 2, 1, 2, 1, 2], jnp.int8))
    state = ConnectXState(board=board, mover=jnp.int32(1))
    mask = np.asarray(environment.observe(state)["mask"])
    np.testing.assert_array_equal(mask[1], [True, True, True, False, True, True, True])
    np.testing.assert_array_equal(mask[0], [True, False, False, False, False, False, False])


def test_a_learner_plays_a_rival_across_a_batch_of_games():
    def play(carry, params, joint, key, temperature):
        count, *_ = joint.obs["board"].shape
        return carry, jax.random.randint(key, (count,), 0, 7)

    def initialize(key, joint):
        count, *_ = joint.obs["board"].shape
        return jnp.zeros(1), jnp.zeros(count)

    environment = Opponent(Vectorize(ConnectX(), 16), play, initialize)
    state, timestep = environment.init(jax.random.key(0))
    assert timestep.reward.shape == (16,) and timestep.obs["board"].shape == (16, 2, 6, 7, 3)

    step = jax.jit(environment.step)
    for index in range(20):
        state, timestep = step(jax.random.key(index), state, jnp.full((16,), 3, jnp.int32))
    assert bool(jnp.isfinite(timestep.reward).all())


def view(board, mover):
    environment = ConnectX()
    return environment.observe(ConnectXState(board=jnp.asarray(board, jnp.int8), mover=jnp.int32(mover)))


def test_negamax_takes_a_win_in_one():
    board = np.zeros((6, 7), np.int8)
    board[5, :3] = 1
    board[4, :3] = 2
    moves = {int(Negamax(depth=1)(view(board, 0), jax.random.key(seed), 0)) for seed in range(16)}
    assert moves == {3}


def test_negamax_blocks_a_win_in_one_only_when_it_looks_two_plies_ahead():
    board = np.zeros((6, 7), np.int8)
    board[5, :3] = 2
    board[5, 6] = 1
    board[4, 6] = 1
    shallow = {int(Negamax(depth=1)(view(board, 0), jax.random.key(seed), 0)) for seed in range(16)}
    deep = {int(Negamax(depth=2)(view(board, 0), jax.random.key(seed), 0)) for seed in range(16)}
    assert deep == {3} and len(shallow) > 1


def scripted(bot):
    def step(algorithm_state, key, timestep, temperature):
        count, *_ = timestep.obs["board"].shape
        keys = jax.random.split(key, (count, 2))
        seats = jnp.broadcast_to(jnp.arange(2), (count, 2))
        actions = jax.vmap(jax.vmap(bot, in_axes=(None, 0, 0)))(timestep.obs, keys, seats)
        return algorithm_state, actions, {}

    return SimpleNamespace(step=step)


@pytest.mark.parametrize(
    "learner, opponent, low, high",
    [(uniform, uniform, 0.35, 0.65), (Negamax(depth=2), uniform, 0.9, 1.0), (uniform, Negamax(depth=2), 0.0, 0.1)],
)
def test_the_gauntlet_scores_every_game_from_the_learners_seat(learner, opponent, low, high):
    gauntlet = Gauntlet({"rival": opponent}, num_games=64)
    play = gauntlet.match(scripted(learner), Vectorize(ConnectX(), 1), opponent)
    score, over = play(None, jax.random.key(0))
    assert bool(over.all())
    assert low <= float(score.mean()) <= high

    metrics = gauntlet.craft(scripted(learner), Vectorize(ConnectX(), 1), SimpleNamespace(algorithm_state=None), {})
    assert set(metrics.data) == {"gauntlet/rival", "gauntlet/mean"}
