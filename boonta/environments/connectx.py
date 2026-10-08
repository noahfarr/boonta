import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@struct.dataclass
class ConnectXState:
    board: Array
    mover: Array


class ConnectX(Environment):
    num_agents = 2

    def __init__(self, rows: int = 6, columns: int = 7, inarow: int = 4):
        self.rows = rows
        self.columns = columns
        self.inarow = inarow

    def observation_space(self) -> dict[str, Space]:
        return {
            "board": Space((2, self.rows, self.columns, 3), jnp.float32, 0.0, 1.0),
            "mask": Space((2, self.columns), jnp.bool_, 0, 1),
        }

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, self.columns - 1)

    def time_limit(self) -> int:
        return self.rows * self.columns

    def restore(self, view: Array, seat: Array) -> ConnectXState:
        own, other, turn = view[..., 0], view[..., 1], view[..., 2]
        board = own * (seat + 1) + other * (2 - seat)
        mover = jnp.where(turn.any(), seat, 1 - seat)
        return ConnectXState(board=board.astype(jnp.int8), mover=mover)

    def action_mask(self, state: ConnectXState) -> Array:
        movable = state.board[0] == 0
        waiting = jnp.arange(self.columns) == 0
        return jnp.where((jnp.arange(2) == state.mover)[:, None], movable, waiting)

    def observe(self, state: ConnectXState) -> dict[str, Array]:
        pieces = jnp.stack([state.board == 1, state.board == 2], axis=-1)
        views = jnp.stack([pieces, pieces[..., ::-1]])
        turn = jnp.broadcast_to(
            (jnp.arange(2) == state.mover)[:, None, None, None],
            (2, self.rows, self.columns, 1),
        )
        board = jnp.concatenate([views, turn], axis=-1).astype(jnp.float32)
        return {"board": board, "mask": self.action_mask(state)}

    def run(self, board: Array, row: Array, column: Array, piece: Array, step) -> Array:
        down, across = step
        length, alive = jnp.int32(0), jnp.bool_(True)
        for distance in range(1, self.inarow):
            r, c = row + distance * down, column + distance * across
            inside = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.columns)
            cell = board[jnp.clip(r, 0, self.rows - 1), jnp.clip(c, 0, self.columns - 1)]
            alive = alive & inside & (cell == piece)
            length = length + alive
        return length

    def connects(self, board: Array, row: Array, column: Array, piece: Array) -> Array:
        lines = [
            1
            + self.run(board, row, column, piece, (down, across))
            + self.run(board, row, column, piece, (-down, -across))
            for down, across in DIRECTIONS
        ]
        return jnp.any(jnp.stack(lines) >= self.inarow)

    def init(self, key: Key) -> tuple[ConnectXState, Timestep]:
        state = ConnectXState(
            board=jnp.zeros((self.rows, self.columns), jnp.int8),
            mover=jax.random.randint(key, (), 0, 2),
        )
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.zeros((2,), jnp.int32),
            reward=jnp.zeros((2,), jnp.float32),
            terminated=jnp.ones((2,), bool),
            truncated=jnp.zeros((2,), bool),
        )

    def step(
        self, key: Key, state: ConnectXState, action: Array
    ) -> tuple[ConnectXState, Timestep]:
        column = jnp.take(action, state.mover)
        inside = (column >= 0) & (column < self.columns)
        column = jnp.clip(column, 0, self.columns - 1)
        height = jnp.sum(state.board[:, column] != 0)
        legal = inside & (height < self.rows)
        row = jnp.clip(self.rows - 1 - height, 0, self.rows - 1)
        piece = (state.mover + 1).astype(jnp.int8)
        board = jnp.where(legal, state.board.at[row, column].set(piece), state.board)

        won = legal & self.connects(board, row, column, piece)
        full = legal & jnp.all(board != 0)
        sign = jnp.where(jnp.arange(2) == state.mover, 1.0, -1.0)
        reward = jnp.where(won, sign, jnp.where(legal, 0.0, -sign))
        done = won | full | ~legal

        state = ConnectXState(board=board, mover=1 - state.mover)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward.astype(jnp.float32),
            terminated=jnp.broadcast_to(done, (2,)),
            truncated=jnp.zeros((2,), bool),
        )


def uniform(obs: dict[str, Array], key: Key, seat: Array) -> Array:
    return jax.random.categorical(key, jnp.where(obs["mask"][seat], 0.0, -jnp.inf))


class Negamax:
    def __init__(self, depth: int, rows: int = 6, columns: int = 7, inarow: int = 4):
        self.depth = depth
        self.game = ConnectX(rows, columns, inarow)

    def search(self, state: ConnectXState, depth: int) -> Array:
        def child(column):
            action = jnp.full((2,), column, jnp.int32)
            after, timestep = self.game.step(jax.random.key(0), state, action)
            ending = jnp.take(timestep.reward, state.mover) * depth
            if depth == 1:
                return ending
            deeper = -self.search(after, depth - 1).max()
            return jnp.where(timestep.terminated[0], ending, deeper)

        columns = jnp.arange(self.game.columns)
        if depth == self.depth:
            return jax.lax.map(child, columns)
        return jax.vmap(child)(columns)

    def __call__(self, obs: dict[str, Array], key: Key, seat: Array) -> Array:
        state = self.game.restore(obs["board"][seat], seat)
        values = self.search(state, self.depth)
        return jax.random.categorical(
            key, jnp.where(values == values.max(), 0.0, -jnp.inf)
        )


def make(env_id: str = "connectx", **kwargs) -> ConnectX:
    return ConnectX(**kwargs)
