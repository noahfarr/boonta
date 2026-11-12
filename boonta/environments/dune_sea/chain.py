import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, Timestep


@struct.dataclass
class ChainState:
    position: Array


def cell(env):
    return lambda env_state: env_state.position


def shackle_rows(columns: int) -> int:
    return int(np.ceil(0.35 * columns + 0.3))


def layout(length: int) -> np.ndarray:
    """The wheels of a padlock on a near-square grid in reading order, below empty rows for its shackle."""
    columns = int(np.ceil(np.sqrt(length)))
    rows = -(-length // columns)
    body = np.full(rows * columns, -1, np.int32)
    body[:length] = np.arange(length)
    return np.concatenate([np.full((shackle_rows(columns), columns), -1, np.int32), body.reshape(rows, columns)])


# Three by five bitmaps of the digits 0 to 9, for the wheels.
DIGITS = np.array(
    [
        [int(bit) for bit in "".join(glyph)]
        for glyph in (
            ("111", "101", "101", "101", "111"),
            ("010", "110", "010", "010", "111"),
            ("111", "001", "111", "100", "111"),
            ("111", "001", "111", "001", "111"),
            ("101", "101", "111", "001", "001"),
            ("111", "100", "111", "001", "111"),
            ("111", "100", "111", "101", "111"),
            ("111", "001", "010", "010", "010"),
            ("111", "101", "111", "101", "111"),
            ("111", "101", "111", "001", "111"),
        )
    ],
    bool,
).reshape(10, 5, 3)


class Chain(Environment):
    def __init__(
        self, length: int = 64, num_actions: int = 4, hazard: float = 0.2, seed: int = 0, lock: bool = False
    ):
        assert length >= 1 and num_actions >= 2 and 0.0 <= hazard < 1.0
        self.num_cells = length
        self.num_actions = num_actions
        self.hazard = hazard
        # A lock ends the episode on a wrong answer, so only a policy that knows the cells before one reaches it.
        self.lock = lock
        self.start = 0
        self.secrets = jnp.asarray(np.random.default_rng(seed).integers(0, num_actions, length), jnp.int32)
        self.reach = jnp.asarray((1.0 - hazard) ** np.arange(length), jnp.float32)
        self.cell_grid = jnp.asarray(layout(length))

    def observation_space(self) -> Space:
        return Space((self.num_cells,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, self.num_actions - 1)

    def time_limit(self) -> int:
        return self.num_cells

    def correct(self, policy: Array) -> Array:
        return policy[jnp.arange(self.num_cells), self.secrets]

    def onward(self, policy: Array) -> Array:
        """Probability that the episode goes on from each cell to the next."""
        survive = (1.0 - self.hazard) * (self.correct(policy) if self.lock else jnp.ones(self.num_cells))
        return survive.at[-1].set(0.0)

    def reached(self, policy: Array) -> Array:
        """Probability that an episode from the start reaches each cell."""
        return jnp.concatenate([jnp.ones(1), jnp.cumprod(self.onward(policy)[:-1])])

    def expected_return(self, policy: Array) -> Array:
        return jnp.sum(self.reached(policy) * self.correct(policy))

    def values(self, policy: Array) -> Array:
        def back(after, cell):
            correct, onward = cell
            value = correct + onward * after
            return value, value

        _, values = jax.lax.scan(back, jnp.float32(0.0), (self.correct(policy), self.onward(policy)), reverse=True)
        return values

    def advantages(self, policy: Array) -> Array:
        payoff = jax.nn.one_hot(self.secrets, self.num_actions)
        after = jnp.concatenate([self.values(policy)[1:], jnp.zeros(1)])
        last = jnp.arange(self.num_cells) == self.num_cells - 1
        goes_on = jnp.where(last[:, None], 0.0, payoff if self.lock else jnp.ones_like(payoff))
        q = payoff + (1.0 - self.hazard) * goes_on * after[:, None]
        return q - jnp.sum(policy * q, axis=-1, keepdims=True)

    def transitions(self, policy: Array) -> Array:
        onward = jnp.diag(self.onward(policy)[:-1], k=1)
        return onward.at[:, self.start].add(1.0 - jnp.sum(onward, axis=-1))

    def place(self, state: ChainState, cell: Array) -> ChainState:
        return state.replace(position=jnp.int32(cell))

    def observe(self, state: ChainState) -> Array:
        return jax.nn.one_hot(state.position, self.num_cells)

    def init(self, key: Key) -> tuple[ChainState, Timestep]:
        state = ChainState(position=jnp.int32(self.start))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(self, key: Key, state: ChainState, action: Array) -> tuple[ChainState, Timestep]:
        last = self.num_cells - 1
        reward = action == self.secrets[state.position]
        ended = (state.position == last) | jax.random.bernoulli(key, self.hazard) | (self.lock & ~reward)
        state = ChainState(position=jnp.minimum(state.position + 1, last))
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward.astype(jnp.float32),
            terminated=ended,
            truncated=jnp.bool_(False),
        )

    def render(self, state: ChainState, tile: int = 32) -> Array:
        """A padlock: a steel shackle over a brass body holding one number wheel per cell.

        The wheels read left to right and top to bottom. Opened wheels show the digit of their secret
        action, the wheel being answered is framed red, and wheels ahead are dark and blank. Digits are
        drawn for up to ten actions.
        """
        num_rows, num_columns = self.cell_grid.shape
        top = shackle_rows(num_columns)
        y, x = jnp.meshgrid(
            (jnp.arange(num_rows * tile) + 0.5) / tile,
            (jnp.arange(num_columns * tile) + 0.5) / tile,
            indexing="ij",
        )
        row, column = jnp.floor(y).astype(jnp.int32), jnp.floor(x).astype(jnp.int32)
        across, down = x - column - 0.5, y - row - 0.5
        cell = self.cell_grid[row, column]
        held = cell >= 0
        colour = jnp.full((*y.shape, 3), 245.0)

        def paint(mask, value):
            return jnp.where(mask[..., None], jnp.asarray(value, jnp.float32), colour)

        # The shackle: a thick arch whose legs run down into the body.
        middle, outer = num_columns / 2.0, 0.35 * num_columns
        thick = 0.28 * outer
        crown = outer + 0.2
        bend = jnp.hypot(x - middle, y - crown)
        side = jnp.abs(x - middle)
        reach = jnp.where(y < crown, bend, side)
        shackle = (reach < outer) & (reach > outer - thick) & (y < top + 0.5)
        rim = shackle & ((reach > outer - 0.07) | (reach < outer - thick + 0.07))
        colour = paint(shackle, [180.0, 184.0, 192.0])
        colour = paint(rim, [105.0, 108.0, 115.0])

        # The body: a rounded brass block over the wheels.
        left, right, upper, lower, corner = 0.08, num_columns - 0.08, top - 0.25, num_rows - 0.08, 0.45
        outside = jnp.hypot(
            jnp.maximum(jnp.maximum(left + corner - x, x - (right - corner)), 0.0),
            jnp.maximum(jnp.maximum(upper + corner - y, y - (lower - corner)), 0.0),
        )
        colour = paint(outside < corner, [125.0, 92.0, 25.0])
        colour = paint(outside < corner - 0.08, [210.0, 165.0, 60.0])

        # The wheels: a window per cell with a drum that darkens towards its edges.
        window = held & (jnp.abs(across) < 0.33) & (jnp.abs(down) < 0.42)
        drum = held & (jnp.abs(across) < 0.27) & (jnp.abs(down) < 0.36)
        opened = cell < state.position
        current = cell == state.position
        frame = jnp.where(current[..., None], jnp.array([230.0, 20.0, 20.0]), jnp.array([45.0, 38.0, 30.0]))
        colour = jnp.where(window[..., None], frame, colour)
        shade = (1.0 - 0.55 * (down / 0.36) ** 2)[..., None]
        face = jnp.where((opened | current)[..., None], jnp.array([248.0, 242.0, 222.0]), jnp.array([92.0, 92.0, 98.0]))
        colour = jnp.where(drum[..., None], face * shade, colour)
        if self.num_actions <= 10:
            glyph_x = jnp.floor((across + 0.15) / 0.3 * 3).astype(jnp.int32)
            glyph_y = jnp.floor((down + 0.24) / 0.48 * 5).astype(jnp.int32)
            inside = (glyph_x >= 0) & (glyph_x < 3) & (glyph_y >= 0) & (glyph_y < 5)
            digit = jnp.asarray(DIGITS)[
                self.secrets[jnp.clip(cell, 0)], jnp.clip(glyph_y, 0, 4), jnp.clip(glyph_x, 0, 2)
            ]
            colour = paint(drum & opened & inside & digit, [25.0, 25.0, 25.0])
        return colour.astype(jnp.uint8)


class Scorecard:
    def __init__(self, name: str = "chain"):
        self.name = name

    def craft(self, key, podracer, podracer_state, logs):
        from boonta.artisans.artifact import Metrics

        algorithm, environment = podracer.algorithm, podracer.environment
        game = getattr(environment, "unwrapped", environment)
        empty = jnp.zeros(game.num_cells)
        timestep = Timestep(
            obs=jnp.eye(game.num_cells),
            action=empty.astype(jnp.int32),
            reward=empty,
            terminated=empty.astype(bool),
            truncated=empty.astype(bool),
        )
        _, _, _, aux = algorithm.step(podracer_state.algorithm_state, None, key, timestep, temperature=1.0)
        policy = jax.nn.softmax(aux["logits"])
        correct = game.correct(policy)
        data = {
            f"{self.name}/return": game.expected_return(policy),
            f"{self.name}/best_return": game.expected_return(jax.nn.one_hot(game.secrets, game.num_actions)),
            f"{self.name}/correct": jnp.mean(correct),
        }
        data |= {f"{self.name}/correct_{index}": correct[index] for index in range(game.num_cells)}
        return Metrics(self.name, jax.device_get(data))
