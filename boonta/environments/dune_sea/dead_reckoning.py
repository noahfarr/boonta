import jax
import jax.numpy as jnp
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, Timestep

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
COMPASS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], jnp.int32)
TRIANGLE = jnp.array([[0.12, 0.19], [0.87, 0.5], [0.12, 0.81]])
BLACK, GREY = jnp.zeros(3), jnp.full(3, 100.0)
GREEN, RED = jnp.array([0.0, 255.0, 0.0]), jnp.array([255.0, 0.0, 0.0])


@struct.dataclass
class DeadReckoningState:
    target: Array
    position: Array
    heading: Array
    time: Array
    executed: Array


class DeadReckoning(Environment):
    def __init__(
        self,
        radius: int = 3,
        egocentric: bool = False,
        glimpse: float = 0.0,
        slip_probability: float = 0.0,
    ):
        assert radius >= 1
        assert 0.0 <= glimpse <= 1.0
        assert 0.0 <= slip_probability <= 1.0
        self.radius = radius
        self.egocentric = egocentric
        self.glimpse = glimpse
        self.slip_probability = slip_probability
        self.num_actions = 4 if self.egocentric else 5

    def observation_space(self) -> Space:
        return Space(
            (2 * (2 * self.radius + 1) + self.num_actions,), jnp.float32, 0.0, 1.0
        )

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, self.num_actions - 1)

    def time_limit(self) -> int:
        return 4 * self.radius + 4

    def wrap(self, position: Array) -> Array:
        return (position + self.radius) % (2 * self.radius + 1) - self.radius

    def offset(self, state: DeadReckoningState) -> Array:
        offset = self.wrap(state.target - state.position) + self.radius
        return jax.nn.one_hot(offset, 2 * self.radius + 1).reshape(-1)

    def observe(self, state: DeadReckoningState) -> Array:
        executed = jax.nn.one_hot(state.executed, self.num_actions)
        return jnp.concatenate([self.offset(state), executed])

    def init(self, key: Key) -> tuple[DeadReckoningState, Timestep]:
        side = 2 * self.radius + 1
        start_key, target_key = jax.random.split(key)
        start = jax.random.randint(start_key, (), 0, side**2)
        target = jax.random.randint(target_key, (), 0, side**2 - 1)
        target = target + (target >= start)
        cell = lambda index: jnp.stack(jnp.divmod(index, side)).astype(jnp.int32) - self.radius
        state = DeadReckoningState(
            target=cell(target),
            position=cell(start),
            heading=jnp.int32(0),
            time=jnp.int32(0),
            executed=jnp.int32(-1),
        )
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(
        self, key: Key, state: DeadReckoningState, action: Array
    ) -> tuple[DeadReckoningState, Timestep]:
        slip_key, move_key, glimpse_key = jax.random.split(key, 3)
        stopped = action == STOP
        slipped = ~stopped & (jax.random.uniform(slip_key) < self.slip_probability)
        executed = jnp.where(
            slipped, jax.random.randint(move_key, (), 1, self.num_actions), action
        )
        if self.egocentric:
            turn = jnp.where(executed == LEFT, -1, jnp.where(executed == RIGHT, 1, 0))
            heading = (state.heading + turn) % 4
            moved = executed == FORWARD
        else:
            heading = jnp.maximum(executed - 1, 0)
            moved = ~stopped
        position = self.wrap(state.position + moved * COMPASS[heading])
        state = state.replace(
            position=position,
            heading=heading,
            time=state.time + 1,
            executed=executed.astype(jnp.int32),
        )
        reward = jnp.where(stopped & jnp.all(state.position == state.target), 1.0, 0.0)
        visible = jax.random.uniform(glimpse_key) < self.glimpse
        offset = jnp.where(visible, self.offset(state), 0.0)
        executed = jax.nn.one_hot(state.executed, self.num_actions)
        return state, Timestep(
            obs=jnp.concatenate([offset, executed]),
            action=action,
            reward=reward.astype(jnp.float32),
            terminated=stopped,
            truncated=~stopped & (state.time >= self.time_limit()),
        )

    def render(
        self, state: DeadReckoningState, cell: int = 32, subdivisions: int = 3
    ) -> Array:
        grid = 2 * self.radius + 1
        fine = cell * subdivisions
        u = (jnp.arange(fine) + 0.5) / fine
        py, px = jnp.meshgrid(u, u, indexing="ij")
        row, column = jnp.meshgrid(jnp.arange(grid), jnp.arange(grid), indexing="ij")
        x, y = column - self.radius, self.radius - row
        goal = (x == state.target[0]) & (y == state.target[1])
        agent = (x == state.position[0]) & (y == state.position[1])

        theta = jnp.pi / 2 * ((state.heading + 3) % 4)
        cos, sin = jnp.cos(theta), jnp.sin(theta)
        corners = (TRIANGLE - 0.5) @ jnp.array([[cos, sin], [-sin, cos]]) + 0.5
        sides = [
            (px - bx) * (ay - by) - (ax - bx) * (py - by)
            for (ax, ay), (bx, by) in zip(corners, jnp.roll(corners, -1, axis=0))
        ]
        triangle = jnp.all(jnp.stack(sides) >= 0, 0) | jnp.all(jnp.stack(sides) <= 0, 0)
        line = (px < 0.031) | (py < 0.031)

        tile = lambda mask: mask[:, :, None, None, None]
        image = jnp.where(line[..., None], GREY, BLACK)[None, None]
        image = jnp.where(tile(goal), GREEN, image)
        image = jnp.where(tile(agent) & triangle[..., None], RED, image)
        image = image.transpose(0, 2, 1, 3, 4).reshape(
            grid * cell, subdivisions, grid * cell, subdivisions, 3
        )
        return image.mean((1, 3)).round().astype(jnp.uint8)
