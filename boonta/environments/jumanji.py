import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space

WALL = 1
TARGET = 2
AGENT = 3
BOX = 4
MOVES = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)


def unsolve(state, key: Key, iters: int):
    fixed = state.fixed_grid
    size, *_ = fixed.shape
    boxes = (fixed == TARGET).astype(jnp.uint8) * jnp.uint8(BOX)

    def empty(grid, cell):
        inside = jnp.all((cell >= 0) & (cell < size))
        return inside & (fixed[cell[0], cell[1]] != WALL) & (
            grid[cell[0], cell[1]] != BOX
        )

    def spot(key):
        flat = jax.random.permutation(key, size * size)
        free = (fixed.reshape(-1) != WALL) & (boxes.reshape(-1) != BOX)
        pick = flat[jnp.argmax(jnp.take(free, flat))]
        return jnp.stack([pick // size, pick % size])

    agent = spot(key)

    def pull(carry, step_key):
        grid, agent = carry
        d = MOVES[jax.random.randint(step_key, (), 0, 4)]
        front, back = agent + d, agent - d
        can_pull = (
            jnp.all((front >= 0) & (front < size))
            & (grid[front[0], front[1]] == BOX)
            & empty(grid, back)
        )
        can_step = empty(grid, back)
        grid = jnp.where(
            can_pull,
            grid.at[front[0], front[1]].set(0).at[agent[0], agent[1]].set(BOX),
            grid,
        )
        agent = jnp.where(can_pull | can_step, back, agent)
        return (grid, agent), None

    (boxes, agent), _ = jax.lax.scan(
        pull, (boxes, agent), jax.random.split(key, iters)
    )
    variable = boxes.at[agent[0], agent[1]].set(AGENT)
    return state.replace(
        variable_grid=variable.astype(state.variable_grid.dtype),
        agent_location=agent.astype(state.agent_location.dtype),
        step_count=jnp.zeros_like(state.step_count),
    )


def snake(env, bucket: int = 4, fruit: bool = False):
    """Groups Snake states by where the head is and how long the body has grown."""

    def cell(env_state):
        head = env_state.head_position
        features = [head.row, head.col, env_state.length // bucket]
        if fruit:
            features.extend([env_state.fruit_position.row, env_state.fruit_position.col])
        key = jnp.zeros_like(env_state.length, jnp.int32)
        for value in features:
            key = key * jnp.int32(1000003) + value.astype(jnp.int32)
        return key

    return cell


class Jumanji(Environment):
    def __init__(
        self,
        env,
        fixed: int | None = None,
        pool: int | None = None,
        pulls: int | None = None,
        pulled: float = 0.5,
    ):
        self._env = env
        self._fixed = fixed
        self._pool = pool
        self._pulls = pulls
        self._pulled = pulled

    def draw(self, key: Key) -> Key:
        if self._fixed is not None:
            return jax.random.key(self._fixed)
        if self._pool is not None:
            slot = jax.random.randint(key, (), 0, self._pool)
            return jax.random.key(slot)
        return key

    def stack(self, observation) -> Array:
        grid = observation.grid
        clock = jnp.broadcast_to(
            jnp.asarray(observation.step_count, grid.dtype), grid.shape[:-1] + (1,)
        )
        return jnp.concatenate([grid, clock], axis=-1)

    def observe(self, state) -> Array:
        return self.stack(self._env._state_to_observation(state))

    def frame(self, timestep, action: Array) -> Timestep:
        return Timestep(
            obs=self.stack(timestep.observation),
            action=action,
            reward=jnp.asarray(timestep.reward, jnp.float32),
            terminated=timestep.step_type == 2,
            truncated=jnp.zeros_like(timestep.step_type == 2),
            info={},
        )

    def init(self, key: Key):
        draw_key, pull_key, coin_key = jax.random.split(key, 3)
        state, timestep = self._env.reset(self.draw(draw_key))
        if self._pulls is not None:
            pulled = jax.random.uniform(coin_key) < self._pulled
            state = jax.tree.map(
                lambda a, b: jnp.where(pulled, a, b),
                unsolve(state, pull_key, self._pulls),
                state,
            )
            timestep = timestep.replace(
                observation=timestep.observation._replace(
                    grid=jnp.stack(
                        [state.variable_grid, state.fixed_grid], axis=-1
                    ),
                    step_count=state.step_count,
                )
            )
        first = self.frame(timestep, jnp.int32(0))
        return state, first.replace(terminated=jnp.ones_like(first.terminated))

    def step(self, key: Key, state, action: Array):
        action = jnp.asarray(action, jnp.int32)
        state, timestep = self._env.step(state, action)
        return state, self.frame(timestep, action)

    def observation_space(self) -> Space:
        spec = self._env.observation_spec.grid
        shape = spec.shape[:-1] + (spec.shape[-1] + 1,)
        return Space(shape=shape, dtype=spec.dtype, low=0, high=255)

    def action_space(self) -> Space:
        spec = self._env.action_spec
        return Space(
            shape=(), dtype=jnp.int32, low=0, high=spec.num_values - 1
        )

    def time_limit(self) -> int:
        return int(self._env.time_limit)

    def render(self, state) -> np.ndarray:
        viewer = self._env._viewer
        # Jumanji viewers draw to a window unless built for rgb_array, so switch them over.
        viewer._display = viewer._display_rgb_array
        image = viewer.render(getattr(state, "unwrapped", state))
        viewer.close()
        return np.asarray(image)[..., :3]


def make(
    env_id,
    fixed: int | None = None,
    pool: int | None = None,
    pulls: int | None = None,
    pulled: float = 0.5,
    dataset: str | None = None,
    box_bonus: float | None = None,
    **kwargs,
):
    import jumanji

    if dataset is not None:
        from jumanji.environments.routing.sokoban.generator import \
            HuggingFaceDeepMindGenerator

        kwargs["generator"] = HuggingFaceDeepMindGenerator(dataset)
    if box_bonus is not None:
        from jumanji.environments.routing.sokoban import reward as rw

        class Scaled(rw.DenseReward):
            def __call__(self, state, action, next_state):
                boxes = rw.SINGLE_BOX_BONUS * (
                    self.count_targets(next_state) - self.count_targets(state)
                )
                solved = self.count_targets(next_state) == rw.N_BOXES
                return (
                    box_bonus * boxes
                    + rw.LEVEL_COMPLETE_BONUS * solved
                    + rw.STEP_BONUS
                )

        kwargs["reward_fn"] = Scaled()
    return Jumanji(
        jumanji.make(env_id, **kwargs),
        fixed=fixed,
        pool=pool,
        pulls=pulls,
        pulled=pulled,
    )
