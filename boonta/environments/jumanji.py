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


def reseed(env_state, key):
    """Gives restored states their own randomness, so copies of one snapshot draw different fruit."""
    from .wrappers import WrapperState

    if isinstance(env_state, WrapperState):
        return env_state.replace(env_state=reseed(env_state.env_state, key))
    fresh = jax.random.split(key, env_state.key.shape[0])
    if not jax.dtypes.issubdtype(env_state.key.dtype, jax.dtypes.prng_key):
        fresh = jax.random.key_data(fresh).astype(env_state.key.dtype)
    return env_state.replace(key=fresh)


def pac_man(env, bucket: int = 32):
    """Groups PacMan states by where PacMan is, the pellets left in buckets, and the power-ups left."""

    def cell(env_state):
        player = env_state.player_locations
        power_ups = jnp.sum(jnp.any(env_state.power_up_locations != 0, axis=-1), axis=-1)
        features = [player.x, player.y, env_state.pellets // bucket, power_ups]
        key = jnp.zeros_like(env_state.pellets, jnp.int32)
        for value in features:
            key = key * jnp.int32(1000003) + jnp.asarray(value).astype(jnp.int32)
        return key

    return cell


FRIGHTENED = 30  # steps of frightened ghosts after a power-up, fixed by Jumanji's PacMan


def pac_man_planes(observation) -> Array:
    """Jumanji's PacMan observation as image planes: the open floor (Jumanji's grid, 1 where PacMan
    can walk), PacMan, ghosts, power-ups, pellets and the frightened time left. Ghosts, power-ups and
    pellets are stored as (column, row), and eaten ones move to (0, 0), a wall."""
    grid = observation.grid.astype(jnp.float32)

    def plane(points):
        points = points.astype(jnp.int32)
        kept = jnp.any(points != 0, axis=-1).astype(jnp.float32)
        return jnp.zeros_like(grid).at[points[:, 1], points[:, 0]].add(kept)

    player = observation.player_locations
    planes = [
        grid,
        jnp.zeros_like(grid).at[player.x, player.y].set(1.0),
        plane(observation.ghost_locations),
        plane(observation.power_up_locations),
        jnp.minimum(plane(observation.pellet_locations), 1.0),
        jnp.full_like(grid, jnp.maximum(observation.frightened_state_time, 0) / FRIGHTENED),
    ]
    return jnp.stack(planes, axis=-1)


def game_2048(env, bucket: int = 32):
    """Groups 2048 states by their age, the bucketed sum of the tiles, and their largest tile.

    Every move that changes the board spawns a 2 or a 4 and merges keep the sum, so the sum
    counts how far into the game a board is. The exponent of the largest tile says which stage
    of the game it has reached.
    """

    def cell(env_state):
        board = env_state.board
        tiles = jnp.where(board > 0, jnp.left_shift(1, board), 0)
        age = jnp.sum(tiles, axis=(-2, -1)) // bucket
        largest = jnp.max(board, axis=(-2, -1))
        return (age.astype(jnp.int32) * jnp.int32(1000003) + largest.astype(jnp.int32)) * jnp.int32(
            1664525
        )

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
        # Board games such as Game2048 observe a board of integers and carry their legal moves.
        self._board = hasattr(env.observation_spec, "board")
        self._pac_man = hasattr(env.observation_spec, "pellet_locations")
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
        if self._board:
            return {"board": observation.board, "mask": observation.action_mask}
        if self._pac_man:
            return pac_man_planes(observation)
        grid = observation.grid
        clock = jnp.broadcast_to(
            jnp.asarray(observation.step_count, grid.dtype), grid.shape[:-1] + (1,)
        )
        return jnp.concatenate([grid, clock], axis=-1)

    def observe(self, state) -> Array:
        if self._board:
            return {"board": state.board, "mask": state.action_mask}
        if self._pac_man:
            return self.stack(self._env._observation_from_state(state))
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

    def observation_space(self) -> Space | dict[str, Space]:
        if self._board:
            spec = self._env.observation_spec.board
            rows, columns = spec.shape
            # A board of n cells holds tiles up to 2**(n + 1), stored as their exponents.
            mask = self._env.observation_spec.action_mask
            return {
                "board": Space(shape=spec.shape, dtype=spec.dtype, low=0, high=rows * columns + 1),
                "mask": Space(shape=mask.shape, dtype=jnp.bool_, low=0, high=1),
            }
        spec = self._env.observation_spec.grid
        if self._pac_man:
            return Space(shape=(*spec.shape, 6), dtype=jnp.float32, low=0.0, high=4.0)
        shape = spec.shape[:-1] + (spec.shape[-1] + 1,)
        return Space(shape=shape, dtype=spec.dtype, low=0, high=255)

    def action_space(self) -> Space:
        spec = self._env.action_spec
        return Space(
            shape=(), dtype=jnp.int32, low=0, high=spec.num_values - 1
        )

    def action_mask(self, state) -> Array | None:
        if self._board:
            return state.action_mask
        return super().action_mask(state)

    def time_limit(self) -> int:
        if not hasattr(self._env, "time_limit"):
            raise NotImplementedError(f"{type(self._env).__name__} has no time limit")
        return int(self._env.time_limit)

    def render(self, state) -> np.ndarray:
        viewer = self._env._viewer
        # Jumanji viewers draw to a window unless built for rgb_array, so switch them over.
        viewer._display = viewer._display_rgb_array
        state = getattr(state, "unwrapped", state)
        if self._pac_man:
            # The PacMan viewer draws with JAX's .at[] updates, which NumPy arrays lack.
            state = jax.tree.map(jnp.asarray, state)
        image = viewer.render(state)
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
    if env_id.startswith("PacMan"):
        # PacMan keeps its step count in the state and ends at its limit as a termination, so a
        # restored snapshot would carry its clock along and running out of time would read as
        # dying. Boonta's limit truncates instead, and a restart resets it.
        from .wrappers import TimeLimit

        time_limit = kwargs.pop("time_limit", 1000)
        env = Jumanji(jumanji.make(env_id, time_limit=NEVER, **kwargs))
        return TimeLimit(env, time_limit)
    return Jumanji(
        jumanji.make(env_id, **kwargs),
        fixed=fixed,
        pool=pool,
        pulls=pulls,
        pulled=pulled,
    )


NEVER = 2**30
