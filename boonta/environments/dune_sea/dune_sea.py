import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space
from .generators import Generator, uniform_generator
from .render import render
from .state import MAX_ENERGY, Agent, Cell, DuneSeaState

ENERGY_COST = 0.01
ENERGY_GAIN = 0.25
NOOP, FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT, INTERACT, BROADCAST = range(9)
NUM_ACTIONS = 9
DIRECTIONS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
MOVE_OFFSET = jnp.array([0, 0, 2, 3, 1, 0, 0, 0, 0], dtype=jnp.int32)
STEP_SIZE = jnp.array([0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=jnp.int32)
TURN_OFFSET = jnp.array([0, 0, 0, 0, 0, 3, 1, 0, 0], dtype=jnp.int32)


NUM_CHANNELS = len(Cell.__dataclass_fields__) + 2


class DuneSea(Environment):

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        num_agents: int = 1,
        view_size: int = 7,
        time_limit: int = 256,
        generator: Generator = uniform_generator,
    ):
        self.height = height
        self.width = width
        self.num_agents = num_agents
        self.view_size = view_size
        self._time_limit = time_limit
        self.generator = generator

    def init(self, key: Key) -> tuple[DuneSeaState, Timestep]:
        cells, agent = self.generator(key, self.height, self.width, self.num_agents)
        state = DuneSeaState(cells=cells, agent=agent, time=jnp.int32(0))
        timestep = Timestep(
            obs=self.observe(state),
            action=jnp.zeros(self.num_agents, jnp.int32),
            reward=jnp.zeros(self.num_agents, jnp.float32),
            terminated=jnp.zeros(self.num_agents, bool),
            truncated=jnp.zeros(self.num_agents, bool),
            info={},
        )
        return state, timestep

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        action = jnp.asarray(action, jnp.int32)
        grid_size = jnp.array([self.height, self.width])
        direction = state.agent.direction

        def move(positions, agent_index):
            position = positions[agent_index]
            agent_action = action[agent_index]
            move_direction = DIRECTIONS[
                (direction[agent_index] + MOVE_OFFSET[agent_action]) % 4
            ]
            target_position = position + STEP_SIZE[agent_action] * move_direction
            in_bounds = jnp.all((target_position >= 0) & (target_position < grid_size))
            y, x = jnp.clip(target_position, 0, grid_size - 1)
            occupied = jnp.any(jnp.all(positions == target_position, axis=-1))
            can_move = in_bounds & (state.cells.wall[y, x] == 0) & ~occupied
            position = jnp.where(can_move, target_position, position)
            return positions.at[agent_index].set(position), None

        turn_order = jax.random.permutation(key, self.num_agents)
        positions, _ = jax.lax.scan(move, state.agent.position, turn_order)

        def mine(wall, agent_index):
            target_position = positions[agent_index] + DIRECTIONS[direction[agent_index]]
            in_bounds = jnp.all((target_position >= 0) & (target_position < grid_size))
            y, x = jnp.clip(target_position, 0, grid_size - 1)
            can_mine = (action[agent_index] == INTERACT) & in_bounds & (wall[y, x] > 0)
            return wall.at[y, x].set(jnp.where(can_mine, 0.0, wall[y, x])), can_mine

        wall, mined_in_turn_order = jax.lax.scan(mine, state.cells.wall, turn_order)
        mined = jnp.zeros(self.num_agents, bool).at[turn_order].set(mined_in_turn_order)

        y, x = positions[:, 0], positions[:, 1]
        can_collect = (action == INTERACT) & ~mined & (state.cells.energy[y, x] > 0)
        energy = jnp.where(
            can_collect, state.agent.energy + ENERGY_GAIN, state.agent.energy
        )
        energy = jnp.clip(energy - ENERGY_COST, 0.0, MAX_ENERGY)
        energy_delta = energy - state.agent.energy
        cells = state.cells.replace(
            wall=wall,
            energy=state.cells.energy.at[y, x].set(
                jnp.where(can_collect, 0.0, state.cells.energy[y, x])
            )
        )

        agent = Agent(
            position=positions,
            direction=(direction + TURN_OFFSET[action]) % 4,
            energy=energy,
            broadcast=action == BROADCAST,
            inventory=state.agent.inventory + mined.astype(jnp.int32),
        )
        state = state.replace(cells=cells, agent=agent, time=state.time + 1)
        timestep = Timestep(
            obs=self.observe(state),
            action=action,
            reward=energy_delta,
            terminated=state.agent.energy <= 0.0,
            truncated=jnp.full(self.num_agents, state.time >= self._time_limit),
            info={},
        )
        return state, timestep

    def observe(self, state: DuneSeaState) -> dict[str, Array]:
        radius = self.view_size // 2
        y, x = state.agent.position[:, 0], state.agent.position[:, 1]
        agents = jnp.zeros((self.height, self.width), jnp.float32).at[y, x].set(1.0)
        wall = jnp.pad(state.cells.wall, radius, constant_values=1.0)
        broadcasts = jnp.zeros((self.height, self.width), jnp.float32)
        broadcasts = broadcasts.at[y, x].set(state.agent.broadcast.astype(jnp.float32))
        others = [state.cells.goal, state.cells.energy, agents, broadcasts]
        grid = jnp.stack([wall] + [jnp.pad(plane, radius) for plane in others], axis=-1)

        def egocentric_view(position, direction):
            view = jax.lax.dynamic_slice(
                grid,
                (position[0], position[1], 0),
                (self.view_size, self.view_size, NUM_CHANNELS),
            )
            rotations = jnp.stack([jnp.rot90(view, k, axes=(0, 1)) for k in range(4)])
            return rotations[(direction + 1) % 4]

        views = jax.vmap(egocentric_view)(state.agent.position, state.agent.direction)
        return {
            "view": views,
            "energy": state.agent.energy / MAX_ENERGY,
            "inventory": state.agent.inventory.astype(jnp.float32),
        }

    def observation_space(self) -> dict[str, Space]:
        return {
            "view": Space(
                shape=(self.num_agents, self.view_size, self.view_size, NUM_CHANNELS),
                dtype=jnp.float32,
                low=0.0,
                high=1.0,
            ),
            "energy": Space(
                shape=(self.num_agents,), dtype=jnp.float32, low=0.0, high=1.0
            ),
            "inventory": Space(
                shape=(self.num_agents,), dtype=jnp.float32, low=0.0, high=jnp.inf
            ),
        }

    def action_space(self) -> Space:
        return Space(
            shape=(self.num_agents,), dtype=jnp.int32, low=0, high=NUM_ACTIONS - 1
        )

    def time_limit(self) -> int:
        return self._time_limit

    def render(self, state: DuneSeaState, scale: int = 2) -> Array:
        return render(state, scale)
