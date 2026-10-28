from collections.abc import Sequence

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space
from .generators import Generator
from .render import TILE, render
from .state import (MATURE_ENERGY, MAX_ENERGY, MAX_PLANT_ENERGY, SEED_ENERGY,
                    Agent, Cell, DuneSeaState)

ENERGY_COST = 0.01
CATCH_AMOUNT = 0.5
(
    NOOP,
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    INTERACT,
    BROADCAST,
    PLACE,
) = range(10)
NUM_ACTIONS = 10
DIRECTIONS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
MOVE_OFFSET = jnp.array([0, 0, 2, 3, 1, 0, 0, 0, 0, 0], dtype=jnp.int32)
STEP_SIZE = jnp.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=jnp.int32)
TURN_OFFSET = jnp.array([0, 0, 0, 0, 0, 3, 1, 0, 0, 0], dtype=jnp.int32)
FACING_CHANNELS = 4

BASE_CHANNELS = 7
NUM_CHANNELS = BASE_CHANNELS + FACING_CHANNELS
OBSERVATION_DTYPE = jnp.bfloat16


class DuneSea(Environment):
    thread_safe_render = True

    def __init__(
        self,
        height: int,
        width: int,
        num_agents: int,
        view_size: int,
        time_limit: int,
        generator: Generator,
        teams: Sequence[int],
        eaters: Sequence[bool],
        thieves: Sequence[bool],
        energy_cost: float = ENERGY_COST,
        catch_amount: float = CATCH_AMOUNT,
        tag_from_behind: bool = False,
        growth_steps: int | None = None,
    ):
        self.height = height
        self.width = width
        self.num_agents = num_agents
        self.view_size = view_size
        self._time_limit = time_limit
        self.generator = generator
        self.energy_cost = energy_cost
        self.catch_amount = catch_amount
        self.growth = None if growth_steps is None else (MAX_PLANT_ENERGY - SEED_ENERGY) / growth_steps
        self.tag_from_behind = tag_from_behind
        self.num_actions = NUM_ACTIONS
        self.num_channels = NUM_CHANNELS
        self.teams = jnp.asarray(teams, jnp.int32)
        self.eaters = jnp.asarray(eaters, bool)
        self.thieves = jnp.asarray(thieves, bool)

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

    def move(self, state: DuneSeaState, action: Array, turn_order: Array) -> Array:
        grid_size = jnp.array([self.height, self.width])
        alive = state.agent.energy > 0

        def stride(positions, agent_index):
            position = positions[agent_index]
            agent_action = action[agent_index]
            move_direction = DIRECTIONS[
                (state.agent.direction[agent_index] + MOVE_OFFSET[agent_action]) % 4
            ]
            moving = STEP_SIZE[agent_action] > 0

            def is_free(cell):
                in_bounds = jnp.all((cell >= 0) & (cell < grid_size))
                y, x = jnp.clip(cell, 0, grid_size - 1)
                occupied = jnp.any(jnp.all(positions == cell, axis=-1) & alive)
                return (
                    in_bounds
                    & (state.cells.wall[y, x] == 0)
                    & ~occupied
                )

            first = position + move_direction
            position = jnp.where(moving & is_free(first), first, position)
            return positions.at[agent_index].set(position), None

        positions, _ = jax.lax.scan(stride, state.agent.position, turn_order)
        return positions

    def interact(
        self,
        state: DuneSeaState,
        action: Array,
        positions: Array,
        turn_order: Array,
    ) -> tuple[Cell, Array, Array]:
        grid_size = jnp.array([self.height, self.width])
        alive = state.agent.energy > 0
        indices = jnp.arange(self.num_agents)

        def act(carry, agent_index):
            wall, energy_grid, bush, loss = carry
            facing_position = (
                positions[agent_index] + DIRECTIONS[state.agent.direction[agent_index]]
            )
            in_bounds = jnp.all((facing_position >= 0) & (facing_position < grid_size))
            y, x = jnp.clip(facing_position, 0, grid_size - 1)
            interacting = (action[agent_index] == INTERACT) & in_bounds

            at_facing = (
                alive
                & (indices != agent_index)
                & jnp.all(positions == facing_position, axis=-1)
            )
            victim = jnp.argmax(at_facing)
            remaining = state.agent.energy[victim] - loss[victim]
            can_steal = (
                interacting
                & (self.catch_amount > 0)
                & self.thieves[agent_index]
                & at_facing.any()
                & (remaining > 0)
                & (
                    state.agent.direction[victim] == state.agent.direction[agent_index]
                    if self.tag_from_behind
                    else True
                )
            )
            stolen = jnp.where(
                can_steal, jnp.minimum(self.catch_amount, remaining), 0.0
            )

            can_mine = interacting & ~can_steal & (wall[y, x] > 0)
            can_collect = (
                interacting
                & ~can_steal
                & ~can_mine
                & self.eaters[agent_index]
                & (energy_grid[y, x] > 0)
            )
            can_clear = (
                interacting & ~can_steal & ~can_mine & ~can_collect & (bush[y, x] > 0)
            )
            can_place = (
                (action[agent_index] == PLACE)
                & in_bounds
                & (state.agent.inventory[agent_index] > 0)
                & (wall[y, x] == 0)
                & (energy_grid[y, x] == 0)
                & (bush[y, x] == 0)
                & ~at_facing.any()
            )
            gain = stolen + jnp.where(can_collect, energy_grid[y, x], 0.0)
            wall = wall.at[y, x].set(
                jnp.where(can_mine, 0.0, jnp.where(can_place, 1.0, wall[y, x]))
            )
            energy_grid = energy_grid.at[y, x].set(
                jnp.where(can_collect, 0.0, energy_grid[y, x])
            )
            bush = bush.at[y, x].set(jnp.where(can_clear, 0.0, bush[y, x]))
            loss = loss.at[victim].add(stolen)
            return (wall, energy_grid, bush, loss), (can_mine, can_place, gain)

        (wall, energy_grid, bush, loss), (mined, placed, gain) = jax.lax.scan(
            act,
            (
                state.cells.wall,
                state.cells.energy,
                state.cells.bush,
                jnp.zeros(self.num_agents, jnp.float32),
            ),
            turn_order,
        )
        in_agent_order = (
            lambda values: jnp.zeros_like(values).at[turn_order].set(values)
        )
        energy_change = in_agent_order(gain) - loss
        inventory_change = in_agent_order(mined).astype(jnp.int32) - in_agent_order(
            placed
        ).astype(jnp.int32)
        return (
            state.cells.replace(wall=wall, energy=energy_grid, bush=bush),
            energy_change,
            inventory_change,
        )

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        alive = state.agent.energy > 0
        chosen = jnp.asarray(action, jnp.int32)
        action = jnp.where(alive, chosen, NOOP)
        action = jnp.where(action >= self.num_actions, NOOP, action)
        turn_order = jax.random.permutation(key, self.num_agents)
        positions = self.move(state, action, turn_order)
        cells, energy_change, inventory_change = self.interact(
            state, action, positions, turn_order
        )

        if self.growth is not None:
            cells = cells.replace(energy=self.ripen(state.cells.energy, cells.energy))
        energy_change = energy_change + self.drain(state, positions, cells)
        upkeep = jnp.where(alive, self.energy_cost, 0.0)
        energy = jnp.clip(state.agent.energy + energy_change - upkeep, 0.0, MAX_ENERGY)
        agent = Agent(
            position=positions,
            direction=(state.agent.direction + TURN_OFFSET[action]) % 4,
            energy=energy,
            broadcast=action == BROADCAST,
            inventory=state.agent.inventory + inventory_change,
        )
        reward = energy - state.agent.energy
        state = state.replace(cells=cells, agent=agent, time=state.time + 1)
        timestep = Timestep(
            obs=self.observe(state),
            action=chosen,
            reward=reward,
            terminated=state.agent.energy <= 0.0,
            truncated=jnp.full(self.num_agents, state.time >= self._time_limit),
            info={},
        )
        return state, timestep

    def ripen(self, previous: Array, energy: Array) -> Array:
        harvested = (previous > 0) & (energy == 0)
        replanted = harvested & (previous >= MATURE_ENERGY)
        grown = jnp.minimum(energy + self.growth, MAX_PLANT_ENERGY)
        energy = jnp.where(energy > 0, grown, energy)
        return jnp.where(replanted, SEED_ENERGY, energy)

    def drain(self, state: DuneSeaState, positions: Array, cells: Cell) -> Array:
        return jnp.zeros(self.num_agents, jnp.float32)

    def field_of_view(self, state: DuneSeaState) -> Array:
        radius = self.view_size // 2
        padded = jnp.pad(state.cells.wall, radius, constant_values=1.0)
        alive = state.agent.energy > 0

        rows, columns = jnp.meshgrid(
            jnp.arange(self.height), jnp.arange(self.width), indexing="ij"
        )

        def seen_by(position):
            window = self.visibility(padded, position)
            relative_row, relative_column = rows - position[0], columns - position[1]
            inside = (jnp.abs(relative_row) <= radius) & (
                jnp.abs(relative_column) <= radius
            )
            row = jnp.clip(relative_row + radius, 0, 2 * radius)
            column = jnp.clip(relative_column + radius, 0, 2 * radius)
            return inside & window[row, column]

        return jax.vmap(seen_by)(state.agent.position) & alive[:, None, None]

    def visibility(self, padded_wall: Array, position: Array) -> Array:
        radius = self.view_size // 2
        offsets = jnp.arange(-radius, radius + 1)
        dy, dx = jnp.meshgrid(offsets, offsets, indexing="ij")
        fractions = jnp.arange(1, 2 * radius + 1) / (2 * radius + 1)
        sample_y = jnp.round(position[0] + dy[..., None] * fractions).astype(jnp.int32)
        sample_x = jnp.round(position[1] + dx[..., None] * fractions).astype(jnp.int32)
        at_target = (sample_y == position[0] + dy[..., None]) & (
            sample_x == position[1] + dx[..., None]
        )
        at_origin = (sample_y == position[0]) & (sample_x == position[1])
        solid = padded_wall[sample_y + radius, sample_x + radius] > 0
        return ~jnp.any(solid & ~at_target & ~at_origin, axis=-1)

    def observe(self, state: DuneSeaState) -> dict[str, Array]:
        radius = self.view_size // 2
        size = (self.view_size, self.view_size)
        alive = state.agent.energy > 0
        in_bush = (
            state.cells.bush[state.agent.position[:, 0], state.agent.position[:, 1]] > 0
        )
        padded_wall = jnp.pad(state.cells.wall, radius, constant_values=1.0)
        shared = [
            padded_wall,
            *(
                jnp.pad(plane, radius)
                for plane in (state.cells.energy, state.cells.bush)
            ),
        ]
        local = jnp.arange(self.view_size)

        def egocentric_view(position, direction, team):
            window = lambda plane: jax.lax.dynamic_slice(
                plane, (position[0], position[1]), size
            )
            nearby = jnp.abs(state.agent.position - position).max(-1) <= 1
            shown = alive & (~in_bush | nearby)
            relative = state.agent.position - position + radius
            at = (
                (local[None, :, None] == relative[:, 0, None, None])
                & (local[None, None, :] == relative[:, 1, None, None])
            ).astype(jnp.float32)
            present = lambda mask: jnp.tensordot(mask.astype(jnp.float32), at, axes=1)
            wall, energy, bush = (window(plane) for plane in shared)
            visible = self.visibility(padded_wall, position)
            view = jnp.stack(
                [
                    wall,
                    energy,
                    present(shown & (self.teams == team)),
                    present(shown & (self.teams != team)),
                    present(shown & state.agent.broadcast),
                    bush,
                ],
                axis=-1,
            )
            view = jnp.concatenate(
                [view * visible[..., None], visible[..., None]], axis=-1
            )
            turned = (state.agent.direction - direction) % 4
            facing = jnp.stack(
                [present(shown & (turned == k)) for k in range(FACING_CHANNELS)],
                axis=-1,
            )
            view = jnp.concatenate([view, facing * visible[..., None]], axis=-1)
            rotations = jnp.stack([jnp.rot90(view, k, axes=(0, 1)) for k in range(4)])
            return rotations[(direction + 1) % 4]

        views = jax.vmap(egocentric_view)(
            state.agent.position, state.agent.direction, self.teams
        )
        observation = {
            "view": views,
            "energy": state.agent.energy / MAX_ENERGY,
            "inventory": state.agent.inventory,
            "team": self.teams,
        }
        return jax.tree.map(lambda leaf: leaf.astype(OBSERVATION_DTYPE), observation)

    def observation_space(self) -> dict[str, Space]:
        return {
            "view": Space(
                shape=(
                    self.num_agents,
                    self.view_size,
                    self.view_size,
                    self.num_channels,
                ),
                dtype=OBSERVATION_DTYPE,
                low=0.0,
                high=1.0,
            ),
            "energy": Space(
                shape=(self.num_agents,), dtype=OBSERVATION_DTYPE, low=0.0, high=1.0
            ),
            "inventory": Space(
                shape=(self.num_agents,), dtype=OBSERVATION_DTYPE, low=0.0, high=jnp.inf
            ),
            "team": Space(
                shape=(self.num_agents,), dtype=OBSERVATION_DTYPE, low=0.0, high=1.0
            ),
        }

    def action_space(self) -> Space:
        return Space(
            shape=(self.num_agents,), dtype=jnp.int32, low=0, high=self.num_actions - 1
        )

    def time_limit(self) -> int:
        return self._time_limit

    def render(
        self,
        state: DuneSeaState,
        scale: int = 1,
        storm: float = 1.0,
        previous: DuneSeaState | None = None,
        blend: float = 1.0,
    ) -> Array:
        offsets = None
        if previous is not None:
            delta = previous.agent.position - state.agent.position
            teleported = jnp.abs(delta).max(-1, keepdims=True) > 2
            offsets = jnp.round(
                jnp.where(teleported, 0, delta) * TILE * (1.0 - blend)
            ).astype(jnp.int32)
        return render(state, scale, self.teams, self.field_of_view(state), storm, offsets, blend)
