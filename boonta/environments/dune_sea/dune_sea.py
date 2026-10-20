from collections.abc import Sequence

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space
from .generators import Generator, uniform_generator
from .render import TILE, render
from .state import MAX_ENERGY, Agent, Cell, DuneSeaState

ENERGY_COST = 0.01
CATCH_AMOUNT = 0.5
NOOP, FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT, INTERACT, BROADCAST, PLACE, DASH = range(11)
NUM_ACTIONS = 10
DIRECTIONS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
MOVE_OFFSET = jnp.array([0, 0, 2, 3, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
STEP_SIZE = jnp.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2], dtype=jnp.int32)
TURN_OFFSET = jnp.array([0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0], dtype=jnp.int32)
FACING_CHANNELS = 4

NUM_CHANNELS = 8


class DuneSea(Environment):

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        num_agents: int = 1,
        view_size: int = 7,
        time_limit: int = 256,
        generator: Generator = uniform_generator,
        teams: Sequence[int] | None = None,
        eaters: Sequence[bool] | None = None,
        thieves: Sequence[bool] | None = None,
        energy_cost: float = ENERGY_COST,
        catch_amount: float = CATCH_AMOUNT,
        sight_drain: float = 0.0,
        dash_cost: float | None = None,
        tag_from_behind: bool = False,
        observe_facing: bool = False,
        observe_oasis: bool = False,
        beacon_slots: int = 0,
    ):
        self.height = height
        self.width = width
        self.num_agents = num_agents
        self.view_size = view_size
        self._time_limit = time_limit
        self.generator = generator
        self.energy_cost = energy_cost
        self.catch_amount = catch_amount
        self.sight_drain = sight_drain
        self.dash_cost = dash_cost
        self.tag_from_behind = tag_from_behind
        self.observe_facing = observe_facing
        self.observe_oasis = observe_oasis
        self.beacon_slots = beacon_slots
        self.num_actions = NUM_ACTIONS + (dash_cost is not None)
        self.num_channels = NUM_CHANNELS + (FACING_CHANNELS if observe_facing else 0) + (1 if observe_oasis else 0)
        self.teams = jnp.asarray(teams if teams is not None else [0] * num_agents, jnp.int32)
        self.eaters = jnp.asarray(eaters if eaters is not None else [True] * num_agents, bool)
        self.thieves = jnp.asarray(thieves if thieves is not None else [False] * num_agents, bool)

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

        def move_one(positions, agent_index):
            position = positions[agent_index]
            agent_action = action[agent_index]
            move_direction = DIRECTIONS[
                (state.agent.direction[agent_index] + MOVE_OFFSET[agent_action]) % 4
            ]
            step_size = STEP_SIZE[agent_action]

            def is_free(cell):
                in_bounds = jnp.all((cell >= 0) & (cell < grid_size))
                y, x = jnp.clip(cell, 0, grid_size - 1)
                occupied = jnp.any(jnp.all(positions == cell, axis=-1) & alive)
                return in_bounds & (state.cells.wall[y, x] == 0) & ~occupied

            first = position + move_direction
            second = position + 2 * move_direction
            can_first = (step_size >= 1) & is_free(first)
            can_second = can_first & (step_size == 2) & is_free(second)
            position = jnp.where(can_second, second, jnp.where(can_first, first, position))
            return positions.at[agent_index].set(position), None

        positions, _ = jax.lax.scan(move_one, state.agent.position, turn_order)
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

        def interact_one(carry, agent_index):
            wall, energy_grid, bush, loss = carry
            facing_position = positions[agent_index] + DIRECTIONS[state.agent.direction[agent_index]]
            in_bounds = jnp.all((facing_position >= 0) & (facing_position < grid_size))
            y, x = jnp.clip(facing_position, 0, grid_size - 1)
            interacting = (action[agent_index] == INTERACT) & in_bounds
            striking = interacting | ((action[agent_index] == DASH) & in_bounds)

            at_facing = alive & (indices != agent_index) & jnp.all(positions == facing_position, axis=-1)
            victim = jnp.argmax(at_facing)
            remaining = state.agent.energy[victim] - loss[victim]
            can_steal = (
                striking
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
            stolen = jnp.where(can_steal, jnp.minimum(self.catch_amount, remaining), 0.0)

            can_mine = interacting & ~can_steal & (wall[y, x] > 0)
            can_collect = (
                interacting
                & ~can_steal
                & ~can_mine
                & self.eaters[agent_index]
                & (energy_grid[y, x] > 0)
            )
            can_clear = (
                interacting
                & ~can_steal
                & ~can_mine
                & ~can_collect
                & (bush[y, x] > 0)
            )
            can_place = (
                (action[agent_index] == PLACE)
                & in_bounds
                & (state.agent.inventory[agent_index] > 0)
                & (wall[y, x] == 0)
                & (energy_grid[y, x] == 0)
                & (state.cells.goal[y, x] == 0)
                & (bush[y, x] == 0)
                & ~at_facing.any()
            )
            gain = stolen + jnp.where(can_collect, energy_grid[y, x], 0.0)
            wall = wall.at[y, x].set(jnp.where(can_mine, 0.0, jnp.where(can_place, 1.0, wall[y, x])))
            energy_grid = energy_grid.at[y, x].set(jnp.where(can_collect, 0.0, energy_grid[y, x]))
            bush = bush.at[y, x].set(jnp.where(can_clear, 0.0, bush[y, x]))
            loss = loss.at[victim].add(stolen)
            return (wall, energy_grid, bush, loss), (can_mine, can_place, gain)

        (wall, energy_grid, bush, loss), (mined, placed, gain) = jax.lax.scan(
            interact_one,
            (state.cells.wall, state.cells.energy, state.cells.bush, jnp.zeros(self.num_agents, jnp.float32)),
            turn_order,
        )
        in_agent_order = lambda values: jnp.zeros_like(values).at[turn_order].set(values)
        energy_change = in_agent_order(gain) - loss
        inventory_change = in_agent_order(mined).astype(jnp.int32) - in_agent_order(placed).astype(jnp.int32)
        return state.cells.replace(wall=wall, energy=energy_grid, bush=bush), energy_change, inventory_change

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        alive = state.agent.energy > 0
        chosen = jnp.asarray(action, jnp.int32)
        action = jnp.where(alive, chosen, NOOP)
        action = jnp.where(action >= self.num_actions, NOOP, action)
        turn_order = jax.random.permutation(key, self.num_agents)
        positions = self.move(state, action, turn_order)
        cells, energy_change, inventory_change = self.interact(state, action, positions, turn_order)

        if self.sight_drain > 0:
            energy_change = energy_change + self.sight(state, positions, cells.wall, cells.bush)
        dashing = jnp.where(action == DASH, self.dash_cost or 0.0, 0.0)
        upkeep = jnp.where(alive, self.energy_cost + dashing, 0.0)
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

    def sight(self, state: DuneSeaState, positions: Array, wall: Array, bush: Array) -> Array:
        radius = self.view_size // 2
        alive = state.agent.energy > 0
        padded = jnp.pad(wall, radius, constant_values=1.0)

        def seen_from(position):
            visible = self.visibility(padded, position)
            relative = positions - position
            inside = (jnp.abs(relative) <= radius).all(-1)
            rows = jnp.clip(relative[:, 0] + radius, 0, 2 * radius)
            columns = jnp.clip(relative[:, 1] + radius, 0, 2 * radius)
            hidden = in_bush & (jnp.abs(relative).max(-1) > 1)
            return inside & visible[rows, columns] & ~hidden

        in_bush = bush[positions[:, 0], positions[:, 1]] > 0
        sees = jax.vmap(seen_from)(positions)
        sees = sees & (self.thieves & alive)[:, None] & (~self.thieves & alive)[None, :]
        watchers = sees.sum(0)
        drained = jnp.minimum(self.sight_drain, state.agent.energy / jnp.maximum(watchers, 1))
        return (sees * drained[None, :]).sum(1) - drained * watchers

    def fov(self, state: DuneSeaState) -> Array:
        radius = self.view_size // 2
        padded = jnp.pad(state.cells.wall, radius, constant_values=1.0)
        alive = state.agent.energy > 0

        rows, columns = jnp.meshgrid(jnp.arange(self.height), jnp.arange(self.width), indexing="ij")

        def seen_by(position):
            window = self.visibility(padded, position)
            relative_row, relative_column = rows - position[0], columns - position[1]
            inside = (jnp.abs(relative_row) <= radius) & (jnp.abs(relative_column) <= radius)
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
        at_target = (sample_y == position[0] + dy[..., None]) & (sample_x == position[1] + dx[..., None])
        at_origin = (sample_y == position[0]) & (sample_x == position[1])
        solid = padded_wall[sample_y + radius, sample_x + radius] > 0
        return ~jnp.any(solid & ~at_target & ~at_origin, axis=-1)

    def observe(self, state: DuneSeaState) -> dict[str, Array]:
        radius = self.view_size // 2
        size = (self.view_size, self.view_size)
        alive = state.agent.energy > 0
        in_bush = state.cells.bush[state.agent.position[:, 0], state.agent.position[:, 1]] > 0
        padded_wall = jnp.pad(state.cells.wall, radius, constant_values=1.0)
        shared = [
            padded_wall,
            *(jnp.pad(plane, radius) for plane in (state.cells.goal, state.cells.energy, state.cells.bush)),
        ]
        local = jnp.arange(self.view_size)

        def egocentric_view(position, direction, team):
            window = lambda plane: jax.lax.dynamic_slice(plane, (position[0], position[1]), size)
            nearby = jnp.abs(state.agent.position - position).max(-1) <= 1
            shown = alive & (~in_bush | nearby)
            relative = state.agent.position - position + radius
            at = (
                (local[None, :, None] == relative[:, 0, None, None])
                & (local[None, None, :] == relative[:, 1, None, None])
            ).astype(jnp.float32)
            present = lambda mask: jnp.tensordot(mask.astype(jnp.float32), at, axes=1)
            wall, goal, energy, bush = (window(plane) for plane in shared)
            visible = self.visibility(padded_wall, position)
            view = jnp.stack(
                [
                    wall,
                    goal,
                    energy,
                    present(shown & (self.teams == team)),
                    present(shown & (self.teams != team)),
                    present(shown & state.agent.broadcast),
                    bush,
                ],
                axis=-1,
            )
            view = jnp.concatenate([view * visible[..., None], visible[..., None]], axis=-1)
            if self.observe_facing:
                turned = (state.agent.direction - direction) % 4
                facing = jnp.stack([present(shown & (turned == k)) for k in range(FACING_CHANNELS)], axis=-1)
                view = jnp.concatenate([view, facing * visible[..., None]], axis=-1)
            if self.observe_oasis:
                water = window(jnp.pad(jnp.clip(state.cells.oasis, 0.0, 1.0), radius))
                view = jnp.concatenate([view, (water * visible)[..., None]], axis=-1)
            rotations = jnp.stack([jnp.rot90(view, k, axes=(0, 1)) for k in range(4)])
            return rotations[(direction + 1) % 4]

        views = jax.vmap(egocentric_view)(state.agent.position, state.agent.direction, self.teams)
        observation = {
            "view": views,
            "energy": state.agent.energy / MAX_ENERGY,
            "inventory": state.agent.inventory.astype(jnp.float32),
            "team": self.teams.astype(jnp.float32),
        }
        if self.beacon_slots > 0:
            observation["beacons"] = self.beacons(state)
        return observation

    def beacons(self, state: DuneSeaState) -> Array:
        alive = state.agent.energy > 0
        scale = float(max(self.height, self.width))

        def nearest_calls(position, direction, index):
            relative = state.agent.position - position
            distance = jnp.abs(relative).sum(-1)
            calling = alive & state.agent.broadcast & (jnp.arange(self.num_agents) != index)
            score, nearest = jax.lax.top_k(
                jnp.where(calling, -distance.astype(jnp.float32), -jnp.inf), min(self.beacon_slots, self.num_agents)
            )
            present = score > -jnp.inf
            offset = relative[nearest]
            ahead = (offset * DIRECTIONS[direction]).sum(-1)
            right = (offset * DIRECTIONS[(direction + 1) % 4]).sum(-1)
            found = present.astype(jnp.float32)
            slots = jnp.stack([ahead / scale * found, right / scale * found, found], axis=-1)
            return jnp.pad(slots, ((0, self.beacon_slots - slots.shape[0]), (0, 0)))

        return jax.vmap(nearest_calls)(state.agent.position, state.agent.direction, jnp.arange(self.num_agents))

    def observation_space(self) -> dict[str, Space]:
        return {
            "view": Space(
                shape=(self.num_agents, self.view_size, self.view_size, self.num_channels),
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
            "team": Space(shape=(self.num_agents,), dtype=jnp.float32, low=0.0, high=1.0),
            **(
                {"beacons": Space(shape=(self.num_agents, self.beacon_slots, 3), dtype=jnp.float32, low=-1.0, high=1.0)}
                if self.beacon_slots > 0
                else {}
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
        scale: int = 2,
        show_fov: bool = False,
        storm: float = 1.0,
        previous: DuneSeaState | None = None,
        blend: float = 1.0,
    ) -> Array:
        offsets = None
        if previous is not None:
            delta = previous.agent.position - state.agent.position
            teleported = jnp.abs(delta).max(-1, keepdims=True) > 2
            offsets = jnp.round(jnp.where(teleported, 0, delta) * TILE * (1.0 - blend)).astype(jnp.int32)
        fov = self.fov(state) if show_fov else None
        return render(state, scale, self.teams, fov, storm, offsets, blend)
