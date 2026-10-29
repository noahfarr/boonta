import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space
from .generators import Generator, cave_generator
from .render import TILE, render
from .state import (GIFT, GROWTH_STEPS, MATURE_ENERGY, MAX_ENERGY,
                    MAX_PLANT_ENERGY, MIN_GIFT, SEED_ENERGY, STARTING_ENERGY,
                    THRESHOLD, Agent, Cell, JundlandState)

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
ACTION_COST = jnp.array([1.0, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.5, 2.0], dtype=jnp.float32)
MINE_COST = 3.0
TYPICAL_COST = 1.5
FACING_CHANNELS = 4

BASE_CHANNELS = 7
NUM_CHANNELS = BASE_CHANNELS + FACING_CHANNELS
OBSERVATION_DTYPE = jnp.bfloat16

CARRYING_CAPACITY = 2.0
INITIAL_POPULATION = 32
MUTATION = 0.05
CATCH_AMOUNT = 75.0
SEED_FIELD = 5
MAX_SEEDS = 3
SCENT_SPREAD = 0.2
SCENT_DECAY = 0.003
SCENT_WARMUP = 300
CAMERA = 32
EVALUATION_STEPS = 1000


def sample(plane: Array, cells: Array, outside: float) -> Array:
    plane = jnp.asarray(plane)
    height, width = plane.shape
    rows, columns = cells[..., 0], cells[..., 1]
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    return jnp.where(inside, plane[jnp.clip(rows, 0, height - 1), jnp.clip(columns, 0, width - 1)], outside)


def orientations(view_size: int) -> tuple[np.ndarray, np.ndarray]:
    radius = view_size // 2
    pixels = np.arange(view_size**2).reshape(view_size, view_size)
    sources = np.stack([np.rot90(pixels, k) for k in range(4)])
    offsets = np.stack(np.divmod(sources, view_size), -1) - radius
    inverse = np.zeros((4, view_size, view_size, 2), np.int32)
    for k in range(4):
        rows, columns = np.divmod(sources[k], view_size)
        inverse[k, rows, columns] = np.stack(np.indices((view_size, view_size)), -1)
    return offsets.astype(np.int32), inverse


class Jundland(Environment):
    thread_safe_render = True

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        num_agents: int = 128,
        view_size: int = 7,
        time_limit: int | None = None,
        generator: Generator = cave_generator,
        carrying_capacity: float = CARRYING_CAPACITY,
        initial_population: int = INITIAL_POPULATION,
        starting_energy: float = STARTING_ENERGY,
        threshold: float = THRESHOLD,
        gift: float = GIFT,
        mutation: float = MUTATION,
        catch_amount: float = CATCH_AMOUNT,
        seed_field: int = SEED_FIELD,
        max_seeds: int = MAX_SEEDS,
        initial_dirt: int = 0,
    ):
        self.height = height
        self.width = width
        self.num_agents = num_agents
        self.view_size = view_size
        self._time_limit = time_limit
        self.generator = generator
        self.growth = (MAX_PLANT_ENERGY - SEED_ENERGY) / GROWTH_STEPS
        self.num_plants = round(carrying_capacity * num_agents * TYPICAL_COST / self.growth)
        self.initial_population = min(initial_population, num_agents)
        self.starting_energy = starting_energy
        self.threshold = threshold
        self.gift = gift
        self.mutation = mutation
        self.catch_amount = catch_amount
        self.seed_field = seed_field
        self.max_seeds = max_seeds
        self.initial_dirt = initial_dirt
        self.num_actions = NUM_ACTIONS
        self.num_channels = NUM_CHANNELS
        self.teams = jnp.arange(num_agents, dtype=jnp.int32)
        offsets, pixels = orientations(view_size)
        self.offsets = jnp.asarray(offsets)
        self.pixels = jnp.asarray(pixels)

    def init(self, key: Key) -> tuple[JundlandState, Timestep]:
        cells, agent = self.generator(key, self.height, self.width, self.num_agents, self.num_plants)
        cells = cells.replace(
            scent=jax.lax.fori_loop(0, SCENT_WARMUP, lambda _, scent: self.diffuse(scent, cells), cells.scent)
        )
        founders = jnp.arange(self.num_agents) < self.initial_population
        agent = agent.replace(
            energy=jnp.where(founders, self.starting_energy, 0.0).astype(jnp.float32),
            inventory=jnp.full(self.num_agents, self.initial_dirt, jnp.int32),
            threshold=jnp.full(self.num_agents, self.threshold, jnp.float32),
            gift=jnp.full(self.num_agents, self.gift, jnp.float32),
            age=jnp.zeros(self.num_agents, jnp.int32),
        )
        state = JundlandState(cells=cells, agent=agent, time=jnp.int32(0))
        timestep = Timestep(
            obs=self.observe(state),
            action=jnp.zeros(self.num_agents, jnp.int32),
            reward=jnp.zeros(self.num_agents, jnp.float32),
            terminated=jnp.zeros(self.num_agents, bool),
            truncated=jnp.zeros(self.num_agents, bool),
            info=self.tally(
                state,
                state.agent.energy > 0,
                jnp.zeros(self.num_agents, bool),
                jnp.zeros(self.num_agents, bool),
                jnp.zeros(self.num_agents, jnp.int32),
                jnp.zeros(self.num_agents, jnp.float32),
                jnp.zeros(self.num_agents, jnp.float32),
                jnp.zeros(self.num_agents, bool),
            ),
        )
        return state, timestep

    def move(
        self, state: JundlandState, action: Array, turn_order: Array
    ) -> tuple[Array, Array, Array]:
        alive = state.agent.energy > 0
        indices = jnp.arange(self.num_agents)
        rank = jnp.zeros(self.num_agents, jnp.int32).at[turn_order].set(indices)
        start = state.agent.position
        moving = STEP_SIZE[action] > 0
        target = start + DIRECTIONS[(state.agent.direction + MOVE_OFFSET[action]) % 4]
        in_bounds = ((target >= 0) & (target < jnp.array([self.height, self.width]))).all(-1)
        open_ground = in_bounds & (sample(state.cells.wall, target, 1.0) == 0)
        earlier = rank[None, :] < rank[:, None]
        others = alive[None, :] & (indices[:, None] != indices[None, :])

        def resolve(positions):
            seen = jnp.where(earlier[..., None], positions[None], start[None])
            occupied = jnp.all(seen == target[:, None], -1) & others
            free = open_ground & ~occupied.any(-1)
            return jnp.where((moving & free)[:, None], target, start), seen, occupied

        def settle(carry):
            positions, _ = carry
            settled, *_ = resolve(positions)
            return settled, (settled != positions).any()

        positions, _ = jax.lax.while_loop(lambda carry: carry[1], settle, (start, jnp.array(True)))
        _, seen, occupied = resolve(positions)

        victim = jnp.argmax(occupied, -1)
        gazed_at = seen[indices, victim] + DIRECTIONS[state.agent.direction[victim]]
        attempt = (
            moving
            & in_bounds
            & occupied.any(-1)
            & (self.catch_amount > 0)
            & ~jnp.all(gazed_at == start, -1)
        )
        before = (earlier & attempt[None, :] & (victim[None, :] == victim[:, None])).sum(-1)
        stolen = jnp.where(
            attempt, jnp.clip(state.agent.energy[victim] - before * self.catch_amount, 0.0, self.catch_amount), 0.0
        )
        loss = jnp.zeros(self.num_agents, jnp.float32).at[victim].add(stolen)
        return positions, stolen, loss

    def interact(
        self,
        state: JundlandState,
        action: Array,
        positions: Array,
        turn_order: Array,
    ) -> tuple[Cell, Array, Array, Array]:
        alive = state.agent.energy > 0
        indices = jnp.arange(self.num_agents)
        rank = jnp.zeros(self.num_agents, jnp.int32).at[turn_order].set(indices)
        facing = positions + DIRECTIONS[state.agent.direction]
        in_bounds = ((facing >= 0) & (facing < jnp.array([self.height, self.width]))).all(-1)
        y, x = jnp.where(in_bounds[:, None], facing, 0).T
        shared = jnp.all(facing[:, None] == facing[None], -1) & in_bounds[:, None] & in_bounds[None]
        queue = (shared & (rank[None, :] < rank[:, None])).sum(-1)
        blocked = (jnp.all(positions[None] == facing[:, None], -1) & alive & (indices[:, None] != indices)).any(-1)
        planes = (state.cells.wall, state.cells.energy, state.cells.bush)
        faced = jnp.stack([plane[y, x] for plane in planes], -1)
        interacting = (action == INTERACT) & in_bounds
        placing = (action == PLACE) & in_bounds & (state.agent.inventory > 0) & ~blocked
        mined = placed = jnp.zeros(self.num_agents, bool)
        gain = jnp.zeros(self.num_agents, jnp.float32)

        for turn in range(len(DIRECTIONS)):
            acting = queue == turn
            wall, energy, bush = faced[:, 0], faced[:, 1], faced[:, 2]
            can_mine = acting & interacting & (wall > 0)
            can_collect = acting & interacting & ~can_mine & (energy > 0)
            can_clear = acting & interacting & ~can_mine & ~can_collect & (bush > 0)
            can_place = acting & placing & (wall == 0) & (energy == 0) & (bush == 0)
            after = jnp.stack(
                [
                    jnp.where(can_mine, 0.0, jnp.where(can_place, 1.0, wall)),
                    jnp.where(can_collect, 0.0, energy),
                    jnp.where(can_clear, 0.0, bush),
                ],
                -1,
            )
            source = jnp.argmax(shared & acting[None, :], -1)
            touched = (shared & acting[None, :]).any(-1)
            faced = jnp.where(touched[:, None], after[source], faced)
            mined, placed = mined | can_mine, placed | can_place
            gain = gain + jnp.where(can_collect, energy, 0.0)

        rows = jnp.where(in_bounds, y, self.height)
        wall, energy, bush = (
            plane.at[rows, x].set(faced[:, channel], mode="drop") for channel, plane in enumerate(planes)
        )
        return state.cells.replace(wall=wall, energy=energy, bush=bush), gain, mined, placed

    def grow(self, energy: Array) -> Array:
        return jnp.where(energy > 0, jnp.minimum(energy + self.growth, MAX_PLANT_ENERGY), energy)

    def sow(self, key: Key, cells: Cell, positions: Array, alive: Array, origins: Array, sown: Array) -> Array:
        grid_size = jnp.array([self.height, self.width])
        radius = self.seed_field // 2
        span = jnp.arange(-radius, radius + 1)
        field = jnp.stack(jnp.meshgrid(span, span, indexing="ij"), -1).reshape(-1, 2)

        count_key, spot_key = jax.random.split(key)
        count = jax.random.randint(count_key, (self.num_agents, 1), 0, self.max_seeds + 1)
        remaining = field.shape[0] - jnp.arange(self.max_seeds)
        draws = jax.random.randint(spot_key, (self.num_agents, self.max_seeds), 0, remaining)

        spots = []
        for index in range(self.max_seeds):
            spot = draws[:, index]
            for _ in range(index):
                spot = draws[:, index] + sum(earlier <= spot for earlier in spots)
            spots.append(spot)
        spots = jnp.stack(spots, -1)
        planted = sown[:, None] & (jnp.arange(self.max_seeds) < count)
        targets = (origins[:, None] + field[spots]).reshape(-1, 2)
        planted = planted.reshape(-1)
        in_bounds = jnp.all((targets >= 0) & (targets < grid_size), axis=-1)
        y, x = jnp.clip(targets, 0, grid_size - 1).T
        rooted = (
            planted
            & in_bounds
            & (cells.wall[y, x] == 0)
            & (cells.energy[y, x] == 0)
            & ~(jnp.all(targets[:, None] == positions[None], -1) & alive).any(-1)
        )
        return cells.energy.at[y, x].max(jnp.where(rooted, SEED_ENERGY, 0.0))

    def diffuse(self, scent: Array, cells: Cell) -> Array:
        open_ground = cells.wall == 0
        padded = jnp.pad(scent, 1)
        reachable = jnp.pad(open_ground, 1)
        neighbours = ((slice(2, None), slice(1, -1)), (slice(None, -2), slice(1, -1)), (slice(1, -1), slice(2, None)), (slice(1, -1), slice(None, -2)))
        flow = sum(jnp.where(reachable[window], padded[window] - scent, 0.0) for window in neighbours)
        scent = (1.0 - SCENT_DECAY) * (scent + SCENT_SPREAD * flow) + cells.energy / MAX_PLANT_ENERGY
        return jnp.where(open_ground, scent, 0.0)

    def smell(self, state: JundlandState) -> Array:
        around = state.agent.direction[:, None] + jnp.arange(len(DIRECTIONS))
        cells = jnp.concatenate([state.agent.position[:, None], state.agent.position[:, None] + DIRECTIONS[around % 4]], 1)
        return 1.0 - jnp.exp(-sample(state.cells.scent, cells, 0.0))

    def tally(
        self,
        state: JundlandState,
        acting: Array,
        died: Array,
        born: Array,
        action: Array,
        harvest: Array,
        taken: Array,
        placed: Array,
    ) -> dict[str, Array]:
        alive = state.agent.energy > 0
        survivors = alive.sum()
        actors = acting.sum()
        harvested = harvest > 0
        ratio = lambda part, whole, valid: jnp.where(valid, part.sum() / jnp.maximum(whole, 1), jnp.nan)
        return {
            "num_alive": survivors.astype(jnp.float32),
            "deaths": died.mean(),
            "births": born.mean(),
            "lifespan": ratio(state.agent.age * died, died.sum(), died.any()),
            "threshold": ratio(state.agent.threshold * alive, survivors, survivors > 0),
            "gift": ratio(state.agent.gift * alive, survivors, survivors > 0),
            "energy": ratio(state.agent.energy * alive, survivors, survivors > 0),
            "plants": (state.cells.energy > 0).sum() / self.num_agents,
            "harvests": ratio(harvested, actors, actors > 0),
            "mature_harvests": ratio(harvest >= MATURE_ENERGY, harvested.sum(), harvested.any()),
            "thefts": ratio(taken > 0, actors, actors > 0),
            "broadcasts": ratio(acting & (action == BROADCAST), actors, actors > 0),
            "placements": ratio(placed, actors, actors > 0),
        }

    def step(
        self, key: Key, state: JundlandState, action: Array
    ) -> tuple[JundlandState, Timestep]:
        order_key, seed_key, birth_key = jax.random.split(key, 3)
        alive = state.agent.energy > 0
        chosen = jnp.asarray(action, jnp.int32)
        action = jnp.where(alive, chosen, NOOP)
        action = jnp.where(action >= self.num_actions, NOOP, action)
        turn_order = jax.random.permutation(order_key, self.num_agents)
        positions, taken, lost = self.move(state, action, turn_order)
        cells, harvest, mined, placed = self.interact(state, action, positions, turn_order)

        facing = positions + DIRECTIONS[state.agent.direction]
        cells = cells.replace(energy=self.grow(cells.energy))
        cells = cells.replace(energy=self.sow(seed_key, cells, positions, alive, facing, harvest >= MATURE_ENERGY))
        cells = cells.replace(scent=self.diffuse(cells.scent, cells))

        cost = ACTION_COST[action] + jnp.where(mined, MINE_COST - ACTION_COST[INTERACT], 0.0)
        energy = state.agent.energy + harvest + taken - lost - jnp.where(alive, cost, 0.0)
        energy = jnp.clip(energy, 0.0, MAX_ENERGY)
        dead = energy <= 0.0
        agent = Agent(
            position=positions,
            direction=(state.agent.direction + TURN_OFFSET[action]) % 4,
            energy=energy,
            broadcast=action == BROADCAST,
            inventory=state.agent.inventory + mined.astype(jnp.int32) - placed.astype(jnp.int32),
            threshold=state.agent.threshold,
            gift=state.agent.gift,
            age=state.agent.age + alive.astype(jnp.int32),
        )
        died = alive & dead
        lifespans = agent.age
        agent, born = self.reproduce(birth_key, agent, cells.wall)
        state = state.replace(cells=cells, agent=agent, time=state.time + 1)
        timestep = Timestep(
            obs=self.observe(state),
            action=chosen,
            reward=(~dead).astype(jnp.float32),
            terminated=dead,
            truncated=self.truncate(state, dead),
            info=self.tally(state.replace(agent=state.agent.replace(age=lifespans)), alive, died, born, action, harvest, taken, placed),
        )
        return state, timestep

    def truncate(self, state: JundlandState, dead: Array) -> Array:
        out_of_time = self._time_limit is not None and state.time >= self._time_limit
        return ~dead & out_of_time

    def reproduce(self, key: Key, agent: Agent, wall: Array) -> tuple[Agent, Array]:
        side_key, direction_key, threshold_key, gift_key = jax.random.split(key, 4)
        indices = jnp.arange(self.num_agents)
        alive = agent.energy > 0
        parents = alive & (agent.energy > agent.threshold)
        rank = jnp.cumsum(parents) - 1
        empty = jnp.nonzero(~alive, size=self.num_agents, fill_value=self.num_agents)[0]
        slot = jnp.where(parents, empty[jnp.clip(rank, 0, self.num_agents - 1)], self.num_agents)

        sides = jnp.argsort(jax.random.uniform(side_key, (self.num_agents, len(DIRECTIONS))), -1)
        cells = agent.position[:, None] + DIRECTIONS[sides]
        occupied = (jnp.all(cells[:, :, None] == agent.position[None, None], -1) & alive).any(-1)
        free = (sample(wall, cells, 1.0) == 0) & ~occupied
        spot = cells[indices, jnp.argmax(free, -1)]
        ready = parents & free.any(-1) & (slot < self.num_agents)
        clash = (jnp.all(spot[:, None] == spot[None], -1) & ready[None] & (indices[None] < indices[:, None])).any(-1)
        breeding = ready & ~clash

        gift = jnp.clip(
            agent.gift * (1.0 + self.mutation * jax.random.normal(gift_key, (self.num_agents,))), MIN_GIFT, MAX_ENERGY - 1.0
        )
        threshold = jnp.clip(
            agent.threshold * (1.0 + self.mutation * jax.random.normal(threshold_key, (self.num_agents,))),
            gift + 1.0,
            MAX_ENERGY,
        )
        target = jnp.where(breeding, slot, self.num_agents)
        place = lambda values, child: values.at[target].set(child, mode="drop")
        energy = agent.energy - jnp.where(breeding, agent.gift, 0.0)
        agent = Agent(
            position=place(agent.position, spot),
            direction=place(agent.direction, jax.random.randint(direction_key, (self.num_agents,), 0, 4)),
            energy=place(energy, agent.gift),
            broadcast=place(agent.broadcast, False),
            inventory=place(agent.inventory, 0),
            threshold=place(agent.threshold, threshold),
            gift=place(agent.gift, gift),
            age=place(agent.age, 0),
        )
        return agent, place(jnp.zeros(self.num_agents, bool), True)

    def field_of_view(self, state: JundlandState) -> Array:
        radius = self.view_size // 2
        alive = state.agent.energy > 0
        rows, columns = jnp.meshgrid(jnp.arange(self.height), jnp.arange(self.width), indexing="ij")

        def seen_by(position):
            window = self.visibility(state.cells.wall, position, self.offsets[0])
            relative_row, relative_column = rows - position[0], columns - position[1]
            inside = (jnp.abs(relative_row) <= radius) & (jnp.abs(relative_column) <= radius)
            row = jnp.clip(relative_row + radius, 0, 2 * radius)
            column = jnp.clip(relative_column + radius, 0, 2 * radius)
            return inside & window[row, column]

        return jax.vmap(seen_by)(state.agent.position) & alive[:, None, None]

    def visibility(self, wall: Array, position: Array, offsets: Array) -> Array:
        radius = self.view_size // 2
        fractions = jnp.arange(1, 2 * radius + 1) / (2 * radius + 1)
        target = position + offsets
        samples = jnp.round(position + offsets[..., None, :] * fractions[:, None]).astype(jnp.int32)
        at_target = jnp.all(samples == target[..., None, :], -1)
        at_origin = jnp.all(samples == position, -1)
        solid = sample(wall, samples, 1.0) > 0
        return ~jnp.any(solid & ~at_target & ~at_origin, axis=-1)

    def observe(self, state: JundlandState) -> dict[str, Array]:
        radius = self.view_size // 2
        alive = state.agent.energy > 0
        in_bush = sample(state.cells.bush, state.agent.position, 0.0) > 0
        indices = jnp.arange(self.num_agents)

        def egocentric_view(position, direction, index):
            orientation = (direction + 1) % 4
            offsets = self.offsets[orientation]
            cells = position + offsets
            relative = state.agent.position - position
            nearby = jnp.abs(relative).max(-1) <= 1
            inside = (jnp.abs(relative) <= radius).all(-1)
            shown = alive & (~in_bush | nearby) & inside
            pixel = self.pixels[orientation][
                jnp.clip(relative[:, 0] + radius, 0, 2 * radius), jnp.clip(relative[:, 1] + radius, 0, 2 * radius)
            ]
            turned = (state.agent.direction - direction) % 4
            itself = indices == index
            marks = jnp.stack(
                [itself, ~itself, state.agent.broadcast, *(turned == k for k in range(FACING_CHANNELS))], -1
            ) & shown[:, None]
            agents = jnp.zeros((self.view_size, self.view_size, marks.shape[-1]), jnp.float32)
            agents = agents.at[pixel[:, 0], pixel[:, 1]].add(marks.astype(jnp.float32))
            visible = self.visibility(state.cells.wall, position, offsets)
            terrain = jnp.stack(
                [
                    sample(state.cells.wall, cells, 1.0),
                    sample(state.cells.energy, cells, 0.0) / MAX_PLANT_ENERGY,
                    agents[..., 0],
                    agents[..., 1],
                    agents[..., 2],
                    sample(state.cells.bush, cells, 0.0),
                ],
                axis=-1,
            )
            return jnp.concatenate(
                [terrain * visible[..., None], visible[..., None], agents[..., 3:] * visible[..., None]], axis=-1
            )

        views = jax.vmap(egocentric_view)(state.agent.position, state.agent.direction, indices)
        observation = {
            "view": views,
            "energy": state.agent.energy / MAX_ENERGY,
            "inventory": state.agent.inventory,
            "scent": self.smell(state),
            "genes": jnp.stack([state.agent.threshold, state.agent.gift], -1) / MAX_ENERGY,
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
            "scent": Space(
                shape=(self.num_agents, len(DIRECTIONS) + 1), dtype=OBSERVATION_DTYPE, low=0.0, high=1.0
            ),
            "genes": Space(shape=(self.num_agents, 2), dtype=OBSERVATION_DTYPE, low=0.0, high=1.0),
        }

    def action_space(self) -> Space:
        return Space(
            shape=(self.num_agents,), dtype=jnp.int32, low=0, high=self.num_actions - 1
        )

    def time_limit(self) -> int:
        return EVALUATION_STEPS if self._time_limit is None else self._time_limit

    def render(
        self,
        state: JundlandState,
        scale: int = 1,
        previous: JundlandState | None = None,
        blend: float = 1.0,
        camera: int | None = CAMERA,
    ) -> Array:
        size = (min(camera or self.height, self.height), min(camera or self.width, self.width))
        centre = np.asarray(state.agent.position[0])
        corner = tuple(int(np.clip(centre[axis] - size[axis] // 2, 0, limit - size[axis])) for axis, limit in enumerate((self.height, self.width)))
        offsets = None
        if previous is not None:
            delta = previous.agent.position - state.agent.position
            teleported = jnp.abs(delta).max(-1, keepdims=True) > 2
            offsets = jnp.round(
                jnp.where(teleported, 0, delta) * TILE * (1.0 - blend)
            ).astype(jnp.int32)
        return render(
            state, scale, jnp.zeros_like(self.teams), self.field_of_view(state), offsets, corner, size
        )
