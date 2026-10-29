import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DuneSea
from .generators import Generator, cave_generator
from .state import GROWTH_STEPS, Cell, DuneSeaState

STARTING_ENERGY = 0.5
ENERGY_COST = 0.002
INITIAL_DIRT = 4
STORM_PERIOD = 150
STORM_LENGTH = 40
STORM_WARNING = 20
STORM_DRAIN = 0.01


class Shelter(DuneSea):

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        num_agents: int = 4,
        view_size: int = 7,
        time_limit: int = 600,
        generator: Generator = cave_generator,
        growth_steps: int = GROWTH_STEPS,
        starting_energy: float = STARTING_ENERGY,
        energy_cost: float = ENERGY_COST,
        initial_dirt: int = INITIAL_DIRT,
        storm_period: int = STORM_PERIOD,
        storm_length: int = STORM_LENGTH,
        storm_warning: int = STORM_WARNING,
        storm_drain: float = STORM_DRAIN,
    ):
        base_generator = generator

        def generate(key, height, width, num_agents):
            cells, agent = base_generator(key, height, width, num_agents)
            return cells, agent.replace(
                energy=jnp.full(num_agents, starting_energy, jnp.float32),
                inventory=jnp.full(num_agents, initial_dirt, agent.inventory.dtype),
            )

        super().__init__(
            height,
            width,
            num_agents,
            view_size,
            time_limit,
            generate,
            growth_steps=growth_steps,
            teams=list(range(num_agents)),
            eaters=[True] * num_agents,
            thieves=[False] * num_agents,
            energy_cost=energy_cost,
        )
        self.storm_period = storm_period
        self.storm_length = storm_length
        self.storm_warning = storm_warning
        self.storm_drain = storm_drain

    def storming(self, time: Array) -> Array:
        return time % self.storm_period >= self.storm_period - self.storm_length

    def walled(self, positions: Array, wall: Array) -> Array:
        padded = jnp.pad(wall, 1, constant_values=1.0)

        def sealed(position):
            row, column = position
            around = jnp.stack(
                [
                    padded[row, column + 1],
                    padded[row + 2, column + 1],
                    padded[row + 1, column],
                    padded[row + 1, column + 2],
                ]
            )
            return (around > 0).sum()

        return jax.vmap(sealed)(positions)

    def drain(self, state: DuneSeaState, positions: Array, cells: Cell) -> Array:
        alive = state.agent.energy > 0
        open_sides = 4 - self.walled(positions, cells.wall)
        stormy = self.storming(state.time + 1)
        return jnp.where(alive & stormy, -self.storm_drain * open_sides / 4, 0.0)

    def weather(self, state: DuneSeaState) -> Array:
        phase = (state.time + 1) % self.storm_period
        calm = self.storm_period - self.storm_length - self.storm_warning
        ramp = jnp.clip((phase - calm) / max(self.storm_warning, 1), 0.0, 1.0)
        return jnp.full(self.num_agents, ramp, jnp.float32)

    def count(self, state: DuneSeaState) -> dict[str, Array]:
        alive = state.agent.energy > 0
        stormy = self.storming(state.time)
        sealed = self.walled(state.agent.position, state.cells.wall) == 4
        return {
            "sheltered": (alive & stormy & sealed).sum().astype(jnp.float32),
            "exposed": (alive & stormy & ~sealed).sum().astype(jnp.float32),
        }

    def init(self, key: Key) -> tuple[DuneSeaState, Timestep]:
        state, timestep = super().init(key)
        return state, timestep.replace(info={**timestep.info, **self.count(state)})

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        state, timestep = super().step(key, state, action)
        return state, timestep.replace(info={**timestep.info, **self.count(state)})

    def render(
        self,
        state: DuneSeaState,
        scale: int = 1,
        storm: float | None = None,
        previous: DuneSeaState | None = None,
        blend: float = 1.0,
    ) -> Array:
        if storm is None:
            storm = float(self.weather(state).max())
        return super().render(state, scale, storm, previous, blend)
