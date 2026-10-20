import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DuneSea
from .generators import Generator, cave_generator
from .state import DuneSeaState

BASE_SPAWN_RATE = 0.002


def neighbour_count(grid: Array) -> Array:
    padded = jnp.pad(grid, 1)
    return (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )


class Commons(DuneSea):

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        num_agents: int = 1,
        view_size: int = 7,
        time_limit: int = 512,
        generator: Generator = cave_generator,
        spawn_rate: float = BASE_SPAWN_RATE,
    ):
        super().__init__(height, width, num_agents, view_size, time_limit, generator)
        self.spawn_rate = spawn_rate

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        state, timestep = super().step(key, state, action)
        cells = state.cells
        adjacent_energy = neighbour_count(cells.energy)
        spawn_chance = jnp.clip(self.spawn_rate * 2.0**adjacent_energy, 0.0, 1.0)
        spawn_roll = jax.random.bernoulli(key, spawn_chance)
        can_spawn = (cells.energy == 0) & (cells.wall == 0)
        new_energy = jnp.where(spawn_roll & can_spawn, 1.0, cells.energy)
        state = state.replace(cells=cells.replace(energy=new_energy))
        return state, timestep.replace(obs=self.observe(state))
