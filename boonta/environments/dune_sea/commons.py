import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DuneSea
from .generators import Generator, cave_generator
from .state import MATURE_ENERGY, MAX_PLANT_ENERGY, SEED_ENERGY, DuneSeaState

GROWTH_STEPS = 100


class Commons(DuneSea):

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        num_agents: int = 1,
        view_size: int = 7,
        time_limit: int = 512,
        generator: Generator = cave_generator,
        growth_steps: int = GROWTH_STEPS,
        **roles,
    ):
        super().__init__(height, width, num_agents, view_size, time_limit, generator, **roles)
        self.growth = (MAX_PLANT_ENERGY - SEED_ENERGY) / growth_steps

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        previous = state.cells.energy
        state, timestep = super().step(key, state, action)
        energy = state.cells.energy
        harvested = (previous > 0) & (energy == 0)
        replanted = harvested & (previous >= MATURE_ENERGY)
        grown = jnp.minimum(energy + self.growth, MAX_PLANT_ENERGY)
        energy = jnp.where(energy > 0, grown, energy)
        energy = jnp.where(replanted, SEED_ENERGY, energy)
        state = state.replace(cells=state.cells.replace(energy=energy))
        return state, timestep.replace(obs=self.observe(state))
