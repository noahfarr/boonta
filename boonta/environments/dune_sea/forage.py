import jax.numpy as jnp

from .dune_sea import DuneSea
from .generators import Generator, cave_generator
from .state import GROWTH_STEPS

STARTING_ENERGY = 0.5
ENERGY_COST = 0.003


class Forage(DuneSea):

    def __init__(
        self,
        height: int = 24,
        width: int = 24,
        num_agents: int = 4,
        view_size: int = 7,
        time_limit: int = 400,
        generator: Generator = cave_generator,
        growth_steps: int = GROWTH_STEPS,
        starting_energy: float = STARTING_ENERGY,
        energy_cost: float = ENERGY_COST,
    ):
        base_generator = generator

        def generate(key, height, width, num_agents):
            cells, agent = base_generator(key, height, width, num_agents)
            return cells, agent.replace(
                energy=jnp.full(num_agents, starting_energy, jnp.float32)
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
