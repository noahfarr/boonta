import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DuneSea
from .generators import Generator, cave_generator
from .state import GROWTH_STEPS, DuneSeaState

TAG_AMOUNT = 0.3
STARTING_ENERGY = 0.5
ENERGY_COST = 0.003


class Tag(DuneSea):

    def __init__(
        self,
        height: int = 40,
        width: int = 40,
        num_agents: int = 12,
        view_size: int = 7,
        time_limit: int = 600,
        generator: Generator = cave_generator,
        growth_steps: int = GROWTH_STEPS,
        tag_amount: float = TAG_AMOUNT,
        starting_energy: float = STARTING_ENERGY,
        energy_cost: float = ENERGY_COST,
    ):
        base_generator = generator

        def generate(key, height, width, num_agents):
            cells, agent = base_generator(key, height, width, num_agents)
            return cells, agent.replace(energy=jnp.full(num_agents, starting_energy, jnp.float32))

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
            thieves=[True] * num_agents,
            catch_amount=tag_amount,
            energy_cost=energy_cost,
        )

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        state, timestep = super().step(key, state, action)
        last_one_standing = (state.agent.energy > 0).sum() <= 1
        return state, timestep.replace(terminated=timestep.terminated | last_one_standing)
