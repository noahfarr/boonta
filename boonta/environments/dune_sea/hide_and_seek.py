import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .commons import GROWTH_STEPS, Commons
from .generators import Generator, cave_generator
from .state import DuneSeaState

ENERGY_COST = 0.003
INITIAL_DIRT = 4
SIGHT_DRAIN = 0.05


class HideAndSeek(Commons):

    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        num_hiders: int = 10,
        num_seekers: int = 10,
        view_size: int = 7,
        time_limit: int = 700,
        generator: Generator = cave_generator,
        growth_steps: int = GROWTH_STEPS,
        energy_cost: float = ENERGY_COST,
        catch_amount: float = 0.0,
        initial_dirt: int = INITIAL_DIRT,
        sight_drain: float = SIGHT_DRAIN,
    ):
        hiders, seekers = [True] * num_hiders, [True] * num_seekers
        is_hider = jnp.arange(num_hiders + num_seekers) < num_hiders
        base_generator = generator

        def generate(key, height, width, num_agents):
            cells, agent = base_generator(key, height, width, num_agents)
            dirt = jnp.where(is_hider, initial_dirt, 0).astype(agent.inventory.dtype)
            return cells, agent.replace(inventory=dirt)

        super().__init__(
            height,
            width,
            num_hiders + num_seekers,
            view_size,
            time_limit,
            generate,
            growth_steps,
            teams=[0] * num_hiders + [1] * num_seekers,
            eaters=hiders + [False] * num_seekers,
            thieves=[False] * num_hiders + seekers,
            energy_cost=energy_cost,
            catch_amount=catch_amount,
            sight_drain=sight_drain,
        )

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        state, timestep = super().step(key, state, action)
        alive = state.agent.energy > 0
        game_over = ~(alive & ~self.thieves).any() | ~(alive & self.thieves).any()
        return state, timestep.replace(terminated=timestep.terminated | game_over)
