import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DuneSea
from .generators import (MAZE_BRAID, MAZE_PRIZE_DISTANCE, Generator,
                         maze_generator)
from .state import Cell, DuneSeaState

STARTING_ENERGY = 0.5
ENERGY_COST = 0.0015


class Maze(DuneSea):

    def __init__(
        self,
        height: int = 21,
        width: int = 21,
        num_agents: int = 4,
        view_size: int = 7,
        time_limit: int = 300,
        braid: float = MAZE_BRAID,
        prize_distance: int = MAZE_PRIZE_DISTANCE,
        starting_energy: float = STARTING_ENERGY,
        energy_cost: float = ENERGY_COST,
        generator: Generator | None = None,
    ):
        assert height % 2 == 1 and width % 2 == 1, "a maze needs odd sides"
        base_generator = generator or (
            lambda key, height, width, num_agents: maze_generator(
                key, height, width, num_agents, braid, prize_distance
            )
        )

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
            teams=list(range(num_agents)),
            eaters=[True] * num_agents,
            thieves=[False] * num_agents,
            energy_cost=energy_cost,
        )

    def interact(
        self, state: DuneSeaState, action: Array, positions: Array, turn_order: Array
    ) -> tuple[Cell, Array, Array]:
        cells, gain, inventory_change = super().interact(
            state, action, positions, turn_order
        )
        return (
            cells.replace(wall=state.cells.wall),
            gain,
            jnp.zeros_like(inventory_change),
        )

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        state, timestep = super().step(key, state, action)
        found = ~(state.cells.energy > 0).any()
        return state, timestep.replace(terminated=timestep.terminated | found)
