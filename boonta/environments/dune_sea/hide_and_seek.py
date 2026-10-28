import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DuneSea
from .generators import Generator, cave_generator
from .state import GROWTH_STEPS, Cell, DuneSeaState

ENERGY_COST = 0.0015
INITIAL_DIRT = 4
SIGHT_DRAIN = 0.05


class HideAndSeek(DuneSea):

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
            growth_steps=growth_steps,
            teams=[0] * num_hiders + [1] * num_seekers,
            eaters=hiders + [False] * num_seekers,
            thieves=[False] * num_hiders + seekers,
            energy_cost=energy_cost,
            catch_amount=catch_amount,
        )
        self.sight_drain = sight_drain

    def drain(self, state: DuneSeaState, positions: Array, cells: Cell) -> Array:
        wall, bush = cells.wall, cells.bush
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
        drained = jnp.minimum(
            self.sight_drain, state.agent.energy / jnp.maximum(watchers, 1)
        )
        return (sees * drained[None, :]).sum(1) - drained * watchers

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        state, timestep = super().step(key, state, action)
        alive = state.agent.energy > 0
        game_over = ~(alive & ~self.thieves).any() | ~(alive & self.thieves).any()
        return state, timestep.replace(terminated=timestep.terminated | game_over)
