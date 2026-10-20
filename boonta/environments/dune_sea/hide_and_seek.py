import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .commons import GROWTH_STEPS, Commons
from .dune_sea import NOOP
from .generators import Generator, cave_generator
from .state import DuneSeaState

PREPARATION_STEPS = 40


class HideAndSeek(Commons):

    def __init__(
        self,
        height: int = 24,
        width: int = 24,
        num_hiders: int = 2,
        num_seekers: int = 2,
        view_size: int = 7,
        time_limit: int = 400,
        generator: Generator = cave_generator,
        growth_steps: int = GROWTH_STEPS,
        preparation_steps: int = PREPARATION_STEPS,
    ):
        hiders, seekers = [True] * num_hiders, [True] * num_seekers
        super().__init__(
            height,
            width,
            num_hiders + num_seekers,
            view_size,
            time_limit,
            generator,
            growth_steps,
            teams=[0] * num_hiders + [1] * num_seekers,
            eaters=hiders + [False] * num_seekers,
            thieves=[False] * num_hiders + seekers,
        )
        self.preparation_steps = preparation_steps

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        frozen = self.thieves & (state.time < self.preparation_steps)
        action = jnp.where(frozen, NOOP, jnp.asarray(action, jnp.int32))
        state, timestep = super().step(key, state, action)
        alive = state.agent.energy > 0
        game_over = ~(alive & ~self.thieves).any() | ~(alive & self.thieves).any()
        return state, timestep.replace(terminated=timestep.terminated | game_over)
