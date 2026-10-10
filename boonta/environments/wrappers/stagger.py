import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep, broadcast

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class StaggerState(WrapperState):
    budget: Array


class Stagger(Wrapper):
    def __init__(self, env, spread: int):
        super().__init__(env)
        self.spread = spread

    def init(self, key: Key) -> tuple[StaggerState, Timestep]:
        env_key, spread_key = jax.random.split(key)
        env_state, timestep = self._env.init(env_key)
        budget = jax.random.randint(
            spread_key, jnp.shape(timestep.truncated), 0, self.spread
        )
        return StaggerState(env_state, budget), timestep

    def step(
        self, key: Key, state: StaggerState, action: Array
    ) -> tuple[StaggerState, Timestep]:
        step_key, reset_key = jax.random.split(key)
        env_state, timestep = self._env.step(step_key, state.env_state, action)

        fire = (state.budget == 0) & ~timestep.done.astype(bool)
        initial_env_state, initial_timestep = self._env.init(reset_key)

        def select(initial_leaf, leaf):
            return jnp.where(broadcast(fire, leaf), initial_leaf, leaf)

        env_state = jax.tree.map(select, initial_env_state, env_state)
        obs = jax.tree.map(select, initial_timestep.obs, timestep.obs)
        truncated = timestep.truncated | fire.astype(timestep.truncated.dtype)
        budget = jnp.where(state.budget < 0, state.budget, state.budget - 1)

        return StaggerState(env_state, budget), timestep.replace(
            obs=obs, truncated=truncated
        )
