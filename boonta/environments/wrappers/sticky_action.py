import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class StickyActionState(WrapperState):
    action: Array


class StickyAction(Wrapper):
    def __init__(self, env, probability: float = 0.1):
        super().__init__(env)
        self._probability = probability

    def init(self, key: Key) -> tuple[StickyActionState, Timestep]:
        env_state, timestep = self._env.init(key)
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        return StickyActionState(env_state, action), timestep

    def step(
        self, key: Key, state: StickyActionState, action: Array
    ) -> tuple[StickyActionState, Timestep]:
        sticky_key, env_key = jax.random.split(key)
        repeat = jax.random.uniform(sticky_key, action.shape) < self._probability
        executed = jnp.where(repeat, state.action, action)
        env_state, timestep = self._env.step(env_key, state.env_state, executed)
        return StickyActionState(env_state, executed), timestep.replace(action=action)
