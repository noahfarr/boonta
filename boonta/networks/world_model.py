from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array


@struct.dataclass(frozen=True)
class Prediction:
    state: Array
    reward: Array
    termination: Array


class WorldModel(nn.Module):
    dynamics: Callable
    reward: Callable
    termination: Callable
    observation: Callable

    def __call__(self, state: Array, action: Array, context: Array, command: Array, goal: Array) -> Prediction:
        *batch, _ = jnp.shape(state)
        if self.is_initializing():
            self.observe(state, context, command)
        following = self.dynamics(state=state, action=action, context=context)
        reward = self.reward(state=state, action=action, following=following, command=command, goal=goal)
        termination = self.termination(state=following, command=command, goal=goal)
        return Prediction(
            state=following, reward=jnp.reshape(reward, batch), termination=jnp.reshape(termination, batch)
        )

    def observe(self, state: Array, context: Array, command: Array) -> Array:
        return self.observation(state=state, context=context, command=command)
