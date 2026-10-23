from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array


@struct.dataclass
class Observation:
    observation: Array
    state: Array
    context: Array
    command: Array


def _unchanged(command: Array) -> Array:
    return command


class WorldModel(nn.Module):
    dynamics_fn: Callable
    reward_fn: Callable
    termination_fn: Callable
    observation_fn: Callable
    goal: Callable[[Array], Array] = _unchanged
    advance: Callable[[Array], Array] = _unchanged

    def __call__(self, observation: Observation, action: Array) -> tuple[Observation, Array, Array]:
        state, context, command = observation.state, observation.context, observation.command
        *batch, _ = jnp.shape(state)
        goal = self.goal(command)
        following = self.dynamics_fn(state=state, action=action, context=context)
        reward = self.reward_fn(state=state, action=action, following=following, command=command, goal=goal)
        termination = self.termination_fn(state=following, command=command, goal=goal)
        return (
            self.observe(following, context, self.advance(command)),
            jnp.reshape(reward, batch),
            jnp.reshape(termination, batch),
        )

    def observe(self, state: Array, context: Array, command: Array) -> Observation:
        observation = self.observation_fn(state=state, context=context, command=command)
        return Observation(observation=observation, state=state, context=context, command=command)
