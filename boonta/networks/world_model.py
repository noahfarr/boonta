from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array


def _unchanged(command: Array) -> Array:
    return command


class WorldModel(nn.Module):
    dynamics_fn: Callable
    reward_fn: Callable
    termination_fn: Callable
    observation_fn: Callable
    goal: Callable[[Array], Array] = _unchanged
    advance: Callable[[Array], Array] = _unchanged

    def __call__(
        self, state: Array, action: Array, context: Array, command: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        *batch, _ = jnp.shape(state)
        goal = self.goal(command)
        following = self.dynamics_fn(state=state, action=action, context=context)
        reward = self.reward_fn(state=state, action=action, following=following, command=command, goal=goal)
        termination = self.termination_fn(state=following, command=command, goal=goal)
        command = self.advance(command)
        observation = self.observation_fn(state=following, context=context, command=command)
        return following, observation, command, jnp.reshape(reward, batch), jnp.reshape(termination, batch)
