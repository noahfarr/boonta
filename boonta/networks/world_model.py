from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, PyTree


def _unchanged(command: Array) -> Array:
    return command


class WorldModel(nn.Module):
    dynamics_fn: Callable
    reward_fn: Callable
    termination_fn: Callable
    observation_fn: Callable
    goal: Callable[[Array], Array] = _unchanged
    advance: Callable[[Array], Array] = _unchanged

    def __call__(self, world_model_state: PyTree, action: Array) -> tuple[PyTree, Array, Array]:
        state, context, command = world_model_state.state, world_model_state.context, world_model_state.command
        *batch, _ = jnp.shape(state)
        goal = self.goal(command)
        following = self.dynamics_fn(state=state, action=action, context=context)
        reward = self.reward_fn(state=state, action=action, following=following, command=command, goal=goal)
        termination = self.termination_fn(state=following, command=command, goal=goal)
        command = self.advance(command)
        return (
            world_model_state.replace(
                observation=self.observation_fn(state=following, context=context, command=command),
                state=following,
                command=command,
            ),
            jnp.reshape(reward, batch),
            jnp.reshape(termination, batch),
        )
