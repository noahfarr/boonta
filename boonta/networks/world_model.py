from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array


class WorldModel(nn.Module):
    dynamics_fn: Callable
    reward_fn: Callable
    termination_fn: Callable
    observation_fn: Callable

    def __call__(
        self, state: Array, action: Array, **kwargs: Array
    ) -> tuple[Array, Array, Array, Array]:
        *batch_shape, _ = jnp.shape(state)
        next_state = self.dynamics_fn(state=state, action=action, **kwargs)
        reward = self.reward_fn(
            state=state, action=action, next_state=next_state, **kwargs
        )
        termination = self.termination_fn(state=next_state, **kwargs)
        observation = self.observation_fn(state=next_state, **kwargs)
        return (
            next_state,
            observation,
            jnp.reshape(reward, batch_shape),
            jnp.reshape(termination, batch_shape),
        )
