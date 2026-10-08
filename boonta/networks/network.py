from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.utils import broadcast
from boonta.utils.typing import Array, Carry, Key

from .layers import Identity


class Network(nn.Module):
    feature_extractor: nn.Module = Identity()
    torso: nn.Module | None = None
    head: nn.Module = Identity()

    @nn.compact
    def __call__(
        self,
        obs: Array,
        action: Array | None = None,
        reward: Array | None = None,
        done: Array | None = None,
        carry: Carry = None,
        temperature: float | Array | None = None,
        **kwargs: Array,
    ) -> Any:
        if done is not None:

            def blank(leaf: Array) -> Array:
                return jnp.where(broadcast(done, leaf), jnp.zeros_like(leaf), leaf)

            action = jax.tree.map(blank, action)
            reward = jax.tree.map(blank, reward)

        x = self.feature_extractor(obs, action, reward, done, **kwargs)
        head = self.head
        if temperature is not None:
            head = partial(head, temperature=temperature)

        if self.torso is None:
            self.sow("intermediates", "features", x, reduce_fn=lambda _, value: value)
            return head(x, **kwargs)

        carry, x = self.torso(carry, x, done, **kwargs)
        self.sow("intermediates", "features", x, reduce_fn=lambda _, value: value)
        return carry, head(x, **kwargs)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        if self.torso is None:
            return None
        return self.torso.initialize_carry(key, input_shape)
