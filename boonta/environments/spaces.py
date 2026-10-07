from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from boonta.utils import canonicalize_dtype
from boonta.utils.typing import Array, Key


@dataclass
class Space:
    shape: tuple[int, ...] = ()
    dtype: jnp.dtype = jnp.float32
    low: float | Sequence[float] | Array = 0.0
    high: float | Sequence[float] | Array = 1.0

    def __post_init__(self):
        self.dtype = canonicalize_dtype(self.dtype)
        self.low = jnp.asarray(self.low)
        self.high = jnp.asarray(self.high)

    def sample(self, key: Key) -> Array:
        if jnp.issubdtype(self.dtype, jnp.integer):
            return jax.random.randint(
                key, self.shape, self.low, self.high + 1, self.dtype
            )
        return jax.random.uniform(key, self.shape, self.dtype, self.low, self.high)

    @property
    def num_actions(self) -> int | tuple[int, ...]:
        categories = self.high - self.low + 1
        if categories.shape == ():
            return int(categories)
        return tuple(int(n) for n in categories)
