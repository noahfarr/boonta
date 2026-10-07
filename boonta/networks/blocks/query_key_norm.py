from collections.abc import Callable

import jax.numpy as jnp
from flax import linen as nn

from boonta.utils.typing import Array, Dtype


class QueryKeyNorm(nn.Module):
    positional_embedding: Callable[
        [Array, Array, Array, Array], tuple[Array, Array, Array | None]
    ] = lambda query, key, query_positions, key_positions: (query, key, None)
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self, query: Array, key: Array, query_positions: Array, key_positions: Array
    ) -> tuple[Array, Array, Array | None]:
        query = nn.RMSNorm(
            dtype=self.dtype, param_dtype=self.param_dtype, name="query"
        )(query)
        key = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="key")(
            key
        )
        return self.positional_embedding(query, key, query_positions, key_positions)
