import copy
from collections.abc import Sequence
from itertools import chain

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Dtype

from ..blocks import GLU, Block, Residual, Stack, Stateless


def llama(
    blocks: Sequence[Block],
    num_layers: int,
    features: int,
    expansion_factor: int = 4,
    hidden_dim: int | None = None,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
) -> Stack:
    def layer(block: Block) -> tuple[Residual, Residual]:
        return (
            Residual(
                Stack(
                    (
                        Stateless(nn.RMSNorm(dtype=dtype, param_dtype=param_dtype)),
                        block,
                    )
                )
            ),
            Residual(
                Stack(
                    (
                        Stateless(nn.RMSNorm(dtype=dtype, param_dtype=param_dtype)),
                        GLU(
                            features,
                            expansion_factor,
                            hidden_dim=hidden_dim,
                            dtype=dtype,
                            param_dtype=param_dtype,
                        ),
                    )
                )
            ),
        )

    layers = chain.from_iterable(
        layer(copy.deepcopy(blocks[index % len(blocks)])) for index in range(num_layers)
    )
    return Stack((*layers, Stateless(nn.RMSNorm(dtype=dtype, param_dtype=param_dtype))))
