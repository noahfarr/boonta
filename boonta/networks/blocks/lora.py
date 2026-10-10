import math
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.utils import dequantize
from boonta.utils.typing import Array, Carry, Key, PyTree

from ..pretrained import init_fn
from .block import Block


class LoRA(Block):
    block: Block
    params: PyTree
    rank: int = 4
    alpha: float = 1.0
    layers: tuple[type[nn.Module], ...] = (nn.Dense, nn.DenseGeneral)
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        base = self.variable("frozen_params", "base", init_fn(self.params)).value
        base = jax.lax.stop_gradient(dequantize(base))
        apply = partial(self.block.apply, {"params": base})

        def adapts_layer(context) -> bool:
            return context.method_name == "__call__" and isinstance(context.module, self.layers)

        shapes = {}

        def record_kernels(next_fun, args, kwargs, context):
            outputs = next_fun(*args, **kwargs)
            if adapts_layer(context):
                inputs, *_ = args
                kernel = context.module.get_variable("params", "kernel")
                contracted = (kernel.ndim + inputs.ndim - outputs.ndim) // 2
                shapes[context.module.path] = (kernel.shape[:contracted], kernel.shape[contracted:])
            return outputs

        with nn.intercept_methods(record_kernels):
            jax.eval_shape(apply, carry, x, done)

        adapters = {}
        for path, (input_shape, output_shape) in shapes.items():
            name = "_".join(path)
            down = self.param(
                f"{name}_a",
                lambda key, shape=input_shape: self.kernel_init(
                    key, (math.prod(shape), self.rank)
                ).reshape(*shape, self.rank),
            )
            up = self.param(f"{name}_b", nn.initializers.zeros_init(), (self.rank, *output_shape))
            adapters[path] = (down, up)

        scale = self.alpha / self.rank

        def add_low_rank(next_fun, args, kwargs, context):
            outputs = next_fun(*args, **kwargs)
            if not adapts_layer(context):
                return outputs
            inputs, *_ = args
            down, up = adapters[context.module.path]
            low = jnp.tensordot(inputs, down.astype(inputs.dtype), axes=down.ndim - 1)
            delta = jnp.tensordot(low, up.astype(low.dtype), axes=1)
            return outputs + (scale * delta).astype(outputs.dtype)

        with nn.intercept_methods(add_low_rank):
            return apply(carry, x, done)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.block.initialize_carry(key, input_shape)
