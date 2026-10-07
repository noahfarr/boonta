import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Key

from .block import Block


class Tower(Block):
    block: Block
    num_layers: int = 1

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        if carry is None:
            batch_size, _, *features = x.shape
            carry = self.initialize_carry(jax.random.key(0), (batch_size, *features))

        def layer(block, activations, state, mask):
            state, activations = block(state, activations, mask)
            return activations, state

        x, carries = nn.scan(
            layer,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            length=self.num_layers,
            in_axes=(1, nn.broadcast),
            out_axes=1,
            unroll=self.num_layers,
        )(self.block, x, carry, done)
        return carries, x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        carry = self.block.initialize_carry(key, input_shape)
        return jax.tree.map(
            lambda leaf: jnp.repeat(
                jnp.expand_dims(leaf, 1), self.num_layers, axis=1
            ),
            carry,
        )
