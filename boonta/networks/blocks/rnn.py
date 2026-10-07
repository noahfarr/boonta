import jax
import jax.numpy as jnp
from flax import linen as nn

from boonta.utils import broadcast
from boonta.utils.typing import Array, Carry, Key

from .block import Block


def reset_carry(done: Array, carry: Carry, initial_carry: Carry) -> Carry:
    return jax.tree.map(
        lambda initial, leaf: jnp.where(broadcast(done, leaf), initial, leaf),
        initial_carry,
        carry,
    )


class RNNCellBase(nn.Module):
    features: int

    def __call__(self, carry: Carry, x: Array) -> tuple[Carry, Array]:
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        raise NotImplementedError


class RNN(Block):
    cell: RNNCellBase

    @nn.compact
    def __call__(self, carry: Carry, inputs: Array, done: Array) -> tuple[Carry, Array]:
        batch_size, _sequence_length, *features = inputs.shape
        initial_carry = self.cell.initialize_carry(
            jax.random.key(0), (batch_size, *features)
        )
        if carry is None:
            carry = initial_carry

        def scan_fn(cell, carry, inputs):
            features, done = inputs
            carry = reset_carry(done, carry, initial_carry)
            return cell(carry, features)

        scan = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        return scan(self.cell, carry, (inputs, done))

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        *batch_size, _ = input_shape
        return self.cell.initialize_carry(key, (*batch_size, self.cell.features))
