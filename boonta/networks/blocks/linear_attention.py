import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from jax.ad_checkpoint import checkpoint_name

from boonta.utils.typing import Array, Carry, Key

from .block import Block


@struct.dataclass
class LinearAttentionInputs:
    query: Array
    key: Array
    value: Array
    log_decay: Array
    erase_gate: Array
    write_gate: Array


@struct.dataclass
class LinearAttentionCarry:
    state: Array
    cell: Carry


class LinearAttentionCellBase(nn.Module):
    def __call__(
        self, carry: Carry, x: Array, done: Array
    ) -> tuple[Carry, LinearAttentionInputs]:
        raise NotImplementedError

    def output(self, outputs: Array, x: Array) -> Array:
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(
        self, key: Key, input_shape: tuple[int, ...]
    ) -> LinearAttentionCarry:
        raise NotImplementedError


def unpack(inputs: LinearAttentionInputs) -> tuple[Array, ...]:
    return tuple(
        x.astype(jnp.float32)
        for x in (
            inputs.query,
            inputs.key,
            inputs.value,
            inputs.log_decay,
            inputs.erase_gate,
            inputs.write_gate,
        )
    )


def recurrent(
    inputs: LinearAttentionInputs, done: Array, state: Array
) -> tuple[Array, Array]:
    def step(state: Array, inputs) -> tuple[Array, Array]:
        query, key, value, log_decay, erase_gate, write_gate, done = inputs
        state = jnp.where(done[:, None, None, None], 0.0, state)
        state = state * jnp.exp(log_decay)[..., None, None]
        readout = jnp.einsum("bhk,bhkv->bhv", key, state)
        written = value * write_gate[..., None] - readout * erase_gate[..., None]
        state = state + jnp.einsum("bhk,bhv->bhkv", key, written)
        return state, jnp.einsum("bhk,bhkv->bhv", query, state)

    state, outputs = jax.lax.scan(
        step,
        state.astype(jnp.float32),
        tuple(jnp.moveaxis(x, 1, 0) for x in (*unpack(inputs), done)),
    )
    return jnp.moveaxis(outputs, 0, 1), state


def chunkwise(
    inputs: LinearAttentionInputs,
    done: Array,
    state: Array,
    chunk_size: int = 64,
) -> tuple[Array, Array]:
    batch_size, sequence_length, num_heads, value_dim = inputs.value.shape
    size = min(chunk_size, sequence_length)
    padding = -sequence_length % size
    num_chunks = (sequence_length + padding) // size

    def chunk(x: Array) -> Array:
        x = jnp.pad(x, [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
        x = jnp.moveaxis(x.reshape(batch_size, num_chunks, size, *x.shape[2:]), 1, 0)
        if x.ndim > 3:
            x = jnp.swapaxes(x, 2, 3)
        return x

    dtype = inputs.value.dtype

    def matmul(lhs: Array, rhs: Array) -> Array:
        return jnp.matmul(
            lhs.astype(dtype), rhs.astype(dtype), preferred_element_type=jnp.float32
        )

    def prepare(query, key, value, log_decay, erase_gate, write_gate, resets):
        cumulative = jnp.cumsum(log_decay, axis=-1)
        lower = jnp.tril(jnp.ones((size, size), bool))
        same = (resets[..., :, None] == resets[..., None, :]) & lower
        gap = cumulative[..., :, None] - cumulative[..., None, :]
        pairwise = jnp.where(same, jnp.exp(jnp.where(same, gap, 0.0)), 0.0)
        fresh = resets == 0
        head = jnp.where(fresh, jnp.exp(cumulative), 0.0)
        final = cumulative[..., -1:]
        tail = jnp.where(resets == resets[..., -1:], jnp.exp(final - cumulative), 0.0)
        persist = jnp.where(fresh[..., -1], jnp.exp(final[..., 0]), 0.0)

        system = erase_gate[..., None] * matmul(key, jnp.swapaxes(key, -1, -2)) * pairwise
        system = jnp.tril(system, -1) + jnp.eye(size, dtype=system.dtype)
        erased_key = key * erase_gate[..., None]
        solved = jax.scipy.linalg.solve_triangular(
            system,
            jnp.concatenate(
                [value * write_gate[..., None], erased_key * head[..., None]], axis=-1
            ),
            lower=True,
            unit_diagonal=True,
        )
        solved = checkpoint_name(solved, "solved")
        targets, readouts = jnp.split(solved, [value_dim], axis=-1)
        intra = matmul(query, jnp.swapaxes(key, -1, -2)) * pairwise
        return query * head[..., None], key * tail[..., None], targets, readouts, intra, persist

    query, key, value = (chunk(x) for x in (inputs.query, inputs.key, inputs.value))
    log_decay, erase_gate, write_gate = (
        chunk(x.astype(jnp.float32))
        for x in (inputs.log_decay, inputs.erase_gate, inputs.write_gate)
    )
    resets = jnp.cumsum(chunk(done.astype(jnp.int32)), axis=-1)[:, :, None]
    policy = jax.checkpoint_policies.save_only_these_names("solved")
    chunks = jax.checkpoint(prepare, policy=policy)(
        query, key, value, log_decay, erase_gate, write_gate, resets
    )

    def step(state: Array, inputs) -> tuple[Array, Array]:
        query, key, targets, readouts, intra, persist = inputs
        correction = targets - matmul(readouts, state)
        output = matmul(query, state) + matmul(intra, correction)
        state = state * persist[..., None, None] + matmul(jnp.swapaxes(key, -1, -2), correction)
        return state, output

    state, outputs = jax.lax.scan(step, state.astype(jnp.float32), chunks)
    outputs = jnp.moveaxis(jnp.swapaxes(outputs, 2, 3), 0, 1)
    outputs = outputs.reshape(batch_size, num_chunks * size, num_heads, value_dim)
    return outputs[:, :sequence_length], state


class LinearAttention(Block):
    cell: LinearAttentionCellBase
    chunk_size: int = 64

    @nn.compact
    def __call__(
        self, carry: LinearAttentionCarry | None, x: Array, done: Array
    ) -> tuple[LinearAttentionCarry, Array]:
        _, sequence_length, *_ = x.shape
        if carry is None:
            carry = self.initialize_carry(jax.random.key(0), x.shape)
        cell_carry, inputs = self.cell(carry.cell, x, done)
        if sequence_length == 1:
            outputs, state = recurrent(inputs, done, carry.state)
        else:
            outputs, state = chunkwise(inputs, done, carry.state, self.chunk_size)
        carry = LinearAttentionCarry(state=state.astype(carry.state.dtype), cell=cell_carry)
        return carry, self.cell.output(outputs, x)

    @nn.nowrap
    def initialize_carry(
        self, key: Key, input_shape: tuple[int, ...]
    ) -> LinearAttentionCarry:
        return self.cell.initialize_carry(key, input_shape)
