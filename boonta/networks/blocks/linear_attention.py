import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array, Carry, Key

from .block import Block


@struct.dataclass
class LinearAttentionInputs:
    query: Array
    key: Array
    value: Array
    log_decay: Array | None = None
    beta: Array | None = None


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
    def state_shape(self) -> tuple[int, int, int]:
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        raise NotImplementedError


def fill(inputs: LinearAttentionInputs) -> tuple[Array, ...]:
    batch_size, sequence_length, num_heads, _ = inputs.value.shape
    gates = jnp.zeros((batch_size, sequence_length, num_heads), jnp.float32)
    log_decay = gates if inputs.log_decay is None else inputs.log_decay
    beta = gates + 1.0 if inputs.beta is None else inputs.beta
    return tuple(
        x.astype(jnp.float32)
        for x in (inputs.query, inputs.key, inputs.value, log_decay, beta)
    )


def recurrent(
    inputs: LinearAttentionInputs, done: Array, state: Array, delta_rule: bool = True
) -> tuple[Array, Array]:
    query, key, value, log_decay, beta = fill(inputs)

    def step(state: Array, inputs) -> tuple[Array, Array]:
        query, key, value, log_decay, beta, done = inputs
        state = jnp.where(done[:, None, None, None], 0.0, state)
        state = state * jnp.exp(log_decay)[..., None, None]
        if delta_rule:
            value = value - jnp.einsum("bhk,bhkv->bhv", key, state)
        state = state + jnp.einsum("bhk,bhv->bhkv", key, value * beta[..., None])
        return state, jnp.einsum("bhk,bhkv->bhv", query, state)

    state, outputs = jax.lax.scan(
        step,
        state.astype(jnp.float32),
        tuple(jnp.moveaxis(x, 1, 0) for x in (query, key, value, log_decay, beta, done)),
    )
    return jnp.moveaxis(outputs, 0, 1), state


def chunkwise(
    inputs: LinearAttentionInputs,
    done: Array,
    state: Array,
    delta_rule: bool = True,
    chunk_size: int = 64,
) -> tuple[Array, Array]:
    query, key, value, log_decay, beta = fill(inputs)
    batch_size, sequence_length, num_heads, value_dim = value.shape
    size = min(chunk_size, sequence_length)
    padding = -sequence_length % size
    num_chunks = (sequence_length + padding) // size

    def chunk(x: Array) -> Array:
        x = jnp.pad(x, [(0, 0), (0, padding)] + [(0, 0)] * (x.ndim - 2))
        x = jnp.moveaxis(x.reshape(batch_size, num_chunks, size, *x.shape[2:]), 1, 0)
        if x.ndim > 3:
            x = jnp.swapaxes(x, 2, 3)
        return x

    query, key, value, log_decay, beta = (
        chunk(x) for x in (query, key, value, log_decay, beta)
    )
    resets = jnp.cumsum(chunk(done.astype(jnp.int32)), axis=-1)[:, :, None]

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

    targets = value * beta[..., None]
    readouts = jnp.zeros_like(key)
    if delta_rule:
        weighted_key = key * beta[..., None]
        system = jnp.einsum("...id,...jd->...ij", weighted_key, key) * pairwise
        system = jnp.tril(system, -1) + jnp.eye(size, dtype=system.dtype)
        solved = jax.scipy.linalg.solve_triangular(
            system,
            jnp.concatenate([targets, weighted_key * head[..., None]], axis=-1),
            lower=True,
            unit_diagonal=True,
        )
        targets, readouts = jnp.split(solved, [value_dim], axis=-1)
    intra = jnp.einsum("...id,...jd->...ij", query, key) * pairwise
    query = query * head[..., None]
    key = key * tail[..., None]

    def step(state: Array, inputs) -> tuple[Array, Array]:
        query, key, targets, readouts, intra, persist = inputs
        correction = targets - readouts @ state
        output = query @ state + intra @ correction
        state = state * persist[..., None, None] + jnp.swapaxes(key, -1, -2) @ correction
        return state, output

    state, outputs = jax.lax.scan(
        step,
        state.astype(jnp.float32),
        (query, key, targets, readouts, intra, persist),
    )
    outputs = jnp.moveaxis(jnp.swapaxes(outputs, 2, 3), 0, 1)
    outputs = outputs.reshape(batch_size, num_chunks * size, num_heads, value_dim)
    return outputs[:, :sequence_length], state


class LinearAttention(Block):
    cell: LinearAttentionCellBase
    delta_rule: bool = True
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
            outputs, state = recurrent(inputs, done, carry.state, self.delta_rule)
        else:
            outputs, state = chunkwise(
                inputs, done, carry.state, self.delta_rule, self.chunk_size
            )
        carry = LinearAttentionCarry(state=state.astype(carry.state.dtype), cell=cell_carry)
        return carry, self.cell.output(outputs, x)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> LinearAttentionCarry:
        batch_size, *_ = input_shape
        return LinearAttentionCarry(
            state=jnp.zeros((batch_size, *self.cell.state_shape()), jnp.float32),
            cell=self.cell.initialize_carry(key, input_shape),
        )
