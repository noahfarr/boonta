import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from boonta.utils.typing import Array

BLOCK = 128


def _forward(a: Array, b: Array, init: Array) -> Array:
    batch, steps, features = a.shape

    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((batch, steps, features), a.dtype),
        grid=(batch, features // BLOCK),
        grid_names=("batch", "group"),
    )
    def kernel(a_ref, b_ref, init_ref, out_ref):
        row = jax.lax.axis_index("batch")
        lanes = pl.ds(jax.lax.axis_index("group") * BLOCK, BLOCK)

        def body(t, carry):
            carry = (
                a_ref[row, t, lanes].astype(jnp.float32) * carry
                + b_ref[row, t, lanes].astype(jnp.float32)
            )
            out_ref[row, t, lanes] = carry.astype(out_ref.dtype)
            return carry

        jax.lax.fori_loop(
            0, steps, body, init_ref[row, lanes].astype(jnp.float32)
        )

    return kernel(a, b, init)


def _backward(a: Array, cotangent: Array) -> Array:
    cotangent = cotangent.astype(a.dtype)
    batch, steps, features = a.shape

    @functools.partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((batch, steps, features), a.dtype),
        grid=(batch, features // BLOCK),
        grid_names=("batch", "group"),
    )
    def kernel(a_ref, cotangent_ref, db_ref):
        row = jax.lax.axis_index("batch")
        lanes = pl.ds(jax.lax.axis_index("group") * BLOCK, BLOCK)

        def body(i, carry):
            t = steps - 1 - i
            carry = (
                cotangent_ref[row, t, lanes].astype(jnp.float32)
                + a_ref[row, t + 1, lanes].astype(jnp.float32) * carry
            )
            db_ref[row, t, lanes] = carry.astype(db_ref.dtype)
            return carry

        last = cotangent_ref[row, steps - 1, lanes].astype(jnp.float32)
        db_ref[row, steps - 1, lanes] = last.astype(db_ref.dtype)
        jax.lax.fori_loop(1, steps, body, last)

    return kernel(a, cotangent)


@jax.custom_vjp
def scan(a: Array, b: Array, init: Array) -> Array:
    return _forward(a, b, init)


def _scan_fwd(a, b, init):
    h = _forward(a, b, init)
    return h, (a, h, init)


def _scan_bwd(residual, cotangent):
    a, h, init = residual
    db = _backward(a, cotangent)
    shifted = jnp.concatenate([init[:, None], h[:, :-1]], axis=1)
    da = db * shifted
    return da.astype(a.dtype), db.astype(a.dtype), (db[:, 0] * a[:, 0]).astype(init.dtype)


scan.defvjp(_scan_fwd, _scan_bwd)


def pallas(a: Array, b: Array, init: Array) -> Array:
    batch, steps, *extra, features = a.shape
    if features % BLOCK:
        return associative(a, b, init).astype(a.dtype)
    if a.ndim == 3:
        return scan(a, b, init)

    def fold(x):
        return jnp.moveaxis(x, 1, -2).reshape(-1, steps, features)

    out = scan(fold(a), fold(b), init.reshape(-1, features))
    return jnp.moveaxis(out.reshape(batch, *extra, steps, features), -2, 1)


def associative(a: Array, b: Array, init: Array) -> Array:
    dtype = jnp.result_type(a, b, jnp.float32)
    a, b, init = a.astype(dtype), b.astype(dtype), init.astype(dtype)

    def operator(lhs, rhs):
        a_i, b_i = lhs
        a_j, b_j = rhs
        return a_j * a_i, a_j * b_i + b_j

    a, b = jax.lax.associative_scan(operator, (a, b), axis=1)
    return b + a * jnp.expand_dims(init, 1)


def sequential(a: Array, b: Array, init: Array) -> Array:
    def step(carry, inputs):
        a_t, b_t = inputs
        carry = a_t.astype(jnp.float32) * carry + b_t.astype(jnp.float32)
        return carry, carry.astype(a.dtype)

    _, out = jax.lax.scan(
        step,
        init.astype(jnp.float32),
        (jnp.moveaxis(a, 1, 0), jnp.moveaxis(b, 1, 0)),
        unroll=True,
    )
    return jnp.moveaxis(out, 0, 1)
