from functools import partial

import jax
import jax.numpy as jnp


def kernel(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.T.reshape(shape)


def scale(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.reshape(shape)


def centered_scale(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return (weight.astype(jnp.float32) + 1.0).reshape(shape)


def rows(start: int, stop: int, weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight[start:stop].T.reshape(shape)


def taps(start: int, stop: int, weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight[start:stop, 0].T.reshape(shape)


def half(part: int, weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    features, num_heads, head_dim = shape
    return weight.T.reshape(features, num_heads, 2, head_dim)[:, :, part]


def linear_layer(key_width: int, value_width: int) -> dict:
    query, key, value = (0, key_width), (key_width, 2 * key_width), (
        2 * key_width,
        2 * key_width + value_width,
    )
    entries = {
        ("gate", "kernel"): ("in_proj_z.weight", kernel),
        ("alpha", "kernel"): ("in_proj_a.weight", kernel),
        ("beta", "kernel"): ("in_proj_b.weight", kernel),
        ("log_rate",): ("A_log", scale),
        ("step_bias",): ("dt_bias", scale),
        ("norm", "scale"): ("norm.weight", scale),
        ("output_projection", "kernel"): ("out_proj.weight", kernel),
    }
    for name, bounds in (("query", query), ("key", key), ("value", value)):
        entries[(name, "kernel")] = ("in_proj_qkv.weight", partial(rows, *bounds))
        entries[(f"{name}_convolution", "kernel")] = (
            "conv1d.weight",
            partial(taps, *bounds),
        )
    return {
        ("linear_attention", "cell", *keys): (f"linear_attn.{name}", transform)
        for keys, (name, transform) in entries.items()
    }


ATTENTION = {
    ("attention", "query", "kernel"): ("self_attn.q_proj.weight", partial(half, 0)),
    ("attention", "output_gate", "projection", "kernel"): (
        "self_attn.q_proj.weight",
        partial(half, 1),
    ),
    ("attention", "key", "kernel"): ("self_attn.k_proj.weight", kernel),
    ("attention", "value", "kernel"): ("self_attn.v_proj.weight", kernel),
    ("attention", "output_projection", "kernel"): ("self_attn.o_proj.weight", kernel),
    ("attention", "positional_embedding", "query", "scale"): (
        "self_attn.q_norm.weight",
        centered_scale,
    ),
    ("attention", "positional_embedding", "key", "scale"): (
        "self_attn.k_norm.weight",
        centered_scale,
    ),
}

SHARED = {
    ("attention_norm", "scale"): ("input_layernorm.weight", centered_scale),
    ("swi_glu_norm", "scale"): ("post_attention_layernorm.weight", centered_scale),
    ("swi_glu", "gate", "kernel"): ("mlp.gate_proj.weight", kernel),
    ("swi_glu", "up", "kernel"): ("mlp.up_proj.weight", kernel),
    ("swi_glu", "down", "kernel"): ("mlp.down_proj.weight", kernel),
}


def table(stack, prefix: str) -> dict:
    key_width = stack.linear_num_heads * stack.linear_head_dim
    value_width = stack.linear_num_value_heads * stack.linear_value_head_dim
    layer = {**SHARED, **ATTENTION, **linear_layer(key_width, value_width)}
    entries = {("output_norm", "scale"): (f"{prefix}.norm.weight", centered_scale)}
    for index in range(stack.num_layers):
        for keys, (name, transform) in layer.items():
            entries[(f"layers_{index}", *keys)] = (
                f"{prefix}.layers.{index}.{name}",
                transform,
            )
    return entries


def template(stack) -> dict:
    key = jax.random.key(0)
    carry = stack.initialize_carry(key, (1, stack.features))
    inputs = jax.ShapeDtypeStruct((1, 1, stack.features), jnp.float32)
    done = jax.ShapeDtypeStruct((1, 1), jnp.bool_)
    return jax.eval_shape(stack.init, key, carry, inputs, done)["params"]


def load_qwen3_5(
    stack, state: dict[str, jax.Array], prefix: str = "model.language_model"
) -> dict:
    sources = table(stack, prefix)

    def leaf(path, target):
        keys = tuple(entry.key for entry in path)
        source, transform = sources[keys]
        weight = transform(jnp.asarray(state[source]), target.shape)
        return weight.astype(target.dtype)

    params = jax.tree_util.tree_map_with_path(leaf, template(stack))
    return {"params": params}
