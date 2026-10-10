from functools import partial

import jax
import jax.numpy as jnp


def kernel(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.T.reshape(shape)


def scale(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.reshape(shape)


def centered_scale(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return (weight.astype(jnp.float32) + 1.0).reshape(shape)


def first(total: int, width: int) -> int:
    return 0


def second(total: int, width: int) -> int:
    return width


def last(total: int, width: int) -> int:
    return total - width


def rows(offset, weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    start = offset(weight.shape[0], shape[-1])
    return weight[start : start + shape[-1]].T.reshape(shape)


def taps(offset, weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    start = offset(weight.shape[0], shape[-1])
    return weight[start : start + shape[-1], 0].T.reshape(shape)


def head_chunk(chunk: int, weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    features, num_heads, head_dim = shape
    return weight.T.reshape(features, num_heads, 2, head_dim)[:, :, chunk]


DELTA = {
    ("gate", "kernel"): ("in_proj_z.weight", kernel),
    ("alpha", "kernel"): ("in_proj_a.weight", kernel),
    ("beta", "kernel"): ("in_proj_b.weight", kernel),
    ("log_rate",): ("A_log", scale),
    ("step_bias",): ("dt_bias", scale),
    ("norm", "scale"): ("norm.weight", scale),
    ("output_projection", "kernel"): ("out_proj.weight", kernel),
    **{
        (name, "kernel"): ("in_proj_qkv.weight", partial(rows, offset))
        for name, offset in (("query", first), ("key", second), ("value", last))
    },
    **{
        (f"{name}_convolution", "kernel"): ("conv1d.weight", partial(taps, offset))
        for name, offset in (("query", first), ("key", second), ("value", last))
    },
}

ATTENTION = {
    ("query", "kernel"): ("q_proj.weight", partial(head_chunk, 0)),
    ("output_gate", "projection", "kernel"): ("q_proj.weight", partial(head_chunk, 1)),
    ("key", "kernel"): ("k_proj.weight", kernel),
    ("value", "kernel"): ("v_proj.weight", kernel),
    ("output_projection", "kernel"): ("o_proj.weight", kernel),
    ("positional_embedding", "query", "scale"): ("q_norm.weight", centered_scale),
    ("positional_embedding", "key", "scale"): ("k_norm.weight", centered_scale),
}

MIXER = {
    ("blocks", "blocks_0", "layer", "scale"): ("input_layernorm.weight", centered_scale),
    **{
        ("blocks", "blocks_1", "cell", *keys): (f"linear_attn.{name}", transform)
        for keys, (name, transform) in DELTA.items()
    },
    **{
        ("blocks", "blocks_1", *keys): (f"self_attn.{name}", transform)
        for keys, (name, transform) in ATTENTION.items()
    },
}

FEEDFORWARD = {
    ("blocks", "blocks_0", "layer", "scale"): (
        "post_attention_layernorm.weight",
        centered_scale,
    ),
    ("blocks", "blocks_1", "gate", "kernel"): ("mlp.gate_proj.weight", kernel),
    ("blocks", "blocks_1", "up", "kernel"): ("mlp.up_proj.weight", kernel),
    ("blocks", "blocks_1", "down", "kernel"): ("mlp.down_proj.weight", kernel),
}


def template(stack, features: int) -> dict:
    key = jax.random.key(0)
    carry = stack.initialize_carry(key, (1, features))
    inputs = jax.ShapeDtypeStruct((1, 1, features), jnp.float32)
    done = jax.ShapeDtypeStruct((1, 1), jnp.bool_)
    return jax.eval_shape(stack.init, key, carry, inputs, done)["params"]


def source(keys: tuple[str, ...], num_layers: int, prefix: str):
    block, *rest = keys
    index = int(block.removeprefix("blocks_"))
    if index == 2 * num_layers:
        return f"{prefix}.norm.weight", centered_scale
    layer, side = divmod(index, 2)
    name, transform = (MIXER, FEEDFORWARD)[side][tuple(rest)]
    return f"{prefix}.layers.{layer}.{name}", transform


def load_qwen3_5(
    stack, state: dict[str, jax.Array], prefix: str = "model.language_model"
) -> dict:
    features = state[f"{prefix}.embed_tokens.weight"].shape[1]
    params = template(stack, features)
    num_layers = len(params) // 2

    def leaf(path, target):
        name, transform = source(tuple(entry.key for entry in path), num_layers, prefix)
        weight = transform(jnp.asarray(state[name]), target.shape)
        return weight.astype(target.dtype)

    return {"params": jax.tree_util.tree_map_with_path(leaf, params)}
