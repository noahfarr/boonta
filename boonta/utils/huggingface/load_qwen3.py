import jax
import jax.numpy as jnp


def kernel(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.T.reshape(shape)


def scale(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.reshape(shape)


LAYER = {
    ("attention_norm", "scale"): ("input_layernorm.weight", scale),
    ("swi_glu_norm", "scale"): ("post_attention_layernorm.weight", scale),
    ("attention", "query", "kernel"): ("self_attn.q_proj.weight", kernel),
    ("attention", "key", "kernel"): ("self_attn.k_proj.weight", kernel),
    ("attention", "value", "kernel"): ("self_attn.v_proj.weight", kernel),
    ("attention", "output_projection", "kernel"): ("self_attn.o_proj.weight", kernel),
    ("attention", "positional_embedding", "query", "scale"): (
        "self_attn.q_norm.weight",
        scale,
    ),
    ("attention", "positional_embedding", "key", "scale"): (
        "self_attn.k_norm.weight",
        scale,
    ),
    ("swi_glu", "gate", "kernel"): ("mlp.gate_proj.weight", kernel),
    ("swi_glu", "up", "kernel"): ("mlp.up_proj.weight", kernel),
    ("swi_glu", "down", "kernel"): ("mlp.down_proj.weight", kernel),
}

ROOT = {
    ("output_norm", "scale"): ("model.norm.weight", scale),
}


def table(num_layers: int) -> dict:
    entries = dict(ROOT)
    for layer in range(num_layers):
        for keys, (name, transform) in LAYER.items():
            entries[(f"layers_{layer}", *keys)] = (
                f"model.layers.{layer}.{name}",
                transform,
            )
    return entries


def template(stack) -> dict:
    key = jax.random.key(0)
    carry = stack.initialize_carry(key, (1, stack.features))
    inputs = jax.ShapeDtypeStruct((1, 1, stack.features), jnp.float32)
    done = jax.ShapeDtypeStruct((1, 1), jnp.bool_)
    return jax.eval_shape(stack.init, key, carry, inputs, done)["params"]


def load_qwen3(stack, state: dict[str, jax.Array]) -> dict:
    sources = table(stack.num_layers)

    def leaf(path, target):
        keys = tuple(entry.key for entry in path)
        source, transform = sources[keys]
        weight = transform(jnp.asarray(state[source]), target.shape)
        return weight.astype(target.dtype)

    params = jax.tree_util.tree_map_with_path(leaf, template(stack))
    return {"params": params}
