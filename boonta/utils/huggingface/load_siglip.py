import jax
import jax.numpy as jnp


def reshape(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.reshape(shape)


def kernel(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.T.reshape(shape)


def conv_kernel(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return jnp.transpose(weight, (2, 3, 1, 0)).reshape(shape)


LAYER = {
    ("attention_norm", "scale"): ("layer_norm1.weight", reshape),
    ("attention_norm", "bias"): ("layer_norm1.bias", reshape),
    ("attention", "query", "kernel"): ("self_attn.q_proj.weight", kernel),
    ("attention", "query", "bias"): ("self_attn.q_proj.bias", reshape),
    ("attention", "key", "kernel"): ("self_attn.k_proj.weight", kernel),
    ("attention", "key", "bias"): ("self_attn.k_proj.bias", reshape),
    ("attention", "value", "kernel"): ("self_attn.v_proj.weight", kernel),
    ("attention", "value", "bias"): ("self_attn.v_proj.bias", reshape),
    ("attention", "output_projection", "kernel"): ("self_attn.out_proj.weight", kernel),
    ("attention", "output_projection", "bias"): ("self_attn.out_proj.bias", reshape),
    ("ffn_norm", "scale"): ("layer_norm2.weight", reshape),
    ("ffn_norm", "bias"): ("layer_norm2.bias", reshape),
    ("ffn", "Dense_0", "kernel"): ("mlp.fc1.weight", kernel),
    ("ffn", "Dense_0", "bias"): ("mlp.fc1.bias", reshape),
    ("ffn", "Dense_1", "kernel"): ("mlp.fc2.weight", kernel),
    ("ffn", "Dense_1", "bias"): ("mlp.fc2.bias", reshape),
}

ROOT = {
    ("output_norm", "scale"): ("post_layernorm.weight", reshape),
    ("output_norm", "bias"): ("post_layernorm.bias", reshape),
    ("patch_embedding", "projection", "kernel"): (
        "embeddings.patch_embedding.weight",
        conv_kernel,
    ),
    ("patch_embedding", "projection", "bias"): (
        "embeddings.patch_embedding.bias",
        reshape,
    ),
    ("patch_embedding", "position_embedding"): (
        "embeddings.position_embedding.weight",
        reshape,
    ),
}


def table(num_layers: int, prefix: str = "vision_model") -> dict:
    entries = {
        keys: (f"{prefix}.{name}", transform)
        for keys, (name, transform) in ROOT.items()
    }
    for layer in range(num_layers):
        for keys, (name, transform) in LAYER.items():
            entries[(f"layers_{layer}", *keys)] = (
                f"{prefix}.encoder.layers.{layer}.{name}",
                transform,
            )
    return entries


def template(stack, image_size: int) -> dict:
    key = jax.random.key(0)
    carry = stack.initialize_carry(key, (1, image_size, image_size, 3))
    image = jax.ShapeDtypeStruct((1, image_size, image_size, 3), jnp.float32)
    done = jax.ShapeDtypeStruct((1,), jnp.bool_)
    return jax.eval_shape(stack.init, key, carry, image, done)["params"]


def load_siglip(
    stack, state: dict[str, jax.Array], image_size: int, prefix: str = "vision_model"
) -> dict:
    sources = table(stack.num_layers, prefix)

    def leaf(path, target):
        keys = tuple(entry.key for entry in path)
        source, transform = sources[keys]
        weight = transform(jnp.asarray(state[source]), target.shape)
        return weight.astype(target.dtype)

    params = jax.tree_util.tree_map_with_path(leaf, template(stack, image_size))
    return {"params": params}
