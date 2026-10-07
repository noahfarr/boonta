import jax
import jax.numpy as jnp

from .load_siglip import load_siglip


def kernel(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.T.reshape(shape)


def norm_scale(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return (weight + 1.0).reshape(shape)


def reshape(weight: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return weight.reshape(shape)


GEMMA_LAYER = {
    ("attention_norm", "scale"): ("input_layernorm.weight", norm_scale),
    ("attention", "query", "kernel"): ("self_attn.q_proj.weight", kernel),
    ("attention", "key", "kernel"): ("self_attn.k_proj.weight", kernel),
    ("attention", "value", "kernel"): ("self_attn.v_proj.weight", kernel),
    ("attention", "output_projection", "kernel"): ("self_attn.o_proj.weight", kernel),
    ("ge_glu_norm", "scale"): ("post_attention_layernorm.weight", norm_scale),
    ("ge_glu", "gate", "kernel"): ("mlp.gate_proj.weight", kernel),
    ("ge_glu", "up", "kernel"): ("mlp.up_proj.weight", kernel),
    ("ge_glu", "down", "kernel"): ("mlp.down_proj.weight", kernel),
}

TOP_LEVEL = {
    ("image_embedding", "kernel"): (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
        kernel,
    ),
    ("image_embedding", "bias"): (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias",
        reshape,
    ),
    ("language_embedding", "embedding"): (
        "paligemma_with_expert.paligemma.lm_head.weight",
        reshape,
    ),
    ("state_embedding", "kernel"): ("state_proj.weight", kernel),
    ("state_embedding", "bias"): ("state_proj.bias", reshape),
    ("action_embedding", "kernel"): ("action_in_proj.weight", kernel),
    ("action_embedding", "bias"): ("action_in_proj.bias", reshape),
    ("noisy_action_embedding", "Dense_0", "kernel"): (
        "action_time_mlp_in.weight",
        kernel,
    ),
    ("noisy_action_embedding", "Dense_0", "bias"): ("action_time_mlp_in.bias", reshape),
    ("noisy_action_embedding", "Dense_1", "kernel"): (
        "action_time_mlp_out.weight",
        kernel,
    ),
    ("noisy_action_embedding", "Dense_1", "bias"): (
        "action_time_mlp_out.bias",
        reshape,
    ),
    ("velocity", "kernel"): ("action_out_proj.weight", kernel),
    ("velocity", "bias"): ("action_out_proj.bias", reshape),
}

EXPERT_PREFIXES = (
    "paligemma_with_expert.paligemma.model.language_model",
    "paligemma_with_expert.gemma_expert.model",
)

VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"


def expert_key(index: int, keys: tuple[str, ...]) -> tuple[str, ...]:
    if keys[0] == "attention":
        return ("attention", f"{keys[1]}_{index}", *keys[2:])
    return (f"{keys[0]}_{index}", *keys[1:])


def table(num_layers: int) -> dict:
    sources = dict(TOP_LEVEL)
    for index, prefix in enumerate(EXPERT_PREFIXES):
        sources[(f"output_norm_{index}", "scale")] = (
            f"{prefix}.norm.weight",
            norm_scale,
        )
        for layer in range(num_layers):
            for keys, (name, transform) in GEMMA_LAYER.items():
                sources[(f"layers_{layer}", *expert_key(index, keys))] = (
                    f"{prefix}.layers.{layer}.{name}",
                    transform,
                )
    return sources


def template(
    stack, image_size: int, proprioception_dim: int, language_length: int
) -> dict:
    key = jax.random.key(0)
    obs = {
        "image": jax.ShapeDtypeStruct((1, image_size, image_size, 3), jnp.float32),
        "language": jax.ShapeDtypeStruct((1, language_length), jnp.int32),
        "language_mask": jax.ShapeDtypeStruct((1, language_length), jnp.bool_),
        "proprioception": jax.ShapeDtypeStruct((1, proprioception_dim), jnp.float32),
    }
    noisy_actions = jax.ShapeDtypeStruct(
        (1, stack.action_horizon, stack.action_dim), jnp.float32
    )
    time = jax.ShapeDtypeStruct((1,), jnp.float32)
    return jax.eval_shape(stack.init, key, obs, noisy_actions, time)["params"]


def load_pi0(
    stack,
    state: dict[str, jax.Array],
    image_size: int,
    proprioception_dim: int,
    language_length: int,
) -> dict:
    sources = table(stack.num_layers)

    def leaf(path, target):
        keys = tuple(entry.key for entry in path)
        source, transform = sources[keys]
        weight = transform(jnp.asarray(state[source]), target.shape)
        return weight.astype(target.dtype)

    full_template = template(stack, image_size, proprioception_dim, language_length)
    rest_template = {k: v for k, v in full_template.items() if k != "vision"}

    params = dict(jax.tree_util.tree_map_with_path(leaf, rest_template))
    params["vision"] = load_siglip(
        stack.vision, state, image_size, prefix=VISION_PREFIX
    )["params"]
    return {"params": params}
