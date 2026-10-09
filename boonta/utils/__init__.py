from typing import Literal

import jax
import jax.numpy as jnp
import lox

from .axis import (
    add_batch_axis,
    add_feature_axis,
    add_time_axis,
    broadcast,
    concatenate,
    remove_batch_axis,
    remove_feature_axis,
    remove_time_axis,
    flatten,
    place,
    take,
)
from .huggingface import (
    load_config,
    load_gemma,
    load_pi0,
    load_qwen3,
    load_qwen3_5,
    load_siglip,
    load_template,
    load_tokenizer,
    load_vocab,
    load_weights,
)
from .lift import vmap
from .quantization import (
    QuantizedArray,
    dequantize,
    materialize_gradients,
    quantize,
    quantized_dot_general,
    quantizing,
)
from .checkpoint import load_checkpoint, newest
from .system_monitor import SystemMonitor
from .sharding import mesh, sharded, vary
from .timestep import Timestep
from .transition import Transition
from .typing import Array, Key, PyTree

__all__ = [
    "QuantizedArray",
    "Timestep",
    "Transition",
    "add_batch_axis",
    "add_feature_axis",
    "add_time_axis",
    "broadcast",
    "canonicalize_dtype",
    "concatenate",
    "conditional_update",
    "dequantize",
    "get_attention_implementation",
    "load_checkpoint",
    "newest",
    "SystemMonitor",
    "load_config",
    "load_gemma",
    "load_pi0",
    "load_qwen3",
    "load_qwen3_5",
    "load_siglip",
    "load_template",
    "load_tokenizer",
    "load_vocab",
    "load_weights",
    "log_schedule",
    "mesh",
    "place",
    "quantize",
    "quantized_dot_general",
    "quantizing",
    "remove_batch_axis",
    "remove_feature_axis",
    "remove_time_axis",
    "sharded",
    "take",
    "vary",
    "vmap",
]


def canonicalize_dtype(dtype):
    return jax.dtypes.canonicalize_dtype(dtype)


def conditional_update(new_tensors, old_tensors, condition):
    return jax.tree.map(
        lambda new, old: jnp.where(condition, new, old), new_tensors, old_tensors
    )


def get_attention_implementation(
    head_dim, *lengths
) -> tuple[Literal["xla", "cudnn"], jnp.dtype]:
    capabilities = [
        float(device.compute_capability)
        for device in jax.local_devices()
        if device.platform == "gpu" and "nvidia" in device.device_kind.lower()
    ]
    if not capabilities:
        return "xla", jnp.float32
    capability = min(capabilities)
    limit = 256 if capability >= 9.0 else 128
    if (
        capability >= 8.0
        and head_dim % 8 == 0
        and head_dim <= limit
        and all(length % 2 == 0 for length in lengths)
    ):
        return "cudnn", jnp.bfloat16
    return "xla", jnp.float32


def log_schedule(schedule, prefix):
    def wrapped(count):
        value = schedule(count)
        lox.log({prefix: value})
        return value

    return wrapped
