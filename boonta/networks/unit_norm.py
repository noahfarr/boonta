import jax
import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict, unflatten_dict

from boonta.utils.typing import PyTree


def normalize_dense(params: dict, path: tuple) -> None:
    kernel_path = path + ("kernel",)
    kernel = params[kernel_path]
    normalize = lambda k: k / jnp.linalg.norm(k, axis=0, keepdims=True)

    if kernel.ndim > 2:
        params[kernel_path] = jax.vmap(normalize)(kernel)
        return

    normalize(kernel)


def normalize_batch_norm(params: dict, path: tuple) -> None:
    scale_key, bias_key = path + ("scale",), path + ("bias",)
    scale, bias = params[scale_key], params[bias_key]
    *_, dim = scale.shape
    factor = jnp.sqrt(dim) * jax.lax.rsqrt(
        jnp.sum(scale**2 + bias**2, axis=-1, keepdims=True) + 1e-8
    )
    params[scale_key] = scale * factor
    params[bias_key] = bias * factor


def normalize_rms_norm(params: dict, path: tuple) -> None:
    scale_key = path + ("scale",)

    scale = params[scale_key]
    *_, dim = scale.shape
    factor = jnp.sqrt(dim) * jax.lax.rsqrt(
        jnp.sum(scale**2, axis=-1, keepdims=True) + 1e-8
    )
    params[scale_key] = scale * factor


NORMALIZERS = {
    "Dense": normalize_dense,
    "BatchNorm": normalize_batch_norm,
    "RMSNorm": normalize_rms_norm,
}


def normalize(params_by_path: dict, path: tuple) -> bool:
    for component in path:
        prefix, *_ = component.split("_")
        if prefix in NORMALIZERS:
            NORMALIZERS[prefix](params_by_path, path)
            return True
    return False


def unit_norm_params(params: PyTree, root: str | None = None) -> PyTree:
    target = params
    if root is not None:
        target = params[root]
    params_by_path = dict(flatten_dict(target))
    paths = sorted({path[:-1] for path in params_by_path})
    matched = False
    for path in paths:
        if normalize(params_by_path, path):
            matched = True
    if not matched:
        raise ValueError(
            f"unit-norm projection matched no {sorted(NORMALIZERS)} layers"
        )
    normalized = unflatten_dict(params_by_path)
    if root is None:
        return normalized
    return {**params, root: normalized}


def unit_norm_weights(root: str | None = None) -> optax.GradientTransformation:
    def update_fn(updates: PyTree, params: PyTree) -> PyTree:
        new_params = optax.apply_updates(params, updates)
        projected = unit_norm_params(new_params, root)
        return jax.tree.map(lambda p, proj: proj - p, params, projected)

    return optax.stateless(update_fn)
