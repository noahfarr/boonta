import flax.linen as nn
import jax
import numpy as np
from flax.core import unfreeze

from boonta.utils.typing import PyTree


def init_fn(weights: PyTree):
    def load(*args):
        host = jax.tree.map(np.asarray, unfreeze(weights))
        shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), host)
        return jax.pure_callback(lambda: host, shape, vmap_method="broadcast_all")

    return load


class Pretrained(nn.Module):
    module: nn.Module
    weights: PyTree

    @nn.compact
    def __call__(self, *args, **kwargs):
        params = self.variable("params", "pretrained", init_fn(self.weights)).value
        return self.module.apply({"params": params}, *args, **kwargs)
