import flax.linen as nn
import jax

from boonta.utils import QuantizedArray
from boonta.utils.typing import PyTree


class LoRA(nn.Module):
    rank: int = 4
    alpha: float = 1.0
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, params: PyTree) -> PyTree:
        def adapt_leaf(path, leaf):
            if isinstance(leaf, QuantizedArray):
                leaf = leaf.dequantize()
            if leaf.ndim != 2:
                return jax.lax.stop_gradient(leaf)
            name = "_".join(key.key for key in path)
            in_features, out_features = leaf.shape
            a = self.param(f"{name}_a", self.kernel_init, (in_features, self.rank))
            b = self.param(f"{name}_b", self.bias_init, (self.rank, out_features))
            delta = ((self.alpha / self.rank) * (a @ b)).astype(leaf.dtype)
            return jax.lax.stop_gradient(leaf) + delta

        return jax.tree_util.tree_map_with_path(
            adapt_leaf, params, is_leaf=lambda leaf: isinstance(leaf, QuantizedArray)
        )
