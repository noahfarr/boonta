import jax
import jax.numpy as jnp
from flax import struct

from .typing import Array, PyTree


@struct.dataclass(frozen=True)
class EMA:
    mean: PyTree
    var: PyTree
    steps: Array
    decay: float = struct.field(pytree_node=False, default=0.99)

    @classmethod
    def blank(cls, template: PyTree, axis: tuple[int, ...], decay: float = 0.99):
        keep = lambda leaf: tuple(
            size for index, size in enumerate(jnp.shape(leaf)) if index not in axis
        )
        return cls(
            mean=jax.tree.map(lambda leaf: jnp.zeros(keep(leaf), jnp.float32), template),
            var=jax.tree.map(lambda leaf: jnp.zeros(keep(leaf), jnp.float32), template),
            steps=jnp.array(0, jnp.float32),
            decay=decay,
        )

    def absorb(self, batch: PyTree, axis: tuple[int, ...]) -> "EMA":
        blend = lambda held, fresh: self.decay * held + (1.0 - self.decay) * fresh
        return self.replace(
            mean=jax.tree.map(
                lambda leaf, held: blend(held, jnp.mean(leaf, axis=axis)),
                batch,
                self.mean,
            ),
            var=jax.tree.map(
                lambda leaf, held: blend(held, jnp.var(leaf, axis=axis)),
                batch,
                self.var,
            ),
            steps=self.steps + 1.0,
        )

    def settled(self) -> tuple[PyTree, PyTree]:
        correction = 1.0 - self.decay ** jnp.maximum(self.steps, 1.0)
        return (
            jax.tree.map(lambda leaf: leaf / correction, self.mean),
            jax.tree.map(lambda leaf: leaf / correction, self.var),
        )

    def whiten(self, batch: PyTree, clip: float = jnp.inf) -> PyTree:
        mean, var = self.settled()
        return jax.tree.map(
            lambda leaf, mu, sigma: jnp.clip(
                (leaf - mu) / jnp.sqrt(sigma + 1e-8), -clip, clip
            ),
            batch,
            mean,
            var,
        )

    def spread(self) -> PyTree:
        _, var = self.settled()
        return jax.tree.map(lambda leaf: jnp.sqrt(leaf + 1e-8), var)

    def scale(self, batch: Array) -> Array:
        return batch / self.spread()
