"""Tiled categorical likelihoods with the same Dense parameters and logits.

Only the computation changes: no extra parameters, tying, or prediction heads.
The projection is rematerialized per tile during backward to bound V-sized
activation memory. Sampling is intended for a single next-token position.
"""
import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class ChunkedDistribution:
    features: object
    kernel: object
    bias: object
    temperature: object
    chunk_size: int = struct.field(pytree_node=False, default=128)

    def project(self, features):
        logits = features @ self.kernel
        if self.bias is not None:
            logits = logits + self.bias
        logits = logits.astype(jnp.float32)
        return logits / jnp.where(self.temperature > 0, self.temperature, 1.0)

    def statistics(self, targets=None):
        shape = self.features.shape[:-1]
        x = self.features.reshape(-1, self.features.shape[-1])
        size = x.shape[0]
        padding = (-size) % self.chunk_size
        x = jnp.pad(x, ((0, padding), (0, 0))).reshape(-1, self.chunk_size, x.shape[-1])
        y = jnp.zeros(size, jnp.int32) if targets is None else targets.reshape(-1)
        y = jnp.pad(y, (0, padding)).reshape(-1, self.chunk_size)

        @jax.checkpoint
        def tile(_, inputs):
            features, actions = inputs
            logp = jax.nn.log_softmax(self.project(features), axis=-1)
            selected = jnp.take_along_axis(logp, actions[:, None], axis=-1)[:, 0]
            entropy = -(jnp.exp(logp) * logp).sum(-1)
            return None, (selected, entropy)

        _, (selected, entropy) = jax.lax.scan(tile, None, (x, y))
        return (selected.reshape(-1)[:size].reshape(shape),
                entropy.reshape(-1)[:size].reshape(shape))

    def log_prob(self, targets):
        return self.statistics(targets)[0]

    def entropy(self):
        return self.statistics()[1]

    def at(self, positions):
        """Select per-row prefix endpoints before projecting the vocabulary."""
        features = self.features[jnp.arange(self.features.shape[0]), positions]
        return self.replace(features=features)

    @property
    def logits(self):
        if self.features.size // self.features.shape[-1] > self.chunk_size:
            raise ValueError("Select a next-token position before materializing logits")
        return self.project(self.features)

    def sample(self, seed):
        logits = self.logits
        return jnp.where(self.temperature > 0, jax.random.categorical(seed, logits),
                         jnp.argmax(logits, axis=-1)).astype(jnp.int32)

    def sample_and_log_prob(self, seed):
        action = self.sample(seed)
        logp = jax.nn.log_softmax(self.logits, axis=-1)
        return action, jnp.take_along_axis(logp, action[..., None], -1)[..., 0]

    def kl_divergence(self, other):
        if self.features.shape != other.features.shape:
            raise ValueError("KL requires identical history positions")
        shape = self.features.shape[:-1]
        x = self.features.reshape(-1, self.features.shape[-1])
        y = other.features.reshape(x.shape)
        size = len(x)
        padding = (-size) % self.chunk_size
        tiles = lambda z: jnp.pad(z, ((0, padding), (0, 0))).reshape(-1, self.chunk_size, z.shape[-1])
        def tile(_, inputs):
            a, b = inputs
            p = jax.nn.log_softmax(self.project(a), -1)
            q = jax.nn.log_softmax(other.project(b), -1)
            return None, (jnp.exp(p) * (p-q)).sum(-1)
        _, kl = jax.lax.scan(jax.checkpoint(tile), None, (tiles(x), tiles(y)))
        return kl.reshape(-1)[:size].reshape(shape)


class ChunkedCategorical(nn.Module):
    layer: nn.Dense
    chunk_size: int = 128

    def setup(self):
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        nn.share_scope(self, self.layer)

    def __call__(self, x, temperature=1.0, **kwargs):
        # Initialize/access the existing Dense scope on a single position.
        self.layer(x[..., :1, :], **kwargs)
        kernel = self.get_variable('params', 'kernel').astype(x.dtype)
        bias = self.get_variable('params', 'bias') if self.layer.use_bias else None
        if bias is not None:
            bias = bias.astype(x.dtype)
        return ChunkedDistribution(x, kernel, bias, jnp.asarray(temperature), self.chunk_size)
