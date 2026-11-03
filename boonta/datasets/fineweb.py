import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.utils import Timestep, Transition


@struct.dataclass(frozen=True)
class FineWebState:
    transitions: Transition = struct.field(metadata={"axis": "data"})


class FineWeb:
    def __init__(self, dataset_id, length=1024, pool_size=1200, seed=42):
        self.tokens = np.memmap(dataset_id, dtype="<u2", mode="r")
        token_count, *_ = self.tokens.shape
        if token_count <= length:
            raise ValueError("Token file is shorter than one training window")
        self.length, self.pool_size, self.seed = length, pool_size, seed

    def pool(self, seed):
        token_count, *_ = self.tokens.shape
        starts = np.random.default_rng(seed).integers(
            0, token_count - self.length, self.pool_size
        )
        windows = np.stack(
            [self.tokens[start : start + self.length + 1] for start in starts]
        ).astype(np.int32)
        shape = (self.pool_size, self.length)
        terminated = np.zeros(shape, bool)
        terminated[:, 0] = True
        return FineWebState(
            Transition(
                first=Timestep(
                    obs=windows[:, :-1],
                    action=np.zeros(shape, np.int32),
                    reward=np.zeros(shape, np.float32),
                    terminated=terminated,
                    truncated=np.zeros(shape, bool),
                ),
                second=Timestep(obs=None, action=windows[:, 1:]),
                aux={},
            )
        )

    def init(self):
        return self.pool(self.seed)

    def update(self, state, key, sharding):
        seed = int(np.asarray(jax.random.bits(key, (), dtype=np.uint32)))
        return jax.device_put(self.pool(seed), sharding)

    def sample(self, state, key, batch_shape, *, step=None):
        batch_size, length = batch_shape
        if length != self.length:
            raise ValueError(f"FineWeb expects (batch, {self.length}), got {batch_shape}")
        start = (step // self.length) % self.pool_size
        indices = start + jnp.arange(batch_size)
        return jax.tree.map(lambda leaf: jnp.asarray(leaf)[indices], state.transitions)

    def close(self):
        pass
