from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
from flax import struct

from boonta.utils import Key, Timestep, Transition, canonicalize_dtype


@struct.dataclass(frozen=True)
class MinariState:
    transitions: Transition = struct.field(metadata={"axis": "data"})


class Minari:
    def __init__(self, dataset_id: str, pool_size: int = 0, num_devices: int = 1):
        import minari

        self.episodes = minari.load_dataset(dataset_id, download=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pool")
        self.pending = None
        whole = pool_size >= self.episodes.total_steps
        self.pool_size = 0 if whole else pool_size // num_devices * num_devices
        if self.pool_size:
            self.transitions = jax.tree.map(
                lambda leaf: np.zeros((self.pool_size, *leaf.shape[1:]), leaf.dtype),
                jax.tree.map(stack, convert(self.episodes[0])),
            )
        else:
            rows = self.episodes.total_steps // num_devices * num_devices
            self.transitions = load(self.episodes, self.episodes.episode_indices, rows)

    def init(self) -> MinariState:
        return MinariState(self.transitions)

    def stage(self, key: Key, sharding) -> MinariState:
        generator = np.random.default_rng(np.asarray(jax.random.key_data(key)))
        indices = generator.permutation(self.episodes.episode_indices)
        return jax.device_put(
            MinariState(load(self.episodes, indices, self.pool_size)), sharding
        )

    def update(self, state: MinariState, key: Key, sharding) -> MinariState:
        if not self.pool_size:
            return state
        if self.pending is None:
            self.pending = self.executor.submit(self.stage, key, sharding)
        state = self.pending.result()
        self.pending = self.executor.submit(
            self.stage, jax.random.fold_in(key, 1), sharding
        )
        return state

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def sample(
        self, state: MinariState, key: Key, batch_shape: tuple[int, ...]
    ) -> Transition:
        assert len(batch_shape) == 1, (
            f"minari samples single transitions, so batch_shape must be "
            f"(batch,), got {batch_shape}"
        )
        leaf, *_ = jax.tree.leaves(state.transitions)
        size, *_ = leaf.shape
        index = jax.random.randint(key, batch_shape, 0, size)
        return jax.tree.map(lambda x: x[index], state.transitions)


def convert(episode) -> Transition:
    def prior(leaf):
        leaf = np.asarray(leaf)
        return np.concatenate([np.zeros_like(leaf[:1]), leaf[:-1]])

    starts = np.zeros_like(np.asarray(episode.terminations))
    starts[0] = True
    return Transition(
        first=Timestep(
            obs=jax.tree.map(lambda leaf: leaf[:-1], episode.observations),
            action=jax.tree.map(prior, episode.actions),
            reward=prior(episode.rewards),
            terminated=starts,
            truncated=np.zeros_like(starts),
        ),
        second=Timestep(
            obs=jax.tree.map(lambda leaf: leaf[1:], episode.observations),
            action=episode.actions,
            reward=episode.rewards,
            terminated=episode.terminations,
            truncated=episode.truncations,
        ),
    )


def stack(*leaves) -> np.ndarray:
    joined = np.concatenate(leaves)
    return joined.astype(canonicalize_dtype(joined.dtype))


def load(episodes, indices, rows: int) -> Transition:
    chosen, size = [], 0
    for episode in episodes.iterate_episodes(indices):
        chosen.append(convert(episode))
        size += len(episode.rewards)
        if size >= rows:
            break
    return jax.tree.map(lambda *leaves: stack(*leaves)[:rows], *chosen)
