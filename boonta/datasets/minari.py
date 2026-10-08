import jax
import numpy as np
from flax import struct

from boonta.utils import Key, Timestep, Transition, canonicalize_dtype

from .pool import Still, Stream


@struct.dataclass(frozen=True)
class MinariState:
    transitions: Transition = struct.field(metadata={"axis": "data"})


class Minari:
    def __init__(self, transitions: Transition, stream: Still | Stream = Still()):
        self.transitions = transitions
        self.stream = stream

    def init(self) -> MinariState:
        return MinariState(self.transitions)

    def update(self, state: MinariState, key: Key, sharding) -> MinariState:
        return self.stream.update(state, key, sharding)

    def close(self) -> None:
        self.stream.close()

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


def trim(transitions: Transition, rows: int) -> Transition:
    return jax.tree.map(lambda leaf: leaf[:rows], transitions)


def gather(episodes, num_devices: int = 1) -> Transition:
    transitions = jax.tree.map(
        stack, *[convert(e) for e in episodes.iterate_episodes()]
    )
    leaf, *_ = jax.tree.leaves(transitions)
    return trim(transitions, len(leaf) // num_devices * num_devices)


def fill(episodes, generator: np.random.Generator, rows: int) -> Transition:
    chosen, size = [], 0
    for episode in episodes.iterate_episodes(
        generator.permutation(episodes.episode_indices)
    ):
        chosen.append(convert(episode))
        size += len(episode.rewards)
        if size >= rows:
            break
    return trim(jax.tree.map(stack, *chosen), rows)


def load(dataset_id: str, num_devices: int = 1) -> Transition:
    import minari

    return gather(minari.load_dataset(dataset_id, download=True), num_devices)


def pool(episodes, pool_size: int, num_devices: int = 1) -> Minari:
    rows = pool_size // num_devices * num_devices
    blank = jax.tree.map(
        lambda leaf: np.zeros((rows, *leaf.shape[1:]), leaf.dtype),
        fill(episodes, np.random.default_rng(0), 1),
    )
    return Minari(
        blank, Stream(lambda generator: MinariState(fill(episodes, generator, rows)))
    )


def make(
    dataset_id: str, pool_size: int = 0, num_devices: int = 1, **kwargs
) -> Minari:
    import minari

    episodes = minari.load_dataset(dataset_id, download=True)
    if 0 < pool_size < episodes.total_steps:
        return pool(episodes, pool_size, num_devices)
    return Minari(gather(episodes, num_devices))
