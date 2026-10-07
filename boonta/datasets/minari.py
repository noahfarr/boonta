import jax
import numpy as np
from flax import struct

from boonta.utils import Key, Timestep, Transition, canonicalize_dtype

from .disk import write


@struct.dataclass(frozen=True)
class MinariState:
    transitions: Transition = struct.field(metadata={"axis": "data"})


@struct.dataclass(frozen=True)
class Minari:
    transitions: Transition

    def init(self) -> MinariState:
        return MinariState(self.transitions)

    def update(self, state: MinariState, key: Key, sharding) -> MinariState:
        return state

    def close(self) -> None:
        pass

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


def load(dataset_id: str, num_devices: int = 1) -> Transition:
    import minari

    episodes = minari.load_dataset(dataset_id, download=True)
    transitions = jax.tree.map(
        stack, *[convert(e) for e in episodes.iterate_episodes()]
    )
    leaf, *_ = jax.tree.leaves(transitions)
    rows = len(leaf) // num_devices * num_devices
    return jax.tree.map(lambda leaf: leaf[:rows], transitions)


def export(dataset_id: str, directory: str) -> None:
    import minari

    write(directory, minari.load_dataset(dataset_id, download=True).iterate_episodes())


def make(dataset_id: str, num_devices: int = 1, **kwargs) -> Minari:
    return Minari(load(dataset_id, num_devices))
