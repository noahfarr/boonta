from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np
from flax import struct

from boonta.utils import Key, Timestep, Transition, canonicalize_dtype

FIELDS = ("obs", "action", "reward", "terminated", "truncated")


@struct.dataclass(frozen=True)
class DiskState:
    transitions: Transition = struct.field(metadata={"axis": "data"})


def dump(path: Path, tree) -> None:
    if isinstance(tree, dict):
        for name, subtree in tree.items():
            dump(path / name, subtree)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.parent / f"{path.name}.npy", tree.astype(canonicalize_dtype(tree.dtype)))


def mount(path: Path):
    file = path.parent / f"{path.name}.npy"
    if file.exists():
        return np.load(file, mmap_mode="r")
    names = [child.name.removesuffix(".npy") for child in sorted(path.iterdir())]
    return {name: mount(path / name) for name in names}


def write(directory: str | Path, episodes) -> None:
    columns = {name: [] for name in FIELDS}
    starts, size = [], 0
    for episode in episodes:
        starts.append(size)
        size += len(episode.rewards)
        columns["obs"].append(episode.observations)
        columns["action"].append(episode.actions)
        columns["reward"].append(episode.rewards)
        columns["terminated"].append(episode.terminations)
        columns["truncated"].append(episode.truncations)

    directory = Path(directory)
    for name, values in columns.items():
        dump(directory / name, jax.tree.map(lambda *leaves: np.concatenate(leaves), *values))
    np.save(directory / "starts.npy", np.asarray(starts, np.int64))


class Disk:
    def __init__(self, directory: str | Path, pool_size: int = 0, num_devices: int = 1):
        directory = Path(directory)
        self.columns = {name: mount(directory / name) for name in FIELDS}
        self.starts = np.load(directory / "starts.npy")
        self.size = len(self.columns["reward"])
        whole = not 0 < pool_size < self.size
        self.rows = (self.size if whole else pool_size) // num_devices * num_devices
        self.transitions = self.gather(np.arange(self.rows)) if whole else None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="disk")
        self.pending = None

    def gather(self, index: np.ndarray) -> Transition:
        episode = np.searchsorted(self.starts, index, side="right") - 1
        start = index == self.starts[episode]
        row = index + episode
        before = np.maximum(index - 1, 0)
        obs, action, reward, terminated, truncated = (
            self.columns[name] for name in FIELDS
        )

        def prior(leaf):
            values = np.asarray(leaf[before])
            mask = start.reshape(start.shape + (1,) * (values.ndim - 1))
            return np.where(mask, np.zeros_like(values), values)

        return Transition(
            first=Timestep(
                obs=jax.tree.map(lambda leaf: np.asarray(leaf[row]), obs),
                action=jax.tree.map(prior, action),
                reward=prior(reward),
                terminated=start,
                truncated=np.zeros_like(start),
            ),
            second=Timestep(
                obs=jax.tree.map(lambda leaf: np.asarray(leaf[row + 1]), obs),
                action=jax.tree.map(lambda leaf: np.asarray(leaf[index]), action),
                reward=np.asarray(reward[index]),
                terminated=np.asarray(terminated[index]),
                truncated=np.asarray(truncated[index]),
            ),
        )

    def stage(self, key: Key, sharding) -> DiskState:
        generator = np.random.default_rng(np.asarray(jax.random.key_data(key)))
        index = np.sort(generator.choice(self.size, self.rows, replace=False))
        return jax.device_put(DiskState(self.gather(index)), sharding)

    def init(self) -> DiskState:
        if self.transitions is not None:
            return DiskState(self.transitions)
        empty = self.gather(np.arange(0))
        return DiskState(
            jax.tree.map(
                lambda leaf: np.zeros((self.rows, *leaf.shape[1:]), leaf.dtype), empty
            )
        )

    def update(self, state: DiskState, key: Key, sharding) -> DiskState:
        if self.transitions is not None:
            return state
        if self.pending is None:
            self.pending = self.executor.submit(self.stage, key, sharding)
        state = self.pending.result()
        self.pending = self.executor.submit(
            self.stage, jax.random.fold_in(key, 1), sharding
        )
        return state

    def sample(
        self, state: DiskState, key: Key, batch_shape: tuple[int, ...]
    ) -> Transition:
        assert len(batch_shape) == 1, (
            f"disk samples single transitions, so batch_shape must be "
            f"(batch,), got {batch_shape}"
        )
        index = jax.random.randint(key, batch_shape, 0, self.rows)
        return jax.tree.map(lambda x: x[index], state.transitions)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


def make(dataset_id: str, pool_size: int = 0, num_devices: int = 1, **kwargs) -> Disk:
    return Disk(dataset_id, pool_size, num_devices)
