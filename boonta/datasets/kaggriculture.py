from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax.experimental import io_callback

from boonta.utils import Key, Timestep, Transition
from boonta.utils.typing import Array

PLAYERS = 2


@struct.dataclass(frozen=True)
class KaggricultureState:
    pass


class Kaggriculture:
    def __init__(self, transitions: Transition, starts: Array, horizon: int = 64):
        self.transitions = jax.tree.map(np.asarray, transitions)
        self.starts = np.asarray(starts)
        self.horizon = horizon
        self.shapes = {}

    def draw(self, pick) -> Transition:
        index = self.starts[np.asarray(pick)][:, None] + np.arange(self.horizon)[None, :]
        return jax.tree.map(lambda leaf: leaf[index], self.transitions)

    def init(self) -> KaggricultureState:
        return KaggricultureState()

    def update(self, state: KaggricultureState, key: Key, sharding) -> KaggricultureState:
        return state

    def close(self) -> None:
        pass

    def sample(
        self, state: KaggricultureState, key: Key, batch_shape: tuple[int, ...]
    ) -> Transition:
        (batch_size,) = batch_shape
        if batch_size not in self.shapes:
            probe = self.draw(np.zeros(batch_size, np.int64))
            self.shapes[batch_size] = jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), probe
            )
        pick = jax.random.randint(key, (batch_size,), 0, self.starts.shape[0])
        return io_callback(self.draw, self.shapes[batch_size], pick, ordered=False)

    def __len__(self) -> int:
        leaf, *_ = jax.tree.leaves(self.transitions)
        return leaf.shape[0]


def windows(terminated, horizon):
    ends = np.flatnonzero(terminated)
    if not ends.size:
        ends = np.array([terminated.size - 1])
    starts = np.concatenate([[0], ends[:-1] + 1])
    spans = [
        np.arange(start, end + 2 - horizon)
        for start, end in zip(starts, ends)
        if end + 1 - start >= horizon
    ]
    return np.concatenate(spans) if spans else np.zeros(0, np.int64)


def build(transitions: Transition, horizon: int = 64) -> Kaggriculture:
    starts = windows(np.asarray(transitions.first.terminated), horizon)
    if starts.size == 0:
        raise ValueError(f"no episode is {horizon} rows long")
    return Kaggriculture(transitions, starts, horizon)


def transition(obs, action, terminated, reward_scale: float = 1e4) -> Transition:
    action = dict(action)
    value = action.pop("value", None)
    widened = {}
    for name, leaf in action.items():
        other = np.zeros_like(leaf) if name in ("filled", "allowed") else leaf
        widened[name] = np.stack([leaf, other], axis=1)
    weight = np.zeros((terminated.size, PLAYERS), np.float32)
    weight[:, 0] = 1.0
    aux = {"weight": weight}
    if value is not None:
        aux["value"] = value.astype(np.float32) / reward_scale
    return Transition(
        first=Timestep(obs=obs, terminated=terminated, truncated=np.zeros_like(terminated)),
        second=Timestep(obs=None, action=widened),
        aux=aux,
    )


def save(path, parts) -> Path:
    obs, action, terminated = parts
    assert action["chosen"].ndim == 2, (
        f"expected (rows, units) chosen, got {action['chosen'].shape}; "
        "merge saved players with `merge` instead of saving a loaded Transition"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"obs_{name}": leaf for name, leaf in obs.items()},
        **{f"act_{name}": leaf for name, leaf in action.items()},
        terminated=terminated,
    )
    return path


def merge(paths):
    parts = [dict(np.load(path)) for path in paths]
    joined = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    obs = {name[4:]: leaf for name, leaf in joined.items() if name.startswith("obs_")}
    action = {name[4:]: leaf for name, leaf in joined.items() if name.startswith("act_")}
    return obs, action, joined["terminated"]


def load(path, reward_scale: float = 1e4) -> Transition:
    with np.load(path) as data:
        obs = {name[4:]: data[name] for name in data.files if name.startswith("obs_")}
        action = {name[4:]: data[name] for name in data.files if name.startswith("act_")}
        return transition(obs, action, data["terminated"], reward_scale)


def pool(paths, reward_scale: float = 1e4) -> Transition:
    parts = [load(path, reward_scale) for path in paths]
    return jax.tree.map(lambda *leaves: np.concatenate(leaves), *parts)


def make(
    dataset_id: str, directory: str, horizon: int = 64, reward_scale: float = 1e4
) -> Kaggriculture:
    return build(load(Path(directory) / f"{dataset_id}.npz", reward_scale), horizon)
