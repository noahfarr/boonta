from typing import Protocol

from boonta.utils import Key, Transition
from boonta.utils.typing import PyTree

DatasetState = PyTree


class Dataset(Protocol):
    def init(self) -> DatasetState: ...

    def update(
        self, state: DatasetState, key: Key, sharding: PyTree
    ) -> DatasetState: ...

    def sample(
        self, state: DatasetState, key: Key, batch_shape: tuple[int, ...]
    ) -> Transition: ...

    def close(self) -> None: ...
