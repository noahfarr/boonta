from typing import Protocol

from boonta.utils import Key, Transition
from boonta.utils.typing import PyTree

DatasetState = PyTree


class Dataset(Protocol):
    """Offline batches; sample's optional step counts allocated training positions.

    Indexed samplers use step to select a deterministic batch. Random samplers
    may ignore it; callers outside training may omit it.
    """
    def init(self) -> DatasetState: ...

    def update(
        self, state: DatasetState, key: Key, sharding: PyTree
    ) -> DatasetState: ...

    def sample(
        self, state: DatasetState, key: Key, batch_shape: tuple[int, ...], *, step=None
    ) -> Transition: ...

    def close(self) -> None: ...
