from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np

from boonta.utils import Key
from boonta.utils.typing import PyTree


class Still:
    def update(self, state: PyTree, key: Key, sharding: PyTree) -> PyTree:
        return state

    def close(self) -> None:
        pass


class Stream:
    def __init__(self, draw: Callable[[np.random.Generator], PyTree]):
        self.draw = draw
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pool")
        self.pending = None

    def stage(self, key: Key, sharding: PyTree) -> PyTree:
        generator = np.random.default_rng(np.asarray(jax.random.key_data(key)))
        return jax.device_put(self.draw(generator), sharding)

    def update(self, state: PyTree, key: Key, sharding: PyTree) -> PyTree:
        if self.pending is None:
            self.pending = self.executor.submit(self.stage, key, sharding)
        state = self.pending.result()
        self.pending = self.executor.submit(
            self.stage, jax.random.fold_in(key, 1), sharding
        )
        return state

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
