import operator
from collections.abc import Callable, Sequence

import numpy as np

from .artifact import Checkpoint


class Checkpointer:

    name = "model"

    def __init__(
        self,
        score: Callable | None = None,
        fields: Sequence[str] = ("algorithm_state",),
    ):
        self.score = score or operator.methodcaller("get", "episode_statistics/episode_return")
        self.fields = tuple(fields)

    def craft(self, key, podracer, state, logs) -> Checkpoint:
        parts = {
            field: getattr(state, field)
            for field in self.fields
            if getattr(state, field) is not None
        }
        returns = self.score(logs) if logs else None
        if returns is None:
            return Checkpoint(self.name, parts)
        return Checkpoint(self.name, parts, float(np.nanmean(returns)))
