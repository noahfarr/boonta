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
        self.score = score or operator.itemgetter("episode_statistics/episode_return")
        self.fields = tuple(fields)

    def craft(self, algorithm, environment, state, logs) -> Checkpoint:
        parts = {field: getattr(state, field) for field in self.fields}
        if not logs:
            return Checkpoint(self.name, parts)
        return Checkpoint(self.name, parts, float(np.nanmean(self.score(logs))))
