from dataclasses import dataclass

import jax
import numpy as np

from boonta.curricula.pbt import leading_member

from .artifact import Metrics


@dataclass
class Leaderboard:
    name: str = "leaderboard"
    key: str = "episode_statistics/episode_return"

    def craft(self, algorithm, environment, state, logs) -> Metrics:
        if self.key not in logs:
            return Metrics(self.name, {})
        envs, *_ = state.timestep.reward.shape
        returns = np.asarray(jax.device_get(logs[self.key]), np.float64).reshape(
            -1, algorithm.members, algorithm.seeds, envs // algorithm.copies
        )
        members = np.moveaxis(returns, 1, 0).reshape(algorithm.members, -1)
        finished = ~np.isnan(members)
        totals = np.where(finished, members, 0.0).sum(axis=1)
        counts = finished.sum(axis=1)
        member_returns = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
        leader = int(jax.device_get(leading_member(algorithm, state.algorithm_state)))
        return Metrics(
            self.name,
            {
                f"{self.name}/best_return": np.take(member_returns, leader),
                f"{self.name}/mean": np.nanmean(member_returns),
                f"{self.name}/spread": np.nanstd(member_returns),
            },
        )
