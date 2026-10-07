from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
import trueskill
from boonta.utils import Array, Key, Metrics, PyTree

Scores = np.ndarray


class Opponent(Protocol):
    def __call__(self, obs: PyTree, key: Key, seat: int) -> Array: ...


class Player(Protocol):
    def __call__(self, algorithm: object, params: PyTree) -> Opponent: ...


MU = 25.0
SIGMA = MU / 3.0
BETA = MU / 6.0
TAU = MU / 300.0

RATER = trueskill.TrueSkill(
    mu=MU, sigma=SIGMA, beta=BETA, tau=TAU, draw_probability=0.0
)


def conservative(rating: trueskill.Rating) -> float:
    return rating.mu - 3.0 * rating.sigma


def rate(
    ratings: dict[str, trueskill.Rating], results: list[tuple[str, str, float]]
) -> None:
    for a, b, score in results:
        if score > 0.5:
            ratings[a], ratings[b] = trueskill.rate_1vs1(
                ratings[a], ratings[b], env=RATER
            )
        elif score < 0.5:
            ratings[b], ratings[a] = trueskill.rate_1vs1(
                ratings[b], ratings[a], env=RATER
            )
        else:
            ratings[a], ratings[b] = trueskill.rate_1vs1(
                ratings[a], ratings[b], drawn=True, env=RATER
            )


@dataclass
class TrueSkill:
    player: Player
    opponents: dict[str, Opponent]
    name: str = "trueskill"
    ratings: dict[str, trueskill.Rating] = field(default_factory=dict, init=False)
    epoch: int = field(default=0, init=False)

    def play(self, environment, first: Opponent, second: Opponent, key: Key) -> Scores:
        env = environment.unwrapped
        state, timestep = env.init(key)

        def step(carry, index):
            state, timestep, key = carry
            key, first_key, second_key, env_key = jax.random.split(key, 4)
            action = jnp.stack(
                [
                    first(timestep.obs, first_key, 0),
                    second(timestep.obs, second_key, 1),
                ],
                1,
            )
            state, timestep = env.step(env_key, state, action)
            return (state, timestep, key), None

        (state, timestep, key), _ = jax.lax.scan(
            step, (state, timestep, key), jnp.arange(env.time_limit())
        )
        reward = np.asarray(timestep.reward)
        env.close(state)
        left, right = reward[:, 0], reward[:, 1]
        return np.where(left > right, 1.0, np.where(left < right, 0.0, 0.5))

    def subjects(self, state: PyTree) -> dict[str, PyTree]:
        def walk(node, path):
            if not isinstance(node, Mapping) or "params" in node:
                yield ":".join(path) or "learner", node
                return
            for key, child in node.items():
                yield from walk(child, path + [str(key)])

        return dict(walk(state.algorithm_state.params, []))

    def craft(
        self, algorithm: object, environment: object, state: PyTree, logs: dict
    ) -> Metrics:
        subjects = self.subjects(state)
        for name in (*subjects, *self.opponents):
            self.ratings.setdefault(name, RATER.create_rating())

        key = jax.random.PRNGKey(self.epoch)
        results = []

        names = sorted(self.opponents)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        for index, (first, second) in enumerate(pairs):
            scores = self.play(
                environment,
                self.opponents[first],
                self.opponents[second],
                jax.random.fold_in(key, 1_000 + index),
            )
            results.extend((first, second, float(s)) for s in np.asarray(scores))

        for offset, (subject, params) in enumerate(sorted(subjects.items())):
            learner = self.player(algorithm, params)
            for index, name in enumerate(sorted(self.opponents)):
                scores = self.play(
                    environment,
                    learner,
                    self.opponents[name],
                    jax.random.fold_in(key, offset * 100 + index),
                )
                results.extend((subject, name, float(s)) for s in np.asarray(scores))

        rate(self.ratings, results)
        self.epoch += 1

        data = {}
        for subject in subjects:
            prefix = self.name if subject == "learner" else f"{self.name}/{subject}"
            data[f"{prefix}/rating"] = conservative(self.ratings[subject])
            data[f"{prefix}/mu"] = self.ratings[subject].mu
            data[f"{prefix}/sigma"] = self.ratings[subject].sigma
        for name in self.opponents:
            data[f"{self.name}/{name}"] = conservative(self.ratings[name])
        return Metrics(self.name, data)
