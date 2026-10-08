from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from boonta.environments.wrappers import Vectorize
from boonta.utils import Array, Key, PyTree

from .artifact import Metrics

Bot = Callable[[PyTree, Key, Array], Array]


class Gauntlet:

    def __init__(
        self,
        opponents: Mapping[str, Bot],
        num_games: int = 512,
        opening: int = 4,
        seed: int = 0,
        name: str = "gauntlet",
    ):
        self.opponents = dict(opponents)
        self.num_games = num_games
        self.opening = opening
        self.key = jax.random.key(seed)
        self.name = name
        self.matches = {}
        self.epoch = 0

    def match(self, algorithm, environment, opponent: Bot):
        game = environment.unwrapped
        games = Vectorize(game, self.num_games)
        seats = jnp.arange(self.num_games) % 2
        learner = jnp.arange(2) == seats[:, None]

        def play(algorithm_state: PyTree, key: Key) -> tuple[Array, Array]:
            init_key, key = jax.random.split(key)
            environment_state, timestep = games.init(init_key)

            def turn(carry, ply):
                environment_state, timestep, score, over, key = carry
                key, learner_key, bot_key, opening_key, step_key = jax.random.split(key, 5)
                _, learned, _ = algorithm.step(
                    algorithm_state, learner_key, timestep, temperature=0.0
                )
                bot_keys = jax.random.split(bot_key, self.num_games)
                rival = jax.vmap(opponent)(timestep.obs, bot_keys, 1 - seats)
                chosen = jnp.where(learner, learned, rival[:, None])
                opened = jax.random.categorical(
                    opening_key, jnp.where(timestep.obs["mask"], 0.0, -jnp.inf)
                )
                action = jnp.where(ply < self.opening, opened, chosen)
                environment_state, timestep = games.step(step_key, environment_state, action)
                reward = jnp.sum(jnp.where(learner, timestep.reward, 0.0), axis=-1)
                score = jnp.where(over, score, (reward + 1.0) / 2.0)
                over = over | timestep.terminated.all(axis=-1)
                return (environment_state, timestep, score, over, key), None

            score = jnp.full((self.num_games,), 0.5)
            over = jnp.zeros((self.num_games,), bool)
            (_, _, score, over, _), _ = jax.lax.scan(
                turn,
                (environment_state, timestep, score, over, key),
                jnp.arange(game.time_limit()),
            )
            return score, over

        return jax.jit(play)

    def craft(self, algorithm, environment, state, logs) -> Metrics:
        key = jax.random.fold_in(self.key, self.epoch)
        self.epoch += 1
        data = {}
        for index, (name, opponent) in enumerate(self.opponents.items()):
            if name not in self.matches:
                self.matches[name] = self.match(algorithm, environment, opponent)
            score, _ = self.matches[name](
                state.algorithm_state, jax.random.fold_in(key, index)
            )
            data[f"{self.name}/{name}"] = float(np.mean(score))
        data[f"{self.name}/mean"] = float(np.mean(list(data.values())))
        return Metrics(self.name, data)
