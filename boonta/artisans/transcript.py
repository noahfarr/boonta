from functools import partial

import jax
import numpy as np

from boonta.utils import load_tokenizer

from .artifact import Text


class Transcript:

    def __init__(self, repo_id: str, name: str = "generation", num_examples: int = 4):
        self._tokenizer = load_tokenizer(repo_id)
        self.name = name
        self._num_examples = num_examples

    def rollout(self, algorithm, environment, num_steps, algorithm_state, key):
        init_key, key = jax.random.split(key)
        environment_state, timestep = environment.init(init_key)
        _, carry = algorithm.init(init_key, timestep)

        def step(loop, key):
            algorithm_state, carry, environment_state, timestep = loop
            algorithm_key, environment_key = jax.random.split(key)
            algorithm_state, carry, action, _ = algorithm.step(
                algorithm_state, carry, algorithm_key, timestep, temperature=0.0
            )
            environment_state, timestep = environment.step(
                environment_key, environment_state, action
            )
            return (algorithm_state, carry, environment_state, timestep), action

        (_, _, environment_state, _), actions = jax.lax.scan(
            step,
            (algorithm_state, carry, environment_state, timestep),
            jax.random.split(key, num_steps),
        )
        environment.close(environment_state)
        return actions

    def craft(self, key, podracer, podracer_state, logs) -> Text:
        algorithm, environment = podracer.algorithm, podracer.environment
        actions = np.asarray(
            jax.jit(
                partial(
                    self.rollout, algorithm, environment, environment.time_limit() + 1
                )
            )(podracer_state.algorithm_state, key)
        )
        *_, num_envs = actions.shape
        transcript = "\n\n".join(
            f"{env}: {self._tokenizer.decode(actions[:, env].tolist())}"
            for env in range(min(self._num_examples, num_envs))
        )
        return Text(self.name, transcript)
