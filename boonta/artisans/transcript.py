from functools import partial

import jax
import numpy as np

from boonta.utils import load_tokenizer

from .artifact import Text
from .rollout import rollout


class Transcript:

    def __init__(self, repo_id: str, name: str = "generation", num_examples: int = 4):
        self._tokenizer = load_tokenizer(repo_id)
        self.name = name
        self._num_examples = num_examples

    def craft(self, key, algorithm, environment, state, logs) -> Text:
        play = jax.jit(
            partial(
                rollout,
                algorithm,
                environment,
                num_steps=environment.time_limit() + 1,
                record=lambda environment_state, action: action,
            )
        )
        actions = np.asarray(play(state.algorithm_state, key))
        *_, num_envs = actions.shape
        transcript = "\n\n".join(
            f"{env}: {self._tokenizer.decode(actions[:, env].tolist())}"
            for env in range(min(self._num_examples, num_envs))
        )
        return Text(self.name, transcript)
