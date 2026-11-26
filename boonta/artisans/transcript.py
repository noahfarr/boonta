from functools import partial

import jax
import numpy as np

from boonta.podracers.sebulba import SebulbaState
from boonta.utils import load_tokenizer

from .artifact import Text


class Transcript:

    def __init__(self, repo_id: str, name: str = "generation", num_examples: int = 4):
        self._tokenizer = load_tokenizer(repo_id)
        self.name = name
        self._num_examples = num_examples

    def rollout(self, podracer, state, key):
        reset_key, lap_key, call_key, key = jax.random.split(key, 4)
        environment_state, timestep = podracer.environment.init(reset_key)
        _, carry = podracer.algorithm.init(reset_key, timestep)
        state = state.replace(
            timestep=timestep, environment_state=environment_state, carry=carry
        )
        state = podracer.pit(call_key, podracer.lap(lap_key, state))
        num_steps = podracer.environment.time_limit() + 1

        def step(state, key):
            state, transition = podracer.rollout(state, key, temperature=0.0)
            return state, transition.second.action

        state, actions = jax.lax.scan(step, state, jax.random.split(key, num_steps))
        podracer.environment.close(state.environment_state)
        return actions

    def craft(self, key, podracer, state, logs) -> Text:
        if isinstance(state, SebulbaState):
            device, *_ = podracer.config.actor.devices.flat
            actor, *_ = state.actors
            state, key = jax.device_put(
                (actor.board(state.algorithm_state, state.pit_state), key), device
            )
        actions = np.asarray(jax.jit(partial(self.rollout, podracer))(state, key))
        *_, num_envs = actions.shape
        transcript = "\n\n".join(
            f"{env}: {self._tokenizer.decode(actions[:, env].tolist())}"
            for env in range(min(self._num_examples, num_envs))
        )
        return Text(self.name, transcript)
