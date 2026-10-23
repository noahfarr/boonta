from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep

from .environment import Environment
from .spaces import Space


@struct.dataclass
class Observation:
    observation: Array
    state: Array
    context: Array
    command: Array


@struct.dataclass
class WorldModelState:
    params: PyTree = struct.field(default=None, metadata={"axis": None})
    state: Array = None
    context: Array = None
    command: Array = None


class _Networks(nn.Module):
    dynamics_fn: Callable
    reward_fn: Callable
    termination_fn: Callable
    observation_fn: Callable

    def __call__(self, state: Array, action: Array, context: Array, command: Array, goal: Array) -> tuple:
        *batch, _ = jnp.shape(state)
        if self.is_initializing():
            self.observe(state, context, command)
        following = self.dynamics_fn(state=state, action=action, context=context)
        reward = self.reward_fn(state=state, action=action, following=following, command=command, goal=goal)
        termination = self.termination_fn(state=following, command=command, goal=goal)
        return following, jnp.reshape(reward, batch), jnp.reshape(termination, batch)

    def observe(self, state: Array, context: Array, command: Array) -> Array:
        return self.observation_fn(state=state, context=context, command=command)


def _unchanged(command: Array) -> Array:
    return command


class WorldModel(Environment):
    def __init__(
        self,
        environment: Environment,
        dynamics_fn: Callable,
        reward_fn: Callable,
        termination_fn: Callable,
        observation_fn: Callable,
        goal: Callable[[Array], Array] = _unchanged,
        advance: Callable[[Array], Array] = _unchanged,
    ):
        self.environment = environment
        self.networks = _Networks(
            dynamics_fn=dynamics_fn,
            reward_fn=reward_fn,
            termination_fn=termination_fn,
            observation_fn=observation_fn,
        )
        self.goal = goal
        self.advance = advance

    def initialize(self, key: Key, theta: tuple, action: Array) -> PyTree:
        state, context, command = theta
        return self.networks.init(key, state, action, context, command, self.goal(command))

    def learned(self, params: PyTree) -> set[str]:
        return {name.removesuffix("_fn") for name in params.get("params", {})}

    def calibrate(self, params: PyTree, theta: tuple, action: Array) -> PyTree:
        state, context, command = theta
        _, statistics = self.networks.apply(
            params, state, action, context, command, self.goal(command), mutable=["statistics"]
        )
        return {**params, **statistics}

    def init(self, key: Key) -> tuple[WorldModelState, Timestep]:
        return WorldModelState(), Timestep(obs=None)

    def update(
        self, state: WorldModelState, key: Key, params: PyTree = None, theta: tuple = None, **kwargs
    ) -> WorldModelState:
        if params is not None:
            state = state.replace(params=params)
        if theta is not None:
            model_state, context, command = theta
            state = state.replace(state=model_state, context=context, command=command)
        return state

    def observe(self, state: WorldModelState) -> Observation:
        observation = self.networks.apply(
            state.params, state.state, state.context, state.command, method=_Networks.observe
        )
        return Observation(observation=observation, state=state.state, context=state.context, command=state.command)

    def step(self, key: Key, state: WorldModelState, action: Array) -> tuple[WorldModelState, Timestep]:
        following, reward, termination = self.networks.apply(
            state.params, state.state, action, state.context, state.command, self.goal(state.command)
        )
        state = state.replace(state=following, command=self.advance(state.command))
        flags = jnp.zeros_like(reward, bool)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward,
            terminated=flags,
            truncated=flags,
            info={"termination": termination},
        )

    def observation_space(self) -> Space:
        return self.environment.observation_space()

    def action_space(self) -> Space:
        return self.environment.action_space()
