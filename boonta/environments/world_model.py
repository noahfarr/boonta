from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep

from .environment import Environment
from .spaces import Space


@struct.dataclass
class WorldModelState:
    params: PyTree = struct.field(metadata={"axis": None})
    state: Array = None
    context: Array = None
    command: Array = None


class _Heads(nn.Module):
    dynamics: Callable
    reward: Callable
    termination: Callable
    observation: Callable

    def __call__(self, state: Array, action: Array, context: Array, command: Array, goal: Array) -> tuple:
        *batch, _ = jnp.shape(state)
        if self.is_initializing():
            self.observe(state, context, command)
        following = self.dynamics(state=state, action=action, context=context)
        reward = self.reward(state=state, action=action, following=following, command=command, goal=goal)
        termination = self.termination(state=following, command=command, goal=goal)
        return following, jnp.reshape(reward, batch), jnp.reshape(termination, batch)

    def observe(self, state: Array, context: Array, command: Array) -> Array:
        return self.observation(state=state, context=context, command=command)


def _unchanged(command: Array) -> Array:
    return command


class WorldModel(Environment):
    def __init__(
        self,
        environment: Environment,
        dynamics: Callable,
        reward: Callable,
        termination: Callable,
        observation: Callable,
        pack: Callable,
        goal: Callable[[Array], Array] = _unchanged,
        advance: Callable[[Array], Array] = _unchanged,
    ):
        self.environment = environment
        self.network = _Heads(dynamics=dynamics, reward=reward, termination=termination, observation=observation)
        self.pack = pack
        self.goal = goal
        self.advance = advance

    def init(self, key: Key) -> tuple[WorldModelState, Timestep]:
        raise NotImplementedError("a world model starts from a known state, through start(params, state, ...)")

    def start(self, params: PyTree, state: Array, context: Array, command: Array) -> tuple[WorldModelState, Timestep]:
        observation = self.network.apply(params, state, context, command, method=_Heads.observe)
        *batch, _ = jnp.shape(state)
        flags = jnp.zeros(batch, bool)
        return WorldModelState(params=params, state=state, context=context, command=command), Timestep(
            obs=self.pack(state=state, observation=observation, context=context, command=command),
            action=jnp.zeros((*batch, *self.action_space().shape), self.action_space().dtype),
            reward=jnp.zeros(batch),
            terminated=flags,
            truncated=flags,
            info={"termination": jnp.full(batch, -jnp.inf)},
        )

    def step(self, key: Key, state: WorldModelState, action: Array) -> tuple[WorldModelState, Timestep]:
        following, reward, termination = self.network.apply(
            state.params, state.state, action, state.context, state.command, self.goal(state.command)
        )
        command = self.advance(state.command)
        observation = self.network.apply(state.params, following, state.context, command, method=_Heads.observe)
        flags = jnp.zeros_like(reward, bool)
        return state.replace(state=following, command=command), Timestep(
            obs=self.pack(state=following, observation=observation, context=state.context, command=command),
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
