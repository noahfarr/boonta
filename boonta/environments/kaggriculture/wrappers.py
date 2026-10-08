import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, PyTree

from ..wrappers.wrapper import Wrapper, WrapperState
from . import PLAYERS
from .actions import ORDERS, UNITS, route


@struct.dataclass
class DispatchState(WrapperState):
    obs: PyTree


@struct.dataclass
class DeltaMoneyState(WrapperState):
    previous_money: Array
    previous_terminated: Array


class Dispatch(Wrapper):
    def init(self, key):
        env_state, timestep = self._env.init(key)
        rows = timestep.obs["money"].shape[0]
        blank = {
            "chosen": jnp.zeros((rows, PLAYERS, UNITS), jnp.int32),
            "market": jnp.zeros((rows, PLAYERS, ORDERS), jnp.int32),
            "qty": jnp.zeros((rows, PLAYERS, ORDERS), jnp.int32),
            "travel": jnp.zeros((rows, PLAYERS), jnp.float32),
        }
        return DispatchState(env_state=env_state, obs=timestep.obs), timestep.replace(action=blank)

    def step(self, key, state, action):
        env_state, timestep = self._env.step(key, state.env_state, route(state.obs, action))
        return DispatchState(env_state=env_state, obs=timestep.obs), timestep.replace(action=action)


class DeltaMoney(Wrapper):
    def init(self, key):
        env_state, timestep = self._env.init(key)
        money = timestep.obs["money"]
        state = DeltaMoneyState(
            env_state=env_state,
            previous_money=money,
            previous_terminated=jnp.zeros_like(timestep.terminated),
        )
        return state, timestep.replace(reward=jnp.zeros_like(money))

    def step(self, key, state, action):
        env_state, timestep = self._env.step(key, state.env_state, action)
        money = timestep.obs["money"]
        delta = jnp.where(state.previous_terminated, 0.0, money - state.previous_money)
        state = DeltaMoneyState(
            env_state=env_state,
            previous_money=money,
            previous_terminated=timestep.terminated,
        )
        return state, timestep.replace(reward=delta.astype(jnp.float32))
