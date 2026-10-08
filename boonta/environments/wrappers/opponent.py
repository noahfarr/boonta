from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class OpponentState(WrapperState):
    params: PyTree = struct.field(metadata={"axis": None})
    carry: PyTree
    joint: Timestep


def stack(action: Array, rival: Array) -> Array:
    return jnp.stack([action, rival], axis=-1)


def first(leaf: Array) -> Array:
    return jnp.take(leaf, 0, axis=-1)


class Opponent(Wrapper):
    def __init__(
        self,
        env,
        play: Callable,
        initialize: Callable[[Key, Timestep], tuple[PyTree, PyTree]],
        groups: int = 1,
        seat: Callable[[Array], Array] = first,
        join: Callable[[Array, Array], Array] = stack,
        temperature: float = 1.0,
    ):
        super().__init__(env)
        self._play = play
        self._initialize = initialize
        self._groups = groups
        self._seat = seat
        self._join = join
        self._temperature = temperature

    @property
    def num_agents(self) -> int:
        return 1

    def narrow(self, joint: Timestep) -> Timestep:
        return joint.replace(
            action=self._seat(joint.action),
            reward=self._seat(joint.reward),
            terminated=self._seat(joint.terminated),
            truncated=self._seat(joint.truncated),
        )

    def group(self, tree: PyTree, index: int) -> PyTree:
        def take(leaf):
            width = leaf.shape[0] // self._groups
            return leaf[index * width : (index + 1) * width]

        return jax.tree.map(take, tree)

    def init(self, key: Key) -> tuple[OpponentState, Timestep]:
        env_key, opponent_key = jax.random.split(key)
        env_state, joint = self._env.init(env_key)
        params, carry = self._initialize(opponent_key, joint)
        params = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (self._groups, *jnp.shape(leaf))), params
        )
        state = OpponentState(env_state, params=params, carry=carry, joint=joint)
        return state, self.narrow(joint)

    def step(
        self, key: Key, state: OpponentState, action: Array
    ) -> tuple[OpponentState, Timestep]:
        env_key, opponent_key = jax.random.split(key)
        carries, rivals = [], []
        for index in range(self._groups):
            carry, rival = self._play(
                self.group(state.carry, index),
                jax.tree.map(lambda leaf: leaf[index], state.params),
                self.group(state.joint, index),
                jax.random.fold_in(opponent_key, index),
                self._temperature,
            )
            carries.append(carry)
            rivals.append(rival)
        carry = jax.tree.map(lambda *leaves: jnp.concatenate(leaves), *carries)
        rival = jnp.concatenate(rivals)
        env_state, joint = self._env.step(
            env_key, state.env_state, self._join(action, rival)
        )
        state = state.replace(env_state=env_state, carry=carry, joint=joint)
        return state, self.narrow(joint)

    def action_mask(self, state: OpponentState) -> Array | None:
        mask = self._env.action_mask(state.env_state)
        if mask is None:
            return None
        return self._seat(jnp.moveaxis(mask, -2, -1))

    def update(self, state: OpponentState, opponents: PyTree = None, **kwargs):
        if kwargs:
            state = state.replace(env_state=self._env.update(state.env_state, **kwargs))
        if opponents is None:
            return state
        params = jax.tree.map(
            lambda old, new: jnp.asarray(new, old.dtype), state.params, opponents
        )
        _, carry = self._initialize(jax.random.key(0), state.joint)
        return state.replace(params=params, carry=carry)
