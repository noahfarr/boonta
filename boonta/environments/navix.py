from typing import Any

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class Navix(Environment):

    def __init__(self, env, seed: int | None = None):
        self._env = env
        self.seed = seed
        self._renderer = None

    def reset(self, key: Key) -> Any:
        if self.seed is None:
            return self._env.reset(key)
        # Every reset builds the layout of one seed, which does not depend on the key and so is computed once
        # for a whole batch. The environment's own randomness still follows the key.
        timestep = self._env.reset(jax.random.key(self.seed))
        return timestep.replace(state=timestep.state.replace(key=key))

    def init(self, key: Key) -> tuple[Any, Timestep]:
        state = self.reset(key)
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs=state.observation,
            action=action,
            reward=jnp.zeros(shapes.reward.shape, shapes.reward.dtype),
            terminated=jnp.ones(shapes.terminated.shape, shapes.terminated.dtype),
            truncated=jnp.zeros(shapes.truncated.shape, shapes.truncated.dtype),
            info=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes.info
            ),
        )
        return state, timestep

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        timestep = self._env.step(state, action)
        step_type = timestep.step_type
        ts = Timestep(
            obs=timestep.observation,
            action=action,
            reward=timestep.reward,
            terminated=step_type == 2,
            truncated=step_type == 1,
            info={},
        )
        return timestep, ts

    def observe(self, state: Any) -> Array:
        return state.observation

    def observation_space(self) -> Space:
        shape = jax.eval_shape(self._env.reset, jax.random.key(0)).observation
        if jnp.issubdtype(shape.dtype, jnp.integer):
            info = jnp.iinfo(shape.dtype)
            low, high = info.min, info.max
        else:
            low, high = -jnp.inf, jnp.inf
        return Space(shape=shape.shape, dtype=shape.dtype, low=low, high=high)

    def action_space(self) -> Space:
        return Space(
            shape=(),
            dtype=jnp.int32,
            low=0,
            high=len(self._env.action_set) - 1,
        )

    def time_limit(self) -> int:
        return int(self._env.max_steps)

    def render(self, state: Any) -> Array:
        if self._renderer is None:
            from navix import observations

            self._renderer = jax.jit(lambda timestep: observations.rgb(timestep.state))
        return self._renderer(state)


def make(env_id, seed: int | None = None, **kwargs):
    import navix

    env = navix.make(env_id, **kwargs)
    # The environments wrapped by boonta reset themselves, and NAVIX resets one
    # step late, so its own reset has to go.
    return Navix(env.replace(disable_autoreset=True), seed=seed)


def cell(env, direction: bool = False):
    """Groups states by where the agent stands, what it carries and how each door stands."""

    def one(timestep):
        state = timestep.state
        player = state.get_player()
        features = [*jnp.reshape(player.position, (-1,)), jnp.reshape(player.pocket, ())]
        if direction:
            features.append(jnp.reshape(player.direction, ()))
        try:
            features.extend(jnp.reshape(state.get_doors().symbolic_state, (-1,)))
        except KeyError:
            pass
        key = jnp.int32(0)
        for value in features:
            key = key * jnp.int32(1000003) + value.astype(jnp.int32)
        return key

    return lambda env_state: jax.vmap(one)(getattr(env_state, "unwrapped", env_state))

