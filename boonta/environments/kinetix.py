from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import spaces as gymnax_spaces

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class Kinetix(Environment):
    def __init__(self, env, params):
        self._env = env
        self._params = params

    def init(self, key: Key) -> tuple[Any, Timestep]:
        obs, state = self._env.reset_env(key, self._params, None)
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros(shapes.reward.shape, shapes.reward.dtype),
            terminated=jnp.ones(shapes.terminated.shape, shapes.terminated.dtype),
            truncated=jnp.zeros(shapes.truncated.shape, shapes.truncated.dtype),
            info=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes.info
            ),
        )
        return state, timestep

    def update(self, state: Any, theta: Any = None, **kwargs) -> Any:
        if theta is None:
            return state
        return theta

    def observe(self, state: Any) -> Any:
        return self._env.get_obs(state)

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        obs, state, reward, done, info = self._env.step_env(
            key, state, action, self._params
        )
        truncated = state.timestep >= self._params.max_timesteps
        terminated = done & ~truncated
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return state, timestep

    def observation_space(self) -> Space:
        space = self._env.observation_space(self._params)
        return Space(
            shape=space.shape, dtype=space.dtype, low=space.low, high=space.high
        )

    def action_space(self) -> Space:
        from kinetix.environment import MultiDiscrete

        space = self._env.action_space(self._params)
        if isinstance(space, gymnax_spaces.Discrete):
            return Space(shape=space.shape, dtype=space.dtype, low=0, high=space.n - 1)
        if isinstance(space, MultiDiscrete):
            return Space(
                shape=space.shape,
                dtype=jnp.int32,
                low=0,
                high=space.number_of_dims_per_distribution - 1,
            )
        return Space(
            shape=space.shape, dtype=space.dtype, low=space.low, high=space.high
        )

    def time_limit(self) -> int:
        return int(self._params.max_timesteps)


def make(
    env_id: str | list[str] | None = None,
    action_type: str = "discrete",
    observation_type: str = "symbolic_flat",
    env_params=None,
    static_env_params=None,
    ued_params=None,
    levels=None,
):
    from jax2d.engine import PhysicsEngine
    from kinetix.environment import (ActionType, EnvParams, ObservationType,
                                     StaticEnvParams, UEDParams,
                                     make_kinetix_env, sample_kinetix_level)

    env_params = env_params or EnvParams()
    static_env_params = static_env_params or StaticEnvParams()
    ued_params = ued_params or UEDParams()
    physics_engine = PhysicsEngine(static_env_params)

    if levels is not None:
        count = jax.tree.leaves(levels)[0].shape[0]

        def reset_fn(key):
            drawn = jax.random.randint(key, (), 0, count)
            return jax.tree.map(lambda leaf: leaf[drawn], levels)
    elif env_id is None:

        def reset_fn(key):
            return sample_kinetix_level(
                key, physics_engine, env_params, static_env_params, ued_params
            )
    else:
        from kinetix.util.saving import load_evaluation_levels

        names = [env_id] if isinstance(env_id, str) else list(env_id)
        levels, static_env_params = load_evaluation_levels(
            names, static_env_params_override=static_env_params
        )

        def reset_fn(key):
            drawn = jax.random.randint(key, (), 0, len(names))
            return jax.tree.map(lambda leaf: leaf[drawn], levels)

    env = make_kinetix_env(
        action_type=ActionType.from_string(action_type),
        observation_type=ObservationType.from_string(observation_type),
        reset_fn=reset_fn,
        env_params=env_params,
        static_env_params=static_env_params,
        auto_reset=False,
    )
    return Kinetix(env, env_params)
