from typing import Any

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class XLandMiniGrid(Environment):
    def __init__(self, env, params, benchmark=None):
        self._env = env
        self.params = params
        self.benchmark = benchmark

    def sample_params(self, key: Key, params):
        if self.benchmark is None:
            return params
        return params.replace(ruleset=self.benchmark.sample_ruleset(key))

    def init(self, key: Key) -> tuple[Any, Timestep]:
        sample_key, reset_key = jax.random.split(key)
        state = self._env.reset(self.sample_params(sample_key, self.params), reset_key)
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

    def step(
        self, key: Key, state: Any, action: Array
    ) -> tuple[Any, Timestep]:
        timestep = self._env.step(self.params, state, action)
        terminated = timestep.last() & (timestep.discount == 0.0)
        truncated = timestep.last() & (timestep.discount == 1.0)
        ts = Timestep(
            obs=timestep.observation,
            action=action,
            reward=timestep.reward,
            terminated=terminated,
            truncated=truncated,
            info={},
        )
        return timestep, ts

    def observation_space(self) -> Space:
        return Space(
            shape=self._env.observation_shape(self.params),
            dtype=jnp.uint8,
            low=0,
            high=255,
        )

    def action_space(self) -> Space:
        return Space(
            shape=(),
            dtype=jnp.int32,
            low=0,
            high=self._env.num_actions(self.params) - 1,
        )

    def time_limit(self) -> int:
        return int(self.params.max_steps)


def make(env_id, benchmark=None, **kwargs):
    import xminigrid
    from xminigrid.benchmarks import load_benchmark

    env, env_params = xminigrid.make(env_id, **kwargs)

    if benchmark is not None:
        benchmark = load_benchmark(benchmark)

    return XLandMiniGrid(
        env,
        env_params,
        benchmark=benchmark,
    )
