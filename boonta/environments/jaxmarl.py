from typing import Any

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class JaxMARL(Environment):
    def __init__(self, env):
        from jaxmarl.environments.multi_agent_env import MultiAgentEnv

        self._env = env
        self._supports_action_mask = (
            type(env).get_avail_actions is not MultiAgentEnv.get_avail_actions
        )

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    def init(self, key: Key) -> tuple[Any, Timestep]:
        obs, state = self._env.reset(key)
        obs = jnp.stack([obs[agent] for agent in self._env.agents])
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        info = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes.info)
        if self._supports_action_mask:
            info = {**info, "action_mask": self.action_mask(state)}
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros(shapes.reward.shape, shapes.reward.dtype),
            terminated=jnp.ones(shapes.terminated.shape, shapes.terminated.dtype),
            truncated=jnp.zeros(shapes.truncated.shape, shapes.truncated.dtype),
            info=info,
        )
        return state, timestep

    def step(
        self, key: Key, state: Any, action: Array
    ) -> tuple[Any, Timestep]:
        actions = dict(zip(self._env.agents, action, strict=True))
        obs, state, reward, done, info = self._env.step_env(key, state, actions)
        obs, reward = (
            jnp.stack([values[agent] for agent in self._env.agents])
            for values in (obs, reward)
        )
        done = jnp.stack([done[agent] | done["__all__"] for agent in self._env.agents])
        if self._supports_action_mask:
            info = {**info, "action_mask": self.action_mask(state)}
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=reward,
            terminated=done,
            truncated=jnp.zeros_like(done),
            info=info,
        )
        return state, timestep

    def action_mask(self, state: Any) -> Array:
        avail_actions = self._env.get_avail_actions(state)
        return jnp.stack([avail_actions[agent] for agent in self._env.agents]).astype(
            bool
        )

    def observation_space(self) -> Space:
        agent, *_ = self._env.agents
        space = self._env.observation_space(agent)
        return Space(
            shape=(self._env.num_agents, *space.shape),
            dtype=space.dtype,
            low=space.low,
            high=space.high,
        )

    def action_space(self) -> Space:
        from jaxmarl.environments import spaces as jaxmarl_spaces

        agent, *_ = self._env.agents
        space = self._env.action_space(agent)
        match space:
            case jaxmarl_spaces.Discrete(n=n):
                return Space(
                    shape=(self._env.num_agents,), dtype=jnp.int32, low=0, high=n - 1
                )
            case jaxmarl_spaces.Box(low=low, high=high, shape=shape, dtype=dtype):
                return Space(
                    shape=(self._env.num_agents, *shape),
                    dtype=dtype,
                    low=low,
                    high=high,
                )
            case _:
                raise ValueError(f"Unsupported action space type: {type(space)}")

    def horizon(self) -> int:
        return int(self._env.max_steps)


def make(env_id, **kwargs):
    import jaxmarl

    if isinstance(kwargs.get("scenario"), str):
        from jaxmarl.environments.smax import map_name_to_scenario

        kwargs["scenario"] = map_name_to_scenario(kwargs["scenario"])

    env = jaxmarl.make(env_id, **kwargs)
    return JaxMARL(env)
