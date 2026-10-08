import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class Mapox(Environment):

    def __init__(self, env, length):
        self._env = env
        self._length = length

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    def init(self, key: Key) -> tuple[Array, Timestep]:
        state, ts = self._env.reset(key)
        action_space = self.action_space()
        timestep = Timestep(
            obs={"view": ts.obs, "mask": ts.action_mask},
            action=jnp.zeros(action_space.shape, action_space.dtype),
            reward=jnp.zeros_like(ts.reward, dtype=jnp.float32),
            terminated=jnp.ones_like(ts.terminated),
            truncated=jnp.zeros_like(ts.terminated),
            info={},
        )
        return state, timestep

    def step(self, key: Key, state: Array, action: Array) -> tuple[Array, Timestep]:
        action = jnp.asarray(action, jnp.int32)
        state, ts = self._env.step(state, action, key)
        timestep = Timestep(
            obs={"view": ts.obs, "mask": ts.action_mask},
            action=action,
            reward=ts.reward.astype(jnp.float32),
            terminated=ts.terminated,
            truncated=jnp.zeros_like(ts.terminated),
            info={},
        )
        return state, timestep

    def observation_space(self) -> dict[str, Space]:
        spec = self._env.observation_spec
        sizes = jnp.asarray(spec.max_value)
        return {
            "view": Space(
                shape=(self.num_agents, *spec.shape),
                dtype=spec.dtype,
                low=0,
                high=sizes - 1,
            ),
            "mask": Space(
                shape=(self.num_agents, self._env.action_spec.n),
                dtype=jnp.bool_,
                low=0,
                high=1,
            ),
        }

    def action_space(self) -> Space:
        return Space(
            shape=(self.num_agents,),
            dtype=jnp.int32,
            low=0,
            high=self._env.action_spec.n - 1,
        )

    def time_limit(self) -> int:
        return self._length - 1


def make(env_id, length=512, **kwargs):
    from mapox import EnvironmentConfig, EnvironmentFactory
    from pydantic import TypeAdapter

    config = TypeAdapter(EnvironmentConfig).validate_python(
        {"env_type": env_id, **kwargs}
    )
    env, _ = EnvironmentFactory().create_env(config, length=length)
    return Mapox(env, length)
