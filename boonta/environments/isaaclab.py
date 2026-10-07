import atexit

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space
from .wrappers import Batched

simulation_app = None


def launch(**kwargs):
    global simulation_app
    if simulation_app is None:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(kwargs).app
        atexit.register(simulation_app.close)
    return simulation_app


@struct.dataclass
class EnvState:
    pass


class IsaacLab(Environment):
    def __init__(self, env):
        self._env = env
        env = env.unwrapped
        obs_space = env.single_observation_space["policy"]
        action_space = env.single_action_space

        self.num_envs = env.num_envs
        self._obs_space = Space(
            shape=obs_space.shape,
            dtype=obs_space.dtype,
            low=obs_space.low,
            high=obs_space.high,
        )
        self._action_space = Space(
            shape=action_space.shape,
            dtype=action_space.dtype,
            low=action_space.low,
            high=action_space.high,
        )
        self._horizon = int(env.max_episode_length)

    def init(
        self, key: Key
    ) -> tuple[EnvState, Timestep]:
        def _reset(seed):
            obs, _ = self._env.reset(seed=int(seed))
            return np.asarray(obs["policy"].detach().cpu(), dtype=self._obs_space.dtype)

        seed = jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max)
        obs_struct = jax.ShapeDtypeStruct(
            (self.num_envs, *self._obs_space.shape), self._obs_space.dtype
        )
        obs = io_callback(_reset, obs_struct, seed)

        state = EnvState()
        action = jnp.zeros(
            (self.num_envs, *self._action_space.shape), self._action_space.dtype
        )
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

    def step(
        self,
        key: Key,
        state: EnvState,
        action: Array,
    ) -> tuple[EnvState, Timestep]:
        def _step(action):
            import torch

            device = self._env.unwrapped.device
            action = torch.as_tensor(np.asarray(action), device=device)
            obs, reward, terminated, truncated, _ = self._env.step(action)
            obs = np.asarray(obs["policy"].detach().cpu(), dtype=self._obs_space.dtype)
            reward = np.asarray(reward.detach().cpu(), dtype=np.float32)
            terminated = np.asarray(terminated.detach().cpu(), dtype=np.bool_)
            truncated = np.asarray(truncated.detach().cpu(), dtype=np.bool_)
            return obs, reward, terminated, truncated

        obs, reward, terminated, truncated = io_callback(
            _step,
            (
                jax.ShapeDtypeStruct(
                    (self.num_envs, *self._obs_space.shape), self._obs_space.dtype
                ),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_),
            ),
            action,
        )
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={},
        )
        return state, timestep

    def observation_space(self) -> Space:
        return self._obs_space

    def action_space(self) -> Space:
        return self._action_space

    def horizon(self) -> int:
        return self._horizon


def make(
    env_id,
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = True,
    enable_cameras: bool = False,
    **kwargs,
):
    launch(device=device, headless=headless, enable_cameras=enable_cameras)

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 -- registers Isaac-* task ids with gymnasium
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(env_id, device=device, num_envs=num_envs, **kwargs)
    env = gym.make(env_id, cfg=env_cfg)
    env = IsaacLab(env)
    return Batched(env, num_envs=env.num_envs)
