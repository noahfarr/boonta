import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space
from ..wrappers import Batched


@struct.dataclass
class MjlabState:
    pass


class Mjlab(Environment):
    def __init__(self, env, executed: str | None = None):
        self.env = env
        self.num_envs = env.num_envs
        self.executed = executed
        self._observation_space = {
            group: Space(shape=shape, dtype=jnp.float32, low=-jnp.inf, high=jnp.inf)
            for group, shape in env.observation_manager.group_obs_dim.items()
            if group != executed
        }
        self._action_space = Space(
            shape=(env.action_manager.total_action_dim,), dtype=jnp.float32, low=-jnp.inf, high=jnp.inf
        )

    def observation_space(self) -> dict[str, Space]:
        return self._observation_space

    def action_space(self) -> Space:
        return self._action_space

    def time_limit(self) -> int:
        return int(self.env.max_episode_length)

    def _batched(self, space: Space, dtype=None) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.num_envs, *space.shape), dtype or space.dtype)

    def _observations(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {group: self._batched(space) for group, space in self._observation_space.items()}

    def _observe(self, obs) -> dict[str, np.ndarray]:
        return {group: obs[group].detach().cpu().numpy() for group in self._observation_space}

    def _reset(self, seed):
        obs, _ = self.env.reset(seed=int(seed))
        return self._observe(obs)

    def _advance(self, action):
        import torch

        obs, reward, terminated, truncated, _ = self.env.step(
            torch.as_tensor(np.asarray(action), device=self.env.device)
        )
        if self.executed is not None:
            action = obs[self.executed].detach().cpu().numpy()
        return (
            self._observe(obs),
            np.asarray(action, np.float32),
            reward.detach().cpu().numpy().astype(np.float32),
            terminated.detach().cpu().numpy(),
            truncated.detach().cpu().numpy(),
        )

    def init(self, key: Key) -> tuple[MjlabState, Timestep]:
        seed = jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max)
        obs = io_callback(self._reset, self._observations(), seed)
        flags = jnp.zeros((self.num_envs,), bool)
        return MjlabState(), Timestep(
            obs=obs,
            action=jnp.zeros((self.num_envs, *self._action_space.shape), self._action_space.dtype),
            reward=jnp.zeros((self.num_envs,), jnp.float32),
            terminated=~flags,
            truncated=flags,
            info={},
        )

    def step(self, key: Key, state: MjlabState, action: Array) -> tuple[MjlabState, Timestep]:
        flags = jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_)
        obs, action, reward, terminated, truncated = io_callback(
            self._advance,
            (
                self._observations(),
                self._batched(self._action_space),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
                flags,
                flags,
            ),
            action,
        )
        return state, Timestep(
            obs=obs, action=action, reward=reward, terminated=terminated, truncated=truncated, info={}
        )


def make(
    env_id,
    num_envs: int = 1,
    device: str = "cuda:0",
    play: bool = False,
    motion_file: str | None = None,
    executed: str | None = None,
):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from . import motion_tracking  # noqa: F401

    cfg = load_env_cfg(env_id, play=play)
    cfg.scene.num_envs = num_envs
    if motion_file is not None:
        cfg.commands["motion"].motion_file = motion_file
    env = Mjlab(ManagerBasedRlEnv(cfg, device=device), executed=executed)
    return Batched(env, num_envs=num_envs)
