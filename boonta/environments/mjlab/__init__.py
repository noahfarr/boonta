from contextlib import contextmanager

import jax
import jax.numpy as jnp
from flax import struct
from jax.experimental.buffer_callback import buffer_callback

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space
from ..wrappers import Batched


@struct.dataclass
class MjlabState:
    clock: Array = struct.field(metadata={"axis": None})


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
        clock = jax.ShapeDtypeStruct((), jnp.int32)
        flags = jax.ShapeDtypeStruct((self.num_envs,), jnp.int8)
        self._reset = buffer_callback(self._restart, (clock, self._observations()), has_side_effect=True)
        self._advance = buffer_callback(
            self._proceed,
            (
                clock,
                self._observations(),
                self._batched(self._action_space),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
                flags,
                flags,
            ),
            has_side_effect=True,
        )

    def observation_space(self) -> dict[str, Space]:
        return self._observation_space

    def action_space(self) -> Space:
        return self._action_space

    def time_limit(self) -> int:
        return int(self.env.max_episode_length)

    def render(self, state: MjlabState, **kwargs):
        del state
        return self.env.render(**kwargs)

    def close(self, state: MjlabState) -> None:
        del state

    def _batched(self, space: Space, dtype=None) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.num_envs, *space.shape), dtype or space.dtype)

    def _observations(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {group: self._batched(space) for group, space in self._observation_space.items()}

    @contextmanager
    def _on(self, context):
        import torch
        import warp as wp

        stream = torch.cuda.ExternalStream(context.stream, device=self.env.device)
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            yield

    def _fill(self, targets, obs) -> None:
        import torch

        for group, target in targets.items():
            torch.from_dlpack(target).copy_(obs[group])

    def _restart(self, context, out, clock, seed) -> None:
        import torch

        with self._on(context):
            obs, _ = self.env.reset(seed=int(torch.from_dlpack(seed).item()))
            next_clock, observations = out
            torch.from_dlpack(next_clock).copy_(torch.from_dlpack(clock) + 1)
            self._fill(observations, obs)

    def _proceed(self, context, out, clock, action) -> None:
        import torch

        with self._on(context):
            commanded = torch.from_dlpack(action).to(dtype=torch.float32).clone()
            obs, reward, terminated, truncated, _ = self.env.step(commanded)
            next_clock, observations, executed, rewards, terminations, truncations = out
            torch.from_dlpack(next_clock).copy_(torch.from_dlpack(clock) + 1)
            self._fill(observations, obs)
            torch.from_dlpack(executed).copy_(commanded if self.executed is None else obs[self.executed])
            torch.from_dlpack(rewards).copy_(reward)
            torch.from_dlpack(terminations).copy_(terminated)
            torch.from_dlpack(truncations).copy_(truncated)

    def init(self, key: Key) -> tuple[MjlabState, Timestep]:
        seed = jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max)
        clock, obs = self._reset(jnp.zeros((), jnp.int32), seed)
        flags = jnp.zeros((self.num_envs,), bool)
        return MjlabState(clock=clock), Timestep(
            obs=obs,
            action=jnp.zeros((self.num_envs, *self._action_space.shape), self._action_space.dtype),
            reward=jnp.zeros((self.num_envs,), jnp.float32),
            terminated=~flags,
            truncated=flags,
            info={},
        )

    def step(self, key: Key, state: MjlabState, action: Array) -> tuple[MjlabState, Timestep]:
        del key
        clock, obs, executed, reward, terminated, truncated = self._advance(state.clock, action)
        return MjlabState(clock=clock), Timestep(
            obs=obs,
            action=executed,
            reward=reward,
            terminated=terminated.astype(bool),
            truncated=truncated.astype(bool),
            info={},
        )


def make(
    env_id,
    num_envs: int = 1,
    device: str = "cuda:0",
    play: bool = False,
    motion_file: str | None = None,
    executed: str | None = None,
    render_mode: str | None = None,
):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(env_id, play=play)
    cfg.scene.num_envs = num_envs
    if motion_file is not None:
        cfg.commands["motion"].motion_file = motion_file
    env = Mjlab(
        ManagerBasedRlEnv(cfg, device=device, render_mode=render_mode),
        executed=executed,
    )
    return Batched(env, num_envs=num_envs)
