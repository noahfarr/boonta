import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class Jaxtari(Environment):
    def __init__(self, env, max_frames_per_episode, frame_skip):
        self._env = env
        self._time_limit = max_frames_per_episode // frame_skip

    def init(self, key: Key) -> tuple:
        obs, state = self._env.reset(key)
        action_space = self.action_space()
        timestep = Timestep(
            obs=obs,
            action=jnp.zeros(action_space.shape, action_space.dtype),
            reward=jnp.zeros((), jnp.float32),
            terminated=jnp.ones((), bool),
            truncated=jnp.zeros((), bool),
            info={},
        )
        return state, timestep

    def step(self, key: Key, state, action: Array) -> tuple:
        action = jnp.asarray(action, jnp.int32)
        obs, state, reward, terminated, truncated, _ = self._env.step(state, action)
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.asarray(reward, jnp.float32),
            terminated=terminated & ~truncated,
            truncated=truncated,
            info={},
        )
        return state, timestep

    def observation_space(self) -> Space:
        space = self._env.observation_space()
        return Space(
            shape=space.shape, dtype=space.dtype, low=space.low, high=space.high
        )

    def action_space(self) -> Space:
        space = self._env.action_space()
        return Space(shape=space.shape, dtype=space.dtype, low=0, high=space.n - 1)

    def time_limit(self) -> int:
        return self._time_limit


def make(
    env_id,
    frame_stack_size=4,
    frame_skip=4,
    clip_reward=True,
    sticky_actions=0.25,
    episodic_life=False,
    first_fire=True,
    noop_max=30,
    max_frames_per_episode=108_000,
    native_downscaling=False,
    mods=None,
):
    import jaxtari
    from jaxtari.wrappers import AtariWrapper, PixelObsWrapper

    env = AtariWrapper(
        jaxtari.make(env_id, mods=mods),
        sticky_actions=sticky_actions,
        episodic_life=episodic_life,
        first_fire=first_fire,
        noop_max=noop_max,
        max_frames_per_episode=max_frames_per_episode,
    )
    env = PixelObsWrapper(
        env,
        do_pixel_resize=True,
        pixel_resize_shape=(84, 84),
        grayscale=True,
        use_native_downscaling=native_downscaling,
        frame_stack_size=frame_stack_size,
        frame_skip=frame_skip,
        clip_reward=clip_reward,
    )
    return Jaxtari(env, max_frames_per_episode, frame_skip)
