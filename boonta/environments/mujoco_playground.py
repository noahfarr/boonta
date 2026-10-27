from typing import Any

import jax
import jax.numpy as jnp
import lox
import numpy as np
from jax.tree_util import GetAttrKey

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space
from .wrappers.wrapper import Wrapper


class MuJoCoPlayground(Environment):
    def __init__(self, env, default_camera: dict | None = None):
        import mujoco
        from mujoco.mjx._src.types import tree_path_to_attr_str

        self._env = env
        self.default_camera = -1
        if default_camera is not None:
            self.default_camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(env.mj_model, self.default_camera)
            for name, value in default_camera.items():
                value = np.asarray(value, dtype=np.float64)
                setattr(
                    self.default_camera, name, value if value.ndim else float(value)
                )
        shapes = jax.eval_shape(self._reset, jax.random.key(0))
        leaves, self.treedef = jax.tree_util.tree_flatten_with_path(shapes)
        batch_dims = {}
        if env._config.get("impl") == "warp":
            from mujoco.mjx.warp import types

            batch_dims = types._BATCH_DIM["Data"]
        buffers = []
        for path, leaf in leaves:
            match path:
                case (GetAttrKey(name="data"), *field) if not batch_dims.get(
                    tree_path_to_attr_str(tuple(field)), True
                ):
                    buffers.append(leaf)
                case _:
                    buffers.append(None)
        self.buffers = tuple(buffers)

    def _detach(self, state: Any) -> Any:
        leaves = jax.tree.leaves(state)
        return jax.tree.unflatten(
            self.treedef,
            [
                leaf if buffer is None else None
                for leaf, buffer in zip(leaves, self.buffers)
            ],
        )

    def _attach(self, state: Any) -> Any:
        kept = iter(jax.tree.leaves(state))
        return jax.tree.unflatten(
            self.treedef,
            [
                next(kept) if buffer is None else jnp.zeros(buffer.shape, buffer.dtype)
                for buffer in self.buffers
            ],
        )

    def _reset(self, key: Key) -> Any:
        state = self._env.reset(key)
        steps = jnp.zeros(key.shape[:-1], dtype=jnp.int32)
        return state.replace(info={**state.info, "steps": steps})

    def init(self, key: Key) -> tuple[Any, Timestep]:
        state = self._detach(self._reset(key))
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs=state.obs,
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
        state = self._env.step(self._attach(state), action)
        steps = state.info["steps"] + 1
        isfinite = ~jnp.isfinite(state.obs).all(axis=-1) | ~jnp.isfinite(state.reward)
        state = state.replace(
            obs=jnp.where(isfinite[..., None], 0.0, state.obs),
            reward=jnp.where(isfinite, 0.0, state.reward),
            done=jnp.where(isfinite, 1.0, state.done),
        )
        terminated = state.done.astype(jnp.bool)
        horizon = jnp.array(self._env._config.episode_length, dtype=jnp.int32)
        truncated = (steps >= horizon) & ~terminated
        state = self._detach(state.replace(info={**state.info, "steps": steps}))
        timestep = Timestep(
            obs=state.obs,
            action=action,
            reward=state.reward,
            terminated=terminated,
            truncated=truncated,
            info={
                **state.info,
                **{f"reward_terms/{name}": value for name, value in state.metrics.items()},
            },
        )
        return state, timestep

    def observe(self, state: Any) -> Array:
        return state.obs

    def observation_space(self) -> Space:
        return Space(
            shape=(self._env.observation_size,),
            dtype=jnp.float32,
            low=-jnp.inf,
            high=jnp.inf,
        )

    def action_space(self) -> Space:
        low, high = self._env.mjx_model.actuator_ctrlrange.T
        return Space(
            shape=(self._env.action_size,), dtype=jnp.float32, low=low, high=high
        )

    def time_limit(self) -> int:
        return int(self._env._config.episode_length)

    def render(
        self,
        state: Any,
        height: int = 480,
        width: int = 640,
        camera: str | None = None,
    ) -> Array:
        return self._env.render(
            self._attach(state),
            height=height,
            width=width,
            camera=self.default_camera if camera is None else camera,
        )


def make(env_id, config_overrides=None, default_camera=None, **kwargs):
    from mujoco_playground import registry

    if (config_overrides or {}).get("impl") == "warp":
        import warp

        warp.config.quiet = True
    env = registry.load(env_id, config_overrides=config_overrides, **kwargs)
    if env._config.get("impl") == "warp":
        model = env._mjx_model
        env._mjx_model = model.replace(
            opt=model.opt.replace(_impl=model.opt._impl.replace(warn_overflow=0))
        )
    return MuJoCoPlayground(env, default_camera)


def restart_clock(env_state, key):
    return env_state.replace(
        info={**env_state.info, "steps": jnp.zeros_like(env_state.info["steps"])}
    )


def pick_cube_progress(env):
    game = env._env

    def cell(env_state):
        data, info = env_state.data, env_state.info
        box = data.xpos[..., game._obj_body, :]
        gripper = data.site_xpos[..., game._gripper_site, :]
        features = [
            info["reached_box"].astype(jnp.int32),
            jnp.digitize(
                jnp.linalg.norm(box - gripper, axis=-1),
                jnp.array([0.012, 0.03, 0.06, 0.12, 0.25]),
            ),
            jnp.digitize(
                box[..., 2] - game._init_obj_pos[2], jnp.array([0.01, 0.03, 0.06, 0.1])
            ),
            jnp.digitize(
                jnp.linalg.norm(info["target_pos"] - box, axis=-1),
                jnp.array([0.02, 0.05, 0.1, 0.2]),
            ),
        ]
        key = jnp.zeros(jnp.shape(info["reached_box"]), jnp.int32)
        for feature in features:
            key = key * 8 + feature.astype(jnp.int32)
        return key

    return cell


class LogRewardTerms(Wrapper):
    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        state, timestep = self._env.step(key, state, action)
        lox.log(
            {
                name: jnp.where(timestep.done, value, jnp.nan)
                for name, value in timestep.info.items()
                if name.startswith("reward_terms/")
                and jnp.shape(value) == jnp.shape(timestep.done)
            }
        )
        return state, timestep

    def update(self, state, key, **kwargs):
        return self._env.update(state, key, **kwargs)

    def action_mask(self, state):
        return self._env.action_mask(state)

    def observe(self, state):
        return self._env.observe(state)
