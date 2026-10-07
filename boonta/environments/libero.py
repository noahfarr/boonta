import itertools
import os
import threading

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space

PROPRIOCEPTION_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
PROPRIOCEPTION_DIM = 3 + 4 + 2
ACTION_DIM = 7


@struct.dataclass
class EnvState:
    env_id: Array


class Libero(Environment):
    def __init__(
        self,
        task_suite_name: str,
        task_id: int,
        num_envs: int = 1,
        image_size: int = 128,
    ):
        from libero.libero import benchmark, get_libero_path

        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        task = task_suite.get_task(task_id)
        self.language_instruction = task.language
        self.bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self.init_states = task_suite.get_task_init_states(task_id)
        self.image_size = image_size
        self.num_envs = num_envs
        self.simulators = {}
        self.lock = threading.Lock()
        self.slots = itertools.count()

        self.image_shape = jax.ShapeDtypeStruct((image_size, image_size, 3), jnp.uint8)
        self.proprioception_shape = jax.ShapeDtypeStruct(
            (PROPRIOCEPTION_DIM,), jnp.float32
        )

    def extract_obs(self, obs):
        image = np.ascontiguousarray(obs["agentview_image"][::-1])
        proprioception = np.concatenate(
            [obs[key] for key in PROPRIOCEPTION_KEYS]
        ).astype(np.float32)
        return image, proprioception

    def simulator(self, slot):
        if slot not in self.simulators:
            from libero.libero.envs import OffScreenRenderEnv

            self.simulators[slot] = OffScreenRenderEnv(
                bddl_file_name=self.bddl_file,
                camera_heights=self.image_size,
                camera_widths=self.image_size,
            )
        return self.simulators[slot]

    def reset_simulator(self, sim, seed):
        sim.seed(int(seed))
        sim.reset()
        init_state = self.init_states[int(seed) % len(self.init_states)]
        return sim.set_init_state(init_state)

    def init(
        self, key: Key
    ) -> tuple[EnvState, Timestep]:
        def _reset(seed):
            with self.lock:
                slot = next(self.slots) % self.num_envs
                obs = self.reset_simulator(self.simulator(slot), seed)
            image, proprioception = self.extract_obs(obs)
            return np.int32(slot), image, proprioception

        seed = jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max)
        env_id, image, proprioception = io_callback(
            _reset,
            (
                jax.ShapeDtypeStruct((), jnp.int32),
                self.image_shape,
                self.proprioception_shape,
            ),
            seed,
        )
        state = EnvState(env_id=env_id)
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs={"image": image, "proprioception": proprioception},
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
        def _step(env_id, action):
            with self.lock:
                sim = self.simulator(int(env_id))
                obs, reward, terminated, _ = sim.step(
                    np.asarray(action, dtype=np.float32)
                )
                terminated = bool(terminated) or bool(sim.check_success())
            image, proprioception = self.extract_obs(obs)
            return image, proprioception, np.float32(reward), np.bool_(terminated)

        image, proprioception, reward, terminated = io_callback(
            _step,
            (
                self.image_shape,
                self.proprioception_shape,
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
            ),
            state.env_id,
            action,
        )
        state = EnvState(env_id=state.env_id)
        timestep = Timestep(
            obs={"image": image, "proprioception": proprioception},
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=jnp.zeros_like(terminated),
            info={},
        )
        return state, timestep

    def action_space(self) -> Space:
        return Space(shape=(ACTION_DIM,), dtype=jnp.float32, low=-1.0, high=1.0)

    def observation_space(self) -> dict[str, Space]:
        return {
            "image": Space(
                shape=(self.image_size, self.image_size, 3),
                dtype=jnp.uint8,
                low=0,
                high=255,
            ),
            "proprioception": Space(
                shape=(PROPRIOCEPTION_DIM,),
                dtype=jnp.float32,
                low=-jnp.inf,
                high=jnp.inf,
            ),
        }


def make(
    env_id,
    task_suite_name: str,
    num_envs: int = 1,
    image_size: int = 128,
    **kwargs,
):
    env = Libero(
        task_suite_name=task_suite_name,
        task_id=int(env_id),
        num_envs=num_envs,
        image_size=image_size,
    )
    return env
