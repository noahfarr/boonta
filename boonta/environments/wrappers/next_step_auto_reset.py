import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class NextStepAutoResetState(WrapperState):
    needs_reset: Array


class NextStepAutoReset(Wrapper):

    def init(
        self, key: Key
    ) -> tuple[NextStepAutoResetState, Timestep]:
        env_state, timestep = self._env.init(key)
        return NextStepAutoResetState(env_state, jnp.bool_(False)), timestep

    def step(
        self,
        key: Key,
        state: NextStepAutoResetState,
        action: Array,
    ) -> tuple[NextStepAutoResetState, Timestep]:
        key_step, key_reset = jax.random.split(key)

        env_state_step, timestep_step = self._env.step(
            key_step, state.env_state, action
        )
        env_state_reset, timestep_reset = self._env.init(key_reset)

        def select(reset_leaf, step_leaf):
            return jax.lax.select(state.needs_reset, reset_leaf, step_leaf)

        env_state = jax.tree.map(select, env_state_reset, env_state_step)
        obs = jax.tree.map(select, timestep_reset.obs, timestep_step.obs)
        terminated = select(
            jnp.zeros_like(timestep_step.terminated), timestep_step.terminated
        )
        truncated = select(
            jnp.zeros_like(timestep_step.truncated), timestep_step.truncated
        )
        reward = jax.tree.map(
            lambda r: select(jnp.zeros_like(r), r), timestep_step.reward
        )
        info = jax.tree.map(
            lambda leaf: select(jnp.zeros_like(leaf), leaf), timestep_step.info
        )

        timestep = Timestep(
            obs=obs,
            action=timestep_step.action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return NextStepAutoResetState(env_state, terminated | truncated), timestep
