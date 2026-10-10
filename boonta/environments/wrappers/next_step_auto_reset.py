import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from .auto_reset import AutoReset
from .wrapper import WrapperState


@struct.dataclass
class NextStepAutoResetState(WrapperState):
    needs_reset: Array


class NextStepAutoReset(AutoReset):

    def init(self, key: Key) -> tuple[NextStepAutoResetState, Timestep]:
        env_state, timestep = super().init(key)
        return NextStepAutoResetState(env_state, jnp.zeros(self.num_envs, bool)), timestep

    def step(
        self,
        key: Key,
        state: NextStepAutoResetState,
        action: Array,
    ) -> tuple[NextStepAutoResetState, Timestep]:
        def restart(key, env_state, needs_reset, action):
            key_step, key_reset = jax.random.split(key)
            env_state_step, timestep_step = self._env.step(key_step, env_state, action)
            env_state_reset, timestep_reset = self._env.init(key_reset)

            def select(reset_leaf, step_leaf):
                return jax.lax.select(needs_reset, reset_leaf, step_leaf)

            env_state = jax.tree.map(select, env_state_reset, env_state_step)
            timestep = Timestep(
                obs=jax.tree.map(select, timestep_reset.obs, timestep_step.obs),
                action=timestep_step.action,
                reward=jax.tree.map(
                    lambda r: select(jnp.zeros_like(r), r), timestep_step.reward
                ),
                terminated=select(
                    jnp.zeros_like(timestep_step.terminated), timestep_step.terminated
                ),
                truncated=select(
                    jnp.zeros_like(timestep_step.truncated), timestep_step.truncated
                ),
                info=jax.tree.map(
                    lambda leaf: select(jnp.zeros_like(leaf), leaf), timestep_step.info
                ),
            )
            return env_state, timestep.done.all(), timestep

        keys = jax.random.split(key, self.num_envs)
        env_state, needs_reset, timestep = jax.vmap(restart)(
            keys, state.env_state, state.needs_reset, action
        )
        return NextStepAutoResetState(env_state, needs_reset), timestep

    def update(self, state: NextStepAutoResetState, key: Key, **kwargs):
        return state.replace(env_state=super().update(state.env_state, key, **kwargs))

    def action_mask(self, state: NextStepAutoResetState):
        return super().action_mask(state.env_state)

    def observe(self, state: NextStepAutoResetState):
        return super().observe(state.env_state)
