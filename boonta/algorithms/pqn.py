from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import (
    Timestep,
    Transition,
    add_feature_axis,
    canonicalize_dtype,
    flatten,
    remove_feature_axis,
)
from boonta.utils.typing import Array, Key, PyTree

from . import bb8
from .bb8 import BB8Config
from .action_value_estimators import q_lambda


@struct.dataclass(frozen=True)
class PQNConfig:
    num_minibatches: int
    update_epochs: int
    gamma: float
    q_lambda: float
    bb8: BB8Config = struct.field(pytree_node=False, default=BB8Config())


@struct.dataclass(frozen=True)
class PQNState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState
    bb8: PyTree = None


@dataclass
class PQN:
    cfg: PQNConfig
    network: nn.Module
    exploration_schedule: optax.Schedule
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()
    action_value_estimator: Callable = q_lambda

    def init(self, key: Key, timestep: Timestep) -> tuple[PQNState, PyTree]:
        params = self.network.init(key, timestep.obs, temperature=1.0)
        optimizer_state = self.optimizer.init(params["params"])

        state = PQNState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=optimizer_state,
            bb8=bb8.init_state(self.cfg.bb8, ("td",)),
        )

        return state, None

    def step(
        self,
        state: PQNState,
        carry: PyTree,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[PQNState, PyTree, Array, PyTree]:
        dist = self.network.apply(
            state.params,
            timestep.obs,
            temperature=self.exploration_schedule(state.step) * temperature,
        )
        action = dist.sample(seed=key)
        return state, carry, action, {"q_values": dist.preferences}

    def update(self, state: PQNState, key: Key, transitions: Transition) -> PQNState:
        def loss_fn(
            params: PyTree, transitions: Transition
        ) -> tuple[Array, tuple[PyTree, Array]]:
            dist, variables = self.network.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable=True,
            )
            q_values = dist.preferences
            q_value = remove_feature_axis(
                jnp.take_along_axis(
                    q_values, add_feature_axis(transitions.second.action), axis=-1
                )
            )
            target_q_value = transitions.aux["target_q_value"]
            td_error = (q_value - target_q_value) * (1.0 - transitions.second.truncated)
            loss = 0.5 * (td_error**2).mean()
            def apply(params: PyTree) -> PyTree:
                return self.network.apply(
                    params, transitions.first.obs, temperature=1.0
                )

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params,
                    apply=apply,
                    transitions=transitions,
                    dist=dist,
                    q_values=q_values,
                    variables=variables,
                )
            return loss, (variables, q_value)

        num_steps, num_envs = transitions.second.reward.shape
        batch_size = num_steps * num_envs

        obs = jax.tree.map(
            lambda leaf: jnp.take(leaf, -1, axis=0), transitions.second.obs
        )
        q_values = self.network.apply(
            state.params, obs, temperature=1.0
        ).preferences
        target_q_value = self.action_value_estimator(
            transitions, q_values, self.cfg.gamma, self.cfg.q_lambda
        )
        transitions = transitions.replace(
            aux={**transitions.aux, "target_q_value": target_q_value}
        )
        if self.cfg.bb8.noise_scale_probe:
            state = self.probe_noise_scale(state, transitions, loss_fn)

        transitions = jax.tree.map(
            lambda x: flatten(x, start_dim=0, end_dim=1), transitions
        )

        def minibatch_fn(state: PQNState, indices: Array) -> tuple[PQNState, tuple]:
            minibatch = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), transitions
            )
            target_q_value = minibatch.aux["target_q_value"]

            (loss, (variables, q_value)), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(state.params, minibatch)
            explained_variance = 1 - jnp.var(target_q_value - q_value) / (
                jnp.var(target_q_value) + 1e-8
            )
            lox.log(
                {
                    "critic/loss": loss,
                    "critic/explained_variance": explained_variance,
                    "critic/q_value": q_value.mean(),
                }
            )

            updates, optimizer_state = self.optimizer.update(
                grads["params"], state.optimizer_state, state.params["params"]
            )
            variables = {
                name: variables.get(name, value) for name, value in state.params.items()
            }
            params = {
                **variables,
                "params": optax.apply_updates(state.params["params"], updates),
            }

            state = state.replace(
                params=params,
                optimizer_state=optimizer_state,
            )
            return state, None

        def epoch_fn(state, key):
            permutation = jax.random.permutation(key, batch_size)
            minibatch_indices = permutation.reshape(self.cfg.num_minibatches, -1)

            state, _ = jax.lax.scan(minibatch_fn, state, minibatch_indices)

            return state, None

        keys = jax.random.split(key, self.cfg.update_epochs)
        state, _ = jax.lax.scan(epoch_fn, state, keys)

        return state

    def probe_noise_scale(
        self, state: PQNState, transitions: Transition, loss_fn: Callable
    ) -> PQNState:
        """Logs the noise scale of the TD loss at the pre-update parameters (see `bb8`)."""
        cfg = self.cfg.bb8
        num_steps, num_envs = transitions.second.reward.shape

        def gradients(batch: Transition) -> dict[str, PyTree]:
            grads, _ = jax.grad(loss_fn, has_aux=True, allow_int=True)(state.params, batch)
            return {"td": grads["params"]}

        grads = jax.vmap(gradients)(bb8.group_columns(transitions, cfg.num_groups))
        probe_state, logs = bb8.probe(
            cfg,
            state.bb8,
            grads,
            self.optimizer,
            state.optimizer_state,
            state.params["params"],
            num_envs,
            num_steps,
        )
        lox.log(logs)
        return state.replace(bb8={**state.bb8, **probe_state})
