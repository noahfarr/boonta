from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.networks.low_rank_perturbations import (
    LowRankPerturbedMLP,
    MirroredDensePerturbation,
)
from boonta.utils import Array, Key, PyTree, Timestep, Transition, canonicalize_dtype

if TYPE_CHECKING:
    from boonta.environments.spaces import Space


@struct.dataclass(frozen=True)
class EGGROLLConfig:
    population_size: int = 64
    rank: int = 1
    sigma: float = 0.05
    input_support: str = "none"

    def __post_init__(self):
        if self.input_support not in ("none", "pair", "population"):
            raise ValueError("input_support must be none, pair, or population")
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError("EGGROLL population_size must be even and at least two")
        if self.rank < 1 or not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("EGGROLL requires positive rank and finite positive sigma")


@struct.dataclass(frozen=True)
class EGGROLLState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState
    noise_key: Key


def _sample_layer(key, kernel, pairs, rank):
    input_key, output_key, bias_key = jax.random.split(key, 3)
    input_size, output_size = kernel.shape
    return MirroredDensePerturbation(
        inputs=jax.random.normal(input_key, (pairs, input_size, rank), kernel.dtype)
        * rank ** -0.5,
        outputs=jax.random.normal(output_key, (pairs, rank, output_size), kernel.dtype),
        bias=jax.random.normal(bias_key, (pairs, output_size), kernel.dtype),
    )


@dataclass
class EGGROLL:
    cfg: EGGROLLConfig
    network: LowRankPerturbedMLP
    action_space: Space
    optimizer: optax.GradientTransformation

    def __post_init__(self):
        self.discrete = jnp.issubdtype(self.action_space.dtype, jnp.integer)
        if self.discrete and self.action_space.shape:
            raise ValueError("EGGROLL requires a scalar discrete or continuous action space")
        output_size = (
            self.action_space.num_actions
            if self.discrete
            else math.prod(self.action_space.shape)
        )
        if not self.network.widths or self.network.widths[-1] != output_size:
            raise ValueError("EGGROLL policy output width must match the action space")

    def init(self, key: Key, timestep: Timestep) -> tuple[EGGROLLState, None]:
        if timestep.reward.ndim != 1 or timestep.reward.shape[0] % self.cfg.population_size:
            raise ValueError(
                "EGGROLL requires single-agent rollouts with num_envs divisible by population_size"
            )
        params_key, noise_key = jax.random.split(key)
        params = self.network.init(params_key, timestep.obs)["params"]
        state = EGGROLLState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=self.optimizer.init(params),
            noise_key=noise_key,
        )
        return state, None

    def draw(self, params, key):
        return tuple(
            _sample_layer(
                jax.random.fold_in(key, index),
                params[f"dense_{index}"]["kernel"],
                self.cfg.population_size // 2,
                self.cfg.rank,
            )
            for index in range(len(self.network.widths))
        )

    def step(
        self,
        state: EGGROLLState,
        carry: PyTree,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[EGGROLLState, PyTree, Array, dict]:
        noise = self.draw(state.params, state.noise_key) if temperature else None
        outputs = self.network.apply(
            {"params": state.params},
            timestep.obs,
            noise,
            sigma=self.cfg.sigma * temperature,
        )
        if self.discrete:
            action = jnp.argmax(outputs, axis=-1) + self.action_space.low
        else:
            action = jnp.tanh(outputs).reshape((-1, *self.action_space.shape))
            low, high = self.action_space.low, self.action_space.high
            bounded = jnp.isfinite(low) & jnp.isfinite(high)
            lower = jnp.where(bounded, low, -1.0)
            upper = jnp.where(bounded, high, 1.0)
            action = jnp.clip(lower + (action + 1) * 0.5 * (upper - lower), low, high)
        return state, carry, action.astype(self.action_space.dtype), {}

    def update(
        self, state: EGGROLLState, key: Key, transitions: Transition
    ) -> EGGROLLState:
        rewards = transitions.second.reward.astype(jnp.float32)
        done = transitions.second.done
        alive = jnp.concatenate(
            (jnp.ones_like(done[:1]), jnp.cumprod(~done[:-1], axis=0).astype(bool)),
            axis=0,
        )
        returns = jnp.where(alive, rewards, 0).sum(axis=0)
        fitness = returns.reshape(self.cfg.population_size, -1).mean(axis=1)
        paired_scores = fitness[::2] - fitness[1::2]
        paired_scores = paired_scores / jnp.maximum(fitness.std(), 1e-8)
        weights = -paired_scores / (self.cfg.population_size * self.cfg.sigma)
        gradients = {}
        for index, noise in enumerate(self.draw(state.params, state.noise_key)):
            inputs = noise.inputs
            if index == 0 and self.cfg.input_support != "none":
                obs = transitions.first.obs.reshape(*alive.shape, inputs.shape[1])
                used = (obs != 0) & alive[..., None]
                support = used.reshape(len(obs), inputs.shape[0], -1, inputs.shape[1]).any(axis=(0, 2))
                if self.cfg.input_support == "population":
                    support = support.any(axis=0, keepdims=True)
                inputs = inputs * support[..., None]
            inputs = inputs.transpose(1, 0, 2).reshape(inputs.shape[1], -1)
            outputs = noise.outputs * weights[:, None, None]
            outputs = outputs.reshape(-1, outputs.shape[-1])
            gradients[f"dense_{index}"] = {
                "kernel": inputs @ outputs,
                "bias": weights @ noise.bias,
            }
        updates, optimizer_state = self.optimizer.update(
            gradients, state.optimizer_state, state.params
        )
        lox.log(
            {"eggroll/fitness_mean": fitness.mean(), "eggroll/fitness_max": fitness.max()}
        )
        return state.replace(
            params=optax.apply_updates(state.params, updates),
            optimizer_state=optimizer_state,
            noise_key=key,
        )
