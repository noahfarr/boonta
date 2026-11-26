import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.networks.low_rank_perturbations import MirroredDensePerturbation
from boonta.utils import Array, Key, PyTree, Timestep, Transition, canonicalize_dtype


@struct.dataclass(frozen=True)
class EGGROLLConfig:
    population_size: int = 64
    rank: int = 1
    sigma: float = 0.05
    temperature: float = 0.0

    def __post_init__(self):
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
    network: nn.Module
    optimizer: optax.GradientTransformation

    def init(self, key: Key, timestep: Timestep) -> tuple[EGGROLLState, None]:
        if timestep.reward.ndim != 1 or timestep.reward.shape[0] % self.cfg.population_size:
            raise ValueError(
                "EGGROLL requires single-agent rollouts with num_envs divisible by population_size"
            )
        params_key, noise_key = jax.random.split(key)
        params = self.network.init(params_key, timestep.obs, temperature=1.0)["params"]
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
            for index in range(len(params))
        )

    def step(
        self,
        state: EGGROLLState,
        carry: PyTree,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[EGGROLLState, PyTree, Array, dict]:
        dist = self.network.apply(
            {"params": state.params},
            timestep.obs,
            temperature=self.cfg.temperature * temperature,
            noise=self.draw(state.params, state.noise_key),
            sigma=self.cfg.sigma * temperature,
        )
        return state, carry, dist.sample(seed=key), {}

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
            _, input_size, _ = noise.inputs.shape
            inputs = noise.inputs.transpose(1, 0, 2).reshape(input_size, -1)
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
