"""Randomly initialized Qwen3 behavior cloning on AgentTrove token sequences."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import datasets
from boonta.algorithms.recurrent_bc import RecurrentBC
from boonta.networks import Categorical, Network
from boonta.networks.chunked_categorical import ChunkedCategorical
from boonta.utils import Timestep

from . import schedules


class TokenEmbedding(nn.Module):
    vocab_size: int
    features: int
    dtype: object = jnp.bfloat16

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None):
        return nn.Embed(self.vocab_size, self.features, dtype=self.dtype)(obs)


class TokenInitialization:
    """Supply the initialization timestep required by the offline podracer.

    AgentTrove has no interactive environment; evaluation uses token likelihood
    from training logs, rather than environment rollouts.
    """

    num_agents = 1

    def __init__(self, num_envs, pad_id):
        self.num_envs, self.pad_id = num_envs, pad_id

    def init(self, key):
        return None, Timestep(
            obs=jnp.full((self.num_envs,), self.pad_id, jnp.int32),
            action=jnp.zeros(self.num_envs, jnp.int32),
            reward=jnp.zeros(self.num_envs, jnp.float32),
            terminated=jnp.ones(self.num_envs, bool),
            truncated=jnp.zeros(self.num_envs, bool),
        )

    def step(self, key, state, action):
        raise NotImplementedError("AgentTrove is offline; use token likelihood, not rollouts")

    def close(self, state):
        pass


def make_network(cfg):
    spec = cfg.network
    dtype = jnp.dtype(spec.dtype)
    dense = nn.Dense(spec.vocab_size, use_bias=False, dtype=dtype)
    chunk = spec.get("logit_chunk_size", 0)
    head = ChunkedCategorical(dense, chunk) if chunk else Categorical(dense)
    return Network(
        feature_extractor=TokenEmbedding(spec.vocab_size, cfg.cell.features, dtype),
        torso=instantiate(cfg.stack),
        head=head,
    )


def make(cfg):
    if int(cfg.num_steps) > 2**31-1 and not jax.config.x64_enabled:
        raise ValueError("This token budget requires JAX_ENABLE_X64=true before Python starts")
    dataset = datasets.make(**cfg.dataset)
    if getattr(dataset, "sampling", "replacement") == "permutation":
        batch = int(cfg.algorithm.batch_size)
        draws = int(cfg.num_steps) // dataset.length
        if int(cfg.num_steps) % (batch * dataset.length * int(cfg.num_epochs)):
            raise ValueError("Budget must divide exactly across batch and refresh cycles")
        if draws // int(cfg.num_epochs) != dataset.pool_size:
            raise ValueError("Each permutation pool must be consumed exactly once per train call")
        list(dataset.rows.sample_rows(dataset.length, 0, dataset.seed))
        eligible = int((dataset.rows.index["length"] <= dataset.length).sum())
        if draws > eligible:
            raise ValueError("Without-replacement draw budget exceeds the eligible train split")
    spec = cfg.network
    if dataset.tokenizer.get_vocab_size() > spec.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds network.vocab_size")
    if dataset.length > cfg.cell.context_length:
        raise ValueError("Dataset length exceeds the Qwen3 context length")
    network = make_network(cfg)
    algorithm = RecurrentBC(
        cfg=instantiate(cfg.algorithm), network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adamw(
                schedules.learning_rate(cfg, cfg.algorithm.batch_size * dataset.length),
                weight_decay=cfg.optimizer.weight_decay,
            ),
        ),
    )
    env = TokenInitialization(cfg.environment.num_envs, dataset.pad_id)
    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    return {"algorithm": algorithm, "environment": env, "dataset": dataset, "pit": pit, "lap": lap}
