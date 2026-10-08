import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_grpo import RecurrentGRPO
from boonta.environments.wrappers import (MCP, GroupedAutoReset, LogAction,
                                          Prompt, RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import Categorical, Chunked, LoRA, Network, Qwen3, Unembed
from boonta.networks.pretrained import init_fn
from boonta.utils import (load_config, load_qwen3, load_tokenizer,
                          load_weights, materialize_gradients, quantize)

REPO_ID = "Qwen/Qwen3-0.6B-Base"
WORD_LENGTH = 5
CAPACITY = 32
FEEDBACK = ["0", "1", "2", "3"]
SYSTEM_PROMPT = (
    "You are playing Wordle. Guess the secret five-letter word by calling the wordle tool.\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "wordle", "description": "Submit a five-letter guess.", '
    '"parameters": {"type": "object", "properties": {"guess": {"type": "string"}}, "required": ["guess"]}}}\n'
    "</tools>\n"
    "Respond with a tool call:\n"
    '<tool_call>\n{"name": "wordle", "arguments": {"guess": "crane"}}\n</tool_call>'
)


class TokenFeatureExtractor(nn.Module):
    embed: nn.Embed

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None):
        return jnp.concatenate(
            [self.embed(action)[..., None, :], self.embed(obs)], axis=-2
        )


def make(cfg):
    env = environments.make(**cfg.environment)

    config = load_config(REPO_ID)
    vocab_size = config["vocab_size"]
    tokenizer = load_tokenizer(REPO_ID)

    start = tokenizer.token_to_id("<tool_call>")
    end = tokenizer.token_to_id("</tool_call>")
    pad = tokenizer.token_to_id("<|endoftext|>")
    feedback = jnp.asarray(
        [tokenizer.token_to_id(token) for token in FEEDBACK], jnp.int32
    )

    def to_action(arguments, cursor):
        def parse(arguments, cursor):
            text = tokenizer.decode([int(token) for token in arguments[: int(cursor)]])
            try:
                guess = json.loads(text)["arguments"]["guess"].lower()
                letters = [ord(character) - ord("a") for character in guess]
            except Exception:
                letters = []
            letters = [letter for letter in letters if 0 <= letter < 26][:WORD_LENGTH]
            letters = letters + [0] * (WORD_LENGTH - len(letters))
            return np.asarray(letters, np.int32)

        return jax.pure_callback(
            parse,
            jax.ShapeDtypeStruct((WORD_LENGTH,), jnp.int32),
            arguments,
            cursor,
            vmap_method="sequential",
        )

    def to_tokens(obs):
        cells = obs.reshape(WORD_LENGTH, 3)
        revealed = cells.sum(-1) > 0
        return jnp.where(revealed, feedback[cells.argmax(-1)], feedback[3]).astype(
            jnp.int32
        )

    env = MCP(
        env,
        to_action=to_action,
        to_tokens=to_tokens,
        start=start,
        end=end,
        pad=pad,
        vocab_size=vocab_size,
        capacity=CAPACITY,
        observation_shape=(WORD_LENGTH,),
        action_shape=(),
    )
    prompt = np.asarray(tokenizer.encode(SYSTEM_PROMPT).ids, np.int32)
    env = Prompt(env, prompt, pad)
    env = SameStepAutoReset(env)
    env = LogAction(env)
    env = Vectorize(env, num_envs=cfg.environment.num_envs)
    env = GroupedAutoReset(
        env,
        num_steps=cfg.podracer.config.num_steps,
        group_size=cfg.algorithm.group_size,
    )
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)

    dtype = jnp.dtype(cfg.network.dtype)
    param_dtype = jnp.dtype(cfg.network.param_dtype)
    qwen3 = Qwen3(
        features=config["hidden_size"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        num_groups=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        hidden_dim=config["intermediate_size"],
        max_wavelength=config["rope_theta"],
        context_length=env.time_limit() * (1 + WORD_LENGTH),
        dtype=dtype,
        param_dtype=param_dtype,
    )
    weights = load_weights(REPO_ID)
    pretrained_params = jax.tree.map(np.asarray, load_qwen3(qwen3, weights)["params"])
    quantized_params = jax.tree.map(np.asarray, quantize(pretrained_params))
    embeddings = np.asarray(
        jnp.asarray(weights["model.embed_tokens.weight"], dtype=param_dtype)
    )
    torso = Chunked(
        LoRA(
            block=qwen3,
            params=quantized_params,
            rank=8,
            alpha=16.0,
        )
    )

    embed = nn.Embed(
        vocab_size,
        config["hidden_size"],
        dtype=dtype,
        param_dtype=param_dtype,
        embedding_init=init_fn(embeddings),
    )
    network = Network(
        feature_extractor=TokenFeatureExtractor(embed=embed),
        torso=torso,
        head=Categorical(Unembed(embed=embed)),
    )

    algorithm = RecurrentGRPO(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            materialize_gradients(),
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adam(cfg.optimizer.lr),
        ),
    )

    return {"algorithm": algorithm, "environment": env}
