import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_grpo import RecurrentGRPO
from boonta.environments.wrappers import MCP, Group, RecordEpisodeStatistics
from boonta.networks import (Categorical, Chunk, Chunked, LoRA, Network, Qwen3,
                             Unembed)
from boonta.networks.architectures.qwen3 import partition_rules as qwen3_rules
from boonta.networks.blocks.lora import partition_rules as lora_rules
from boonta.networks.pretrained import init_fn
from boonta.utils import (load_config, load_qwen3, load_tokenizer,
                          load_weights, materialize_gradients, quantize)

CAPACITY = 32
BUDGET = 4096
REPLIES = (
    "The secret number is higher.",
    "The secret number is lower.",
    "Correct.",
    "Guess a whole number from {low} to {high}.",
    "Make your first guess.",
)
SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are playing a guessing game. Find the secret whole number from {low} to {high} "
    "in at most {guesses} guesses by calling the guess tool.\n"
    "<tools>\n"
    '{{"type": "function", "function": {{"name": "guess", "description": "Guess the secret number.", '
    '"parameters": {{"type": "object", "properties": {{"number": {{"type": "integer"}}}}, "required": ["number"]}}}}}}\n'
    "</tools>\n"
    "Respond with a tool call in this format:\n"
    '<tool_call>\n{{"name": "guess", "arguments": {{"number": {example}}}}}\n</tool_call>\n'
    "The tool replies whether the secret number is higher or lower than your guess."
)
RESPONSE = (
    "<|im_end|>\n<|im_start|>user\n<tool_response>\n",
    "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
)


class TokenFeatureExtractor(nn.Module):
    embed: nn.Embed
    pad: int

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None):
        tokens = jnp.concatenate([action[..., None], obs], axis=-1)
        valid = jnp.concatenate([~done.astype(bool)[..., None], obs != self.pad], axis=-1)
        return Chunk(self.embed(tokens), valid)


def make(cfg):
    env = environments.make(**cfg.environment)
    num_guesses = env.time_limit()

    repo_id = cfg.network.repo_id
    config = load_config(repo_id)
    vocab_size = config["vocab_size"]
    tokenizer = load_tokenizer(repo_id)

    start = tokenizer.token_to_id("<tool_call>")
    end = tokenizer.token_to_id("</tool_call>")
    pad = tokenizer.token_to_id("<|endoftext|>")
    opening, closing = (
        jnp.asarray(tokenizer.encode(text).ids, jnp.int32) for text in RESPONSE
    )
    bounds = {"low": env.low, "high": env.high}
    replies = [tokenizer.encode(text.format(**bounds)).ids for text in REPLIES]
    reply_length = max(len(reply) for reply in replies)
    replies = jnp.asarray(
        [reply + [pad] * (reply_length - len(reply)) for reply in replies], jnp.int32
    )
    observation_length = len(opening) + reply_length + len(closing)

    def to_action(arguments, cursor):
        def parse(arguments, cursor):
            text = tokenizer.decode([int(token) for token in arguments[: int(cursor)]])
            try:
                number = int(json.loads(text)["arguments"]["number"])
            except Exception:
                number = 0
            return np.int32(number)

        return jax.pure_callback(
            parse,
            jax.ShapeDtypeStruct((), jnp.int32),
            arguments,
            cursor,
            vmap_method="sequential",
        )

    def to_tokens(obs):
        return jnp.concatenate([opening, replies[obs], closing])

    prompt = tokenizer.encode(
        SYSTEM_PROMPT.format(**bounds, guesses=num_guesses, example=(env.low + env.high) // 2)
    ).ids
    env = MCP(
        env,
        to_action=to_action,
        to_tokens=to_tokens,
        start=start,
        end=end,
        pad=pad,
        vocab_size=vocab_size,
        capacity=CAPACITY,
        observation_shape=(observation_length,),
        action_shape=(),
        prompt=prompt,
    )
    game_length = len(prompt) + env.time_limit() + (num_guesses + 1) * observation_length
    env = instantiate(cfg.environment.auto_reset, env)
    env = Group(
        env,
        num_steps=cfg.podracer.config.num_steps,
        group_size=cfg.algorithm.group_size,
    )

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
        context_length=game_length + game_length % 2,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    weights = load_weights(repo_id)
    pretrained_params = jax.tree.map(np.asarray, load_qwen3(qwen3, weights)["params"])
    quantized_params = jax.tree.map(np.asarray, quantize(pretrained_params))
    embeddings = np.asarray(
        jnp.asarray(weights["model.embed_tokens.weight"], dtype=param_dtype)
    )
    embed = nn.Embed(
        vocab_size,
        config["hidden_size"],
        dtype=dtype,
        param_dtype=param_dtype,
        embedding_init=init_fn(embeddings),
    )
    torso = Chunked(
        LoRA(
            block=qwen3,
            params=quantized_params,
            rank=8,
            alpha=16.0,
        ),
        budget=BUDGET,
    )
    network = Network(
        feature_extractor=TokenFeatureExtractor(embed=embed, pad=pad),
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

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {
        "algorithm": algorithm,
        "environment": env,
        "pit": pit,
        "lap": lap,
        "partition_rules": qwen3_rules + lora_rules,
    }
