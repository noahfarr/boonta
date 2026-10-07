import argparse
import time
from contextlib import nullcontext

import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.networks import Network, Qwen3
from boonta.utils import (dequantize, load_config, load_qwen3, load_tokenizer,
                          load_weights, quantize, quantizing)


class TokenEmbedding(nn.Module):
    vocab_size: int
    features: int
    dtype: jnp.dtype | None = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None):
        return nn.Embed(
            self.vocab_size,
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="embed_tokens",
        )(obs)


def build(repo_id, context_length, dtype, param_dtype, weight_bits=None):
    config = load_config(repo_id)
    weights = load_weights(repo_id)

    torso = Qwen3(
        features=config["hidden_size"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        num_groups=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        hidden_dim=config["intermediate_size"],
        max_wavelength=config["rope_theta"],
        context_length=context_length,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    network = Network(
        feature_extractor=TokenEmbedding(
            config["vocab_size"], config["hidden_size"], dtype, param_dtype
        ),
        torso=torso,
        head=nn.Dense(
            config["vocab_size"], use_bias=False, dtype=dtype, param_dtype=param_dtype
        ),
    )

    embedding = jnp.asarray(weights["model.embed_tokens.weight"]).astype(param_dtype)
    head = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
    params = {
        "feature_extractor": {"embed_tokens": {"embedding": embedding}},
        "torso": load_qwen3(torso, weights)["params"],
        "head": {"kernel": jnp.asarray(head).T.astype(param_dtype)},
    }
    if weight_bits is not None:
        params["torso"] = quantize(params["torso"], bits=weight_bits, dtype=param_dtype)
        params["head"] = quantize(params["head"], bits=weight_bits, dtype=param_dtype)
    return network, {"params": params}


def main(
    repo_id,
    prompt,
    budget,
    batch_size,
    context_length,
    seed,
    weight_bits,
    activation_bits,
):
    tokenizer = load_tokenizer(repo_id)
    prompt_ids = tokenizer.encode(prompt).ids
    if context_length is None:
        context_length = 1 << (len(prompt_ids) + budget - 1).bit_length()
    assert (
        len(prompt_ids) <= context_length
    ), f"prompt length {len(prompt_ids)} exceeds context length {context_length}"

    context = (
        quantizing(activations=activation_bits)
        if activation_bits is not None
        else nullcontext()
    )
    with context:
        network, variables = build(
            repo_id,
            context_length=context_length,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            weight_bits=weight_bits,
        )

        @jax.jit
        def prefill(variables, carry, tokens):
            done = jnp.zeros(tokens.shape, dtype=jnp.bool_).at[:, 0].set(True)
            carry, logits = network.apply(
                dequantize(variables), tokens, None, None, done, carry
            )
            return carry, jnp.argmax(logits[:, -1], axis=-1)

        @jax.jit
        def decode(variables, carry, token):
            def step(state, _):
                carry, token = state
                batch_size, *_ = token.shape
                done = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
                carry, logits = network.apply(
                    dequantize(variables), token[:, None], None, None, done, carry
                )
                token = jnp.argmax(logits[:, -1], axis=-1)
                return (carry, token), token

            _, tokens = jax.lax.scan(step, (carry, token), None, length=budget - 1)
            return tokens

        tokens = jnp.broadcast_to(
            jnp.asarray(prompt_ids, dtype=jnp.int32), (batch_size, len(prompt_ids))
        )
        input_shape = (batch_size, network.torso.features)

        def initialize_carry():
            return network.initialize_carry(jax.random.key(seed), input_shape)

        carry, token = jax.block_until_ready(
            prefill(variables, initialize_carry(), tokens)
        )
        jax.block_until_ready(decode(variables, carry, token))

        start = time.monotonic()
        completions = jax.block_until_ready(decode(variables, carry, token))
        decode_time = time.monotonic() - start

    completion, *_ = jnp.concatenate([token[:, None], completions.T], axis=1)
    print(f"{prompt}{tokenizer.decode(completion.tolist())}")
    print(f"inference/TPS: {batch_size * (budget - 1) / decode_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-bits", type=int, default=None)
    parser.add_argument("--activation-bits", type=int, default=None)
    args = parser.parse_args()
    main(
        args.repo_id,
        args.prompt,
        args.budget,
        args.batch_size,
        args.context_length,
        args.seed,
        args.weight_bits,
        args.activation_bits,
    )
