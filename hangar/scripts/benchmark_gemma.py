import argparse
import time

import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.networks import Gemma, Network
from boonta.utils import load_config, load_gemma, load_tokenizer, load_weights


class TokenEmbedding(nn.Module):
    vocab_size: int
    features: int
    dtype: jnp.dtype | None = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None):
        embeddings = nn.Embed(
            self.vocab_size,
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="embed_tokens",
        )(obs)
        normalizer = jnp.asarray(self.features**0.5, dtype=embeddings.dtype)
        return embeddings * normalizer


def build(repo_id, context_length, dtype, param_dtype):
    config = load_config(repo_id)
    weights = load_weights(repo_id)

    torso = Gemma(
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
        "torso": load_gemma(torso, weights)["params"],
        "head": {"kernel": jnp.asarray(head).T.astype(param_dtype)},
    }
    return network, {"params": params}


def main(repo_id, prompt, max_new_tokens, context_length, seed):
    tokenizer = load_tokenizer(repo_id)
    prompt_ids = tokenizer.encode(prompt).ids
    assert (
        len(prompt_ids) <= context_length
    ), f"prompt length {len(prompt_ids)} exceeds context length {context_length}"

    network, variables = build(
        repo_id,
        context_length=context_length,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )

    @jax.jit
    def prefill(variables, carry, tokens):
        done = jnp.zeros(tokens.shape, dtype=jnp.bool_)
        carry, logits = network.apply(variables, tokens, None, None, done, carry)
        return carry, jnp.argmax(logits[:, -1], axis=-1)

    @jax.jit
    def decode(variables, carry, token):
        def step(state, _):
            carry, token = state
            batch_size, *_ = token.shape
            done = jnp.zeros((batch_size, 1), dtype=jnp.bool_)
            carry, logits = network.apply(
                variables, token[:, None], None, None, done, carry
            )
            token = jnp.argmax(logits[:, -1], axis=-1)
            return (carry, token), token

        _, tokens = jax.lax.scan(step, (carry, token), None, length=max_new_tokens - 1)
        return tokens

    tokens = jnp.asarray(prompt_ids, dtype=jnp.int32)[None]
    batch_size, *_ = tokens.shape
    input_shape = (batch_size, network.torso.features)

    def initialize_carry():
        return network.initialize_carry(jax.random.key(seed), input_shape)

    carry, token = jax.block_until_ready(prefill(variables, initialize_carry(), tokens))
    jax.block_until_ready(decode(variables, carry, token))

    start = time.monotonic()
    completions = jax.block_until_ready(decode(variables, carry, token))
    decode_time = time.monotonic() - start

    completion, *_ = jnp.concatenate([token[:, None], completions.T], axis=1)
    print(f"{prompt}{tokenizer.decode(completion.tolist())}")
    print(f"inference/TPS: {(max_new_tokens - 1) / decode_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="google/gemma-2b")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        args.repo_id,
        args.prompt,
        args.max_new_tokens,
        args.context_length,
        args.seed,
    )
