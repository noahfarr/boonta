import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import torch
from transformers import GPT2LMHeadModel

from boonta.networks import GPT2
from hangar.recipes.recurrent_bc_fineweb import make_network

MODEL_ID = "gpt2"
EXPECTED_PARAMETERS = 124_439_808
LOSS_TOLERANCE = 1e-4
VOCAB_SIZE = 50257
FEATURES = 768
NUM_LAYERS = 12
NUM_HEADS = 12
CONTEXT_LENGTH = 1024


def array(reference, name):
    return reference[name].detach().cpu().numpy()


def layer_parameters(reference, index, num_heads, head_dim):
    prefix = f"transformer.h.{index}"
    features = num_heads * head_dim
    query_key_value_kernel = array(reference, f"{prefix}.attn.c_attn.weight")
    query_key_value_bias = array(reference, f"{prefix}.attn.c_attn.bias")
    attention = {
        name: {
            "kernel": kernel.reshape(features, num_heads, head_dim),
            "bias": bias.reshape(num_heads, head_dim),
        }
        for name, kernel, bias in zip(
            ("query", "key", "value"),
            np.split(query_key_value_kernel, 3, axis=-1),
            np.split(query_key_value_bias, 3),
        )
    }
    attention["output_projection"] = {
        "kernel": array(reference, f"{prefix}.attn.c_proj.weight").reshape(
            num_heads, head_dim, features
        ),
        "bias": array(reference, f"{prefix}.attn.c_proj.bias"),
    }
    return {
        "attention_norm": {
            "scale": array(reference, f"{prefix}.ln_1.weight"),
            "bias": array(reference, f"{prefix}.ln_1.bias"),
        },
        "attention": attention,
        "mlp_norm": {
            "scale": array(reference, f"{prefix}.ln_2.weight"),
            "bias": array(reference, f"{prefix}.ln_2.bias"),
        },
        "mlp_hidden": {
            "kernel": array(reference, f"{prefix}.mlp.c_fc.weight"),
            "bias": array(reference, f"{prefix}.mlp.c_fc.bias"),
        },
        "mlp_output": {
            "kernel": array(reference, f"{prefix}.mlp.c_proj.weight"),
            "bias": array(reference, f"{prefix}.mlp.c_proj.bias"),
        },
    }


def network_parameters(reference, stack):
    head_dim = stack.features // stack.num_heads
    return {
        "feature_extractor": {
            "embed": {"embedding": array(reference, "transformer.wte.weight")}
        },
        "torso": {
            "position_embedding": {"embedding": array(reference, "transformer.wpe.weight")},
            **{
                f"layers_{index}": layer_parameters(reference, index, stack.num_heads, head_dim)
                for index in range(stack.num_layers)
            },
            "output_norm": {
                "scale": array(reference, "transformer.ln_f.weight"),
                "bias": array(reference, "transformer.ln_f.bias"),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, default=Path("data/fineweb/val.bin"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--length", type=int, default=256)
    args = parser.parse_args()

    positions = args.length + 1
    windows = np.memmap(args.tokens, dtype="<u2", mode="r")[: args.batch_size * positions]
    windows = np.asarray(windows, np.int32).reshape(args.batch_size, positions)
    inputs, targets = windows[:, :-1], windows[:, 1:]

    reference_model = GPT2LMHeadModel.from_pretrained(MODEL_ID).eval()
    with torch.no_grad():
        reference_logits = reference_model(torch.as_tensor(inputs, dtype=torch.long)).logits
    reference_logits = reference_logits.cpu().numpy()
    reference_log_probs = jax.nn.log_softmax(jnp.asarray(reference_logits), axis=-1)
    reference_loss = -jnp.take_along_axis(reference_log_probs, targets[..., None], axis=-1).mean()

    stack = GPT2(
        features=FEATURES,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        context_length=CONTEXT_LENGTH,
        dtype=jnp.float32,
    )
    network = make_network(VOCAB_SIZE, FEATURES, stack, jnp.float32)
    params = network_parameters(reference_model.state_dict(), stack)
    parameter_count = sum(leaf.size for leaf in jax.tree.leaves(params))
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(f"Expected {EXPECTED_PARAMETERS} parameters, got {parameter_count}")

    done = np.zeros(inputs.shape, bool)
    done[:, 0] = True
    carry = network.initialize_carry(None, inputs.shape)
    _, dist = network.apply(
        {"params": params}, inputs, done=done, carry=carry, temperature=1.0
    )
    loss = -dist.log_prob(jnp.asarray(targets)).mean()
    largest_log_prob_gap = float(jnp.abs(dist.logits - reference_log_probs).max())
    print(f"parameters: {parameter_count}")
    print(f"reference loss: {float(reference_loss):.6f}")
    print(f"boonta loss:    {float(loss):.6f}")
    print(f"largest log-prob gap: {largest_log_prob_gap:.6f}")
    if abs(float(loss - reference_loss)) > LOSS_TOLERANCE:
        raise ValueError("Boonta GPT-2 does not match the reference model")
    print("passed")


if __name__ == "__main__":
    main()
