import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from benchmark_qwen3 import build
from chat_tui import ChatApp

from boonta.utils import dequantize, load_template, load_tokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SUGGESTED = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]


class Qwen3Chat:
    def __init__(
        self,
        repo_id="Qwen/Qwen3-1.7B",
        system="You are a helpful assistant.",
        budget=1024,
        context_length=8192,
        seed=0,
        weight_bits=None,
        think=True,
        temperature=None,
        top_k=20,
        top_p=None,
    ):
        self.repo_id = repo_id
        self.system = system
        self.budget = budget
        self.context_length = context_length
        self.think = think
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = load_tokenizer(repo_id)
        self.template = load_template(repo_id)
        self.stop_ids = {
            self.tokenizer.token_to_id(IM_END),
            self.tokenizer.token_to_id("<|endoftext|>"),
        }
        self.network, variables = build(
            repo_id,
            context_length=context_length,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            weight_bits=weight_bits,
        )
        self.variables = variables

        def sample(key, logits, temperature, top_p):
            logits = logits.astype(jnp.float32) / temperature
            values, indices = jax.lax.top_k(logits, top_k)
            probs = jax.nn.softmax(values, axis=-1)
            keep = jnp.cumsum(probs, axis=-1) - probs < top_p
            values = jnp.where(keep, values, -jnp.inf)
            choice = jax.random.categorical(key, values, axis=-1)
            return jnp.take_along_axis(indices, choice[:, None], axis=-1)[:, 0]

        @jax.jit
        def decode(carry, token, key, temperature, top_p, done):
            carry, logits = self.network.apply(
                dequantize(variables), token[:, None], None, None, done, carry
            )
            return carry, sample(key, logits[:, -1], temperature, top_p)

        self.decode = decode
        self.input_shape = (1, self.network.torso.features)
        self.seed = seed
        self.reset()
        self.warmup()

    def warmup(self):
        token = self.feed([0], *self.sampling())
        jax.block_until_ready(token)
        self.reset()

    def unload(self):
        for array in jax.tree.leaves((self.variables, self.carry)):
            array.delete()
        self.decode = None
        jax.clear_caches()

    def reset(self):
        self.carry = self.network.initialize_carry(
            jax.random.key(self.seed), self.input_shape
        )
        self.key = jax.random.key(self.seed + 1)
        self.fresh = True
        self.messages = (
            [{"role": "system", "content": self.system}] if self.system else []
        )
        self.rendered = ""
        self.tps = 0.0
        self.used = 0

    def next_key(self):
        self.key, key = jax.random.split(self.key)
        return key

    def sampling(self):
        if self.temperature is not None and self.top_p is not None:
            return self.temperature, self.top_p
        if self.think:
            return 0.6, 0.95
        return 0.7, 0.8

    def vram(self):
        stats = jax.local_devices()[0].memory_stats()
        used = stats.get("bytes_in_use", 0) / 1e9
        limit = stats.get("bytes_limit", 0) / 1e9
        if limit:
            return f"{used:.1f}/{limit:.0f} GB"
        return f"{used:.1f} GB"

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def step(self, token, temperature, top_p):
        done = jnp.array([[self.fresh]], dtype=jnp.bool_)
        self.carry, token = self.decode(
            self.carry, token, self.next_key(), temperature, top_p, done
        )
        self.fresh = False
        return token

    def feed(self, ids, temperature, top_p, token=None):
        for tid in ids:
            token = self.step(jnp.array([tid], dtype=jnp.int32), temperature, top_p)
        return token

    def stream(self, message):
        self.messages.append({"role": "user", "content": message})
        prompt = self.template.render(
            messages=self.messages,
            add_generation_prompt=True,
            enable_thinking=self.think,
        )
        if len(self.encode(prompt)) > self.context_length:
            self.messages.pop()
            raise ValueError(
                f"conversation is over the {self.context_length} context limit"
            )

        rendered_before = self.rendered
        delta = prompt[len(rendered_before) :]
        marker = f"{IM_START}assistant\n"
        cut = delta.rfind(marker) + len(marker)
        temperature, top_p = self.sampling()
        token = self.feed(self.encode(delta[:cut]), temperature, top_p)
        after_marker = self.carry
        token = self.feed(self.encode(delta[cut:]), temperature, top_p, token)

        generated, printed = [], 0
        start = time.monotonic()
        for _ in range(self.budget):
            if int(token[0]) in self.stop_ids:
                self.step(token, temperature, top_p)
                break
            next_token = self.step(token, temperature, top_p)
            generated.append(int(token[0]))
            text = self.tokenizer.decode(generated)
            if not text.endswith("�"):
                yield text[printed:]
                printed = len(text)
            token = next_token
        elapsed = time.monotonic() - start
        self.tps = len(generated) / elapsed if elapsed > 0 else 0.0
        text = self.tokenizer.decode(generated)
        tail = text[printed:]
        if tail:
            yield tail

        answer = (
            text.split("</think>", 1)[-1].lstrip("\n") if "</think>" in text else text
        )
        self.messages.append({"role": "assistant", "content": answer})
        suffix = answer + IM_END + "\n"
        self.rendered = rendered_before + delta[:cut] + suffix
        self.carry = after_marker
        self.feed(self.encode(suffix), temperature, top_p)
        self.used = len(self.encode(self.rendered))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-bits", type=int, default=None)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--accent", default="#5b8def")
    args = parser.parse_args()

    def build_engine(model=None):
        return Qwen3Chat(
            repo_id=model or args.repo_id,
            system=args.system,
            budget=args.budget,
            context_length=args.context_length,
            seed=args.seed,
            weight_bits=args.weight_bits,
            think=args.think,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    def toggle_think(app, arg):
        app.engine.think = not app.engine.think
        return f"thinking {'on' if app.engine.think else 'off'}"

    ChatApp(
        build_engine,
        title="infer",
        assistant="qwen",
        model=args.repo_id,
        accent=args.accent,
        commands={"/thinking": ("toggle reasoning", toggle_think)},
        models=SUGGESTED,
    ).run()
