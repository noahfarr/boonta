"""Bounded, epoch-refreshed token pools for AgentTrove trajectories."""

from pathlib import Path
import hashlib
import json
import random

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.utils import Timestep, Transition, load_tokenizer


@struct.dataclass(frozen=True)
class AgentTroveState:
    transitions: Transition = struct.field(metadata={"axis": "data"})
    next_cursor: object = struct.field(default_factory=lambda: np.int32(0))


def normalize_messages(row):
    """Validate both supported schemas without inferring roles from content."""
    messages = row.get("messages") or row.get("conversations")
    if not messages:
        if "messages" not in row and "conversations" not in row:
            raise ValueError(f"Unsupported AgentTrove columns: {sorted(row)}")
        # The concatenated dataset includes null/empty trajectories. They
        # contribute no supervised tokens, just like user-only windows.
        return []
    aliases = {"human": "user", "gpt": "assistant", "observation": "tool"}
    normalized = []
    for message in messages:
        role = message.get("role") or message.get("from")
        role = aliases.get(role, role)
        content = message.get("content")
        if content is None:
            content = message.get("value")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError(f"Unsupported AgentTrove message: role={role!r}")
        normalized.append({"role": role, "content": content})
    return normalized


def validate_tokenizer(tokenizer):
    if getattr(tokenizer, "truncation", None) or getattr(tokenizer, "padding", None):
        raise ValueError("AgentTrove requires tokenizer truncation and padding to be disabled")
    ids = [tokenizer.token_to_id(token) for token in
           ("<|endoftext|>", "<|im_start|>", "<|im_end|>")]
    if any(token is None for token in ids):
        raise ValueError("AgentTrove requires a Qwen ChatML tokenizer")
    return ids[0]


def encode_messages(row, tokenizer):
    """Encode ChatML once, retaining body offsets for host-side accounting."""
    for message in normalize_messages(row):
        header = tokenizer.encode(
            f"<|im_start|>{message['role']}\n", add_special_tokens=False
        )
        body = tokenizer.encode(message["content"] + "<|im_end|>\n", add_special_tokens=False)
        yield message, header.ids, body


def serialize(encodings, assistant_only=True):
    tokens, weights = [], []
    for message, header, encoding in encodings:
        body = encoding.ids
        tokens.extend(header + body)
        weights.extend([not assistant_only] * len(header))
        weights.extend([not assistant_only or message["role"] == "assistant"] * len(body))
    return np.asarray(tokens, np.int32), np.asarray(weights, np.float32)


def tokenize(row, tokenizer, assistant_only=True):
    """Serialize ShareGPT or role/content turns as Qwen ChatML, without a prompt."""
    return serialize(encode_messages(row, tokenizer), assistant_only)


def complete_trajectory(row, tokens, assistant_weights, length):
    """A whole recorded interaction with a task prefix, not verified success."""
    messages = normalize_messages(row)
    turns = [m for m in messages if m["role"] != "system"]
    return bool(
        turns and turns[0]["role"] == "user" and turns[0]["content"].strip()
        and turns[-1]["role"] == "assistant"
        and len(tokens) <= length and assistant_weights[1:].any()
    )


def windows(row, tokenizer, length, pad_id, assistant_only=True):
    # Selection is independent of loss mode. Both objectives see exactly the
    # same whole trajectories, targets, framing, padding, and resets.
    if "_tokens" in row:
        tokens, assistant_weights = row["_tokens"], row["_mask"]
        if not row["eligible"] or len(tokens) > length:
            return
    else:
        tokens, assistant_weights = tokenize(row, tokenizer, assistant_only=True)
        if not complete_trajectory(row, tokens, assistant_weights, length):
            return
    size = len(tokens) - 1
    weight = np.zeros(length, np.float32)
    weight[:size] = assistant_weights[1:] if assistant_only else 1.0
    inputs = np.full(length, pad_id, np.int32)
    targets = inputs.copy()
    inputs[:size], targets[:size] = tokens[:-1], tokens[1:]
    starts = np.zeros(length, bool)
    starts[0] = True
    assistant = np.zeros(length, np.float32)
    assistant[:size] = assistant_weights[1:]
    valid = np.arange(length) < size
    yield Transition(
        first=Timestep(
            obs=inputs, action=np.zeros(length, np.int32),
            reward=np.zeros(length, np.float32),
            terminated=starts, truncated=np.zeros(length, bool),
        ),
        second=Timestep(obs=None, action=targets),
        aux={"weight": weight, "assistant_weight": assistant, "valid": valid,
             "record_index": np.int32(row.get("index", -1))},
    )


class CachedRows:
    """Compact host index and memory-mapped whole trajectories from a cache."""

    def __init__(self, path, split, split_manifest=None):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if (not self.manifest.get("complete") or not self.manifest.get("cache_written")
                or self.manifest.get("format_version") != 1):
            raise ValueError("AgentTrove cache must be complete and format_version=1")
        size = self.manifest["cached_tokens"]
        if ((self.path / "tokens.bin").stat().st_size != 4 * size
                or (self.path / "mask.bin").stat().st_size != size):
            raise ValueError("AgentTrove cache binary size does not match its manifest")
        if split not in {"train", "validation"}:
            raise ValueError("Cached AgentTrove split must be train or validation")
        self.assignments = None
        self.split_manifest = None
        if split_manifest is not None:
            split_path = Path(split_manifest)
            specification = json.loads(split_path.read_text())
            if (specification["cache_revision"] != self.manifest["revision"]
                    or specification["cache_rows"] != self.manifest["rows"]
                    or specification["tokenizer_sha256"] != self.manifest["tokenizer_runtime_sha256"]):
                raise ValueError("Grouped split does not match the token cache")
            path = split_path.parent / specification["split_file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != specification["split_sha256"]:
                raise ValueError("Grouped split checksum mismatch")
            self.assignments = np.fromfile(path, dtype="u1")
            if len(self.assignments) != self.manifest["rows"] or (self.assignments > 2).any():
                raise ValueError("Invalid grouped split codes")
            self.split_manifest = specification
        self.split = split
        self.index = None
        self.orders = {}

    def sample_rows(self, length, size, seed):
        # Index metadata once on the host, never inside the JAX training scan.
        # Sampling the prefix of a bounded shuffle cannot cover a source-ordered
        # corpus. This compact index supports uniform whole-corpus pool refresh.
        if self.index is None:
            index = np.empty(self.manifest["rows"], dtype=[
                ("index", "<i8"), ("offset", "<i8"), ("length", "<i4")
            ])
            count = seen = 0
            with (self.path / "records.jsonl").open() as records:
                for line in records:
                    item = json.loads(line)
                    seen += 1
                    code = (int(self.assignments[item["index"]]) if self.assignments is not None
                            else int(item["split"] == "validation"))
                    if code == int(self.split == "validation") and item["cached"]:
                        if count >= len(index):
                            raise ValueError("Cache row count exceeds manifest")
                        index[count] = item["index"], item["offset"], item["length"]
                        count += 1
            if seen != self.manifest["rows"]:
                raise ValueError("Cache row count does not match manifest")
            self.index = index[:count]
        eligible = np.flatnonzero(self.index["length"] <= length)
        selected = random.Random(seed).sample(range(len(eligible)), min(size, len(eligible)))
        if not selected:
            return
        tokens = np.memmap(self.path / "tokens.bin", mode="r", dtype="<i4")
        mask = np.memmap(self.path / "mask.bin", mode="r", dtype="u1")
        for choice in selected:
            item = self.index[eligible[choice]]
            start, end = int(item["offset"]), int(item["offset"] + item["length"])
            yield {"index": int(item["index"]), "eligible": True,
                   "_tokens": tokens[start:end], "_mask": mask[start:end]}

    def ordered_rows(self, length, start, size, seed):
        # Build the same compact index used for ordinary pool sampling.
        if self.index is None:
            list(self.sample_rows(length, 0, seed))
        eligible = np.flatnonzero(self.index["length"] <= length)
        if start < 0 or start + size > len(eligible):
            raise ValueError("Without-replacement budget exceeds eligible trajectories")
        key = (length, seed)
        if key not in self.orders:
            self.orders[key] = np.random.default_rng(seed).permutation(eligible)
        tokens = np.memmap(self.path / "tokens.bin", mode="r", dtype="<i4")
        mask = np.memmap(self.path / "mask.bin", mode="r", dtype="u1")
        for choice in self.orders[key][start:start + size]:
            item = self.index[choice]
            offset, end = int(item["offset"]), int(item["offset"] + item["length"])
            yield {"index": int(item["index"]), "eligible": True,
                   "_tokens": tokens[offset:end], "_mask": mask[offset:end]}



class AgentTrove:
    def __init__(
        self, dataset_id="open-thoughts/AgentTrove", tokenizer_id="Qwen/Qwen3-0.6B",
        length=4096, pool_size=1024, num_devices=1, split="train",
        assistant_only=True, shuffle_buffer=1000, seed=42, revision=None,
        tokenizer=None, rows=None, window_policy="complete_trajectory", cache_dir=None, sampling="replacement", split_manifest=None,
    ):
        if length < 1 or pool_size < 1 or num_devices < 1 or pool_size % num_devices:
            raise ValueError("length and pool_size must be positive; pool_size must divide evenly across devices")
        if window_policy != "complete_trajectory":
            raise ValueError("Only the complete_trajectory window policy is supported")
        if tokenizer is None and Path(tokenizer_id).is_file():
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tokenizer_id))
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(tokenizer_id)
        self.pad_id = validate_tokenizer(self.tokenizer)
        self.length, self.pool_size = length, pool_size
        self.assistant_only, self.seed = assistant_only, seed
        if cache_dir is not None:
            if rows is not None:
                raise ValueError("Specify cache_dir or rows, not both")
            rows = CachedRows(cache_dir, split, split_manifest)
            if rows.split_manifest is not None and length > rows.split_manifest["max_length"]:
                raise ValueError("Context exceeds the grouped split eligibility limit")
            if length > rows.manifest["cache_max_length"]:
                raise ValueError("Context length exceeds the preprocessed cache limit")
            fingerprint = hashlib.sha256(self.tokenizer.to_str().encode()).hexdigest()
            if fingerprint != rows.manifest.get("tokenizer_runtime_sha256"):
                raise ValueError("Tokenizer does not match the preprocessing cache")
            if revision is not None and revision != rows.manifest["revision"]:
                raise ValueError("Dataset revision does not match the preprocessing cache")
        if rows is None:
            from datasets import load_dataset

            rows = load_dataset(dataset_id, split=split, streaming=True, revision=revision)
            if shuffle_buffer:
                rows = rows.shuffle(seed=seed, buffer_size=shuffle_buffer)
        if sampling not in {"replacement", "permutation"}:
            raise ValueError("Unknown AgentTrove sampling policy")
        if sampling == "permutation" and not isinstance(rows, CachedRows):
            raise ValueError("Permutation sampling requires a certified cache")
        self.sampling = sampling
        self.rows = rows
        self.initial = None

    def pool(self, seed, start=0):
        if hasattr(self.rows, "set_epoch"):
            self.rows.set_epoch(seed)
        selected = []
        if self.sampling == "permutation":
            rows = self.rows.ordered_rows(self.length, start, self.pool_size, self.seed)
        else:
            rows = (self.rows.sample_rows(self.length, self.pool_size, seed)
                    if isinstance(self.rows, CachedRows) else self.rows)
        for row in rows:
            for window in windows(row, self.tokenizer, self.length, self.pad_id, self.assistant_only):
                selected.append(window)
                if len(selected) == self.pool_size:
                    return AgentTroveState(jax.tree.map(lambda *xs: np.stack(xs), *selected),
                                           np.int32(start + self.pool_size))
        if not selected:
            raise ValueError(f"AgentTrove split contains no trainable complete trajectories within {self.length} tokens")
        # Small local fixtures/splits still produce a pool with a stable shape.
        selected = [selected[i % len(selected)] for i in range(self.pool_size)]
        return AgentTroveState(jax.tree.map(lambda *xs: np.stack(xs), *selected))

    def init(self):
        if self.initial is None:
            self.initial = self.pool(self.seed).replace(next_cursor=np.int32(0))
        return self.initial

    def update(self, state, key, sharding):
        seed = int(np.asarray(jax.random.bits(key, (), dtype=np.uint32)))
        start = int(np.asarray(state.next_cursor)) if self.sampling == "permutation" else 0
        return jax.device_put(self.pool(seed, start), sharding)

    def sample_at(self, state, key, batch_shape, step):
        if self.sampling == "replacement":
            return self.sample(state, key, batch_shape)
        if len(batch_shape) != 2 or batch_shape[1] != self.length:
            raise ValueError("Permutation sampling requires (batch, context)")
        start = (step // self.length) % self.pool_size
        indices = start + jnp.arange(batch_shape[0])
        return jax.tree.map(lambda x: jnp.asarray(x)[indices], state.transitions)

    def sample(self, state, key, batch_shape):
        if self.sampling == "permutation":
            raise ValueError("Permutation sampling must use Quadinaros.sample_at with the training step")
        if len(batch_shape) != 2 or batch_shape[1] != self.length:
            raise ValueError(f"AgentTrove expects (batch, {self.length}), got {batch_shape}")
        index = jax.random.randint(key, (batch_shape[0],), 0, self.pool_size)
        return jax.tree.map(lambda x: jnp.asarray(x)[index], state.transitions)

    def close(self):
        pass
