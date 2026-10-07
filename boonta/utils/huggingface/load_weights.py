from pathlib import Path

import jax
from huggingface_hub import snapshot_download
from safetensors.flax import load_file


def load_weights(repo_id: str) -> dict:
    directory = Path(snapshot_download(repo_id, allow_patterns=["*.safetensors"]))
    cpu = jax.devices("cpu")[0]
    state: dict[str, jax.Array] = {}
    with jax.default_device(cpu):
        for shard in sorted(directory.glob("*.safetensors")):
            state.update(load_file(shard))
    return state
