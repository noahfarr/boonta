import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def load_vocab(repo_id: str) -> dict[str, int]:
    tokenizer = json.loads(Path(hf_hub_download(repo_id, "tokenizer.json")).read_text())
    return tokenizer["model"]["vocab"]
