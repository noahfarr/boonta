import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def load_config(repo_id: str) -> dict:
    return json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text())
