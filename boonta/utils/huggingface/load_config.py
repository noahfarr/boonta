import json
from pathlib import Path


def load_config(repo_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text())
