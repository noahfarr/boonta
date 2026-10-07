import json
from pathlib import Path

import jinja2
from huggingface_hub import hf_hub_download


def load_template(repo_id: str) -> jinja2.Template:
    config = json.loads(
        Path(hf_hub_download(repo_id, "tokenizer_config.json")).read_text()
    )
    return jinja2.Environment().from_string(config["chat_template"])
