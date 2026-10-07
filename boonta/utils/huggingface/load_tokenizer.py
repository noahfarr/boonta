from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def load_tokenizer(repo_id: str) -> Tokenizer:
    return Tokenizer.from_file(hf_hub_download(repo_id, "tokenizer.json"))
