import argparse
import json
from pathlib import Path

from datasets import load_dataset
import numpy as np
from tokenizers import Tokenizer

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TOKENIZER_ID = "gpt2"
END_OF_TEXT = "<|endoftext|>"
DOCUMENTS_PER_BATCH = 1024
TOKEN_DTYPE = "<u2"


def encoded_documents(rows, tokenizer, end_of_text_id):
    documents = []
    for row in rows:
        documents.append(row["text"])
        if len(documents) == DOCUMENTS_PER_BATCH:
            for encoding in tokenizer.encode_batch(documents, add_special_tokens=False):
                yield [*encoding.ids, end_of_text_id]
            documents = []
    for encoding in tokenizer.encode_batch(documents, add_special_tokens=False):
        yield [*encoding.ids, end_of_text_id]


def write_tokens(path, documents, token_budget):
    written = 0
    with path.open("wb") as file:
        for document in documents:
            remaining = token_budget - written
            np.asarray(document[:remaining], TOKEN_DTYPE).tofile(file)
            written += min(len(document), remaining)
            if written == token_budget:
                return written
    raise ValueError(f"{path} received only {written} of {token_budget} tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/fineweb"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--validation-tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_ID)
    end_of_text_id = tokenizer.token_to_id(END_OF_TEXT)
    rows = load_dataset(
        DATASET_ID, DATASET_CONFIG, split="train", streaming=True, revision=args.revision
    )
    documents = encoded_documents(rows, tokenizer, end_of_text_id)
    validation_tokens = write_tokens(args.output / "val.bin", documents, args.validation_tokens)
    print(f"wrote {validation_tokens} validation tokens", flush=True)
    train_tokens = write_tokens(args.output / "train.bin", documents, args.train_tokens)
    print(f"wrote {train_tokens} train tokens", flush=True)
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "revision": args.revision,
        "tokenizer_id": TOKENIZER_ID,
        "end_of_text_id": end_of_text_id,
        "token_dtype": TOKEN_DTYPE,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
