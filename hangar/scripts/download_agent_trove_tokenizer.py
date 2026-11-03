"""Download only the pinned Qwen3 tokenizer used for AgentTrove accounting."""

import argparse
import hashlib
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download


REPO_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/tokenizer.json"))
    args = parser.parse_args()
    source = Path(hf_hub_download(REPO_ID, "tokenizer.json", revision=REVISION))
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Downloaded tokenizer does not match the accounting fingerprint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if hashlib.sha256(args.output.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f"Refusing to overwrite a different tokenizer: {args.output}")
    else:
        shutil.copyfile(source, args.output)
    print(f"Verified tokenizer: {args.output.resolve()}")


if __name__ == "__main__":
    main()
