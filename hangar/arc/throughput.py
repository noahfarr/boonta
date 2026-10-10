import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--mode", default="zeroshot")
    args = parser.parse_args()
    rows = []
    for run in map(Path, args.runs):
        records = [json.loads(line) for line in (run / f"{args.mode}.jsonl").read_text().splitlines()]
        records = [record for record in records if record.get("error") is None]
        started = int((run / "started").read_text())
        ready = int((run / "ready").read_text())
        finished = int((run / "finished").read_text()) if (run / "finished").exists() else max(r["end"] for r in records)
        tokens = sum(record["completion_tokens"] for record in records)
        prompts = sum(record["prompt_tokens"] for record in records)
        rows.append(
            {
                "run": run.name,
                "generations": len(records),
                "completion_tokens": tokens,
                "prompt_tokens": prompts,
                "startup_seconds": ready - started,
                "active_seconds": finished - ready,
                "tokens_per_second": tokens / (finished - ready),
            }
        )
        print(json.dumps(rows[-1]))
    records = [
        json.loads(line)
        for run in map(Path, args.runs)
        for line in (run / f"{args.mode}.jsonl").read_text().splitlines()
    ]
    records = [record for record in records if record.get("error") is None]
    lengths = np.array([record["completion_tokens"] for record in records])
    seconds = np.array([record["end"] - record["start"] for record in records])
    summary = {
        "generations": len(records),
        "completion_tokens_total": int(lengths.sum()),
        "completion_tokens_mean": float(lengths.mean()),
        "completion_tokens_median": float(np.median(lengths)),
        "forced_fraction": float(np.mean([bool(record.get("forced")) for record in records])),
        "latency_seconds_mean": float(seconds.mean()),
        "latency_seconds_p90": float(np.percentile(seconds, 90)),
        "aggregate_tokens_per_second_per_gpu": float(
            sum(row["completion_tokens"] for row in rows) / sum(row["active_seconds"] for row in rows)
        ),
        "gpu_seconds": sum(row["startup_seconds"] + row["active_seconds"] for row in rows),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
