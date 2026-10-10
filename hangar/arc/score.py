import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def pick(records, mode):
    ordered = sorted(records, key=lambda record: record["index"])
    if mode == "zeroshot":
        return [record.get("prediction") for record in ordered[:2]]
    votes = Counter()
    first = {}
    for position, record in enumerate(ordered):
        if record.get("prediction") is not None:
            key = json.dumps(record["prediction"])
            votes[key] += 1
            first.setdefault(key, position)
    ranked = sorted(votes, key=lambda key: (-votes[key], first[key]))
    return [json.loads(key) for key in ranked[:2]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--mode", choices=["zeroshot", "augment"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for path in args.results:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            if record.get("error") is None:
                grouped[(record["task"], record["test_index"])].append(record)
    names = sorted(path.stem for path in Path(args.data).glob("*.json"))
    tasks = {}
    total = 0.0
    for name in names:
        task = json.loads((Path(args.data) / f"{name}.json").read_text())
        tests = []
        for test_index, pair in enumerate(task["test"]):
            records = grouped.get((name, test_index), [])
            attempts = pick(records, args.mode)
            solved = any(attempt == pair["output"] for attempt in attempts)
            tests.append(
                {
                    "solved": solved,
                    "samples": len(records),
                    "valid": sum(record.get("prediction") is not None for record in records),
                    "correct_samples": sum(bool(record.get("correct")) for record in records),
                    "truncated": sum(record.get("finish_reason") == "length" for record in records),
                    "completion_tokens": sum(record.get("completion_tokens", 0) for record in records),
                    "prompt_tokens": sum(record.get("prompt_tokens", 0) for record in records),
                    "seconds": sum(record["end"] - record["start"] for record in records),
                    "attempts": attempts,
                }
            )
        score = sum(test["solved"] for test in tests) / len(tests)
        total += score
        tasks[name] = {"score": score, "tests": tests}
    summary = {
        "mode": args.mode,
        "score": total,
        "tasks": len(names),
        "percent": 100 * total / len(names),
        "test_outputs": sum(len(task["tests"]) for task in tasks.values()),
        "test_outputs_solved": sum(test["solved"] for task in tasks.values() for test in task["tests"]),
        "samples": sum(test["samples"] for task in tasks.values() for test in task["tests"]),
        "valid": sum(test["valid"] for task in tasks.values() for test in task["tests"]),
        "truncated": sum(test["truncated"] for task in tasks.values() for test in task["tests"]),
        "completion_tokens": sum(test["completion_tokens"] for task in tasks.values() for test in task["tests"]),
        "prompt_tokens": sum(test["prompt_tokens"] for task in tasks.values() for test in task["tests"]),
        "any_sample_correct": sum(
            any(test["correct_samples"] for test in task["tests"]) for task in tasks.values()
        ),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "tasks": tasks}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
