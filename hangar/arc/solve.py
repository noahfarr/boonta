import argparse
import asyncio
import json
import random
import re
import time
import zlib
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

PROMPT = """You are solving an ARC-AGI puzzle. Each grid is a rectangle of cells holding colours 0-9, written one row per line with one digit per cell and no spaces.

The training examples below each show an input grid and the output grid that a single hidden transformation produces from it. Work out the transformation, check it against every training example, then apply it to the test input.

{examples}

Test input:
{test}

Reply with the test output grid only, inside one fenced block like this:
```
0120
3300
```"""


def render(grid):
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def compose(train, test_input):
    examples = "\n\n".join(
        f"Example {index + 1}\nInput:\n{render(pair['input'])}\nOutput:\n{render(pair['output'])}"
        for index, pair in enumerate(train)
    )
    return PROMPT.format(examples=examples, test=render(test_input))


def parse(content):
    if not content:
        return None
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", content, flags=re.S)
    if not blocks:
        return None
    rows = [line.strip() for line in blocks[-1].strip().splitlines()]
    if not rows or any(not re.fullmatch(r"[0-9]+", row) for row in rows):
        return None
    if len({len(row) for row in rows}) != 1 or len(rows) > 30 or len(rows[0]) > 30:
        return None
    return [[int(cell) for cell in row] for row in rows]


def transforms(count, seed):
    rng = random.Random(seed)
    chosen = [(0, False, list(range(10)))]
    dihedral = [(k, f) for f in (False, True) for k in range(4) if (k, f) != (0, False)]
    rng.shuffle(dihedral)
    for k, f in dihedral[: count - 1]:
        colours = list(range(1, 10))
        rng.shuffle(colours)
        chosen.append((k, f, [0] + colours))
    return chosen


def forward(grid, transform):
    k, f, palette = transform
    array = np.rot90(np.asarray(grid), k)
    if f:
        array = np.fliplr(array)
    return np.asarray(palette)[array].tolist()


def backward(grid, transform):
    k, f, palette = transform
    array = np.argsort(palette)[np.asarray(grid)]
    if f:
        array = np.fliplr(array)
    return np.rot90(array, -k).tolist()


def augment(task, transform):
    def move(pair):
        return {key: forward(value, transform) for key, value in pair.items()}

    return [move(pair) for pair in task["train"]], [move(pair) for pair in task["test"]]


def plan(tasks, mode, augmentations):
    jobs = []
    for name, task in tasks.items():
        for test_index, _ in enumerate(task["test"]):
            if mode == "zeroshot":
                identity = (0, False, list(range(10)))
                jobs.append((name, test_index, 0, identity, 1.0, 1))
                jobs.append((name, test_index, 1, identity, 0.7, 2))
            else:
                for index, transform in enumerate(transforms(augmentations, zlib.crc32(name.encode()))):
                    jobs.append((name, test_index, index, transform, 1.0, 100 + index))
    return jobs


async def attempt(client, semaphore, args, task, job, sink):
    name, test_index, index, transform, temperature, seed = job
    train, test = augment(task, transform)
    prompt = compose(train, test[test_index]["input"])
    async with semaphore:
        start = time.time()
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=temperature,
                top_p=0.95,
                seed=seed,
                extra_body={"top_k": 20, "min_p": 0.0, "chat_template_kwargs": {"reasoning_effort": args.effort}},
            )
            error = None
        except Exception as exception:
            response, error = None, repr(exception)
        end = time.time()
    record = {
        "task": name,
        "test_index": test_index,
        "index": index,
        "transform": [transform[0], transform[1], transform[2]],
        "temperature": temperature,
        "seed": seed,
        "start": start,
        "end": end,
        "error": error,
    }
    if response is not None:
        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None) or ""
        grid = parse(content)
        prediction = backward(grid, transform) if grid is not None else None
        target = task["test"][test_index].get("output")
        record |= {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": choice.finish_reason,
            "content": content[-4000:],
            "reasoning_chars": len(reasoning),
            "reasoning_tail": reasoning[-1500:],
            "prediction": prediction,
            "correct": prediction is not None and target is not None and prediction == target,
        }
    sink.write(json.dumps(record) + "\n")
    sink.flush()
    print(name, test_index, index, record.get("completion_tokens"), record.get("correct"), round(end - start), flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["zeroshot", "augment"], default="zeroshot")
    parser.add_argument("--augmentations", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    names = sorted(path.stem for path in Path(args.data).glob("*.json"))[args.shard :: args.shards]
    if args.limit:
        names = names[: args.limit]
    tasks = {name: json.loads((Path(args.data) / f"{name}.json").read_text()) for name in names}
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            record = json.loads(line)
            if record.get("error") is None:
                done.add((record["task"], record["test_index"], record["index"]))
    jobs = [job for job in plan(tasks, args.mode, args.augmentations) if job[:3] not in done]
    jobs.sort(key=lambda job: -sum(len(pair["input"]) * len(pair["input"][0]) for pair in tasks[job[0]]["train"]))
    print(f"{len(tasks)} tasks, {len(jobs)} jobs to run", flush=True)
    client = AsyncOpenAI(base_url=args.url, api_key="EMPTY", timeout=7200, max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    with out.open("a") as sink:
        await asyncio.gather(*(attempt(client, semaphore, args, tasks[job[0]], job, sink) for job in jobs))


if __name__ == "__main__":
    asyncio.run(main())
