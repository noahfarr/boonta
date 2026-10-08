import argparse
import json
from pathlib import Path

import numpy as np

from boonta.datasets import kaggriculture as store
from boonta.environments.kaggriculture import replays


def slug(name):
    kept = [character if character.isalnum() or character in "-_" else "_" for character in name]
    return "".join(kept).strip("_") or "unnamed"


def earnings(path):
    with np.load(path) as data:
        if "act_value" not in data.files:
            return None, 0
        value, terminated = data["act_value"], data["terminated"]
    ends = np.flatnonzero(terminated)
    if not ends.size:
        return None, 0
    starts = np.concatenate([[0], ends[:-1] + 1])
    return float(np.mean(value[starts])), int(ends.size)


def keep(path, parts, append):
    if append and path.exists():
        spare = path.with_suffix(".part.npz")
        store.save(spare, parts)
        parts = store.merge([path, spare])
        spare.unlink()
    return store.save(path, parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--module-version", default="1.32.7")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=4)
    parser.add_argument("--pool", default="board")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--min-episodes", type=int, default=3)
    args = parser.parse_args()

    experts = args.out / "experts"
    if not args.reuse:
        paths = sorted(
            path for path in args.replays.glob("*.json") if path.name not in ("manifest.json", "index.json")
        )
        if not paths:
            raise SystemExit(f"no replays under {args.replays}")
        print(f"replays: {len(paths)}")
        for name, parts in replays.by_player(paths, module_version=args.module_version).items():
            path = keep(experts / f"{slug(name)}.npz", parts, args.append)
            print(f"  {name[:26]:<26} {parts[2].size:6d} rows -> {path.name}")

    written = {path.stem: path for path in sorted(experts.glob("*.npz"))}
    if not written:
        raise SystemExit(f"no per-player files under {experts}")

    scored = []
    for name, path in written.items():
        mean, count = earnings(path)
        if mean is not None and count >= args.min_episodes:
            scored.append((mean, count, name))
    scored.sort(reverse=True)
    order = [name for _, _, name in scored]
    print(f"ranked {len(order)} players by mean episode return")
    for mean, count, name in scored[:6]:
        print(f"  {name[:28]:<28} ${mean:>8,.0f} over {count:3d} episodes")

    if args.holdout >= len(order):
        raise SystemExit(f"holdout {args.holdout} needs more than {len(order)} ranked players")
    spacing = len(order) / float(args.holdout)
    marks = {int(index * spacing + spacing / 2) for index in range(args.holdout)}
    held = [name for index, name in enumerate(order) if index in marks]
    pool = [name for index, name in enumerate(order) if index not in marks][: args.top]

    pooled = args.out / "pooled"
    store.save(pooled / f"{args.pool}.npz", store.merge([written[name] for name in pool]))
    store.save(pooled / "holdout.npz", store.merge([written[name] for name in held]))
    manifest = {
        "module_version": args.module_version,
        "players": len(written),
        "pool": pool,
        "holdout": held,
    }
    (pooled / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"pool {args.pool}: {', '.join(pool)}")
    print(f"holdout: {', '.join(held)}")


if __name__ == "__main__":
    main()
