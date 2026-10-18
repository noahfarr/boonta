import argparse
import collections
import glob
import gzip
import json
import tempfile

import hydra
import jax
from hydra.core.hydra_config import HydraConfig

from hangar import recipes


def compose(overrides):
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(
            config_name="train", overrides=list(overrides), return_hydra_config=True
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def top_ops(directory, count):
    (path,) = glob.glob(f"{directory}/**/*.trace.json.gz", recursive=True)[-1:]
    with gzip.open(path) as file:
        events = json.load(file)["traceEvents"]
    names = {
        event["pid"]: event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    devices = {pid for pid, name in names.items() if "GPU" in name or "TPU" in name}
    devices = devices or set(names)
    totals = collections.Counter()
    for event in events:
        if event.get("ph") == "X" and event["pid"] in devices:
            totals[event["name"]] += event["dur"]
    total = sum(totals.values())
    for name, duration in totals.most_common(count):
        print(f"{duration / 1e3:10.1f} ms  {duration / total:6.1%}  {name[:100]}")
    print(f"{total / 1e3:10.1f} ms  total device time over {len(devices)} devices")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--num-updates", type=int, default=2)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--directory", default=None)
    args = parser.parse_args()

    cfg = compose(args.overrides)
    podracer = recipes.make(cfg)
    init_key, warm_key, trace_key = jax.random.split(jax.random.key(cfg.seed), 3)

    state = podracer.init(init_key)
    state, _ = jax.block_until_ready(podracer.train(state, warm_key, args.num_updates))

    directory = args.directory or tempfile.mkdtemp(prefix="boonta-profile-")
    with jax.profiler.trace(directory, create_perfetto_trace=True):
        state, _ = jax.block_until_ready(
            podracer.train(state, trace_key, args.num_updates)
        )
    podracer.close(state)

    print(f"trace written to {directory}")
    top_ops(directory, args.top)
