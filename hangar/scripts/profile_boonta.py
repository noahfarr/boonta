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


def durations(directory):
    (path,) = glob.glob(f"{directory}/**/*.trace.json.gz", recursive=True)[-1:]
    with gzip.open(path) as file:
        events = json.load(file)["traceEvents"]
    names = {
        event["pid"]: event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    devices = {pid for pid, name in names.items() if "GPU" in name or "TPU" in name}
    totals = collections.Counter()
    for event in events:
        if event.get("ph") == "X" and event["pid"] in (devices or names):
            totals[event["name"]] += event["dur"]
    return totals


def profile(overrides, num_updates, seed, directory):
    cfg = compose(overrides)
    podracer = recipes.make(cfg)
    init_key, warm_key, trace_key = jax.random.split(jax.random.key(seed), 3)

    def train(state, key):
        state, _ = jax.block_until_ready(podracer.train(state, key, num_updates))
        return state

    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 1

    state = train(podracer.init(init_key), warm_key)
    with jax.profiler.trace(directory, profiler_options=options):
        state = train(state, trace_key)
    podracer.close(state)
    return durations(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--num-updates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--directory", default=None)
    args = parser.parse_args()
    directory = args.directory or tempfile.mkdtemp(prefix="boonta-profile-")
    totals = profile(args.overrides, args.num_updates, args.seed, directory)
    print(f"trace: {directory}")
    total = sum(totals.values())
    for name, duration in totals.most_common(args.top):
        print(f"{duration / 1e3:10,.1f} ms  {duration / total:6.1%}  {name[:100]}")
