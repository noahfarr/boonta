import argparse
import subprocess
import time
from datetime import date
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig

from boonta.utils import Transition
from hangar import recipes

COLUMNS = [
    "environment",
    "algorithm",
    "num_envs",
    "batch_size",
    "environment/SPS",
    "rollout/SPS",
    "update/SPS",
    "training/SPS",
    "device",
    "commit",
    "date",
]
LEDGER = Path(__file__).parent.parent / "benchmarks.md"


def compose(overrides):
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(
            config_name="config", overrides=list(overrides), return_hydra_config=True
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def sample(key, action, space):
    if jnp.issubdtype(action.dtype, jnp.integer):
        return jax.random.randint(
            key, action.shape, space.low, jnp.asarray(space.high) + 1, action.dtype
        )
    return jax.random.uniform(key, action.shape, action.dtype, space.low, space.high)


def benchmark(overrides, num_updates, seed):
    cfg = compose(overrides)
    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment

    num_envs = podracer.config.num_envs
    num_steps = podracer.config.num_steps
    num_agents = getattr(environment, "num_agents", 1)
    batch_size = num_envs * num_steps * num_agents
    space = environment.action_space()

    env_key, collect_key, update_key, init_key, train_key = jax.random.split(
        jax.random.key(seed), 5
    )

    @jax.jit
    def rollout(environment_state, timestep, key):
        def step(carry, key):
            environment_state, timestep = carry
            action_key, step_key = jax.random.split(key)
            action = sample(action_key, timestep.action, space)
            return environment.step(step_key, environment_state, action), None

        keys = jax.random.split(key, num_steps)
        carry, _ = jax.lax.scan(step, (environment_state, timestep), keys)
        return carry

    @jax.jit
    def collect(algorithm_state, environment_state, timestep, key):
        def step(carry, key):
            algorithm_state, environment_state, timestep = carry
            algorithm_key, environment_key = jax.random.split(key)
            algorithm_state, action, aux = algorithm.step(
                algorithm_state, algorithm_key, timestep, 1.0
            )
            environment_state, next_timestep = environment.step(
                environment_key, environment_state, action
            )
            transition = Transition(first=timestep, second=next_timestep, aux=aux)
            return (algorithm_state, environment_state, next_timestep), transition

        keys = jax.random.split(key, num_steps)
        return jax.lax.scan(step, (algorithm_state, environment_state, timestep), keys)

    update = jax.jit(algorithm.update)

    environment_state, timestep = environment.init(env_key)
    jax.block_until_ready(rollout(environment_state, timestep, env_key))

    start = time.monotonic()
    jax.block_until_ready(rollout(environment_state, timestep, env_key))
    env_time = time.monotonic() - start

    algorithm_state = algorithm.init(init_key, timestep)
    carry, _ = collect(algorithm_state, environment_state, timestep, collect_key)
    carry, transitions = jax.block_until_ready(collect(*carry, collect_key))

    start = time.monotonic()
    carry, transitions = jax.block_until_ready(collect(*carry, collect_key))
    rollout_time = time.monotonic() - start

    algorithm_state, *_ = carry
    algorithm_state = update(algorithm_state, update_key, transitions)
    jax.block_until_ready(update(algorithm_state, update_key, transitions))

    start = time.monotonic()
    jax.block_until_ready(update(algorithm_state, update_key, transitions))
    update_time = time.monotonic() - start
    environment.close(environment_state)

    def train(state):
        state = podracer.train(state, train_key, num_updates)
        if isinstance(state, tuple):
            state, _ = state
        return state

    state = jax.block_until_ready(train(train(podracer.init(init_key))))

    start = time.monotonic()
    state = jax.block_until_ready(train(state))
    train_time = time.monotonic() - start
    podracer.close(state)

    return {
        "num_envs": num_envs,
        "batch_size": batch_size,
        "environment/SPS": batch_size / env_time,
        "rollout/SPS": batch_size / rollout_time,
        "update/SPS": batch_size / update_time,
        "training/SPS": num_updates * batch_size / train_time,
    }


def combinations():
    root = Path(__file__).parent.parent / "config" / "environment"
    for path in sorted(root.rglob("*.yaml")):
        environment = str(path.relative_to(root).with_suffix(""))
        try:
            cfg = compose([f"environment={environment}"])
            namespace = cfg.environment.namespace
            suite = cfg.environment.get("suite", namespace)
        except Exception:
            continue
        for algorithm, *key in recipes.register:
            if tuple(key) != (namespace, suite):
                continue
            cfg = compose([f"algorithm={algorithm}", f"environment={environment}"])
            if "rollout" in cfg:
                yield algorithm, environment


def read(path):
    if not path.exists():
        return {}
    lines = [line for line in path.read_text().splitlines() if line.startswith("|")]
    rows = [
        dict(zip(COLUMNS, (cell.strip() for cell in line.strip("|").split("|"))))
        for line in lines[2:]
    ]
    return {(row["environment"], row["algorithm"], row["device"]): row for row in rows}


def record(path, row):
    table = read(path)
    table[row["environment"], row["algorithm"], row["device"]] = row
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in sorted(
        table.values(),
        key=lambda row: (
            row["environment"],
            row["algorithm"].removeprefix("recurrent_"),
            row["algorithm"].startswith("recurrent_"),
            row["device"],
        ),
    ):
        lines.append("| " + " | ".join(row[column] for column in COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n")


def stamp():
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
    return {
        "device": jax.devices()[0].device_kind,
        "commit": commit,
        "date": date.today().isoformat(),
    }


def main(overrides, num_updates, seed, everything, match, ledger):
    if not everything:
        for name, value in benchmark(overrides, num_updates, seed).items():
            print(f"{name}: {value:,.0f}")
        return

    rows = []
    for algorithm, environment in combinations():
        name = f"{algorithm} {environment}"
        if match not in name:
            continue
        print(name, flush=True)
        try:
            results = benchmark(
                [f"algorithm={algorithm}", f"environment={environment}", *overrides],
                num_updates,
                seed,
            )
        except Exception as error:
            print(f"  failed: {type(error).__name__}: {str(error).splitlines()[0][:120]}")
            continue
        finally:
            jax.clear_caches()
        rows.append((name, results))
        print("  " + "  ".join(f"{k}: {v:,.0f}" for k, v in results.items()), flush=True)
        if ledger:
            numbers = {column: f"{value:,.0f}" for column, value in results.items()}
            record(
                LEDGER,
                {"algorithm": algorithm, "environment": environment, **numbers, **stamp()},
            )

    width = max((len(name) for name, _ in rows), default=0)
    columns = ["environment/SPS", "rollout/SPS", "update/SPS", "training/SPS"]
    print()
    print(f"{'':{width}}  " + "  ".join(f"{c:>16}" for c in columns))
    for name, results in rows:
        print(f"{name:{width}}  " + "  ".join(f"{results[c]:>16,.0f}" for c in columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--num-updates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--match", default="")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.record and (args.overrides or not args.all):
        parser.error("--record takes --all and no overrides")
    main(args.overrides, args.num_updates, args.seed, args.all, args.match, args.record)
