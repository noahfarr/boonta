"""Writes and submits the paper's experiments as Determined experiments on 42.

    python hangar/scripts/42/experiments.py list
    python hangar/scripts/42/experiments.py write --out outputs/42
    python hangar/scripts/42/experiments.py submit --only check --project-id 338

Each run trains with `boonta train` from the checkout on /pfss (see run.sh) and
writes metrics.npz to /pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/runs/paper/<name>.
Scaling runs take a whole node (8 slots) and use the first N GPUs, so that no other
job shares the node and distorts the throughput.
"""

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml

WORKSPACE = "/pfss/mlde/workspaces/mlde_wsp_RAI_WS"
HERE = Path(__file__).parent
DEVICES = (1, 2, 4, 8)
NODE = 8

BREAKOUT = "gymnax/minatar/breakout"
CRAFTAX = "craftax/craftax_classic/symbolic"

# Per-device work stays fixed, so the global batch grows with the devices.
WEAK = {
    "ppo-breakout": {
        "args": f"algorithm=ppo environment={BREAKOUT}",
        "envs": 4096,
        "steps": 100_000_000,
    },
    "ppo-craftax": {
        "args": f"algorithm=ppo environment={CRAFTAX}",
        "envs": 1024,
        "steps": 50_000_000,
    },
    "dqn-breakout": {
        "args": f"algorithm=dqn environment={BREAKOUT}",
        "envs": 128,
        "steps": 5_000_000,
        "per_device": lambda d: f"replay.capacity={262_144 * d} buffer.sample_batch_size={512 * d}",
    },
}

# The global batch stays fixed, so the learning dynamics do not change with the devices.
STRONG = {
    "ppo-breakout": {
        "args": f"algorithm=ppo environment={BREAKOUT}",
        "envs": 8192,
        "steps": 50_000_000,
    },
    "ppo-craftax": {
        "args": f"algorithm=ppo environment={CRAFTAX}",
        "envs": 4096,
        "steps": 200_000_000,
    },
    "dqn-breakout": {
        "args": f"algorithm=dqn environment={BREAKOUT}",
        "envs": 1024,
        "steps": 5_000_000,
    },
}


def experiments(seeds):
    runs = [
        {
            "name": "check-ppo-breakout-2gpu",
            "group": "check",
            "slots": 2,
            "args": f"algorithm=ppo environment={BREAKOUT} podracer.config.mesh.count=2 num_steps=20_000_000 seed=0",
        },
        {
            "name": "check-recurrent-ppo-mingru-2gpu",
            "group": "check",
            "slots": 2,
            "args": f"algorithm=recurrent_ppo environment={BREAKOUT} cell=min_gru torso=ssm podracer.config.mesh.count=2 num_steps=5_000_000 seed=0",
        },
        {
            "name": "check-ppo-breakout-sebulba",
            "group": "check",
            "slots": 2,
            "args": f"algorithm=ppo environment={BREAKOUT} podracer=sebulba podracer.config.actor.start=1 num_steps=20_000_000 seed=0",
        },
    ]
    for workload, spec in WEAK.items():
        for devices in DEVICES:
            extra = spec["per_device"](devices) if "per_device" in spec else ""
            runs.append(
                {
                    "name": f"weak-{workload}-{devices}gpu",
                    "group": "weak",
                    "slots": NODE,
                    "args": f"{spec['args']} environment.num_envs={spec['envs'] * devices} "
                    f"podracer.config.mesh.count={devices} num_steps={spec['steps'] * devices} "
                    f"{extra} seed=0".replace("  ", " "),
                }
            )
    for workload, spec in STRONG.items():
        for devices in DEVICES:
            for seed in seeds:
                runs.append(
                    {
                        "name": f"strong-{workload}-{devices}gpu-s{seed}",
                        "group": "strong",
                        "slots": NODE,
                        "args": f"{spec['args']} environment.num_envs={spec['envs']} "
                        f"podracer.config.mesh.count={devices} num_steps={spec['steps']} seed={seed}",
                    }
                )
    for seed in seeds:
        runs.append(
            {
                "name": f"podracers-anakin-s{seed}",
                "group": "podracers",
                "slots": 1,
                "args": f"algorithm=ppo environment={BREAKOUT} podracer.config.mesh.count=1 seed={seed}",
            }
        )
        runs.append(
            {
                "name": f"podracers-sebulba-s{seed}",
                "group": "podracers",
                "slots": 2,
                "args": f"algorithm=ppo environment={BREAKOUT} podracer=sebulba podracer.config.actor.start=1 seed={seed}",
            }
        )
    return runs


def config(run, project):
    return {
        "name": f"boonta-paper-{run['name']}",
        "workspace": "RAI_WS",
        "project": project,
        "max_restarts": 0,
        "resources": {"resource_pool": "42_Compute", "slots_per_trial": run["slots"]},
        "searcher": {
            "name": "single",
            "metric": "training/episode_return",
            "smaller_is_better": False,
            "max_length": {"batches": 1},
        },
        "bind_mounts": [
            {
                "host_path": WORKSPACE,
                "container_path": WORKSPACE,
                "propagation": "rprivate",
                "read_only": False,
            }
        ],
        "environment": {
            "environment_variables": [
                f"RUN_NAME={run['name']}",
                f"BOONTA_ARGS={run['args']}",
            ]
        },
        "entrypoint": "bash run.sh",
    }


def write(runs, out, project):
    context = out / "context"
    context.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "run.sh", context / "run.sh")
    paths = []
    for run in runs:
        path = out / f"{run['name']}.yaml"
        path.write_text(yaml.safe_dump(config(run, project), sort_keys=False))
        paths.append(path)
    return context, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "write", "submit"])
    parser.add_argument("--only", nargs="*", help="groups or run names to include")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("outputs/42"))
    parser.add_argument("--project", default="recurrent-streaming-rl")
    parser.add_argument("--project-id", type=int, default=338)
    parser.add_argument("--dry-run", action="store_true", help="print the det commands instead of running them")
    args = parser.parse_args()

    runs = experiments(args.seeds)
    if args.only:
        runs = [r for r in runs if r["group"] in args.only or r["name"] in args.only]

    if args.command == "list":
        for run in runs:
            print(f"{run['group']:10s} {run['slots']} slots  {run['name']:40s} {run['args']}")
        print(f"{len(runs)} runs, {sum(r['slots'] for r in runs)} slots in total")
        return

    context, paths = write(runs, args.out, args.project)
    print(f"wrote {len(paths)} experiment configs and {context / 'run.sh'}")
    if args.command == "write":
        return

    det = shlex.split(os.environ.get("DET", "det"))
    for path in paths:
        command = [*det, "experiment", "create", "--project_id", str(args.project_id), str(path), str(context)]
        print(" ".join(shlex.quote(part) for part in command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
