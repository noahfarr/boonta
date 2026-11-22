"""Writes and submits the paper's experiments as Determined experiments on 42.

    python hangar/scripts/42/experiments.py list [--only GROUP ...]
    python hangar/scripts/42/experiments.py write --out outputs/42
    python hangar/scripts/42/experiments.py submit --only podracers --project-id 338

Each run trains with `boonta train` from the checkout on /pfss (see run.sh) and
writes metrics.npz to /pfss/mlde/workspaces/mlde_wsp_RAI_WS/ar44ruca/runs/paper/<name>.
Scaling runs take a whole node (8 slots) and use the first N GPUs, so that no other
job shares the node and distorts the throughput. The README lists which paper
table or figure each group feeds.
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

MINATAR = ("breakout", "asterix", "freeway", "seaquest", "space_invaders")
MUJOCO = ("hopper", "walker2d", "halfcheetah", "ant", "humanoid")
OFFLINE = ("hopper", "walker2d", "halfcheetah")
SMAX = ("2s3z", "3s5z", "5m_vs_6m", "6h_vs_8z")

# PPO on MinAtar Breakout with the settings of quickstart.ipynb. hangar's defaults
# learn Breakout far more slowly, which matters for curves and time to target.
PPO_BREAKOUT = (
    "algorithm=ppo environment=gymnax/minatar/breakout environment.num_envs=8192 "
    "podracer.config.num_steps=16 algorithm.num_minibatches=16 algorithm.update_epochs=4 "
    "algorithm.clip_coefficient=0.14 algorithm.entropy_coefficient=0.003 algorithm.gamma=0.989 "
    "algorithm.gae_lambda=0.77 algorithm.normalize_advantage=true optimizer.lr=2.4e-3 "
    "optimizer.max_grad_norm=1.5"
)
PPO_CRAFTAX = "algorithm=ppo environment=craftax/craftax_classic/symbolic"
DQN_BREAKOUT = "algorithm=dqn environment=gymnax/minatar/breakout"
SEBULBA = "podracer=sebulba podracer.config.actor.start=1"


def with_num_envs(args, num_envs):
    parts = [part for part in args.split() if not part.startswith("environment.num_envs=")]
    return " ".join([*parts, f"environment.num_envs={num_envs}"])


def run(name, group, args, slots=1):
    return {"name": name, "group": group, "slots": slots, "args": " ".join(args.split())}


def check():
    return [
        run("check-ppo-breakout-2gpu", "check", f"{PPO_BREAKOUT} podracer.config.mesh.count=2 num_steps=20_000_000 seed=0", 2),
        run("check-recurrent-ppo-mingru-2gpu", "check", "algorithm=recurrent_ppo environment=gymnax/minatar/breakout cell=min_gru torso=ssm podracer.config.mesh.count=2 num_steps=5_000_000 seed=0", 2),
        run("check-ppo-breakout-sebulba", "check", f"{PPO_BREAKOUT} {SEBULBA} num_steps=20_000_000 seed=0", 2),
    ]


def podracers(seeds):
    """Figure 2. The same PPO and JAX environment with Anakin and with Sebulba."""
    runs = []
    for seed in seeds:
        runs.append(run(f"podracers-anakin-s{seed}", "podracers", f"{PPO_BREAKOUT} podracer.config.mesh.count=1 num_steps=20_000_000 seed={seed}"))
        runs.append(run(f"podracers-sebulba-s{seed}", "podracers", f"{PPO_BREAKOUT} {SEBULBA} num_steps=20_000_000 seed={seed}", 2))
    return runs


# Per-GPU work stays fixed, so the global batch grows with the number of GPUs.
WEAK = {
    "ppo-breakout": (PPO_BREAKOUT, 8192, 50_000_000, lambda d: ""),
    "ppo-craftax": (PPO_CRAFTAX, 1024, 50_000_000, lambda d: ""),
    "dqn-breakout": (DQN_BREAKOUT, 128, 5_000_000, lambda d: f"replay.capacity={262_144 * d} buffer.sample_batch_size={512 * d}"),
}

# The global batch stays fixed, so a run learns the same on any number of GPUs.
STRONG = {
    "ppo-breakout": (PPO_BREAKOUT, 8192, 20_000_000),
    "ppo-craftax": (PPO_CRAFTAX, 4096, 200_000_000),
    "dqn-breakout": (DQN_BREAKOUT, 1024, 10_000_000),
}


def weak():
    """Figure 3, left. Throughput with a fixed number of environments per GPU."""
    runs = []
    for workload, (args, envs, steps, extra) in WEAK.items():
        for devices in DEVICES:
            args_d = with_num_envs(args, envs * devices)
            runs.append(run(f"weak-{workload}-{devices}gpu", "weak",
                            f"{args_d} podracer.config.mesh.count={devices} num_steps={steps * devices} {extra(devices)} seed=0", NODE))
    return runs


def strong(seeds):
    """Figure 3, right. Speedup and time to a target return with a fixed global batch."""
    runs = []
    for workload, (args, envs, steps) in STRONG.items():
        for devices in DEVICES:
            for seed in seeds:
                runs.append(run(f"strong-{workload}-{devices}gpu-s{seed}", "strong",
                                f"{with_num_envs(args, envs)} podracer.config.mesh.count={devices} num_steps={steps} seed={seed}", NODE))
    return runs


def autoreset(seeds):
    """Appendix C. Throughput and return with each auto-reset strategy."""
    return [
        run(f"autoreset-{strategy}-s{seed}", "autoreset",
            f"{PPO_BREAKOUT} environment/auto_reset={strategy} podracer.config.mesh.count=1 num_steps=20_000_000 seed={seed}")
        for strategy in ("same_step", "next_step", "optimistic")
        for seed in seeds
    ]


def reproduction(seeds, few_seeds):
    """Table 5. Returns against published results, with hangar's settings for each combination."""
    runs = []
    for game in MINATAR:
        env = f"environment=gymnax/minatar/{game}"
        ppo = PPO_BREAKOUT if game == "breakout" else f"algorithm=ppo {env}"
        for seed in seeds:
            runs.append(run(f"reproduce-ppo-{game}-s{seed}", "reproduction", f"{ppo} podracer.config.mesh.count=1 seed={seed}"))
            runs.append(run(f"reproduce-dqn-{game}-s{seed}", "reproduction", f"algorithm=dqn {env} podracer.config.mesh.count=1 seed={seed}"))
            runs.append(run(f"reproduce-pqn-{game}-s{seed}", "reproduction", f"algorithm=pqn {env} podracer.config.mesh.count=1 seed={seed}"))
    for task in MUJOCO:
        for seed in few_seeds:
            runs.append(run(f"reproduce-sac-{task}-s{seed}", "reproduction", f"algorithm=sac environment=brax/mujoco/{task} podracer.config.mesh.count=1 seed={seed}"))
    for task in OFFLINE:
        for algorithm in ("bc", "iql"):
            for seed in few_seeds:
                runs.append(run(f"reproduce-{algorithm}-{task}-s{seed}", "reproduction", f"algorithm={algorithm} environment=brax/mujoco/{task} podracer.config.mesh.count=1 seed={seed}"))
    for scenario in SMAX:
        for algorithm in ("ippo", "mappo"):
            for seed in few_seeds:
                runs.append(run(f"reproduce-{algorithm}-{scenario}-s{seed}", "reproduction", f"algorithm={algorithm} environment=jaxmarl/smax/{scenario} podracer.config.mesh.count=1 seed={seed}"))
    return runs


def curricula(seeds, few_seeds):
    """Section 5, curricula. Recurrent PPO on JaxUED mazes with PLR and with domain randomization."""
    maze = "algorithm=recurrent_ppo environment=jaxued/maze"
    runs = []
    for seed in seeds:
        runs.append(run(f"curricula-plr-s{seed}", "curricula", f"{maze} curriculum=plr podracer.config.mesh.count=1 seed={seed}"))
        runs.append(run(f"curricula-dr-s{seed}", "curricula", f"{maze} podracer.config.mesh.count=1 seed={seed}"))
    for seed in few_seeds:
        runs.append(run(f"curricula-plr-sebulba-s{seed}", "curricula", f"{maze} curriculum=plr {SEBULBA} seed={seed}", 2))
    return runs


def experiments(seeds=range(10), few_seeds=range(5), scaling_seeds=range(3)):
    return [
        *check(),
        *podracers(seeds),
        *weak(),
        *strong(scaling_seeds),
        *autoreset(scaling_seeds),
        *reproduction(seeds, few_seeds),
        *curricula(seeds, few_seeds),
    ]


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
    parser.add_argument("--only", nargs="*", help="groups, run names or name prefixes to include")
    parser.add_argument("--seeds", type=int, default=10, help="seeds for curves and returns")
    parser.add_argument("--few-seeds", type=int, default=5, help="seeds for the slower reproduction and Sebulba runs")
    parser.add_argument("--scaling-seeds", type=int, default=3, help="seeds for strong scaling and auto-reset")
    parser.add_argument("--out", type=Path, default=Path("outputs/42"))
    parser.add_argument("--project", default="recurrent-streaming-rl")
    parser.add_argument("--project-id", type=int, default=338)
    parser.add_argument("--dry-run", action="store_true", help="print the det commands instead of running them")
    args = parser.parse_args()

    runs = experiments(range(args.seeds), range(args.few_seeds), range(args.scaling_seeds))
    if args.only:
        runs = [r for r in runs if r["group"] in args.only or any(r["name"].startswith(o) for o in args.only)]

    if args.command == "list":
        for run in runs:
            print(f"{run['group']:12s} {run['slots']} slots  {run['name']:40s} {run['args']}")
        groups = {}
        for run in runs:
            count, slots = groups.get(run["group"], (0, 0))
            groups[run["group"]] = (count + 1, slots + run["slots"])
        for group, (count, slots) in groups.items():
            print(f"{group:12s} {count:4d} runs {slots:5d} slots")
        print(f"{'total':12s} {len(runs):4d} runs {sum(r['slots'] for r in runs):5d} slots")
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
