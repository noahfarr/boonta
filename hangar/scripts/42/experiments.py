import argparse
import json
import os
import re
import shlex
import shutil
import ssl
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml

WORKSPACE = "/pfss/mlde/workspaces/mlde_wsp_RAI_WS"
HERE = Path(__file__).parent
RUNS = f"{WORKSPACE}/ar44ruca/runs/paper"
DEVICES = (1, 2, 4, 8)
NODE = 8

MINATAR = ("breakout", "asterix", "freeway", "space_invaders")
MUJOCO = ("hopper", "walker2d", "halfcheetah", "ant", "humanoid")
OFFLINE = ("hopper", "walker2d", "halfcheetah")
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
HELD_OUT_MAZES = ("SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped", "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3")

DETERMINISTIC = ("XLA_FLAGS=--xla_gpu_deterministic_ops=true",)
CHECKPOINTS = ("BOONTA_LOGGER=[file,orbax]",)
TRAINING_ONLY = ("num_steps", "num_epochs", "environment.num_envs", "podracer.config.num_steps")


def with_num_envs(args, num_envs):
    parts = [part for part in args.split() if not part.startswith("environment.num_envs=")]
    return " ".join([*parts, f"environment.num_envs={num_envs}"])


def run(name, group, args, slots=1, env=()):
    return {"name": name, "group": group, "slots": slots, "args": " ".join(args.split()), "env": list(env)}


def check():
    return [
        run("check-ppo-breakout-2gpu", "check", f"{PPO_BREAKOUT} podracer.config.mesh.count=2 num_steps=20_000_000 seed=0", 2),
        run("check-recurrent-ppo-mingru-2gpu", "check", "algorithm=recurrent_ppo environment=gymnax/minatar/breakout cell=min_gru torso=ssm podracer.config.mesh.count=2 num_steps=5_000_000 seed=0", 2),
        run("check-ppo-breakout-sebulba", "check", f"{PPO_BREAKOUT} {SEBULBA} num_steps=20_000_000 seed=0", 2),
    ]


def determinism():
    return [
        run(f"determinism-{copy}", "determinism",
            f"{PPO_BREAKOUT} podracer.config.mesh.count=1 num_steps=20_000_000 seed=0",
            env=(*DETERMINISTIC, f"JAX_COMPILATION_CACHE_DIR=/tmp/jax-cache-{copy}"))
        for copy in ("a", "b")
    ]


def podracers(seeds):
    runs = []
    for seed in seeds:
        runs.append(run(f"podracers-anakin-s{seed}", "podracers", f"{PPO_BREAKOUT} podracer.config.mesh.count=1 num_steps=20_000_000 seed={seed}", env=CHECKPOINTS))
        runs.append(run(f"podracers-sebulba-s{seed}", "podracers", f"{PPO_BREAKOUT} {SEBULBA} num_steps=20_000_000 seed={seed}", 2, env=CHECKPOINTS))
    return runs


WEAK = {
    "ppo-breakout": (PPO_BREAKOUT, 8192, 50_000_000, lambda d: ""),
    "ppo-craftax": (PPO_CRAFTAX, 1024, 50_000_000, lambda d: ""),
    "dqn-breakout": (DQN_BREAKOUT, 128, 5_000_000, lambda d: f"replay.capacity={262_144 * d} buffer.sample_batch_size={512 * d}"),
}

STRONG = {
    "ppo-breakout": (PPO_BREAKOUT, 65536, 200_000_000),
    "ppo-craftax": (PPO_CRAFTAX, 32768, 1_000_000_000),
}


def weak():
    runs = []
    for workload, (args, envs, steps, extra) in WEAK.items():
        for devices in DEVICES:
            args_d = with_num_envs(args, envs * devices)
            runs.append(run(f"weak-{workload}-{devices}gpu", "weak",
                            f"{args_d} podracer.config.mesh.count={devices} num_steps={steps * devices} {extra(devices)} seed=0", NODE))
    return runs


def strong(seeds):
    runs = []
    for workload, (args, envs, steps) in STRONG.items():
        for devices in DEVICES:
            for seed in seeds:
                runs.append(run(f"strong-{workload}-{devices}gpu-s{seed}", "strong",
                                f"{with_num_envs(args, envs)} podracer.config.mesh.count={devices} num_steps={steps} seed={seed}", NODE))
    return runs


def autoreset(seeds):
    return [
        run(f"autoreset-{strategy}-s{seed}", "autoreset",
            f"{PPO_BREAKOUT} environment/auto_reset={strategy} podracer.config.mesh.count=1 num_steps=20_000_000 seed={seed}")
        for strategy in ("same_step", "next_step", "optimistic")
        for seed in seeds
    ]


def reproduction(seeds, few_seeds):
    runs = []
    for game in MINATAR:
        env = f"environment=gymnax/minatar/{game}"
        ppo = PPO_BREAKOUT if game == "breakout" else f"algorithm=ppo {env}"
        for seed in seeds:
            runs.append(run(f"reproduce-ppo-{game}-s{seed}", "reproduction", f"{ppo} podracer.config.mesh.count=1 seed={seed}", env=CHECKPOINTS))
            runs.append(run(f"reproduce-dqn-{game}-s{seed}", "reproduction", f"algorithm=dqn {env} podracer.config.mesh.count=1 seed={seed}", env=CHECKPOINTS))
            runs.append(run(f"reproduce-pqn-{game}-s{seed}", "reproduction", f"algorithm=pqn {env} podracer.config.mesh.count=1 seed={seed}", env=CHECKPOINTS))
    for task in MUJOCO:
        for seed in few_seeds:
            runs.append(run(f"reproduce-sac-{task}-s{seed}", "reproduction", f"algorithm=sac environment=brax/mujoco/{task} podracer.config.mesh.count=1 seed={seed}"))
    for task in OFFLINE:
        for algorithm in ("bc", "iql"):
            for seed in few_seeds:
                runs.append(run(f"reproduce-{algorithm}-{task}-s{seed}", "reproduction", f"algorithm={algorithm} environment=brax/mujoco/{task} podracer.config.mesh.count=1 seed={seed}", env=CHECKPOINTS))
    return runs


DQN_MINATAR_PAPER = (
    "algorithm=dqn environment.num_envs=1 algorithm.updates_per_step=1 algorithm.tau=0.001 "
    "replay.capacity=100000 buffer.sample_batch_size=32 buffer.min_length_time_axis=5000 "
    "optimizer.lr=2.5e-4 exploration.end=0.1 exploration.fraction=0.021 "
    "num_steps=5_000_000 num_epochs=50"
)


MINATAR_PAPER = ("asterix", "breakout", "freeway", "seaquest", "space_invaders")


def dqn_paper(seeds):
    official = [
        run(f"dqn-paper-official-{game}-s{seed}", "dqn-paper", f"{DQN_MINATAR_PAPER} environment=gymnasium/minatar/{game} podracer.config.mesh.count=1 seed={seed}")
        for game in MINATAR_PAPER
        for seed in seeds
    ]
    gymnax = [
        run(f"dqn-paper-{game}-s{seed}", "dqn-paper", f"{DQN_MINATAR_PAPER} environment=gymnax/minatar/{game} podracer.config.mesh.count=1 seed={seed}")
        for game in MINATAR
        for seed in seeds
    ]
    return official + gymnax


CARTPOLE = "algorithm=ppo environment=gymnasium/cartpole"
PRECISION = "algorithm=recurrent_pupo environment=gymnax/minatar/breakout cell=min_gru podracer.config.mesh.count=1 num_steps=20_000_000"


def breadth(seeds):
    return [
        *(run(f"cartpole-s{seed}", "cartpole", f"{CARTPOLE} {SEBULBA} seed={seed}", 2) for seed in seeds),
        *(run(f"wordle-s{seed}", "wordle", f"algorithm=recurrent_grpo environment=dune_sea/wordle podracer.config.num_steps=6 seed={seed}") for seed in range(3)),
    ]


def precision(seeds):
    return [
        run(f"precision-{dtype}-s{seed}", "precision", f"{PRECISION} network.dtype={dtype} seed={seed}")
        for dtype in ("float32", "bfloat16")
        for seed in seeds
    ]


OVERHEAD_MINATAR = (
    "algorithm=ppo environment=gymnax/minatar/space_invaders podracer.config.num_steps=128 podracer.config.mesh.count=1 "
    "algorithm.update_epochs=4 algorithm.num_minibatches=8 algorithm.gamma=0.99 algorithm.gae_lambda=0.95 "
    "algorithm.clip_coefficient=0.2 algorithm.clip_value_loss=true algorithm.entropy_coefficient=0.01 "
    "algorithm.value_coefficient=0.5 algorithm.normalize_advantage=true optimizer.lr=5e-3 optimizer.max_grad_norm=0.5 "
    "+optimizer.anneal=linear +optimizer.eps=1e-5"
)
OVERHEAD_BRAX = (
    "algorithm=ppo environment=brax/mujoco/hopper environment.kwargs.backend=positional environment.num_envs=2048 "
    "podracer.config.num_steps=10 podracer.config.mesh.count=1 algorithm.update_epochs=4 algorithm.num_minibatches=32 "
    "algorithm.gamma=0.99 algorithm.gae_lambda=0.95 algorithm.clip_coefficient=0.2 algorithm.clip_value_loss=true "
    "algorithm.entropy_coefficient=0.0 algorithm.value_coefficient=0.5 algorithm.normalize_advantage=true "
    "optimizer.lr=3e-4 optimizer.max_grad_norm=0.5 +optimizer.eps=1e-5 num_steps=50_000_000"
)
OVERHEAD_ENVS = {64: 10_000_000, 1024: 20_000_000, 8192: 50_000_000}


def overhead(seeds):
    runs = []
    for envs, steps in OVERHEAD_ENVS.items():
        for seed in seeds:
            runs.append(run(f"overhead-purejaxrl-minatar-{envs}-s{seed}", "overhead", "", env=(f"BASELINE=minatar --seed {seed} --num-envs {envs} --total-timesteps {steps}",)))
            runs.append(run(f"overhead-boonta-minatar-{envs}-s{seed}", "overhead", f"{OVERHEAD_MINATAR} environment.num_envs={envs} num_steps={steps} seed={seed}"))
    for seed in seeds:
        runs.append(run(f"overhead-purejaxrl-brax-s{seed}", "overhead", "", env=(f"BASELINE=brax --seed {seed}",)))
        runs.append(run(f"overhead-boonta-brax-s{seed}", "overhead", f"{OVERHEAD_BRAX} seed={seed}"))
    return runs


PRETRAINING = "algorithm=recurrent_bc environment=fineweb"


def pretraining(seeds):
    runs = [run(f"pretraining-s{seed}", "pretraining", f"{PRETRAINING} seed={seed}") for seed in seeds]
    for devices in DEVICES:
        sequences = 16 * devices
        runs.append(run(
            f"pretraining-scaling-{devices}gpu", "pretraining-scaling",
            f"{PRETRAINING} algorithm.batch_size={sequences} podracer.config.mesh.count={devices} "
            f"num_steps={300 * sequences * 1024} num_epochs=3 seed=0",
            NODE,
        ))
    return runs


def multinode(seeds):
    devices = 2 * NODE
    runs = []
    for workload, (args, envs, steps, extra) in WEAK.items():
        runs.append(run(f"weak-{workload}-{devices}gpu", "multinode", f"{with_num_envs(args, envs * devices)} podracer.config.mesh.count={devices} num_steps={steps * devices} {extra(devices)} seed=0", devices))
    for workload, (args, envs, steps) in STRONG.items():
        for seed in seeds:
            runs.append(run(f"strong-{workload}-{devices}gpu-s{seed}", "multinode", f"{with_num_envs(args, envs)} podracer.config.mesh.count={devices} num_steps={steps} seed={seed}", devices))
    return runs


def actors():
    return [
        run(f"actors-{count}-s0", "actors", f"{CARTPOLE} podracer=sebulba podracer.config.actor.start=1 podracer.config.actor.count={count} environment.num_envs={64 * count} num_steps=2_000_000 seed=0", count + 1)
        for count in (1, 2, 4, 7)
    ]


def probe():
    return [
        run("probe-wordle", "probe", "algorithm=recurrent_grpo environment=dune_sea/wordle podracer.config.num_steps=6 num_steps=4096 num_epochs=2 seed=0"),
        run("probe-multinode", "probe", f"{PPO_BREAKOUT} podracer.config.mesh.count=16 environment.num_envs=16384 num_steps=20_000_000 num_epochs=2 seed=0", 16),
        run("probe-cartpole", "probe", f"{CARTPOLE} {SEBULBA} num_steps=65536 num_epochs=2 seed=0", 2),
        run("probe-actors", "probe", f"{CARTPOLE} podracer=sebulba podracer.config.actor.start=1 podracer.config.actor.count=2 num_steps=65536 num_epochs=2 seed=0", 3),
    ]


def evaluation_args(trained):
    args = [arg for arg in trained["args"].split() if arg.split("=")[0] not in TRAINING_ONLY]
    args = [arg for arg in args if not arg.startswith(("podracer=", "podracer.config.actor"))]
    if "algorithm=dqn" in args:
        args.append("environment.num_envs=128")
    return [*args, "~artisan"]


def checkpoint_steps(trained):
    import sys

    sys.path.insert(0, str(HERE.parents[2]))
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from hangar import resolvers  # noqa: F401

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(HERE.parents[1] / "config"), version_base=None):
        cfg = compose("train", overrides=trained["args"].split())
    batch = int(cfg.environment.num_envs) * int(cfg.podracer.config.num_steps)
    epoch = int(cfg.num_steps) // (batch * int(cfg.num_epochs)) * batch
    return [epoch * k for k in range(int(cfg.num_epochs) + 1)]


def evaluation(seeds, few_seeds):
    runs = []
    for trained in reproduction(seeds, few_seeds):
        if trained["env"] != list(CHECKPOINTS):
            continue
        name = "evaluate-" + trained["name"].removeprefix("reproduce-")
        checkpoint = f"checkpoint={RUNS}/{trained['name']}"
        runs.append(run(name, "evaluation", " ".join([*evaluation_args(trained), checkpoint]), env=("BOONTA_COMMAND=evaluate",)))
    return runs


def curves(seeds, few_seeds):
    trained = [*podracers(seeds), *(r for r in reproduction(seeds, few_seeds) if r["env"] == list(CHECKPOINTS) and "minatar" in r["args"])]
    runs = []
    for one in trained:
        steps = checkpoint_steps(one)
        paths = [f"{RUNS}/{one['name']}/checkpoints/latest/{step}" for step in steps]
        if one["group"] == "reproduction":
            paths[0] = "null"
        name = "curve-" + one["name"].removeprefix("reproduce-")
        runs.append(run(name, "curves", " ".join(["-m", *evaluation_args(one), "checkpoint=" + ",".join(paths)]), env=("BOONTA_COMMAND=evaluate",)))
    return runs


def curricula(seeds, few_seeds):
    maze = "algorithm=recurrent_ppo environment=jaxued/maze"
    runs = []
    for seed in seeds:
        runs.append(run(f"curricula-plr-s{seed}", "curricula", f"{maze} curriculum=plr podracer.config.mesh.count=1 seed={seed}", env=CHECKPOINTS))
        runs.append(run(f"curricula-dr-s{seed}", "curricula", f"{maze} podracer.config.mesh.count=1 seed={seed}", env=CHECKPOINTS))
    for seed in few_seeds:
        runs.append(run(f"curricula-plr-sebulba-s{seed}", "curricula", f"{maze} curriculum=plr {SEBULBA} seed={seed}", 2, env=CHECKPOINTS))
    return runs


def mazes(seeds, few_seeds):
    levels = ",".join(f"[{maze}]" for maze in HELD_OUT_MAZES)
    return [
        run(
            "mazes-" + trained["name"].removeprefix("curricula-"),
            "mazes",
            f"-m algorithm=recurrent_ppo environment=jaxued/maze checkpoint={RUNS}/{trained['name']} environment.kwargs.levels={levels} ~artisan",
            env=("BOONTA_COMMAND=evaluate",),
        )
        for trained in curricula(seeds, few_seeds)
    ]


SMOKE_STEPS = {"dqn": 16_384, "sac": 16_384, "bc": 1_024, "iql": 1_024}


def smoke():
    runs = []
    for full in [*reproduction(range(1), range(1)), *curricula(range(1), range(1))]:
        algorithm = re.search(r"algorithm=(\S+)", full["args"]).group(1)
        steps = SMOKE_STEPS.get(algorithm, 524_288)
        name = "smoke-" + full["name"].removesuffix("-s0")
        args = re.sub(r"(^|\s)num_steps=\S+", "", full["args"])
        runs.append(run(name, "smoke", f"{args} num_steps={steps} num_epochs=2", full["slots"], full["env"]))
    return runs


def experiments(seeds=range(10), few_seeds=range(5), scaling_seeds=range(3)):
    return [
        *check(),
        *determinism(),
        *podracers(seeds),
        *weak(),
        *strong(scaling_seeds),
        *autoreset(seeds),
        *reproduction(seeds, few_seeds),
        *evaluation(seeds, few_seeds),
        *curves(seeds, few_seeds),
        *dqn_paper(seeds),
        *curricula(seeds, few_seeds),
        *mazes(seeds, few_seeds),
        *smoke(),
        *breadth(seeds),
        *precision(range(3)),
        *actors(),
        *multinode(range(3)),
        *overhead(range(5)),
        *pretraining(range(3)),
        *probe(),
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
                *run["env"],
            ]
        },
        "entrypoint": "python3 -m determined.launch.torch_distributed bash run.sh" if run["slots"] > NODE else "bash run.sh",
    }


def write(runs, out, project):
    context = out / "context"
    context.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "run.sh", context / "run.sh")
    shutil.copy(HERE / "purejaxrl.py", context / "purejaxrl.py")
    paths = []
    for run in runs:
        path = out / f"{run['name']}.yaml"
        path.write_text(yaml.safe_dump(config(run, project), sort_keys=False))
        paths.append(path)
    return context, paths


def active_slots(project_id):
    master = os.environ["DET_MASTER"]
    url = f"{master}/api/v1/experiments?projectId={project_id}&states=STATE_ACTIVE&limit=1000"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['DET_USER_TOKEN']}"})
    context = ssl._create_unverified_context() if os.environ.get("DET_MASTER_CERT_FILE") == "noverify" else None
    with urllib.request.urlopen(request, context=context) as response:
        experiments = json.load(response).get("experiments", [])
    return sum(int(e["config"]["resources"]["slots_per_trial"]) for e in experiments)


def wait_for_slots(project_id, slots, max_slots):
    while active_slots(project_id) + slots > max_slots:
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "write", "submit"])
    parser.add_argument("--only", nargs="*", help="groups, run names or name prefixes to include")
    parser.add_argument("--max-slots", type=int, help="wait until the project holds at most this many slots, needs DET_MASTER and DET_USER_TOKEN")
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
    for run, path in zip(runs, paths):
        command = [*det, "experiment", "create", "--project_id", str(args.project_id), str(path), str(context)]
        print(" ".join(shlex.quote(part) for part in command))
        if not args.dry_run:
            if args.max_slots:
                wait_for_slots(args.project_id, run["slots"], args.max_slots)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
