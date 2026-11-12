"""Part B batch-size probes: how one PPO update's captured signal depends on the batch it sees.

At each checkpoint of a plain PPO run (policy pi):

- the evaluation set is 8 default-size rollouts (1024 environments x 64 steps) from the true
  start under pi, after a burn-in so that its states cover every episode age;
- for each N in NS (times the default 1024 environments, 64-step rollouts) 16 independent fresh
  batches are drawn; each gets one full PPO update from a copy of the checkpoint (same epochs),
  giving pi';
- on the evaluation set: c_out = mean[(pi'/pi - 1) A], delta = mean KL(pi || pi'), rho_low and
  rho_high with S_raw and S_cell of the evaluation set; on the training batch: c_in and c_out/c_in;
- the fit rho^2(N) = a N / (N + B*) per checkpoint, for rho_high (rho_low only rescales a).

Two schemes: the number of minibatches fixed at the recipe's 8 (minibatch size grows with N), and
the minibatch size fixed at the default 8192 (the number of minibatches grows with N; at N below
1/8 a single minibatch holds the whole, smaller batch).

Usage:
  uv run --no-sync python hangar/scripts/batch_probe.py RUN_DIR [--epochs 10 30 60 90] [--out DIR]
RUN_DIR is a training run saved with logger=[file,orbax] artisan=checkpoint.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4)
DEFAULT_ENVS = 1024
NUM_STEPS = 64
NUM_BATCHES = 16
NUM_EVAL_ROLLOUTS = 8
BURN_IN = 512
SCARED = 5
FRIGHTENED = 30


def compose(num_envs: int):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    import hangar.resolvers  # noqa: F401  (registers the eval resolver)

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra_compose(
            "train",
            overrides=[
                "algorithm=ppo",
                "environment=jumanji/pac_man",
                "curriculum=default",
                f"environment.num_envs={num_envs}",
                "logger=[]",
                "artisan=[]",
            ],
        )


def build(num_envs: int):
    """The recipe's PPO and environment, with the network reading the exact uint8 planes of
    encode(): the same parameters and the same arithmetic as in training."""
    from hangar.recipes import register

    cfg = compose(num_envs)
    components = register[("ppo", "jumanji", "pac_man")](cfg)
    algorithm = components["algorithm"]
    extractor = algorithm.network.feature_extractor
    network = algorithm.network.clone(
        feature_extractor=extractor.clone(
            observation_extractor=extractor.observation_extractor.clone(scared_divisor=float(FRIGHTENED))
        )
    )
    return cfg, dataclasses.replace(algorithm, network=network), components["environment"]


def encode(obs):
    """PacMan planes as exact uint8: every plane holds small counts, and the frightened plane is
    an integer timer over 30."""
    planes = obs.at[..., SCARED].multiply(FRIGHTENED)
    return jnp.round(planes).astype(jnp.uint8)


def inner_state(env_state):
    while not hasattr(env_state, "pellets"):
        env_state = env_state.env_state
    return env_state


def collect(algorithm, environment, params_state, key, num_rollouts: int, gamma: float, lam: float):
    """Burns in, then collects num_rollouts consecutive 64-step rollouts under pi. Returns uint8
    observations, actions, pi's log-probabilities, raw GAE advantages and PacMan cell keys, each
    flattened over rollouts, steps and environments, plus the last rollout's transitions."""
    from boonta.algorithms.advantage_estimators import generalized_advantage_estimator
    from boonta.environments.jumanji import pac_man
    from boonta.utils import Transition

    cell_fn = pac_man(None, bucket=32)
    init_key, burn_key, key = jax.random.split(key, 3)
    env_state, timestep = environment.init(init_key)

    def step(carry, key):
        env_state, timestep = carry
        act_key, env_key = jax.random.split(key)
        planes = timestep.replace(obs=encode(timestep.obs))
        _, _, action, aux = algorithm.step(params_state, None, act_key, planes, 1.0)
        cells = cell_fn(inner_state(env_state))
        next_state, next_timestep = environment.step(env_key, env_state, action)
        # Transitions hold the exact uint8 planes, a quarter of the float observations' memory.
        transition = Transition(
            first=planes, second=next_timestep.replace(obs=encode(next_timestep.obs)), aux=aux, carry=None
        )
        return (next_state, next_timestep), (transition, cells)

    @jax.jit
    def burn(env_state, timestep, key):
        (env_state, timestep), _ = jax.lax.scan(step, (env_state, timestep), jax.random.split(key, BURN_IN))
        return env_state, timestep

    @jax.jit
    def rollout(env_state, timestep, key):
        (env_state, timestep), (transitions, cells) = jax.lax.scan(
            step, (env_state, timestep), jax.random.split(key, NUM_STEPS)
        )
        _, last_value = algorithm.network.apply(params_state.params, transitions.second.obs[-1], temperature=1.0)
        advantages, _ = generalized_advantage_estimator(
            transitions, transitions.aux["value"], last_value[..., 0], gamma, lam
        )
        return env_state, timestep, transitions, cells, advantages

    env_state, timestep = burn(env_state, timestep, burn_key)
    out = {"obs": [], "action": [], "log_prob": [], "advantage": [], "cell": [], "valid": []}
    last = None
    for rollout_key in jax.random.split(key, num_rollouts):
        env_state, timestep, transitions, cells, advantages = rollout(env_state, timestep, rollout_key)
        flat = lambda leaf: jnp.reshape(leaf, (-1, *leaf.shape[2:]))  # noqa: E731
        out["obs"].append(flat(transitions.first.obs))
        out["action"].append(flat(transitions.second.action).astype(jnp.int32))
        out["log_prob"].append(flat(transitions.aux["log_prob"]))
        out["advantage"].append(flat(advantages))
        out["cell"].append(flat(cells))
        out["valid"].append(~flat(transitions.second.truncated).astype(bool))
        last = transitions
    return {name: jnp.concatenate(parts) for name, parts in out.items()}, last


def cell_signal(data) -> float:
    """S_cell on a fixed set: sum over cells of visit share times the debiased between-action
    variance of the per-action mean advantage (Advantage.signal with undecayed counts), clipped
    at zero."""
    from boonta.curricula.psr import GainState, signal

    cells = np.asarray(data["cell"])
    valid = np.asarray(data["valid"])
    _, rows = np.unique(cells, return_inverse=True)
    num_cells = rows.max() + 1
    num_actions = 5
    action = np.asarray(data["action"])
    advantage = np.asarray(data["advantage"], np.float64)
    counts = np.zeros((num_cells, num_actions))
    sums = np.zeros((num_cells, num_actions))
    squares = np.zeros((num_cells, num_actions))
    np.add.at(counts, (rows[valid], action[valid]), 1.0)
    np.add.at(sums, (rows[valid], action[valid]), advantage[valid])
    np.add.at(squares, (rows[valid], action[valid]), advantage[valid] ** 2)
    state = GainState(
        action_counts=jnp.asarray(counts, jnp.float32),
        advantage_sums=jnp.asarray(sums, jnp.float32),
        square_sums=jnp.asarray(squares, jnp.float32),
        probability_sums=jnp.zeros((num_cells, num_actions)),
        cell_visits=jnp.asarray(counts.sum(-1), jnp.float32),
        scale=jnp.zeros(()),
    )
    visits = counts.sum(-1)
    return float(np.sum(visits / visits.sum() * np.maximum(np.asarray(signal(state)), 0.0)))


def log_probs(algorithm, params, planes, chunk: int = 32768):
    outputs = []
    for start in range(0, planes.shape[0], chunk):
        dist, _ = algorithm.network.apply(params, planes[start : start + chunk], temperature=1.0)
        outputs.append(jax.nn.log_softmax(dist.logits.astype(jnp.float32), axis=-1))
    return jnp.concatenate(outputs)


def measure(new_log, old_log, data):
    taken_new = jnp.take_along_axis(new_log, data["action"][:, None], axis=-1)[:, 0]
    taken_old = jnp.take_along_axis(old_log, data["action"][:, None], axis=-1)[:, 0]
    valid = data["valid"]
    count = jnp.maximum(jnp.sum(valid), 1)
    c = jnp.sum(jnp.where(valid, (jnp.exp(taken_new - taken_old) - 1.0) * data["advantage"], 0.0)) / count
    kl = jnp.sum(jnp.exp(old_log) * (old_log - new_log), axis=-1)
    delta = jnp.sum(jnp.where(valid, kl, 0.0)) / count
    return float(c), float(delta)


def probe_checkpoint(checkpoint: str, seed: int):
    from boonta.curricula.psr import restore_policy

    results = {"checkpoint": checkpoint, "schemes": {}}
    cfg, algorithm, environment = build(DEFAULT_ENVS)
    gamma, lam = cfg.algorithm.gamma, cfg.algorithm.gae_lambda
    params_state, *_ = restore_policy(algorithm, environment, checkpoint, jax.random.key(seed))
    key = jax.random.key(seed + 1000)
    eval_key, key = jax.random.split(key)
    evaluation, _ = collect(algorithm, environment, params_state, eval_key, NUM_EVAL_ROLLOUTS, gamma, lam)
    old_eval = log_probs(algorithm, params_state.params, evaluation["obs"])
    valid = evaluation["valid"]
    s_raw = float(jnp.sum(jnp.where(valid, evaluation["advantage"] ** 2, 0.0)) / jnp.sum(valid))
    s_cell = cell_signal(evaluation)
    results.update(s_raw=s_raw, s_cell=s_cell, eval_samples=int(valid.sum()))
    print(f"  evaluation set: {int(valid.sum())} samples, S_raw {s_raw:.3f}, S_cell {s_cell:.4f}")

    default_minibatch = DEFAULT_ENVS * NUM_STEPS // cfg.algorithm.num_minibatches
    schemes = ("fixed_count", "fixed_size")
    rows = {scheme: [] for scheme in schemes}
    for n in NS:
        num_envs = int(round(n * DEFAULT_ENVS))
        batch = num_envs * NUM_STEPS
        _, algorithm_n, environment_n = build(num_envs)
        counts = {
            "fixed_count": cfg.algorithm.num_minibatches,
            "fixed_size": max(1, batch // default_minibatch),
        }
        updates = {
            scheme: jax.jit(
                dataclasses.replace(
                    algorithm_n, cfg=algorithm_n.cfg.replace(num_minibatches=counts[scheme])
                ).update
            )
            for scheme in schemes
        }
        values = {scheme: [] for scheme in schemes}
        for batch_key in jax.random.split(jax.random.fold_in(key, num_envs), NUM_BATCHES):
            collect_key, update_key = jax.random.split(batch_key)
            # One fresh batch, updated under both schemes from the same copy of the checkpoint.
            train_data, transitions = collect(algorithm_n, environment_n, params_state, collect_key, 1, gamma, lam)
            old_train = log_probs(algorithm, params_state.params, train_data["obs"])
            for scheme in schemes:
                updated = updates[scheme](params_state, update_key, transitions)
                c_out, delta = measure(log_probs(algorithm, updated.params, evaluation["obs"]), old_eval, evaluation)
                c_in, _ = measure(log_probs(algorithm, updated.params, train_data["obs"]), old_train, train_data)
                values[scheme].append(
                    {
                        "c_out": c_out,
                        "c_in": c_in,
                        "delta": delta,
                        "rho_low": c_out / np.sqrt(max(2 * delta * s_raw, 1e-30)),
                        "rho_high": c_out / np.sqrt(max(2 * delta * s_cell, 1e-30)),
                    }
                )
        for scheme in schemes:
            per_batch = values[scheme]
            summary = {name: float(np.mean([v[name] for v in per_batch])) for name in per_batch[0]}
            summary.update(
                N=n,
                num_envs=num_envs,
                num_minibatches=counts[scheme],
                rho_high_sq=float(np.mean([v["rho_high"] ** 2 for v in per_batch])),
                rho_low_sq=float(np.mean([v["rho_low"] ** 2 for v in per_batch])),
                holdout_ratio=summary["c_out"] / summary["c_in"] if summary["c_in"] else float("nan"),
                per_batch=per_batch,
            )
            print(
                f"  {scheme:11s} N={n:<6.4g} c_out {summary['c_out']:+.5f} c_in {summary['c_in']:+.5f} "
                f"ratio {summary['holdout_ratio']:.2f} delta {summary['delta']:.5f} rho_high {summary['rho_high']:.3f}",
                flush=True,
            )
            rows[scheme].append(summary)
    for scheme in schemes:
        results["schemes"][scheme] = {
            "rows": rows[scheme],
            "fit": fit([r["N"] for r in rows[scheme]], [r["rho_high_sq"] for r in rows[scheme]]),
        }
    return results


def fit(ns, rho_sq):
    from scipy.optimize import curve_fit

    ns, rho_sq = np.asarray(ns), np.asarray(rho_sq)

    def model(n, a, b):
        return a * n / (n + b)

    try:
        (a, b), _ = curve_fit(model, ns, rho_sq, p0=(rho_sq.max(), 1.0), bounds=([0, 1e-6], [np.inf, 1e6]))
        residual = rho_sq - model(ns, a, b)
        r2 = 1 - np.sum(residual**2) / np.sum((rho_sq - rho_sq.mean()) ** 2)
        return {"a": float(a), "B_star": float(b), "r2": float(r2)}
    except RuntimeError as error:
        return {"error": str(error)}


def main():
    global NUM_BATCHES, NS
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 30, 60, 90])
    parser.add_argument("--out", default=None)
    parser.add_argument("--batches", type=int, default=NUM_BATCHES, help="fresh batches per N")
    parser.add_argument("--ns", type=float, nargs="+", default=list(NS), help="batch sizes, times 1024 envs")
    args = parser.parse_args()
    NUM_BATCHES, NS = args.batches, tuple(args.ns)
    run = Path(args.run)
    out = Path(args.out or run / "batch_probe")
    out.mkdir(parents=True, exist_ok=True)
    saved = run / "checkpoints" / "latest"
    steps = sorted(int(path.name) for path in saved.iterdir() if path.name.isdigit())
    from omegaconf import OmegaConf

    seed = int(OmegaConf.load(run / ".hydra" / "config.yaml").seed)
    for epoch in args.epochs:
        step = steps[min(epoch, len(steps)) - 1]
        checkpoint = str(saved / str(step))
        print(f"checkpoint at epoch {epoch} (step {step})")
        result = probe_checkpoint(checkpoint, seed + epoch)
        result.update(epoch=epoch, step=step)
        (out / f"epoch_{epoch:03d}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
