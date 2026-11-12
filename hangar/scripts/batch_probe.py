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
LOOK_STATES = 4096  # evaluation states that get a simulator-based S
LOOK_KEYS = 4  # lookahead rollouts per (state, action) in each of the two independent estimates
LOOK_HORIZON = 64  # steps of each lookahead, the first with the action under test, then pi
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


def vectorized(environment, env_state):
    """The Vectorize wrapper under the environment's other wrappers, and its batched inner state."""
    from boonta.environments.wrappers import WrapperState
    from boonta.environments.wrappers.vectorize import Vectorize

    env, state = environment, env_state
    while not isinstance(env, Vectorize):
        env = env._env
        state = state.env_state if isinstance(state, WrapperState) else state
    return env, state


def collect(algorithm, environment, params_state, key, num_rollouts: int, gamma: float, lam: float, keep: int = 0):
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
        inner = vectorized(environment, env_state)[1] if keep else None
        next_state, next_timestep = environment.step(env_key, env_state, action)
        # Transitions hold the exact uint8 planes, a quarter of the float observations' memory.
        transition = Transition(
            first=planes, second=next_timestep.replace(obs=encode(next_timestep.obs)), aux=aux, carry=None
        )
        return (next_state, next_timestep), (transition, cells, inner)

    @jax.jit
    def burn(env_state, timestep, key):
        (env_state, timestep), _ = jax.lax.scan(step, (env_state, timestep), jax.random.split(key, BURN_IN))
        return env_state, timestep

    num_envs = environment.num_envs

    @jax.jit
    def rollout(env_state, timestep, key):
        scan_key, keep_key = jax.random.split(key)
        (env_state, timestep), (transitions, cells, inner) = jax.lax.scan(
            step, (env_state, timestep), jax.random.split(scan_key, NUM_STEPS)
        )
        kept = None
        if keep:
            # A random subset of this rollout's states (step, environment), kept whole for lookahead.
            picked = jax.random.choice(keep_key, NUM_STEPS * num_envs, (keep,), replace=False)
            kept = (
                jax.tree.map(lambda leaf: jnp.reshape(leaf, (-1, *leaf.shape[2:]))[picked], inner),
                jnp.reshape(transitions.first.obs, (-1, *transitions.first.obs.shape[2:]))[picked],
            )
        _, last_value = algorithm.network.apply(params_state.params, transitions.second.obs[-1], temperature=1.0)
        advantages, _ = generalized_advantage_estimator(
            transitions, transitions.aux["value"], last_value[..., 0], gamma, lam
        )
        return env_state, timestep, transitions, cells, advantages, kept

    env_state, timestep = burn(env_state, timestep, burn_key)
    out = {"obs": [], "action": [], "log_prob": [], "advantage": [], "cell": [], "valid": []}
    last, kept_states, kept_obs = None, [], []
    for rollout_key in jax.random.split(key, num_rollouts):
        env_state, timestep, transitions, cells, advantages, kept = rollout(env_state, timestep, rollout_key)
        if keep:
            kept_states.append(kept[0])
            kept_obs.append(kept[1])
        flat = lambda leaf: jnp.reshape(leaf, (-1, *leaf.shape[2:]))  # noqa: E731
        out["obs"].append(flat(transitions.first.obs))
        out["action"].append(flat(transitions.second.action).astype(jnp.int32))
        out["log_prob"].append(flat(transitions.aux["log_prob"]))
        out["advantage"].append(flat(advantages))
        out["cell"].append(flat(cells))
        out["valid"].append(~flat(transitions.second.truncated).astype(bool))
        last = transitions
    data = {name: jnp.concatenate(parts) for name, parts in out.items()}
    if keep:
        data["look_states"] = jax.tree.map(lambda *leaves: jnp.concatenate(leaves), *kept_states)
        data["look_obs"] = jnp.concatenate(kept_obs)
    return data, last


def lookahead_signal(algorithm, environment, params_state, data, key, gamma: float, chunk: int = 512) -> dict:
    """S_look = mean_s sum_a pi(a|s) A1(s, a) A2(s, a), where A1 and A2 are two independent
    lookahead estimates of the advantage of every action. Each estimate of Q(s, a) averages
    LOOK_KEYS rollouts that take a, follow pi for LOOK_HORIZON - 1 more steps with fresh ghost
    randomness, and bootstrap with the critic; A = Q - sum_b pi(b|s) Q(s, b) within each estimate.
    The two estimates share nothing but the state, so their product is unbiased for Var_pi(A)."""
    from boonta.environments.jumanji import reseed

    vec, _ = vectorized(environment, environment.init(jax.random.key(0))[0])
    inner = vec._env
    num_actions = 5

    @jax.jit
    def q_values(states, key):
        size = jax.tree.leaves(states)[0].shape[0]
        width = size * num_actions * LOOK_KEYS
        tiled = jax.tree.map(lambda leaf: jnp.repeat(leaf, num_actions * LOOK_KEYS, axis=0), states)
        first_action = jnp.tile(jnp.repeat(jnp.arange(num_actions), LOOK_KEYS), size)
        seed_key, key = jax.random.split(key)
        tiled = reseed(tiled, seed_key)

        def step(carry, key):
            states, alive, total, discount, forced = carry
            act_key, env_key = jax.random.split(key)
            obs = encode(jax.vmap(inner.observe)(states))
            dist, _ = algorithm.network.apply(params_state.params, obs, temperature=1.0)
            action = jnp.where(forced, first_action, dist.sample(seed=act_key))
            states, timestep = jax.vmap(inner.step)(jax.random.split(env_key, width), states, action)
            total = total + alive * discount * timestep.reward.astype(jnp.float32)
            alive = alive * (1.0 - timestep.terminated.astype(jnp.float32))
            return (states, alive, total, discount * gamma, jnp.zeros_like(forced)), None

        start = (tiled, jnp.ones(width), jnp.zeros(width), jnp.ones(()), jnp.ones(width, bool))
        (states, alive, total, discount, _), _ = jax.lax.scan(step, start, jax.random.split(key, LOOK_HORIZON))
        _, value = algorithm.network.apply(params_state.params, encode(jax.vmap(inner.observe)(states)), temperature=1.0)
        q = total + alive * discount * value[..., 0]
        return jnp.reshape(q, (size, num_actions, LOOK_KEYS)).mean(-1)

    states, obs = data["look_states"], data["look_obs"]
    count = obs.shape[0]
    probs = jnp.exp(log_probs(algorithm, params_state.params, obs))
    estimates = []
    for replicate_key in jax.random.split(key, 2):
        parts = []
        for index, start in enumerate(range(0, count, chunk)):
            piece = jax.tree.map(lambda leaf: leaf[start : start + chunk], states)
            parts.append(q_values(piece, jax.random.fold_in(replicate_key, index)))
        q = jnp.concatenate(parts)
        estimates.append(q - jnp.sum(probs * q, axis=-1, keepdims=True))
    first, second = estimates
    product = jnp.sum(probs * first * second, axis=-1)
    single = jnp.sum(probs * first**2, axis=-1)
    return {
        "s_look": float(jnp.mean(product)),
        "s_look_se": float(jnp.std(product) / np.sqrt(count)),
        "s_look_single": float(jnp.mean(single)),
        "look_states": int(count),
    }


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
    evaluation, _ = collect(
        algorithm, environment, params_state, eval_key, NUM_EVAL_ROLLOUTS, gamma, lam, keep=LOOK_STATES // NUM_EVAL_ROLLOUTS
    )
    look_key, key = jax.random.split(key)
    look = lookahead_signal(algorithm, environment, params_state, evaluation, look_key, gamma)
    s_look = look["s_look"]
    for name in ("look_states", "look_obs"):
        evaluation.pop(name)
    old_eval = log_probs(algorithm, params_state.params, evaluation["obs"])
    valid = evaluation["valid"]
    s_raw = float(jnp.sum(jnp.where(valid, evaluation["advantage"] ** 2, 0.0)) / jnp.sum(valid))
    s_cell = cell_signal(evaluation)
    results.update(s_raw=s_raw, s_cell=s_cell, eval_samples=int(valid.sum()), **look)
    print(
        f"  evaluation set: {int(valid.sum())} samples, S_raw {s_raw:.3f}, S_cell {s_cell:.4f}, "
        f"S_look {s_look:.4f} +- {look['s_look_se']:.4f} (single-estimate {look['s_look_single']:.4f})",
        flush=True,
    )

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
                        "rho_look": c_out / np.sqrt(max(2 * delta * s_look, 1e-30)) if s_look > 0 else float("nan"),
                        "c_out_per_sample": c_out / batch,
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
                rho_look_sq=float(np.mean([v["rho_look"] ** 2 for v in per_batch])),
                holdout_ratio=summary["c_out"] / summary["c_in"] if summary["c_in"] else float("nan"),
                per_batch=per_batch,
            )
            print(
                f"  {scheme:11s} N={n:<6.4g} c_out {summary['c_out']:+.5f} c_in {summary['c_in']:+.5f} "
                f"ratio {summary['holdout_ratio']:.2f} delta {summary['delta']:.5f} rho_high {summary['rho_high']:.3f} "
                f"rho_look {summary['rho_look']:.3f} c_out/sample {summary['c_out_per_sample']:.2e}",
                flush=True,
            )
            rows[scheme].append(summary)
    for scheme in schemes:
        results["schemes"][scheme] = {
            "rows": rows[scheme],
            "fit": fit([r["N"] for r in rows[scheme]], [r["rho_high_sq"] for r in rows[scheme]]),
            "fit_look": fit([r["N"] for r in rows[scheme]], [r["rho_look_sq"] for r in rows[scheme]]),
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
