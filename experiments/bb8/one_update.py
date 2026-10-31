"""E1: the law for one update. From a checkpoint, one update on a fraction of a fresh rollout's environment columns,
over a grid of learning-rate multipliers, and how much it improves J.

Adapted from exp/psr-breakout's hangar/scripts/one_update.py (commit 7fa63581 and later). Differences:
- the update is the algorithm's own `update` (PPO or PQN), with the config's `num_minibatches` scaled with the
  fraction, so the minibatch size stays fixed (one minibatch when the batch is smaller than a minibatch), and the
  optimizer rebuilt with a traced learning rate (Adam as trained, or SGD with momentum / Lion from a zero state);
- ΔJ is measured with common random numbers over `--episodes` episodes from reset, under the training-time policy
  (PPO: stochastic; PQN: ε-greedy at the checkpoint's ε, and greedy reported separately);
- the noise scale is the Phase 1 probe (`algorithm.bb8`) applied to a large batch: `--big` consecutive rollouts laid
  side by side as independent T-step columns (B ≫ B_noise), reported as raw, fresh and full, per loss.

Per checkpoint the output holds, for every condition (fraction, multiplier, update seed): ΔJ per episode (npz), its
mean, KL(π_before ‖ π_after) on the fresh rollout's states (PPO), ‖Δθ‖, and the large-probe noise scales.

    python experiments/bb8/one_update.py <run dir> --steps 7864320 --output out.json [--optimizer adam|sgd|lion]
"""

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.flatten_util import ravel_pytree

from boonta import environments
from boonta.algorithms import bb8
from boonta.algorithms.advantage_estimators import generalized_advantage_estimator
from boonta.algorithms.bb8 import BB8Config
from boonta.utils import remove_feature_axis

MULTIPLIERS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
FRACTIONS = (1 / 128, 1 / 32, 1 / 8, 5 / 16, 1.0)  # 0.78%, 3.1%, 12.5%, 31.25%, 100%


def build(run, extra=()):
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    import hangar
    from hangar import recipes, resolvers  # noqa: F401

    overrides = [o for o in OmegaConf.load(Path(run) / ".hydra" / "overrides.yaml") if not o.startswith("logger=")]
    with initialize_config_dir(config_dir=str(Path(hangar.__file__).parent / "config"), version_base=None):
        cfg = compose("train", overrides=[*overrides, "logger=[]", "artisan=[]", *extra], return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    return cfg, recipes.make(cfg)


def restore(path, template):
    """The checkpoint's algorithm state. Checkpoints written before the `difference` and `trust_region` spaces hold the
    old probe layout (gradient, update, fresh); those are restored with that layout and their probe statistics are
    replaced by fresh zeros (E1 does not use them)."""
    with ocp.StandardCheckpointer() as checkpointer:
        try:
            return checkpointer.restore(path, target=template)
        except ValueError:
            old = jax.tree_util.tree_map_with_path(lambda p, x: x, template.bb8)
            for key in ("noise", "signal"):
                old[key] = {name: {k: v for k, v in spaces.items() if k in ("gradient", "update", "fresh")}
                            for name, spaces in template.bb8[key].items()}
            restored = checkpointer.restore(path, target=template.replace(bb8=old))
    return restored.replace(bb8=template.bb8)


def desynchronize(podracer, state, key, max_noops):
    """Random no-op starts: environment e takes U{0..max_noops} no-op actions (action 0 in MinAtar) after reset, so
    the environments are out of phase before the policy warm-up. Steps are masked per environment."""
    if max_noops <= 0:
        return state
    environment = podracer.environment
    offset_key, step_key = jax.random.split(key)
    num_envs = jax.tree.leaves(state.timestep.reward)[0].shape[0]
    offsets = jax.random.randint(offset_key, (num_envs,), 0, max_noops + 1)

    @jax.jit
    def run(state):
        def body(carry, t):
            environment_state, timestep = carry
            action = jnp.zeros_like(timestep.action)
            stepped_state, stepped = environment.step(jax.random.fold_in(step_key, t), environment_state, action)
            keep = t < offsets
            select = lambda new, old: jnp.where(keep.reshape(-1, *([1] * (new.ndim - 1))), new, old)
            return (jax.tree.map(select, stepped_state, environment_state), jax.tree.map(select, stepped, timestep)), None

        (environment_state, timestep), _ = jax.lax.scan(body, (state.environment_state, state.timestep), jnp.arange(max_noops))
        return state.replace(environment_state=environment_state, timestep=timestep)

    return run(state)


def make_optimizer(kind, learning_rate, max_grad_norm):
    inner = {
        "adam": lambda: optax.adam(learning_rate),
        "sgd": lambda: optax.sgd(learning_rate, momentum=0.9),
        "lion": lambda: optax.lion(learning_rate),
    }[kind]()
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), inner)


def main(args):
    run = Path(args.run)
    cfg, podracer = build(run)
    algorithm = podracer.algorithm
    name = "ppo" if hasattr(algorithm.cfg, "clip_coefficient") else "pqn"
    num_envs, T = int(cfg.environment.num_envs), int(cfg.podracer.config.num_steps)
    minibatch_envs = num_envs // int(algorithm.cfg.num_minibatches)
    base_lr = float(args.base_lr or cfg.optimizer.lr)
    max_grad_norm = float(cfg.optimizer.max_grad_norm)
    game = environments.make(**cfg.environment)
    horizon = game.time_limit() + 1
    print(f"{run} {name}: {num_envs} envs, T {T}, minibatch {minibatch_envs} envs, optimizer {args.optimizer}, "
          f"base lr {base_lr}, horizon {horizon}, device {jax.devices()[0].device_kind}", flush=True)

    def variant(num_minibatches, optimizer, probe=False, num_groups=64):
        return dataclasses.replace(
            algorithm,
            cfg=algorithm.cfg.replace(
                num_minibatches=num_minibatches,
                bb8=BB8Config(noise_scale_probe=probe, num_groups=num_groups, decay=0.9),
            ),
            optimizer=optimizer,
        )

    updates = {}

    def update(state, batch, key, learning_rate):
        columns = batch.second.reward.shape[1]
        num_minibatches = max(1, columns // minibatch_envs)
        if num_minibatches not in updates:
            def fn(state, batch, key, learning_rate):
                optimizer = make_optimizer(args.optimizer, learning_rate, max_grad_norm)
                out, _ = lox.spool(variant(num_minibatches, optimizer).update)(state, key, batch)
                return out
            updates[num_minibatches] = jax.jit(fn)
        return updates[num_minibatches](state, batch, key, jnp.asarray(learning_rate, jnp.float32))

    @jax.jit
    def probe(state, big, learning_rate):
        optimizer = make_optimizer(args.optimizer, learning_rate, max_grad_norm)
        probed = variant(1, optimizer, probe=True, num_groups=64)
        probed = dataclasses.replace(probed, cfg=probed.cfg.replace(update_epochs=1))
        state = state.replace(bb8=jax.tree.map(jnp.zeros_like, _probe_state(probed)))
        _, logs = lox.spool(probed.update)(state, jax.random.key(0), big)
        return {k: v for k, v in logs.items() if k.startswith("bb8/")}

    def _probe_state(probed):
        from boonta.algorithms import bb8
        return bb8.init(("actor", "critic", "total") if name == "ppo" else ("td",))

    rollout = jax.jit(lambda state, keys: jax.lax.scan(lambda s, k: podracer.rollout(s, k, 1.0), state, keys))
    advance = jax.jit(lambda state, keys: jax.lax.scan(lambda s, k: (podracer.rollout(s, k, 1.0)[0], None), state, keys)[0])

    def policy_actions(params, obs, step_key, epsilon, greedy):
        if name == "ppo":
            dist, _ = algorithm.network.apply(params, obs, temperature=1.0)
            gumbel = jax.random.gumbel(step_key, dist.logits.shape)
            return jnp.argmax(dist.logits + gumbel, axis=-1)
        preferences = algorithm.network.apply(params, obs, temperature=1.0).preferences
        best = jnp.argmax(preferences, axis=-1)
        explore_key, action_key = jax.random.split(step_key)
        random_action = jax.random.randint(action_key, best.shape, 0, preferences.shape[-1])
        explore = (jax.random.uniform(explore_key, best.shape) < epsilon) & ~greedy
        return jnp.where(explore, random_action, best)

    @jax.jit
    def episode_returns(params, key, epsilon, greedy):
        states, timesteps = jax.vmap(game.init)(jax.random.split(jax.random.fold_in(key, 0), args.episodes))

        def body(carry, t):
            states, timesteps, alive, total = carry
            actions = policy_actions(params, timesteps.obs, jax.random.fold_in(jax.random.fold_in(key, 1), t), epsilon, greedy)
            states, timesteps = jax.vmap(game.step)(jax.random.split(jax.random.fold_in(jax.random.fold_in(key, 2), t), args.episodes), states, actions)
            total = total + jnp.where(alive, timesteps.reward, 0.0)
            return (states, timesteps, alive & ~timesteps.done, total), None

        alive = jnp.ones(args.episodes, bool)
        (_, _, alive, total), _ = jax.lax.scan(body, (states, timesteps, alive, jnp.zeros(args.episodes)), jnp.arange(horizon))
        return total, alive

    @jax.jit
    def gae(params, transitions):
        last = jax.tree.map(lambda leaf: leaf[-1], transitions.second.obs)
        _, bootstrap = algorithm.network.apply(params, last, temperature=1.0)
        return generalized_advantage_estimator(transitions, transitions.aux["value"], remove_feature_axis(bootstrap),
                                               algorithm.cfg.gamma, algorithm.cfg.gae_lambda)

    @jax.jit
    def surrogate(after, heldout):
        """Held-out surrogate improvement: E[(π'(a|s) / π(a|s) − 1) A] per block of 128 columns, per rollout."""
        def one(part):
            obs, action, old_log_prob, advantage = part
            flat = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), obs)
            dist, _ = algorithm.network.apply(after, flat, temperature=1.0)
            log_prob = dist.log_prob(action.reshape(-1)).reshape(advantage.shape)
            gain = (jnp.exp(log_prob - old_log_prob) - 1.0) * advantage  # (T, N)
            return gain.mean(axis=0).reshape(-1, 128).mean(axis=1)
        return jax.lax.map(one, heldout)

    @jax.jit
    def mean_kl(before, after, obs):
        old, _ = algorithm.network.apply(before, obs, temperature=1.0)
        new, _ = algorithm.network.apply(after, obs, temperature=1.0)
        old_log, new_log = jax.nn.log_softmax(old.logits, -1), jax.nn.log_softmax(new.logits, -1)
        return jnp.mean(jnp.sum(jnp.exp(old_log) * (old_log - new_log), -1))

    report = {"run": str(run), "algorithm": name, "optimizer": args.optimizer, "base_lr": base_lr, "num_envs": num_envs,
              "T": T, "minibatch_envs": minibatch_envs, "episodes": args.episodes, "device": jax.devices()[0].device_kind,
              "checkpoints": {}}
    arrays = {}
    for step in args.steps:
        started = time.time()
        key = jax.random.key(1_000_003 * step % (2**31) + args.key_offset)
        init_key, warm_key, batch_key, big_key, eval_key = jax.random.split(key, 5)
        template = podracer.init(init_key)
        path = os.path.abspath(run / "checkpoints" / "latest" / str(step) / "algorithm_state")
        algorithm_state = restore(path, template.algorithm_state)
        if args.optimizer != "adam":
            algorithm_state = algorithm_state.replace(
                optimizer_state=make_optimizer(args.optimizer, base_lr, max_grad_norm).init(algorithm_state.params["params"]))
        params = algorithm_state.params
        origin, _ = ravel_pytree(params["params"])
        state = template.replace(algorithm_state=algorithm_state)
        state = desynchronize(podracer, state, jax.random.fold_in(warm_key, 1), args.noops)
        state = advance(state, jax.random.split(warm_key, args.warmup))
        state, batch = rollout(state, jax.random.split(batch_key, T))
        bigs = [batch]
        for rollout_key in jax.random.split(big_key, args.big - 1):
            state, more = rollout(state, jax.random.split(rollout_key, T))
            bigs.append(more)
        # Env-major: column e * big + k is environment e in rollout k, so a probe group (contiguous columns) holds
        # whole environments across all rollouts and groups stay independent.
        big = bb8.stack_rollouts(bigs)
        del bigs, more
        noise = {k: float(np.mean(v)) for k, v in probe(algorithm_state, big, base_lr).items()}
        single = {k: float(np.mean(v)) for k, v in probe(algorithm_state, batch, base_lr).items()}
        del big
        heldout = None
        if name == "ppo" and args.heldout > 0:
            # Held-out batch for the surrogate proxy: later rollouts of the pre-update policy, independent of the
            # update's batch. Advantages from the pre-update critic, unnormalized.
            parts = []
            for rollout_key in jax.random.split(jax.random.fold_in(key, 7), args.heldout):
                state, more = rollout(state, jax.random.split(rollout_key, T))
                advantages, _ = gae(params, more)
                parts.append((more.first.obs, more.second.action, more.aux["log_prob"], advantages))
            heldout = jax.tree.map(lambda *leaves: jnp.stack(leaves), *parts)
            del parts, more
        observations = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), batch.first.obs)
        epsilon = float(algorithm.exploration_schedule(algorithm_state.step)) if name == "pqn" else 0.0
        before, before_alive = episode_returns(params, eval_key, epsilon, False)
        entry = {"epsilon": epsilon, "j_before": float(before.mean()), "truncated_before": float(before_alive.mean()),
                 "noise_big": noise, "noise_single": single, "big_columns": args.big * num_envs, "conditions": []}
        if name == "pqn":
            greedy_before, _ = episode_returns(params, eval_key, epsilon, True)
            entry["j_before_greedy"] = float(greedy_before.mean())
        print(f"step {step}: J {entry['j_before']:.3f}, eps {epsilon:.3f}; large-probe fresh noise scale (envs): " + ", ".join(
            f"{k.split('/')[1]} {v:.0f}" for k, v in noise.items() if k.endswith("/fresh/noise_scale_envs")), flush=True)

        if args.check:
            plain = dataclasses.replace(algorithm, cfg=algorithm.cfg.replace(bb8=BB8Config()))
            reference = jax.jit(lambda s, k, b: lox.spool(plain.update)(s, k, b)[0])(algorithm_state, jax.random.key(7), batch)
            mine = update(algorithm_state, batch, jax.random.key(7), base_lr)
            a, _ = ravel_pytree(reference.params["params"])
            b, _ = ravel_pytree(mine.params["params"])
            entry["check"] = {"bitwise": bool(jnp.array_equal(a, b)), "max_abs_diff": float(jnp.max(jnp.abs(a - b))),
                              "update_norm": float(jnp.linalg.norm(a - origin))}
            print("check against the algorithm's update:", entry["check"], flush=True)

        def measure(test, chosen, learning_rate, seed, extra):
            after = update(algorithm_state, jax.tree.map(lambda leaf: jnp.take(leaf, chosen, axis=1), batch),
                           jax.random.key(seed), learning_rate)
            moved, _ = ravel_pytree(after.params["params"])
            result = {"test": test, "seed": seed, "columns": int(len(chosen)), "learning_rate": float(learning_rate),
                      **extra, "update_norm": float(jnp.linalg.norm(moved - origin))}
            if name == "ppo":
                result["kl"] = float(mean_kl(params, after.params, observations))
            if heldout is not None:
                blocks = np.asarray(surrogate(after.params, heldout), np.float32).reshape(-1)
                result["proxy"] = float(blocks.mean())
                arrays[f"{step}/{len(entry['conditions'])}/proxy"] = blocks
            if not extra.get("kl_only"):
                returns, alive = episode_returns(after.params, eval_key, epsilon, False)
                difference = np.asarray(returns - before, np.float32)
                result["delta_j"] = float(difference.mean())
                result["truncated"] = float(alive.mean())
                index = len(entry["conditions"])
                arrays[f"{step}/{index}"] = difference
                if name == "pqn":
                    greedy, _ = episode_returns(after.params, eval_key, epsilon, True)
                    gdiff = np.asarray(greedy - greedy_before, np.float32)
                    result["delta_j_greedy"] = float(gdiff.mean())
                    arrays[f"{step}/{index}/greedy"] = gdiff
            entry["conditions"].append(result)
            return result

        def subset(fraction, seed):
            count = max(1, int(round(fraction * num_envs)))
            rng = np.random.default_rng([step, seed, count])
            return jnp.asarray(np.sort(rng.choice(num_envs, count, replace=False)))

        if "grid" in args.tests:
            for seed in range(args.seeds):
                for fraction in FRACTIONS:
                    chosen = subset(fraction, seed)
                    for multiplier in args.multipliers:
                        measure("grid", chosen, base_lr * multiplier, seed, {"fraction": fraction, "multiplier": multiplier})
                print(f"  grid seed {seed}: {time.time() - started:.0f} s", flush=True)
        if "tune" in args.tests:
            for seed in range(args.seeds):
                for multiplier in args.tune_multipliers:
                    measure("tune", jnp.arange(num_envs), base_lr * multiplier, seed, {"fraction": 1.0, "multiplier": multiplier})
        if "matched_kl" in args.tests and name == "ppo":
            full = [c["kl"] for c in entry["conditions"] if c["test"] == "grid" and c["fraction"] == 1.0 and c["multiplier"] == 1.0]
            target = float(np.median(full)) if full else None
            entry["matched_kl_target"] = target
            for seed in range(args.seeds):
                for fraction in FRACTIONS:
                    chosen = subset(fraction, seed)
                    low, high = np.log(1 / 256), np.log(16.0)
                    for _ in range(args.bisections):
                        middle = 0.5 * (low + high)
                        kl = measure("matched_kl_search", chosen, base_lr * np.exp(middle), seed,
                                     {"fraction": fraction, "kl_only": True})["kl"]
                        low, high = (middle, high) if kl < target else (low, middle)
                    multiplier = float(np.exp(0.5 * (low + high)))
                    measure("matched_kl", chosen, base_lr * multiplier, seed, {"fraction": fraction, "multiplier": multiplier})
            print(f"  matched KL: {time.time() - started:.0f} s", flush=True)
        entry["seconds"] = time.time() - started
        report["checkpoints"][str(step)] = entry
        Path(args.output).write_text(json.dumps(report))
        np.savez_compressed(Path(args.output).with_suffix(".npz"), **arrays)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tests", nargs="+", default=["grid", "matched_kl"])
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "lion"])
    parser.add_argument("--base-lr", type=float, default=None, help="default: the run's optimizer.lr")
    parser.add_argument("--multipliers", type=float, nargs="+", default=list(MULTIPLIERS),
                        help="grid multipliers (an extension run with other values is merged by analyze_e1.py)")
    parser.add_argument("--tune-multipliers", type=float, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, default=4, help="update seeds (minibatch shuffles and column subsets)")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--noops", type=int, default=0, help="random no-op starts, at most this many per environment")
    parser.add_argument("--big", type=int, default=8, help="rollouts in the large probe batch")
    parser.add_argument("--heldout", type=int, default=16, help="rollouts in the held-out batch of the surrogate proxy (PPO)")
    parser.add_argument("--bisections", type=int, default=8)
    parser.add_argument("--key-offset", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args())
