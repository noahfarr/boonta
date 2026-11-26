import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from boonta import environments
from boonta.algorithms.eggroll import EGGROLL, EGGROLLConfig
from boonta.environments.wrappers import Group, SameStepAutoReset
from boonta.networks.low_rank_perturbations import LowRankPerturbedMLP
from boonta.podracers import anakin
from boonta.utils import Transition, load_checkpoint, mesh


MODES = ("none", "population", "pair")


def emit(**record):
    print(json.dumps(record), flush=True)


def make_evaluator(env, network):
    @jax.jit
    def evaluate(params, key):
        key, reset_key = jax.random.split(key)
        state, ts = env.init(reset_key)
        alive = jnp.ones_like(ts.done)

        def step(carry):
            key, state, ts, alive, returns, length = carry
            key, step_key = jax.random.split(key)
            actions = network.apply({"params": params}, ts.obs).argmax(axis=-1)
            state, ts = env.step(step_key, state, actions)
            return key, state, ts, alive & ~ts.done, returns + alive * ts.reward, length + 1

        _, _, _, alive, returns, _ = jax.lax.while_loop(
            lambda c: c[3].any() & (c[5] < 8192), step,
            (key, state, ts, alive, jnp.zeros_like(ts.reward), 0),
        )
        return returns.mean(), alive.any()

    return evaluate


def train(args, env, algorithm):
    evaluate = make_evaluator(env, algorithm.network)
    runner = anakin.make(anakin.AnakinConfig(args.horizon, mesh(1)), algorithm, Group(env, num_steps=args.horizon))
    for seed in args.seeds:
        init_key, train_key, _ = jax.random.split(jax.random.key(seed), 3)
        state = runner.init(init_key)
        for epoch, key in enumerate(jax.random.split(train_key, 10), start=1):
            state, _ = runner.train(state, key, args.steps // (10 * runner.batch_size))
            if epoch in (5, 10):
                score, capped = evaluate(state.algorithm_state.params, jax.random.key(92711))
                assert not bool(capped)
                emit(mode=algorithm.cfg.input_support, game=args.game, seed=seed,
                     step=int(state.algorithm_state.step), episode_return=float(score))
        runner.close(state)


def frozen(args, env, algorithm):
    _, initial_ts = env.init(jax.random.key(0))
    initial, _ = algorithm.init(jax.random.key(1), initial_ts)
    algorithms = [replace(algorithm, cfg=algorithm.cfg.replace(input_support=m)) for m in MODES]

    @jax.jit
    def evaluation_observations(params, key):
        reset_key, rollout_key = jax.random.split(key)
        state, ts = env.init(reset_key)

        def step(carry, key):
            state, ts = carry
            actions = algorithm.network.apply({"params": params}, ts.obs).argmax(axis=-1)
            state, after = env.step(key, state, actions)
            return (state, after), ts.obs[:16]

        _, observations = jax.lax.scan(step, (state, ts), jax.random.split(rollout_key, args.horizon))
        return observations.reshape(-1, *observations.shape[2:])

    @jax.jit
    def sample(params, key, observations):
        reset_key, noise_key, rollout_key = jax.random.split(key, 3)
        state, ts = env.init(reset_key)
        noise = algorithm.draw(params, noise_key)

        def step(carry, key):
            state, ts = carry
            action = algorithm.network.apply({"params": params}, ts.obs, noise, sigma=algorithm.cfg.sigma).argmax(-1)
            state, after = env.step(key, state, action)
            return (state, after), Transition(first=ts, second=after)

        _, transitions = jax.lax.scan(step, (state, ts), jax.random.split(rollout_key, args.horizon))
        def action_scores(weights):
            logits = algorithm.network.apply({"params": weights}, observations)
            return logits - logits.mean(axis=-1, keepdims=True)

        updates = []
        score_updates = []
        for method in algorithms:
            updated, _ = lox.spool(method.update)(initial.replace(params=params, noise_key=noise_key), noise_key, transitions)
            delta = jax.tree.map(jnp.subtract, updated.params, params)
            updates.append(ravel_pytree(delta)[0])
            _, directional_change = jax.jvp(action_scores, (params,), (delta,))
            score_updates.append(directional_change.ravel())
        return jnp.stack(updates), jnp.stack(score_updates)

    for step in args.checkpoints:
        path = args.run / "checkpoints/latest" / str(step) / "algorithm_state"
        params = load_checkpoint(str(path))["params"]
        keys = jax.random.split(jax.random.key(771), args.batches)
        observations = evaluation_observations(params, jax.random.key(92712))
        rows = [jax.device_get(sample(params, k, observations)) for k in keys]

        def trace(values):
            centered = np.stack(values).astype(np.float64)
            centered -= centered.mean(axis=0)
            return np.einsum("bmp,bmp->bm", centered, centered).mean(axis=0)

        variance, score_variance = (trace(values) for values in zip(*rows, strict=True))
        for mode, value, score in zip(MODES, variance, score_variance, strict=True):
            emit(game=args.game, step=step, mode=mode, batches=args.batches,
                 logit_variance_reduction_pct=float(100 * (1 - score / score_variance[0])),
                 whole_variance_reduction_pct=float(100 * (1 - value / variance[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game")
    parser.add_argument("--horizon", type=int, default=64)
    modes = parser.add_subparsers(dest="experiment", required=True)
    training = modes.add_parser("train")
    training.add_argument("mode", choices=MODES)
    training.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    training.add_argument("--steps", type=int, default=5_000_000)
    probe = modes.add_parser("frozen")
    probe.add_argument("run", type=Path)
    probe.add_argument("checkpoints", nargs="+", type=int)
    probe.add_argument("--batches", type=int, default=256)
    args = parser.parse_args()
    env = SameStepAutoReset(environments.make("gymnax", args.game), 256)
    space = env.action_space()
    algorithm = EGGROLL(EGGROLLConfig(input_support=getattr(args, "mode", "none")),
                        LowRankPerturbedMLP((64, 64, space.num_actions)), space,
                        optax.sgd(1) if args.experiment == "frozen" else optax.adam(0.01))
    if args.experiment == "frozen":
        frozen(args, env, algorithm)
    else:
        train(args, env, algorithm)


if __name__ == "__main__":
    main()
