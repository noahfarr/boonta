import argparse
import json
import os
import pathlib
import time
from contextlib import contextmanager

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import lox
from jax import export

from boonta.environments.wrappers import SameStepAutoReset
from boonta.podracers.sebulba import ActorState
from boonta.utils import Timestep, Transition, take

import pilots
from menu import COMBOS, DATASETS, PILOTS, PODRACERS, TRACKS

jax.config.update("jax_traceback_in_locations_limit", 0)

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "build"
DATA = HERE / "data"
KEY = jax.ShapeDtypeStruct((2,), jnp.uint32)
EPISODE_RETURN = "episode_statistics/episode_return"


@contextmanager
def unconstrained():
    original = jax.lax.with_sharding_constraint
    jax.lax.with_sharding_constraint = lambda x, shardings: x
    try:
        yield
    finally:
        jax.lax.with_sharding_constraint = original


def spec(leaf):
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)


def describe(leaves):
    return [[list(leaf.shape), str(leaf.dtype)] for leaf in leaves]


def wrap(key):
    return jax.random.wrap_key_data(key)


def emit(directory, name, function, specs):
    started = time.time()
    exported = export.export(jax.jit(function))(*specs)
    text = exported.mlir_module()
    (directory / f"{name}.mlir").write_text(text)
    print(f"  {name}: {len(text) / 1e3:.0f} kB in {time.time() - started:.1f}s", flush=True)
    return {
        "file": f"{name}.mlir",
        "kept": list(exported.module_kept_var_idx),
        "num_inputs": len(specs),
        "outputs": describe(exported.out_avals),
    }


def no_typed_keys(tree):
    typed = [
        leaf
        for leaf in jax.tree.leaves(tree)
        if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
    ]
    assert not typed, "state holds typed keys"


def positions(treedef, select):
    indices = jax.tree.unflatten(treedef, list(range(treedef.num_leaves)))
    return [int(index) for index in jax.tree.leaves(select(indices))]


def episodes(logs, num_updates):
    returns = jnp.asarray(logs.get(EPISODE_RETURN, jnp.full((num_updates,), jnp.nan)))
    returns = returns.reshape(num_updates, -1).astype(jnp.float32)
    ended = ~jnp.isnan(returns)
    count = jnp.sum(ended, axis=1).astype(jnp.float32)
    total = jnp.sum(jnp.where(ended, returns, 0.0), axis=1)
    return jnp.stack([total, count], axis=1)


def summary(logs):
    returns = jnp.asarray(logs.get(EPISODE_RETURN, jnp.full((1,), jnp.nan)))
    returns = returns.reshape(-1).astype(jnp.float32)
    ended = ~jnp.isnan(returns)
    return jnp.stack(
        [jnp.sum(jnp.where(ended, returns, 0.0)), jnp.sum(ended).astype(jnp.float32)]
    )


def watch(directory, combo, algorithm):
    track = combo["track"]
    environment = SameStepAutoReset(pilots.game(track), num_envs=1)
    network = pilots.policy(combo["pilot"], algorithm)

    def initialize(key):
        environment_state, timestep = environment.init(wrap(key))
        return jax.tree.map(lambda leaf: leaf[0], (environment_state, timestep))

    environment_shape, timestep_shape = jax.eval_shape(initialize, KEY)
    carry_leaves, carry_def = jax.tree.flatten((environment_shape, timestep_shape))
    obs = timestep_shape.obs
    params_shape = jax.eval_shape(
        lambda: network.init(
            jax.random.key(0), jnp.zeros((1, *obs.shape), obs.dtype), temperature=1.0
        )
    )
    params_leaves, params_def = jax.tree.flatten(params_shape)

    def watch_init(key):
        return tuple(jax.tree.leaves(initialize(key)))

    def watch_step(*args):
        params = jax.tree.unflatten(params_def, args[: len(params_leaves)])
        environment_state, timestep = jax.tree.map(
            lambda leaf: leaf[None],
            jax.tree.unflatten(carry_def, args[len(params_leaves) : -1]),
        )
        action_key, step_key = jax.random.split(wrap(args[-1]))
        dist = network.apply(params, timestep.obs, temperature=0.0)
        if isinstance(dist, tuple):
            dist, _ = dist
        action = dist.sample(seed=action_key)
        environment_state, timestep = jax.tree.map(
            lambda leaf: leaf[0], environment.step(step_key, environment_state, action)
        )
        return (
            *jax.tree.leaves((environment_state, timestep)),
            jnp.asarray(timestep.obs, jnp.float32),
            jnp.asarray(timestep.reward, jnp.float32),
            jnp.asarray(timestep.done, jnp.int32),
        )

    programs = {}
    frame_shape = timestep_shape.obs
    if track in pilots.PICTURES:
        unwrap = lambda carry: getattr(carry[0], "unwrapped", carry[0])
        game_leaves, game_def = jax.tree.flatten(unwrap((environment_shape, timestep_shape)))

        def watch_render(*leaves):
            picture = pilots.picture(track, jax.tree.unflatten(game_def, leaves))
            return jnp.asarray(picture, jnp.float32)

        frame_shape = jax.eval_shape(watch_render, *game_leaves)
        programs["render"] = {
            **emit(directory, "watch_render", watch_render, list(map(spec, game_leaves))),
            "inputs": positions(carry_def, unwrap),
        }

    return {
        **programs,
        "init": emit(directory, "watch_init", watch_init, [KEY]),
        "step": emit(
            directory,
            "watch_step",
            watch_step,
            [*map(spec, params_leaves), *map(spec, carry_leaves), KEY],
        ),
        "num_carry": len(carry_leaves),
        "num_params": len(params_leaves),
        "obs_shape": list(frame_shape.shape),
    }


def build_anakin(directory, combo):
    h, updates = combo["h"], combo["updates_per_call"]
    racer = pilots.anakin(combo["pilot"], combo["track"], h)
    state_shape = jax.eval_shape(racer.init, jax.random.key(0))
    no_typed_keys(state_shape)
    leaves, treedef = jax.tree.flatten(state_shape)
    train = lox.spool(lox.strip(racer.train, tags=["evaluation"]))

    def init(key):
        return tuple(jax.tree.leaves(racer.init(wrap(key))))

    def train_call(*args):
        state = jax.tree.unflatten(treedef, args[:-1])
        state, logs = train(state, wrap(args[-1]), updates)
        return (*jax.tree.leaves(state), episodes(logs, updates))

    manifest = {
        "num_state": len(leaves),
        "state": describe(leaves),
        "params": positions(treedef, lambda state: state.algorithm_state.params),
        "init": emit(directory, "init", init, [KEY]),
        "train": emit(directory, "train", train_call, [*map(spec, leaves), KEY]),
        "batch_size": racer.batch_size,
    }
    return manifest, racer.algorithm


def build_sebulba(directory, combo):
    h = combo["h"]
    racer = pilots.sebulba(combo["pilot"], combo["track"], h)
    unroll = lox.spool(lox.strip(racer.unroll, tags=["evaluation"]))
    update = lox.spool(racer.update)

    def initialize(key):
        environment_key, algorithm_key = jax.random.split(key)
        environment_state, timestep = racer.environment.init(
            jax.random.fold_in(environment_key, 0)
        )
        algorithm_state = racer.algorithm.init(algorithm_key, timestep)
        carry = getattr(algorithm_state, "carry", None)
        actor_state = ActorState(
            timestep=timestep,
            environment_state=environment_state,
            carry=take(carry, 0, 1),
        )
        rollout_state = racer.pit(racer.lap(actor_state.board(algorithm_state)))
        num_laps = rollout_state.algorithm_state.step - algorithm_state.step
        actor_state = ActorState(
            timestep=rollout_state.timestep,
            environment_state=rollout_state.environment_state,
            carry=getattr(rollout_state.algorithm_state, "carry", None),
        )
        algorithm_state = algorithm_state.replace(step=algorithm_state.step + num_laps)
        return algorithm_state, actor_state

    algorithm_shape, actor_shape = jax.eval_shape(initialize, jax.random.key(0))
    no_typed_keys((algorithm_shape, actor_shape))
    algorithm_leaves, algorithm_def = jax.tree.flatten(algorithm_shape)
    actor_leaves, actor_def = jax.tree.flatten(actor_shape)
    num_algorithm, num_actor = len(algorithm_leaves), len(actor_leaves)

    def init(key):
        algorithm_state, actor_state = initialize(wrap(key))
        return (*jax.tree.leaves(algorithm_state), *jax.tree.leaves(actor_state))

    def rollout_body(algorithm_state, actor_state, key):
        rollout_state = actor_state.board(algorithm_state)
        before = rollout_state.algorithm_state.step
        (rollout_state, transitions), logs = unroll(rollout_state, key)
        num_laps = rollout_state.algorithm_state.step - before
        actor_state = ActorState(
            timestep=rollout_state.timestep,
            environment_state=rollout_state.environment_state,
            carry=getattr(rollout_state.algorithm_state, "carry", None),
        )
        return actor_state, transitions, num_laps, logs

    transitions_shape = jax.eval_shape(
        lambda algorithm_state, actor_state: rollout_body(
            algorithm_state, actor_state, jax.random.key(0)
        )[1],
        algorithm_shape,
        actor_shape,
    )
    transition_leaves, transition_def = jax.tree.flatten(transitions_shape)
    num_transitions = len(transition_leaves)

    def rollout(*args):
        algorithm_state = jax.tree.unflatten(algorithm_def, args[:num_algorithm])
        actor_state = jax.tree.unflatten(
            actor_def, args[num_algorithm : num_algorithm + num_actor]
        )
        actor_state, transitions, num_laps, logs = rollout_body(
            algorithm_state, actor_state, wrap(args[-1])
        )
        return (
            *jax.tree.leaves(actor_state),
            *jax.tree.leaves(transitions),
            jnp.asarray(num_laps, jnp.int32).reshape(1),
            summary(logs),
        )

    def update_call(*args):
        algorithm_state = jax.tree.unflatten(algorithm_def, args[:num_algorithm])
        transitions = jax.tree.unflatten(
            transition_def, args[num_algorithm : num_algorithm + num_transitions]
        )
        num_laps, key = args[-2], args[-1]
        algorithm_state = algorithm_state.replace(
            step=algorithm_state.step + num_laps[0].astype(algorithm_state.step.dtype)
        )
        algorithm_state, _ = update(algorithm_state, wrap(key), transitions)
        return tuple(jax.tree.leaves(algorithm_state))

    laps = jax.ShapeDtypeStruct((1,), jnp.int32)
    manifest = {
        "num_algorithm": num_algorithm,
        "num_actor": num_actor,
        "num_transitions": num_transitions,
        "algorithm": describe(algorithm_leaves),
        "params": positions(algorithm_def, lambda state: state.params),
        "init": emit(directory, "init", init, [KEY]),
        "rollout": emit(
            directory,
            "rollout",
            rollout,
            [*map(spec, algorithm_leaves), *map(spec, actor_leaves), KEY],
        ),
        "update": emit(
            directory,
            "update",
            update_call,
            [*map(spec, algorithm_leaves), *map(spec, transition_leaves), laps, KEY],
        ),
        "batch_size": racer.batch_size,
    }
    return manifest, racer.algorithm


DATASET_FIELDS = ("obs", "next_obs", "action", "reward", "terminated", "truncated")


def dataset_specs(track):
    header = json.loads((DATA / f"{track}.json").read_text())
    return header, [
        jax.ShapeDtypeStruct(tuple(header["fields"][name]["shape"]), header["fields"][name]["dtype"])
        for name in DATASET_FIELDS
    ]


def transitions_from(arrays):
    obs, next_obs, action, reward, terminated, truncated = arrays
    blank = jnp.zeros_like(terminated)
    return Transition(
        first=Timestep(
            obs=obs,
            action=jnp.zeros_like(action),
            reward=jnp.zeros_like(reward),
            terminated=blank,
            truncated=blank,
        ),
        second=Timestep(
            obs=next_obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        ),
    )


def build_quadinaros(directory, combo):
    h, updates = combo["h"], combo["updates_per_call"]
    header, data_specs = dataset_specs(combo["track"])
    num_data = len(data_specs)

    def racer_for(arrays):
        return pilots.quadinaros(
            combo["pilot"], combo["track"], h, transitions_from(arrays)
        )

    template = racer_for(data_specs)
    state_shape = jax.eval_shape(
        lambda key, *arrays: racer_for(arrays).init(wrap(key)), KEY, *data_specs
    )
    no_typed_keys(state_shape)
    state_shape = state_shape.replace(dataset_state=None)
    leaves, treedef = jax.tree.flatten(state_shape)

    def init(key, *arrays):
        state = racer_for(arrays).init(wrap(key))
        return tuple(jax.tree.leaves(state.replace(dataset_state=None)))

    def fit(*args):
        arrays = args[len(leaves) : len(leaves) + num_data]
        racer = racer_for(arrays)
        state = jax.tree.unflatten(treedef, args[: len(leaves)])
        state = state.replace(dataset_state=racer.dataset.init())
        state, logs = lox.spool(lox.strip(racer.fit, tags=["evaluation"]))(
            state, wrap(args[-1]), updates
        )
        return tuple(jax.tree.leaves(state.replace(dataset_state=None)))

    evaluate = lox.spool(lox.strip(template.evaluate, tags=["training"]))

    def evaluate_call(*args):
        state = jax.tree.unflatten(treedef, args[:-1])
        _, logs = evaluate(state, wrap(args[-1]), combo["evaluation_steps"])
        return summary(logs)

    with unconstrained():
        fit_program = emit(directory, "fit", fit, [*map(spec, leaves), *data_specs, KEY])
    manifest = {
        "num_state": len(leaves),
        "state": describe(leaves),
        "params": positions(treedef, lambda state: state.algorithm_state.params),
        "dataset": f"../../data/{combo['track']}.bin",
        "dataset_header": header,
        "init": emit(directory, "init", init, [KEY, *data_specs]),
        "fit": fit_program,
        "evaluate": emit(directory, "evaluate", evaluate_call, [*map(spec, leaves), KEY]),
        "batch_size": template.batch_size,
        "evaluation_every": combo["evaluation_every"],
    }
    return manifest, template.algorithm


BUILDERS = {"anakin": build_anakin, "sebulba": build_sebulba, "quadinaros": build_quadinaros}


def identity(combo):
    return f"{combo['podracer']}-{combo['pilot']}-{combo['track']}"


def build(combo):
    directory = BUILD / identity(combo)
    directory.mkdir(parents=True, exist_ok=True)
    print(identity(combo), flush=True)
    manifest, algorithm = BUILDERS[combo["podracer"]](directory, combo)
    manifest["watch"] = watch(directory, combo, algorithm)
    manifest.update(
        podracer=combo["podracer"],
        pilot=combo["pilot"],
        track=combo["track"],
        num_envs=combo["h"]["num_envs"],
        updates_per_call=combo.get("updates_per_call", 1),
        hyperparameters=combo["h"],
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def entry(combo, manifest):
    return {
        "podracer": combo["podracer"],
        "pilot": combo["pilot"],
        "track": combo["track"],
        "num_envs": combo["h"]["num_envs"],
        "steps_per_update": manifest["batch_size"],
        "solved": combo["solved"],
        "baseline": combo["baseline"],
        "learned": combo["learned"],
        "path": f"build/{identity(combo)}/",
        **({"measured": combo["measured"]} if "measured" in combo else {}),
    }


def write_catalog():
    runs = []
    for combo in COMBOS:
        path = BUILD / identity(combo) / "manifest.json"
        if path.exists():
            runs.append(entry(combo, json.loads(path.read_text())))
    catalog = {
        "podracers": PODRACERS,
        "pilots": PILOTS,
        "tracks": TRACKS,
        "runs": runs,
    }
    (HERE / "catalog.json").write_text(json.dumps(catalog, indent=1))
    print(f"catalog: {len(runs)} runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--datasets", action="store_true")
    parser.add_argument("--catalog", action="store_true")
    arguments = parser.parse_args()
    if arguments.datasets:
        import collect

        for track in DATASETS:
            collect.collect(track)
    if not arguments.catalog:
        for combo in COMBOS:
            if arguments.only and identity(combo) not in arguments.only:
                continue
            build(combo)
    write_catalog()


if __name__ == "__main__":
    main()
