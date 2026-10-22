import argparse
import importlib
import os
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from hydra.core.hydra_config import HydraConfig
from jax.flatten_util import ravel_pytree
from omegaconf import OmegaConf

from boonta.algorithms.advantage_estimators import generalized_advantage_estimator
from boonta.utils import remove_feature_axis
from hangar import recipes
from hangar import resolvers  # noqa: F401

ADAM_DECAY, ADAM_EPSILON = 0.999, 1e-8
DROPPED = ("curriculum", "logger", "+logger", "+artisan", "+loggers", "+artisans", "environment.num_envs")


def compose(run: Path, num_envs: int, curriculum: bool = False):
    overrides = OmegaConf.load(run / ".hydra" / "overrides.yaml")
    if curriculum:
        dropped = tuple(item for item in DROPPED if item != "curriculum")
        extra = []
    else:
        dropped, extra = DROPPED, ["curriculum=default"]
    kept = [item for item in overrides if not item.startswith(dropped)]
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(
            config_name="train",
            overrides=[*kept, *extra, f"environment.num_envs={num_envs}"],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def checkpoint_steps(run: Path, wanted: list[int] | None) -> list[int]:
    held = sorted(
        int(item.name) for item in (run / "checkpoints" / "latest").iterdir() if item.name.isdigit()
    )
    if wanted:
        return [step for step in held if step in wanted]
    return sorted({held[0], held[len(held) // 2], held[-1]})


def resolve(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def ranks(values):
    order = np.argsort(values)
    ranked = np.empty(len(values))
    ranked[order] = np.arange(len(values))
    return ranked


def concatenate(chunks):
    return jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=1), *chunks)


def columns(transitions, index):
    return jax.tree.map(lambda leaf: leaf[:, index], transitions)


def main(
    run, steps, num_envs, num_chunks, warmup, repeats, seed, output,
    cell_fn=None, headroom_samples=8192, curves=True, weightings=(), reference_chunks=40,
    curriculum_batches=False, scaling=0, hindsight=(), better_policy=None, best_episodes=0.0,
    pool_steps=1024, pool_every=32,
):
    run = Path(run)
    cfg = compose(run, num_envs)
    podracer = recipes.make(cfg)
    algorithm = podracer.algorithm
    ppo = algorithm.cfg
    num_steps = int(cfg.rollout.num_steps)

    cell_of = None if cell_fn is None else resolve(cell_fn)(env=podracer.environment)

    def recorder(podracer):
        def record(state, key):
            if cell_of is None:
                cells = jnp.zeros(num_envs, jnp.int32)
            else:
                cells = cell_of(state.environment_state.unwrapped)
            state, transition = podracer.rollout(state, key, 1.0)
            return state, (transition, cells)

        return jax.jit(lambda state, keys: jax.lax.scan(record, state, keys))

    rollout = recorder(podracer)
    if curriculum_batches:
        from boonta.utils.checkpoint import load_checkpoint

        curriculum_cfg = compose(run, num_envs, curriculum=True)
        curriculum_podracer = recipes.make(curriculum_cfg)
        curriculum_rollout = recorder(curriculum_podracer)
        place = jax.jit(curriculum_podracer.pit.__call__)
        curriculum_name = curriculum_cfg.curriculum.get("_target_", "curriculum").split(".")[-2]
    advance = jax.jit(
        lambda state, keys: jax.lax.scan(
            lambda state, key: (podracer.rollout(state, key, 1.0)[0], None), state, keys
        )[0]
    )
    update = jax.jit(algorithm.update)

    def advantages_of(params, transitions):
        last_obs = jax.tree.map(lambda leaf: leaf[-1], transitions.second.obs)
        _, bootstrap = algorithm.network.apply(params, last_obs, temperature=1.0)
        advantages, _ = generalized_advantage_estimator(
            transitions,
            transitions.aux["value"],
            remove_feature_axis(bootstrap),
            ppo.gamma,
            ppo.gae_lambda,
        )
        return advantages

    def policy_loss(weights, params, transitions, advantages, sample_weights):
        dist, _ = algorithm.network.apply(
            {**params, "params": weights}, transitions.first.obs, temperature=1.0
        )
        return -jnp.mean(sample_weights * advantages * dist.log_prob(transitions.second.action))

    @jax.jit
    def policy_gradient(params, transitions, advantages, sample_weights=1.0):
        grads = jax.grad(policy_loss)(
            params["params"], params, transitions, advantages, sample_weights
        )
        flat, _ = ravel_pytree(grads)
        return flat

    @jax.jit
    def sample_power(params, obs, actions, advantages):
        def power(obs, action, advantage):
            def objective(weights):
                dist, _ = algorithm.network.apply(
                    {**params, "params": weights},
                    jax.tree.map(lambda leaf: leaf[None], obs),
                    temperature=1.0,
                )
                return advantage * dist.log_prob(action[None])[0]

            grads = jax.grad(objective)(params["params"])
            return sum(jnp.sum(leaf.astype(jnp.float32) ** 2) for leaf in jax.tree.leaves(grads))

        chunked = jax.tree.map(lambda leaf: leaf.reshape(-1, 256, *leaf.shape[1:]), (obs, actions, advantages))
        return jax.lax.map(lambda chunk: jax.vmap(power)(*chunk), chunked).reshape(-1)

    def headroom(params, train, train_advantages, cells, key):
        flat = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), train)
        advantages, cells = train_advantages.reshape(-1), cells.reshape(-1)
        total, *_ = advantages.shape
        index = jax.random.choice(key, total, (headroom_samples,), replace=False)
        power = np.asarray(sample_power(
            params, jax.tree.map(lambda leaf: leaf[index], flat.first.obs),
            flat.second.action[index], advantages[index],
        ))
        batch_gradient = policy_gradient(params, train, train_advantages)
        signal = max(float(batch_gradient @ batch_gradient) - power.mean() / total, 1e-30)
        noise = power.mean() - signal
        state_noise = np.sqrt(power).mean() ** 2 - signal
        sampled_cells = np.asarray(cells)[np.asarray(index)]
        labels, inverse, counts = np.unique(sampled_cells, return_inverse=True, return_counts=True)
        cell_power = np.bincount(inverse, weights=power) / counts
        share = counts / counts.sum()
        cell_noise = (share * np.sqrt(cell_power)).sum() ** 2 - signal
        cell_mass = np.sort(share * cell_power)[::-1] / (share * cell_power).sum()

        logits = jax.nn.softmax(flat.aux["logits"][index].astype(jnp.float32), axis=-1)
        chosen = jnp.take_along_axis(logits, flat.second.action[index][:, None], axis=-1)[:, 0]
        spread = 1.0 - 2.0 * chosen + jnp.sum(logits**2, axis=-1)
        logit_power = np.asarray(advantages[index] ** 2 * spread)

        batch = num_envs * num_steps
        def ceiling(placed_noise):
            return (1.0 + noise / signal / batch) / (1.0 + max(placed_noise, 0.0) / signal / batch)

        return dict(
            noise_scale_iid=noise / signal,
            state_multiplier=noise / max(state_noise, 1e-30),
            cell_multiplier=noise / max(cell_noise, 1e-30),
            state_ceiling=ceiling(state_noise),
            cell_ceiling=ceiling(cell_noise),
            num_cells=len(labels),
            top_cell_share=cell_mass[: max(1, len(labels) // 100)].sum(),
            top_tenth_share=cell_mass[: max(1, len(labels) // 10)].sum(),
            logit_rank_correlation=np.corrcoef(ranks(power), ranks(logit_power))[0, 1],
        )

    def sample_weights(scheme, mix, alpha, transitions, advantages, cells):
        logits = jax.nn.softmax(transitions.aux["logits"].astype(jnp.float32), axis=-1)
        chosen = jnp.take_along_axis(logits, transitions.second.action[..., None], axis=-1)[..., 0]
        spread = 1.0 - 2.0 * chosen + jnp.sum(logits**2, axis=-1)
        power = np.asarray(advantages**2 * spread)
        if scheme == "cell":
            labels, inverse = np.unique(np.asarray(cells), return_inverse=True)
            cell_power = np.bincount(inverse.ravel(), weights=power.ravel()) / np.bincount(inverse.ravel())
            power = cell_power[inverse].reshape(power.shape)
        priority = power**alpha
        priority = priority / max(priority.mean(), 1e-30)
        return jnp.asarray((1.0 - mix) + mix * priority, jnp.float32)

    def candidate(label, algorithm_state, transitions, advantages, weights, update_key):
        transitions = transitions.replace(aux={**transitions.aux, "weight": weights})
        return dict(
            label=label,
            gradient=policy_gradient(algorithm_state.params, transitions, advantages, weights),
            params=update(algorithm_state, update_key, transitions).params,
            effective=float(jnp.mean(weights) ** 2 / jnp.mean(weights**2)),
        )

    def rho_candidates(algorithm_state, train, train_advantages, train_cells, specs, key):
        num_columns = num_envs * num_chunks
        candidates = []
        for spec in specs:
            scheme, mix, alpha = (spec.split(":") + ["0.75", "1.0"])[:3] if spec != "none" else ("none", 0.0, 1.0)
            mix, alpha = float(mix), float(alpha)
            for _ in range(repeats):
                key, subset_key, update_key = jax.random.split(key, 3)
                index = jax.random.choice(subset_key, num_columns, (num_envs,), replace=False)
                subset, advantages = columns(train, index), train_advantages[:, index]
                if scheme == "none":
                    weights = jnp.ones_like(advantages)
                else:
                    weights = sample_weights(scheme, mix, alpha, subset, advantages, train_cells[:, index])
                label = "rho" if spec == "none" else f"rho {spec}"
                candidates.append(candidate(label, algorithm_state, subset, advantages, weights, update_key))
        return candidates

    def load_better(template):
        better_run, _, better_step = better_policy.partition(":")
        held_steps = sorted(
            int(item.name) for item in (Path(better_run) / "checkpoints" / "latest").iterdir()
            if item.name.isdigit()
        )
        chosen = int(better_step) if better_step else held_steps[-1]
        path = os.path.abspath(Path(better_run) / "checkpoints" / "latest" / str(chosen) / "algorithm_state")
        with ocp.StandardCheckpointer() as checkpointer:
            return checkpointer.restore(path, target=template.algorithm_state).params, chosen

    def from_better_states(better_state, algorithm_state, key):
        refresh_key, rollout_key = jax.random.split(key)
        better_state = advance(better_state, jax.random.split(refresh_key, 4 * num_steps))
        current = better_state.replace(algorithm_state=algorithm_state)
        _, (transitions, cells) = rollout(current, jax.random.split(rollout_key, num_steps))
        return better_state, transitions, cells

    def snapshot_pool(state, key):
        def stride(state, keys):
            def step(state, key):
                state, transition = podracer.rollout(state, key, 1.0)
                reward = jnp.reshape(transition.second.reward, (num_envs, -1)).sum(axis=-1)
                done = jnp.reshape(transition.second.done, (num_envs, -1)).all(axis=-1)
                return state, (done, reward)

            snapshot = (state.environment_state, state.timestep)
            state, (done, reward) = jax.lax.scan(step, state, keys)
            return state, (snapshot, done, reward)

        stride = jax.jit(stride)
        snapshots, dones, rewards = [], [], []
        for _ in range(pool_steps // pool_every):
            key, stride_key = jax.random.split(key)
            state, (snapshot, done, reward) = stride(state, jax.random.split(stride_key, pool_every))
            snapshots.append(snapshot)
            dones.append(np.asarray(done))
            rewards.append(np.asarray(reward))
        done, reward = np.concatenate(dones), np.concatenate(rewards)
        episode_return = np.full(done.shape, np.nan)
        running = np.zeros(num_envs)
        start = np.full(num_envs, -1)
        for t in range(len(done)):
            running += reward[t]
            for env in np.flatnonzero(done[t]):
                if start[env] >= 0:
                    episode_return[start[env]:t + 1, env] = running[env]
                start[env] = t + 1
                running[env] = 0.0
        if np.isnan(episode_return).all():
            raise RuntimeError("no episode finished inside the snapshot pool; raise --pool-steps")
        return state, snapshots, episode_return[::pool_every]

    def pooled_batch(state, algorithm_state, snapshots, chosen, key):
        stride_index, env_index = chosen
        environment_state, timestep = jax.tree.map(lambda leaf: leaf[stride_index, env_index], snapshots)
        start = state.replace(environment_state=environment_state, timestep=timestep, algorithm_state=algorithm_state)
        _, (transitions, cells) = rollout(start, jax.random.split(key, num_steps))
        return transitions

    def best_episode_candidates(state, algorithm_state, key):
        params = algorithm_state.params
        key, pool_key = jax.random.split(key)
        state, snapshots, snapshot_returns = snapshot_pool(state, pool_key)
        snapshots = jax.tree.map(lambda *leaves: jnp.stack(leaves), *snapshots)
        finished = ~np.isnan(snapshot_returns)
        threshold = np.nanquantile(snapshot_returns, 1.0 - best_episodes)
        groups = {
            "own episodes": np.argwhere(finished),
            f"own best {best_episodes:.0%} episodes": np.argwhere(finished & (snapshot_returns >= threshold)),
        }
        candidates = []
        for label, pool in groups.items():
            for _ in range(repeats):
                key, pick_key, rollout_key, update_key = jax.random.split(key, 4)
                rows = np.asarray(jax.random.choice(pick_key, len(pool), (num_envs,)))
                transitions = pooled_batch(
                    state, algorithm_state, snapshots, (pool[rows, 0], pool[rows, 1]), rollout_key
                )
                advantages = advantages_of(params, transitions)
                candidates.append(candidate(
                    label, algorithm_state, transitions, advantages, jnp.ones_like(advantages), update_key,
                ))
        summary = dict(
            episodes_threshold=float(threshold),
            mean_episode_return=float(np.nanmean(snapshot_returns)),
            pooled_snapshots=int(finished.sum()),
        )
        return state, candidates, summary

    def better_candidates_and_reference(state, algorithm_state, better_params, key):
        params = algorithm_state.params
        key, warm_key = jax.random.split(key)
        better_state = state.replace(algorithm_state=algorithm_state.replace(params=better_params))
        better_state = advance(better_state, jax.random.split(warm_key, warmup))
        candidates = []
        for _ in range(repeats):
            key, batch_key, update_key = jax.random.split(key, 3)
            better_state, transitions, _ = from_better_states(better_state, algorithm_state, batch_key)
            advantages = advantages_of(params, transitions)
            candidates.append(candidate(
                "better-policy states", algorithm_state, transitions, advantages,
                jnp.ones_like(advantages), update_key,
            ))
        halves = [0.0, 0.0]
        for chunk in range(reference_chunks):
            key, batch_key = jax.random.split(key)
            better_state, transitions, _ = from_better_states(better_state, algorithm_state, batch_key)
            halves[chunk % 2] = halves[chunk % 2] + policy_gradient(
                params, transitions, advantages_of(params, transitions)
            )
        first, second = (half / (reference_chunks / 2) for half in halves)
        halves_cosine = float(first @ second / (jnp.linalg.norm(first) * jnp.linalg.norm(second)))
        accuracy = np.sqrt(2 * max(halves_cosine, 0.0) / (1 + max(halves_cosine, 0.0)))
        return candidates, ((first + second) / 2, accuracy)

    def evaluate(state, algorithm_state, candidates, key, current_step=0, others=None):
        params = algorithm_state.params
        halves = [0.0, 0.0]
        gains, kls = np.zeros(len(candidates)), np.zeros(len(candidates))
        for chunk in range(reference_chunks):
            key, chunk_key = jax.random.split(key)
            state, (held, _) = rollout(state, jax.random.split(chunk_key, num_steps))
            held_advantages = advantages_of(params, held)
            halves[chunk % 2] = halves[chunk % 2] + policy_gradient(params, held, held_advantages)
            for i, entry in enumerate(candidates):
                gain, kl = improvement(entry["params"], held, held_advantages)
                gains[i] += float(gain) / reference_chunks
                kls[i] += float(kl) / reference_chunks
        first, second = (half / (reference_chunks / 2) for half in halves)
        reference = (first + second) / 2
        reference_noise = float(jnp.sum((first - second) ** 2)) / 4
        signal = max(float(reference @ reference) - reference_noise, 1e-30)
        halves_cosine = float(first @ second / (jnp.linalg.norm(first) * jnp.linalg.norm(second)))
        half_accuracy = max(halves_cosine, 0.0)

        true_norm = np.sqrt(signal)
        direction = reference / jnp.linalg.norm(reference)

        def components(gradient):
            along = float(gradient @ direction)
            across = float(jnp.linalg.norm(gradient - along * direction))
            return along / true_norm, across / true_norm

        adam = next(
            node for node in jax.tree.leaves(
                algorithm_state.optimizer_state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState)
            )
            if isinstance(node, optax.ScaleByAdamState)
        )
        second_moment, _ = ravel_pytree(adam.nu)
        corrected = second_moment / (1.0 - ADAM_DECAY ** jnp.maximum(adam.count, 1))
        preconditioner = 1.0 / (jnp.sqrt(corrected) + ADAM_EPSILON)
        policy_mask = reference != 0.0
        old_weights, _ = ravel_pytree(params["params"])
        steepest = -preconditioner * reference
        steepest_direction = steepest / jnp.linalg.norm(steepest)

        def preconditioned_cosine(gradient):
            return float(
                gradient @ (preconditioner * reference)
                / jnp.sqrt((gradient @ (preconditioner * gradient)) * (reference @ (preconditioner * reference)))
            )

        futures = {}
        for spec in hindsight:
            label, future_run = spec.split("=", 1)
            held_steps = sorted(
                int(item.name) for item in (Path(future_run) / "checkpoints" / "latest").iterdir()
                if item.name.isdigit()
            )
            later = [held for held in held_steps if held > current_step]
            if not later:
                continue
            path = os.path.abspath(Path(future_run) / "checkpoints" / "latest" / str(later[0]) / "algorithm_state")
            with ocp.StandardCheckpointer() as checkpointer:
                future = checkpointer.restore(path, target=algorithm_state)
            future_weights, _ = ravel_pytree(future.params["params"])
            heading = jnp.where(policy_mask, future_weights - old_weights, 0.0)
            futures[label] = (heading / jnp.linalg.norm(heading), later[0])

        labels_ahead = list(futures)
        for i, first_label in enumerate(labels_ahead):
            for second_label in labels_ahead[i + 1:]:
                print(
                    f"step {current_step} futures '{first_label}' (to {futures[first_label][1]:,}) and "
                    f"'{second_label}' (to {futures[second_label][1]:,}) point the same way with cosine "
                    f"{float(futures[first_label][0] @ futures[second_label][0]):.3f}"
                )

        def hindsight_cosines(new_params):
            new_weights, _ = ravel_pytree(new_params["params"])
            step = jnp.where(policy_mask, new_weights - old_weights, 0.0)
            step = step / jnp.linalg.norm(step)
            return {label: float(step @ heading) for label, (heading, _) in futures.items()}

        def step_components(new_params):
            new_weights, _ = ravel_pytree(new_params["params"])
            step = jnp.where(policy_mask, new_weights - old_weights, 0.0)
            along = float(step @ steepest_direction)
            across = float(jnp.linalg.norm(step - along * steepest_direction))
            predicted_gain = -float(reference @ step)
            return along, across, predicted_gain

        results = {}
        for label in dict.fromkeys(entry["label"] for entry in candidates):
            chosen = [i for i, entry in enumerate(candidates) if entry["label"] == label]
            mean_gradient = sum(candidates[i]["gradient"] for i in chosen) / len(chosen)
            cosines, errors, along, across = [], [], [], []
            adam_cosines, step_along, step_across, predicted = [], [], [], []
            toward = {label: [] for label in futures}
            for i in chosen:
                gradient = candidates[i]["gradient"]
                parallel, perpendicular = components(gradient)
                along.append(parallel)
                across.append(perpendicular)
                adam_cosines.append(preconditioned_cosine(gradient))
                moved_along, moved_across, gain = step_components(candidates[i]["params"])
                step_along.append(moved_along)
                step_across.append(moved_across)
                predicted.append(gain)
                for future_label, value in hindsight_cosines(candidates[i]["params"]).items():
                    toward[future_label].append(value)
                cosines.append(float(
                    gradient @ reference / (jnp.linalg.norm(gradient) * jnp.linalg.norm(reference))
                ))
                errors.append((float(jnp.sum((gradient - reference) ** 2)) - reference_noise) / signal)
            results[label] = dict(
                **{
                    f"cosine to {other}": np.asarray([
                        float(candidates[i]["gradient"] @ other_gradient
                              / (jnp.linalg.norm(candidates[i]["gradient"]) * jnp.linalg.norm(other_gradient)))
                        for i in chosen
                    ])
                    for other, (other_gradient, _) in (others or {}).items()
                },
                gain=gains[chosen], kl=kls[chosen], cosine=np.asarray(cosines),
                error=np.asarray(errors), along=np.asarray(along), across=np.asarray(across),
                mean_components=np.asarray(components(mean_gradient)),
                adam_cosine=np.asarray(adam_cosines), step_along=np.asarray(step_along),
                step_across=np.asarray(step_across), predicted_gain=np.asarray(predicted),
                **{f"toward {future_label}": np.asarray(values) for future_label, values in toward.items()},
                effective=np.asarray([candidates[i]["effective"] for i in chosen]),
            )
        for other, (other_gradient, other_accuracy) in (others or {}).items():
            print(
                f"step {current_step} the {other} gradient (accuracy {other_accuracy:.3f}) has cosine "
                f"{float(reference @ other_gradient / (jnp.linalg.norm(reference) * jnp.linalg.norm(other_gradient))):.3f} "
                f"with the gradient of J_rho"
            )
        reliability = dict(
            reference_samples=reference_chunks * num_envs * num_steps,
            halves_cosine=halves_cosine,
            reference_accuracy=np.sqrt(2 * half_accuracy / (1 + half_accuracy)),
        )
        return state, results, reliability

    def restore_curriculum(step, algorithm_state, key):
        template = curriculum_podracer.init(key)
        restored = load_checkpoint(run / "checkpoints" / "latest" / str(step), template)
        return restored.replace(algorithm_state=algorithm_state)

    def scaled(step, algorithm_state, state, key):
        params = algorithm_state.params
        fractions = (1 / 16, 1 / 8, 1 / 4, 1 / 2)
        gradients = {}

        def add(label, transitions, advantages, weights, chunk_key):
            weights = jnp.broadcast_to(jnp.asarray(weights, jnp.float32), advantages.shape)
            full = np.asarray(policy_gradient(params, transitions, advantages, weights))
            gradients.setdefault(label, {}).setdefault(1.0, []).append(full)
            for fraction in fractions:
                chunk_key, subset_key = jax.random.split(chunk_key)
                index = jax.random.choice(subset_key, num_envs, (int(num_envs * fraction),), replace=False)
                sub = np.asarray(policy_gradient(
                    params, columns(transitions, index), advantages[:, index], weights[:, index]
                ))
                gradients[label].setdefault(fraction, []).append(sub)

        if curriculum_batches:
            key, restore_key = jax.random.split(key)
            restored = restore_curriculum(step, algorithm_state, restore_key)
        for _ in range(scaling):
            key, rho_key, place_key, rollout_key, subset_key, other_key = jax.random.split(key, 6)
            state, (transitions, cells) = rollout(state, jax.random.split(rho_key, num_steps))
            advantages = advantages_of(params, transitions)
            add("rho", transitions, advantages, 1.0, subset_key)
            for spec in weightings:
                scheme, mix, alpha = (spec.split(":") + ["0.75", "1.0"])[:3]
                weights = sample_weights(scheme, float(mix), float(alpha), transitions, advantages, cells)
                add(f"rho {spec}", transitions, advantages, weights, subset_key)
            if curriculum_batches:
                restored = place(place_key, restored)
                restored, (transitions, cells) = curriculum_rollout(
                    restored, jax.random.split(rollout_key, num_steps)
                )
                advantages = advantages_of(params, transitions)
                restarted = ~jnp.asarray(transitions.second.info["rho"])
                started = ~restarted
                add(curriculum_name, transitions, advantages, 1.0, other_key)
                add(f"{curriculum_name}: rho block", transitions, advantages,
                    started / jnp.maximum(jnp.mean(started), 1e-6), other_key)
                add(f"{curriculum_name}: restarted block", transitions, advantages,
                    restarted / jnp.maximum(jnp.mean(restarted), 1e-6), other_key)

        halves = [0.0, 0.0]
        for chunk in range(reference_chunks):
            key, chunk_key = jax.random.split(key)
            state, (held, _) = rollout(state, jax.random.split(chunk_key, num_steps))
            halves[chunk % 2] = halves[chunk % 2] + np.asarray(
                policy_gradient(params, held, advantages_of(params, held))
            )
        first, second = (half / (reference_chunks / 2) for half in halves)
        reference = (first + second) / 2
        reference_noise = float(np.sum((first - second) ** 2)) / 4
        true_norm = np.sqrt(max(float(reference @ reference) - reference_noise, 1e-30))
        direction = reference / np.linalg.norm(reference)
        halves_cosine = float(first @ second / (np.linalg.norm(first) * np.linalg.norm(second)))
        accuracy = np.sqrt(2 * max(halves_cosine, 0.0) / (1 + max(halves_cosine, 0.0)))

        batch = num_envs * num_steps
        curves = {}
        for label, by_size in gradients.items():
            sizes, cosine, cosine_spread, along, across = [], [], [], [], []
            groups = {fraction: [[g] for g in by_size[fraction]] for fraction in fractions}
            full = by_size[1.0]
            multiple = 1
            while multiple <= len(full):
                groups[float(multiple)] = [full[i:i + multiple] for i in range(0, len(full) - multiple + 1, multiple)]
                multiple *= 2
            for size in sorted(groups):
                values = [np.mean(group, axis=0) for group in groups[size]]
                cosines = [float(v @ direction / np.linalg.norm(v)) for v in values]
                parallel = [float(v @ direction) / true_norm for v in values]
                perpendicular = [
                    float(np.linalg.norm(v - (v @ direction) * direction)) / true_norm for v in values
                ]
                sizes.append(size * batch)
                cosine.append(np.mean(cosines))
                cosine_spread.append(np.std(cosines))
                along.append(np.mean(parallel))
                across.append(np.mean(perpendicular))
            curves[label] = dict(
                samples=np.asarray(sizes), cosine=np.asarray(cosine),
                cosine_spread=np.asarray(cosine_spread), along=np.asarray(along), across=np.asarray(across),
            )
        return state, curves, dict(reference_accuracy=accuracy, reference_samples=reference_chunks * batch)

    def curriculum_candidates(step, algorithm_state, train_cells, key):
        key, init_key = jax.random.split(key)
        restored = restore_curriculum(step, algorithm_state, init_key)
        batches = []
        for _ in range(repeats):
            key, place_key, rollout_key = jax.random.split(key, 3)
            restored = place(place_key, restored)
            restored, (transitions, cells) = curriculum_rollout(
                restored, jax.random.split(rollout_key, num_steps)
            )
            batches.append((transitions, cells))
        rho_labels, rho_counts = np.unique(np.asarray(train_cells), return_counts=True)
        rho_share = dict(zip(rho_labels.tolist(), (rho_counts / rho_counts.sum()).tolist()))
        pooled = np.concatenate([np.asarray(cells).ravel() for _, cells in batches])
        labels, counts = np.unique(pooled, return_counts=True)
        batch_share = dict(zip(labels.tolist(), (counts / counts.sum()).tolist()))
        candidates = []
        for transitions, cells in batches:
            advantages = advantages_of(algorithm_state.params, transitions)
            key, update_key, corrected_key = jax.random.split(key, 3)
            candidates.append(candidate(
                curriculum_name, algorithm_state, transitions, advantages,
                jnp.ones_like(advantages), update_key,
            ))
            correction = np.vectorize(lambda cell: rho_share.get(cell, 0.0) / batch_share[cell])(np.asarray(cells))
            candidates.append(candidate(
                f"{curriculum_name} corrected", algorithm_state, transitions, advantages,
                jnp.asarray(correction, jnp.float32), corrected_key,
            ))
        restart_share = float(np.mean([np.mean(~np.asarray(t.second.info["rho"])) for t, _ in batches]))
        unseen = float(np.mean([rho_share.get(cell, 0.0) == 0.0 for cell in pooled]))
        return candidates, dict(restart_share=restart_share, unseen_by_rho=unseen)

    @jax.jit
    def improvement(params, held, held_advantages):
        dist, _ = algorithm.network.apply(params, held.first.obs, temperature=1.0)
        ratio = jnp.exp(dist.log_prob(held.second.action) - held.aux["log_prob"])
        old = jax.nn.log_softmax(held.aux["logits"].astype(jnp.float32), axis=-1)
        new = jax.nn.log_softmax(dist.logits.astype(jnp.float32), axis=-1)
        kl = jnp.sum(jnp.exp(old) * (old - new), axis=-1)
        return jnp.mean(held_advantages * (ratio - 1.0)), jnp.mean(kl)

    key = jax.random.key(seed)
    init_key, key = jax.random.split(key)
    template = podracer.init(init_key)
    output = Path(output or run / "batch_saturation")
    output.mkdir(parents=True, exist_ok=True)
    report = {}

    for step in checkpoint_steps(run, steps):
        path = os.path.abspath(run / "checkpoints" / "latest" / str(step) / "algorithm_state")
        with ocp.StandardCheckpointer() as checkpointer:
            algorithm_state = checkpointer.restore(path, target=template.algorithm_state)
        state = template.replace(algorithm_state=algorithm_state)
        params = algorithm_state.params

        key, warmup_key, *chunk_keys = jax.random.split(key, 2 + num_chunks + 1)
        state = advance(state, jax.random.split(warmup_key, warmup))
        chunks, chunk_cells = [], []
        for chunk_key in chunk_keys:
            state, (transitions, cells) = rollout(state, jax.random.split(chunk_key, num_steps))
            chunks.append(transitions)
            chunk_cells.append(cells)
        train, held = concatenate(chunks[:-1]), chunks[-1]
        train_cells = jnp.concatenate(chunk_cells[:-1], axis=1)
        train_advantages = advantages_of(params, train)
        held_advantages = advantages_of(params, held)
        num_columns = num_envs * num_chunks

        if headroom_samples:
            key, headroom_key = jax.random.split(key)
            room = headroom(params, train, train_advantages, train_cells, headroom_key)
            report.setdefault(step, {}).update({f"headroom/{name}": value for name, value in room.items()})
            print(
                f"step {step}: per-sample noise scale {room['noise_scale_iid']:,.0f}; "
                f"best placement multiplies the batch by {room['state_multiplier']:.2f} (states) / "
                f"{room['cell_multiplier']:.2f} (cells, {room['num_cells']} seen), "
                f"improvement ceiling per update x{room['state_ceiling']:.2f} / x{room['cell_ceiling']:.2f}; "
                f"top 1% / 10% of cells hold {room['top_cell_share']:.0%} / {room['top_tenth_share']:.0%} "
                f"of gradient power; rank corr. with advantage^2 x spread {room['logit_rank_correlation']:.2f}"
            )
        if scaling:
            key, scaling_key = jax.random.split(key)
            state, curves_by_label, reliability = scaled(step, algorithm_state, state, scaling_key)
            report.setdefault(step, {}).update(
                {f"scaling/{label}/{name}": value for label, curve in curves_by_label.items() for name, value in curve.items()}
            )
            report[step].update({f"scaling-reference/{name}": value for name, value in reliability.items()})
            print(f"step {step} scaling reference accuracy {reliability['reference_accuracy']:.3f}")
            for label, curve in curves_by_label.items():
                print(
                    f"step {step} {label:>28}: cosine by batch size "
                    + " ".join(f"{int(n / 1000)}k:{c:.2f}" for n, c in zip(curve["samples"], curve["cosine"]))
                    + " | along "
                    + " ".join(f"{a:.2f}" for a in curve["along"])
                )
            plot_scaling(report, output / "scaling.png")
            continue
        if weightings or curriculum_batches or better_policy or best_episodes:
            key, candidates_key, curriculum_key, evaluate_key = jax.random.split(key, 4)
            candidates = rho_candidates(
                algorithm_state, train, train_advantages, train_cells, ("none", *weightings), candidates_key
            )
            if curriculum_batches:
                extra, composition = curriculum_candidates(step, algorithm_state, train_cells, curriculum_key)
                candidates += extra
                report.setdefault(step, {}).update({f"composition/{k}": v for k, v in composition.items()})
                print(
                    f"step {step} {curriculum_name} batches: {composition['restart_share']:.0%} of samples "
                    f"restarted, {composition['unseen_by_rho']:.1%} in cells the rho pool never visited"
                )
            others = {}
            if best_episodes:
                key, best_key = jax.random.split(key)
                state, extra, summary = best_episode_candidates(state, algorithm_state, best_key)
                candidates += extra
                print(
                    f"step {step} own episodes: {summary['pooled_snapshots']:,} snapshots from finished "
                    f"episodes, mean return {summary['mean_episode_return']:.2f}, best-episode threshold "
                    f"{summary['episodes_threshold']:.2f}"
                )
            if better_policy:
                key, better_key = jax.random.split(key)
                better_params, better_step = load_better(template)
                extra, others["better-policy states"] = better_candidates_and_reference(
                    state, algorithm_state, better_params, better_key
                )
                candidates += extra
                print(f"step {step} better policy: {better_policy.split(':')[0]} at {better_step:,}")
            state, compared, reliability = evaluate(
                state, algorithm_state, candidates, evaluate_key, step, others
            )
            report.setdefault(step, {}).update({f"reference/{name}": value for name, value in reliability.items()})
            print(
                f"step {step} reference: {reliability['reference_samples']:,} samples, cosine between its "
                f"halves {reliability['halves_cosine']:.3f}, so cosine with the true gradient about "
                f"{reliability['reference_accuracy']:.3f}"
            )
            baseline = compared["rho"]["gain"].mean()
            for label, entry in compared.items():
                report[step].update({f"batch/{label}/{name}": value for name, value in entry.items()})
                print(
                    f"step {step} {label:>22}: cosine {entry['cosine'].mean():.3f} "
                    f"(Adam geometry {entry['adam_cosine'].mean():.3f}), real step along/across the "
                    f"steepest Adam direction {entry['step_along'].mean():.3g}/{entry['step_across'].mean():.3g}, "
                    f"predicted J gain {entry['predicted_gain'].mean():.3g} (x{entry['predicted_gain'].mean() / compared['rho']['predicted_gain'].mean():.2f}), "
                    f"(spread {entry['cosine'].std():.2f}), relative error {entry['error'].mean():.2f}, "
                    f"held-out gain {entry['gain'].mean():.3g} "
                    f"(x{entry['gain'].mean() / baseline if baseline > 0 else float('nan'):.2f} of rho, "
                    f"spread {entry['gain'].std():.2g}), KL {entry['kl'].mean():.4f}, "
                    f"effective batch {entry['effective'].mean():.0%}"
                    + "".join(
                        f", {name} {entry[name].mean():.3f}"
                        for name in entry if name.startswith(("toward ", "cosine to "))
                    )
                )
        if not curves:
            continue

        held_gradient = policy_gradient(params, held, held_advantages)
        sizes = [2**power for power in range(4, int(np.log2(num_columns)) + 1)]
        cosine, squared_norm, gain, kl = (np.zeros((len(sizes), repeats)) for _ in range(4))
        for i, size in enumerate(sizes):
            for r in range(repeats):
                key, subset_key, update_key = jax.random.split(key, 3)
                index = jax.random.choice(subset_key, num_columns, (size,), replace=False)
                subset, subset_advantages = columns(train, index), train_advantages[:, index]
                gradient = policy_gradient(params, subset, subset_advantages)
                cosine[i, r] = float(
                    gradient @ held_gradient
                    / (jnp.linalg.norm(gradient) * jnp.linalg.norm(held_gradient) + 1e-30)
                )
                squared_norm[i, r] = float(gradient @ gradient)
                updated = update(algorithm_state, update_key, subset)
                gain[i, r], kl[i, r] = map(float, improvement(updated.params, held, held_advantages))

        samples = np.asarray(sizes) * num_steps
        small, big = samples[0], samples[-1]
        small_norm, big_norm = squared_norm[0].mean(), squared_norm[-1].mean()
        signal = (big * big_norm - small * small_norm) / (big - small)
        noise = (small_norm - big_norm) / (1.0 / small - 1.0 / big)
        noise_scale = noise / signal if signal > 0 else np.inf
        report.setdefault(step, {}).update(
            samples=samples, cosine=cosine, squared_norm=squared_norm, gain=gain, kl=kl,
            noise_scale=noise_scale, standard_batch=num_envs * num_steps,
        )
        print(
            f"step {step}: noise scale {noise_scale:,.0f} samples "
            f"(standard batch {num_envs * num_steps:,}); "
            f"gain at 1/4/all of the batch "
            f"{gain[len(sizes) - 3].mean():.3g} / {gain[-1].mean():.3g}"
        )

    np.savez(
        output / "batch_saturation.npz",
        **{f"{step}/{name}": value for step, entry in report.items() for name, value in entry.items()},
    )
    if curves:
        plot(report, output / "batch_saturation.png")
    if any(name.startswith("batch/") for entry in report.values() for name in entry):
        plot_gradients(report, output / "gradients.png")
        plot_steps(report, output / "steps.png")
    print(f"wrote {output}")


def plot_scaling(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [step for step, entry in report.items() if any(name.startswith("scaling/") for name in entry)]
    figure, axes = plt.subplots(2, len(steps), figsize=(5.5 * len(steps), 9), squeeze=False)
    for column, step in enumerate(steps):
        entry = report[step]
        labels = list(dict.fromkeys(name.split("/")[1] for name in entry if name.startswith("scaling/")))
        top, bottom = axes[0][column], axes[1][column]
        for label in labels:
            samples = entry[f"scaling/{label}/samples"]
            cosine, spread = entry[f"scaling/{label}/cosine"], entry[f"scaling/{label}/cosine_spread"]
            line, = top.plot(samples, cosine, marker="o", label=label)
            top.fill_between(samples, cosine - spread, cosine + spread, alpha=0.15, color=line.get_color())
            bottom.plot(samples, entry[f"scaling/{label}/along"], marker="o", color=line.get_color())
            bottom.plot(samples, entry[f"scaling/{label}/across"], marker="x", linestyle=":", color=line.get_color())
        accuracy = entry["scaling-reference/reference_accuracy"]
        top.axhline(accuracy, color="black", linestyle="--", lw=1, label=f"reference accuracy {accuracy:.2f}")
        top.set_title(f"step {step:,}: cosine with the true gradient")
        bottom.set_title("along (o, solid) and across (x, dotted) the true gradient")
        bottom.axhline(1.0, color="black", lw=1)
        for axis in (top, bottom):
            axis.set_xscale("log")
            axis.set_xlabel("samples in the batch")
        bottom.set_yscale("log")
        top.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def plot_steps(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [step for step, entry in report.items() if any(name.endswith("/step_along") for name in entry)]
    figure, axes = plt.subplots(1, len(steps), figsize=(5.5 * len(steps), 5), squeeze=False)
    for axis, step in zip(axes[0], steps):
        entry = report[step]
        labels = list(dict.fromkeys(name.split("/")[1] for name in entry if name.startswith("batch/")))
        unit = np.mean(entry["batch/rho/step_along"])
        reach = 1.0
        for label in labels:
            along = entry[f"batch/{label}/step_along"] / unit
            across = entry[f"batch/{label}/step_across"] / unit
            points = axis.scatter(along, across, alpha=0.5, s=25)
            colour = points.get_facecolor()[0]
            axis.annotate("", xy=(along.mean(), across.mean()), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=colour, lw=2))
            gain = entry[f"batch/{label}/predicted_gain"].mean() / entry["batch/rho/predicted_gain"].mean()
            axis.plot([], [], color=colour, lw=2, label=f"{label} (J gain x{gain:.2f})")
            reach = max(reach, np.max(np.abs(along)), np.max(across))
        axis.set_title(f"step {step:,}: the actual PPO step (clipping, Adam, full loss)")
        axis.set_xlabel("along the steepest Adam direction for J_rho (units: rho's mean)")
        axis.set_ylabel("across it")
        axis.axhline(0, color="grey", lw=0.5)
        axis.axvline(0, color="grey", lw=0.5)
        axis.set_xlim(-0.3 * reach, 1.1 * reach)
        axis.set_ylim(0, 1.1 * reach)
        axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def plot_gradients(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [step for step, entry in report.items() if any(name.startswith("batch/") for name in entry)]
    figure, axes = plt.subplots(1, len(steps), figsize=(5.5 * len(steps), 5), squeeze=False)
    for axis, step in zip(axes[0], steps):
        entry = report[step]
        labels = list(dict.fromkeys(name.split("/")[1] for name in entry if name.startswith("batch/")))
        reach = 1.0
        for label in labels:
            along, across = entry[f"batch/{label}/along"], entry[f"batch/{label}/across"]
            mean_along, mean_across = entry[f"batch/{label}/mean_components"]
            points = axis.scatter(along, across, alpha=0.5, s=25)
            colour = points.get_facecolor()[0]
            axis.annotate(
                "", xy=(mean_along, mean_across), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=colour, lw=2),
            )
            axis.plot([], [], color=colour, lw=2, label=f"{label} (cos {entry[f'batch/{label}/cosine'].mean():.2f})")
            reach = max(reach, np.max(np.abs(along)), np.max(across), abs(mean_along), mean_across)
        axis.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=3))
        axis.plot([], [], color="black", lw=3, label="true gradient")
        accuracy = entry.get("reference/reference_accuracy", np.nan)
        axis.set_title(f"step {step:,} (reference accuracy {accuracy:.2f})")
        axis.set_xlabel("component along the true gradient (units of its length)")
        axis.set_ylabel("component perpendicular to it")
        axis.axhline(0, color="grey", lw=0.5)
        axis.axvline(0, color="grey", lw=0.5)
        axis.set_xlim(-0.3 * reach, 1.1 * reach)
        axis.set_ylim(0, 1.1 * reach)
        axis.set_aspect("equal")
        axis.legend(fontsize=8, loc="upper left")
    figure.suptitle("Batch gradients split along and across the true gradient of J_rho; dots are single batches, arrows their mean")
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("cosine", "cosine with held-out gradient"),
        ("gain", "held-out surrogate improvement"),
        ("kl", "held-out KL(old || new)"),
    ]
    for step, entry in report.items():
        for axis, (name, title) in zip(axes, panels):
            values = entry[name]
            line, = axis.plot(entry["samples"], values.mean(axis=1), marker="o", label=f"{step:,}")
            axis.fill_between(
                entry["samples"], values.min(axis=1), values.max(axis=1), alpha=0.2, color=line.get_color()
            )
            axis.set_title(title)
        for axis in axes:
            axis.axvline(entry["standard_batch"], color="grey", linestyle=":")
        if np.isfinite(entry["noise_scale"]):
            axes[0].axvline(entry["noise_scale"], color=line.get_color(), linestyle="--")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("samples in the batch")
    axes[0].legend(title="checkpoint step")
    figure.tight_layout()
    figure.savefig(path, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Does a PPO update keep improving as its batch grows? Gradient and update "
        "saturation curves at a run's checkpoints, on batches from rho."
    )
    parser.add_argument("run", help="hydra run directory with checkpoints/latest")
    parser.add_argument("--steps", type=int, nargs="*", help="checkpoint steps (default: first, middle, last)")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-chunks", type=int, default=4, help="rollout windows pooled into the training batch")
    parser.add_argument("--warmup", type=int, default=1000, help="policy steps before collecting")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--cell-fn", help="cell function import path, e.g. boonta.environments.craftax.progress")
    parser.add_argument("--headroom-samples", type=int, default=8192, help="0 skips the headroom")
    parser.add_argument("--no-curves", dest="curves", action="store_false", help="skip the batch-size curves")
    parser.add_argument(
        "--weighting", nargs="*", default=[],
        help="reweightings of a standard rho batch to compare with the unweighted update, each "
        "scheme:mix:alpha with scheme 'sample' (advantage^2 x spread per sample) or 'cell' "
        "(its mean per cell, needs --cell-fn); weight = (1 - mix) + mix * priority^alpha / mean",
    )
    parser.add_argument(
        "--reference-chunks", type=int, default=40,
        help="standard batches streamed into the reference gradient and held-out gains",
    )
    parser.add_argument(
        "--curriculum-batches", action="store_true",
        help="also restore the run's curriculum (archive and ledger) and compare its real batches, "
        "as is and importance-corrected per cell, with rho batches of the same size",
    )
    parser.add_argument(
        "--scaling", type=int, default=0,
        help="standard batches per type for the bias-versus-noise sweep over batch sizes (0 skips it)",
    )
    parser.add_argument(
        "--hindsight", nargs="*", default=[],
        help="label=run_dir pairs; compares each batch's real step with the direction from this "
        "checkpoint to that run's next checkpoint (runs must share these parameters, e.g. forks)",
    )
    parser.add_argument(
        "--better-policy",
        help="run_dir[:step] of a stronger policy; its states (not its parameters) seed batches of "
        "the current policy, whose gradient approximates improving where the better policy goes",
    )
    parser.add_argument(
        "--best-episodes", type=float, default=0.0,
        help="fraction of the current policy's own top-returning episodes from rho whose states seed "
        "restart batches (compared with seeding from all its own episodes); 0 skips it",
    )
    parser.add_argument("--pool-steps", type=int, default=1024, help="policy steps recorded for the snapshot pool")
    parser.add_argument("--pool-every", type=int, default=32, help="keep a snapshot every this many steps")
    args = parser.parse_args()
    main(
        args.run, args.steps, args.num_envs, args.num_chunks, args.warmup,
        args.repeats, args.seed, args.output, args.cell_fn, args.headroom_samples, args.curves,
        args.weighting, args.reference_chunks, args.curriculum_batches, args.scaling, args.hindsight,
        args.better_policy, args.best_episodes, args.pool_steps, args.pool_every,
    )
