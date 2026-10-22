import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from render_checkpoint import load
from signal_map import gradients, policy_table

from hangar import resolvers  # noqa: F401

GAP = 20


def stationary(transitions, start, iterations=4000):
    distribution = jax.nn.one_hot(start, transitions.shape[0])
    distribution, _ = jax.lax.scan(lambda row, _: (row @ transitions, None), distribution, None, length=iterations)
    return distribution


def occupancy(starts, transitions, num_steps):
    def step(carry, _):
        row, total = carry
        return (row @ transitions, total + row), None

    (_, total), _ = jax.lax.scan(step, (starts, jnp.zeros_like(starts)), None, length=num_steps)
    return total


def archive_of(environment_state):
    from boonta.environments.wrappers.archive_auto_reset import ArchiveAutoResetState

    return next(
        node
        for node in jax.tree.leaves(environment_state, is_leaf=lambda node: isinstance(node, ArchiveAutoResetState))
        if isinstance(node, ArchiveAutoResetState)
    )


def prd_restarts(selector, wrapper, checkpoint, algorithm, num_cells, key, num_draws=16):
    archive = archive_of(checkpoint.environment_state)
    cells = jnp.squeeze(archive.archive_state.keys, axis=-1)

    def draw(key):
        slots = selector.select(
            checkpoint.pit_state, wrapper, archive, key, algorithm=algorithm, algorithm_state=checkpoint.algorithm_state
        )
        return jnp.zeros(num_cells).at[cells[slots]].add(1.0)

    counts = jnp.sum(jax.vmap(draw)(jax.random.split(key, num_draws)), axis=0)
    return counts / jnp.sum(counts)


def erd_restarts(wrapper, checkpoint, num_cells):
    archive_state = archive_of(checkpoint.environment_state).archive_state
    cells = jnp.squeeze(archive_state.keys, axis=-1)
    eligible = wrapper.archive.eligible(archive_state)
    counts = jnp.zeros(num_cells).at[jnp.where(eligible, cells, num_cells)].add(1.0, mode="drop")
    return counts / jnp.sum(counts)


def per_cell_scores(score, selector, checkpoint, wrapper, algorithm, num_cells):
    archive = archive_of(checkpoint.environment_state)
    per_slot = score(
        selector, checkpoint.pit_state, wrapper, archive, algorithm=algorithm, algorithm_state=checkpoint.algorithm_state
    )
    archive_state = archive.archive_state
    cells = jnp.squeeze(archive_state.keys, axis=-1)
    return jnp.zeros(num_cells).at[jnp.where(archive_state.mask, cells, num_cells)].set(per_slot, mode="drop")


def cosine(left, right):
    return float(left @ right / (jnp.linalg.norm(left) * jnp.linalg.norm(right) + 1e-30))


def diagnose(selector, wrapper, checkpoint, algorithm, objective, origin, unravel, per_cell, optimum, num_cells):
    from scipy.stats import spearmanr

    from boonta.curricula.prd import Diagonal, Kernel, precondition

    archive_state = archive_of(checkpoint.environment_state).archive_state
    held = np.zeros(num_cells, bool)
    held[np.asarray(jnp.squeeze(archive_state.keys, axis=-1))[np.asarray(archive_state.mask)]] = True

    def agreement(scores):
        return spearmanr(np.asarray(scores)[held], np.asarray(optimum)[held]).statistic

    print("rank correlation with the optimal restarts, over the archive's cells:")
    diagonal = per_cell_scores(Diagonal(), selector, checkpoint, wrapper, algorithm, num_cells)
    print(f"  diagonal score from this ledger           {agreement(diagonal):+.3f}")
    if not isinstance(selector.score, Kernel):
        return
    optimizer_state = checkpoint.algorithm_state.optimizer_state
    adam = next(leaf for leaf in jax.tree.leaves(optimizer_state, is_leaf=lambda node: hasattr(node, "nu")) if hasattr(leaf, "nu"))
    second, _ = ravel_pytree(adam.nu)
    seen = second > 1e-12
    exact, _ = ravel_pytree(precondition(unravel(jax.grad(objective)(origin)), optimizer_state))
    exact = jnp.where(seen, exact, 0.0)
    estimated, _ = ravel_pytree(checkpoint.pit_state.direction)
    kernel = per_cell_scores(selector.score, selector, checkpoint, wrapper, algorithm, num_cells)
    print(f"  kernel score as PRD computes it           {agreement(kernel):+.3f}")
    print(f"  kernel with exact advantages              {agreement(per_cell @ estimated):+.3f}")
    print(f"  kernel with exact advantages and direction {agreement(per_cell @ exact):+.3f}")
    print(
        "cosine between the kernel's direction and the exact preconditioned gradient of J_rho, "
        f"over the parameters Adam has seen: {cosine(jnp.where(seen, estimated, 0.0), exact):+.3f}"
    )


def optimal_policy(game, policy, iterations=100):
    num_actions = policy.shape[-1]

    def improve(policy, _):
        return jax.nn.one_hot(jnp.argmax(game.advantages(policy) + 1e-9 * policy, axis=-1), num_actions), None

    policy, _ = jax.lax.scan(improve, policy, None, length=iterations)
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--checkpoint", help="checkpoint to update from (default: the run's latest)")
    parser.add_argument("--kl", type=float, default=0.01, help="size of the update: mean KL on the states it is measured on")
    parser.add_argument(
        "--trust",
        choices=["max", "batch", "rho"],
        default="max",
        help="bound the KL of every state, as PPO's clipping bounds every sample, or the mean on the batch's or rho's states",
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--tile", type=int, default=16)
    arguments = parser.parse_args()
    run = Path(arguments.run).resolve()
    trained_with = OmegaConf.load(run / ".hydra" / "config.yaml")
    cfg = load(run, [f"environment.num_envs={trained_with.environment.num_envs}", *arguments.overrides])

    from boonta.artisans.overlay import paint
    from boonta.utils import load_checkpoint, resolve_checkpoint
    from hangar import recipes

    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment
    game, selector = environment.unwrapped, podracer.pit.selector
    key = jax.random.key(cfg.seed)
    checkpoint = load_checkpoint(resolve_checkpoint(arguments.checkpoint or run), podracer.init(key))
    step = int(checkpoint.algorithm_state.step)
    print(f"checkpoint at step {step:,}")

    variables = checkpoint.algorithm_state.params
    origin, unravel = ravel_pytree(variables["params"])
    _, observed, policy = policy_table(algorithm, checkpoint.algorithm_state, game, key)
    per_cell = gradients(algorithm, checkpoint.algorithm_state, observed, policy * game.advantages(policy))
    transitions = game.transitions(policy)
    visits_rho = stationary(transitions, game.start)
    continuing = 1.0 - selector.num_placed(environment.num_envs) / environment.num_envs
    num_steps = podracer.config.num_steps

    def log_policy(flat):
        dist, _ = algorithm.network.apply({**variables, "params": unravel(flat)}, observed, temperature=1.0)
        return dist.logits

    def objective(flat):
        return game.expected_return(jnp.exp(log_policy(flat)))

    before = float(objective(origin))

    def stepped(direction, batch):
        direction = direction / jnp.linalg.norm(direction)
        _, change = jax.jvp(log_policy, (origin,), (direction,))
        per_state = jnp.sum(policy * change**2, axis=-1)
        curvature = {
            "max": jnp.max(per_state),
            "batch": jnp.sum(batch * per_state) / jnp.sum(batch),
            "rho": jnp.sum(visits_rho * per_state),
        }[arguments.trust]
        return origin + jnp.sqrt(2.0 * arguments.kl / curvature) * direction

    def batch_of(restarts, share_continuing):
        return occupancy(share_continuing * visits_rho + (1.0 - share_continuing) * restarts, transitions, num_steps)

    def updated(restarts, share_continuing):
        batch = batch_of(restarts, share_continuing)
        return stepped(batch @ per_cell, batch)

    def improvement(restarts, share_continuing):
        return objective(updated(restarts, share_continuing)) - before

    def divergence(restarts, share_continuing):
        moved = log_policy(updated(restarts, share_continuing))
        per_state = jnp.sum(policy * (jnp.log(policy) - moved), axis=-1)
        reached = visits_rho > 1e-6 * jnp.max(visits_rho)
        return float(jnp.sum(visits_rho * per_state)), float(jnp.max(jnp.where(reached, per_state, 0.0))), float(jnp.max(per_state))

    def optimise(share_continuing, initial):
        loss = jax.jit(jax.value_and_grad(lambda logits: -improvement(jax.nn.softmax(logits), share_continuing)))
        optimiser = optax.adam(0.1)
        logits = jnp.log(initial + 1e-6)
        state = optimiser.init(logits)
        best, best_logits = jnp.inf, logits
        for _ in range(arguments.iterations):
            value, gradient = loss(logits)
            best, best_logits = (value, logits) if value < best else (best, best_logits)
            updates, state = optimiser.update(gradient, state)
            logits = optax.apply_updates(logits, updates)
        return jax.nn.softmax(best_logits)

    signal = jnp.sum(policy * game.advantages(policy) ** 2, axis=-1)
    alignment = per_cell @ (visits_rho @ per_cell)
    prd = prd_restarts(selector, podracer.pit.environment, checkpoint, algorithm, game.num_cells, key)
    restarts = {
        "rho": visits_rho,
        "prd": prd,
        "erd": erd_restarts(podracer.pit.environment, checkpoint, game.num_cells),
        "signal oracle": signal / jnp.sum(signal),
        "alignment oracle": jnp.maximum(alignment, 0.0) / jnp.sum(jnp.maximum(alignment, 0.0)),
    }
    candidates = {name: optimise(continuing, initial) for name, initial in restarts.items()}
    optimum = max(candidates.values(), key=lambda restarts: float(improvement(restarts, continuing)))
    unconstrained = optimise(0.0, continuing * visits_rho + (1.0 - continuing) * optimum)

    print(f"J_rho before the update: {before:.4f}; every update moves the policy by KL {arguments.kl} on {dict(max='every single state', batch="its batch's states", rho="rho's states")[arguments.trust]}")
    print(f"{continuing:.0%} of environments continue from rho, the rest restart where the arm says")
    rho = float(improvement(visits_rho, continuing))
    print(f"{'arm':<26} {'J_rho change':>14} {'x rho':>8} {'KL on rho':>10} {'max KL, reached':>16} {'max KL, any':>12}")
    for name, restart in [*restarts.items(), ("optimum", optimum)]:
        change = float(improvement(restart, continuing))
        kl = divergence(restart, continuing)
        print(f"{name:<26} {change:>+14.4f} {change / rho:>8.2f} {kl[0]:>10.4f} {kl[1]:>16.3f} {kl[2]:>12.3f}")
    change = float(improvement(unconstrained, 0.0))
    kl = divergence(unconstrained, 0.0)
    print(f"{'optimum, restarting all':<26} {change:>+14.4f} {change / rho:>8.2f} {kl[0]:>10.4f} {kl[1]:>16.3f} {kl[2]:>12.3f}")

    diagnose(
        selector, podracer.pit.environment, checkpoint, algorithm, objective, origin, unravel, per_cell, optimum, game.num_cells
    )

    trained = load_checkpoint(resolve_checkpoint(run), podracer.init(key)).algorithm_state
    behaviors = {
        "trained behavior": policy_table(algorithm, trained, game, key)[2],
        "optimal behavior": optimal_policy(game, policy),
    }
    true_gradient = jax.grad(objective)(origin)
    sources = {
        "rho batch": batch_of(visits_rho, 1.0),
        "PRD batch": batch_of(prd, continuing),
        **{name: stationary(game.transitions(behavior), game.start) for name, behavior in behaviors.items()},
        "optimal restart batch": batch_of(optimum, continuing),
    }
    directions = {name: (batch @ per_cell, batch) for name, batch in sources.items()}
    directions["true gradient of J_rho"] = (true_gradient, visits_rho)
    advantages = game.advantages(policy)

    def contribution(direction, batch):
        moved = stepped(direction, batch) - origin
        _, change = jax.jvp(log_policy, (origin,), (moved,))
        per_state = visits_rho * jnp.sum(advantages * policy * change, axis=-1)
        return per_state * (true_gradient @ moved) / jnp.sum(per_state), objective(origin + moved) - before

    print(f"gradients, each scaled to the same step (max KL {arguments.kl} per state):")
    print(f"  {'source':<24} {'cos with true gradient':>24} {'J_rho change':>14}")
    maps = {}
    for name, (direction, batch) in directions.items():
        maps[name], change = contribution(direction, batch)
        print(f"  {name:<24} {cosine(direction, true_gradient):>+24.3f} {float(change):>+14.4f}")

    tile = arguments.tile
    start, _ = game.init(key)
    render = np.asarray(game.render(start, tile=tile), np.uint8)
    cell_grid = np.asarray(game.cell_grid)
    num_rows, *_ = render.shape
    gap = np.full((num_rows, tile // 2, 3), GAP, np.uint8)
    shown = (visits_rho, prd, optimum)
    panels = [render, *(paint(np.asarray(values), cell_grid, tile) for values in shown)]
    image = np.concatenate([part for panel in panels for part in (gap, panel)][1:], axis=1)
    path = run / f"optimal_restarts_{step}.png"
    Image.fromarray(image).save(path)
    print(f"render | rho's visits | PRD's restarts | optimal restarts  ->  {path}")

    peak = max(float(jnp.max(jnp.abs(values))) for values in maps.values())
    rows = [("checkpoint", render), *((name, paint(np.asarray(values), cell_grid, tile, peak)) for name, values in maps.items())]
    _, width, _ = render.shape
    strips = []
    for name, panel in rows:
        label = Image.new("RGB", (width, 2 * tile), (GAP, GAP, GAP))
        ImageDraw.Draw(label).text((4, tile // 2), name, fill=(230, 230, 230))
        strips += [np.asarray(label), panel]
    path = run / f"gradients_{step}.png"
    Image.fromarray(np.concatenate(strips, axis=0)).save(path)
    print(f"each panel: how much the step changes J_rho through each cell's policy, on one shared scale  ->  {path}")


if __name__ == "__main__":
    main()
