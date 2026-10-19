import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from boonta.environments.dune_sea import NCA, measure_liveliness, render_field

MIN_ACTIVITY = 0.005
MAX_ACTIVITY = 0.5
MIN_DIVERSITY = 0.2
MIN_STRUCTURE = 0.2


def judge_rules(liveliness: dict) -> np.ndarray:
    activity, diversity, structure = (
        np.asarray(liveliness[name]) for name in ("activity", "diversity", "structure")
    )
    return (
        (activity > MIN_ACTIVITY)
        & (activity < MAX_ACTIVITY)
        & (diversity > MIN_DIVERSITY)
        & (structure > MIN_STRUCTURE)
    )


def survey_rules(args) -> None:
    nca = NCA(
        num_symbols=args.symbols,
        hidden_size=args.hidden_size,
        temperature=args.temperature,
        init_gain=args.init_gain,
    )

    def play_rule(key):
        rule_key, field_key, step_key = jax.random.split(key, 3)
        field = nca.seed_field(field_key, args.size, args.size)
        trajectory = nca.run_rollout(nca.draw_rule(rule_key), field, step_key, args.steps)
        return trajectory, measure_liveliness(trajectory, args.symbols)

    def score_rule(key):
        return play_rule(key)[1]

    keys = jax.vmap(jax.random.key)(jnp.arange(args.first_rule, args.first_rule + args.rules))
    liveliness = jax.jit(lambda keys: jax.lax.map(score_rule, keys, batch_size=args.batch))(keys)
    lively = judge_rules(liveliness)
    print(f"{lively.sum()} of {args.rules} rules are lively ({100 * lively.mean():.1f}%)")

    ranking = np.asarray(liveliness["structure"]) * np.asarray(liveliness["diversity"])
    chosen = [int(index) for index in np.argsort(-np.where(lively, ranking, -np.inf))[: args.show]]
    chosen = [index for index in chosen if lively[index]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = np.linspace(0, args.steps - 1, args.frames).astype(int)
    replay = jax.jit(play_rule)
    rows = []
    for index in chosen:
        trajectory, _ = replay(keys[index])
        rule_id = args.first_rule + index
        print(
            f"rule {rule_id:5d}  activity {float(liveliness['activity'][index]):.3f}  "
            f"diversity {float(liveliness['diversity'][index]):.2f}  "
            f"structure {float(liveliness['structure'][index]):.2f}"
        )
        frames = [np.asarray(render_field(trajectory[t], args.symbols, args.scale)) for t in snapshots]
        rows.append(np.concatenate(frames, axis=1))
        if args.gif:
            stride = max(args.steps // 128, 1)
            movie = [
                Image.fromarray(np.asarray(render_field(frame, args.symbols, args.scale)))
                for frame in jnp.asarray(trajectory)[::stride]
            ]
            movie[0].save(
                output / f"rule_{rule_id}.gif", save_all=True, append_images=movie[1:], duration=40, loop=0
            )
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(output / "survey.png")
        print(f"wrote {output / 'survey.png'}")


def main():
    parser = argparse.ArgumentParser(description="Search random dune_sea NCA rules for lively ones.")
    parser.add_argument("--rules", type=int, default=2048)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--first-rule", type=int, default=0)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--symbols", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--init-gain", type=float, default=3.0)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--output", default="outputs/dune_sea")
    survey_rules(parser.parse_args())


if __name__ == "__main__":
    main()
