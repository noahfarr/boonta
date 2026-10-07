import argparse
import time
from pathlib import Path

import hydra
import jax
import tqdx
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from boonta.utils.checkpoint import load_checkpoint, newest
from boonta.utils.wandb import load_artifact
from hangar import recipes


def compose(overrides):
    with hydra.initialize(version_base=None, config_path="../config"):
        cfg = hydra.compose(
            config_name="config", overrides=list(overrides), return_hydra_config=True
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def resolve(run, num_envs):
    path = Path(run)
    if path.is_dir():
        overrides = next(path.glob("**/.hydra/overrides.yaml"), None)
        if overrides is None:
            raise FileNotFoundError(f"No .hydra/overrides.yaml found under {path}")
        cfg = compose([*OmegaConf.load(overrides), f"environment.num_envs={num_envs}"])
        checkpoint = Path(newest(path)) / "algorithm_state"
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"No algorithm_state checkpoint under {path}; train with "
                "+artisan=[checkpointer] logger=orbax"
            )
        return cfg, lambda: {"algorithm_state": load_checkpoint(checkpoint)}

    wandb_run = wandb.Api().run(run)
    cfg = compose([*wandb_run.metadata["args"], f"environment.num_envs={num_envs}"])
    return cfg, lambda: load_artifact(wandb_run.project, wandb_run.id, entity=wandb_run.entity)


def main(run, seed, num_steps, num_envs):
    cfg, load_fields = resolve(run, num_envs)
    podracer = recipes.make(cfg)
    algorithm, environment = podracer.algorithm, podracer.environment

    init_key, reset_key = jax.random.split(jax.random.key(seed))
    _, timestep = environment.init(reset_key)
    state = algorithm.init(init_key, timestep)
    state = state.replace(params=load_fields()["algorithm_state"]["params"])

    @jax.jit
    def rollout(carry, params, timestep):
        sequence = timestep.to_sequence()

        def step(carry, _):
            return algorithm.network.apply(
                params,
                sequence.obs,
                sequence.action,
                sequence.reward,
                sequence.done,
                carry=carry,
                temperature=1.0,
            )

        return tqdx.scan(step, carry, None, length=num_steps)

    carry, logits = jax.block_until_ready(
        rollout(state.carry, state.params, timestep)
    )

    start = time.monotonic()
    carry, _logits = jax.block_until_ready(rollout(carry, state.params, timestep))
    elapsed = time.monotonic() - start
    print(f"inference/SPS: {num_steps * num_envs / elapsed:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run", help="hydra run directory, or an entity/project/run_id wandb path"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=128)
    args = parser.parse_args()
    main(args.run, args.seed, args.num_steps, args.num_envs)
