from pathlib import Path

import pandas as pd
import wandb
from flax import serialization
from joblib import Memory
from tqdm.rich import tqdm

memory = Memory("./.wandb.cache", verbose=0)


@memory.cache
def fetch_wandb(
    entity: str,
    project: str,
    keys: list[str] | None = None,
    filters: dict | None = None,
    samples: int = 500,
) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters, lazy=False)

    histories = pd.DataFrame()
    for run in tqdm(runs):
        df = run.history(
            samples=samples,
            keys=keys,
            pandas=True,
        )
        if df.empty:
            continue

        df["run_id"] = run.id

        histories = pd.concat([histories, df], ignore_index=True)

    if histories.empty:
        raise ValueError("No histories found")

    configs = pd.json_normalize(
        [
            {
                "run_id": run.id,
                "run_name": run.name,
                "group": run.group,
                **run.config,
            }
            for run in runs
        ]
    )

    df = pd.merge(histories, configs, on="run_id", how="left")
    return df


def load_artifact(
    project: str,
    run_id: str,
    version: str = "latest",
    entity: str | None = None,
    target=None,
):
    api = wandb.Api()
    entity = entity or api.default_entity
    artifact = api.artifact(f"{entity}/{project}/model-{run_id}:{version}")
    directory = artifact.download()
    data = (Path(directory) / "model.msgpack").read_bytes()
    if target is None:
        return serialization.msgpack_restore(data)
    return serialization.from_bytes(target, data)
