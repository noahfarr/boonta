from . import algorithms, data, environments, networks, outputs, podracers, sweeps
from .sections import Evaluation, Network, Training
from .store import store

store(
    dict(
        defaults=[
            "_self_",
            {"podracer": "anakin"},
            {"algorithm": "ppo"},
            {"environment": "gymnax/minatar/breakout"},
            {"logger": "dashboard"},
            {"optional hyperparameters": "${hyperparameters:${algorithm},${environment}}"},
            {"curriculum": "default"},
            {"scoring": "best"},
            {"optional search_space": "${search_space:${oc.select:sweep,null},${algorithm},${environment}}"},
        ],
        total_timesteps=5_000_000,
        seed="${trial:}",
        training=Training(),
        evaluation=Evaluation(),
        network=Network(),
        score="${eval:'\"evaluation\" if ${evaluation.num_steps} else \"training\"'}/episode_return",
        loggers={},
        artisans={},
        hydra=dict(
            run=dict(dir="outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}-${oc.env:SLURM_ARRAY_TASK_ID,0}"),
            sweep=dict(
                dir="sweeps/${hydra.runtime.choices.algorithm}/${hydra.runtime.choices.environment}/${now:%Y-%m-%d_%H-%M-%S}"
            ),
        ),
    ),
    name="config",
)

store.add_to_hydra_store(overwrite_ok=True)
