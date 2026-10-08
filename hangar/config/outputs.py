from boonta.artisans import Checkpointer, Render, Transcript
from boonta.loggers import DashboardLogger, FileLogger, OrbaxLogger, WandbLogger

from .store import builds, fbuilds, store

logger = store(group="logger")
artisan = store(group="artisan")

logger(
    fbuilds(
        DashboardLogger,
        total_timesteps="${total_timesteps}",
        refresh_per_second=1,
        summary={
            "Algorithm": "${hydra:runtime.choices.algorithm}",
            "Environment": "${environment.env_id}",
            "Total Timesteps": "${total_timesteps}",
            "Seed": "${seed}",
        },
    ),
    name="dashboard",
    package="loggers.dashboard",
)
logger(
    fbuilds(FileLogger, directory="${hydra:runtime.output_dir}", filename="metrics.npz"),
    name="file",
    package="loggers.file",
)
logger(
    fbuilds(OrbaxLogger, directory="${hydra:runtime.output_dir}/checkpoints", max_to_keep=100),
    name="orbax",
    package="loggers.orbax",
)
logger(
    fbuilds(
        WandbLogger,
        project="reinforcement-learning",
        name="${hydra:runtime.choices.algorithm}_${environment.env_id}_${seed}",
        mode="online",
        directory="${hydra:runtime.output_dir}",
        seed="${seed}",
        zen_exclude=("cfg",),
    ),
    name="wandb",
    package="loggers.wandb",
)

artisan(builds(Checkpointer), name="checkpointer", package="artisans.checkpointer")
artisan(fbuilds(Render, name="render", max_frames=256, fps=16), name="render", package="artisans.render")
artisan(
    fbuilds(Transcript, repo_id="Qwen/Qwen3-0.6B-Base"),
    name="transcript",
    package="artisans.generation",
)
