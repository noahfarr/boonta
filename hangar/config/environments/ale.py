from ..sections import Environment
from . import environment

environment(
    Environment(
        namespace="ale",
        suite="ale",
        env_id="montezuma_revenge",
        num_envs=128,
        kwargs=dict(
            num_envs="${environment.num_envs}",
            frame_skip=4,
            num_threads=28,
            fraction=0.0,
            capacity=4096,
            depth=8,
            warmup_episodes=1024,
            novelty=0.0,
            snapshot_every=0,
            snapshot="",
        ),
    ),
    name="ale/montezuma",
)
