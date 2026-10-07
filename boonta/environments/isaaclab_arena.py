from .isaaclab import IsaacLab, launch
from .wrappers import Batched


def make(
    env_id,
    task,
    embodiment: str = "franka_ik",
    assets=(),
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = True,
    enable_cameras: bool = False,
    **kwargs,
):
    launch(device=device, headless=headless, enable_cameras=enable_cameras)

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import (
        ArenaEnvBuilder, ArenaEnvBuilderCfg)
    from isaaclab_arena.environments.isaaclab_arena_environment import \
        IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    asset_registry = AssetRegistry()

    def resolve(asset):
        if isinstance(asset, str):
            return asset_registry.get_asset_by_name(asset)()
        return asset

    arena_environment = IsaacLabArenaEnvironment(
        name=env_id,
        embodiment=resolve(embodiment),
        scene=Scene([resolve(asset) for asset in assets]),
        task=task,
        **kwargs,
    )
    builder_cfg = ArenaEnvBuilderCfg(num_envs=num_envs, device=device)
    env = ArenaEnvBuilder(arena_environment, builder_cfg).make_registered()

    env = IsaacLab(env)
    return Batched(env, num_envs=env.num_envs)
