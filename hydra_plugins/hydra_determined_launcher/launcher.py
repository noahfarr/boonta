try:
    from hydra_plugins.hydra_determined_launcher._launcher import (
        DeterminedExecutor,
        DeterminedInfoWatcher,
        DeterminedJob,
        DeterminedJobEnvironment,
        DeterminedSubmititLauncher,
    )
except ModuleNotFoundError as error:
    if error.name not in {"submitit", "hydra_plugins.hydra_submitit_launcher"}:
        raise
