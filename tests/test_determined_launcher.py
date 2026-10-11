import json

import pytest
import yaml

pytest.importorskip("hydra_plugins.hydra_submitit_launcher")

from hydra_plugins.hydra_determined_launcher._launcher import (  # noqa: E402
    DeterminedExecutor, DeterminedInfoWatcher)


def test_the_watcher_describes_only_unfinished_experiments():
    watcher = DeterminedInfoWatcher()
    for ident in ("12", "3", "40"):
        watcher.register_job(ident)
    watcher._finished.add("40")
    assert watcher._make_command() == ["det", "experiment", "describe", "--json", "3,12"]


def test_the_watcher_reads_states_from_describe():
    output = json.dumps(
        [
            {"experiment": {"id": 3, "state": "STATE_ACTIVE"}},
            {"experiment": {"id": 12, "state": "STATE_COMPLETED"}},
            {"experiment": {"id": 40, "state": "STATE_ERROR"}},
            {"experiment": {"id": 41, "state": "STATE_CANCELED"}},
        ]
    )
    states = {ident: info["State"] for ident, info in DeterminedInfoWatcher().read_info(output).items()}
    assert states == {"3": "RUNNING", "12": "COMPLETED", "40": "FAILED", "41": "CANCELLED"}


def test_the_submission_never_restarts_by_default(tmp_path):
    executor = DeterminedExecutor(folder=tmp_path)
    executor.update_parameters(resource_pool="pool", workspace="space", timeout_min=10)
    config = yaml.safe_load(executor._make_submission_file_text("python run.py", "uid"))
    assert config["max_restarts"] == 0
    assert config["workspace"] == "space"
    assert config["resources"] == {"slots_per_trial": 1, "resource_pool": "pool"}
    assert config["entrypoint"] == "timeout -k 60 600 python run.py"
