from __future__ import annotations

import json
import os
import re
import shlex
import typing as tp
from pathlib import Path

import yaml
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher
from submitit.core import core, job_environment, utils

RUNNING = {
    "QUEUED",
    "STARTING",
    "PULLING",
    "ACTIVE",
    "RUNNING",
    "PAUSED",
    "STOPPING_COMPLETED",
    "STOPPING_CANCELED",
    "STOPPING_ERROR",
    "STOPPING_KILLED",
}
FINISHED = {"COMPLETED": "COMPLETED", "ERROR": "FAILED", "CANCELED": "CANCELLED"}


class DeterminedInfoWatcher(core.InfoWatcher):
    def _make_command(self) -> tp.Optional[tp.List[str]]:
        pending = sorted(self._registered - self._finished, key=int)
        if not pending:
            return None
        return ["det", "experiment", "describe", "--json", ",".join(pending)]

    def read_info(self, string: tp.Union[bytes, str]) -> tp.Dict[str, tp.Dict[str, str]]:
        if isinstance(string, bytes):
            string = string.decode()
        info: tp.Dict[str, tp.Dict[str, str]] = {}
        for entry in json.loads(string or "[]"):
            experiment = entry.get("experiment", {})
            ident = str(experiment.get("id", ""))
            if not ident:
                continue
            state = self.translate(str(experiment.get("state", "UNKNOWN")))
            info[ident] = {"JobID": ident, "State": state}
        return info

    @staticmethod
    def translate(state: str) -> str:
        state = state.upper().removeprefix("STATE_")
        if state in RUNNING:
            return "RUNNING"
        return FINISHED.get(state, "UNKNOWN")

    def get_state(self, job_id: str, mode: str = "standard") -> str:
        info = self.get_info(job_id, mode=mode)
        return info.get("State") or "UNKNOWN"


class DeterminedJobEnvironment(job_environment.JobEnvironment):
    _env = {
        "job_id": "DET_EXPERIMENT_ID",
        "num_tasks": "SUBMITIT_DETERMINED_NUM_TASKS",
        "num_nodes": "SUBMITIT_DETERMINED_NUM_NODES",
        "node": "SUBMITIT_DETERMINED_NODE",
        "global_rank": "SUBMITIT_DETERMINED_RANK",
        "local_rank": "SUBMITIT_DETERMINED_LOCAL_RANK",
    }

    def _requeue(self, countdown: int) -> None:
        pass


class DeterminedJob(core.Job[core.R]):
    _cancel_command = "det experiment kill"
    watcher = DeterminedInfoWatcher(delay_s=30)

    def _interrupt(self, timeout: bool = False) -> None:
        self.cancel(check=False)


class DeterminedExecutor(core.PicklingExecutor):
    job_class = DeterminedJob

    @classmethod
    def name(cls) -> str:
        return "determined"

    @classmethod
    def affinity(cls) -> int:
        return 2 if utils.CommandFunction(["which", "det"], verbose=False).command else -1

    @classmethod
    def _equivalence_dict(cls) -> tp.Dict[str, str]:
        return {
            "name": "name",
            "timeout_min": "timeout_min",
            "mem_gb": "mem_gb",
            "nodes": "nodes",
            "cpus_per_task": "cpus_per_task",
            "gpus_per_node": "slots_per_trial",
            "tasks_per_node": "tasks_per_node",
            "stderr_to_stdout": "stderr_to_stdout",
        }

    @classmethod
    def _valid_parameters(cls) -> tp.Set[str]:
        return {
            "name",
            "workspace",
            "project",
            "resource_pool",
            "priority",
            "slots_per_trial",
            "image",
            "environment_variables",
            "bind_mounts",
            "max_restarts",
            "searcher",
            "extra_config",
            "context_dir",
            "python",
            "timeout_min",
            "mem_gb",
            "nodes",
            "cpus_per_task",
            "tasks_per_node",
            "stderr_to_stdout",
        }

    def _num_tasks(self) -> int:
        return 1

    def _make_submission_file_text(self, command: str, uid: str) -> str:
        params = dict(self.parameters)
        timeout_min = params.get("timeout_min")
        entrypoint = (
            command
            if not timeout_min
            else f"timeout -k 60 {int(timeout_min) * 60} {command}"
        )
        config: tp.Dict[str, tp.Any] = {
            "name": params.get("name", f"submitit_{uid}"),
            "entrypoint": entrypoint,
            "max_restarts": params.get("max_restarts", 0),
            "searcher": params.get(
                "searcher", {"name": "single", "metric": "submitit", "max_length": {"batches": 1}}
            ),
            "resources": {"slots_per_trial": params.get("slots_per_trial", 1)},
        }
        for key in ("workspace", "project"):
            if params.get(key):
                config[key] = params[key]
        if params.get("resource_pool"):
            config["resources"]["resource_pool"] = params["resource_pool"]
        if params.get("priority") is not None:
            config["resources"]["priority"] = params["priority"]
        exported = [
            "SUBMITIT_EXECUTOR=determined",
            "SUBMITIT_DETERMINED_NUM_TASKS=1",
            "SUBMITIT_DETERMINED_NUM_NODES=1",
            "SUBMITIT_DETERMINED_NODE=0",
            "SUBMITIT_DETERMINED_RANK=0",
            "SUBMITIT_DETERMINED_LOCAL_RANK=0",
        ]
        environment: tp.Dict[str, tp.Any] = {
            "environment_variables": exported + list(params.get("environment_variables") or [])
        }
        if params.get("image"):
            environment["image"] = params["image"]
        config["environment"] = environment
        if params.get("bind_mounts"):
            config["bind_mounts"] = list(params["bind_mounts"])
        merge(config, params.get("extra_config") or {})
        return yaml.safe_dump(config, sort_keys=False, width=100000, default_flow_style=False)

    def _make_submission_command(self, submission_file_path: Path) -> tp.List[str]:
        context = self.parameters.get("context_dir") or str(self.folder)
        Path(context).mkdir(parents=True, exist_ok=True)
        return ["det", "experiment", "create", str(submission_file_path), str(context)]

    @staticmethod
    def _get_job_id_from_submission_command(string: tp.Union[bytes, str]) -> str:
        if isinstance(string, bytes):
            string = string.decode()
        found = re.search(r"[Cc]reated experiment (?P<id>[0-9]+)", string)
        if found is None:
            raise utils.FailedSubmissionError(
                f"could not read experiment id from det output: {string}"
            )
        return found.group("id")

    @property
    def _submitit_command_str(self) -> str:
        python = self.parameters.get("python") or shlex.quote(os.sys.executable)
        return " ".join([python, "-u -m submitit.core._submit", shlex.quote(str(self.folder))])


class DeterminedSubmititLauncher(BaseSubmititLauncher):
    _EXECUTOR = "determined"


def merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            merge(dst[key], value)
        else:
            dst[key] = value
    return dst
