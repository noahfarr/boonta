import atexit
import signal
import sys
from collections import defaultdict
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table

from boonta.artisans import Text
from boonta.utils import PyTree

from .logger import Primary


def uncover() -> None:
    stream = sys.__stdout__
    if stream is not None and not stream.closed:
        stream.write("\x1b[?25h")
        stream.flush()


class DashboardLogger(Primary):
    def __init__(
        self,
        total_timesteps=0,
        refresh_per_second=10,
        summary=None,
        max_rows=12,
        **kwargs,
    ):
        self.summary = summary or {}
        self.max_rows = max_rows
        self.generations = {}
        self._metrics = {}
        self._step = 0

        self.console = Console()

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TimeRemainingColumn(),
            expand=True,
            console=self.console,
        )
        self.progress_task = self.progress.add_task("Progress", total=total_timesteps)

        dashboard = self.build_dashboard({}, 0, self.progress, self.progress_task)

        self.live = Live(
            dashboard,
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=False,
        )
        self.live.start()
        atexit.register(uncover)
        for received in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(received)
            signal.signal(received, self.interrupted(previous))

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        step = int(np.asarray(steps).max())
        metrics = {
            k: np.array([np.asarray(row).mean() for row in v], dtype=np.float64)
            for k, v in data.items()
        }
        self._metrics, self._step = metrics, step
        self.progress.update(self.progress_task, completed=step)
        dashboard = self.build_dashboard(
            metrics, step, self.progress, self.progress_task
        )
        self.live.update(dashboard, refresh=True)

    def log_artifact(self, artifact, step: int, **kwargs) -> None:
        if isinstance(artifact, Text):
            self.generations[artifact.name] = artifact.data
            self.live.update(
                self.build_dashboard(
                    self._metrics, self._step, self.progress, self.progress_task
                ),
                refresh=True,
            )

    def log_summary(self, data: PyTree, **kwargs) -> None:
        pass

    def interrupted(self, previous):
        def handle(received, frame):
            uncover()
            if callable(previous):
                previous(received, frame)
            elif previous == signal.SIG_IGN:
                return
            elif received == signal.SIGINT:
                raise KeyboardInterrupt
            else:
                raise SystemExit(128 + received)

        return handle

    def finish(self) -> None:
        self.live.stop()
        self.console.show_cursor(True)
        uncover()
        atexit.unregister(uncover)

    def group(self, data: dict[str, PyTree]) -> dict[str, dict[str, Any]]:
        groups = defaultdict(dict)
        for key, value in data.items():
            if "/" in key:
                prefix, name = key.split("/", 1)
                groups[prefix][name] = value
            else:
                groups[""][key] = value
        return dict(groups)

    def build_table(self, heading: str, metrics: dict[str, PyTree]) -> Table:
        table = Table(box=None, expand=True)
        table.add_column(heading, justify="left", width=20, style="yellow")
        table.add_column("Value", justify="right", width=10, style="green")
        items = list(metrics.items())
        cap = self.max_rows
        hidden = 0
        if cap and len(items) > cap:
            items = sorted(items, key=lambda kv: -abs(float(np.mean(kv[1]))))
            hidden = len(items) - (cap - 1)
            items = items[: cap - 1]
        for name, value in items:
            mean, std = np.mean(value), np.std(value)
            if 0 < abs(mean) < 0.001:
                fmt = ".3e"
            elif abs(mean) >= 10000:
                fmt = "_.0f"
            else:
                fmt = ".3f"
            value_str = f"{mean:{fmt}} ± {std:{fmt}}" if std != 0 else f"{mean:{fmt}}"
            table.add_row(name, value_str)
        if hidden:
            table.add_row(f"(+{hidden} more)", "", style="dim")
        return table

    def preview(self, text: str, head: int = 6, tail: int = 6) -> str:
        words = text.split()
        if len(words) <= head + tail:
            return " ".join(words)
        return " ".join(words[:head]) + " … " + " ".join(words[-tail:])

    def build_artifacts(self) -> Table:
        table = Table(box=None, expand=True)
        table.add_column("Artifact", justify="left", width=20, style="cyan")
        table.add_column("Preview", justify="left", style="white", overflow="ellipsis")
        for name, text in self.generations.items():
            table.add_row(name, self.preview(text))
        return table

    def build_dashboard(
        self, data: dict[str, PyTree], step: int, progress: Progress, task: Any
    ) -> Table:
        dashboard = Table(
            box=box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="white",
        )

        dynamic_summary = {
            k.split("/", 1)[1]: v for k, v in data.items() if k.startswith("summary/")
        }
        items = [*self.summary.items(), *dynamic_summary.items()]
        if data:
            items.append(("Step", f"{int(step):_}"))
        left = Table(box=None, expand=True)
        left.add_column("Summary", justify="left", width=16, style="white")
        left.add_column("Value", justify="right", width=8, style="white")
        right = Table(box=None, expand=True)
        right.add_column("Summary", justify="left", width=16, style="white")
        right.add_column("Value", justify="right", width=8, style="white")
        for i, (key, value) in enumerate(items):
            table = left if i % 2 == 0 else right
            value_str = f"{value:_}" if isinstance(value, int) else f"{value}"
            table.add_row(key, value_str, style="white")
        summary_row = Table(box=None, expand=True, pad_edge=False)
        summary_row.add_row(left, right)
        dashboard.add_row(summary_row)

        groups = self.group(data)
        groups.pop("summary", None)
        group_names = list(groups.keys())

        for i in range(0, len(group_names), 2):
            pair = group_names[i : i + 2]
            tables = [self.build_table(name, groups[name]) for name in pair]
            row = Table(box=None, expand=True, pad_edge=False)
            row.add_row(*tables)
            dashboard.add_row(row)

        if self.generations:
            dashboard.add_row(self.build_artifacts())

        dashboard.add_row("")
        progress.update(task, completed=int(step))
        dashboard.add_row(progress)

        return dashboard
