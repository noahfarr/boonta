import time

import jax
import numpy as np
import psutil
import pynvml


class SystemMonitor:
    def __init__(self):
        self.compilation_time = 0.0
        self.time = time.monotonic()
        self.step = 0
        self.work = 0.0
        jax.monitoring.register_event_duration_secs_listener(self.record_compilation)
        try:
            pynvml.nvmlInit()
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            self.gpu = None
        self.device = jax.local_devices()[0]

    def start(self):
        self.mark = time.monotonic()

    def stop(self):
        self.work += time.monotonic() - self.mark

    def record_compilation(self, event: str, duration: float, **kwargs) -> None:
        if event.startswith("/jax/core/compile/"):
            self.compilation_time += duration

    def metrics(self, step: int) -> dict[str, np.ndarray]:
        now = time.monotonic()
        memory_stats = self.device.memory_stats() or {}
        data = {
            "monitor/compilation_time": self.compilation_time,
            "monitor/SPS": (step - self.step) / (now - self.time),
            "monitor/overhead": max(now - self.time - self.work, 0.0),
            "cpu/utilization": psutil.cpu_percent() / 100,
            "cpu/memory_consumption": psutil.virtual_memory().used / 2**30,
            "gpu/memory_consumption": memory_stats.get("bytes_in_use", 0) / 2**30,
        }
        if self.gpu is not None:
            data["gpu/utilization"] = (
                pynvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu / 100
            )
            data["gpu/power_consumption"] = (
                pynvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000
            )
        self.time, self.step, self.work = now, step, 0.0
        return {k: np.full((1, 1), v) for k, v in data.items()}
