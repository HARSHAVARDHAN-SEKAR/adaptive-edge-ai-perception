"""Runtime resource telemetry feeding the adaptive scheduler.

Sources, in order of preference:
  1. NVIDIA NVML (desktop/laptop GPUs)          -> gpu util, vram, power
  2. Jetson tegrastats / jetson-stats (jtop)    -> gpu util, power rails
  3. psutil                                     -> cpu, ram (always available)

Also tracks a rolling FPS estimate from pipeline ticks — FPS is the single
most important signal for the scheduler ("are we keeping up?").
"""

from __future__ import annotations

import collections
import time
from dataclasses import asdict, dataclass

try:
    import psutil

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    _HAS_PSUTIL = False

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
except Exception:  # pragma: no cover
    _HAS_NVML = False
    _NVML_HANDLE = None

try:
    from jtop import jtop as _jtop  # jetson-stats, Jetson only

    _HAS_JTOP = True
except Exception:  # pragma: no cover
    _HAS_JTOP = False


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    ram_percent: float
    gpu_percent: float  # -1 if unavailable
    gpu_mem_percent: float  # -1 if unavailable
    power_watts: float  # -1 if unavailable
    fps: float  # rolling pipeline FPS
    source: str

    def to_dict(self):
        return asdict(self)


class ResourceMonitor:
    def __init__(self, fps_window: int = 30):
        self._tick_times: collections.deque[float] = collections.deque(
            maxlen=fps_window
        )
        self._jtop_ctx = None
        if _HAS_JTOP:
            try:
                self._jtop_ctx = _jtop()
                self._jtop_ctx.start()
            except Exception:
                self._jtop_ctx = None

    # -- pipeline hooks ------------------------------------------------------
    def tick(self) -> None:
        """Call once per processed frame."""
        self._tick_times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._tick_times) < 2:
            return 0.0
        span = self._tick_times[-1] - self._tick_times[0]
        return (len(self._tick_times) - 1) / span if span > 0 else 0.0

    # -- snapshot --------------------------------------------------------------
    def snapshot(self) -> ResourceSnapshot:
        cpu = psutil.cpu_percent() if _HAS_PSUTIL else -1.0
        ram = psutil.virtual_memory().percent if _HAS_PSUTIL else -1.0
        gpu = gmem = pwr = -1.0
        source = "psutil"

        if _HAS_NVML:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                gpu = float(util.gpu)
                gmem = 100.0 * mem.used / mem.total
                pwr = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0
                source = "nvml"
            except Exception:
                pass
        elif self._jtop_ctx is not None and self._jtop_ctx.ok():
            try:
                stats = self._jtop_ctx.stats
                gpu = float(stats.get("GPU", -1))
                pwr = float(stats.get("Power TOT", -1)) / 1000.0
                source = "jtop"
            except Exception:
                pass

        return ResourceSnapshot(time.time(), cpu, ram, gpu, gmem, pwr, self.fps, source)

    def close(self) -> None:
        if self._jtop_ctx is not None:
            try:
                self._jtop_ctx.close()
            except Exception:
                pass
