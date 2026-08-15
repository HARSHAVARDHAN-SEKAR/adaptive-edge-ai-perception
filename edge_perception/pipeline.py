"""Adaptive Edge AI Perception Pipeline.

Per frame:
    frame -> [scheduler.decide] -> run selected models -> fuse -> uncertainty
          -> emit SceneUnderstanding + telemetry

Models are loaded lazily and kept warm once loaded (loading is the expensive
part; on Jetson a TensorRT engine load can take seconds, so we never unload
inside a mission unless memory pressure demands it).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .fusion.perception_fusion import PerceptionFusion, SceneUnderstanding
from .fusion.uncertainty import scene_confidence
from .models import base as model_base
from .models import detection, segmentation, depth, pose, vlm  # noqa: F401 (registers models)
from .scheduler.model_selector import AdaptiveScheduler, Decision, SchedulerConfig, Mode
from .scheduler.resource_monitor import ResourceMonitor


@dataclass
class FrameResult:
    frame_index: int
    scene: SceneUnderstanding
    decision: Decision
    fps: float
    total_latency_ms: float
    model_latencies: Dict[str, float]
    decision_confidence: Optional[float]
    narration: Optional[str] = None       # last VLM scene narration
    skipped: bool = False                  # DEGRADED frame-skip reuse

    def to_dict(self):
        return {
            "frame": self.frame_index,
            "mode": self.decision.mode.value,
            "models": self.decision.models,
            "switched": self.decision.switched,
            "reason": self.decision.reason,
            "fps": round(self.fps, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "model_latencies_ms": {k: round(v, 2)
                                   for k, v in self.model_latencies.items()},
            "decision_confidence": (round(self.decision_confidence, 3)
                                    if self.decision_confidence is not None
                                    else None),
            "narration": self.narration,
            "skipped": self.skipped,
            "backends": self.scene.backends_used,
            "scene": self.scene.to_dict(),
        }


class PerceptionPipeline:
    def __init__(self,
                 scheduler_config: Optional[SchedulerConfig] = None,
                 device: str = "cpu",
                 backend: str = "auto",
                 adaptive: bool = True,
                 fixed_models: Optional[List[str]] = None,
                 log_path: Optional[str] = None,
                 vlm_on_engagement: bool = True,
                 degraded_frame_skip: int = 2):
        self.scheduler = AdaptiveScheduler(scheduler_config)
        self.monitor = ResourceMonitor()
        self.fusion = PerceptionFusion()
        self.device, self.backend = device, backend
        self.adaptive = adaptive
        self.fixed_models = fixed_models or ["yolo_large", "yolo_nano_seg",
                                             "midas_small", "yolo_pose"]
        self._models: Dict[str, model_base.PerceptionModel] = {}
        self._log = open(log_path, "w") if log_path else None
        self._last_risk = 0.0
        self._last_min_conf = 0.0
        # On-demand VLM: narrates the scene once on escalation to ENGAGED.
        # Narration has an absolute TTL so it cannot describe a stale scene
        # indefinitely if the scheduler remains ENGAGED.
        self.vlm_on_engagement = vlm_on_engagement
        self.narration: Optional[str] = None
        self._narration_frame: Optional[int] = None
        self.narration_ttl_frames = 40      # absolute TTL from narration frame
        # improvement: in DEGRADED mode, run inference only every Nth frame
        # and reuse the last scene in between (thermal/compute rescue)
        self.degraded_frame_skip = max(1, degraded_frame_skip)
        self._skip_counter = 0
        self._last_result: Optional[FrameResult] = None

    # ------------------------------------------------------------------
    def _get_model(self, name: str) -> model_base.PerceptionModel:
        if name not in self._models:
            m = model_base.create(name, device=self.device, backend=self.backend)
            m.load()
            self._models[name] = m
        return self._models[name]

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        t0 = time.perf_counter()
        resources = self.monitor.snapshot()

        if self.adaptive:
            decision = self.scheduler.decide(self._last_risk,
                                             self._last_min_conf, resources)
        else:
            decision = Decision(Mode.ENGAGED, list(self.fixed_models),
                                "fixed baseline", False, time.time())

        # -- DEGRADED frame-skip: reuse last scene every Nth frame -----------
        if (decision.mode == Mode.DEGRADED and self._last_result is not None
                and self.degraded_frame_skip > 1):
            self._skip_counter += 1
            if self._skip_counter % self.degraded_frame_skip != 0:
                self.monitor.tick()
                prev = self._last_result
                result = FrameResult(
                    frame_index=frame_index, scene=prev.scene,
                    decision=decision, fps=self.monitor.fps,
                    total_latency_ms=(time.perf_counter() - t0) * 1000.0,
                    model_latencies={}, decision_confidence=prev.decision_confidence,
                    narration=self.narration, skipped=True)
                if self._log:
                    self._log.write(json.dumps(result.to_dict()) + "\n")
                return result
        else:
            self._skip_counter = 0

        outputs, latencies = [], {}
        for name in decision.models:
            out = self._get_model(name)(frame)
            outputs.append(out)
            latencies[name] = out.latency_ms

        scene = self.fusion.fuse(outputs, frame.shape)
        conf = scene_confidence(scene)
        self._last_risk = scene.scene_risk
        self._last_min_conf = scene.min_detection_conf

        # -- on-demand VLM: narrate once per escalation to ENGAGED -----------
        if (self.vlm_on_engagement and decision.switched
                and decision.mode == Mode.ENGAGED):
            try:
                vlm_out = self._get_model("vlm_scene")(frame)
                self.narration = vlm_out.extra.get("description")
                self._narration_frame = frame_index
                latencies["vlm_scene"] = vlm_out.latency_ms
            except Exception:
                # The optional VLM must never take down core perception.
                # backend='real' remains strict when the VLM is called directly;
                # in the pipeline narration is best-effort and explicitly optional.
                self.narration = None
                self._narration_frame = None

        # Absolute TTL: stale narration is misleading even if the scheduler
        # remains ENGAGED because risk is still elevated.
        if (self.narration is not None
                and self._narration_frame is not None
                and frame_index - self._narration_frame
                > self.narration_ttl_frames):
            self.narration = None
            self._narration_frame = None

        self.monitor.tick()
        result = FrameResult(
            frame_index=frame_index,
            scene=scene,
            decision=decision,
            fps=self.monitor.fps,
            total_latency_ms=(time.perf_counter() - t0) * 1000.0,
            model_latencies=latencies,
            decision_confidence=conf,
            narration=self.narration,
        )
        self._last_result = result
        if self._log:
            self._log.write(json.dumps(result.to_dict()) + "\n")
        return result

    # ------------------------------------------------------------------
    def run(self, frames: Iterable[np.ndarray],
            max_frames: Optional[int] = None,
            on_result=None) -> List[FrameResult]:
        results = []
        for i, frame in enumerate(frames):
            if max_frames is not None and i >= max_frames:
                break
            r = self.process(frame, i)
            results.append(r)
            if on_result:
                on_result(r)
        return results

    def close(self):
        if self._log:
            self._log.close()
        self.monitor.close()
