"""Heuristic decision confidence (NOT calibrated statistical uncertainty).

Combines per-source heuristic confidences into one score a planner can gate
on. These values are hand-assigned heuristics; calibrated uncertainty would
require ensembles / MC dropout and ECE/NLL/Brier evaluation — listed as
future work in the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .perception_fusion import FusedObject, SceneUnderstanding

SOURCE_WEIGHTS = {"detection": 0.4, "depth": 0.3, "tracking": 0.2, "pose": 0.1}


@dataclass
class ObjectConfidence:
    object_id: int
    detection: float
    depth: float
    tracking: float
    pose: float
    decision_confidence: float          # heuristic, uncalibrated

    def to_dict(self):
        return {k: round(v, 3) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def _geo_mean(values: Dict[str, float]) -> float:
    logs, wsum = 0.0, 0.0
    for k, v in values.items():
        w = SOURCE_WEIGHTS[k]
        logs += w * np.log(max(v, 1e-3))
        wsum += w
    return float(np.exp(logs / wsum))


def estimate(obj: FusedObject, depth_available: bool) -> ObjectConfidence:
    detection = obj.confidence
    if obj.rel_depth is None:
        depth = 0.35 if not depth_available else 0.5
    else:  # relative depth degrades with range
        depth = float(np.clip(0.95 - 0.6 * obj.rel_depth, 0.3, 0.95))
    tracking = 0.9 if obj.velocity_px_s is not None else 0.5
    pose = 0.85 if obj.activity not in ("n/a", "unknown") else 0.5
    return ObjectConfidence(
        obj.object_id, detection, depth, tracking, pose,
        _geo_mean({"detection": detection, "depth": depth,
                   "tracking": tracking, "pose": pose}))


def scene_confidence(scene: SceneUnderstanding) -> Optional[float]:
    depth_avail = scene.depth_scale is not None
    confs = [estimate(o, depth_avail).decision_confidence
             for o in scene.objects]
    return float(np.mean(confs)) if confs else None
