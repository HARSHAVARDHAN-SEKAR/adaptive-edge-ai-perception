"""Human pose estimation + coarse activity (YOLOv8n-pose)."""
from __future__ import annotations

import numpy as np

from .base import (ModelOutput, PerceptionModel, PoseKeypoints, mock_rng, mock_sleep,
                   mock_world, register)

_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 5, 6, 11, 12


def classify_activity(kp_xy: np.ndarray, kp_conf: np.ndarray) -> str:
    pts = [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP]
    if kp_conf[pts].min() < 0.3:
        return "unknown"
    shoulders = kp_xy[[_L_SHOULDER, _R_SHOULDER]].mean(axis=0)
    hips = kp_xy[[_L_HIP, _R_HIP]].mean(axis=0)
    torso = hips - shoulders
    return "fallen" if abs(torso[0]) > abs(torso[1]) else "upright"


def pose_center(kp_xy: np.ndarray, kp_conf: np.ndarray,
                thr: float = 0.3) -> np.ndarray:
    """Center from VALID keypoints only (low-conf points are excluded)."""
    valid = kp_conf >= thr
    return kp_xy[valid].mean(axis=0) if valid.any() else kp_xy.mean(axis=0)


@register("yolo_pose")
class YoloPose(PerceptionModel):
    task = "pose"
    cost = 2
    weights = "yolov8n-pose.pt"

    def _load_real(self) -> bool:
        from ultralytics import YOLO
        self._model = YOLO(self.weights)
        return True

    def _unload_impl(self) -> None:
        self._model = None

    def _infer_real(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        res = self._model.predict(frame_bgr, verbose=False, device=self.device)[0]
        if res.keypoints is not None:
            for kp, b in zip(res.keypoints, res.boxes):
                xy = kp.xy[0].cpu().numpy()
                cf = (kp.conf[0].cpu().numpy() if kp.conf is not None
                      else np.ones(len(xy)))
                out.poses.append(PoseKeypoints(
                    confidence=float(b.conf), keypoints_xy=xy,
                    keypoints_conf=cf, activity=classify_activity(xy, cf)))
        return out

    def _infer_mock(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        mock_sleep(15.0)
        rng = mock_rng(frame_bgr, self.name)
        for obj in mock_world(frame_bgr):     # skeleton INSIDE the person box
            if not obj.has_pose:
                continue
            x1, y1, x2, y2 = obj.box_xyxy
            cx = (x1 + x2) / 2
            xy = np.stack([np.full(17, cx) + rng.normal(0, (x2 - x1) * 0.08, 17),
                           np.linspace(y1 + 0.05 * (y2 - y1),
                                       y2 - 0.02 * (y2 - y1), 17)], axis=1)
            cf = np.clip(rng.normal(0.8, 0.1, 17), 0.2, 1.0)
            out.poses.append(PoseKeypoints(
                float(np.clip(rng.normal(0.8, 0.08), 0.3, 0.97)),
                xy, cf, classify_activity(xy, cf)))
        return out
