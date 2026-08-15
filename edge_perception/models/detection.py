"""Object detection tier: YOLO nano / small / large (Ultralytics)."""
from __future__ import annotations

import numpy as np

from .base import (Detection, ModelOutput, PerceptionModel, mock_rng, mock_sleep,
                   mock_world, register)


class _YoloBase(PerceptionModel):
    task = "detect"
    weights: str = "yolov8n.pt"
    _mock_latency_ms: float = 12.0
    _mock_conf: float = 0.70
    _mock_miss_far: float = 0.0     # chance to miss far/small objects (tiers)

    def _load_real(self) -> bool:
        from ultralytics import YOLO
        self._model = YOLO(self.weights)
        return True

    def _unload_impl(self) -> None:
        self._model = None

    def _infer_real(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        res = self._model.predict(frame_bgr, verbose=False, device=self.device)[0]
        names = res.names
        for b in res.boxes:
            out.detections.append(Detection(
                label=names[int(b.cls)],
                confidence=float(b.conf),
                box_xyxy=tuple(float(v) for v in b.xyxy[0].tolist())))
        return out

    def _infer_mock(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        mock_sleep(self._mock_latency_ms)
        rng = mock_rng(frame_bgr, self.name)
        for i, obj in enumerate(mock_world(frame_bgr)):
            # lighter tiers occasionally miss far objects — the accuracy/
            # speed trade-off the scheduler exploits, encoded in the mock
            if obj.rel_depth > 0.6 and rng.random() < self._mock_miss_far:
                continue
            conf = self._mock_conf + (0.18 if obj.rel_depth < 0.3 else 0.0)
            out.detections.append(Detection(
                label=obj.label,
                confidence=float(np.clip(rng.normal(conf, 0.05), 0.25, 0.97)),
                box_xyxy=obj.box_xyxy))
        return out


@register("yolo_nano")
class YoloNano(_YoloBase):
    cost = 1
    weights = "yolov8n.pt"
    _mock_latency_ms, _mock_conf, _mock_miss_far = 10.0, 0.64, 0.35


@register("yolo_small")
class YoloSmall(_YoloBase):
    cost = 2
    weights = "yolov8s.pt"
    _mock_latency_ms, _mock_conf, _mock_miss_far = 22.0, 0.72, 0.15


@register("yolo_large")
class YoloLarge(_YoloBase):
    cost = 4
    weights = "yolov8l.pt"
    _mock_latency_ms, _mock_conf, _mock_miss_far = 65.0, 0.84, 0.0
