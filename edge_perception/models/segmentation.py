"""Instance segmentation tier: YOLOv8n-seg (Ultralytics).

Named honestly: this is YOLO instance segmentation, not FastSAM. To use
actual FastSAM on Jetson, point `weights` at FastSAM-s.pt and adapt
post-processing. Registry keeps the legacy alias "fast_sam" for config
compatibility; the canonical name is "yolo_nano_seg".
"""
from __future__ import annotations

import numpy as np

from .base import (ModelOutput, PerceptionModel, SegmentationMask, mock_rng, mock_sleep,
                   mock_world, register)


@register("yolo_nano_seg")
class YoloNanoSeg(PerceptionModel):
    task = "segment"
    cost = 3
    weights = "yolov8n-seg.pt"

    def _load_real(self) -> bool:
        from ultralytics import YOLO
        self._model = YOLO(self.weights)
        return True

    def _unload_impl(self) -> None:
        self._model = None

    def _infer_real(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        res = self._model.predict(frame_bgr, verbose=False, device=self.device)[0]
        if res.masks is not None:
            names = res.names
            for m, b in zip(res.masks.data, res.boxes):
                out.masks.append(SegmentationMask(
                    label=names[int(b.cls)],
                    confidence=float(b.conf),
                    mask=m.cpu().numpy().astype(bool),
                    box_xyxy=tuple(float(v) for v in b.xyxy[0].tolist())))
        return out

    def _infer_mock(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        mock_sleep(35.0)
        h, w = frame_bgr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        rng = mock_rng(frame_bgr, self.name)
        for obj in mock_world(frame_bgr):     # SAME world as the detector
            x1, y1, x2, y2 = obj.box_xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = max((x2 - x1) / 2, 1), max((y2 - y1) / 2, 1)
            mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
            out.masks.append(SegmentationMask(
                obj.label,
                float(np.clip(rng.normal(0.78, 0.05), 0.3, 0.97)),
                mask, obj.box_xyxy))
        return out


# legacy alias for existing configs
from .base import _REGISTRY as _R          # noqa: E402
_R["fast_sam"] = YoloNanoSeg
