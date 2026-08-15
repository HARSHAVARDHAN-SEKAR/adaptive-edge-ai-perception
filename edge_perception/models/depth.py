"""Monocular RELATIVE depth estimation (MiDaS small via torch.hub).

IMPORTANT: monocular models like MiDaS produce RELATIVE depth, not metric
distance. Outputs here are normalized to [0, 1] (0 = nearest, 1 = farthest)
and labeled `depth_scale="relative"` everywhere downstream. For metric
range use ZoeDepth, an RGB-D/stereo camera, or LiDAR fusion.
"""

from __future__ import annotations

import numpy as np

from .base import ModelOutput, PerceptionModel, mock_sleep, mock_world, register


@register("midas_small")
class MidasSmall(PerceptionModel):
    task = "depth"
    cost = 3

    def _load_real(self) -> bool:
        import torch

        self._torch = torch
        self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self._model.to(self.device).eval()
        tfm = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = tfm.small_transform
        return True

    def _unload_impl(self) -> None:
        self._model = None
        self._transform = None

    def _infer_real(self, frame_bgr: np.ndarray) -> ModelOutput:
        import cv2

        torch = self._torch
        out = ModelOutput(self.name, self.task, 0.0)
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        batch = self._transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self._model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
        inv = pred.cpu().numpy()
        # MiDaS outputs inverse relative depth; normalize to [0,1] far-ness.
        # np.ptp (not ndarray.ptp) — the method was removed in NumPy 2.x.
        inv = (inv - inv.min()) / (np.ptp(inv) + 1e-6)
        out.depth_map = (1.0 - inv).astype(np.float32)
        out.depth_scale = "relative"
        return out

    def _infer_mock(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        mock_sleep(28.0)
        h, w = frame_bgr.shape[:2]
        # ground-camera prior: bottom of frame is near
        base = np.linspace(0.15, 0.95, h)[::-1][:, None] * np.ones((h, w), np.float32)
        # imprint the SAME world objects at their true relative depth
        for obj in mock_world(frame_bgr):
            x1, y1, x2, y2 = (int(v) for v in obj.box_xyxy)
            base[max(y1, 0) : min(y2, h), max(x1, 0) : min(x2, w)] = obj.rel_depth
        out.depth_map = base.astype(np.float32)
        out.depth_scale = "relative"
        return out
