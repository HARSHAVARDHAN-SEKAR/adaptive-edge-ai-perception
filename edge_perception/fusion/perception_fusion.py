"""Multi-model perception fusion.

Per-object scene description:
    person | rel_depth 0.22 (near) | approaching | upright | risk 0.61

Corrections vs v1 (per external review):
  - depth is RELATIVE [0..1], never presented as metres
  - track association requires SAME CLASS and uses IoU + centroid gating
  - velocity is pixels/second from real timestamps, not pixels/frame
  - "approaching" uses box looming (area growth rate), robust to FPS changes
  - mask association requires matching class label
  - risk model is transparent and documented as a heuristic
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..models.base import Detection, ModelOutput
from ..models.pose import pose_center

CLASS_RISK = {"person": 0.6, "dog": 0.4, "car": 0.5, "backpack": 0.15, "chair": 0.05}
W_CLASS, W_DEPTH, W_MOTION, W_POSE = 0.35, 0.35, 0.15, 0.15
LOOM_RATE = 0.06  # fractional area growth per second -> approaching
GATE_PX = 80.0  # centroid gate for track association
GATE_IOU = 0.1  # minimum IoU for track association


@dataclass
class FusedObject:
    object_id: int
    label: str
    confidence: float
    box_xyxy: tuple
    rel_depth: float | None = None  # 0 near .. 1 far (RELATIVE)
    velocity_px_s: tuple | None = None  # (dx, dy) pixels/second
    approaching: bool = False
    activity: str = "n/a"
    has_mask: bool = False
    risk: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["box_xyxy"] = [round(v, 1) for v in self.box_xyxy]
        if self.rel_depth is not None:
            d["rel_depth"] = round(self.rel_depth, 3)
        d["risk"] = round(self.risk, 3)
        return d


@dataclass
class SceneUnderstanding:
    timestamp: float
    objects: list[FusedObject] = field(default_factory=list)
    scene_risk: float = 0.0
    min_detection_conf: float = 0.0
    models_used: list[str] = field(default_factory=list)
    backends_used: dict[str, str] = field(default_factory=dict)
    depth_scale: str | None = None  # "relative" when depth present

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "scene_risk": round(self.scene_risk, 3),
            "objects": [o.to_dict() for o in self.objects],
            "models_used": self.models_used,
            "backends_used": self.backends_used,
            "depth_scale": self.depth_scale,
        }


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class _Track:
    label: str
    center: tuple
    box: tuple
    area: float
    t: float


class PerceptionFusion:
    def __init__(self):
        self._next_id = 0
        self._tracks: dict[int, _Track] = {}

    # ------------------------------------------------------------------
    def fuse(
        self, outputs: list[ModelOutput], frame_shape: tuple
    ) -> SceneUnderstanding:
        h, w = frame_shape[:2]
        now = time.time()
        scene = SceneUnderstanding(
            timestamp=now,
            models_used=[o.model_name for o in outputs],
            backends_used={o.model_name: o.actual_backend for o in outputs},
        )

        detections: list[Detection] = []
        depth_map = None
        masks, poses = [], []
        for o in outputs:
            detections.extend(o.detections)
            masks.extend(o.masks)
            poses.extend(o.poses)
            if o.depth_map is not None:
                depth_map = o.depth_map
                scene.depth_scale = o.depth_scale

        if detections:
            scene.min_detection_conf = min(d.confidence for d in detections)

        new_tracks: dict[int, _Track] = {}
        for det in detections:
            oid, prev = self._associate(det, new_tracks)
            obj = FusedObject(
                object_id=oid,
                label=det.label,
                confidence=det.confidence,
                box_xyxy=det.box_xyxy,
            )

            # -- relative depth: median of the central box region ----------
            if depth_map is not None:
                x1, y1, x2, y2 = (int(v) for v in det.box_xyxy)
                cx1 = x1 + (x2 - x1) // 4
                cx2 = x2 - (x2 - x1) // 4
                cy1 = y1 + (y2 - y1) // 4
                cy2 = y2 - (y2 - y1) // 4
                patch = depth_map[
                    max(cy1, 0) : min(max(cy2, cy1 + 1), h),
                    max(cx1, 0) : min(max(cx2, cx1 + 1), w),
                ]
                if patch.size:
                    obj.rel_depth = float(np.clip(np.median(patch), 0.0, 1.0))

            # -- mask association: IoU AND same class ----------------------
            for m in masks:
                if (
                    m.box_xyxy
                    and m.label == det.label
                    and _iou(det.box_xyxy, m.box_xyxy) > 0.4
                ):
                    obj.has_mask = True
                    break

            # -- pose association (persons only, valid-keypoint center) ----
            if det.label == "person":
                x1, y1, x2, y2 = det.box_xyxy
                for p in poses:
                    kx, ky = pose_center(p.keypoints_xy, p.keypoints_conf)
                    if x1 <= kx <= x2 and y1 <= ky <= y2:
                        obj.activity = p.activity
                        break

            # -- motion: px/s from real timestamps; looming -> approaching -
            if prev is not None:
                dt = max(now - prev.t, 1e-3)
                dx = (det.center[0] - prev.center[0]) / dt
                dy = (det.center[1] - prev.center[1]) / dt
                obj.velocity_px_s = (round(dx, 1), round(dy, 1))
                if prev.area > 0:
                    growth_rate = (det.area - prev.area) / prev.area / dt
                    obj.approaching = growth_rate > LOOM_RATE

            obj.risk = self._risk(obj, w)
            scene.objects.append(obj)
            new_tracks[oid] = _Track(det.label, det.center, det.box_xyxy, det.area, now)

        self._tracks = new_tracks
        scene.scene_risk = max((o.risk for o in scene.objects), default=0.0)
        return scene

    # ------------------------------------------------------------------
    def _associate(self, det: Detection, taken: dict[int, _Track]):
        """Same-class association by IoU then centroid distance."""
        best_id, best_score, best_track = None, 0.0, None
        for oid, tr in self._tracks.items():
            if oid in taken or tr.label != det.label:  # class must match
                continue
            iou = _iou(det.box_xyxy, tr.box)
            d = (
                (det.center[0] - tr.center[0]) ** 2
                + (det.center[1] - tr.center[1]) ** 2
            ) ** 0.5
            if iou >= GATE_IOU or d < GATE_PX:
                score = iou + max(0.0, 1.0 - d / GATE_PX)
                if score > best_score:
                    best_id, best_score, best_track = oid, score, tr
        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
            return best_id, None
        return best_id, best_track

    @staticmethod
    def _risk(obj: FusedObject, frame_w: int) -> float:
        """Heuristic risk in [0,1]; weights documented in the report."""
        class_term = CLASS_RISK.get(obj.label, 0.1)
        if obj.rel_depth is not None:
            depth_term = 1.0 - obj.rel_depth  # near -> high
        else:  # apparent-size proxy
            x1, _, x2, _ = obj.box_xyxy
            depth_term = float(np.clip((x2 - x1) / frame_w * 2.0, 0, 1))
        motion_term = 1.0 if obj.approaching else 0.0
        pose_term = 1.0 if obj.activity == "fallen" else 0.0
        risk = (
            W_CLASS * class_term
            + W_DEPTH * depth_term
            + W_MOTION * motion_term
            + W_POSE * pose_term
        )
        return float(np.clip(risk * obj.confidence + (1 - obj.confidence) * 0.2, 0, 1))
