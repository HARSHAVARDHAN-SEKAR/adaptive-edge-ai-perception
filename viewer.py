#!/usr/bin/env python3
"""Visual overlay viewer — see what the perception engine sees.

Draws on each frame:
  - bounding boxes colored by risk (teal -> amber -> red)
  - label with confidence, distance, activity
  - segmentation masks (alpha blend)
  - pose skeletons
  - HUD banner: scheduler mode, active models, FPS, scene risk
  - attention timeline strip along the bottom (mode history)

Examples:
    # record a demo clip of the synthetic mission (no hardware, no display)
    python viewer.py --source synthetic --frames 300 --record assets/demo.mp4

    # live window on your webcam with real models
    python viewer.py --source 0 --backend real --show

    # annotate a single image with real models
    python viewer.py --image assets/test_bus.jpg --backend real \
        --record assets/annotated_bus.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from edge_perception.pipeline import FrameResult, PerceptionPipeline
from edge_perception.sources import CameraSource, SyntheticSource

# BGR palette (matches the dashboard)
TEAL, AMBER, RED, VIOLET = (178, 184, 89), (60, 162, 229), (90, 106, 217), (212, 127, 143)
INK, PANEL = (218, 210, 201), (35, 30, 26)
MODE_COLOR = {"PATROL": TEAL, "ALERT": AMBER, "ENGAGED": RED, "DEGRADED": VIOLET}

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def risk_color(risk: float):
    return RED if risk >= 0.55 else AMBER if risk >= 0.35 else TEAL


class OverlayRenderer:
    def __init__(self, timeline_len: int = 240, watermark: str = ""):
        self._timeline: list[str] = []
        self._timeline_len = timeline_len
        self.watermark = watermark

    def render(self, frame: np.ndarray, r: FrameResult,
               raw_outputs=None) -> np.ndarray:
        img = frame.copy()
        h, w = img.shape[:2]
        s = max(h / 480.0, 0.75)                      # ui scale

        # -- masks (from raw outputs if provided) ---------------------------
        if raw_outputs:
            overlay = img.copy()
            for out in raw_outputs:
                for m in out.masks:
                    mask = m.mask
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h)) > 0
                    overlay[mask] = (0.45 * np.array(TEAL)
                                     + 0.55 * overlay[mask]).astype(np.uint8)
            img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

            # -- pose skeletons ---------------------------------------------
            for out in raw_outputs:
                for p in out.poses:
                    for a, b in SKELETON:
                        if p.keypoints_conf[a] > 0.3 and p.keypoints_conf[b] > 0.3:
                            pa = tuple(int(v) for v in p.keypoints_xy[a])
                            pb = tuple(int(v) for v in p.keypoints_xy[b])
                            cv2.line(img, pa, pb, INK, max(1, int(2 * s)))
                    if p.activity == "fallen":
                        x, y = p.keypoints_xy[p.keypoints_conf > 0.3].mean(axis=0)
                        cv2.putText(img, "FALLEN", (int(x), int(y) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, RED, 2)

        # -- boxes + labels ---------------------------------------------------
        for o in r.scene.objects:
            x1, y1, x2, y2 = (int(v) for v in o.box_xyxy)
            c = risk_color(o.risk)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, max(1, int(2 * s)))
            bits = [f"{o.label} {o.confidence:.2f}"]
            if o.rel_depth is not None:
                bits.append(f"rd {o.rel_depth:.2f}")
            if o.activity not in ("n/a", "unknown"):
                bits.append(o.activity)
            label = " | ".join(bits)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.45 * s, 1)
            ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 6
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 3),
                          PANEL, -1)
            cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * s, c, 1, cv2.LINE_AA)

        # -- HUD banner --------------------------------------------------------
        mode = r.decision.mode.value
        banner_h = int(30 * s)
        cv2.rectangle(img, (0, 0), (w, banner_h), PANEL, -1)
        cv2.putText(img, f"{mode}", (int(8 * s), int(21 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, MODE_COLOR[mode], 2,
                    cv2.LINE_AA)
        short = {
            "yolo_nano": "det-n",
            "yolo_small": "det-s",
            "yolo_large": "det-l",
            "yolo_nano_seg": "seg",
            "midas_small": "depth",
            "yolo_pose": "pose",
        }
        model_text = "+".join(short.get(m, m) for m in r.decision.models)
        hud = (f"fps {r.fps:4.1f} | risk {r.scene.scene_risk:.2f} | "
               f"{model_text}")
        cv2.putText(img, hud, (int(110 * s), int(20 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, INK, 1, cv2.LINE_AA)

        # -- provenance watermark (top-right, permanent) -----------------------
        if self.watermark:
            (ww, wh), _ = cv2.getTextSize(self.watermark,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, 1)
            cv2.putText(img, self.watermark, (w - ww - 8, banner_h + wh + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * s, VIOLET, 1,
                        cv2.LINE_AA)

        # -- VLM narration (above the timeline strip) --------------------------
        if r.narration:
            # Wrap narration rather than clipping long text at the right edge.
            words = r.narration.split()
            lines, cur = [], ""
            max_w = w - 18
            for word in words:
                cand = (cur + " " + word).strip()
                tw, _ = cv2.getTextSize(
                    cand, cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, 1)[0]
                if cur and tw > max_w:
                    lines.append(cur)
                    cur = word
                else:
                    cur = cand
            if cur:
                lines.append(cur)
            lines = lines[:2]

            line_h = max(14, int(17 * s))
            ny0 = h - int(14 * s) - line_h * (len(lines) - 1)
            cv2.rectangle(
                img, (0, ny0 - line_h),
                (w, h - max(5, int(5 * s))), PANEL, -1)
            for j, line in enumerate(lines):
                cv2.putText(
                    img, line, (6, ny0 + j * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, AMBER, 1,
                    cv2.LINE_AA)

        # -- attention timeline strip ------------------------------------------
        self._timeline.append(mode)
        self._timeline = self._timeline[-self._timeline_len:]
        strip_h = max(4, int(6 * s))
        seg_w = w / self._timeline_len
        for i, m in enumerate(self._timeline):
            x = int(i * seg_w)
            cv2.rectangle(img, (x, h - strip_h), (int(x + seg_w) + 1, h),
                          MODE_COLOR[m], -1)
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--image", help="annotate a single image instead of a stream")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--show", action="store_true", help="live cv2 window")
    ap.add_argument("--record", help="output .mp4 (stream) or image path")
    ap.add_argument("--fps-out", type=int, default=20, help="recording fps")
    args = ap.parse_args()

    pipe = PerceptionPipeline(backend=args.backend, device=args.device,
                              adaptive=not args.no_adaptive)
    if args.image:
        wm = "REAL IMAGE - MOCK MODEL OUTPUTS" if args.backend == "mock" else ""
    elif args.source == "synthetic":
        wm = ("SYNTHETIC INPUT - MOCK MODEL LATENCIES"
              if args.backend == "mock" else "SYNTHETIC INPUT")
    else:
        wm = "MOCK MODEL OUTPUTS" if args.backend == "mock" else ""
    renderer = OverlayRenderer(watermark=wm)

    # capture raw model outputs for mask/pose drawing
    raw_holder = {}
    orig_fuse = pipe.fusion.fuse
    def fuse_hook(outputs, shape):
        raw_holder["outputs"] = outputs
        return orig_fuse(outputs, shape)
    pipe.fusion.fuse = fuse_hook

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"could not read image: {args.image}")
        pipe.process(frame, 0)                       # warm-up / tracking prime
        r = pipe.process(frame, 1)
        annotated = renderer.render(frame, r, raw_holder.get("outputs"))
        out_path = args.record or "annotated.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"annotated image: {out_path} | {len(r.scene.objects)} objects, "
              f"mode {r.decision.mode.value}")
        pipe.close()
        return

    src = (SyntheticSource(n_frames=args.frames) if args.source == "synthetic"
           else CameraSource(int(args.source) if args.source.isdigit()
                             else args.source))
    writer = None
    try:
        for i, frame in enumerate(src.frames()):
            if i >= args.frames:
                break
            r = pipe.process(frame, i)
            vis = renderer.render(frame, r, raw_holder.get("outputs"))
            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps_out, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if args.show:
                cv2.imshow("edge perception", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
            print(f"recorded: {args.record}")
        if args.show:
            cv2.destroyAllWindows()
        pipe.close()


if __name__ == "__main__":
    main()
