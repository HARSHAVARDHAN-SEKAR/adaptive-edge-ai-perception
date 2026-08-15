#!/usr/bin/env python3
"""Tier agreement with large-model pseudo-labels (NOT ground-truth accuracy).

The largest model's detections serve as pseudo-labels and lighter tiers are
scored against them (IoU>0.5, same class). This measures MODEL AGREEMENT,
not real accuracy — for real accuracy use a labelled dataset (COCO val) and
report mAP50 / mAP50-95. Use several hundred frames, not one image, before
citing these numbers.

    python -m benchmark.accuracy_test --images assets/ --backend real
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edge_perception.models import base as model_base
from edge_perception.models import detection  # noqa: F401
from edge_perception.sources import SyntheticSource


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(preds, gts, thr=0.5):
    matched = set()
    tp = 0
    for p in preds:
        for i, g in enumerate(gts):
            if i in matched or g.label != p.label:
                continue
            if iou(p.box_xyxy, g.box_xyxy) >= thr:
                matched.add(i)
                tp += 1
                break
    fp, fn = len(preds) - tp, len(gts) - tp
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="assets", help="folder of jpg/png test images")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reference", default="yolo_large")
    ap.add_argument("--out", default="assets/bench_tier_agreement.md")
    args = ap.parse_args()

    import cv2

    paths = sorted(Path(args.images).glob("*.[jp][pn]g"))
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:  # fall back to synthetic frames
        frames = list(SyntheticSource(n_frames=30).frames())
        print("[info] no images found — using synthetic frames")

    ref = model_base.create(args.reference, device=args.device, backend=args.backend)
    gts = [ref(f).detections for f in frames]
    ref.unload()

    lines = [
        (
            f"# Tier agreement with {args.reference} pseudo-labels "
            f"({len(frames)} frames, IoU>0.5)"
        ),
        "",
        "| Tier | Precision | Recall | F1 |",
        "|---|---|---|---|",
    ]
    for tier in ("yolo_nano", "yolo_small"):
        m = model_base.create(tier, device=args.device, backend=args.backend)
        TP = FP = FN = 0
        for f, gt in zip(frames, gts):
            tp, fp, fn = score(m(f).detections, gt)
            TP, FP, FN = TP + tp, FP + fp, FN + fn
        m.unload()
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lines.append(f"| {tier} | {prec:.2f} | {rec:.2f} | {f1:.2f} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
