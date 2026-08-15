#!/usr/bin/env python3
"""Experiment 3 - multi-model scene-understanding demonstration.

This experiment compares a detection-only description with fused detections,
instance segmentation, monocular RELATIVE depth and human pose.

IMPORTANT:
- MiDaS output is relative depth, not metres.
- backend='real' fails if a requested real model cannot be loaded.
- backend='mock' is a deterministic systems/demo path, not an accuracy test.

Examples:
    python -m benchmark.understanding_demo --backend mock
    python -m benchmark.understanding_demo \
        --image assets/test_bus.jpg --backend real
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_perception.fusion.perception_fusion import PerceptionFusion
from edge_perception.fusion.uncertainty import estimate
from edge_perception.models import base as model_base
from edge_perception.models import (depth, detection, pose, segmentation,  # noqa: F401
                                    vlm)
from edge_perception.sources import SyntheticSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--backend", choices=("mock", "auto", "real"), default="mock")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="assets/understanding_demo.json")
    args = ap.parse_args()

    if args.image:
        import cv2
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"could not read image: {args.image}")
        input_kind = "image"
    else:
        frame = list(SyntheticSource(n_frames=120).frames())[110]
        input_kind = "synthetic"

    backends = {}

    def run(name):
        m = model_base.create(name, device=args.device, backend=args.backend)
        out = m(frame)
        backends[name] = {
            "requested": out.requested_backend,
            "actual": out.actual_backend,
        }
        m.unload()
        return out

    # BEFORE: detection only
    det_only = run("yolo_nano")
    before = [f"{d.label} detected" for d in det_only.detections]

    # AFTER: full multi-model fusion
    outputs = [run(n) for n in
               ("yolo_large", "yolo_nano_seg", "midas_small", "yolo_pose")]
    fusion = PerceptionFusion()
    fusion.fuse(outputs, frame.shape)  # prime tracker
    scene = fusion.fuse(outputs, frame.shape)
    narration = run("vlm_scene").extra.get("description", "")

    after = []
    depth_used = any(o.depth_map is not None for o in outputs)
    for o in scene.objects:
        conf = estimate(o, depth_used)
        after.append({
            "object": o.label,
            "rel_depth": (round(o.rel_depth, 3)
                          if o.rel_depth is not None else None),
            "depth_scale": "relative",
            "activity": o.activity,
            "approaching": o.approaching,
            "collision_risk_pct": round(100 * o.risk),
            "decision_confidence": round(conf.decision_confidence, 2),
        })

    result = {
        "input_kind": input_kind,
        "requested_backend": args.backend,
        "backends": backends,
        "before_detection_only": before,
        "after_multi_model_fusion": after,
        "vlm_scene_narration": narration,
        "note": ("Relative depth is a normalized monocular depth cue "
                 "(0=near, 1=far); it is not metric distance."),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    print("BEFORE (detection only):")
    for b in before:
        print(f"  {b}")
    print("\nAFTER (multi-model fusion):")
    for a in after:
        rd = "n/a" if a["rel_depth"] is None else f"{a['rel_depth']:.3f}"
        print(f"  {a['object']}: rel_depth={rd}, {a['activity']}, "
              f"approaching={a['approaching']}, "
              f"risk={a['collision_risk_pct']}%, "
              f"confidence={a['decision_confidence']}")
    print(f"\nVLM narration: {narration}")
    print("\nNOTE: rel_depth is relative (0=near, 1=far), not metres.")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
