#!/usr/bin/env python3
"""Experiment 1 — per-model throughput/latency benchmark.

Measures FPS, latency p50/p95, and process RSS delta for every registered
model on this machine's backend. On Jetson, run once per format
(PyTorch / ONNX Runtime / TensorRT FP16 / INT8) to fill the optimization
table in the report.

    python -m benchmark.fps_test --frames 100 --backend auto --out assets/bench_models.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edge_perception.models import base as model_base
from edge_perception.models import (  # noqa: F401
    depth,
    detection,
    pose,
    segmentation,
    vlm,
)
from edge_perception.sources import SyntheticSource

try:
    import psutil

    _PROC = psutil.Process()
except Exception:
    _PROC = None


def bench_model(name: str, frames, backend: str, device: str, warmup: int = 5):
    rss0 = _PROC.memory_info().rss if _PROC else 0
    model = model_base.create(name, device=device, backend=backend)
    model.load()
    lat = []
    for i, f in enumerate(frames):
        out = model(f)
        if i >= warmup:
            lat.append(out.latency_ms)
    rss1 = _PROC.memory_info().rss if _PROC else 0
    if not lat:
        raise ValueError(
            f"need more than {warmup} frames for benchmark; got {len(frames)}"
        )
    lat = np.array(lat)
    row = {
        "model": name,
        "task": model.task,
        "backend": model.actual_backend,
        "fps": round(1000.0 / lat.mean(), 1),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "load_time_s": round(model._load_time_s, 2),
        "rss_delta_mb": round((rss1 - rss0) / 1e6, 1),
    }
    model.unload()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="assets/bench_models.md")
    args = ap.parse_args()

    if args.frames <= 5:
        ap.error("--frames must be greater than the 5-frame warmup")
    frames = list(SyntheticSource(n_frames=args.frames).frames())
    canonical = [n for n in model_base.available() if n != "fast_sam"]
    rows = [bench_model(n, frames, args.backend, args.device) for n in canonical]

    header = (
        "| Model | Task | Backend | FPS | p50 (ms) | p95 (ms) | Load (s) | ΔRSS (MB) |"
    )
    sep = "|" + "---|" * 8
    lines = [
        f"# Model benchmark — backend={args.backend}, device={args.device}",
        "",
        header,
        sep,
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['task']} | {r['backend']} | "
            f"{r['fps']} | {r['latency_p50_ms']} | {r['latency_p95_ms']} | "
            f"{r['load_time_s']} | {r['rss_delta_mb']} |"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
