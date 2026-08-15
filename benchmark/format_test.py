#!/usr/bin/env python3
"""Experiment 1 — optimization format comparison: PyTorch vs ONNX vs TensorRT.

Benchmarks the SAME weights (default yolov8n) across every format available
on this machine:
  - PyTorch (ultralytics)          any machine
  - ONNX Runtime (CPU/CUDA EP)     any machine (auto-exports if needed)
  - TensorRT engine                Jetson / NVIDIA GPU with trt installed

    python -m benchmark.format_test --weights yolov8n.pt --runs 30 \
        --out assets/bench_formats.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def _percentiles(lat):
    lat = np.array(lat)
    return (
        round(1000.0 / lat.mean(), 1),
        round(float(np.percentile(lat, 50)), 1),
        round(float(np.percentile(lat, 95)), 1),
    )


def bench_pytorch(weights, img, runs, device):
    from ultralytics import YOLO

    m = YOLO(weights)
    m.predict(img, verbose=False, device=device)  # warm-up
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m.predict(img, verbose=False, device=device)
        lat.append((time.perf_counter() - t0) * 1000)
    return _percentiles(lat)


def bench_onnx(weights, img, runs, imgsz):
    import onnxruntime as ort
    from ultralytics import YOLO

    onnx_path = Path(weights).with_suffix(".onnx")
    if not onnx_path.exists():
        YOLO(weights).export(format="onnx", imgsz=imgsz, dynamic=False)
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    name = sess.get_inputs()[0].name
    import cv2

    x = cv2.resize(img, (imgsz, imgsz))[:, :, ::-1]
    x = (x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).copy()
    sess.run(None, {name: x})  # warm-up
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        lat.append((time.perf_counter() - t0) * 1000)
    return _percentiles(lat), providers[0]


def bench_tensorrt(weights, img, runs, device):
    """Ultralytics loads .engine files natively."""
    from ultralytics import YOLO

    engine = Path(weights).with_suffix(".engine")
    if not engine.exists():
        YOLO(weights).export(format="engine", half=True, device=device)
    m = YOLO(str(engine))
    m.predict(img, verbose=False, device=device)
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m.predict(img, verbose=False, device=device)
        lat.append((time.perf_counter() - t0) * 1000)
    return _percentiles(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--image", default="assets/test_bus.jpg")
    ap.add_argument("--out", default="assets/bench_formats.md")
    args = ap.parse_args()

    import cv2

    img = cv2.imread(args.image)
    if img is None:
        img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)

    rows = []
    fps, p50, p95 = bench_pytorch(args.weights, img, args.runs, args.device)
    rows.append(("PyTorch", args.device, fps, p50, p95))

    try:
        (fps, p50, p95), ep = bench_onnx(args.weights, img, args.runs, args.imgsz)
        rows.append(
            ("ONNX Runtime", ep.replace("ExecutionProvider", ""), fps, p50, p95)
        )
    except Exception as e:
        print(f"[skip] ONNX Runtime: {e}")

    try:
        fps, p50, p95 = bench_tensorrt(args.weights, img, args.runs, args.device)
        rows.append(("TensorRT FP16", args.device, fps, p50, p95))
    except Exception as e:
        print(f"[skip] TensorRT (expected off-Jetson): {type(e).__name__}")

    lines = [
        f"# Format benchmark — {args.weights}, {args.runs} runs, input {args.imgsz}",
        "",
        "| Format | Device/EP | FPS | p50 (ms) | p95 (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    if len(rows) >= 2:
        best = max(rows, key=lambda r: r[2])
        speedup = best[2] / rows[0][2]
        lines += [
            "",
            (
                f"Best format on this machine: **{best[0]}** at "
                f"**{speedup:.2f}x** the PyTorch baseline."
            ),
            "",
            (
                "> Caveat: the PyTorch/TensorRT rows time the full "
                "Ultralytics pipeline (pre+post-processing) while the ONNX "
                "row times raw session.run only — treat the ONNX row as a "
                "lower-bound reference, not a strict comparison. Run on "
                "Jetson for the TensorRT rows."
            ),
        ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
