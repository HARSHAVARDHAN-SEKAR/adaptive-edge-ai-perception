#!/usr/bin/env python3
"""Export YOLO-family weights to ONNX (stage 1 of PyTorch -> ONNX -> TensorRT).

Usage:
    python -m edge_perception.optimizer.export_onnx --weights yolov8n.pt \
        --imgsz 640 --half --out engines/

Ultralytics handles graph tracing; we add dynamic batch off by default
(TensorRT engines for robots are almost always batch=1, fixed-shape —
fixed shapes let TRT pick faster kernels).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def export(weights: str, imgsz: int = 640, half: bool = False,
           out_dir: str = "engines") -> Path:
    from ultralytics import YOLO
    model = YOLO(weights)
    onnx_path = model.export(format="onnx", imgsz=imgsz, half=half,
                             dynamic=False, simplify=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / Path(onnx_path).name
    Path(onnx_path).replace(dst)
    print(f"exported: {dst}")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 export")
    ap.add_argument("--out", default="engines")
    a = ap.parse_args()
    export(a.weights, a.imgsz, a.half, a.out)
