#!/usr/bin/env python3
"""Build a TensorRT engine from ONNX (stage 2, run ON THE JETSON).

TensorRT engines are tuned to the exact GPU they're built on — build on the
deployment device, not the laptop.

Usage (Jetson):
    python -m edge_perception.optimizer.build_tensorrt \
        --onnx engines/yolov8n.onnx --fp16
    python -m edge_perception.optimizer.build_tensorrt \
        --onnx engines/yolov8n.onnx --int8 --calib-dir calib_images/

FP16 is the sweet spot on Orin/Xavier: ~2-3x over FP32 with negligible
mAP loss. INT8 needs a calibration set (500-1000 mission-representative
frames) and should be validated against the FP16 baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build(
    onnx_path: str,
    fp16: bool = True,
    int8: bool = False,
    calib_dir: str | None = None,
    workspace_gb: float = 2.0,
) -> Path:
    import tensorrt as trt  # available in JetPack / TensorRT containers

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    data = Path(onnx_path).read_bytes()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"failed to parse {onnx_path}")

    if int8 and not calib_dir:
        raise SystemExit(
            "--int8 requires --calib-dir with 500-1000 mission-representative frames"
        )
    if int8:
        from pathlib import Path as _P

        n = len(list(_P(calib_dir).glob("*.[jp][pn]g")))
        if n == 0:
            raise SystemExit(f"calibration dir '{calib_dir}' contains no images")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise SystemExit("platform lacks fast FP16; rerun with --no-fp16")
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        if not builder.platform_has_fast_int8:
            raise SystemExit("platform lacks INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        from .int8_calibrator import ImageFolderCalibrator

        config.int8_calibrator = ImageFolderCalibrator(calib_dir)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("engine build failed")

    suffix = "_int8" if int8 else ("_fp16" if fp16 else "_fp32")
    out = (
        Path(onnx_path)
        .with_suffix("")
        .with_name(Path(onnx_path).stem + suffix)
        .with_suffix(".engine")
    )
    out.write_bytes(engine_bytes)
    print(f"engine written: {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-fp16 for FP32",
    )
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    a = ap.parse_args()
    build(a.onnx, a.fp16, a.int8, a.calib_dir, a.workspace_gb)
