# Format benchmark — historical local CPU snapshot

Configuration: `yolov8n.pt`, 20 timed runs, input size 640.

| Format | Device/EP | FPS | p50 (ms) | p95 (ms) |
|---|---|---:|---:|---:|
| PyTorch | CPU | 7.3 | 132.7 | 171.6 |
| ONNX Runtime | CPU | 7.1 | 129.3 | 161.4 |

Best measured throughput in this saved run was **PyTorch (7.3 FPS)**.

> **Measurement caveat:** this snapshot predates the current benchmark-output wording and was not regenerated during the publication audit because the real-model dependencies are not installed in the audit environment. PyTorch timings include the full Ultralytics prediction pipeline, while the ONNX row times raw `session.run`; therefore these rows are **not a strict apples-to-apples speed comparison**. Re-run `python -m benchmark.format_test` in the target environment for current measurements. TensorRT must be benchmarked on the intended Jetson/device.
