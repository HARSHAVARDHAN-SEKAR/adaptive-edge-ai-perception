# Model benchmark — backend=mock, device=cpu

| Model | Task | Backend | FPS | p50 (ms) | p95 (ms) | Load (s) | ΔRSS (MB) |
|---|---|---|---|---|---|---|---|
| midas_small | depth | mock | 28.0 | 36.18 | 41.49 | 0.0 | 2.1 |
| vlm_scene | describe | mock | 5.4 | 187.56 | 233.15 | 0.0 | 0.0 |
| yolo_large | detect | mock | 14.5 | 67.47 | 79.58 | 0.0 | 0.0 |
| yolo_nano | detect | mock | 62.5 | 15.65 | 18.15 | 0.0 | 0.0 |
| yolo_nano_seg | segment | mock | 21.7 | 46.04 | 55.03 | 0.0 | 8.5 |
| yolo_pose | pose | mock | 47.3 | 21.18 | 23.58 | 0.0 | 2.3 |
| yolo_small | detect | mock | 36.9 | 27.0 | 33.5 | 0.0 | 0.0 |
