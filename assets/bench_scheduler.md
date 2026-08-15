# Scheduler evaluation — 120-frame synthetic mission, events at ((30, 41), (80, 91))

> These results use deterministic synthetic scenes and latency-simulating mock models. They evaluate scheduler behaviour, not real model accuracy or Jetson performance. 'Cost proxy units' sums manually assigned per-model cost scores over models that actually executed; it is not measured energy or CPU usage.

| System | Mean FPS | FPS σ | Lat mean (ms) | Lat p95 (ms) | Cost proxy units | Event-window coverage | High-risk coverage | Switches |
|---|---|---|---|---|---|---|---|---|
| fixed_heavy | 6.7 | 0.12 | 148.7 | 169.1 | 1440 | 1.0 | 1.0 | 0 |
| fixed_light | 89.8 | 2.72 | 11.0 | 13.5 | 120 | 0.0 | 0.0 | 0 |
| adaptive | 33.0 | 35.08 | 92.5 | 169.0 | 885 | 0.64 | 0.87 | 5 |

Adaptive uses **39% fewer cost-proxy units** than the fixed heavyweight baseline while running the full model suite on **87%** of genuinely high-risk frames (risk >= 0.55); the fixed light baseline covers 0%.

Event-window coverage is lower than high-risk coverage by design: the scripted event includes an approach ramp whose early frames carry genuinely low risk, and the scheduler correctly stays in ALERT there.
