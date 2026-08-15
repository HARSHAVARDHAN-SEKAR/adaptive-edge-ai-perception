# Adaptive Edge AI Perception Engine for Autonomous Mobile Robots

> **Research question:** Can a mobile robot dynamically choose and adapt its
> AI perception models based on scene risk and available edge compute?

Instead of running one fixed detector forever, an **adaptive scheduler**
decides every frame which perception models run: cheap and fast while
patrolling, full multi-model attention (detection + segmentation + relative
depth + pose) the moment risk rises, graceful degradation when compute
saturates.

```
 Camera ──▶ Adaptive Scheduler ──▶ Model Plan ──▶ Inference ──▶ Fusion ──▶ ROS2
              ▲          ▲          PATROL : yolo_nano                  │
              │          │          ALERT  : yolo_small + depth         │
        resources      risk         ENGAGED: large + seg + depth + pose │
        (FPS/GPU)   (fused scene) ◀─────────────────────────────────────┘
```

The scheduler is two orthogonal state machines composed into one plan:
**attention demand** (PATROL/ALERT/ENGAGED, from scene risk) and **resource
condition** (NORMAL/CONSTRAINED/CRITICAL, from FPS and GPU load), each with
its own hysteresis. Risk is evaluated *before* load shedding, so a dangerous
scene under compute pressure still gets the best feasible plan.

## ⚠️ Validation status — read before citing results

| Environment | Status |
|---|---|
| Python 3.10-3.12, mock backend | CI tested |
| Lite Docker image | CI tested |
| ROS 2 node logic (mocked `rclpy`) | CI tested |
| ROS 2 Humble `colcon build` | CI tested in a Humble container |
| Real-model path | supported; not part of CI |
| Gazebo `edge_bot` runtime | launch + URDF provided; runtime validation recommended |
| Jetson Orin TensorRT / INT8 | scripts provided; device validation required |
| Metric depth | **not implemented** - monocular depth is relative only |

The scheduler results below use **deterministic synthetic scenes and
latency-simulating mock models**. They evaluate *scheduler behaviour*, not
real model accuracy or Jetson performance.

## Results (regenerate with `python -m benchmark.latency_test`)

120-frame synthetic mission, events at [[30, 41], [80, 91]]:

| System | Mean FPS | Lat mean (ms) | Cost proxy units | High-risk full-suite coverage |
|---|---|---|---|---|
| fixed_heavy | 6.7 | 148.7 | 1440 | 100% |
| fixed_light | 89.8 | 11.0 | 120 | 0% |
| adaptive | 33.0 | 92.5 | 885 | 87% |

**Adaptive uses 39% fewer model-cost proxy units than the fixed heavyweight baseline at 4.9x its mean FPS, while running the full model suite on 87% of genuinely high-risk frames** (risk >= 0.55; the light baseline covers 0%). Event-window coverage is lower (64%) by design - each event includes an approach ramp whose early frames carry genuinely low risk, where ALERT is the correct response.

*"Cost proxy units" sums manually assigned per-model cost scores over models
that actually executed. It is a scheduling-cost proxy, **not** measured
energy, CPU time, or TOPS.*
The scheduler benchmark uses 160x120 synthetic frames intentionally: it is a
state-machine/scheduling experiment, not an image-accuracy benchmark. Keeping
the synthetic frames small makes the result practical to reproduce in CI.

## Demo artifacts

- [Synthetic adaptive-mission video](assets/demo_mission.mp4) - 300 frames,
  explicitly watermarked `SYNTHETIC INPUT - MOCK MODEL LATENCIES`.
- [Generated research report](docs/report.pdf) - benchmark numbers are read
  from the committed JSON outputs.
- [Experiment 3 output](assets/understanding_demo.json) - fused objects use
  normalized `rel_depth` (`0=near`, `1=far`), never fake metres.

## Quickstart — any Linux machine (Docker)

```bash
docker compose run test          # 15 tests
docker compose up dashboard      # live telemetry UI -> http://localhost:8090
docker compose run demo          # console mission demo
docker compose run bench         # regenerate experiment tables
docker compose up full           # real YOLO models (CPU)
```

### ROS 2 (Humble) — build, source, launch manually

```bash
docker build -f docker/Dockerfile.ros2 -t edge-perception:ros2 .
docker run --rm -it edge-perception:ros2 bash
# inside the container ROS + the workspace are already sourced:
ros2 run edge_perception_ros perception_node --ros-args -p backend:=mock
ros2 topic echo /perception/status
```

Native build on your machine:
```bash
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge \
                 ros-humble-gazebo-ros-pkgs ros-humble-xacro
pip install -e .                      # installs the core edge_perception package
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch edge_perception_ros edge_bot_sim.launch.py    # own robot + Gazebo
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}" -r 5
```

`edge_bot` is a differential-drive robot defined in
`ros2_ws/src/edge_perception_ros/urdf/edge_bot.urdf.xacro` with a 640×480
camera publishing `/camera/image_raw` — no TurtleBot3 required (a TurtleBot3
launch is also provided).

### Local Python

```bash
pip install -e ".[real,dev]"
python run_pipeline.py --source synthetic --frames 300 --backend mock
python viewer.py --image assets/test_bus.jpg --backend real --record out.jpg
python tests/test_pipeline.py && python tests/test_ros2_node_sim.py
```

## Published ROS 2 topics

| Topic | Type | Notes |
|---|---|---|
| `/perception/objects` | `vision_msgs/Detection2DArray` | class + score; **no depth in `pose.z`** (relative depth isn't metric) |
| `/perception/scene` | `std_msgs/String` (JSON) | per-object `rel_depth`, activity, risk, backend provenance |
| `/perception/risk` | `std_msgs/Float32` | scene risk |
| `/perception/semantic_image_grid` | `std_msgs/String` (JSON) | 16×12 **image-space** class grid — not a metric/bird's-eye occupancy map |
| `/perception/status` | `std_msgs/String` (JSON) | mode, attention, resource, models, backends, FPS, validated e2e latency |

## Honest limitations

- **Depth is relative, not metric.** MiDaS outputs relative depth normalized
  to [0,1]. For metric range use ZoeDepth, RGB-D, stereo, or LiDAR fusion.
- **"Decision confidence" is heuristic**, not calibrated uncertainty (no ECE/
  NLL/Brier evaluation, no ensembles/MC-dropout).
- **Tier "accuracy" is model agreement** with large-model pseudo-labels, not
  mAP against a labelled dataset.
- **`backend='real'` fails hard** if a model can't load — it never silently
  returns mock output. `backend='auto'` falls back with a visible warning,
  and every result records `requested_backend`/`actual_backend`.
- **Tracking** is class-aware IoU + centroid gating; ByteTrack/Kalman is
  future work.
- The **Jetson Dockerfile targets Orin (L4T r36.2)** and is *not* universal
  across TX2/Xavier — match the base tag to your JetPack release.
- The **dashboard has no authentication** and binds all interfaces.

## Repository layout

```
pyproject.toml            core package (pip install -e .)
edge_perception/          models/ scheduler/ fusion/ optimizer/ pipeline.py
benchmark/                latency_test (Exp 2) format_test (Exp 1)
                          accuracy_test understanding_demo (Exp 3)
ros2_ws/src/edge_perception_ros/
                          perception_node.py, urdf/edge_bot.urdf.xacro,
                          launch/{edge_bot_sim,perception_tb3}.launch.py
docker/                   Dockerfile (lite/full) Dockerfile.ros2 Dockerfile.jetson
dashboard/ viewer.py      live UI and video overlay
tests/                    test_pipeline.py (15) test_ros2_node_sim.py
docs/                     report.md, report.pdf (numbers generated from JSON)
```

## License & attribution

AGPL-3.0 (see `LICENSE`), required for compatibility with Ultralytics
YOLOv8. Full component list in [ATTRIBUTIONS.md](ATTRIBUTIONS.md). Note that
AGPL does not automatically cover closed commercial deployment of Ultralytics
models — Ultralytics offers a separate Enterprise license for that.
