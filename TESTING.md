# Testing & manual launch guide

Every command below was executed during development; expected output is shown.

## 0. Prerequisites
Linux (any distro) with **either** Docker **or** Python 3.10+.

## 1. Test suite — proves the engine works on your machine
```bash
docker compose run test          # or: pip install -e . && python tests/test_pipeline.py
```
Expected: `15 tests passed`

```bash
python tests/test_ros2_node_sim.py
```
Expected: `ROS2 NODE SIMULATION: ALL CHECKS PASSED`
(Drives the real ROS node code with mocked rclpy — no ROS install needed.)

## 2. Mission demo — watch the scheduler switch modes
```bash
docker compose run demo
```
Expected: lines such as `-> ALERT risk=0.36`, `-> ENGAGED`, `-> PATROL calm cooldown elapsed`

## 3. Live dashboard
```bash
docker compose up dashboard      # http://localhost:8090
```
Expected: mode banner changing, attention timeline filling, object table with
`rel depth` column, health JSON at `/api/state`.

## 4. Experiments (regenerate every table in README/report)
```bash
docker compose run bench
python -m benchmark.latency_test --frames 120     # Experiment 2
python docs/make_report_pdf.py                     # PDF rebuilt from JSON
```

## 5. Real models
```bash
pip install -e ".[real]"
python viewer.py --image assets/test_bus.jpg --backend real --record out.jpg
```
`--backend real` fails loudly if weights/ultralytics are missing — it never
silently substitutes mock output.

## 6. ROS 2 — build, source, launch MANUALLY

### 6a. Inside Docker (no ROS needed on the host)
```bash
docker build -f docker/Dockerfile.ros2 -t edge-perception:ros2 .
docker run --rm -it edge-perception:ros2 bash
# ROS + workspace are auto-sourced by the entrypoint; verify:
ros2 pkg list | grep edge_perception_ros
ros2 run edge_perception_ros perception_node --ros-args -p backend:=mock &
ros2 topic list
ros2 topic echo /perception/status --once
```

### 6b. Natively, with explicit build + source
```bash
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge \
                 ros-humble-xacro ros-humble-robot-state-publisher \
                 ros-humble-gazebo-ros-pkgs
pip install -e .                       # core package on the PYTHONPATH properly
cd ros2_ws
colcon build --symlink-install
source install/setup.bash              # re-run in EVERY new terminal
ros2 run edge_perception_ros perception_node --ros-args -p backend:=mock
```

### 6c. Full Gazebo simulation with the included robot
```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch edge_perception_ros edge_bot_sim.launch.py
# second terminal — drive it:
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.3}}" -r 5
# third terminal — watch perception:
ros2 topic echo /perception/status
ros2 topic echo /perception/scene
rviz2 -d ros2_ws/src/edge_perception_ros/perception.rviz
```
`edge_bot` (urdf/edge_bot.urdf.xacro) is a differential-drive robot with a
640×480 camera publishing `/camera/image_raw`. A TurtleBot3 variant is also
provided: `ros2 launch edge_perception_ros perception_tb3.launch.py`.

Verify the URDF before launching:
```bash
xacro ros2_ws/src/edge_perception_ros/urdf/edge_bot.urdf.xacro > /tmp/edge_bot.urdf
check_urdf /tmp/edge_bot.urdf
```

## 7. GitHub repository

Target repository:

```text
HARSHAVARDHAN-SEKAR/adaptive-edge-ai-perception
```

For future local updates after cloning the repository:

```bash
git status
git add <changed-files>
git commit -m "Describe the change"
git push
```

Do not commit ROS build products, downloaded model weights, ONNX/TensorRT
engines, virtual environments, caches, or local run logs; `.gitignore`
already excludes them.

## Troubleshooting
| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: edge_perception` under ROS | run `pip install -e .` from the repo root |
| `package 'edge_perception_ros' not found` | `source install/setup.bash` in that terminal |
| `BackendUnavailableError` | intended: `--backend real` without ultralytics/weights |
| e2e latency shows `null` in Gazebo | launch with `use_sim_time:=true` |
| Gazebo starts but no images | check `ros2 topic hz /camera/image_raw` |
