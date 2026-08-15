#!/usr/bin/env python3
"""ROS2 node wrapping the Adaptive Edge AI Perception Engine.

Subscribes:
    /camera/image_raw                  sensor_msgs/Image

Publishes:
    /perception/objects                vision_msgs/Detection2DArray
    /perception/scene                  std_msgs/String (JSON, incl. rel_depth)
    /perception/risk                   std_msgs/Float32
    /perception/semantic_image_grid    std_msgs/String (16x12 IMAGE-SPACE
                                       class grid; NOT a metric/bird's-eye
                                       occupancy map)
    /perception/status                 std_msgs/String (mode, models,
                                       backends, fps, latencies)

Notes:
  - Monocular depth is RELATIVE; it is published inside the scene JSON as
    `rel_depth` and deliberately NOT written into pose.position.z.
  - BoundingBox2D center fields differ across vision_msgs versions
    (Pose2D `center.x/y` on Humble-era, `center.position.x/y` on newer);
    both are supported at runtime.
  - e2e latency (camera stamp -> publish) is validated: zero stamps,
    negative values, and implausible values (>10 s, e.g. wall-vs-sim clock
    mismatch) are published as null. Set use_sim_time:=true under Gazebo.
"""
from __future__ import annotations

import json

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from vision_msgs.msg import (BoundingBox2D, Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)

from edge_perception.pipeline import PerceptionPipeline

_MAX_PLAUSIBLE_E2E_MS = 10_000.0


def _set_bbox_center(bbox: BoundingBox2D, x: float, y: float) -> None:
    """Support both vision_msgs center layouts (Pose2D vs point-based)."""
    c = bbox.center
    if hasattr(c, "position"):          # newer vision_msgs
        c.position.x = x
        c.position.y = y
    else:                                # Humble-era Pose2D: center.x / .y
        c.x = x
        c.y = y


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("edge_perception")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("backend", "auto")
        self.declare_parameter("adaptive", True)

        self.pipeline = PerceptionPipeline(
            device=self.get_parameter("device").value,
            backend=self.get_parameter("backend").value,
            adaptive=self.get_parameter("adaptive").value,
        )
        self.bridge = CvBridge()
        self._i = 0

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image, "/camera/image_raw",
                                 self.on_image, sensor_qos)
        self.pub_objects = self.create_publisher(
            Detection2DArray, "/perception/objects", 10)
        self.pub_scene = self.create_publisher(String, "/perception/scene", 10)
        self.pub_risk = self.create_publisher(Float32, "/perception/risk", 10)
        self.pub_status = self.create_publisher(String, "/perception/status", 10)
        self.pub_grid = self.create_publisher(
            String, "/perception/semantic_image_grid", 10)
        self.get_logger().info("edge_perception node up")

    # ------------------------------------------------------------------
    def _semantic_image_grid(self, scene, shape, cols=16, rows=12):
        """Coarse IMAGE-SPACE class grid (which grid cells of the camera
        image contain which class). Not metric, not map-aligned — a
        planner-ready occupancy map requires depth + intrinsics + TF and is
        listed as future work."""
        h, w = shape[:2]
        grid = [["free"] * cols for _ in range(rows)]
        for o in scene.objects:
            x1, y1, x2, y2 = o.box_xyxy
            r0 = max(0, min(int(y1 / h * rows), rows - 1))
            r1 = max(0, min(int(y2 / h * rows), rows - 1))
            c0 = max(0, min(int(x1 / w * cols), cols - 1))
            c1 = max(0, min(int(x2 / w * cols), cols - 1))
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    grid[r][c] = o.label
        return grid

    def _e2e_latency_ms(self, msg):
        stamp_ns = (msg.header.stamp.sec * 1_000_000_000
                    + msg.header.stamp.nanosec)
        if stamp_ns <= 0:
            return None
        e2e = (self.get_clock().now().nanoseconds - stamp_ns) / 1e6
        if e2e < 0 or e2e > _MAX_PLAUSIBLE_E2E_MS:
            # clock-domain mismatch (e.g. sim stamp vs wall clock without
            # use_sim_time) — do not publish a meaningless number
            return None
        return round(e2e, 1)

    # ------------------------------------------------------------------
    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            result = self.pipeline.process(frame, self._i)
        except Exception as e:
            self.get_logger().error(f"perception failed on frame: {e}")
            return
        self._i += 1

        arr = Detection2DArray()
        arr.header = msg.header
        for obj in result.scene.objects:
            det = Detection2D()
            det.header = msg.header
            x1, y1, x2, y2 = obj.box_xyxy
            det.bbox = BoundingBox2D()
            _set_bbox_center(det.bbox, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = obj.label
            hyp.hypothesis.score = float(obj.confidence)
            # rel_depth intentionally NOT written to pose.z (not metric);
            # consumers read it from /perception/scene JSON
            det.results.append(hyp)
            arr.detections.append(det)
        self.pub_objects.publish(arr)

        self.pub_scene.publish(String(data=json.dumps(result.scene.to_dict())))
        self.pub_risk.publish(Float32(data=float(result.scene.scene_risk)))
        self.pub_grid.publish(String(data=json.dumps({
            "stamp": {"sec": msg.header.stamp.sec,
                      "nanosec": msg.header.stamp.nanosec},
            "cols": 16, "rows": 12, "space": "image",
            "grid": self._semantic_image_grid(result.scene, frame.shape),
        })))
        self.pub_status.publish(String(data=json.dumps({
            "mode": result.decision.mode.value,
            "attention": result.decision.attention.value,
            "resource": result.decision.resource.value,
            "models": result.decision.models,
            "backends": result.scene.backends_used,
            "fps": round(result.fps, 1),
            "inference_latency_ms": round(result.total_latency_ms, 1),
            "e2e_latency_ms": self._e2e_latency_ms(msg),
        })))

    def destroy_node(self):
        self.pipeline.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
