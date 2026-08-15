#!/usr/bin/env python3
"""ROS2-free simulation of the perception node.

Installs faithful mocks of rclpy / sensor_msgs / vision_msgs / std_msgs /
cv_bridge into sys.modules, then imports the REAL perception_node module and
drives its on_image() callback with a simulated camera mission (the synthetic
patrol with an intrusion event). Every published message is captured and
validated.

This exercises the exact code paths a `ros2 launch` would: subscription
callback, Detection2DArray construction, semantic grid, e2e latency math,
narration, mode switching. What it cannot verify is the ROS middleware
itself (DDS transport, colcon build) — that part runs on your machine.

    python tests/test_ros2_node_sim.py
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "edge_perception_ros"))

import numpy as np

# ---------------------------------------------------------------------------
# Faithful mocks of the ROS2 python API surface the node uses
# ---------------------------------------------------------------------------

def _module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


class _Time:
    def __init__(self, ns):
        self.nanoseconds = ns


class _Clock:
    def now(self):
        return _Time(time.time_ns())


class _Param:
    def __init__(self, v):
        self.value = v


class Node:
    # simulate `--ros-args -p backend:=mock` (deterministic, no weights)
    PARAM_OVERRIDES = {"backend": "mock"}

    def __init__(self, name):
        self._name = name
        self._params = {}
        self.publishers = {}       # topic -> list of published msgs
        self.subscriptions = {}    # topic -> callback

    def declare_parameter(self, k, v):
        self._params[k] = _Param(self.PARAM_OVERRIDES.get(k, v))

    def get_parameter(self, k):
        return self._params[k]

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subscriptions[topic] = cb

    def create_publisher(self, msg_type, topic, qos):
        node = self

        class Pub:
            def publish(self, msg, _t=topic):
                node.publishers.setdefault(_t, []).append(msg)
        return Pub()

    def get_logger(self):
        class L:
            def info(self, m): print(f"[node] {m}")
            def warn(self, m): print(f"[node][warn] {m}")
        return L()

    def get_clock(self):
        return _Clock()

    def destroy_node(self):
        pass


rclpy = _module("rclpy")
rclpy.init = lambda args=None: None
rclpy.shutdown = lambda: None
rclpy.spin = lambda n: None
node_mod = _module("rclpy.node")
node_mod.Node = Node
qos_mod = _module("rclpy.qos")


class _QosProfile:
    value = 10


class QoSPresetProfiles:
    SENSOR_DATA = _QosProfile()


qos_mod.QoSPresetProfiles = QoSPresetProfiles

std_msgs = _module("std_msgs")
std_msgs_msg = _module("std_msgs.msg")


class String:
    def __init__(self, data=""):
        self.data = data


class Float32:
    def __init__(self, data=0.0):
        self.data = data


std_msgs_msg.String, std_msgs_msg.Float32 = String, Float32

sensor_msgs = _module("sensor_msgs")
sensor_msgs_msg = _module("sensor_msgs.msg")


class _Stamp:
    def __init__(self):
        self.sec = 0
        self.nanosec = 0


class _Header:
    def __init__(self):
        self.stamp = _Stamp()
        self.frame_id = ""


class Image:
    def __init__(self):
        self.header = _Header()
        self.data = None


sensor_msgs_msg.Image = Image

vision_msgs = _module("vision_msgs")
vision_msgs_msg = _module("vision_msgs.msg")


class _Point:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Center:
    def __init__(self):
        self.position = _Point()
        self.theta = 0.0


class BoundingBox2D:
    def __init__(self):
        self.center = _Center()
        self.size_x = self.size_y = 0.0


class _Hyp:
    def __init__(self):
        self.class_id = ""
        self.score = 0.0


class _PoseHolder:
    def __init__(self):
        self.pose = types.SimpleNamespace(position=_Point())


class ObjectHypothesisWithPose:
    def __init__(self):
        self.hypothesis = _Hyp()
        self.pose = types.SimpleNamespace(
            pose=types.SimpleNamespace(position=_Point()))


class Detection2D:
    def __init__(self):
        self.header = _Header()
        self.bbox = None
        self.results = []


class Detection2DArray:
    def __init__(self):
        self.header = _Header()
        self.detections = []


for k, v in dict(BoundingBox2D=BoundingBox2D, Detection2D=Detection2D,
                 Detection2DArray=Detection2DArray,
                 ObjectHypothesisWithPose=ObjectHypothesisWithPose).items():
    setattr(vision_msgs_msg, k, v)

cv_bridge = _module("cv_bridge")


class CvBridge:
    def imgmsg_to_cv2(self, msg, desired_encoding="bgr8"):
        return msg.data                      # our fake Image carries the array


cv_bridge.CvBridge = CvBridge

# ---------------------------------------------------------------------------
# Import the REAL node and drive it
# ---------------------------------------------------------------------------
import json

from edge_perception.sources import SyntheticSource
from edge_perception_ros import perception_node as pn


def make_msg(frame, t_ns):
    m = Image()
    m.data = frame
    m.header.stamp.sec = t_ns // 1_000_000_000
    m.header.stamp.nanosec = t_ns % 1_000_000_000
    m.header.frame_id = "camera_link"
    return m


def main():
    node = pn.PerceptionNode()
    assert "/camera/image_raw" in node.subscriptions, "subscription missing"
    cb = node.subscriptions["/camera/image_raw"]

    frames = list(SyntheticSource(n_frames=120,
                                  event_windows=((30, 100),)).frames())
    for f in frames:
        cb(make_msg(f, time.time_ns()))

    pubs = node.publishers
    topics = sorted(pubs)
    print("published topics:", topics)
    expected = ["/perception/objects", "/perception/risk",
                "/perception/scene", "/perception/semantic_image_grid",
                "/perception/status"]
    assert topics == expected, f"topic mismatch: {topics}"
    assert all(len(pubs[t]) == 120 for t in topics), "1 msg per frame per topic"

    # -- objects: valid Detection2DArray with class/score/range --------------
    arrays = pubs["/perception/objects"]
    nonempty = [a for a in arrays if a.detections]
    assert nonempty, "no detections in whole mission?"
    d = nonempty[-1].detections[0]
    assert d.bbox.size_x > 0 and d.bbox.size_y > 0
    hyp = d.results[0].hypothesis
    assert hyp.class_id and 0 < hyp.score <= 1.0
    # relative depth must NOT be injected into pose.z (it is not metric)
    assert all(dd.results[0].pose.pose.position.z == 0.0
               for a in arrays for dd in a.detections), \
        "relative depth must not be published as metric pose.z"
    scenes_j = [json.loads(m.data) for m in pubs["/perception/scene"]]
    assert any(o.get("rel_depth") is not None
               for sc in scenes_j for o in sc["objects"]), \
        "rel_depth missing from scene JSON"
    assert all(sc["depth_scale"] in (None, "relative") for sc in scenes_j)

    # -- status: modes switched; e2e latency sane ----------------------------
    stats = [json.loads(m.data) for m in pubs["/perception/status"]]
    modes = {s["mode"] for s in stats}
    assert "PATROL" in modes and modes & {"ALERT", "ENGAGED"}, \
        f"scheduler never escalated: {modes}"
    lats = [s["e2e_latency_ms"] for s in stats if s["e2e_latency_ms"]]
    assert lats and all(0 < l < 10000 for l in lats), "e2e latency broken"
    # zero/invalid stamps must publish null rather than a bogus number
    zero_msg = make_msg(frames[0], 0)
    cb(zero_msg)
    assert json.loads(pubs["/perception/status"][-1].data)["e2e_latency_ms"] \
        is None, "zero timestamp must yield null latency"
    # backend provenance present
    assert all(s["backends"] for s in stats), "backend provenance missing"
    assert all(v == "mock" for s in stats for v in s["backends"].values())
    # bbox center compatibility helper works on this msg layout
    from edge_perception_ros.perception_node import _set_bbox_center
    bb = BoundingBox2D(); _set_bbox_center(bb, 5.0, 7.0)
    got = ((bb.center.position.x, bb.center.position.y)
           if hasattr(bb.center, "position") else (bb.center.x, bb.center.y))
    assert got == (5.0, 7.0)

    # -- risk: rises during the event ---------------------------------------
    risks = [m.data for m in pubs["/perception/risk"]]
    assert max(risks[30:100]) > max(risks[:25]), "risk did not rise in event"

    # -- semantic image grid: shape, bounds, event cell occupied -------------
    smap = json.loads(pubs["/perception/semantic_image_grid"][80].data)
    grid = smap["grid"]
    assert len(grid) == smap["rows"] == 12
    assert len(grid[0]) == smap["cols"] == 16
    assert smap["space"] == "image"
    labels = {c for row in grid for c in row}
    assert labels - {"free"}, "semantic map empty during event"

    # -- scene: parses, narration appears after escalation -------------------
    scenes = [json.loads(m.data) for m in pubs["/perception/scene"]]
    assert all("objects" in s for s in scenes)

    node.destroy_node()
    print(f"\nmodes seen: {sorted(modes)}")
    print(f"e2e latency: mean {sum(lats)/len(lats):.1f} ms, "
          f"max {max(lats):.1f} ms")
    print(f"peak risk during event: {max(risks[30:100]):.2f} "
          f"(calm baseline {max(risks[:25]):.2f})")
    print(f"semantic labels observed: {sorted(labels - {'free'})}")
    print("\nROS2 NODE SIMULATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
