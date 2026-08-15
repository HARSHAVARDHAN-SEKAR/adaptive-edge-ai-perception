"""Test suite — runs on any machine (mock backend, no GPU needed)."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from edge_perception.fusion.perception_fusion import PerceptionFusion
from edge_perception.fusion.uncertainty import scene_confidence
from edge_perception.models import base as model_base
from edge_perception.models import (  # noqa: F401
    depth,
    detection,
    pose,
    segmentation,
    vlm,
)
from edge_perception.models.base import BackendUnavailableError
from edge_perception.pipeline import PerceptionPipeline
from edge_perception.scheduler.model_selector import (
    AdaptiveScheduler,
    Attention,
    Mode,
    Resource,
    SchedulerConfig,
)
from edge_perception.scheduler.resource_monitor import ResourceSnapshot
from edge_perception.sources import SyntheticSource

FRAME = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)


def _snap(fps=20.0, gpu=40.0):
    return ResourceSnapshot(0, 30, 40, gpu, 30, 10, fps, "test")


def test_registry_and_models():
    names = model_base.available()
    assert {
        "yolo_nano",
        "yolo_small",
        "yolo_large",
        "yolo_nano_seg",
        "midas_small",
        "yolo_pose",
        "vlm_scene",
    } <= set(names)
    for n in set(names):
        m = model_base.create(n, backend="mock")
        out = m(FRAME)
        assert out.latency_ms > 0 and out.task == m.task
        assert out.requested_backend == "mock"
        assert out.actual_backend == "mock"
        m.unload()


def test_mock_world_consistency():
    """Detection, masks, depth and pose must derive from the SAME world."""
    f = FRAME.copy()
    f[100:230, 120:200] = 190  # bright intruder -> person
    det = model_base.create("yolo_large", backend="mock")(f)
    seg = model_base.create("yolo_nano_seg", backend="mock")(f)
    dep = model_base.create("midas_small", backend="mock")(f)
    pos = model_base.create("yolo_pose", backend="mock")(f)
    persons = [d for d in det.detections if d.label == "person"]
    assert persons, "intruder person must be detected"
    p = max(persons, key=lambda d: d.area)
    # a mask with the same class overlapping the same box
    assert any(m.label == "person" and m.box_xyxy == p.box_xyxy for m in seg.masks)
    # skeleton center inside the person box
    from edge_perception.models.pose import pose_center

    assert pos.poses, "pose model must find the person"
    kx, ky = pose_center(pos.poses[0].keypoints_xy, pos.poses[0].keypoints_conf)
    x1, y1, x2, y2 = p.box_xyxy
    assert x1 <= kx <= x2 and y1 <= ky <= y2
    # depth in the box reflects the object's near relative depth
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    assert dep.depth_map[cy, cx] < 0.5, "intruder should be near (low rel depth)"
    assert dep.depth_scale == "relative"


def test_real_backend_fails_hard_when_unavailable(monkeypatch=None):
    """backend='real' must never silently return mock results."""
    import builtins

    orig_import = builtins.__import__

    def no_ultralytics(name, *a, **k):
        if name == "ultralytics":
            raise ImportError("simulated missing ultralytics")
        return orig_import(name, *a, **k)

    builtins.__import__ = no_ultralytics
    try:
        for name in ("yolo_nano", "yolo_nano_seg", "yolo_pose"):
            m = model_base.create(name, backend="real")
            try:
                m.load()
                raise AssertionError(f"{name}: real backend should have raised")
            except BackendUnavailableError:
                pass
        # auto must fall back WITH a visible warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = model_base.create("yolo_nano", backend="auto")
            m.load()
            assert m.actual_backend == "mock"
            assert any("FALLING BACK TO MOCK" in str(w.message) for w in caught)
    finally:
        builtins.__import__ = orig_import
    # vlm real: no server in test env -> must also fail hard
    m = model_base.create("vlm_scene", backend="real")
    try:
        m.load()
        raise AssertionError("vlm real backend should have raised")
    except BackendUnavailableError:
        pass


def test_scheduler_escalation_and_hysteresis():
    s = AdaptiveScheduler(SchedulerConfig(cooldown_frames=5))
    assert s.decide(0.1, 0.9, _snap()).mode == Mode.PATROL
    assert s.decide(0.5, 0.9, _snap()).mode == Mode.ALERT
    assert s.decide(0.8, 0.9, _snap()).mode == Mode.ENGAGED
    assert s.decide(0.1, 0.9, _snap()).mode == Mode.ENGAGED  # hysteresis
    for _ in range(5):
        d = s.decide(0.1, 0.9, _snap())
    assert d.mode == Mode.PATROL
    assert s.decide(0.1, 0.30, _snap()).mode == Mode.ALERT  # second opinion


def test_sustained_starvation_does_not_oscillate():
    s = AdaptiveScheduler(SchedulerConfig(min_fps=6.0, resource_recovery_frames=10))
    modes = [s.decide(0.1, 0.9, _snap(fps=2.0)).mode for _ in range(60)]
    assert modes[0] == Mode.DEGRADED
    assert all(m == Mode.DEGRADED for m in modes), (
        f"oscillation under sustained starvation: {set(modes)}"
    )


def test_degraded_exits_only_after_resource_recovery():
    cfg = SchedulerConfig(min_fps=6.0, resource_recovery_frames=10)
    s = AdaptiveScheduler(cfg)
    for _ in range(5):
        assert s.decide(0.1, 0.9, _snap(fps=2.0)).mode == Mode.DEGRADED
    # resources recover — must stay DEGRADED for recovery_frames first
    for i in range(cfg.resource_recovery_frames - 1):
        assert s.decide(0.1, 0.9, _snap(fps=30.0)).mode == Mode.DEGRADED, (
            f"left DEGRADED after only {i + 1} healthy frames"
        )
    assert s.decide(0.1, 0.9, _snap(fps=30.0)).mode == Mode.PATROL


def test_high_risk_under_starvation_uses_safe_plan():
    """Risk must be considered even when compute is CRITICAL: the decision
    records the ENGAGED attention demand while the plan stays feasible."""
    s = AdaptiveScheduler()
    d = s.decide(0.9, 0.9, _snap(fps=2.0))
    assert d.mode == Mode.DEGRADED  # feasible plan
    assert d.attention == Attention.ENGAGED  # demand not silently dropped
    assert d.resource == Resource.CRITICAL
    assert "ENGAGED" in d.reason


def test_engaged_starvation_transitions_to_degraded():
    s = AdaptiveScheduler()
    assert s.decide(0.8, 0.9, _snap(fps=30.0)).mode == Mode.ENGAGED
    d = s.decide(0.8, 0.9, _snap(fps=2.0))
    assert d.mode == Mode.DEGRADED, "ENGAGED under starvation must shed load"
    assert d.attention == Attention.ENGAGED


def test_fusion_and_uncertainty():
    f = FRAME.copy()
    f[100:230, 120:200] = 190
    det = model_base.create("yolo_large", backend="mock")(f)
    dep = model_base.create("midas_small", backend="mock")(f)
    fusion = PerceptionFusion()
    scene = fusion.fuse([det, dep], f.shape)
    assert scene.objects
    for o in scene.objects:
        assert o.rel_depth is not None and 0.0 <= o.rel_depth <= 1.0
        assert 0.0 <= o.risk <= 1.0
    assert scene.depth_scale == "relative"
    assert scene.backends_used == {"yolo_large": "mock", "midas_small": "mock"}
    conf = scene_confidence(scene)
    assert conf is None or 0.0 < conf <= 1.0


def test_tracking_requires_same_class():
    fusion = PerceptionFusion()
    from edge_perception.models.base import Detection, ModelOutput

    o1 = ModelOutput("m", "detect", 0.0)
    o1.detections = [Detection("person", 0.9, (100, 100, 150, 200))]
    s1 = fusion.fuse([o1], (240, 320, 3))
    pid = s1.objects[0].object_id
    o2 = ModelOutput("m", "detect", 0.0)
    o2.detections = [Detection("chair", 0.9, (102, 102, 152, 202))]
    s2 = fusion.fuse([o2], (240, 320, 3))
    assert s2.objects[0].object_id != pid, (
        "a chair must not inherit a person's track id"
    )


def test_pipeline_end_to_end_adaptive():
    pipe = PerceptionPipeline(backend="mock", adaptive=True)
    frames = SyntheticSource(n_frames=120, event_windows=((40, 90),)).frames()
    results = pipe.run(frames)
    pipe.close()
    assert len(results) == 120
    modes = {r.decision.mode for r in results}
    assert Mode.PATROL in modes
    assert modes & {Mode.ALERT, Mode.ENGAGED}
    assert all(r.to_dict()["backends"] is not None for r in results)


def test_vlm_narration_on_engagement_and_expiry():
    pipe = PerceptionPipeline(backend="mock", adaptive=True, vlm_on_engagement=True)
    pipe.narration_ttl_frames = 15
    frames = SyntheticSource(n_frames=160, event_windows=((20, 60),)).frames()
    results = pipe.run(frames)
    pipe.close()
    assert any(r.narration for r in results), "VLM should narrate on ENGAGED"
    switches = [
        r for r in results if r.decision.switched and r.decision.mode == Mode.ENGAGED
    ]
    assert any("vlm_scene" in r.model_latencies for r in switches)
    steady = [
        r
        for r in results
        if r.decision.mode == Mode.ENGAGED and not r.decision.switched
    ]
    assert all("vlm_scene" not in r.model_latencies for r in steady)
    # expiry: narration must be cleared by the end of the calm tail
    tail = results[-10:]
    assert all(r.narration is None for r in tail), "stale narration not expired"


def test_vlm_narration_absolute_ttl_while_engaged():
    """Narration must expire even if elevated risk keeps the mode ENGAGED."""
    pipe = PerceptionPipeline(backend="mock", adaptive=True, vlm_on_engagement=True)
    pipe.narration_ttl_frames = 8
    frames = SyntheticSource(n_frames=80, event_windows=((5, 75),)).frames()
    results = pipe.run(frames)
    pipe.close()

    narrated = [r for r in results if r.narration]
    assert narrated, "expected at least one engagement narration"
    first_frame = narrated[0].frame_index
    assert any(
        r.frame_index > first_frame + pipe.narration_ttl_frames
        and r.decision.mode == Mode.ENGAGED
        and r.narration is None
        for r in results
    ), "narration did not expire while mode remained ENGAGED"


def test_degraded_frame_skip():
    pipe = PerceptionPipeline(
        backend="mock",
        adaptive=True,
        scheduler_config=SchedulerConfig(min_fps=10_000),
        degraded_frame_skip=2,
    )
    results = pipe.run(list(SyntheticSource(n_frames=40).frames()))
    pipe.close()
    degraded = [r for r in results if r.decision.mode == Mode.DEGRADED]
    skipped = [r for r in degraded if r.skipped]
    assert degraded and skipped
    assert all(not r.model_latencies for r in skipped)


def test_pipeline_fixed_baseline():
    pipe = PerceptionPipeline(
        backend="mock", adaptive=False, fixed_models=["yolo_nano"]
    )
    r = pipe.process(FRAME)
    pipe.close()
    assert r.decision.models == ["yolo_nano"]
    assert not r.decision.switched


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
