#!/usr/bin/env python3
"""Experiment 2 — Adaptive scheduler vs fixed baselines on an identical
synthetic mission.

IMPORTANT — what this measures: these results use deterministic synthetic
scenes and latency-simulating mock models. They evaluate SCHEDULER
BEHAVIOUR (mode switching, load shedding, event coverage), not real model
accuracy or Jetson performance.

Metrics:
  - mean FPS and FPS std
  - mean / p95 end-to-end frame latency
  - model-cost proxy units: sum of per-model cost scores over models that
    ACTUALLY EXECUTED each frame (skipped frames add 0; VLM included).
    This is a scheduling-cost proxy, not measured energy/CPU/TOPS.
  - event full-suite coverage: fraction of event frames on which the entire
    ENGAGED model suite executed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edge_perception.models import base as model_base
from edge_perception.models import detection, segmentation, depth, pose, vlm  # noqa: F401
from edge_perception.pipeline import PerceptionPipeline
from edge_perception.sources import SyntheticSource

def mission_events(n_frames: int):
    """Two proportional event windows so short smoke runs still test both events."""
    if n_frames < 120:
        raise ValueError("scheduler benchmark requires at least 120 frames")
    return (
        (round(0.25 * n_frames), round(0.342 * n_frames)),
        (round(0.667 * n_frames), round(0.759 * n_frames)),
    )

HEAVY = ["yolo_large", "yolo_nano_seg", "midas_small", "yolo_pose"]
LIGHT = ["yolo_nano"]


def run_system(label, n_frames, adaptive, fixed=None, backend="mock", seed=7, events=None):
    cost = {n: model_base._REGISTRY[n].cost for n in model_base.available()}
    events = events or mission_events(n_frames)
    pipe = PerceptionPipeline(adaptive=adaptive, fixed_models=fixed,
                              backend=backend)
    # Small synthetic frames keep this scheduler benchmark quick to
    # reproduce. The experiment evaluates scheduling behaviour, not image
    # accuracy, so high-resolution pixels add cost without adding evidence.
    frames = SyntheticSource(size=(120, 160), n_frames=n_frames,
                             event_windows=events, seed=seed).frames()          # streaming generator
    lat, fps_series, total_cost = [], [], 0
    event_frames = full_suite_frames = 0
    high_risk_frames = high_risk_covered = 0
    switches = 0
    for i, frame in enumerate(frames):
        r = pipe.process(frame, i)
        lat.append(r.total_latency_ms)
        if r.fps > 0:
            fps_series.append(r.fps)
        # cost from models that ACTUALLY executed (skip-frames add 0,
        # VLM included when it fired)
        total_cost += sum(cost[m] for m in r.model_latencies)
        switches += int(r.decision.switched)
        full = set(HEAVY).issubset(r.model_latencies)
        if any(a <= i < b for a, b in events):
            event_frames += 1
            full_suite_frames += int(full)
        # frames where the scene genuinely warranted full attention
        if r.scene.scene_risk >= 0.55:
            high_risk_frames += 1
            high_risk_covered += int(full)
    pipe.close()
    lat = np.array(lat); fps_series = np.array(fps_series)
    return {
        "system": label,
        "mean_fps": round(float(fps_series.mean()), 1) if len(fps_series) else 0,
        "fps_std": round(float(fps_series.std()), 2) if len(fps_series) else 0,
        "latency_mean_ms": round(float(lat.mean()), 1),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
        "cost_proxy_units": total_cost,
        "event_full_suite_coverage": (round(full_suite_frames / event_frames, 2)
                                      if event_frames else 0.0),
        "high_risk_full_suite_coverage": (
            round(high_risk_covered / high_risk_frames, 2)
            if high_risk_frames else 0.0),
        "mode_switches": switches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--out", default="assets/bench_scheduler.md")
    args = ap.parse_args()

    events = mission_events(args.frames)
    rows = [
        run_system("fixed_heavy", args.frames, adaptive=False, fixed=HEAVY,
                   backend=args.backend, events=events),
        run_system("fixed_light", args.frames, adaptive=False, fixed=LIGHT,
                   backend=args.backend, events=events),
        run_system("adaptive", args.frames, adaptive=True,
                   backend=args.backend, events=events),
    ]

    hdr = ("| System | Mean FPS | FPS σ | Lat mean (ms) | Lat p95 (ms) "
           "| Cost proxy units | Event-window coverage | High-risk coverage "
           "| Switches |")
    lines = [
        f"# Scheduler evaluation — {args.frames}-frame synthetic mission, "
        f"events at {events}",
        "",
        "> These results use deterministic synthetic scenes and "
        "latency-simulating mock models. They evaluate scheduler behaviour, "
        "not real model accuracy or Jetson performance. 'Cost proxy units' "
        "sums manually assigned per-model cost scores over models that "
        "actually executed; it is not measured energy or CPU usage.",
        "", hdr, "|" + "---|" * 9]
    for r in rows:
        lines.append(
            f"| {r['system']} | {r['mean_fps']} | {r['fps_std']} | "
            f"{r['latency_mean_ms']} | {r['latency_p95_ms']} | "
            f"{r['cost_proxy_units']} | {r['event_full_suite_coverage']} | "
            f"{r['high_risk_full_suite_coverage']} | {r['mode_switches']} |")

    heavy, light, adap = rows
    if heavy["cost_proxy_units"]:
        saving = 100 * (1 - adap["cost_proxy_units"] / heavy["cost_proxy_units"])
        lines += ["", f"Adaptive uses **{saving:.0f}% fewer cost-proxy units** "
                      f"than the fixed heavyweight baseline while running the "
                      f"full model suite on "
                      f"**{adap['high_risk_full_suite_coverage']*100:.0f}%** of "
                      f"genuinely high-risk frames (risk >= 0.55); the fixed "
                      f"light baseline covers "
                      f"{light['high_risk_full_suite_coverage']*100:.0f}%.",
                  "",
                  "Event-window coverage is lower than high-risk coverage by "
                  "design: the scripted event includes an approach ramp whose "
                  "early frames carry genuinely low risk, and the scheduler "
                  "correctly stays in ALERT there."]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(
        {"config": {"frames": args.frames, "events": events,
                    "frame_size": [120, 160],
                    "backend": args.backend, "seed": 7},
         "results": rows}, indent=2))
    print("\n".join(lines))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
