#!/usr/bin/env python3
"""Run the Adaptive Edge AI Perception Engine.

Examples:
    # zero-hardware demo (synthetic patrol with scripted intrusion events)
    python run_pipeline.py --source synthetic --frames 300

    # webcam with real models (downloads YOLO weights on first run)
    python run_pipeline.py --source 0 --backend real --device cuda

    # fixed heavyweight baseline for comparison against the adaptive scheduler
    python run_pipeline.py --source synthetic --no-adaptive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from edge_perception.pipeline import PerceptionPipeline
from edge_perception.scheduler.model_selector import Attention, SchedulerConfig
from edge_perception.sources import CameraSource, SyntheticSource

MODE_GLYPH = {"PATROL": ".", "ALERT": "!", "ENGAGED": "#", "DEGRADED": "~"}


def load_config(path: str) -> SchedulerConfig:
    p = Path(path)
    if not p.exists():
        return SchedulerConfig()
    raw = yaml.safe_load(p.read_text()) or {}
    sched = raw.get("scheduler", {})
    plans = {Attention(k): v for k, v in sched.pop("plans", {}).items()
             if k in Attention.__members__} or None
    cfg = SchedulerConfig(**{k: v for k, v in sched.items()
                             if k in SchedulerConfig.__dataclass_fields__})
    if plans:
        cfg.plans = plans
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="synthetic",
                    help="'synthetic', camera index (0), or video path")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--backend", choices=["auto", "real", "mock"], default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="fixed heavyweight model set (baseline)")
    ap.add_argument("--log", default="assets/run_log.jsonl")
    args = ap.parse_args()

    if args.source == "synthetic":
        source = SyntheticSource(n_frames=args.frames)
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        source = CameraSource(src)

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    pipe = PerceptionPipeline(
        scheduler_config=load_config(args.config),
        device=args.device,
        backend=args.backend,
        adaptive=not args.no_adaptive,
        log_path=args.log,
    )

    def report(r):
        if r.decision.switched:
            print(f"\n[frame {r.frame_index:4d}] MODE -> {r.decision.mode.value:8s} "
                  f"({r.decision.reason}) models={r.decision.models}")
        if r.frame_index % 25 == 0:
            top = max(r.scene.objects, key=lambda o: o.risk, default=None)
            if top:
                rd = (f"{top.rel_depth:.2f}" if top.rel_depth is not None
                      else "?")
                desc = f"{top.label} rel_depth={rd} risk={top.risk:.2f}"
            else:
                desc = "clear"
            print(f"[frame {r.frame_index:4d}] {MODE_GLYPH[r.decision.mode.value]} "
                  f"fps={r.fps:5.1f} lat={r.total_latency_ms:6.1f}ms "
                  f"risk={r.scene.scene_risk:.2f} | {desc}")

    try:
        results = pipe.run(source.frames(), max_frames=args.frames, on_result=report)
    finally:
        pipe.close()

    switches = pipe.scheduler.switch_log()
    print(f"\nProcessed {len(results)} frames | mode switches: {len(switches)}")
    for s in switches:
        print(f"  -> {s['mode']:8s} {s['reason']}")
    print(f"Telemetry log: {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
