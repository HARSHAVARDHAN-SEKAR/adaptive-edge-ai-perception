#!/usr/bin/env python3
"""Live telemetry dashboard for the perception engine.

Runs the pipeline in a background thread (synthetic source by default, or a
camera) and serves the UI + a JSON state endpoint on http://localhost:8090.

    python dashboard/server.py                # synthetic loop, mock backend
    python dashboard/server.py --source 0     # webcam
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_perception.pipeline import PerceptionPipeline           # noqa: E402
from edge_perception.sources import CameraSource, SyntheticSource  # noqa: E402

STATE = {"latest": None, "history": deque(maxlen=400), "switches": [],
         "pipeline_alive": True, "last_error": None, "last_frame_t": None,
         "actual_backend": None}
LOCK = threading.Lock()


def pipeline_loop(source_arg: str, backend: str, device: str):
    try:
        pipe = PerceptionPipeline(backend=backend, device=device, adaptive=True)
    except Exception as e:
        with LOCK:
            STATE["pipeline_alive"] = False
            STATE["last_error"] = f"pipeline init failed: {e}"
        return
    retry_s = 1.0
    while True:
        try:
            if source_arg == "synthetic":
                src = SyntheticSource(n_frames=600)
            else:
                s = int(source_arg) if source_arg.isdigit() else source_arg
                src = CameraSource(s)
            for i, frame in enumerate(src.frames()):
                r = pipe.process(frame, i)
                snap = pipe.monitor.snapshot()
                with LOCK:
                    STATE["latest"] = {**r.to_dict(),
                                       "resources": snap.to_dict()}
                    STATE["history"].append({
                        "t": time.time(), "mode": r.decision.mode.value,
                        "fps": round(r.fps, 1), "risk": r.scene.scene_risk,
                        "latency": round(r.total_latency_ms, 1),
                    })
                    STATE["switches"] = pipe.scheduler.switch_log()[-20:]
                    STATE["last_frame_t"] = time.time()
                    STATE["actual_backend"] = (
                        r.scene.backends_used or None)
                    STATE["pipeline_alive"] = True
                    STATE["last_error"] = None
            retry_s = 1.0                       # source finished cleanly
        except Exception as e:
            with LOCK:
                STATE["pipeline_alive"] = False
                STATE["last_error"] = str(e)
            time.sleep(retry_s)                 # camera retry backoff
            retry_s = min(retry_s * 2, 30.0)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(Path(__file__).parent), **kw)

    def do_GET(self):
        if self.path == "/api/state":
            with LOCK:
                age = (round((time.time() - STATE["last_frame_t"]) * 1000)
                       if STATE["last_frame_t"] else None)
                body = json.dumps({
                    "latest": STATE["latest"],
                    "history": list(STATE["history"]),
                    "switches": STATE["switches"],
                    "health": {
                        "pipeline_alive": STATE["pipeline_alive"],
                        "last_frame_age_ms": age,
                        "last_error": STATE["last_error"],
                        "actual_backend": STATE["actual_backend"],
                    },
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            if self.path == "/":
                self.path = "/index.html"
            super().do_GET()

    def log_message(self, *a):  # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    threading.Thread(target=pipeline_loop,
                     args=(args.source, args.backend, args.device),
                     daemon=True).start()
    print(f"dashboard: http://localhost:{args.port}")
    # NOTE: binds all interfaces with NO auth — on a LAN this exposes robot
    # telemetry to every reachable host. Bind 127.0.0.1 or firewall it.
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
