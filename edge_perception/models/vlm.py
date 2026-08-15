"""Vision-Language Model tier — "what is happening?".

The most expensive tier: an on-demand scene narrator, invoked above ENGAGED
(e.g. once per event, not per frame). Answers in natural language:

    "A nearby person is moving toward the robot and may block the path."

Real backend options (auto-detected):
  - Ollama running a llava model locally:  `ollama pull llava` then
    OLLAMA_URL (default http://localhost:11434).
  - On Jetson: VILA / LLaVA via TensorRT-LLM (see jetson-containers) —
    expose it behind the same HTTP interface.

Mock backend: composes the description from the fused scene instead of
pixels — same output contract, zero hardware.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request

import numpy as np

from .base import ModelOutput, PerceptionModel, mock_sleep, register

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PROMPT = (
    "You are the perception narrator of a mobile robot. In one short "
    "sentence, describe what is happening in this camera frame and "
    "any risk to navigation."
)


@register("vlm_scene")
class VlmScene(PerceptionModel):
    task = "describe"
    cost = 8

    def _load_real(self) -> bool:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as r:
            tags = json.load(r)
        ok = any(
            "llava" in m.get("name", "").lower() or "vila" in m.get("name", "").lower()
            for m in tags.get("models", [])
        )
        if not ok:
            raise RuntimeError(
                "no llava/vila model on the Ollama server — "
                "run `ollama pull llava && ollama serve`"
            )
        return True

    def _infer_real(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        import cv2

        _ok, buf = cv2.imencode(".jpg", frame_bgr)
        payload = json.dumps(
            {
                "model": "llava",
                "prompt": PROMPT,
                "images": [base64.b64encode(buf.tobytes()).decode()],
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            out.extra["description"] = json.load(r).get("response", "").strip()
        return out

    def _infer_mock(self, frame_bgr: np.ndarray) -> ModelOutput:
        out = ModelOutput(self.name, self.task, 0.0)
        if True:
            mock_sleep(180.0)  # VLMs are slow; the scheduler must respect that
            bright = frame_bgr.mean(axis=2) > 150
            h, w = frame_bgr.shape[:2]
            if bright.sum() > 0.02 * h * w:
                out.extra["description"] = (
                    "A person is visible nearby and may partially block "
                    "the planned path."
                )
            elif bright.sum() > 0.005 * h * w:
                out.extra["description"] = (
                    "A person is visible farther away; no immediate "
                    "obstruction is inferred from this frame."
                )
            else:
                out.extra["description"] = (
                    "The corridor ahead appears clear with static clutter "
                    "only; navigation risk is low."
                )
        return out
