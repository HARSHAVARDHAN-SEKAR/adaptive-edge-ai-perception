"""Frame sources.

- CameraSource: webcam / CSI camera / video file via OpenCV.
- SyntheticSource: procedurally generated patrol footage with scripted
  "events" (a figure approaching the camera). Content-seeded mock models
  react to these frames, so the whole adaptive loop can be exercised and
  evaluated with zero hardware.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


class CameraSource:
    def __init__(self, src=0, size: Optional[Tuple[int, int]] = None):
        if not _HAS_CV2:
            raise RuntimeError("opencv-python is required for CameraSource")
        self.cap = cv2.VideoCapture(src)
        if size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame
        self.cap.release()


class SyntheticSource:
    """Patrol scene: drifting background; a scripted 'intruder' rectangle
    grows toward the camera during event windows, driving risk upward."""

    def __init__(self, size=(480, 640), n_frames: int = 300,
                 event_windows=((80, 140), (200, 260)), seed: int = 7):
        self.h, self.w = size
        self.n = n_frames
        self.events = event_windows
        self.rng = np.random.default_rng(seed)

    def _in_event(self, i: int) -> Optional[float]:
        for a, b in self.events:
            if a <= i < b:
                return (i - a) / max(b - a - 1, 1)     # progress 0..1
        return None

    def frames(self) -> Iterator[np.ndarray]:
        base = self.rng.integers(40, 90, (self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.n):
            frame = base.copy()
            # slow global drift = robot moving
            frame = np.roll(frame, shift=i % 7, axis=1)
            # ambient noise
            frame = np.clip(
                frame.astype(np.int16)
                + self.rng.integers(-6, 6, frame.shape, dtype=np.int16),
                0, 255).astype(np.uint8)
            prog = self._in_event(i)
            if prog is not None:
                # intruder grows as it approaches -> more pixels, higher
                # content-sum -> mock models emit closer/larger detections
                bw = int(30 + prog * 0.35 * self.w)
                bh = int(60 + prog * 0.6 * self.h)
                x = int(self.w * 0.55 - bw / 2)
                y = self.h - bh
                frame[y:self.h, x:x + bw] = (200, 180, 160)
            yield frame
