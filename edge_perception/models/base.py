"""Base interfaces for all perception models.

Backend contract (strict — a user must never unknowingly get mock results):
  - "real": any import / weight / load / server failure raises immediately.
  - "auto": real if possible, otherwise mock with a VISIBLE warning.
  - "mock": deterministic synthetic outputs derived from one shared
            synthetic ground truth per frame (see mock_world), so mock
            detection / masks / depth / pose are mutually consistent.

Every ModelOutput reports requested_backend and actual_backend.
"""

from __future__ import annotations

import time
import warnings
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class BackendUnavailableError(RuntimeError):
    """backend='real' was requested but cannot be provided."""


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]  # pixels
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class SegmentationMask:
    label: str
    confidence: float
    mask: np.ndarray  # HxW bool
    box_xyxy: tuple[float, float, float, float] | None = None


@dataclass
class PoseKeypoints:
    confidence: float
    keypoints_xy: np.ndarray  # (17, 2)
    keypoints_conf: np.ndarray  # (17,)
    activity: str = "unknown"


@dataclass
class ModelOutput:
    model_name: str
    task: str
    latency_ms: float
    requested_backend: str = "auto"
    actual_backend: str = "mock"  # "real" | "mock"
    detections: list[Detection] = field(default_factory=list)
    masks: list[SegmentationMask] = field(default_factory=list)
    depth_map: np.ndarray | None = None  # HxW float32 RELATIVE depth
    depth_scale: str = "relative"  # never metric from mono
    poses: list[PoseKeypoints] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Model interface
# --------------------------------------------------------------------------


class PerceptionModel(ABC):
    task: str = "generic"
    name: str = "base"
    cost: int = 1

    def __init__(self, device: str = "cpu", backend: str = "auto"):
        assert backend in ("auto", "real", "mock"), backend
        self.device = device
        self.backend = backend
        self._loaded = False
        self._real_ready = False
        self._load_time_s: float = 0.0

    @property
    def actual_backend(self) -> str:
        return "real" if self._real_ready else "mock"

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.perf_counter()
        if self.backend == "mock":
            self._real_ready = False
        else:
            try:
                self._real_ready = bool(self._load_real())
            except Exception as e:
                self._real_ready = False
                if self.backend == "real":
                    raise BackendUnavailableError(
                        f"{self.name}: backend='real' requested but unavailable: {e}"
                    ) from e
            if self.backend == "real" and not self._real_ready:
                raise BackendUnavailableError(
                    f"{self.name}: backend='real' requested but the real "
                    "implementation could not be loaded"
                )
            if self.backend == "auto" and not self._real_ready:
                warnings.warn(
                    f"[{self.name}] real backend unavailable — FALLING BACK "
                    "TO MOCK outputs (backend='auto')",
                    stacklevel=2,
                )
        self._load_time_s = time.perf_counter() - t0
        self._loaded = True

    def unload(self) -> None:
        self._unload_impl()
        self._loaded = False
        self._real_ready = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # -- inference ---------------------------------------------------------
    def __call__(self, frame_bgr: np.ndarray) -> ModelOutput:
        if not self._loaded:
            self.load()
        t0 = time.perf_counter()
        out = (
            self._infer_real(frame_bgr)
            if self._real_ready
            else self._infer_mock(frame_bgr)
        )
        out.latency_ms = (time.perf_counter() - t0) * 1000.0
        out.model_name = self.name
        out.task = self.task
        out.requested_backend = self.backend
        out.actual_backend = self.actual_backend
        return out

    # -- to implement ------------------------------------------------------
    @abstractmethod
    def _load_real(self) -> bool:
        """Load the real implementation. Return True on success.
        Raise or return False when unavailable."""

    @abstractmethod
    def _infer_real(self, frame_bgr: np.ndarray) -> ModelOutput: ...

    @abstractmethod
    def _infer_mock(self, frame_bgr: np.ndarray) -> ModelOutput: ...

    def _unload_impl(self) -> None:
        pass


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def create(name: str, **kwargs) -> PerceptionModel:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------
# Shared synthetic ground truth (one world per frame; all mocks derive
# from it so boxes, masks, depth and pose are mutually consistent)
# --------------------------------------------------------------------------


@dataclass
class MockObject:
    label: str
    box_xyxy: tuple[float, float, float, float]
    rel_depth: float  # 0 = nearest .. 1 = farthest (RELATIVE)
    has_pose: bool = False


def _frame_seed(frame: np.ndarray) -> int:
    return int(frame[::64, ::64].sum()) % (2**31)


def mock_rng(frame: np.ndarray, salt: str = "") -> np.random.Generator:
    """Deterministic per-frame RNG for mutually reproducible mock outputs."""
    salt32 = zlib.crc32(salt.encode("utf-8")) & 0x7FFFFFFF
    return np.random.default_rng((_frame_seed(frame) ^ salt32) & 0x7FFFFFFF)


def mock_world(frame: np.ndarray) -> list[MockObject]:
    """Deterministic synthetic scene truth for a frame."""
    rng = np.random.default_rng(_frame_seed(frame))
    h, w = frame.shape[:2]
    objs: list[MockObject] = []

    # the scripted intruder: a large bright region -> a close person
    bright = frame.mean(axis=2) > 150
    if bright.sum() > 0.005 * h * w:
        ys, xs = np.nonzero(bright)
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        size_frac = (box[2] - box[0]) / w
        objs.append(
            MockObject(
                "person",
                box,
                rel_depth=float(np.clip(0.55 - size_frac, 0.02, 0.9)),
                has_pose=True,
            )
        )

    # Ambient synthetic patrol clutter is deliberately benign. Scripted
    # bright event windows are the only person intrusions, which keeps the
    # scheduler benchmark controlled and makes event coverage interpretable.
    labels = ["chair", "backpack"]
    probs = [0.65, 0.35]
    for _ in range(int(rng.integers(1, 4))):
        label = str(rng.choice(labels, p=probs))
        if label == "person":
            bw, bh = rng.uniform(0.03, 0.08) * w, rng.uniform(0.08, 0.18) * h
            y1 = rng.uniform(0.05, 0.35) * h
            depth = rng.uniform(0.6, 0.95)
        else:
            bw, bh = rng.uniform(0.08, 0.25) * w, rng.uniform(0.12, 0.4) * h
            y1 = rng.uniform(0, h - bh)
            depth = rng.uniform(0.3, 0.9)
        x1 = rng.uniform(0, w - bw)
        objs.append(
            MockObject(
                label,
                (x1, y1, x1 + bw, min(y1 + bh, h - 1.0)),
                rel_depth=float(depth),
                has_pose=(label == "person"),
            )
        )
    return objs


_MOCK_LATENCY_RNG = np.random.default_rng(20260815)


def mock_sleep(ms_mean: float, ms_jitter: float = 0.15) -> None:
    """Sleep for a reproducible synthetic model-latency sample."""
    delay_ms = _MOCK_LATENCY_RNG.normal(ms_mean, ms_mean * ms_jitter)
    time.sleep(max(0.0, float(delay_ms)) / 1000.0)
