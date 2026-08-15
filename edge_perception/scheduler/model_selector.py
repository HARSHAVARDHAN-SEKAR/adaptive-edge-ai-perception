"""Adaptive Neural Model Scheduler — the core research contribution.

Redesigned as TWO orthogonal state machines composed into one plan
(fixes the starvation oscillation of the v1 single-machine design):

  1. ATTENTION DEMAND (what the scene needs)   PATROL / ALERT / ENGAGED
     - driven by fused scene risk and low-confidence detections
     - escalation immediate, de-escalation after `cooldown_frames` calm frames

  2. RESOURCE CONDITION (what compute allows)  NORMAL / CONSTRAINED / CRITICAL
     - driven by FPS vs target/min and GPU saturation
     - entering CRITICAL is immediate; leaving requires
       `resource_recovery_frames` consecutive healthy frames (own hysteresis)

The published plan = plans[attention] downgraded by resource condition:
  NORMAL       -> full plan for the attention level
  CONSTRAINED  -> one tier lighter (ENGAGED uses the ALERT plan, etc.)
  CRITICAL     -> minimal plan, regardless of attention (DEGRADED mode)

Risk is evaluated BEFORE load shedding, so a high-risk scene under
starvation still gets the best feasible attention rather than being
silently dropped to DEGRADED.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .resource_monitor import ResourceSnapshot


class Mode(str, Enum):
    """Published mode = attention level, or DEGRADED under CRITICAL compute."""

    PATROL = "PATROL"
    ALERT = "ALERT"
    ENGAGED = "ENGAGED"
    DEGRADED = "DEGRADED"


class Attention(str, Enum):
    PATROL = "PATROL"
    ALERT = "ALERT"
    ENGAGED = "ENGAGED"


class Resource(str, Enum):
    NORMAL = "NORMAL"
    CONSTRAINED = "CONSTRAINED"
    CRITICAL = "CRITICAL"


_LIGHTER = {
    Attention.ENGAGED: Attention.ALERT,
    Attention.ALERT: Attention.PATROL,
    Attention.PATROL: Attention.PATROL,
}

DEFAULT_PLANS: dict[Attention, list[str]] = {
    Attention.PATROL: ["yolo_nano"],
    Attention.ALERT: ["yolo_small", "midas_small"],
    Attention.ENGAGED: ["yolo_large", "yolo_nano_seg", "midas_small", "yolo_pose"],
}
MINIMAL_PLAN = ["yolo_nano"]


@dataclass
class SchedulerConfig:
    target_fps: float = 15.0  # below -> CONSTRAINED
    min_fps: float = 4.0  # below -> CRITICAL (hard floor)
    risk_alert: float = 0.35
    risk_engaged: float = 0.55
    conf_second_opinion: float = 0.45
    cooldown_frames: int = 20  # attention de-escalation hysteresis
    resource_recovery_frames: int = 10  # CRITICAL-exit hysteresis
    gpu_high: float = 92.0
    plans: dict[Attention, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_PLANS)
    )


@dataclass
class Decision:
    mode: Mode
    models: list[str]
    reason: str
    switched: bool
    timestamp: float
    attention: Attention = Attention.PATROL
    resource: Resource = Resource.NORMAL


class AdaptiveScheduler:
    def __init__(self, config: SchedulerConfig | None = None):
        self.cfg = config or SchedulerConfig()
        self.attention = Attention.PATROL
        self.resource = Resource.NORMAL
        self.mode = Mode.PATROL
        self._calm_frames = 0
        self._healthy_frames = 0
        self.history: list[Decision] = []

    # ------------------------------------------------------------------
    def _update_attention(self, risk: float, min_conf: float) -> str:
        cfg, prev = self.cfg, self.attention
        if risk >= cfg.risk_engaged:
            self.attention = Attention.ENGAGED
        elif risk >= cfg.risk_alert:
            if self.attention != Attention.ENGAGED:
                self.attention = Attention.ALERT
        elif (
            0.0 < min_conf < cfg.conf_second_opinion
            and self.attention == Attention.PATROL
        ):
            self.attention = Attention.ALERT
        if self.attention != prev:
            self._calm_frames = 0
            return f"risk={risk:.2f}"
        # calm de-escalation
        if risk < cfg.risk_alert:
            self._calm_frames += 1
            if (
                self._calm_frames >= cfg.cooldown_frames
                and self.attention != Attention.PATROL
            ):
                self.attention = Attention.PATROL
                self._calm_frames = 0
                return "calm cooldown elapsed"
        else:
            self._calm_frames = 0
        return "steady"

    def _update_resource(self, res: ResourceSnapshot) -> None:
        cfg = self.cfg
        starving = (0 < res.fps < cfg.min_fps) or res.gpu_percent >= cfg.gpu_high
        constrained = 0 < res.fps < cfg.target_fps
        if starving:
            self.resource = Resource.CRITICAL
            self._healthy_frames = 0
        elif self.resource == Resource.CRITICAL:
            # exit CRITICAL only after sustained recovery (own hysteresis)
            self._healthy_frames += 1
            if self._healthy_frames >= cfg.resource_recovery_frames:
                self.resource = Resource.CONSTRAINED if constrained else Resource.NORMAL
                self._healthy_frames = 0
        else:
            self.resource = Resource.CONSTRAINED if constrained else Resource.NORMAL

    # ------------------------------------------------------------------
    def decide(
        self, risk: float, min_detection_conf: float, resources: ResourceSnapshot
    ) -> Decision:
        prev_mode = self.mode
        att_reason = self._update_attention(risk, min_detection_conf)
        self._update_resource(resources)

        if self.resource == Resource.CRITICAL:
            mode, models = Mode.DEGRADED, list(MINIMAL_PLAN)
            reason = (
                f"compute CRITICAL (fps={resources.fps:.1f}); "
                f"attention demand stays {self.attention.value}"
            )
        else:
            eff = self.attention
            if self.resource == Resource.CONSTRAINED and eff != Attention.ENGAGED:
                # shed load in calm times only; ENGAGED demand overrides
                # CONSTRAINED (risk before normal load shedding) and accepts
                # reduced FPS down to the CRITICAL floor
                eff = _LIGHTER[eff]
            mode = Mode(eff.value)
            models = list(self.cfg.plans[eff])
            if self.resource == Resource.NORMAL:
                reason = att_reason
            elif eff != self.attention:
                reason = f"{att_reason}; CONSTRAINED -> {eff.value} plan"
            else:
                reason = f"{att_reason}; ENGAGED kept despite CONSTRAINED"

        self.mode = mode
        decision = Decision(
            mode=mode,
            models=models,
            reason=reason,
            switched=(mode != prev_mode),
            timestamp=time.time(),
            attention=self.attention,
            resource=self.resource,
        )
        if decision.switched:
            self.history.append(decision)
        return decision

    # ------------------------------------------------------------------
    def switch_log(self) -> list[dict]:
        return [
            {
                "t": d.timestamp,
                "mode": d.mode.value,
                "attention": d.attention.value,
                "resource": d.resource.value,
                "reason": d.reason,
            }
            for d in self.history
        ]
