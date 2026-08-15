#!/usr/bin/env python3
"""Generate docs/report.pdf - research report with measured results."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (CondPageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceBefore=14,
                    textColor=colors.HexColor("#1c2229"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=10,
                    textColor=colors.HexColor("#2a4a48"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], leading=14)
NOTE = ParagraphStyle("Note", parent=styles["BodyText"], fontSize=8.5,
                      textColor=colors.HexColor("#555555"))

TBL = TableStyle([
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c2229")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa4ad")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#eef1f3")]),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
])


def md_table(path):
    """Parse the first markdown table in a benchmark file."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("|") and "---" not in line:
            rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


SCHED_JSON = ROOT / "assets" / "bench_scheduler.json"


def sched_facts():
    """Single source of truth: the committed benchmark JSON."""
    if not SCHED_JSON.exists():
        return None
    d = json.loads(SCHED_JSON.read_text())
    cfg, rows = d["config"], {r["system"]: r for r in d["results"]}
    h, l, a = rows["fixed_heavy"], rows["fixed_light"], rows["adaptive"]
    return {
        "frames": cfg["frames"], "events": cfg["events"],
        "backend": cfg["backend"],
        "saving_pct": round(100 * (1 - a["cost_proxy_units"]
                                   / h["cost_proxy_units"])),
        "fps_ratio": round(a["mean_fps"] / h["mean_fps"], 1),
        "hr_cov_pct": round(100 * a["high_risk_full_suite_coverage"]),
        "ev_cov_pct": round(100 * a["event_full_suite_coverage"]),
        "light_hr_pct": round(100 * l["high_risk_full_suite_coverage"]),
    }


def main():
    F = sched_facts()
    story = []
    A = story.append
    A(Paragraph("Adaptive Multi-Modal Edge Perception<br/>"
                "for Autonomous Mobile Robots", styles["Title"]))
    A(Paragraph("Research report - Adaptive Edge AI Perception Engine",
                styles["Italic"]))
    A(Spacer(1, 6 * mm))

    A(Paragraph("1. Abstract", H1))
    abstract = (
        "Mobile robots cannot run every state-of-the-art perception model "
        "simultaneously on embedded hardware. We present an adaptive "
        "perception engine that selects, per frame, which models run - "
        "trading perception breadth against compute as a function of fused scene risk "
        "and live resource telemetry.")
    if F:
        abstract += (
            f" On a {F['frames']}-frame synthetic patrol mission with two "
            f"scripted events, the adaptive scheduler uses {F['saving_pct']}% "
            f"fewer model-cost proxy units than a fixed heavyweight pipeline "
            f"and achieves {F['fps_ratio']}x its mean FPS, while running the "
            f"full model suite on {F['hr_cov_pct']}% of genuinely high-risk "
            f"frames (fixed light baseline: {F['light_hr_pct']}%).")
    A(Paragraph(abstract, BODY))
    A(Paragraph(
        "Validation status: these scheduler results use deterministic "
        "synthetic scenes and latency-simulating mock models; they evaluate "
        "scheduler behaviour, not real model accuracy or Jetson performance. "
        "Real-model execution is supported but is not part of CI. Jetson "
        "TensorRT, Gazebo end-to-end runtime, metric depth and labelled-"
        "dataset accuracy remain deployment work.", NOTE))

    A(Paragraph("2. Problem", H1))
    A(Paragraph(
        "Edge GPUs (e.g. Jetson Orin Nano, ~40 TOPS at 7-15 W) cannot "
        "sustain a heavy detector + segmentation + depth + pose at camera rate. "
        "Fixed lightweight pipelines can omit useful safety context such as "
        "pose, segmentation and relative-depth cues exactly when it matters. "
        "Perception "
        "should therefore be a scheduled resource, not a static graph.", BODY))

    A(Paragraph("3. Architecture", H1))
    A(Paragraph(
        "Camera -> Adaptive Scheduler -> Model Plan -> Multi-Model Inference -> "
        "Perception Fusion + Confidence -> ROS2. The scheduler is a "
        "hysteresis state machine over four modes (PATROL / ALERT / ENGAGED "
        "/ DEGRADED) driven by two signals: fused scene risk and compute "
        "headroom (FPS, GPU utilization, power).", BODY))

    A(Paragraph("4. Models", H1))
    A(Table([["Task", "Model", "Cost tier", "Role"],
             ["Detect", "YOLOv8 n/s/l", "1/2/4", "object hypotheses"],
             ["Segment", "YOLOv8n-seg", "3", "instance masks"],
             ["Rel. depth", "MiDaS small", "3", "relative depth cue"],
             ["Pose", "YOLOv8n-pose", "2", "human pose / activity cue"],
             ["VLM", "LLaVA / VILA (on-demand)", "8",
              "optional scene narration"]], style=TBL,
            hAlign="LEFT"))


    A(Paragraph("5. Optimization pipeline", H1))
    A(Paragraph(
        "PyTorch -> ONNX (fixed shape, batch 1, simplified) -> TensorRT "
        "FP16/INT8, engines built on-device with entropy calibration on "
        "500-1000 mission frames. Measured on this development machine "
        "(CPU; TensorRT column to be completed on Jetson):", BODY))
    fmt = ROOT / "assets" / "bench_formats.md"
    if fmt.exists():
        A(Table(md_table(fmt), style=TBL, hAlign="LEFT"))

    A(Paragraph("6. Adaptive scheduler", H1))
    A(Paragraph(
        "Escalation is immediate (safety-first); de-escalation requires 20 "
        "consecutive calm frames (10 when compute-starved). Documented "
        "failure mode discovered during development: blocking de-escalation "
        "while compute-starved deadlocks the system in ENGAGED - the slow "
        "plan lowers FPS, which reads as starvation, which prevented calm "
        "frames from accumulating. Fix: starvation accelerates load-shedding "
        "instead of blocking it.", BODY))

    A(Paragraph("7. Experiments and results", H1))

    A(Paragraph("7.1 Experiment 1 - format optimization", H2))
    A(Paragraph("See Section 5 table. To complete on Jetson: TensorRT FP16 "
                "and INT8 rows plus power (W) per format.", BODY))

    A(Paragraph("7.2 Experiment 2 - adaptive vs fixed scheduler", H2))
    A(Paragraph(
        (f"{F['frames']}-frame mission, events at {F['events']}, "
         f"backend={F['backend']} (deterministic, hardware-independent). "
         "Cost proxy units sum manually assigned per-model scores over "
         "models that actually executed - not measured energy."
         if F else "Run `python -m benchmark.latency_test` to generate."),
        BODY))
    sched = ROOT / "assets" / "bench_scheduler.md"
    if sched.exists():
        sched_style = TableStyle(list(TBL.getCommands()) + [
            ("FONTSIZE", (0, 0), (-1, -1), 6.3),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ])
        sched_widths = [20*mm, 15*mm, 13*mm, 19*mm, 19*mm,
                        17*mm, 24*mm, 24*mm, 13*mm]
        A(Table(md_table(sched), style=sched_style, hAlign="LEFT",
                colWidths=sched_widths, repeatRows=1))
    if F:
        A(Paragraph(
            f"The adaptive scheduler runs the full suite on {F['hr_cov_pct']}% "
            f"of high-risk frames while spending {F['saving_pct']}% fewer cost "
            f"units than the fixed heavyweight baseline. Event-window coverage "
            f"({F['ev_cov_pct']}%) is lower by design: the scripted event "
            "includes an approach ramp whose early frames carry genuinely low "
            "risk, where ALERT is the correct response. High FPS variance is "
            "the signature of mode switching, not instability.", BODY))

    A(Paragraph("7.3 Experiment 3 - multi-model understanding demo", H2))
    demo = ROOT / "assets" / "understanding_demo.json"
    if demo.exists():
        d = json.loads(demo.read_text())
        A(Paragraph("Before (detection only): " +
                    "; ".join(d["before_detection_only"][:4]) + " ...", BODY))
        backend_summary = ", ".join(
            f"{name}:{info.get('actual', '?')}"
            for name, info in d.get("backends", {}).items())
        if backend_summary:
            A(Paragraph("Backend provenance: " + backend_summary, NOTE))
        rows = [["object", "rel depth", "activity", "risk %", "confidence"]]
        for o in d["after_multi_model_fusion"][:6]:
            rd = o.get("rel_depth")
            rows.append([o["object"],
                         "n/a" if rd is None else f"{float(rd):.3f}",
                         o["activity"],
                         str(o["collision_risk_pct"]),
                         str(o["decision_confidence"])])
        A(Table(rows, style=TBL, hAlign="LEFT"))
        A(Paragraph(
            "Relative depth uses 0=near and 1=far and is not metric. "
            f"VLM narration: {d.get('vlm_scene_narration', '')}", BODY))

    A(Paragraph("7.4 Tier agreement with large-model pseudo-labels", H2))
    acc = ROOT / "assets" / "bench_tier_agreement.md"
    if acc.exists():
        A(Table(md_table(acc), style=TBL, hAlign="LEFT"))
        A(Paragraph("Lighter tiers scored against yolo_large pseudo-labels "
                    "(IoU>0.5). This measures model AGREEMENT, not accuracy; "
                    "labelled-dataset mAP evaluation is future work.", BODY))

    A(Paragraph("7.5 Experiment 4 - ROS2 end-to-end latency", H2))
    A(Paragraph(
        "The perception node stamps end-to-end latency (camera header stamp "
        "-> publish) into /perception/status on every frame. To run: launch "
        "perception_tb3.launch.py and `ros2 topic echo /perception/status`.",
        BODY))

    A(CondPageBreak(38 * mm))
    A(Paragraph("8. ROS2 deployment", H1))
    A(Paragraph(
        "Topics: /perception/objects (vision_msgs/Detection2DArray; class "
        "and score only), /perception/scene (JSON including relative depth), "
        "/perception/risk, /perception/semantic_image_grid (16x12 image-space "
        "class grid; not a metric occupancy map), and /perception/status "
        "(mode, models, backend provenance, FPS and validated latency). "
        "Relative depth is deliberately not written to pose.z.", BODY))

    A(Paragraph("9. Future work", H1))
    A(Paragraph(
        "Learned scheduler (contextual bandit over model plans, reward = "
        "coverage - lambda*energy) against the hand-tuned state machine; "
        "continual learning loop (low-confidence crops auto-labeled by the "
        "large model, nano fine-tuned and re-exported to TensorRT); VLM tier "
        "served by TensorRT-LLM on Jetson.", BODY))

    A(Paragraph("10. Validation status", H1))
    A(Table([["Environment", "Status"],
             ["Python 3.10-3.12, mock backend", "CI tested"],
             ["Lite Docker image", "CI tested"],
             ["ROS 2 node logic (mocked rclpy)", "CI tested"],
             ["ROS 2 Humble colcon build", "CI tested in Humble container"],
             ["Real-model path", "supported; not part of CI"],
             ["Gazebo edge_bot runtime",
              "launch + URDF provided; runtime validation recommended"],
             ["Jetson Orin TensorRT / INT8",
              "scripts provided; device validation required"],
             ["Metric depth", "not implemented (relative only)"]],
            style=TBL, hAlign="LEFT"))


    A(Paragraph("11. References", H1))
    for ref in [
        "Jocher et al., Ultralytics YOLOv8, https://github.com/ultralytics/"
        "ultralytics (AGPL-3.0).",
        "Ranftl et al., Towards Robust Monocular Depth Estimation, TPAMI "
        "2022 (MiDaS), https://github.com/isl-org/MiDaS.",
        "Bhat et al., ZoeDepth: Zero-shot Transfer by Combining Relative and "
        "Metric Depth, 2023 - recommended for metric range.",
        "NVIDIA TensorRT documentation, https://docs.nvidia.com/deeplearning/"
        "tensorrt/.",
        "Macenski et al., Robot Operating System 2: Design, architecture, and "
        "uses in the wild, Science Robotics 7(66), 2022.",
        "Zhang et al., ByteTrack: Multi-Object Tracking by Associating Every "
        "Detection Box, ECCV 2022 - planned tracker upgrade.",
        "Guo et al., On Calibration of Modern Neural Networks, ICML 2017 - "
        "calibration metrics for future uncertainty work.",
    ]:
        A(Paragraph(f"• {ref}", NOTE))

    A(Spacer(1, 4 * mm))
    A(Paragraph("Reproduce every table: python -m benchmark.latency_test / "
                "format_test / accuracy_test / understanding_demo. All "
                "figures in this report are generated from the committed "
                "benchmark JSON.", NOTE))

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Adaptive Multi-Modal Edge Perception for Autonomous Mobile Robots",
        author="Harshavardhan Coimbatore Sekar",
        subject="Risk-aware adaptive multi-model perception scheduling")
    doc.build(story)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
