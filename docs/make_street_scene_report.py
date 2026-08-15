#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = ROOT / "assets" / "street_test.png"
JSON_PATH = ROOT / "assets" / "street_test_real_cuda.json"
PDF_PATH = ROOT / "docs" / "street_scene_real_inference_report.pdf"


with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)


before = data.get("before_detection_only", [])
fused = data.get("after_multi_model_fusion", [])
backends = data.get("backends", {})


def class_from_detection_string(text: str) -> str:
    return text.replace(" detected", "").strip()


before_counts = Counter(
    class_from_detection_string(x)
    for x in before
)

fused_counts = Counter(
    str(x.get("object", "unknown"))
    for x in fused
)

risks = [
    float(x.get("collision_risk_pct", 0))
    for x in fused
]

confidences = [
    float(x.get("decision_confidence", 0))
    for x in fused
]

depths = [
    float(x["rel_depth"])
    for x in fused
    if x.get("rel_depth") is not None
]

max_risk = max(risks) if risks else 0
min_conf = min(confidences) if confidences else 0
max_conf = max(confidences) if confidences else 0
min_depth = min(depths) if depths else 0
max_depth = max(depths) if depths else 0


styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "CustomTitle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=20,
    leading=24,
    spaceAfter=8 * mm,
)

subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    alignment=TA_CENTER,
    fontSize=10,
    leading=14,
    spaceAfter=8 * mm,
)

heading = styles["Heading2"]
body = styles["BodyText"]
body.leading = 15


doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=18 * mm,
    bottomMargin=18 * mm,
)

story = []

story.append(
    Paragraph(
        "Real Street-Scene Inference Validation",
        title_style,
    )
)

story.append(
    Paragraph(
        "Adaptive Edge AI Perception Engine for Autonomous Mobile Robots",
        subtitle_style,
    )
)

story.append(
    Paragraph(
        "<b>Validation type:</b> Single-image real-backend functional validation<br/>"
        "<b>Requested backend:</b> real<br/>"
        "<b>Execution device:</b> CUDA<br/>"
        "<b>Input:</b> Urban street scene",
        body,
    )
)

story.append(Spacer(1, 8 * mm))


if IMAGE_PATH.exists():
    img = Image(str(IMAGE_PATH))
    max_width = 165 * mm
    max_height = 95 * mm

    scale = min(
        max_width / img.imageWidth,
        max_height / img.imageHeight,
    )

    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale

    story.append(img)
    story.append(Spacer(1, 6 * mm))


story.append(Paragraph("1. Purpose", heading))

story.append(
    Paragraph(
        "This experiment validates functional execution of the real perception "
        "backends on a complex urban street image. It is a functional smoke test "
        "of the real inference path rather than a formal object-detection accuracy, "
        "depth-accuracy, or GPU-performance benchmark.",
        body,
    )
)

story.append(Spacer(1, 5 * mm))


story.append(Paragraph("2. Backend execution status", heading))

backend_table = [
    ["Module", "Requested", "Actual", "Status"]
]

for name, info in backends.items():
    actual = str(info.get("actual", "unknown"))
    requested = str(info.get("requested", "unknown"))

    if actual == "real":
        status = "PASS"
    elif name == "vlm_scene" and actual == "unavailable":
        status = "OPTIONAL / SKIPPED"
    else:
        status = actual.upper()

    backend_table.append([
        name,
        requested,
        actual,
        status,
    ])

table = Table(
    backend_table,
    colWidths=[50 * mm, 30 * mm, 30 * mm, 42 * mm],
)

table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)

story.append(table)
story.append(Spacer(1, 7 * mm))


story.append(Paragraph("3. Detection-only result", heading))

story.append(
    Paragraph(
        f"The baseline detection-only stage produced "
        f"<b>{len(before)} detections</b>.",
        body,
    )
)

before_table = [["Class", "Count"]]

for name, count in sorted(before_counts.items()):
    before_table.append([name, str(count)])

before_table.append(["TOTAL", str(len(before))])

t = Table(before_table, colWidths=[80 * mm, 35 * mm])

t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    )
)

story.append(Spacer(1, 3 * mm))
story.append(t)
story.append(Spacer(1, 7 * mm))


story.append(Paragraph("4. Multi-model fusion result", heading))

story.append(
    Paragraph(
        f"The multi-model fusion stage produced "
        f"<b>{len(fused)} fused object hypotheses</b>.",
        body,
    )
)

fusion_table = [["Class", "Count"]]

for name, count in sorted(fused_counts.items()):
    fusion_table.append([name, str(count)])

fusion_table.append(["TOTAL", str(len(fused))])

t = Table(fusion_table, colWidths=[80 * mm, 35 * mm])

t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    )
)

story.append(Spacer(1, 3 * mm))
story.append(t)
story.append(PageBreak())


story.append(Paragraph("5. Relative depth", heading))

story.append(
    Paragraph(
        "MiDaS produced normalized monocular relative-depth estimates. "
        "The scale uses 0 as relatively near and 1 as relatively far. "
        "<b>These values are not metric distances and must not be interpreted "
        "as metres.</b>",
        body,
    )
)

story.append(Spacer(1, 3 * mm))

story.append(
    Paragraph(
        f"Observed fused relative-depth range in this frame: "
        f"<b>{min_depth:.3f} to {max_depth:.3f}</b>.",
        body,
    )
)

story.append(Spacer(1, 6 * mm))


story.append(Paragraph("6. Collision-risk output", heading))

story.append(
    Paragraph(
        f"The maximum perception-level collision-risk score reported in "
        f"this frame was <b>{max_risk:.0f}%</b>.",
        body,
    )
)

story.append(
    Paragraph(
        "This is an internal heuristic risk score used by the adaptive perception "
        "and fusion pipeline. It is not a calibrated or certified collision "
        "probability.",
        body,
    )
)

story.append(Spacer(1, 6 * mm))


story.append(Paragraph("7. Decision confidence", heading))

story.append(
    Paragraph(
        f"Decision-confidence values ranged from approximately "
        f"<b>{min_conf:.2f} to {max_conf:.2f}</b> in this frame.",
        body,
    )
)

story.append(
    Paragraph(
        "These values are internal decision-support scores rather than calibrated "
        "statistical probabilities.",
        body,
    )
)

story.append(Spacer(1, 6 * mm))


story.append(Paragraph("8. VLM status", heading))

vlm = backends.get("vlm_scene", {})

vlm_actual = vlm.get("actual", "not recorded")

story.append(
    Paragraph(
        f"The VLM backend status was <b>{vlm_actual}</b>. "
        "The VLM component is optional, therefore its unavailability did not "
        "prevent the remaining real perception models from executing.",
        body,
    )
)

story.append(Spacer(1, 6 * mm))


story.append(Paragraph("9. Validation conclusion", heading))

story.append(
    Paragraph(
        "This experiment demonstrates successful functional execution of the "
        "primary real-model perception path, including object detection, "
        "segmentation, monocular relative depth, pose estimation, perception "
        "fusion, risk scoring, and decision-confidence generation.",
        body,
    )
)

story.append(Spacer(1, 4 * mm))


story.append(Paragraph("10. Limitations", heading))

story.append(
    Paragraph(
        "This single-image validation is not a formal accuracy benchmark, "
        "ground-truth segmentation evaluation, calibrated depth test, GPU "
        "performance benchmark, Jetson benchmark, or collision-probability "
        "validation. Broader validation requires annotated datasets, multiple "
        "scenes, environmental variation, temporal sequences, hardware profiling, "
        "and deployment experiments.",
        body,
    )
)

story.append(Spacer(1, 6 * mm))


story.append(Paragraph("11. Evidence", heading))

story.append(
    Paragraph(
        "Input image: <b>assets/street_test.png</b><br/>"
        "Machine-readable result: "
        "<b>assets/street_test_real_cuda.json</b><br/>"
        "Generated report: "
        "<b>docs/street_scene_real_inference_report.pdf</b>",
        body,
    )
)


doc.build(story)

print(f"Generated: {PDF_PATH}")
