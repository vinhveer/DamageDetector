from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont


LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "crack": (229, 57, 53),
    "spall": (251, 140, 0),
    "mold": (67, 160, 71),
    "efflorescence": (3, 169, 244),
}

MODEL_PRESETS = {
    "Balanced demo": {
        "summary": "Detection + severity scoring",
        "min_confidence": 0.55,
        "risk_bias": 1.0,
    },
    "Sensitive inspection": {
        "summary": "More findings, lower threshold",
        "min_confidence": 0.45,
        "risk_bias": 1.18,
    },
    "Conservative QA": {
        "summary": "Fewer findings, higher threshold",
        "min_confidence": 0.68,
        "risk_bias": 0.86,
    },
}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    severity: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area_px(self) -> int:
        return self.width * self.height


def set_page_style() -> None:
    st.set_page_config(
        page_title="DamageDetector Streamlit Demo",
        page_icon=":mag:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
            max-width: 1480px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        div[data-testid="stMetricLabel"] {
            color: #52616f;
        }
        .dd-status {
            border-left: 4px solid #009688;
            background: #f6fbfa;
            padding: 0.8rem 1rem;
            margin: 0.25rem 0 1rem;
        }
        .dd-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            background: #eef4f8;
            color: #263238;
            font-size: 0.82rem;
            margin-right: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_demo_image(width: int = 1280, height: int = 820) -> Image.Image:
    rng = np.random.default_rng(42)
    base = rng.normal(178, 14, size=(height, width, 3)).clip(115, 224).astype(np.uint8)
    gradient = np.linspace(-14, 18, width, dtype=np.float32)
    base = np.clip(base.astype(np.float32) + gradient[None, :, None], 0, 255).astype(np.uint8)
    image = Image.fromarray(base, "RGB").filter(ImageFilter.GaussianBlur(radius=0.6))
    draw = ImageDraw.Draw(image, "RGBA")

    for x in range(70, width, 260):
        draw.line((x, 0, x + 18, height), fill=(95, 105, 112, 32), width=3)
    for y in range(120, height, 230):
        draw.line((0, y, width, y + 10), fill=(245, 247, 248, 38), width=4)

    cracks = [
        [(165, 70), (210, 148), (248, 214), (322, 295), (352, 386), (420, 520)],
        [(715, 95), (690, 185), (720, 265), (665, 360), (690, 455), (638, 570)],
        [(930, 510), (1012, 545), (1078, 616), (1160, 675)],
    ]
    for points in cracks:
        draw.line(points, fill=(35, 39, 44, 210), width=6, joint="curve")
        draw.line(points, fill=(14, 18, 22, 210), width=2, joint="curve")
        for px, py in points[1:-1]:
            branch = [(px, py), (px + int(rng.integers(-80, 80)), py + int(rng.integers(35, 90)))]
            draw.line(branch, fill=(25, 28, 32, 160), width=2)

    spalls = [(500, 185, 650, 305), (822, 260, 945, 365), (255, 590, 390, 718)]
    for box in spalls:
        draw.rounded_rectangle(box, radius=14, fill=(127, 116, 104, 120), outline=(83, 75, 70, 155), width=4)
        inner = (box[0] + 18, box[1] + 18, box[2] - 16, box[3] - 20)
        draw.ellipse(inner, fill=(99, 89, 82, 80))

    for _ in range(34):
        x = int(rng.integers(0, width - 70))
        y = int(rng.integers(0, height - 35))
        r = int(rng.integers(8, 28))
        color = (73, 132, 82, int(rng.integers(24, 62)))
        draw.ellipse((x, y, x + r * 2, y + r), fill=color)

    return image


def image_seed(image: Image.Image) -> int:
    small = image.resize((64, 64)).convert("RGB")
    payload = small.tobytes() + str(image.size).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def simulated_detections(image: Image.Image, preset_name: str) -> list[Detection]:
    width, height = image.size
    preset = MODEL_PRESETS[preset_name]
    rng = np.random.default_rng(image_seed(image) % (2**32))
    labels = ["crack", "spall", "mold", "efflorescence"]
    count = int(rng.integers(7, 14))
    detections: list[Detection] = []

    for idx in range(count):
        label = labels[idx % len(labels)] if idx < 4 else str(rng.choice(labels, p=[0.42, 0.25, 0.2, 0.13]))
        if label == "crack":
            box_w = int(rng.integers(max(60, width // 12), max(90, width // 4)))
            box_h = int(rng.integers(max(120, height // 6), max(150, height // 2)))
        elif label == "spall":
            box_w = int(rng.integers(max(70, width // 14), max(120, width // 5)))
            box_h = int(rng.integers(max(55, height // 14), max(105, height // 4)))
        else:
            box_w = int(rng.integers(max(55, width // 18), max(100, width // 6)))
            box_h = int(rng.integers(max(40, height // 20), max(90, height // 5)))

        x1 = int(rng.integers(0, max(1, width - box_w)))
        y1 = int(rng.integers(0, max(1, height - box_h)))
        x2 = min(width - 1, x1 + box_w)
        y2 = min(height - 1, y1 + box_h)
        confidence = float(np.clip(rng.normal(0.72, 0.12), 0.35, 0.97))
        area_ratio = (box_w * box_h) / max(1, width * height)
        severity = float(np.clip((confidence * 58 + area_ratio * 320) * preset["risk_bias"], 0, 100))
        detections.append(Detection(label, confidence, severity, x1, y1, x2, y2))

    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def filter_detections(
    detections: Iterable[Detection],
    labels: set[str],
    min_confidence: float,
) -> list[Detection]:
    return [
        detection
        for detection in detections
        if detection.label in labels and detection.confidence >= min_confidence
    ]


def draw_overlay(
    image: Image.Image,
    detections: list[Detection],
    *,
    opacity: float,
    line_width: int,
    show_labels: bool,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()
    alpha = int(np.clip(opacity, 0.05, 0.65) * 255)

    for detection in detections:
        color = LABEL_COLORS.get(detection.label, (38, 166, 154))
        coords = (detection.x1, detection.y1, detection.x2, detection.y2)
        draw.rectangle(coords, fill=(*color, max(25, alpha // 3)))
        draw.rectangle(coords, outline=(*color, 245), width=line_width)
        if show_labels:
            label = f"{detection.label} {detection.confidence:.0%}"
            text_box = draw.textbbox((detection.x1, detection.y1), label, font=font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            y = max(0, detection.y1 - text_h - 8)
            draw.rounded_rectangle(
                (detection.x1, y, detection.x1 + text_w + 10, y + text_h + 7),
                radius=4,
                fill=(*color, 230),
            )
            draw.text((detection.x1 + 5, y + 3), label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def detections_dataframe(detections: list[Detection]) -> pd.DataFrame:
    rows = []
    for idx, detection in enumerate(detections, start=1):
        rows.append(
            {
                "id": idx,
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "severity": round(detection.severity, 1),
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
                "area_px": detection.area_px,
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(detections: list[Detection], image: Image.Image) -> dict[str, float]:
    width, height = image.size
    total_area = width * height
    affected_area = sum(detection.area_px for detection in detections)
    max_severity = max((detection.severity for detection in detections), default=0.0)
    avg_confidence = float(np.mean([detection.confidence for detection in detections])) if detections else 0.0
    return {
        "findings": float(len(detections)),
        "affected_pct": min(100.0, affected_area / max(1, total_area) * 100),
        "max_severity": max_severity,
        "avg_confidence": avg_confidence,
    }


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def report_payload(
    image_name: str,
    preset_name: str,
    detections: list[Detection],
    image: Image.Image,
) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_name": image_name,
        "image_size": {"width": image.width, "height": image.height},
        "model_preset": preset_name,
        "metrics": summary_metrics(detections, image),
        "detections": [asdict(detection) for detection in detections],
    }


def render_sidebar() -> tuple[Image.Image, str, str, float, set[str], float, int, bool]:
    st.sidebar.header("Input")
    uploaded = st.sidebar.file_uploader(
        "Image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        image_name = uploaded.name
    else:
        image = create_demo_image()
        image_name = "demo-concrete-panel.png"

    st.sidebar.header("Analysis")
    preset_name = st.sidebar.selectbox("Pipeline", list(MODEL_PRESETS), index=0)
    preset_threshold = float(MODEL_PRESETS[preset_name]["min_confidence"])
    min_confidence = st.sidebar.slider("Min confidence", 0.0, 1.0, preset_threshold, 0.01)
    labels = set(
        st.sidebar.multiselect(
            "Damage classes",
            list(LABEL_COLORS),
            default=list(LABEL_COLORS),
        )
    )

    st.sidebar.header("Overlay")
    opacity = st.sidebar.slider("Mask opacity", 0.05, 0.65, 0.24, 0.01)
    line_width = st.sidebar.slider("Box width", 1, 8, 3)
    show_labels = st.sidebar.checkbox("Show labels", value=True)
    return image, image_name, preset_name, min_confidence, labels, opacity, line_width, show_labels


def render_metrics(metrics: dict[str, float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Findings", f"{int(metrics['findings'])}")
    c2.metric("Affected area", f"{metrics['affected_pct']:.1f}%")
    c3.metric("Max severity", f"{metrics['max_severity']:.1f}")
    c4.metric("Avg confidence", f"{metrics['avg_confidence']:.0%}")


def render_label_breakdown(detections: list[Detection]) -> None:
    if not detections:
        st.info("No findings match the current filters.")
        return

    counts = pd.Series([detection.label for detection in detections]).value_counts()
    pills = []
    for label, count in counts.items():
        color = LABEL_COLORS.get(str(label), (38, 166, 154))
        pills.append(
            f"<span class='dd-pill' style='border-left: 4px solid rgb{color};'>{label}: {count}</span>"
        )
    st.markdown("".join(pills), unsafe_allow_html=True)


def run() -> None:
    set_page_style()
    image, image_name, preset_name, min_confidence, labels, opacity, line_width, show_labels = render_sidebar()

    all_detections = simulated_detections(image, preset_name)
    detections = filter_detections(all_detections, labels, min_confidence)
    annotated = draw_overlay(
        image,
        detections,
        opacity=opacity,
        line_width=line_width,
        show_labels=show_labels,
    )
    metrics = summary_metrics(detections, image)
    dataframe = detections_dataframe(detections)

    st.title("DamageDetector Streamlit Demo")
    st.markdown(
        f"""
        <div class="dd-status">
            <strong>{MODEL_PRESETS[preset_name]["summary"]}</strong>
            &nbsp;|&nbsp; {image_name} &nbsp;|&nbsp; {image.width} x {image.height}px
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metrics(metrics)

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.image(annotated, caption="Annotated inspection image", width="stretch")
        render_label_breakdown(detections)

    with right:
        st.subheader("Findings")
        st.dataframe(
            dataframe,
            width="stretch",
            hide_index=True,
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                ),
                "severity": st.column_config.ProgressColumn(
                    "severity",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.1f",
                ),
            },
        )

        report = report_payload(image_name, preset_name, detections, image)
        csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(report, indent=2).encode("utf-8")
        png_bytes = to_png_bytes(annotated)

        d1, d2, d3 = st.columns(3)
        d1.download_button("CSV", csv_bytes, "damage_findings.csv", "text/csv", width="stretch")
        d2.download_button("JSON", json_bytes, "damage_report.json", "application/json", width="stretch")
        d3.download_button("PNG", png_bytes, "damage_overlay.png", "image/png", width="stretch")

    st.divider()
    tab1, tab2 = st.tabs(["Severity review", "Demo data"])
    with tab1:
        if detections:
            severity_df = dataframe[["id", "label", "confidence", "severity", "area_px"]].sort_values(
                "severity",
                ascending=False,
            )
            st.bar_chart(severity_df, x="id", y="severity", color="label", height=280)
        else:
            st.empty()
    with tab2:
        st.json(report, expanded=False)


if __name__ == "__main__":
    run()
