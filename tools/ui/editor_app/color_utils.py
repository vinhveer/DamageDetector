from __future__ import annotations

from PySide6.QtGui import QColor


def label_color(label: str) -> QColor:
    text = str(label or "").strip()
    if not text:
        return QColor(255, 200, 0)
    if text == "ROI":
        return QColor(0, 255, 0)

    parts = text.split()
    if parts:
        try:
            float(parts[-1])
            text = " ".join(parts[:-1]).strip() or text
        except Exception:
            pass

    palette = [
        QColor(255, 99, 132),
        QColor(54, 162, 235),
        QColor(255, 206, 86),
        QColor(75, 192, 192),
        QColor(153, 102, 255),
        QColor(255, 159, 64),
        QColor(46, 204, 113),
        QColor(231, 76, 60),
        QColor(52, 152, 219),
        QColor(241, 196, 15),
    ]
    idx = sum(ord(ch) for ch in text.lower()) % len(palette)
    return palette[idx]
