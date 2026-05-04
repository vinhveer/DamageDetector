from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_science_style() -> None:
    try:
        import scienceplots  # noqa: F401
    except Exception:
        scienceplots = None

    if scienceplots is not None:
        plt.style.use(["science", "no-latex", "grid"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_report_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 220, close: bool = True) -> Path:
    svg_path = Path(path).with_suffix(".svg")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, dpi=dpi, format="svg", bbox_inches="tight")
    fig.savefig(svg_path.with_suffix(".pdf"), dpi=dpi, format="pdf", bbox_inches="tight")
    if close:
        plt.close(fig)
    return svg_path
