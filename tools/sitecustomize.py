from __future__ import annotations

import matplotlib as mpl

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None

if scienceplots is not None:
    mpl.style.use(["science", "no-latex", "grid"])

mpl.rcParams.update(
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
