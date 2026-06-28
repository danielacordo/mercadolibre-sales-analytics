import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = dict(
    bg = "#0A0E17",
    surface = "#111827",
    surf2 = "#1A2235",
    border = "#1E2D45",
    accent = "#00E5A0",
    gold = "#F5C842",
    red = "#FF4D6A",
    blue = "#4D9FFF",
    purple = "#A78BFA",
    text = "#E8EDF5",
    muted = "#6B7A99",
)

C = COLORS #alias

YEAR_C: dict[int, str] = {
    2023: C["blue"],
    2024: C["accent"],
    2025: C["gold"],
    2026: C["red"],
}

SEG_COLORS: dict[str, str] = {
    "Potential": C["accent"],
    "VIP": C["gold"],
    "Loyal": C["purple"],
    "Occasional": C["blue"],
    "At risk": C["red"],
    "Lost": C["muted"],
}

def apply_theme() -> None:
    """Applys dark theme to all matplotlib figures"""
    mpl.rcParams.update({
        "figure.facecolor": C["bg"],
        "axes.facecolor": C["surface"],
        "axes.edgecolor": C["border"],
        "axes.labelcolor": C["muted"],
        "axes.titlecolor": C["text"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.color": C["border"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.facecolor": C["surf2"],
        "legend.edgecolor": C["border"],
        "legend.labelcolor": C["text"],
        "legend.fontsize": 10,
        "legend.framealpha": 0.95,
        "text.color": C["text"],
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.facecolor": C["bg"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "lines.linewidth": 2,
    })

def styled_fig(w: float = 14, h: float = 6,
               title: str = None, subtitle: str = None) -> plt.Figure:
    """Creates a pre-styled figure with optional title and subtitle"""
    fig = plt.figure(figsize=(w, h), facecolor=C["bg"])
    if title:
        y = 0.97 if subtitle else 0.96
        fig.text(0.5, y, title, ha="center", va="top",
                 fontsize=16, fontweight="bold", color=C["text"])
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center", va="top",
                 fontsize=10, color=C["muted"])
    return fig


def fmt_ars(v: float) -> str:
    """Formats a value as $K or $M ARS"""
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


def spine_style(ax: plt.Axes) -> None:
    """Applys consistent spine/tick styling"""
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=C["muted"])


