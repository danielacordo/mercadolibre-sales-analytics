import os
import sys
import warnings
from pathlib import Path
from typing import Optional
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_theme, styled_fig, C, SEG_COLORS  # noqa: E402

apply_theme()

_RFM_PATH = Path(__file__).parent.parent / "data" / "rfm_clientes.csv"


# Elbow + Silhouette
def plot_elbow_silhouette(k_range, inertias: list, silhouettes: list, k_opt: int, save_path: Optional[str] = None) -> plt.Figure:
    """ Elbow + silhouette chart for K-Means cluster selection """
    fig = styled_fig(14, 5, title="K-Means Cluster Selection - Elbow + Silhouette", subtitle=f"Optimal K = {k_opt} chosen based on silhouette score")
    ax1 = fig.add_axes([0.07, 0.15, 0.40, 0.68])
    ax2 = fig.add_axes([0.57, 0.15, 0.40, 0.68])
    ks = list(k_range)
    ax1.plot(ks, inertias, marker="o", color=C["blue"], lw=2.5, markersize=8, zorder=3)
    ax1.axvline(k_opt, color=C["red"], ls="--", alpha=0.8, label=f"K={k_opt} chosen", lw=1.5)
    ax1.scatter([k_opt], [inertias[ks.index(k_opt)]], color=C["red"], s=120, zorder=5)
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow method", fontsize=12, pad=10)
    ax1.legend(fontsize=9)
    ax2.plot(ks, silhouettes, marker="o", color=C["accent"], lw=2.5, markersize=8, zorder=3)
    ax2.axvline(k_opt, color=C["red"], ls="--", alpha=0.8, label=f"K={k_opt} chosen", lw=1.5)
    ax2.scatter([k_opt], [silhouettes[ks.index(k_opt)]], color=C["red"], s=120, zorder=5)
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette score (higher = better separation)", fontsize=12, pad=10)
    ax2.legend(fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# RFM Clusters Scatter
def plot_rfm_clusters(rfm: pd.DataFrame, silhouette_score: Optional[float] = None, save_path: Optional[str] = None) -> plt.Figure:
    """ Scatter (recency vs total spend) + revenue breakdown by RFM segment

    Top section shows scatter with dot size = purchase frequency.
    Bottom section has action cards for each segment"""
    seg_stats = (rfm.groupby("Segment").agg(count=("Customer", "count"),revenue=("monto_total", "sum"), avg_ticket=("monto_total", "mean"),
                recencia=("recencia", "mean")).reset_index().sort_values("revenue", ascending=False))
    total_rev = seg_stats["revenue"].sum()
    n_customers = len(rfm)
    sil_note = f"  -  Silhouette Score = {silhouette_score:.3f}" if silhouette_score is not None else ""

    fig = styled_fig(16, 9,
        title="RFM Customer Segmentation",
        subtitle=f"{n_customers} unique customers  -  K-Means + Business rules{sil_note}")

    ax1 = fig.add_axes([0.05, 0.38, 0.52, 0.52])
    ax2 = fig.add_axes([0.63, 0.38, 0.34, 0.52])

    for seg in rfm["Segment"].unique():
        sub = rfm[rfm["Segment"] == seg]
        color = SEG_COLORS.get(seg, C["muted"])
        ax1.scatter(sub["recencia"], sub["monto_total"] / 1000, s=sub["frecuencia"] * 40 + 25, color=color, alpha=0.65, zorder=3,
                    label=f"{seg} (n={len(sub)})", edgecolors=color, linewidths=0.3)

    ax1.set_xlabel("Recency (days since last purchase)  -  lower = more recent")
    ax1.set_ylabel("Total spend ($K ARS)")
    ax1.set_title("Recency vs Total Spend\n(dot size = purchase frequency)", fontsize=11, pad=10)
    ax1.legend(fontsize=8.5, loc="upper right", markerscale=0.8)
    ax1.invert_xaxis()

    for _, row in rfm[rfm["Segment"] == "VIP"].nlargest(3, "monto_total").iterrows():
        ax1.annotate(f"VIP\n${row['monto_total']/1000:.0f}K", xy=(row["recencia"], row["monto_total"] / 1000),
                     xytext=(row["recencia"] + 80, row["monto_total"] / 1000 + 2), fontsize=7.5, color=SEG_COLORS["VIP"],
                     arrowprops=dict(arrowstyle="->", color=SEG_COLORS["VIP"] + "99", lw=0.8))

    # Revenue bars
    pcts = seg_stats["revenue"] / total_rev * 100
    bar_colors = [SEG_COLORS.get(s, C["muted"]) for s in seg_stats["Segment"]]
    y = np.arange(len(seg_stats))
    h = ax2.barh(y, pcts, color=bar_colors, alpha=0.88, height=0.55, zorder=3)
    for i, (bar, row) in enumerate(zip(h, seg_stats.itertuples())):
        w = bar.get_width()
        ax2.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"${row.revenue/1000:.0f}K  ({w:.1f}%)", va="center", fontsize=9, color=bar_colors[i], fontweight="bold")
        ax2.text(-0.3, bar.get_y() + bar.get_height() / 2, f"n={row.count}", va="center", ha="right", fontsize=8, color=C["muted"])
    ax2.set_yticks(y)
    ax2.set_yticklabels(seg_stats["Segment"], fontsize=10, fontweight="bold")
    ax2.set_xlabel("% of total revenue")
    ax2.set_title("Revenue by segment", fontsize=11, pad=10)
    ax2.set_xlim(-4, 75)

    # Action cards
    ax3 = fig.add_axes([0.05, 0.04, 0.88, 0.28])
    ax3.axis("off")
    actions = [("Potential", "134 customers", "61.0% rev",
         "Contact within 30 days\nof first purchase. No discount:\noffer new/complementary items."),
        ("VIP", "9 customers", "14.1% rev",
         "Early access to new products,\npriority treatment.\n9 people = $214K ARS."),
        ("At risk", "13 customers", "3.4% rev",
         "Last purchase ~1000 days ago.\nReactivation campaign\nwith targeted offer."),
        ("Loyal", "13 customers", "8.0% rev",
         "Loyalty program.\nSlightly higher frequency\nthan average."),
        ("Lost", "72 customers", "6.2% rev",
         "No outreach prioritized.\nLow probability\nof return."),]
    t = ax3.transAxes
    for i, (seg, count, rev, action) in enumerate(actions):
        color = SEG_COLORS.get(seg, C["muted"])
        xp = 0.01 + i * 0.20
        ax3.text(xp + 0.01, 0.92, seg, transform=t, fontsize=11, fontweight="bold", color=color, va="top")
        ax3.text(xp + 0.01, 0.72, count, transform=t, fontsize=9, color=C["muted"], va="top")
        ax3.text(xp + 0.01, 0.56, rev, transform=t, fontsize=10, fontweight="bold", color=C["text"], va="top")
        ax3.text(xp + 0.01, 0.38, action, transform=t, fontsize=8.5, color=C["muted"], va="top", linespacing=1.4)
        ax3.add_patch(mpl.patches.FancyBboxPatch( (xp, 0.02), 0.185, 0.94, boxstyle="round,pad=0.01", linewidth=1,
            edgecolor=color + "55", facecolor=color + "0A", transform=t, zorder=0))

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# RFM Heatmap
def plot_rfm_heatmap(rfm: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Heatmap of average R/F/M scores by segment.

    Color scale from dark surface to accent green. Each cell annotated with the avg score value """
    seg_order = ["VIP", "Loyal", "Potential", "At risk", "Occasional", "Lost"]
    rfm = rfm.copy()
    rfm["Segment"] = pd.Categorical(rfm["Segment"], categories=seg_order, ordered=True)
    pivot = rfm.groupby("Segment")[["R_score", "F_score", "M_score"]].mean().reindex(seg_order)
    data = pivot.values

    cmap = mcolors.LinearSegmentedColormap.from_list("ml", [C["surf2"], C["accent"] + "88", C["accent"]], N=256)

    fig = styled_fig(16, 8,
        title="RFM Score Heatmap - Recency, Frequency and Monetary Value by Segment",
        subtitle="Average score per dimension by segment  -  Normalized scale 1-4")
    ax = fig.add_axes([0.08, 0.18, 0.88, 0.66])
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=1, vmax=4)

    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(seg_order), 1), minor=True)
    ax.grid(which="minor", color=C["bg"], linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Recency\n(R score)", "Frequency\n(F score)", "Monetary\n(M score)"], fontsize=11, fontweight="bold", color=C["text"])
    ax.set_yticks(range(len(seg_order)))
    counts = rfm["Segment"].value_counts()
    ax.set_yticklabels( [f"{seg}  (n={counts.get(seg, 0)})" for seg in seg_order], fontsize=11, fontweight="bold")

    for i in range(len(seg_order)):
        for j in range(3):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=14, fontweight="bold", color=C["bg"] if val > 2.5 else C["text"])

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors=C["muted"], labelsize=9)
    cbar.set_label("Avg score (1=low, 4=high)", color=C["muted"], fontsize=9)

    for i, seg in enumerate(seg_order):
        ax.add_patch(mpl.patches.FancyBboxPatch((-0.55, i - 0.42), 0.08, 0.84,
            boxstyle="round,pad=0.01", facecolor=SEG_COLORS.get(seg, C["muted"]),
            edgecolor="none", transform=ax.transData, zorder=5, clip_on=False))

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# Customer Value Scatter 
def plot_customer_value(rfm: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Frequency vs total spend scatter + RFM score radar bars.

    Dot size encodes inverse recency (larger = more recent).
    Right panel shows avg R/F/M scores per segment as grouped bars."""
    max_rec = rfm["recencia"].max()
    rfm = rfm.copy()
    rfm["dot_size"] = (max_rec - rfm["recencia"]) / max_rec * 200 + 20
    seg_order = ["VIP", "Loyal", "Potential", "At risk", "Occasional", "Lost"]

    fig = styled_fig(16, 8, title="Customer Value Analysis - Frequency vs Total Spend",
        subtitle="306 unique customers  -  RFM segments overlaid  -  Dot size = inverse recency (larger = more recent)")
    ax1 = fig.add_axes([0.05, 0.15, 0.55, 0.72])
    ax2 = fig.add_axes([0.67, 0.15, 0.30, 0.72])

    for seg in seg_order:
        sub = rfm[rfm["Segment"] == seg]
        color = SEG_COLORS.get(seg, C["muted"])
        ax1.scatter(sub["frecuencia"], sub["monto_total"] / 1000, s=sub["dot_size"], color=color, alpha=0.70,
                    zorder=3 if seg in ["VIP", "Loyal"] else 2, label=f"{seg} (n={len(sub)})", edgecolors=color + "66", linewidths=0.5)

    med_freq = rfm["frecuencia"].median()
    med_val  = rfm["monto_total"].median() / 1000
    ax1.axvline(med_freq, color=C["muted"], lw=1, ls="--", alpha=0.5)
    ax1.axhline(med_val, color=C["muted"], lw=1, ls="--", alpha=0.5)
    for txt, xp, yp, color in [
        ("High value\nHigh freq", rfm["frecuencia"].max() * 0.85, rfm["monto_total"].max() / 1000 * 0.9, C["gold"]),
        ("High value\nLow freq", 1.05,  rfm["monto_total"].max() / 1000 * 0.9, C["purple"]),
        ("Low value\nLow freq", 1.05,  med_val * 0.3, C["muted"]),
    ]:
        ax1.text(xp, yp, txt, fontsize=8, color=color, ha="center" if color == C["gold"] else "left",
                 bbox=dict(boxstyle="round", facecolor=color + "11", edgecolor=color + "33"))

    ax1.set_xlabel("Purchase frequency")
    ax1.set_ylabel("Total spend ($K ARS)")
    ax1.set_title("Frequency vs Total Spend by RFM Segment", fontsize=11, pad=10)
    ax1.legend(fontsize=9, loc="center right", markerscale=0.7)

    # Score bars
    rfm["Segment"] = pd.Categorical(rfm["Segment"], categories=seg_order, ordered=True)
    stats_df = rfm.groupby("Segment")[["R_score", "F_score", "M_score"]].mean().reindex(seg_order)
    bar_h = 0.22
    y = np.arange(len(seg_order))
    for j, (col, color_b, lbl) in enumerate([
        ("R_score", C["accent"], "R"),
        ("F_score", C["blue"], "F"),
        ("M_score", C["gold"], "M"),
    ]):
        ax2.barh(y + j * bar_h - bar_h, stats_df[col], height=bar_h,
                 color=color_b, alpha=0.82, label=lbl, zorder=3)

    ax2.axvline(2.5, color=C["muted"], lw=1, ls="--", alpha=0.6, label="Average")
    ax2.set_yticks(y)
    ax2.set_yticklabels(seg_order, fontsize=10, fontweight="bold")
    ax2.set_xlabel("Avg score (1-4)")
    ax2.set_title("Avg RFM scores\nby segment", fontsize=11, pad=10)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_xlim(0, 4.5)

    for i, seg in enumerate(seg_order):
        ax2.add_patch(mpl.patches.FancyBboxPatch( (-0.55, i - 0.38), 0.1, 0.76, boxstyle="round,pad=0.01", facecolor=SEG_COLORS.get(seg, C["muted"]),
            edgecolor="none", transform=ax2.transData, zorder=5, clip_on=False))

    fig.text(0.5, 0.05,
             "Dot size represents inverse recency: larger dots = more recent buyers. "
             "VIP = high value + most recent.", ha="center", fontsize=9, color=C["muted"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor=C["surf2"], edgecolor=C["border"]))
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# Main Pipeline
def run_rfm_plots(rfm_path: str, plots_dir: str, verbose: bool = True) -> None:
    """ Generate all RFM visualizations and save them to plots_dir """
    rfm = pd.read_csv(rfm_path)
    os.makedirs(plots_dir, exist_ok=True)

    if verbose:
        print()
        print("RFM PLOTS - MercadoLibre Sales Analytics")
        print()

    charts = [("RFM clusters scatter", plot_rfm_clusters, "10_rfm_clusters.png", (rfm,)),
        ("RFM score heatmap", plot_rfm_heatmap, "11_rfm_heatmap.png", (rfm,)),
        ("Customer value", plot_customer_value, "09_customer_value.png", (rfm,)),]
    
    for name, fn, fname, args in charts:
        try:
            save = os.path.join(plots_dir, fname)
            fn(*args, save_path=save)
            plt.close("all")
            if verbose:
                print(f" + {name:25} -> {fname}")
        except Exception as e:
            if verbose:
                print(f" x {name:25} skipped: {e}")

    if verbose:
        print(f"\n Saved {len(charts)} charts to: {plots_dir}")
        print(" RFM plots complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate RFM visualizations.")
    parser.add_argument("--rfm",   type=str, default=str(Path(__file__).parent.parent / "data" / "rfm_clientes.csv"))
    parser.add_argument("--plots", type=str, default=str(Path(__file__).parent.parent / "plots"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_rfm_plots(rfm_path=args.rfm, plots_dir=args.plots, verbose=not args.quiet)
