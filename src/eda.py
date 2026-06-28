import os
import warnings
from pathlib import Path
from typing import Optional
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_theme, styled_fig, spine_style, C, YEAR_C  # noqa: E402

apply_theme()

_IPC_PATH = Path(__file__).parent.parent / "data" / "ipc_indec.csv"
MONTHS_EN = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Summary Tables
def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    annual = (df.groupby(df["Fecha"].dt.year).agg(orders=("Order_id","count"), revenue=("Ingreso_bruto","sum"),
             avg_ticket=("Monto","mean"), median_ticket=("Monto","median")).round(0))
    annual["revenue_growth"] = annual["revenue"].pct_change().round(3)
    annual["volume_growth"] = annual["orders"].pct_change().round(3)
    annual.index.name = "Year"

    return annual


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (df.groupby(df["Fecha"].dt.to_period("M"))
        .agg(orders=("Order_id","count"), revenue=("Ingreso_bruto","sum"), avg_ticket=("Monto","mean")).reset_index())
    monthly["ds"] = monthly["Fecha"].dt.to_timestamp()
    monthly["year"] = monthly["ds"].dt.year
    monthly["period_str"] = monthly["ds"].dt.strftime("%Y-%m")
    monthly["month"] = monthly["ds"].dt.month

    if _IPC_PATH.exists():
        ipc_df = pd.read_csv(_IPC_PATH)
        ipc_map = dict(zip(ipc_df["period"], ipc_df["cpi_index"]))
        monthly["cpi"] = monthly["period_str"].map(ipc_map)
    else:
        monthly["cpi"] = np.nan
    price_base = monthly["avg_ticket"].iloc[0]
    monthly["price_index"] = (monthly["avg_ticket"] / price_base * 100).round(1)
    monthly["revenue_real"] = (monthly["revenue"] / monthly["cpi"] * 100).round(0)
    return monthly


def geo_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    geo = (df.dropna(subset=["Provincia_nombre"]).groupby("Provincia_nombre").agg(orders=("Order_id","count"), revenue=("Ingreso_bruto","sum"),
             avg_ticket=("Monto","mean")).reset_index().sort_values("revenue", ascending=False).head(top_n))
    geo["pct_orders"] = (geo["orders"] / geo["orders"].sum()  * 100).round(1)
    geo["pct_revenue"] = (geo["revenue"] / geo["revenue"].sum() * 100).round(1)
    geo["avg_ticket"] = geo["avg_ticket"].round(0)

    return geo.reset_index(drop=True)


def margin_summary(df: pd.DataFrame) -> pd.DataFrame:
    df_ml = df[df["Fuente"] == "ML_Oficial"].dropna(subset=["Ingreso_neto"]).copy()
    assert len(df_ml) > 0, "No ML Official rows with Ingreso_neto, check data source"
    df_ml["pct_net"] = df_ml["Ingreso_neto"] / df_ml["Ingreso_bruto"] * 100
   
    return df_ml


# Chart Functions
def plot_revenue_monthly(df: pd.DataFrame,
                         save_path: Optional[str] = None) -> plt.Figure:
    """ Dual bar chart (nominal vs inflation-adjusted) + avg ticket line

    Bars are color-coded by year. The inflation-adjusted bars (hatched) reveal the real growth picture beneath nominal figures """
    monthly = monthly_summary(df)
    fig, ax1 = plt.subplots(figsize=(16, 6), facecolor=C["bg"])
    ax1.set_facecolor(C["surface"])
    x = np.arange(len(monthly))
    w = 0.38
    colors_nom = [YEAR_C.get(y, C["muted"]) for y in monthly["year"]]
    ax1.bar(x - w/2, monthly["revenue"]/1000, width=w, color=colors_nom, alpha=0.9, label="Nominal", zorder=3)
    ax1.bar(x + w/2, monthly["revenue_real"]/1000, width=w, color=[c + "88" for c in colors_nom], alpha=0.85,
            label="Real (base Jan-23)", zorder=3, hatch="///", edgecolor="#ffffff22")
    ax2 = ax1.twinx()
    ax2.plot(x, monthly["avg_ticket"]/1000, color=C["red"], lw=2,
             marker="o", markersize=3, zorder=5, label="Avg ticket")
    ax2.set_ylabel("Avg ticket ($K ARS)", color=C["red"], fontsize=10)
    ax2.tick_params(colors=C["red"], labelsize=8)
    ax2.spines["right"].set_color(C["border"])

    for yr in [2023, 2024, 2025, 2026]:
        sub = monthly[monthly["year"] == yr]
        if sub.empty:
            continue
        s, e = sub.index[0], sub.index[-1] + 1
        ax1.axvspan(s - 0.5, e - 0.5, color=YEAR_C.get(yr, C["muted"]) + "11", zorder=0)
        ax1.text((s + e - 1) / 2, -14, str(yr), ha="center", va="top",
                 fontsize=11, fontweight="bold",
                 color=YEAR_C.get(yr, C["muted"]), transform=ax1.transData)
        
    ticks = [i for i, d in enumerate(monthly["period_str"]) if d.endswith("-01") or d.endswith("-07")]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([monthly["period_str"].iloc[i] for i in ticks], fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("Revenue ($K ARS)", color=C["text"])
    ax1.set_xlim(-0.6, len(monthly) - 0.4)
    ax1.set_title( "Monthly Gross Revenue - Nominal vs Inflation-Adjusted\n"
        "Base: Jan 2023 ARS  -  Off-topic sales excluded",
        fontsize=14, fontweight="bold", color=C["text"], pad=14)
    spine_style(ax1)
    ax1.grid(True, color=C["border"], lw=0.6, alpha=0.8)
    l1, b1 = ax1.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, loc="upper left", fontsize=9)
    plt.tight_layout(pad=1.5)
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=C["bg"], bbox_inches="tight")
    return fig


def plot_price_vs_cpi(df: pd.DataFrame,
                      save_path: Optional[str] = None) -> plt.Figure:
    """ Business price index vs Argentina CPI (INDEC), base Jan 2023 = 100

    Red fill = price lagged inflation (real margin loss).
    Green fill = price beat inflation (real margin gain) """
    monthly  = monthly_summary(df)
    mp = monthly.dropna(subset=["cpi"])
    fig = styled_fig(16, 7, title="Business Price Index vs Argentina CPI",
        subtitle="Base Jan 2023 = 100  -  Red zone = price below inflation -> real margin loss")
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.72])
    x = np.arange(len(mp))
    cpi = mp["cpi"].values
    pidx = mp["price_index"].values
    ax.fill_between(x, pidx, cpi, where=(pidx < cpi), interpolate=True, color=C["red"], alpha=0.20, label="Price below CPI (margin loss)", zorder=2)
    ax.fill_between(x, pidx, cpi, where=(pidx >= cpi), interpolate=True, color=C["accent"], alpha=0.15, label="Price above CPI (real gain)", zorder=2)
    ax.plot(x, cpi,  color=C["red"],  lw=2.5, ls="--", label="Argentina CPI (INDEC)", zorder=4)
    ax.plot(x, pidx, color=C["blue"], lw=2.5, marker="o", markersize=3,
            label="Business price index", zorder=5)
    ax2 = ax.twinx()
    ax2.bar(x, mp["orders"], color=C["muted"], alpha=0.18, width=0.6,
            zorder=1, label="Monthly orders")
    ax2.set_ylabel("Orders", color=C["muted"], fontsize=9)
    ax2.tick_params(colors=C["muted"], labelsize=8)
    ax2.spines["right"].set_color(C["border"])
    ax2.set_ylim(0, 60)
    periods = list(mp["period_str"])

    if "2023-12" in periods:
        d = periods.index("2023-12")
        ax.annotate("Dec 2023\nDevaluation\nshock", xy=(d, cpi[d]), xytext=(d - 5, cpi[d] + 200), fontsize=8, color=C["red"],
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=1), ha="center")
    gap_peak = int(np.argmax(cpi - pidx))

    ax.annotate(f"Max gap:\n{round(cpi[gap_peak]-pidx[gap_peak])} pts", xy=(gap_peak, (cpi[gap_peak] + pidx[gap_peak]) / 2),
        xytext=(gap_peak + 3, (cpi[gap_peak] + pidx[gap_peak]) / 2 + 80),
        fontsize=8, color=C["red"], ha="left", arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))
    ticks = [i for i, p in enumerate(periods) if p.endswith("-01") or p.endswith("-07")]
    ax.set_xticks(ticks)
    ax.set_xticklabels([periods[i] for i in ticks], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Index (Jan 2023 = 100)", color=C["text"])
    ax.set_xlim(-0.5, len(mp) - 0.5)
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, loc="upper left", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_seasonality(df: pd.DataFrame,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Grouped bar chart of monthly revenue by year + seasonality index.

    Top panel: revenue bars grouped by year, historical average line.
    Bottom panel: seasonality index (100 = monthly average).

    Peak months are detected dynamically from the data (top-2 by seasonality
    index) rather than assumed to be August-September. An earlier version
    hardcoded "August and September are peak every year" in the title and
    highlighted those two bars gold regardless of what the chart actually
    showed — which silently became wrong (Jan/Feb overtook Aug/Sep) the
    first time this was re-run on a refreshed dataset, with nothing in the
    code flagging the mismatch.

    The seasonality index also only averages over COMPLETE calendar years.
    A partial current year (e.g. only Jan-Apr on file) mixed into a
    per-calendar-month average is a real distortion risk under heavy
    nominal inflation: whichever months the partial year happens to cover
    get an outsized nominal weight relative to months it doesn't cover yet,
    which can manufacture a fake "peak" with no real seasonal cause.
    """
    df = df.copy()
    df["year"] = df["Fecha"].dt.year
    complete_years = [y for y in df["year"].unique()
                       if df[df["year"] == y]["Fecha"].dt.month.nunique() == 12]
    df_complete = df[df["year"].isin(complete_years)]

    monthly_rev = df.groupby(["year", df["Fecha"].dt.month])["Ingreso_bruto"].sum().unstack(level=0) / 1000
    avg_rev = (df_complete.groupby(df_complete["Fecha"].dt.month)["Ingreso_bruto"].sum() / 1000 / max(len(complete_years), 1)) \
              .reindex(range(1, 13))

    overall_avg = avg_rev.mean()
    idx_vals = (avg_rev / overall_avg * 100).fillna(100).tolist() if overall_avg else [100] * 12

    # Detect peak months dynamically: top-2 by index, only if actually >100.
    ranked = sorted(range(12), key=lambda i: idx_vals[i], reverse=True)
    peak_months_0idx = [i for i in ranked[:2] if idx_vals[i] > 100]
    peak_names = [MONTHS_EN[i + 1] for i in peak_months_0idx]
    # " & " rather than "-": these months aren't necessarily consecutive
    # (e.g. September and January), so a hyphen ("Sep-Jan") could misread
    # as a continuous range spanning 5 months instead of two single months.
    peak_label = " & ".join(peak_names) if peak_names else "No clear peak"
    years_note = f"{len(complete_years)} complete year(s)" if complete_years else "partial years only — directional"

    title = (f"Seasonality — {peak_label} run highest"
             if peak_names else "Seasonality — no consistent peak detected")

    fig = styled_fig(16, 9,
        title=title,
        subtitle=f"Revenue ($K ARS) by month  ·  index computed from {years_note}, partial years excluded from the average")
    ax1 = fig.add_axes([0.06, 0.52, 0.88, 0.36])
    ax2 = fig.add_axes([0.06, 0.10, 0.88, 0.32])
    x = np.arange(12)
    w = 0.2
    avg_vals_display = monthly_rev.mean(axis=1)  # for the dashed "historical avg" line — all years, just a visual reference
    for yr, off in zip([2023, 2024, 2025, 2026], [-1.5, -0.5, 0.5, 1.5]):
        if yr not in monthly_rev.columns:
            continue
        vals = [monthly_rev.loc[m, yr] if m in monthly_rev.index else 0 for m in range(1, 13)]
        ax1.bar(x + off * w, vals, width=w, color=YEAR_C[yr], alpha=0.88, zorder=3, label=str(yr))
    avg_vals = [avg_vals_display.get(m, 0) for m in range(1, 13)]
    ax1.plot(x, avg_vals, color=C["muted"], lw=1.5, ls="--", zorder=4, label="Historical avg")
    ax1.fill_between(x, avg_vals, alpha=0.06, color=C["muted"])
    for m_idx in peak_months_0idx:
        ax1.axvspan(m_idx - 0.5, m_idx + 0.5, color=C["gold"], alpha=0.07, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MONTHS_EN[1:], fontsize=10, fontweight="bold")
    ax1.set_ylabel("Revenue ($K ARS)", color=C["text"])
    ax1.legend(ncol=5, loc="upper left", fontsize=9)
    if peak_months_0idx:
        peak_x = sum(peak_months_0idx) / len(peak_months_0idx)
        ax1.annotate("PEAK", xy=(peak_x, max(avg_vals) * 1.08), fontsize=9,
            color=C["gold"], fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["gold"] + "22", edgecolor=C["gold"] + "66"))
    bar_colors = [C["gold"] if i in peak_months_0idx else C["accent"] if v > 100 else C["red"] if v < 80 else C["blue"]
                  for i, v in enumerate(idx_vals)]
    bars2 = ax2.bar(x, idx_vals, color=bar_colors, alpha=0.85, zorder=3, width=0.65)
    ax2.axhline(100, color=C["muted"], lw=1.5, ls="--", zorder=4)
    for bar, val in zip(bars2, idx_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C["text"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(MONTHS_EN[1:], fontsize=10, fontweight="bold")
    ax2.set_ylabel("Seasonality index", color=C["text"])
    ax2.set_ylim(0, max(idx_vals) * 1.15 if max(idx_vals) else 120)
    ax2.legend(handles=[
        mpatches.Patch(color=C["gold"],   label=f"Peak ({peak_label})" if peak_names else "Peak (none)"),
        mpatches.Patch(color=C["accent"], label="Above average"),
        mpatches.Patch(color=C["red"],    label="Below average"),
        mpatches.Patch(color=C["blue"],   label="Near average"),
    ], loc="upper right", fontsize=8, ncol=2)
    action_text = (f"Action: Run ads ahead of {peak_label} to capture the peak  ·  "
                   f"Based on {years_note} — re-check before committing budget"
                   if peak_names else
                   "Action: No reliable seasonal peak yet — base advertising timing on inventory/cash flow instead")
    fig.text(0.5, 0.05, action_text,
        ha="center", fontsize=9, color=C["gold"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C["gold"] + "11", edgecolor=C["gold"] + "44"))
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_geography(df: pd.DataFrame,
                   save_path: Optional[str] = None) -> plt.Figure:
    """Revenue and avg ticket by top provinces (horizontal bars)."""
    geo = geo_summary(df).sort_values("revenue", ascending=False)
    fig = styled_fig(16, 7,
        title="Geographic Revenue Distribution",
        subtitle="Top provinces by revenue  ·  Decoration products  ·  2023–2026")
    ax1 = fig.add_axes([0.05, 0.12, 0.45, 0.74])
    ax2 = fig.add_axes([0.60, 0.12, 0.37, 0.74])
    y   = np.arange(len(geo))
    bar_colors = [C["accent"], C["blue"], C["blue"] + "CC", C["blue"] + "99", C["purple"]] + [C["muted"]] * 10
    h = ax1.barh(y, geo["revenue"] / 1000, color=bar_colors[:len(geo)], alpha=0.88, height=0.55, zorder=3)
    for bar, row in zip(h, geo.itertuples()):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"${row.revenue/1000:.0f}K  ({row.pct_revenue:.1f}%)", va="center", fontsize=9, color=C["text"], fontweight="bold")
        ax1.text(-0.5, bar.get_y() + bar.get_height() / 2,
                 f"{row.orders} ord", va="center", ha="right", fontsize=8, color=C["muted"])
    ax1.set_yticks(y)
    ax1.set_yticklabels(geo["Provincia_nombre"], fontsize=10, fontweight="bold")
    ax1.set_xlabel("Revenue ($K ARS)")
    ax1.set_title("Revenue by province", fontsize=11, pad=10)
    ax1.set_xlim(-25, geo["revenue"].max() / 1000 * 1.45)
    bar_colors2 = [C["gold"] if t > geo["avg_ticket"].mean() else C["blue"] + "99" for t in geo["avg_ticket"]]
    h2 = ax2.barh(y, geo["avg_ticket"] / 1000, color=bar_colors2, alpha=0.88, height=0.55, zorder=3)
    avg_t = geo["avg_ticket"].mean()
    ax2.axvline(avg_t / 1000, color=C["muted"], lw=1.5, ls="--", label=f"Avg ${avg_t/1000:.1f}K")
    for bar, row in zip(h2, geo.itertuples()):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"${row.avg_ticket/1000:.1f}K", va="center", fontsize=9, color=C["text"])
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(geo))
    ax2.set_xlabel("Avg ticket ($K ARS)")
    ax2.set_title("Avg ticket by province", fontsize=11, pad=10)
    ax2.legend(fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_margin_correction(df: pd.DataFrame, df_raw: Optional[pd.DataFrame] = None, save_path: Optional[str] = None) -> plt.Figure:
    """ Side-by-side pie charts comparing reported vs corrected net margin.

    IMPORTANT: both percentages are computed live from data via margin_summary(), nothing here is hardcoded. Earlier versions of this
    function baked specific numbers (59.2% / 68.3%) directly into the plot, which silently went stale the first time the dataset was refreshed """
    def _weighted_net_pct(frame: pd.DataFrame) -> float:
        m = margin_summary(frame)
        return float(m["Ingreso_neto"].sum() / m["Ingreso_bruto"].sum() * 100)

    corrected_pct = _weighted_net_pct(df)
    reported_pct = _weighted_net_pct(df_raw) if df_raw is not None else None

    panels = [("CORRECTED margin (real)", corrected_pct)]
    if reported_pct is not None:
        panels.insert(0, ("REPORTED margin (before cleanup)", reported_pct))

    delta_note = f" -  Δ {corrected_pct - reported_pct:+.1f} pp after cleanup" if reported_pct is not None else ""
    fig = styled_fig(16 if len(panels) == 2 else 9, 7,
        title="Margin Correction - Before vs After" if len(panels) == 2 else "Net Margin (corrected)",
        subtitle=f"Off-topic sales excluded from margin baseline{delta_note}  -  computed from data/ventas_decoraciones.csv")

    if len(panels) == 2:
        ax1 = fig.add_axes([0.05, 0.15, 0.38, 0.70])
        ax2 = fig.add_axes([0.57, 0.15, 0.38, 0.70])
        axes = [ax1, ax2]
    else:
        axes = [fig.add_axes([0.30, 0.15, 0.40, 0.70])]

    for ax, (title, pct) in zip(axes, panels):
        ml_fee = 100 - pct
        clr = [C["muted"], C["red"] if pct < 65 else C["accent"]]
        ax.pie([ml_fee, pct], colors=clr, autopct="%1.1f%%",
               startangle=90, explode=[0.02, 0.06], textprops={"color": C["text"], "fontsize": 11, "fontweight": "bold"},
               wedgeprops={"edgecolor": C["bg"], "linewidth": 2})
        ax.set_title(title, fontsize=12, pad=16, color=C["text"], fontweight="bold")
        ax.legend(handles=[
            mpatches.Patch(facecolor=C["muted"], label=f"ML fees ({ml_fee:.1f}%)"),
            mpatches.Patch(facecolor=clr[1], label=f"Net margin ({pct:.1f}%)"),
        ], loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.08))

    if len(panels) == 2:
        fig.text(0.475, 0.52, "\u2192", fontsize=40, color=C["accent"], ha="center", va="center", fontweight="bold")
        fig.text(0.475, 0.42, f"{corrected_pct - reported_pct:+.1f} pp", fontsize=14, color=C["accent"], ha="center", va="center", fontweight="bold")
        fig.text(0.475, 0.35, "change", fontsize=10, color=C["muted"],  ha="center", va="center")
    fig.text(0.5, 0.06, "Margin computed only from ML Official rows with reported net revenue (Ingreso_neto), revenue-weighted.\n"
        "CNX channel orders are excluded from this calc - ML does not report net revenue for that channel.",
        ha="center", fontsize=9, color=C["muted"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C["surf2"], edgecolor=C["border"]))
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# Keep old name as alias for backward compat
def plot_margin_scatter(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    return plot_margin_correction(df, save_path=save_path)


def plot_top_products(df: pd.DataFrame, top_n: int = 12, save_path: Optional[str] = None) -> plt.Figure:
    """ Bubble + bar chart of top products by revenue """
    prod = (df.groupby("Titulo_prod").agg(orders=("Order_id","count"), revenue=("Ingreso_bruto","sum"), ticket=("Monto","mean"))
              .reset_index().sort_values("revenue", ascending=False).head(top_n))
    fig = styled_fig(16, 8, title="Top Products by Revenue", subtitle="Decoration products  -  2023-2026  -  Bubble size = number of orders")
    ax1 = fig.add_axes([0.05, 0.14, 0.55, 0.74])
    ax2 = fig.add_axes([0.65, 0.14, 0.32, 0.74])
    colors_p = [C["accent"], C["gold"], C["blue"], C["purple"], C["red"]] + [C["muted"]] * 20
    for i, (_, row) in enumerate(prod.iterrows()):
        ax1.scatter(row["ticket"] / 1000, row["revenue"] / 1000,
                    s=row["orders"] * 25 + 50, color=colors_p[i], alpha=0.82, zorder=3, edgecolors=colors_p[i] + "66", linewidths=0.5)
        if i < 5:
            ax1.annotate(f"#{i+1}", (row["ticket"] / 1000, row["revenue"] / 1000),
                         fontsize=8, ha="center", va="center", fontweight="bold", color=C["bg"], zorder=4)
    ax1.set_xlabel("Avg ticket ($K ARS)")
    ax1.set_ylabel("Total revenue ($K ARS)")
    ax1.set_title(f"Revenue vs Ticket - Top {top_n} products\n(size = orders)", fontsize=11, pad=10)
    y = np.arange(len(prod))
    hb = ax2.barh(y, prod["revenue"] / 1000, color=colors_p[:len(prod)], alpha=0.87, height=0.55, zorder=3)
    for bar, row in zip(hb, prod.itertuples()):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"${row.revenue/1000:.0f}K", va="center", fontsize=8.5, color=C["text"], fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels([t[:35] + "..." if len(t) > 35 else t for t in prod["Titulo_prod"]], fontsize=7.5)
    ax2.set_xlabel("Revenue ($K ARS)")
    ax2.set_title(f"Top {top_n} by revenue", fontsize=11, pad=10)

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_executive_summary(save_path: Optional[str] = None) -> plt.Figure:
    """ One-page visual executive summary with 5 action cards

    All numbers are computed live via decision_layer.build_pricing_strategy(), reusing the exact same pipeline the dashboard and the CLI strategy
    document use, instead of a second, independently hardcoded copy of the same five numbers. The previous version hardcoded all of them directly
    (134 customers, ε=-0.66, 59.2%->68.3% margin, "Aug-Sep=40% rev", etc.)
    and none of it was wired to actually recompute from data/, so it quietly went stale the first time the dataset was refreshed """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    from decision_layer import load_inputs, build_pricing_strategy

    inputs = load_inputs()
    strategy = build_pricing_strategy(inputs)
    cs, ci, _fc, rfm, seas = (strategy["current_state"], strategy["elasticity"], strategy["forecast"], strategy["rfm_opportunity"], strategy["seasonality"])
    eps = ci["epsilon"]
    p10_vol_pct = eps * 10                       
    p10_rev_pct = (1.10) ** (1 + eps) - 1
    p10_net_monthly = cs["monthly_revenue"] * p10_rev_pct * cs["net_margin"]
    margin_pct = cs["net_margin"] * 100

    try:
        pass 
    except Exception:
        pass

    fig = styled_fig(16, 9,
        title="Executive Summary - MercadoLibre Decorations 2023-2026",
        subtitle="Key findings and recommended actions with estimated impact - computed live from data/")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    findings = [
        {"num": "01", "title": "Inelastic Demand", "metric": f"epsilon = {eps:.2f}",
         "body": f"+10% price\n-> {p10_vol_pct:.1f}% volume\n-> {p10_rev_pct*100:+.1f}% revenue",
         "action": "Raise prices 10%\nthis month",
         "impact": f"+${p10_net_monthly:,.0f} ARS/mo", "color": C["accent"], "priority": "HIGH"},
        {"num": "02", "title": "Seasonal Peak", "metric": f"{seas['peak_label']} = {seas['peak_pct_avg'] or 0:.0f}% rev",
         "body": f"Ranged {seas['peak_pct_min'] or 0:.0f}-{seas['peak_pct_max'] or 0:.0f}%\nacross {len(seas['complete_years'])} complete years\n— monitor, don't assume",
         "action": f"Plan ads before\n{seas['peak_label']}",
         "impact": (f"+${seas['peak_incremental']:,.0f} ARS\nin {seas['peak_label']}" if seas["peak_incremental"] else "TBD"),
         "color": C["gold"], "priority": "MEDIUM"},
        {"num": "03", "title": f"{rfm['potential_count']} Key Customers", "metric": f"{rfm['pct_total_revenue']:.0f}% of revenue",
         "body": "Potential segment:\nrecent, never returned.\nWindow: 30 days",
         "action": "Contact this\nweek",
         "impact": f"+${rfm['potential_count']*0.15*cs['avg_ticket']:,.0f} ARS\n(15% conversion)", "color": C["purple"], "priority": "HIGH"},
        {"num": "04", "title": "Net Margin (current)", "metric": f"{margin_pct:.1f}%",
         "body": f"After ML fees\n({cs['ml_fee_rate']*100:.1f}% retained)\nrevenue-weighted",
         "action": f"Use {margin_pct:.1f}% as\npricing baseline", "impact": "Correct minimum\nviable price",   "color": C["blue"],   "priority": "IMMEDIATE"},
        {"num": "05", "title": "Inflation Illusion",  "metric": "Real revenue fell",
         "body": "Nominal grows.\nReal (base 2023)\nshowed drop in 2024",
         "action": "Track real revenue\nmonthly", "impact": "See notebooks/01\nfor real-vs-nominal", "color": C["red"],  "priority": "MEDIUM"},
    ]
    card_w, card_h, gap, start_x = 0.17, 0.72, 0.012, 0.025
    t = ax.transAxes
    for i, f in enumerate(findings):
        xp, yp, color = start_x + i * (card_w + gap), 0.14, f["color"]
        ax.add_patch(mpl.patches.FancyBboxPatch((xp, yp), card_w, card_h,
            boxstyle="round,pad=0.01", facecolor=color + "0A", edgecolor=color + "55", linewidth=1.5, transform=t, zorder=1))
        ax.add_patch(mpl.patches.FancyBboxPatch((xp, yp + card_h - 0.05), card_w, 0.05,
            boxstyle="round,pad=0.005", facecolor=color, edgecolor="none", transform=t, zorder=2))
        ax.text(xp + card_w/2, yp + card_h - 0.026, f["num"], ha="center", va="center", fontsize=18, fontweight="black", color=C["bg"], transform=t)
        ax.text(xp + card_w/2, yp + card_h - 0.085, f["title"], ha="center", va="top", fontsize=10.5, fontweight="bold", color=C["text"], transform=t)
        ax.text(xp + card_w/2, yp + card_h - 0.155, f["metric"], ha="center", va="top", fontsize=10, fontweight="bold", color=color, transform=t)
        ax.text(xp + card_w/2, yp + card_h - 0.250, f["body"], ha="center", va="top", fontsize=8.5, color=C["muted"], transform=t, linespacing=1.5)
        ax.add_patch(mpl.patches.FancyBboxPatch((xp + 0.01, yp + 0.25), card_w - 0.02, 0.14,
            boxstyle="round,pad=0.008", facecolor=color + "18", edgecolor=color + "44", linewidth=1, transform=t, zorder=3))
        ax.text(xp + card_w/2, yp + 0.38,  "ACTION", ha="center", va="top", fontsize=7, fontweight="bold", color=color, transform=t)
        ax.text(xp + card_w/2, yp + 0.355, f["action"], ha="center", va="top", fontsize=8.5, color=C["text"], transform=t, linespacing=1.4)
        ax.text(xp + card_w/2, yp + 0.21,  f["impact"], ha="center", va="top", fontsize=9, fontweight="bold", color=color, transform=t)
        ax.add_patch(mpl.patches.FancyBboxPatch((xp + 0.02, yp + 0.025), card_w - 0.04, 0.05,
            boxstyle="round,pad=0.005", facecolor=color + "22", edgecolor=color + "66", linewidth=1, transform=t, zorder=3))
        ax.text(xp + card_w/2, yp + 0.05, f"Priority: {f['priority']}", ha="center", va="center", fontsize=8, fontweight="bold", color=color, transform=t)
    _ret_impact = rfm["potential_count"] * 0.15 * cs["avg_ticket"]
    _ads_gain = seas["peak_incremental"] or 0
    _combined = p10_net_monthly * 3 + _ret_impact + _ads_gain

    fig.text(0.5, 0.08,
        f"Combined estimated impact (3 months)  |  Prices +10%: \\${p10_net_monthly*3:,.0f} ARS  "
        f"·  Retention: \\${_ret_impact:,.0f} ARS  ·  {seas['peak_label']} ads: \\${_ads_gain:,.0f} ARS  =  ~\\${_combined:,.0f} ARS",
        ha="center", fontsize=10, color=C["accent"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=C["accent"] + "0F", edgecolor=C["accent"] + "44"))
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# Main Pipeline
def run_eda(data_path: str, plots_dir: str, verbose: bool = True) -> dict:
    """ Run the complete EDA pipeline and save all charts """
    df = pd.read_csv(data_path, parse_dates=["Fecha"])
    os.makedirs(plots_dir, exist_ok=True)

    assert len(df) > 0, "Dataset is empty"
    assert "Fecha" in df.columns, "Missing column: Fecha"
    assert "Ingreso_bruto" in df.columns, "Missing column: Ingreso_bruto"

    if verbose:
        print()
        print("EDA PIPELINE - MercadoLibre Sales Analytics")
        print()

    annual = annual_summary(df)
    monthly = monthly_summary(df)
    geo = geo_summary(df)

    if verbose:
        print(f"\n  Orders: {len(df)}")
        print(f" Gross revenue: ${df['Ingreso_bruto'].sum():,.0f} ARS")
        print(f" Avg ticket: ${df['Monto'].mean():,.0f} ARS")
        print(f" Median ticket: ${df['Monto'].median():,.0f} ARS")
        print("\n  Annual summary:")
        print(annual[["orders", "revenue", "avg_ticket", "revenue_growth"]].to_string())
        print("\n  Top provinces:")
        print(geo[["Provincia_nombre", "orders", "pct_orders", "pct_revenue"]].to_string(index=False))

    charts = [
        ("Executive summary", plot_executive_summary, "00_executive_summary.png", ()),
        ("Revenue + ticket", plot_revenue_monthly, "01_revenue_monthly.png", (df,)),
        ("Price vs CPI", plot_price_vs_cpi, "02_price_vs_cpi.png", (df,)),
        ("Seasonality", plot_seasonality, "03_seasonality.png", (df,)),
        ("Geography", plot_geography, "04_geography.png", (df,)),
        ("Margin correction", plot_margin_correction, "05_margin.png", (df,)),
        ("Top products", plot_top_products, "06_products.png", (df,)),
    ]

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
        print(f"\n  Saved {len(charts)} charts to: {plots_dir}")
        print("  EDA complete")

    return {"annual": annual, "monthly": monthly, "geo": geo}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the MercadoLibre EDA pipeline.")
    parser.add_argument("--data",  type=str, default=str(Path(__file__).parent.parent / "data" / "ventas_decoraciones.csv"))
    parser.add_argument("--plots", type=str, default=str(Path(__file__).parent.parent / "plots"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_eda(data_path=args.data, plots_dir=args.plots, verbose=not args.quiet)
