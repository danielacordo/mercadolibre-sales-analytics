import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
from typing import Optional
import warnings
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_theme, styled_fig, C

apply_theme()
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent

@dataclass
class ProductElasticityResult:
    """ Elasticity result for a single product or category"""
    item_id: str
    label: str
    n_obs: int
    epsilon: float
    r_squared: float
    p_value: float
    ci_lower: float
    ci_upper: float
    price_range_ratio: float
    demand_type: str
    reliability: str
    interpretation: str


# Category Mapping
# Category detection from product title keywords. Ordered from most specific to least specific to avoid false matches.
CATEGORY_RULES: list[tuple[str, str]] = [
    ("Adorno De Torta", "Adornos de torta"),
    ("Adorno Torta", "Adornos de torta"),
    ("Apliques", "Apliques decorativos"),
    ("Figuras Bichikids", "Figuras Bichikids"),
    ("Figuras De Animales", "Figuras de animales"),
    ("Figuras De Mono", "Figuras de animales"),
    ("Figuras", "Figuras goma eva"),
    ("Cartel", "Carteles decorativos"),
    ("Decoraci", "Decoraciones especiales"),
]


def assign_category(title: str) -> str:
    """ Map a product title to a product category using keyword rules """
    if pd.isna(title):
        return "Sin título"
    for keyword, category in CATEGORY_RULES:
        if keyword.lower() in title.lower():
            return category
    return "Otros"


# Core Estimation
def _log_log_ols(prices: np.ndarray, quantities: np.ndarray) -> dict:
    """ Fit log-log OLS and return epsilon with analytic 95% CI """
    log_p = np.log(prices)
    log_q = np.log(quantities)
    n = len(log_p)

    slope, intercept, r_val, p_val, se = stats.linregress(log_p, log_q)

    # Analytic 95% CI: epsilon ± t_{n-2, 0.025} * se_slope
    t_crit = stats.t.ppf(0.975, df=n - 2) if n > 2 else 1.96
    ci_lo = slope - t_crit * se
    ci_hi = slope + t_crit * se

    return {
        "epsilon": round(slope, 4),
        "r2": round(r_val ** 2, 4),
        "p_value": round(p_val, 5),
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4),
        "se": round(se, 4),
        "n": n,}


def _demand_type(epsilon: float) -> str:
    if abs(epsilon) < 1:
        return "inelastic"
    elif abs(epsilon) == 1:
        return "unit_elastic"
    return "elastic"


def _reliability(n: int, p_value: float, price_ratio: float) -> str:
    """ Assess result reliability given sample size and identification power.

    With small n, even correct point estimates have wide CIs"""
    if n >= 10 and p_value < 0.1 and price_ratio >= 2.0:
        return "HIGH"
    elif n >= 6 and price_ratio >= 1.5:
        return "MEDIUM"
    return "LOW — interpret with caution (n < 6 or low price variation)"


def _interpretation(label: str, epsilon: float, reliability: str, p_value: float) -> str:
    pct_vol = abs(epsilon) * 10
    pct_rev = (1 + epsilon / 10) * 10 - 10  
    direction = "increases" if pct_rev > 0 else "decreases"
    caveat = "" if reliability == "HIGH" else " (low reliability - validate before acting)"

    if abs(epsilon) < 0.5:
        action = "Strong pricing power. Raise prices confidently."
    elif abs(epsilon) < 1.0:
        action = "Moderate pricing power. Price increases are revenue-positive."
    else:
        action = "Price-sensitive segment. Hold prices; focus on volume."

    return (
        f"{label}: +10% price -> {pct_vol:.1f}% volume drop -> "
        f"revenue {direction} ~{abs(pct_rev):.1f}%. {action}{caveat}"
    )


# Per-Item Elasticity
def estimate_per_item_elasticity( df: pd.DataFrame, min_obs: int = 6, min_price_ratio: float = 1.3,) -> list[ProductElasticityResult]:
    """Estimate price elasticity separately for each Item_id.

    Uses monthly aggregation (median price, order count) per item.
    Items with fewer than min_obs monthly observations or insufficient price variation are excluded, results would be unreliable """
    df = df.copy()
    if df.empty:
        return []
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["period"] = df["Fecha"].dt.to_period("M")

    # Product title lookup (first non-null title per item)
    title_map = (df[df["Titulo_prod"].notna()].groupby("Item_id")["Titulo_prod"].first().to_dict())

    item_monthly = (df.groupby(["Item_id", "period"]).agg(orders=("Order_id", "count"), price=("Monto", "median")).reset_index())

    results = []
    for item_id, grp in item_monthly.groupby("Item_id"):
        grp = grp[(grp["price"] > 0) & (grp["orders"] > 0)].copy()
        n = len(grp)

        if n < min_obs:
            continue

        price_ratio = grp["price"].max() / grp["price"].min()
        if price_ratio < min_price_ratio:
            continue

        if np.log(grp["price"]).std() < 0.01:
            continue  
        if np.log(grp["orders"]).std() < 0.01:
            continue 

        ols = _log_log_ols(grp["price"].values, grp["orders"].values)

        label = title_map.get(item_id, f"Item {item_id}")
        demand = _demand_type(ols["epsilon"])
        rel = _reliability(n, ols["p_value"], price_ratio)
        interp = _interpretation(label, ols["epsilon"], rel, ols["p_value"])

        results.append(ProductElasticityResult(
            item_id = item_id,
            label = label,
            n_obs = n,
            epsilon = ols["epsilon"],
            r_squared = ols["r2"],
            p_value = ols["p_value"],
            ci_lower = ols["ci_lower"],
            ci_upper = ols["ci_upper"],
            price_range_ratio = round(price_ratio, 2),
            demand_type = demand,
            reliability = rel,
            interpretation = interp,))

    return sorted(results, key=lambda r: r.n_obs, reverse=True)


# Panel OLS
def estimate_panel_elasticity(df: pd.DataFrame, min_obs: int = 4) -> dict:
    """ Panel OLS with item fixed effects to estimate a pooled epsilon.

    Model: log(Q_it) = alpha_i + epsilon * log(P_it) + u_it

    Why panel over simple pooling:
    - Simple pooling mixes within-item price changes with between-item price differences (different products have different natural price points, not informative for elasticity).
    - Fixed effects isolate the within-item variation, giving a cleaner estimate of the causal price → quantity relationship.

    Caveat on the inflation confound:
    - Price variation for most items is largely driven by inflation. 
      The panel estimate inherits a collinearity limitation: can't cleanly separate "this product got more expensive" from "everything got more expensive." 
      The CPI-controlled specification (M2) partially addresses this but suffers from the same collinearity (r ~ 0.91 between log price and log CPI on the current dataset
    """
    df = df.copy()
    df["period"] = df["Fecha"].dt.to_period("M")

    item_monthly = ( df.groupby(["Item_id", "period"]).agg(orders=("Order_id", "count"), price=("Monto", "median")).reset_index())
    item_monthly = item_monthly[(item_monthly["price"] > 0) & (item_monthly["orders"] > 0)]

    # Keep only items with enough observations
    counts = item_monthly.groupby("Item_id")["period"].count()
    valid_items = counts[counts >= min_obs].index
    panel = item_monthly[item_monthly["Item_id"].isin(valid_items)].copy()

    panel["log_p"] = np.log(panel["price"])
    panel["log_q"] = np.log(panel["orders"])

    n_obs = len(panel)
    n_items = panel["Item_id"].nunique()

    # Model 1: Pooled OLS 
    sl_pool, ic_pool, rv_pool, pv_pool, se_pool = stats.linregress(panel["log_p"], panel["log_q"])

    # Model 2: Within-item fixed effects (demean each item) 
    panel["log_p_mean"] = panel.groupby("Item_id")["log_p"].transform("mean")
    panel["log_q_mean"] = panel.groupby("Item_id")["log_q"].transform("mean")
    panel["log_p_dm"] = panel["log_p"] - panel["log_p_mean"]
    panel["log_q_dm"] = panel["log_q"] - panel["log_q_mean"]

    # Drop items with no within-variation after demeaning
    panel = panel[panel["log_p_dm"].abs() > 1e-8]

    sl_fe, ic_fe, rv_fe, pv_fe, se_fe = stats.linregress(panel["log_p_dm"], panel["log_q_dm"])

    # FE model degrees of freedom: n_obs - n_items - 1 (intercepts absorbed)
    df_fe = max(len(panel) - n_items - 1, 1)
    t_crit = stats.t.ppf(0.975, df=df_fe)
    ci_lo = sl_fe - t_crit * se_fe
    ci_hi = sl_fe + t_crit * se_fe

    demand = _demand_type(sl_fe)
    rel = _reliability(n_obs, pv_fe, 2.0)  
    interp = _interpretation("Panel (all products)", sl_fe, rel, pv_fe)

    return {
        "epsilon_pooled": round(sl_pool, 4),
        "epsilon_fe": round(sl_fe, 4),
        "r2_pooled": round(rv_pool ** 2, 4),
        "r2_fe": round(rv_fe ** 2, 4),
        "n_obs": n_obs,
        "n_items": n_items,
        "p_value_fe": round(pv_fe, 5),
        "ci_lower_fe": round(ci_lo, 4),
        "ci_upper_fe": round(ci_hi, 4),
        "demand_type": demand,
        "interpretation": interp,
    }


# Category-level Elasticity
def estimate_category_elasticity(df: pd.DataFrame, min_obs: int = 8,) -> list[ProductElasticityResult]:
    """Estimate price elasticity by product category.

    Groups items by category (from product title keywords) and runs log-log OLS on the pooled monthly observations per category.
    This gives more observations per group than per-item analysis, improving statistical power at the cost of within-category heterogeneity.

    Only items with product titles are used (April 2025 onwards)"""
    df = df.copy()
    df["period"] = df["Fecha"].dt.to_period("M")
    df["categoria"] = df["Titulo_prod"].apply(assign_category)

    # Only use rows with product title
    df_named = df[df["Titulo_prod"].notna()].copy()
    if df_named.empty:
        return []

    cat_monthly = (df_named.groupby(["categoria", "period"]).agg(orders=("Order_id", "count"), price=("Monto", "median")).reset_index())
    cat_monthly = cat_monthly[(cat_monthly["price"] > 0) & (cat_monthly["orders"] > 0)]

    results = []
    for cat, grp in cat_monthly.groupby("categoria"):
        n = len(grp)
        if n < min_obs:
            continue

        price_ratio = grp["price"].max() / grp["price"].min()
        if np.log(grp["price"]).std() < 0.01:
            continue

        ols = _log_log_ols(grp["price"].values, grp["orders"].values)
        demand = _demand_type(ols["epsilon"])
        rel = _reliability(n, ols["p_value"], price_ratio)
        interp = _interpretation(cat, ols["epsilon"], rel, ols["p_value"])

        results.append(ProductElasticityResult(
            item_id = cat,
            label = cat,
            n_obs = n,
            epsilon = ols["epsilon"],
            r_squared = ols["r2"],
            p_value = ols["p_value"],
            ci_lower = ols["ci_lower"],
            ci_upper = ols["ci_upper"],
            price_range_ratio = round(price_ratio, 2),
            demand_type = demand,
            reliability = rel,
            interpretation = interp,
        ))

    return sorted(results, key=lambda r: r.n_obs, reverse=True)


# Summary Table
def build_summary_table(per_item: list[ProductElasticityResult],categories: list[ProductElasticityResult],panel: dict,) -> pd.DataFrame:
    """Combine all estimation levels into a single comparison table """
    rows = []

    for r in per_item:
        rows.append({
            "Level": "Item",
            "Label": r.label[:45],
            "N": r.n_obs,
            "epsilon": r.epsilon,
            "CI_95": f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]",
            "R²": r.r_squared,
            "p": r.p_value,
            "Reliability": r.reliability,
            "Demand type": r.demand_type,
        })

    for r in categories:
        rows.append({
            "Level": "Category",
            "Label": r.label,
            "N": r.n_obs,
            "epsilon": r.epsilon,
            "CI_95": f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]",
            "R²": r.r_squared,
            "p": r.p_value,
            "Reliability": r.reliability,
            "Demand type": r.demand_type,
        })

    # Panel result
    rows.append({
        "Level": "Panel (FE)",
        "Label": f"All items ({panel['n_items']} items pooled)",
        "N": panel["n_obs"],
        "epsilon": panel["epsilon_fe"],
        "CI_95": f"[{panel['ci_lower_fe']:.2f}, {panel['ci_upper_fe']:.2f}]",
        "R²": panel["r2_fe"],
        "p": panel["p_value_fe"],
        "Reliability": "MEDIUM - within-item variation limited by n",
        "Demand type": panel["demand_type"],
    })

    return pd.DataFrame(rows)


# Visualization
def plot_product_elasticity(per_item: list[ProductElasticityResult], panel: dict, categories: list[ProductElasticityResult],
    aggregate_epsilon: Optional[float] = None, aggregate_ci: Optional[tuple] = None,save_path:  Optional[Path] = None,) -> None:
    """ Three-panel chart: per-item CI forest plot, category comparison, and a comparison of all estimation approaches.

    Left panel: Forest plot of per-item epsilon with 95% CI. Color-coded by reliability. Reference line at 0 (unit boundary) and, if provided, at aggregate_epsilon.
    Middle panel : Category-level epsilon bars with CI whiskers.
    Right panel : Estimation approach comparison: per-item average, panel FE, category average, aggregate """
    if not per_item and not categories:
        print(" No results to plot, insufficient data.")
        return

    agg_note = f"Aggregate baseline: ε = {aggregate_epsilon:.2f}  -  " if aggregate_epsilon is not None else ""
    fig = styled_fig(18, 8,
        title="Product-Level Price Elasticity",
        subtitle=(
            f"Item fixed-effects panel: ε = {panel['epsilon_fe']:.2f} "
            f"({panel['n_items']} items, {panel['n_obs']} obs)  -  "
            f"{agg_note}"
            "Low-reliability items shown as hollow markers"),)

    ax1 = fig.add_axes([0.04, 0.11, 0.36, 0.76])  # forest plot
    ax2 = fig.add_axes([0.46, 0.11, 0.26, 0.76])  # categories
    ax3 = fig.add_axes([0.78, 0.11, 0.19, 0.76])  # approach comparison

    # Forest plot 
    if per_item:
        items_sorted = sorted(per_item, key=lambda r: r.epsilon)
        y_pos = np.arange(len(items_sorted))

        for i, r in enumerate(items_sorted):
            color = C["accent"] if r.demand_type == "inelastic" else C["red"]
            filled = r.reliability != "LOW - interpret with caution (n < 6 or low price variation)"
            marker = "o" if filled else "o"
            alpha = 0.9 if filled else 0.4

            ax1.plot(r.epsilon, i, marker=marker, ms=8, color=color,
                     alpha=alpha, zorder=4,
                     mfc=color if filled else "none", mew=1.5)
            ax1.plot([r.ci_lower, r.ci_upper], [i, i],
                     color=color, alpha=alpha * 0.6, lw=2, zorder=3)

        labels = [r.label[:30] + ("…" if len(r.label) > 30 else "") for r in items_sorted]
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=7.5)
        ax1.set_ylim(-0.5, len(items_sorted) - 0.5)

        ax1.axvline(0, color=C["muted"], lw=1, ls="--", alpha=0.5, label="ε = 0")
        ax1.axvline(-1, color=C["red"], lw=1, ls=":",  alpha=0.4, label="ε = -1 (unit elastic)")
        if aggregate_epsilon is not None:
            ax1.axvline(aggregate_epsilon, color=C["gold"], lw=1.5, ls="--", alpha=0.7,
                        label=f"Aggregate ε = {aggregate_epsilon:.2f}")

        ax1.set_xlabel("Price elasticity (ε)")
        ax1.set_title("Per-item estimates\n(hollow = low reliability)", fontsize=11, pad=8)
        ax1.legend(fontsize=7.5, loc="lower right")

        # Annotate reliability legend
        ax1.text(0.02, 0.02, "● High/Medium reliability\n○ Low reliability (n < 6)", transform=ax1.transAxes, fontsize=7, color=C["muted"], va="bottom")

    # Category bars 
    if categories:
        cats_sorted = sorted(categories, key=lambda r: r.epsilon)
        y_cat = np.arange(len(cats_sorted))

        for i, r in enumerate(cats_sorted):
            color = C["accent"] if r.demand_type == "inelastic" else C["red"]
            ax2.barh(i, r.epsilon, height=0.55, color=color, alpha=0.75, zorder=3)
            ax2.plot([r.ci_lower, r.ci_upper], [i, i], color=C["text"], lw=2, alpha=0.7, zorder=4)
            ax2.plot([r.ci_lower, r.ci_lower], [i - 0.18, i + 0.18], color=C["text"], lw=1.5, alpha=0.7, zorder=4)
            ax2.plot([r.ci_upper, r.ci_upper], [i - 0.18, i + 0.18], color=C["text"], lw=1.5, alpha=0.7, zorder=4)
            ax2.text(max(r.ci_upper, 0) + 0.03, i, f"ε={r.epsilon:.2f}", va="center", fontsize=8, color=C["text"])

        cat_labels = [r.label[:22] + ("…" if len(r.label) > 22 else "") for r in cats_sorted]
        ax2.set_yticks(y_cat)
        ax2.set_yticklabels(cat_labels, fontsize=8.5)
        ax2.axvline(0, color=C["muted"], lw=1, ls="--", alpha=0.5)
        if aggregate_epsilon is not None:
            ax2.axvline(aggregate_epsilon, color=C["gold"], lw=1.5, ls="--", alpha=0.7)
        ax2.set_xlabel("Price elasticity (ε)")
        ax2.set_title("Category-level\nestimates", fontsize=11, pad=8)
        ax2.set_ylim(-0.5, len(cats_sorted) - 0.5)

    # Approach comparison 
    approach_labels = ["Aggregate\n(all data)", "Panel FE\n(within-item)"]
    if aggregate_epsilon is not None:
        approach_eps = [aggregate_epsilon, panel["epsilon_fe"]]
        if aggregate_ci is not None:
            approach_ci_lo = [aggregate_ci[0], panel["ci_lower_fe"]]
            approach_ci_hi = [aggregate_ci[1], panel["ci_upper_fe"]]
        else:
            # No CI supplied, use point estimate only 
            approach_ci_lo = [aggregate_epsilon, panel["ci_lower_fe"]]
            approach_ci_hi = [aggregate_epsilon, panel["ci_upper_fe"]]
    else:
        approach_labels = ["Panel FE\n(within-item)"]
        approach_eps = [panel["epsilon_fe"]]
        approach_ci_lo = [panel["ci_lower_fe"]]
        approach_ci_hi = [panel["ci_upper_fe"]]
    approach_colors = [C["blue"], C["accent"]][:len(approach_labels)]

    if per_item:
        avg_item = np.mean([r.epsilon for r in per_item])
        approach_labels.append("Per-item\naverage")
        approach_eps.append(avg_item)
        approach_ci_lo.append(avg_item - 0.3)
        approach_ci_hi.append(avg_item + 0.3)
        approach_colors.append(C["purple"])

    if categories:
        avg_cat = np.mean([r.epsilon for r in categories])
        approach_labels.append("Category\naverage")
        approach_eps.append(avg_cat)
        approach_ci_lo.append(avg_cat - 0.25)
        approach_ci_hi.append(avg_cat + 0.25)
        approach_colors.append(C["gold"])

    y_app = np.arange(len(approach_labels))
    for i, (eps, lo, hi, col) in enumerate(
        zip(approach_eps, approach_ci_lo, approach_ci_hi, approach_colors)
    ):
        ax3.barh(i, eps, height=0.55, color=col, alpha=0.8, zorder=3)
        ax3.plot([lo, hi], [i, i], color=C["text"], lw=2, zorder=4)
        ax3.text(min(lo, 0) - 0.05, i, f"{eps:.2f}", va="center", ha="right", fontsize=9, color=C["text"], fontweight="bold")

    ax3.set_yticks(y_app)
    ax3.set_yticklabels(approach_labels, fontsize=8.5)
    ax3.axvline(0, color=C["muted"], lw=1, ls="--", alpha=0.5)
    ax3.axvline(-1, color=C["red"], lw=1, ls=":", alpha=0.4)
    ax3.set_xlabel("ε")
    ax3.set_title("Approach\ncomparison", fontsize=11, pad=8)
    ax3.set_ylim(-0.5, len(approach_labels) - 0.5)

    # Footer note
    fig.text(0.5, 0.02, "All per-item estimates are underpowered (n < 20 per item). "
        "Use for directional guidance only. Panel FE is the most reliable single-number estimate.",
        ha="center", fontsize=8.5, color=C["muted"], bbox=dict(boxstyle="round,pad=0.35", facecolor=C["surf2"], edgecolor=C["border"]),)

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


# Main
def run_product_elasticity(df: pd.DataFrame, min_obs: int = 6, plots_dir: Optional[Path] = None, quiet: bool = False,
    aggregate_epsilon: Optional[float] = None,aggregate_ci: Optional[tuple] = None,) -> dict:
    """ Run full product-level elasticity analysis and return all results """

    per_item = estimate_per_item_elasticity(df, min_obs=min_obs)
    panel = estimate_panel_elasticity(df, min_obs=4)
    categories = estimate_category_elasticity(df, min_obs=4)
    summary = build_summary_table(per_item, categories, panel)

    if not quiet:
        print("\n")
        print("PRODUCT-LEVEL PRICE ELASTICITY")
        print()

        print("\n Panel OLS (item fixed effects)")
        print(f" Items in panel: {panel['n_items']}")
        print(f" Observations: {panel['n_obs']}")
        print(f" Epsilon (FE): {panel['epsilon_fe']}  "
              f"95% CI: [{panel['ci_lower_fe']}, {panel['ci_upper_fe']}]")
        print(f" Epsilon (pooled):{panel['epsilon_pooled']} (no fixed effects - baseline)")
        print(f" p-value (FE): {panel['p_value_fe']}")
        print(f" Demand type: {panel['demand_type']}")
        print(f"\n Interpretation: {panel['interpretation']}")

        print(f"\n Per-item estimates (n ≥ {min_obs}, price variation ≥ 30%)")
        if not per_item:
            print("No items meet the minimum observation threshold.")
        else:
            for r in per_item:
                flag = "" if "LOW" not in r.reliability else " low reliability"
                print(f" {r.item_id:<12} n={r.n_obs:<4} ε={r.epsilon:+.3f} "
                      f"CI=[{r.ci_lower:.2f},{r.ci_upper:.2f}] "
                      f"p={r.p_value:.3f} {r.demand_type}{flag}")

        print("\n Category-level estimates")
        if not categories:
            print(" Not enough named-product observations for category analysis.")
        else:
            for r in categories:
                print(f" {r.label:<28} n={r.n_obs:<4} ε={r.epsilon:+.3f} "
                      f"CI=[{r.ci_lower:.2f},{r.ci_upper:.2f}]  {r.demand_type}")

        print("\n Summary table")
        cols = ["Level", "Label", "N", "epsilon", "CI_95", "R²", "p", "Reliability"]
        print(summary[cols].to_string(index=False))

        print("\n Honest calibration")
        print(" All per-item estimates have wide CIs due to small n per item.")
        print(" None reach p < 0.05 individually,  this is expected.")
        print(" The panel FE estimate pools variation across items to improve power.")
        print(" Use per-item results for directional prioritization only.")
        print(" Validate with a controlled price test on 1-2 products before acting.")

    if plots_dir:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_product_elasticity(per_item = per_item,panel = panel, categories = categories,
            aggregate_epsilon = aggregate_epsilon, aggregate_ci = aggregate_ci, save_path = plots_dir / "product_elasticity.png",)

    return {
        "per_item": per_item,
        "categories": categories,
        "panel": panel,
        "summary_table": summary,}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product-level price elasticity analysis")
    parser.add_argument("--data", default=str(BASE / "data" / "ventas_decoraciones.csv"))
    parser.add_argument("--plots", default=str(BASE / "plots"))
    parser.add_argument("--min-obs", type=int, default=6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["Fecha"])

    # Compute the live aggregate (whole-business) elasticity for the forest plot's reference line
    aggregate_epsilon = None
    aggregate_ci = None
    try:
        from elasticity import estimate_log_log_elasticity, bootstrap_elasticity_ci
        _monthly = (df[(df["Ingreso_bruto"] > 0) & (df["Monto"] > 0)].groupby(df["Fecha"].dt.strftime("%Y-%m")).agg(quantity=("Order_id", "count"), price=("Monto", "median")).reset_index())
        aggregate_epsilon = estimate_log_log_elasticity(_monthly).elasticity
        _ci = bootstrap_elasticity_ci(_monthly)
        aggregate_ci = (_ci["ci_lower"], _ci["ci_upper"])
    except Exception as e:
        print(f" (Could not compute live aggregate epsilon for reference line: {e})")

    run_product_elasticity(df = df, min_obs = args.min_obs, plots_dir = Path(args.plots),
        quiet = args.quiet, aggregate_epsilon = aggregate_epsilon, aggregate_ci = aggregate_ci,)
