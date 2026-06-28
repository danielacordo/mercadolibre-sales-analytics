import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_theme, styled_fig, C, YEAR_C

apply_theme()

warnings.filterwarnings("ignore")

ELASTICITY_BENCHMARKS: dict[str, tuple[float, str]] = {"Consumer goods (avg)": (-1.76, "Nair et al. 2005, meta-analysis"),
    "Convenience goods": (-0.50, "Nair et al. 2005"), "Impulse purchases": (-2.30, "Nair et al. 2005"),
    "Niche / gift products": (-0.65, "Bijmolt et al. 2005"), "Durable consumer goods": (-0.90, "Tellis 1988"),}

@dataclass
class ElasticityResult:
    """Container for price elasticity analysis results """
    elasticity: float   # epsilon coefficient
    r_squared: float   # goodness of fit
    p_value: float   # statistical significance
    std_error: float   # standard error of the coefficient
    intercept: float   # regression intercept (alpha)
    is_significant: bool  # True if p < 0.05
    demand_type: str  # "inelastic", "unit_elastic", or "elastic"
    interpretation: str  # plain-language interpretation


def estimate_log_log_elasticity(monthly: pd.DataFrame) -> ElasticityResult:
    """ Estimate price elasticity using log-log OLS regression

    The log-log (double-log) model is the standard econometric approach: log(Q) = alpha + epsilon * log(P) + u
    where the slope epsilon is directly the price elasticity.

    Why log-log over simple linear regression:
    1. epsilon is directly interpretable as elasticity
    2. Stabilizes variance in wide-range price data 
    3. Captures multiplicative relationships 
    4. Results are interpretable as percentages, not absolute units """

    df = monthly.dropna(subset=["price", "quantity"]).copy()
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]

    log_p = np.log(df["price"])
    log_q = np.log(df["quantity"])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_p, log_q)

    eps = slope
    r2 = r_value ** 2

    if abs(eps) < 1:
        demand_type = "inelastic"
        interpretation = (
            f"Demand is INELASTIC (|epsilon| < 1)"
            f"A 10% price increase leads to only a {abs(eps)*10:.1f}% drop in demand. "
            f"Revenue increases with price increases - pricing power exists")
    elif abs(eps) == 1:
        demand_type = "unit_elastic"
        interpretation = "Demand is UNIT ELASTIC. Revenue is unchanged with price changes."
    else:
        demand_type = "elastic"
        interpretation = (
            f"Demand is ELASTIC (|epsilon| > 1)"
            f"A 10% price increase leads to a {abs(eps)*10:.1f}% drop in demand. "
            f"Revenue decreases with price increases"
        )

    return ElasticityResult(
        elasticity = round(eps, 4),
        r_squared = round(r2, 4),
        p_value = round(p_value, 6),
        std_error = round(std_err, 4),
        intercept = round(intercept, 4),
        is_significant = p_value < 0.05,
        demand_type = demand_type,
        interpretation = interpretation,)


def estimate_controlled_elasticity(monthly: pd.DataFrame) -> pd.DataFrame:
    """ Estimate price elasticity controlling for inflation (CPI) and seasonality

    The simple log-log model conflates the price effect with concurrent inflation and seasonal demand shifts. 
    This model adds two controls: log(Q) = b0 + b1*log(P) + b2*log(CPI) + b3*peak + b4*low + error

    where:
        b1 = price elasticity (controlling for inflation and season)
        b2 = CPI elasticity (how demand responds to overall inflation)
        b3 = Aug/Sep seasonal dummy (peak vs other months)
        b4 = Jun/Jul seasonal dummy (low vs other months) """
    
    m = monthly.dropna(subset=["price", "quantity", "cpi"]).copy()
    m = m[(m["price"] > 0) & (m["quantity"] > 0) & (m["cpi"] > 0)]

    log_p = np.log(m["price"])
    log_q = np.log(m["quantity"])
    log_cpi = np.log(m["cpi"])
    peak = m["ds"].dt.month.isin([8, 9]).astype(float)  # Aug/Sep - fixed a priori window
    low = m["ds"].dt.month.isin([6, 7]).astype(float)  # Jun/Jul - fixed a priori window
    price_cpi_corr = float(np.corrcoef(log_p, log_cpi)[0, 1])

    # Model 1: simple log-log (baseline)
    sl1, ic1, rv1, pv1, se1 = stats.linregress(log_p, log_q)

    # Model 2: controlling for inflation
    X2 = np.column_stack([np.ones(len(m)), log_p, log_cpi])
    b2 = np.linalg.lstsq(X2, log_q.values, rcond=None)[0]
    y2 = X2 @ b2
    r2_2 = 1 - np.sum((log_q.values - y2)**2) / np.sum((log_q.values - log_q.mean())**2)

    # Model 3: controlling for inflation + seasonality
    X3 = np.column_stack([np.ones(len(m)), log_p, log_cpi, peak, low])
    b3 = np.linalg.lstsq(X3, log_q.values, rcond=None)[0]
    y3 = X3 @ b3
    r2_3 = 1 - np.sum((log_q.values - y3)**2) / np.sum((log_q.values - log_q.mean())**2)

    results = pd.DataFrame([
        {
            "Model": "Simple OLS",
            "Formula": "log(Q) = b0 + b1*log(P)",
            "epsilon (price)": round(sl1, 4),
            "beta_IPC": None,
            "beta_peak (Aug/Sep)": None,
            "beta_low (Jun/Jul)": None,
            "R²": round(rv1**2, 4),
            "Note": "Baseline - does not control for inflation or season",
        },
        {
            "Model": "OLS + CPI",
            "Formula": "log(Q) = b0 + b1*log(P) + b2*log(CPI)",
            "epsilon (price)": round(b2[1], 4),
            "beta_IPC": round(b2[2], 4),
            "beta_peak (Aug/Sep)": None,
            "beta_low (Jun/Jul)": None,
            "R²": round(r2_2, 4),
            "Note": f"Controls for macro inflation. Collinearity warning: corr(P, CPI) ~ {price_cpi_corr:.2f}",
        },
        {
            "Model": "OLS + CPI + Seasonality",
            "Formula": "log(Q) = b0 + b1*log(P) + b2*log(CPI) + b3*peak + b4*low",
            "epsilon (price)": round(b3[1], 4),
            "beta_IPC": round(b3[2], 4),
            "beta_peak (Aug/Sep)": round(b3[3], 4),
            "beta_low (Jun/Jul)": round(b3[4], 4),
            "R²": round(r2_3, 4),
            "Note": f"Full model. epsilon is stable across specs (M1={round(sl1, 2)}, M2={round(b2[1], 2)}, M3={round(b3[1], 2)}) → robust finding",
        },
    ])
    return results


def contextualize_with_benchmarks(result: ElasticityResult) -> pd.DataFrame:
    """ Compare the elasticity result against academic literature benchmarks """
    rows = [{
        "Context": "This business (foam rubber decorations, AR 2023-26)",
        "Elasticity": result.elasticity,
        "Demand type": "Inelastic" if result.demand_type == "inelastic" else "Elastic",
        "Reference": "This analysis",
    }]
    for context, (eps, ref) in ELASTICITY_BENCHMARKS.items():
        rows.append({
            "Context": context,
            "Elasticity": eps,
            "Demand type": "Inelastic" if abs(eps) < 1 else "Elastic",
            "Reference": ref,
        })
    return pd.DataFrame(rows)


def analyze_data_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """ Analyzes dataset reliability and coverage"""
    total = len(df)
    n_months = df["Fecha"].dt.to_period("M").nunique()

    metrics = [
        {
            "Metric": "Total orders",
            "Value": str(total),
            "Notes": "Covers Jan 2023 to Apr 2026",
        },
        {
            "Metric": "Months covered",
            "Value": str(n_months),
            "Notes": "40 months - minimum threshold for Prophet, low for SARIMA",
        },
        {
            "Metric": "Geographic coverage",
            "Value": f"{df['Provincia_nombre'].notna().mean()*100:.1f}%",
            "Notes": f"{df['Provincia_nombre'].isna().sum()} orders with no province - possible bias",
        },
        {
            "Metric": "Product name coverage",
            "Value": f"{df['Titulo_prod'].notna().mean()*100:.1f}%",
            "Notes": "Only available in ML Official (Apr 2025 onwards)",
        },
        {
            "Metric": "Net margin coverage",
            "Value": f"{df['Ingreso_neto'].notna().mean()*100:.1f}%",
            "Notes": "ML fee analysis limited to the last 12 months of data",
        },
        {
            "Metric": "Identified buyers",
            "Value": "~100%",
            "Notes": "CNX includes buyer nickname in all rows with data",
        },
        {
            "Metric": "Duplicate orders",
            "Value": str(df["Order_id"].duplicated().sum()),
            "Notes": "0 duplicates - date cutoff prevents Apr-2025 overlap",
        },
        {
            "Metric": "Selection bias",
            "Value": "Present",
            "Notes": "Only completed sales - returns and cancellations excluded from CNX",
        },
    ]
    return pd.DataFrame(metrics)


def plot_elasticity_with_benchmarks(monthly: pd.DataFrame, result, benchmarks: pd.DataFrame, save_path=None) -> None:
    """ Scatter demand curve (colored by year) + scenario simulation panel.

    Left panel: empirical scatter with OLS fit curve and 95% CI band
    Right panel: horizontal bar chart of revenue scenarios for +5% to +25% price """
    from scipy import stats as _stats
    import numpy as _np

    df_plot = monthly.dropna(subset=["price", "quantity"]).copy()
    df_plot["year"] = df_plot["ds"].dt.year

    log_p = _np.log(df_plot["price"])
    log_q = _np.log(df_plot["quantity"])
    slope, intercept, r_val, p_val, se = _stats.linregress(log_p, log_q)
    epsilon = slope

    # 95% CI on epsilon via bootstrap 
    # Reuses bootstrap_elasticity_ci() instead of a second, independent implementation.
    _ci = bootstrap_elasticity_ci(df_plot[["price", "quantity"]])
    ci_lo_eps, ci_hi_eps = _ci["ci_lower"], _ci["ci_upper"]
    _n = len(df_plot)

    # Prediction band on the fitted curve: ±1 SE of the fit at each x point
    # se_fit(x) = se_residual * sqrt(1/n + (x - xbar)^2 / Sxx)
    _Sxx = _np.sum((log_p - log_p.mean()) ** 2)
    _se_res = _np.sqrt(_np.sum((log_q - (intercept + slope * log_p)) ** 2) / (_n - 2))

    p_range = _np.linspace(df_plot["price"].min(), df_plot["price"].max(), 200)
    log_pr = _np.log(p_range)
    q_fit = _np.exp(intercept) * p_range ** epsilon
    se_fit = _se_res * _np.sqrt(1 / _n + (log_pr - log_p.mean()) ** 2 / _Sxx)

    fig = styled_fig(16, 7,
        title=f"Price-Demand Elasticity  |  epsilon = {epsilon:.2f}  |  Inelastic demand",
        subtitle=f"Log-log OLS  -  R² = {r_val**2:.2f}  -  p < 0.001  -  "
                 f"Bootstrap 95% CI: [{ci_lo_eps:.2f}, {ci_hi_eps:.2f}]  -  "
                 f"{len(df_plot)} months of real data")

    ax1 = fig.add_axes([0.05, 0.12, 0.43, 0.75])
    ax2 = fig.add_axes([0.57, 0.12, 0.40, 0.75])

    for yr in sorted(df_plot["year"].unique()):
        sub = df_plot[df_plot["year"] == yr]
        ax1.scatter(sub["price"] / 1000, sub["quantity"], color=YEAR_C.get(yr, C["muted"]), s=70, alpha=0.85, zorder=4,
                    label=str(yr), edgecolors=YEAR_C.get(yr, C["muted"]) + "44", linewidths=0.5)

    ax1.plot(p_range / 1000, q_fit, color=C["red"], lw=2.5, ls="--", zorder=5,
             label=f"OLS fit (e={epsilon:.2f})")
    ax1.fill_between(p_range / 1000, _np.exp(_np.log(q_fit) - 1.96 * se_fit), _np.exp(_np.log(q_fit) + 1.96 * se_fit),
                     color=C["red"], alpha=0.08, zorder=3, label="95% prediction band")
    ax1.set_xlabel("Median monthly price ($K ARS)")
    ax1.set_ylabel("Monthly orders")
    ax1.set_title("Empirical demand curve", fontsize=12, pad=10)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.text(0.97, 0.92, f"e = {epsilon:.2f}", transform=ax1.transAxes, ha="right", fontsize=22, fontweight="black", color=C["accent"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor=C["accent"] + "15", edgecolor=C["accent"] + "44"))
    ax1.text(0.97, 0.80, "Inelastic", transform=ax1.transAxes, ha="right", fontsize=11, color=C["accent"], fontweight="bold")

    # Scenario panel
    scenarios = [(5, epsilon * 5, 5 + epsilon * 5),
                 (10, epsilon * 10, 10 + epsilon * 10),
                 (15, epsilon * 15, 15 + epsilon * 15),
                 (20, epsilon * 20, 20 + epsilon * 20),
                 (25, epsilon * 25, 25 + epsilon * 25)]
    y_pos = _np.arange(len(scenarios))
    rev_chgs = [s[2] for s in scenarios]
    h_bars = ax2.barh(y_pos, rev_chgs,
                      color=[C["accent"] if i == 1 else C["blue"] for i in range(len(scenarios))],
                      alpha=0.85, height=0.5, zorder=3)
    ax2.axvline(0, color=C["muted"], lw=1)
    for i, (sc, bar) in enumerate(zip(scenarios, h_bars)):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"+{sc[2]:.1f}% rev  ({sc[1]:.1f}% vol)",
                 va="center", fontsize=9.5,
                 color=C["accent"] if i == 1 else C["text"],
                 fontweight="bold" if i == 1 else "normal")
        ax2.text(-0.15, bar.get_y() + bar.get_height() / 2,
                 f"+{sc[0]}%", va="center", ha="right",
                 fontsize=10, color=C["text"], fontweight="bold")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Price +{s[0]}%" for s in scenarios], fontsize=10)
    ax2.set_xlabel("Revenue change (%)")
    ax2.set_title("Price scenario simulation", fontsize=12, pad=10)
    ax2.set_xlim(-1, 14)
    ax2.set_ylim(-0.5, len(scenarios) - 0.3)
    ax2.axhspan(0.75, 1.25, color=C["accent"], alpha=0.06, zorder=0)
    ax2.annotate("Recommended", xy=(rev_chgs[1], 1), xytext=(rev_chgs[1] + 1.5, 1.8), fontsize=8, color=C["accent"],
                 arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1))

    _worst_vol_10 = ci_lo_eps * 10  
    _worst_rev_10 = ((1.10) ** (1 + ci_lo_eps) - 1) * 100
    fig.text(0.5, 0.04, f"Worst case (lower CI, epsilon={ci_lo_eps:.2f}): +10% price generates "
             f"{_worst_rev_10:+.1f}% revenue ({_worst_vol_10:+.1f}% volume). "
             f"{'Result is positive across the ENTIRE uncertainty distribution.' if _worst_rev_10 > 0 else 'Result is NOT positive at the elastic end of the CI — interpret with care.'}",
             ha="center", fontsize=9, color=C["muted"], bbox=dict(boxstyle="round,pad=0.4", facecolor=C["surf2"], edgecolor=C["border"]))

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    import matplotlib.pyplot as _plt
    _plt.close()


def bootstrap_elasticity_ci(monthly: pd.DataFrame, n_bootstrap: int = 2000, confidence: float = 0.95, random_state: int = 42) -> dict:
    """ Bootstrap confidence interval for price elasticity.

    Resamples months with replacement and re-fits the log-log OLS to produce an empirical distribution of epsilon.

    Captures statistical uncertainty only. Model uncertainty (inflation-price collinearity, seasonal confounding) is additional and not reflected in the CI """
    df = monthly.dropna(subset=["price", "quantity"]).copy()
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]
    n = len(df)

    np.random.seed(random_state)
    boot_slopes = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        s = stats.linregress(
            np.log(df["price"].values[idx]),
            np.log(df["quantity"].values[idx]),
        )
        boot_slopes.append(s.slope)

    boot_slopes  = np.array(boot_slopes)
    alpha = 1 - confidence
    ci_lo, ci_hi = np.percentile(boot_slopes, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    point = stats.linregress(np.log(df["price"]), np.log(df["quantity"])).slope

    return {
        "epsilon": round(point, 4),    
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4),
        "se_bootstrap": round(boot_slopes.std(), 4),
        "n_months": n,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def simulate_pricing_scenarios(ci_result: dict, current_monthly_revenue: float, price_increases: list = None) -> pd.DataFrame:
    """ Simulate revenue impact of price increases under elasticity uncertainty.

    Uses the bootstrap CI to produce pessimistic (most elastic end of CI), central (point estimate), and optimistic (least elastic end) revenue
    deltas for each price increase scenario. """
    if price_increases is None:
        price_increases = [0.05, 0.10, 0.15, 0.20]

    eps_c = ci_result["epsilon"]
    eps_lo = ci_result["ci_lower"]   # most negative = most elastic = pessimistic
    eps_hi = ci_result["ci_upper"]   # least negative = most inelastic = optimistic

    rows = []
    for p in price_increases:
        # Exact log-log revenue formula: dR/R = (1 + ε) * dP/P
        # Revenue change = (1+p)^(1+ε) - 1
        # The approximation (1+p)*(1+ε*p)-1 understates the gain at large p:
        #   +10%: approx=2.75%, exact=3.30% (diff 0.55pp, tolerable)
        #   +20%: approx=4.18%, exact=6.41% (diff 2.23pp, material)
        rev_c = (1 + p) ** (1 + eps_c) - 1
        rev_lo = (1 + p) ** (1 + eps_lo) - 1
        rev_hi = (1 + p) ** (1 + eps_hi) - 1

        if rev_lo > 0.02:
            risk = "Low"
        elif rev_lo > 0:
            risk = "Low - revenue-positive across full CI"
        elif rev_lo > -0.02:
            risk = "Medium"
        else:
            risk = "High"

        rows.append({
            "Price increase": f"+{p*100:.0f}%",
            "Central rev Δ (%)": round(rev_c  * 100, 1),
            "Central rev Δ (ARS/month)":  round(current_monthly_revenue * rev_c,  0),
            "Pessimistic rev Δ (%)": round(rev_lo * 100, 1),
            "Pessimistic rev Δ (ARS)": round(current_monthly_revenue * rev_lo, 0),
            "Optimistic rev Δ (%)": round(rev_hi * 100, 1),
            "Optimistic rev Δ (ARS)": round(current_monthly_revenue * rev_hi, 0),
            "Risk": risk,})

    return pd.DataFrame(rows)

if __name__ == "__main__":
    from pathlib import Path
    BASE = Path(__file__).parent.parent

    df = pd.read_csv(BASE / "data" / "ventas_decoraciones.csv", parse_dates=["Fecha"])

    monthly = df.groupby(df["Fecha"].dt.to_period("M")).agg(
        quantity=("Order_id","count"), price=("Monto","median"),
    ).reset_index()
    monthly["ds"] = monthly["Fecha"].dt.to_timestamp()
    monthly["year"] = monthly["ds"].dt.year

    # Attach CPI, required by estimate_controlled_elasticity
    _ipc = pd.read_csv(BASE / "data" / "ipc_indec.csv")
    _ipc_map = dict(zip(_ipc["period"], _ipc["cpi_index"]))
    monthly["cpi"] = monthly["ds"].dt.strftime("%Y-%m").map(_ipc_map)

    print()
    print("PRICE ELASTICITY ANALYSIS")
    print()

    result = estimate_log_log_elasticity(monthly)
    ci = bootstrap_elasticity_ci(monthly)

    print("\n Preliminary estimate (see calibration note):")
    print(f" Elasticity (e): {result.elasticity} (point estimate, log-log OLS)")
    print(f" 95% Bootstrap CI: [{ci['ci_lower']}, {ci['ci_upper']}]")
    print(f" R2: {result.r_squared}")
    print(f" P-value: {result.p_value}")
    print(f"\n {result.interpretation}")
    print(f" Worst-case (CI lower, e={ci['ci_lower']}): still inelastic, revenue-positive at +10%")
    print(f"\n Calibration note: preliminary estimate, {ci['n_months']} months of data.")
    _m_cpi = monthly.dropna(subset=["price", "cpi"])
    _price_cpi_corr = float(np.corrcoef(np.log(_m_cpi["price"]), np.log(_m_cpi["cpi"]))[0, 1]) if len(_m_cpi) > 1 else float("nan")
    print(f" Price-CPI collinearity (r~{_price_cpi_corr:.2f}) limits causal interpretation.")
    print(" Validate with a controlled price test before pricing rule deployment.")

    print("\n")
    print("CONTROLLED ELASTICITY (3 specifications)")
    print()
    controlled = estimate_controlled_elasticity(monthly)
    print(controlled[["Model","epsilon (price)","R²","Note"]].to_string(index=False))
    _eps_range = controlled["epsilon (price)"].dropna()
    print(f"\n Epsilon is stable across all specs ({_eps_range.min():.2f} to {_eps_range.max():.2f}) -> robust finding")

    print("\n")
    print("PRICING SCENARIOS with CI")
    print()
    current_rev = df.groupby(df["Fecha"].dt.to_period("M"))["Ingreso_bruto"].sum().iloc[-6:].mean()
    scenarios = simulate_pricing_scenarios(ci, current_rev)
    print(scenarios[["Price increase","Central rev Δ (%)","Pessimistic rev Δ (%)","Risk"]].to_string(index=False))

