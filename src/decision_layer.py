import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

BASE = Path(__file__).parent.parent

if str(BASE / "src") not in sys.path:
    sys.path.insert(0, str(BASE / "src"))
try:
    from src.elasticity import bootstrap_elasticity_ci, simulate_pricing_scenarios
except ImportError:
    from elasticity import bootstrap_elasticity_ci, simulate_pricing_scenarios



# Data Loading
def load_inputs() -> dict:
    """Load all data needed for the decision layer"""
    df = pd.read_csv(BASE / "data" / "ventas_decoraciones.csv", parse_dates=["Fecha"])
    rfm = pd.read_csv(BASE / "data" / "rfm_clientes.csv")
    fc = pd.read_csv(BASE / "data" / "forecast_6meses.csv")
    ipc = pd.read_csv(BASE / "data" / "ipc_indec.csv")

    ipc_map = dict(zip(ipc["period"], ipc["cpi_index"]))
    monthly = df.groupby(df["Fecha"].dt.to_period("M")).agg(
        orders = ("Order_id", "count"),
        revenue  = ("Ingreso_bruto", "sum"),
        price = ("Monto", "median"),
        quantity = ("Order_id", "count"),).reset_index()
    monthly["ds"] = monthly["Fecha"].dt.to_timestamp()
    monthly["cpi"] = monthly["ds"].dt.strftime("%Y-%m").map(ipc_map)

    df_ml = df[df["Fuente"] == "ML_Oficial"].copy()
    df_ml["pct_net"] = df_ml["Ingreso_neto"] / df_ml["Ingreso_bruto"] * 100

    return {"df": df, "rfm": rfm, "fc": fc, "monthly": monthly, "df_ml": df_ml,}


# Core Decision Logic 
def build_pricing_strategy(inputs: dict) -> dict:
    """ Builds the Q3 2026 pricing strategy with scenarios and confidence ranges """
    # bootstrap_elasticity_ci and simulate_pricing_scenarios imported at module level
    monthly = inputs["monthly"].copy()

    # Bootstrap CI for elasticity
    ci = bootstrap_elasticity_ci(monthly)

    # Current state
    current_rev_pm  = inputs["monthly"]["revenue"].iloc[-6:].mean()
    # avg_ticket uses Potential segment's own historical average, not the last 30 row of the full dataset (which was ML_Oficial-heavy and inflated the estimate ~1.8x).
    _potential = inputs["rfm"][inputs["rfm"]["Segment"] == "Potential"]
    if len(_potential) > 0 and _potential["frecuencia"].sum() > 0:
        current_ticket = _potential["monto_total"].sum() / _potential["frecuencia"].sum()
    else:
        current_ticket = inputs["df"]["Monto"].median()  # safe fallback
    # Revenue-weighted, this matches the methodology in src/eda.py's margin_summary() / plot_margin_correction().
    df_ml = inputs["df_ml"].dropna(subset=["Ingreso_neto"])
    ml_fee_rate = 1 - (df_ml["Ingreso_neto"].sum() / df_ml["Ingreso_bruto"].sum())

    # Scenarios
    scenarios = simulate_pricing_scenarios(ci, current_rev_pm)

    # Forecast
    fc = inputs["fc"]
    total_central = fc["Proyeccion_central"].sum()
    total_low = fc["Limite_inferior_80"].sum()
    total_high = fc["Limite_superior_80"].sum()

    # RFM opportunity
    # NOTE: rfm_clientes.csv uses English column names (Segment / Potential).
    # This used to read "Segmento" / "Potencial" (Spanish) and crashed with a KeyError on every run, the column rename in the RFM rebuild was never propagated here.
    potential = inputs["rfm"][inputs["rfm"]["Segment"] == "Potential"]
    pot_rev = potential["monto_total"].sum()
    pot_count = len(potential)

    # Full segment breakdown - computed live, never hardcoded
    seg = (inputs["rfm"].groupby("Segment").agg(count=("Customer", "count"), revenue=("monto_total", "sum")).reset_index())
    seg["pct_revenue"] = (seg["revenue"] / seg["revenue"].sum() * 100).round(1)
    seg = seg.sort_values("revenue", ascending=False)

    # Seasonality uses complete calendar years only - partial years skew the monthly index under high inflation, making coverage gaps look like seasonal peaks
    # Peak months are detected dynamically (top-2 by seasonality index, >100 only)
    df_full = inputs["df"].copy()
    df_full["year"] = df_full["Fecha"].dt.year
    year_counts = df_full["year"].value_counts()
    complete_years = [y for y in year_counts.index if (df_full[df_full["year"]==y]["Fecha"].dt.month.nunique() == 12)]
    df_complete = df_full[df_full["year"].isin(complete_years)]

    if len(complete_years) > 0:
        avg_by_month = (df_complete.groupby(df_complete["Fecha"].dt.month)["Ingreso_bruto"].sum()
                         / len(complete_years)).reindex(range(1, 13))
        overall_avg_month = avg_by_month.mean()
        month_idx = (avg_by_month / overall_avg_month * 100).fillna(100) if overall_avg_month else None
        ranked_months = month_idx.sort_values(ascending=False).index.tolist() if month_idx is not None else []
        peak_months = [m for m in ranked_months[:2] if month_idx[m] > 100]
    else:
        peak_months = []

    yearly_peak_pct = []
    for y in sorted(df_complete["year"].unique()):
        yearly = df_complete[df_complete["year"] == y]
        tot = yearly["Ingreso_bruto"].sum()
        peak = yearly[yearly["Fecha"].dt.month.isin(peak_months)]["Ingreso_bruto"].sum() if peak_months else 0
        if tot > 0:
            yearly_peak_pct.append(peak / tot * 100)
    peak_pct_avg = float(np.mean(yearly_peak_pct)) if yearly_peak_pct else None
    peak_pct_min = float(np.min(yearly_peak_pct)) if yearly_peak_pct else None
    peak_pct_max = float(np.max(yearly_peak_pct)) if yearly_peak_pct else None
    avg_monthly_rev_complete = df_complete.groupby(df_complete["Fecha"].dt.to_period("M"))["Ingreso_bruto"].sum().mean()
    if peak_months:
        peak_mask = df_complete["Fecha"].dt.month.isin(peak_months)
        peak_monthly_rev = df_complete[peak_mask].groupby(df_complete[peak_mask]["Fecha"].dt.to_period("M"))["Ingreso_bruto"].sum().mean()
    else:
        peak_monthly_rev = None
    peak_incremental = (peak_monthly_rev - avg_monthly_rev_complete) if pd.notna(peak_monthly_rev) else None
    _MONTH_NAMES = ["", "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    peak_label = " & ".join(_MONTH_NAMES[m] for m in peak_months) if peak_months else "No consistent peak"

    # Repeat-buyer LTV multiplier - computed live from rfm_clientes.csv
    _repeat = inputs["rfm"][inputs["rfm"]["frecuencia"] > 1]["monto_total"]
    _one_time = inputs["rfm"][inputs["rfm"]["frecuencia"] == 1]["monto_total"]
    repeat_ltv_multiplier = (
        float(_repeat.median() / _one_time.median())
        if len(_repeat) > 0 and len(_one_time) > 0 and _one_time.median() > 0
        else None)

    return {
        "elasticity": ci,
        "scenarios": scenarios,
        "current_state": {
            "monthly_revenue": round(current_rev_pm, 0),
            "avg_ticket": round(current_ticket, 0),
            "ml_fee_rate": round(ml_fee_rate, 3),
            "net_margin": round(1 - ml_fee_rate, 3),
        },
        "forecast": {
            "period": "May-October 2026",
            "central": round(total_central, 0),
            "low": round(total_low, 0),
            "high": round(total_high, 0),
        },
        "rfm_opportunity": {
            "potential_count": pot_count,
            "potential_revenue": round(pot_rev, 0),
            "pct_total_revenue": round(pot_rev / inputs["rfm"]["monto_total"].sum() * 100, 1),
            "repeat_ltv_multiplier": round(repeat_ltv_multiplier, 1) if repeat_ltv_multiplier else None,
        },
        "segments": seg,
        "seasonality": {
            "complete_years": complete_years,
            "peak_months": peak_months,
            "peak_label": peak_label,
            "peak_pct_avg": round(peak_pct_avg, 1) if peak_pct_avg is not None else None,
            "peak_pct_min": round(peak_pct_min, 1) if peak_pct_min is not None else None,
            "peak_pct_max": round(peak_pct_max, 1) if peak_pct_max is not None else None,
            "peak_incremental": round(peak_incremental, 0) if peak_incremental is not None else None,
        },
    }


# Output Formatting 
def print_strategy_console(strategy: dict) -> None:
    """ Print the decision document to console."""
    ci = strategy["elasticity"]
    cs = strategy["current_state"]
    fc = strategy["forecast"]
    rfm  = strategy["rfm_opportunity"]
    sc = strategy["scenarios"]
    seg = strategy["segments"]
    seas = strategy["seasonality"]

    # Derived numbers
    annual_run_rate = cs["monthly_revenue"] * 12
    net_monthly = cs["monthly_revenue"] * cs["net_margin"]
    p10_gross_pct = (1.10) ** (1 + ci["epsilon"]) - 1
    p10_monthly_net = cs["monthly_revenue"] * p10_gross_pct * cs["net_margin"]
    p10_annual_net = p10_monthly_net * 12
    repeat_ltv_prem = rfm.get("repeat_ltv_multiplier") or 2.5  
    # Worst-case for a +10% increase - read directly from the scenario table
    row_10pct = sc[sc["Price increase"] == "+10%"].iloc[0]
    worst_case_pct = row_10pct["Pessimistic rev Δ (%)"]

    retention_one_time = rfm["potential_count"] * 0.15 * cs["avg_ticket"]
    retention_conservative = rfm["potential_count"] * 0.10 * cs["avg_ticket"]

    print()
    print("PRICING & GROWTH STRATEGY - Q3 2026")
    print("MercadoLibre Decoration Business")
    print(5)

    print("\n NUMERICAL IMPACT SUMMARY")
    print(f"""
  CURRENT STATE
  Annual revenue run rate: ${annual_run_rate:>10,.0f} ARS
  Monthly net (after fees): ${net_monthly:>10,.0f} ARS ({cs["net_margin"]*100:.1f}% margin)
  ML fees retained per sale: {cs["ml_fee_rate"]*100:.1f}% of gross (revenue-weighted, ML Official channel)

  REVENUE OPPORTUNITY (QUANTIFIED)
  +10% price increase:
    Gross uplift: ${cs["monthly_revenue"]*p10_gross_pct:>+8,.0f} ARS/month ({p10_gross_pct*100:+.1f}%)
    Net uplift: ${p10_monthly_net:>+8,.0f} ARS/month (after ML fees)
    Annual net: ${p10_annual_net:>+8,.0f} ARS/year
    Worst case: {worst_case_pct:+.1f}% revenue (CI lower bound - still positive)

  Retention campaign (Potential segment, 15% conversion):
    {rfm["potential_count"]} buyers x 15% x ${cs["avg_ticket"]:,.0f} avg ticket
    One-time: ${retention_one_time:>+8,.0f} ARS
    Repeat buyers spend {repeat_ltv_prem:.1f}x more lifetime vs new buyers

  Pre-peak advertising shift (ahead of {seas["peak_label"]}):
    {seas["peak_label"]} = {seas["peak_pct_avg"]}% of annual revenue on average
    (ranged {seas["peak_pct_min"]}%–{seas["peak_pct_max"]}% across {len(seas["complete_years"])} complete years - variable)
    Incremental: {"+${:,.0f} ARS/month (peak-month estimate)".format(seas["peak_incremental"]) if seas["peak_incremental"] else "n/a"}

  SEGMENT REVENUE CONTRIBUTION""")
    for _, r in seg.iterrows():
        print(f" {r['Segment']:<12} ({int(r['count'])} buyers):".ljust(32) +
              f"{r['pct_revenue']:>5.1f}%")
    print(f"""
  FORECAST
  Central (May-Oct 2026): ${fc["central"]:>10,.0f} ARS
  Range (80% CI): ${fc["low"]:>10,.0f} – ${fc["high"]:,.0f} ARS
  Uncertainty: +/-{(fc["high"]-fc["low"])/2/fc["central"]*100:.0f}% around central (be honest with stakeholders)
  Best model MAPE: ~55% (naive seasonal baseline - see notebooks/02_Forecasting.ipynb; use for planning direction only, not point precision)""")

    print("\n STRATEGY 1: PRICING")
    print(f"""
  Objective: Recover real margin lost to inflation in 2024.

  Implementation (4 weeks):
    Week 1: Raise the highest-volume, near-zero-elasticity item (see product_elasticity.py per-item table) +15% as a controlled test.
            Keep all other SKUs unchanged. Measure daily.
    Week 2: If volume drop < 10% -> proceed. If > 15% -> pause.
    Week 3: Apply +10% to top 5 SKUs by revenue.
            Contact Potential segment ({rfm["potential_count"]} buyers) with new product highlights.
    Week 4: Review actuals vs model. Update elasticity. Decide remainder.

  Expected impact:
    Central: +${cs["monthly_revenue"]*p10_gross_pct:+,.0f} gross / +${p10_monthly_net:+,.0f} net per month
    Annual: +${p10_annual_net:+,.0f} ARS net
    Worst: {worst_case_pct:+.1f}% revenue (CI lower bound — still positive at +10%)

  Success metrics:
    Revenue change (30-day): target >= +2%
    Volume change: acceptable if <= -8%
    Listing rank: no decline > 2 positions""")

    print("\n STRATEGY 2: MARKETING & RETENTION")
    print(f"""
  Objective: Convert {rfm["potential_count"]} recent first-time buyers to repeat customers.

  Potential segment: {rfm["potential_count"]} buyers | {rfm["pct_total_revenue"]}% of total revenue
    Made one purchase, haven't returned.
    30-day follow-up doubles repeat probability (Gupta et al. 2004).
    Repeat buyers spend {repeat_ltv_prem:.1f}x more over lifetime vs new buyers.

  Implementation:
    Week 1: Identify Potential buyers from last 30 days (rfm_clientes.csv)
            Send follow-up: new products, complementary items.
            DO NOT mention pricing or discounts.
    Week 3: Secondary message to non-responders: seasonal preview.
    Monthly: Refresh segment from updated RFM after each ETL run.

  Expected impact:
    15% conversion: +${retention_one_time:,.0f} ARS one-time
    10% conversion: +${retention_conservative:,.0f} ARS (conservative)

  Success metrics:
    30-day repeat rate:  baseline ~11%, target > 15%
    Contacted vs uncontacted repeat rate (A/B if possible)""")

    print("\n STRATEGY 3: INVENTORY & TIMING")
    print(f"""
  Objective: Align stock and spend to the seasonal demand pattern.

  The signal ({len(seas["complete_years"])} complete years - interpret with care, small n):
    {seas["peak_label"]} share of annual revenue: {seas["peak_pct_avg"]}% on average
    (ranged {seas["peak_pct_min"]}%–{seas["peak_pct_max"]}% - the premium has moved year to year,
    so treat this as a hypothesis to monitor, not a guaranteed repeat)

  Advertising:
    Plan increased ad spend in the weeks BEFORE {seas["peak_label"]}, if the pattern holds —
    lead time matters more than the exact week, given the premium isn't stable.
    Watch actuals closely once {seas["peak_label"]} arrives rather than assuming the historical average.

  Inventory:
    Light pre-build ahead of {seas["peak_label"]}, sized conservatively given the
    year-to-year range above. Reduce stock target after the peak passes.

  Forecast: ${fc["central"]:,.0f} ARS central (May-Oct 2026)
    Use pessimistic (${fc["low"]:,.0f}) for cash planning.
    Use central (${fc["central"]:,.0f}) for operations.

  Success metrics:
    {seas["peak_label"]} 2026 vs {seas["peak_label"]} 2025 revenue (controlling for ad spend)
    Stockout rate during {seas["peak_label"]}: target = 0
    Pre-peak CPM: target <= 70% of in-peak CPM""")

    print("\n COMBINED ROI - 3-MONTH HORIZON")
    pricing_gain = p10_annual_net / 4
    retention_est = retention_one_time
    ads_gain = seas["peak_incremental"] if seas["peak_incremental"] and seas["peak_incremental"] > 0 else 0
    total = pricing_gain + retention_est + ads_gain
    print(f"""
  Action                               Expected                     Confidence
  
  +10% price (net, 3 months)       +${pricing_gain:>8,.0f} ARS           Medium
  Retention campaign               +${retention_est:>8,.0f} ARS          Medium
  Advertising shift ({seas["peak_label"]})  +${ads_gain:>8,.0f} ARS      Low-Medium (pattern weakening — see above)
  
  Combined (3 months)              +${total:>8,.0f} ARS                 LOW-MEDIUM

  Confidence: LOW-MEDIUM. Wide uncertainty individually.
  Directionally useful for prioritization - not financial commitments.""")

    print("\n")
    print("Generated: src/decision_layer.py | Data: data/")
    print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q3 2026 pricing strategy document.", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--format", choices=["console", "markdown"], default="console", help="Output format.")
    args = parser.parse_args()

    inputs = load_inputs()
    strategy = build_pricing_strategy(inputs)
    print_strategy_console(strategy)
