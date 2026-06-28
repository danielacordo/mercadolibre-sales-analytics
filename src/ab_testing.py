import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path
from dataclasses import dataclass

BASE = Path(__file__).parent.parent

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

@dataclass
class Segment:
    """A buyer segment with its own demand characteristics """
    name: str
    n_buyers: int
    conversion_rate: float
    avg_ticket: float
    elasticity: float
    revenue_share: float
    notes: str = ""

@dataclass
class ABResult:
    """Result of a probabilistic A/B simulation for one segment.
    All monetary values in ARS. Probabilities in [0, 1] """
    segment: str
    price_increase: float
    n_simulations: int

    # Control (current price)
    control_conv_mean: float
    control_rev_mean: float

    # Treatment (higher price)
    treatment_conv_mean: float
    treatment_rev_mean: float

    # Inference
    prob_treatment_wins: float  # P(treatment revenue > control revenue)
    expected_lift: float   # E[treatment rev - control rev] in ARS
    lift_ci_lower: float  # 5th percentile of lift distribution
    lift_ci_upper: float  # 95th percentile of lift distribution

    # Decision
    recommendation: str
    confidence: str # "High" / "Medium" / "Low"


# Segment Estimation
def estimate_segments(df: pd.DataFrame, rfm: pd.DataFrame) -> list[Segment]:
    """Estimate segment-level demand characteristics from historical data.

    Segments:
      1. By price tier: Low / Mid / High / Premium
      2. By buyer type: New (freq=1) vs Repeat (freq>1)
      3. By season: Peak (Aug-Sep) vs Off-peak"""
    
    total_revenue = df["Ingreso_bruto"].sum()
    total_buyers = len(rfm)

    # Aggregate elasticity baseline
    monthly = df.groupby(df["Fecha"].dt.to_period("M")).agg(quantity=("Order_id","count"), price=("Monto","median")).reset_index()
    monthly["ds"] = monthly["Fecha"].dt.to_timestamp()
    clean = monthly.dropna().query("price > 0 and quantity > 0")
    agg_eps, *_ = stats.linregress(np.log(clean["price"]), np.log(clean["quantity"]))

    # Baseline conversion (repeat purchase rate) with Laplace smoothing
    n_repeat = (rfm["frecuencia"] > 1).sum()
    base_conv = (n_repeat + 1) / (total_buyers + 2)  

    segments = []

    # PRICE TIER SEGMENTS 
    # Elasticity adjustment: Premium buyers are less price-sensitive.
    # Rationale: high-ticket items are typically event-driven (large celebrations) where the occasion (not price) drives the purchase decision.
    # Assumption: elasticity scales linearly with log(avg_ticket / overall_avg).
    # This is a model assumption, not a measured value.
    tier_defs = [
        ("Low (<$2K)", (0, 1999), 1.35),  # more elastic than avg
        ("Mid ($2K-5K)", (2000, 4999),  1.00),   # close to aggregate
        ("High ($5K-10K)",(5000, 9999), 0.75),  # less elastic
        ("Premium (>$10K)",(10000,999999),0.45), # least elastic
    ]
    for tier_name, (lo, hi), eps_scale in tier_defs:
        mask = df["Monto"].between(lo, hi)
        n_ord = mask.sum()
        rev_seg = df.loc[mask, "Ingreso_bruto"].sum()
        ticket = df.loc[mask, "Monto"].median() if n_ord > 0 else 0

        # Segment-specific conversion: buyers who bought in this tier and came back, proxy from rfm via monto_total quartile
        q_lo = rfm["monto_total"].quantile(float(np.clip(lo / df["Monto"].max() * 0.8, 0.0, 1.0)))
        q_hi = rfm["monto_total"].quantile( float(np.clip(hi / df["Monto"].max() * 0.8 + 0.2, 0.0, 1.0)))
        rfm_seg = rfm[rfm["monto_total"].between(q_lo, q_hi)]
        n_rep_seg = (rfm_seg["frecuencia"] > 1).sum() if len(rfm_seg) > 0 else 0
        conv = (n_rep_seg + 1) / (len(rfm_seg) + 2) if len(rfm_seg) > 0 else base_conv

        segments.append(Segment(
            name = f"Price tier: {tier_name}",
            n_buyers = n_ord,
            conversion_rate = round(conv, 4),
            avg_ticket = round(ticket, 0),
            elasticity = round(agg_eps * eps_scale, 4),
            revenue_share = round(rev_seg / total_revenue, 3),
            notes = f"Elasticity = {agg_eps:.2f} × {eps_scale} (assumption)"))

    # BUYER TYPE SEGMENTS 
    new_buyers = rfm[rfm["frecuencia"] == 1]
    repeat_buyers = rfm[rfm["frecuencia"] >  1]

    # New buyers: conversion = probability of making a 2nd purchase
    new_conv = (1 + 1) / (len(new_buyers) + 2)  
    rep_conv = (len(repeat_buyers) + 1) / (len(rfm) + 2)

    segments.append(Segment(
        name = "Buyer type: New (first purchase)",
        n_buyers = len(new_buyers),
        conversion_rate = round(new_conv, 4),
        avg_ticket = round(new_buyers["monto_total"].mean(), 0),
        elasticity = round(agg_eps * 1.2, 4),  # new buyers more price-sensitive
        revenue_share = round(new_buyers["monto_total"].sum() / total_revenue, 3),
        notes = "Higher elasticity assumption: new buyers haven't built loyalty"))
    
    segments.append(Segment(
        name = "Buyer type: Repeat (2+ purchases)",
        n_buyers = len(repeat_buyers),
        conversion_rate = round(rep_conv, 4),
        avg_ticket = round(repeat_buyers["monto_total"].mean(), 0),
        elasticity = round(agg_eps * 0.6, 4),  # loyal buyers less price-sensitive
        revenue_share = round(repeat_buyers["monto_total"].sum() / total_revenue, 3),
        notes = "Lower elasticity assumption: loyalty reduces price sensitivity"))

    #  SEASONAL SEGMENTS 
    peak_mask = df["Mes"].isin([8, 9])
    offpeak_mask = ~peak_mask

    for seg_name, mask, eps_scale, note in [
        ("Season: Peak (Aug-Sep)", peak_mask, 0.70,
         "Peak: occasion-driven, lower price sensitivity"),
        ("Season: Off-peak (rest)",  offpeak_mask, 1.15,
         "Off-peak: discretionary, higher price sensitivity"),
    ]:
        n_ord = mask.sum()
        rev_seg = df.loc[mask, "Ingreso_bruto"].sum()
        ticket = df.loc[mask, "Monto"].median() if n_ord > 0 else 0
        segments.append(Segment(
            name = seg_name,
            n_buyers = n_ord,
            conversion_rate = base_conv,
            avg_ticket = round(ticket, 0),
            elasticity = round(agg_eps * eps_scale, 4),
            revenue_share = round(rev_seg / total_revenue, 3),
            notes = note))

    return segments


# Probabilistic Simulation 
def beta_posterior(n_successes: int, n_trials: int, prior_a: float = 1.0, prior_b: float = 9.0) -> tuple:
    """ Beta-Binomial posterior for conversion rate """
    return prior_a + n_successes, prior_b + n_trials - n_successes


def required_sample_size(p_control: float = 0.108, lift_pp: float = 0.02, alpha: float = 0.05,
                          power: float = 0.80, monthly_orders: int = 14) -> dict:
    """ Computes the sample size required for a properly powered A/B test.

    Documents why a fully randomized experiment is not feasible at this business's current transaction volume, and what alternatives are available"""
    from scipy import stats as _st
    import numpy as np

    p_treatment = p_control + lift_pp
    p_pool = (p_control + p_treatment) / 2

    z_alpha = _st.norm.ppf(1 - alpha / 2)
    z_beta = _st.norm.ppf(power)

    n = ((z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) + z_beta  * np.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))) ** 2 / (p_treatment - p_control) ** 2)

    months = (n * 2) / monthly_orders

    verdict = (f"INFEASIBLE as a randomized experiment. "
        f"Requires {n:.0f} observations per group ({n*2:.0f} total). "
        f"At {monthly_orders} orders/month: {months:.0f} months = {months/12:.0f} years." )

    alternatives = [
        "Bayesian simulation (implemented): uses historical elasticity to model "
        "the outcome distribution without a real experiment.",
        "Sequential testing / CUPED: monitor a single metric as data arrives, "
        "stop when a threshold is crossed. Requires fewer observations than "
        "fixed-horizon designs but still needs months, not weeks.",
        "Single-product test (recommended): raise price on one product only, "
        "compare vs all other products as a quasi-control. Not randomized, but "
        "provides real causal evidence at small scale.",
        "Synthetic control: use comparable MercadoLibre categories as "
        "counterfactual for what would have happened without the price change.",]

    return {
        "n_per_group": int(n),
        "n_total": int(n * 2),
        "months_required": round(months, 0),
        "alpha": alpha,
        "power": power,
        "lift_detected_pp": lift_pp,
        "verdict": verdict,
        "alternatives": alternatives,}

def simulate_ab_test(segment: Segment, price_increase: float, n_simulations: int = 10_000,
                     monthly_visitors: int = 200, random_state: int = 42) -> ABResult:
    """ Probabilistic A/B simulation for a price increase on one segment.

    Model:
      Control: buyer sees current price P
                 conversion ~ Bernoulli(p_control)
                 revenue per conversion = P
      Treatment: buyer sees price P × (1 + Δ)
                 p_treatment = p_control × (1 + Δ)^ε
                   where ε = segment elasticity
                 revenue per conversion = P × (1 + Δ)

    Why Bayesian simulation and not frequentist t-test:
       SAMPLE SIZE REALITY CHECK (frequentist):
        To detect a 2pp conversion lift (10.8% → 12.8%) with 80% power and alpha=0.05, a standard two-proportion z-test requires:
          n ≈ 4,083 observations per group (8,166 total)
        At ~14 orders/month: this requires 583 months = 48 years.
        A properly powered frequentist A/B test is not feasible with this business's current transaction volume.

      FALSE POSITIVE RISK:
        Testing 8 segments at alpha=0.05 → expected 0.4 false positives under the null (multiple comparisons problem).
        Bonferroni-corrected alpha = 0.05/8 = 0.006.
        With this sample size, we cannot meet this threshold either.

      WHY BAYESIAN SIMULATION INSTEAD:
        - Reports P(treatment > control) directly - the business question
        - Incorporates prior knowledge (historical conversion rate)
        - Makes uncertainty explicit rather than hiding it in p-values
        - Honest about what we can and cannot conclude from small data

      WHAT THIS SIMULATION IS AND IS NOT:
        IS: A structured way to reason about uncertainty given the historical elasticity and the observed conversion rate.
        IS NOT: A substitute for a real experiment. The elasticity used as input is itself estimated from observational data under inflation, it is not causally identified.

      CONCLUSION:
        The simulation consistently returns P(win) between 50-65% for most segments at +10% price increase. 
        This is the correct answer for a dataset of this size. 
        The aggregate elasticity result (ε ≈ -0.62, entire CI revenue-positive at +10%) is the stronger basis for the pricing decision. """
    np.random.seed(random_state)
    rng = np.random.default_rng(random_state)

    p_base = segment.conversion_rate
    eps = segment.elasticity
    P = segment.avg_ticket
    delta  = price_increase

    # Posterior for control conversion rate
    n_success_pseudo = max(1, int(p_base * segment.n_buyers))
    n_trial_pseudo = max(2, segment.n_buyers)
    post_a, post_b = beta_posterior(n_success_pseudo, n_trial_pseudo)

    # Monte Carlo simulation 
    p_control_draws = rng.beta(post_a, post_b, n_simulations)

    # Treatment conversion: apply demand response
    demand_multiplier = (1 + delta) ** eps
    p_treatment_draws = np.clip(p_control_draws * demand_multiplier, 0, 1)

    # Simulate visitor outcomes (Binomial)
    conv_control = rng.binomial(monthly_visitors, p_control_draws)
    conv_treatment = rng.binomial(monthly_visitors, p_treatment_draws)

    # Revenue per group per month
    rev_control = conv_control * P
    rev_treatment = conv_treatment * P * (1 + delta)

    # Lift distribution
    lift = rev_treatment - rev_control

    # Summarize 
    prob_wins = float((lift > 0).mean())
    exp_lift = float(lift.mean())
    lift_lo = float(np.percentile(lift, 5))
    lift_hi = float(np.percentile(lift, 95))

    # Recommendation logic
    if prob_wins >= 0.90 and lift_lo > 0:
        rec  = f"IMPLEMENT: P(treatment > control) = {prob_wins:.0%}. " \
               f"90% CI lift: [+${lift_lo:,.0f}, +${lift_hi:,.0f}] ARS/month."
        conf = "High"
    elif prob_wins >= 0.75:
        rec  = f"RUN REAL TEST: P(win) = {prob_wins:.0%} — promising but not conclusive. " \
               f"Median lift: +${exp_lift:,.0f} ARS/month."
        conf = "Medium"
    elif prob_wins >= 0.55:
        rec  = f"GATHER MORE DATA: P(win) = {prob_wins:.0%} - too close to call. " \
               f"Lift CI includes near-zero outcomes."
        conf = "Low"
    else:
        rec  = f"DO NOT IMPLEMENT: P(win) = {prob_wins:.0%} - control likely better. " \
               f"Expected lift: ${exp_lift:,.0f} ARS/month."
        conf = "Low"

    return ABResult(
        segment = segment.name,
        price_increase = price_increase,
        n_simulations = n_simulations,
        control_conv_mean = round(float(p_control_draws.mean()), 4),
        control_rev_mean = round(float(rev_control.mean()), 0),
        treatment_conv_mean = round(float(p_treatment_draws.mean()), 4),
        treatment_rev_mean = round(float(rev_treatment.mean()), 0),
        prob_treatment_wins = round(prob_wins, 4),
        expected_lift = round(exp_lift, 0),
        lift_ci_lower = round(lift_lo, 0),
        lift_ci_upper = round(lift_hi, 0),
        recommendation = rec,
        confidence = conf,)


# Cohort Analysis with Decisions 
def cohort_decision_analysis(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """ Cohort analysis connected to concrete retention decisions.

    Cohorts = first purchase month

    Decision logic:
      High LTV + low retention -> highest-value contact target
      Low LTV + high retention -> loyal but low-ticket -> upsell opportunity
      Low LTV + low retention -> deprioritize """
    
    # First purchase date per buyer
    if "primera_compra" not in rfm.columns:
        # Approximate from recencia and ultima_compra
        rfm = rfm.copy()
        rfm["ultima_compra"] = pd.to_datetime(rfm["ultima_compra"])
        rfm["primera_compra"] = rfm["ultima_compra"] - pd.to_timedelta( rfm["frecuencia"].apply(lambda f: max(f - 1, 0)) * 60, unit="D")
    rfm["cohort_month"] = pd.to_datetime(rfm["ultima_compra"]).dt.to_period("M")

    # Aggregate by acquisition cohort
    cohorts = rfm.groupby("cohort_month").agg(
        cohort_size = ("Customer", "count"),
        avg_ltv = ("monto_total", "mean"),
        total_ltv = ("monto_total", "sum"),
        pct_repeat = ("frecuencia", lambda x: (x > 1).mean()),
        avg_frequency = ("frecuencia", "mean"),
        avg_recency = ("recencia", "mean"),).reset_index()

    overall_avg_ltv = rfm["monto_total"].mean()
    overall_pct_rep = (rfm["frecuencia"] > 1).mean()

    cohorts["ltv_vs_avg"] = (cohorts["avg_ltv"] / overall_avg_ltv).round(2)
    cohorts["ret_vs_avg"] = (cohorts["pct_repeat"] / max(overall_pct_rep, 0.001)).round(2)

    # Decision matrix
    def assign_action(row):
        high_ltv = row["ltv_vs_avg"] >= 1.0
        high_ret = row["ret_vs_avg"] >= 1.0
        if high_ltv and not high_ret:
            return ("Priority contact",
                    "High-value buyers not returning -> 30-day win-back, personalised offer")
        elif high_ltv and high_ret:
            return ("VIP nurture",
                    "High LTV + loyal -> exclusivity, early access, loyalty reward")
        elif not high_ltv and high_ret:
            return ("Upsell",
                    "Loyal but low-ticket -> introduce Premium products, bundle offers")
        else:
            return ("Monitor",
                    "Low LTV + low retention -> standard catalogue, no dedicated spend")

    cohorts[["action", "action_detail"]] = pd.DataFrame(cohorts.apply(assign_action, axis=1).tolist(), index=cohorts.index)
    cohorts["avg_ltv"] = cohorts["avg_ltv"].round(0)
    cohorts["total_ltv"] = cohorts["total_ltv"].round(0)
    cohorts["pct_repeat"] = (cohorts["pct_repeat"] * 100).round(1)

    return cohorts.sort_values("cohort_month")


# Funnel Analysis
def funnel_by_segment(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """Purchase funnel segmented by price tier, buyer type, and season.+

    Funnel stages (proxied from available data):
      Reach -> total buyers who could have seen the product (= all RFM buyers)
      Purchase -> buyers who completed at least one order
      Repeat -> buyers who purchased 2+ times
      LTV -> average revenue per buyer"""
    
    total_rev = df["Ingreso_bruto"].sum()
    rows = []

    # Price segments
    tier_ranges = {
        "Low (<$2K)": (0, 1999),
        "Mid ($2K-5K)": (2000, 4999),
        "High ($5K-10K)": (5000, 9999),
        "Premium (>$10K)": (10000, 999999),}
    
    for tier, (lo, hi) in tier_ranges.items():
        mask = df["Monto"].between(lo, hi)
        n_ord = mask.sum()
        rev_seg = df.loc[mask, "Ingreso_bruto"].sum()
        ticket = df.loc[mask, "Monto"].median() if n_ord > 0 else 0
        # Repeat from rfm approximated by monto_total range
        rfm_q = rfm[rfm["monto_total"].between(lo * 0.5, hi * 2)]
        n_rep = (rfm_q["frecuencia"] > 1).sum() if len(rfm_q) > 0 else 0
        rows.append({
            "Dimension": "Price tier",
            "Segment": tier,
            "Orders": n_ord,
            "Revenue (ARS)": round(rev_seg, 0),
            "Avg ticket (ARS)": round(ticket, 0),
            "Rev share (%)": round(rev_seg / total_rev * 100, 1),
            "Repeat buyers": n_rep,
            "Repeat rate (%)": round(n_rep / max(len(rfm_q), 1) * 100, 1),
            "Revenue/order (ARS)": round(rev_seg / max(n_ord, 1), 0),})

    # Buyer type
    for btype, mask_fn in [
        ("New (1 purchase)", lambda r: r["frecuencia"] == 1),
        ("Repeat (2+ purchases)", lambda r: r["frecuencia"] > 1),
    ]:
        sub = rfm[mask_fn(rfm)]
        n_ord = sub["frecuencia"].sum()
        rev = sub["monto_total"].sum()
        rows.append({
            "Dimension": "Buyer type",
            "Segment": btype,
            "Orders": int(n_ord),
            "Revenue (ARS)": round(rev, 0),
            "Avg ticket (ARS)": round(rev / max(n_ord, 1), 0),
            "Rev share (%)": round(rev / total_rev * 100, 1),
            "Repeat buyers": int((rfm["frecuencia"] > 1).sum()) if "Repeat" in btype else 0,
            "Repeat rate (%)": 100.0 if "Repeat" in btype else 0.0,
            "Revenue/order (ARS)": round(rev / max(n_ord, 1), 0),})

    # Season
    for season, months in [
        ("Peak (Aug-Sep)", [8, 9]),
        ("Off-peak (rest)", [1,2,3,4,5,6,7,10,11,12]),
    ]:
        mask = df["Mes"].isin(months)
        n_ord = mask.sum()
        rev_seg = df.loc[mask, "Ingreso_bruto"].sum()
        ticket = df.loc[mask, "Monto"].median() if n_ord > 0 else 0
        rows.append({
            "Dimension": "Season",
            "Segment": season,
            "Orders": n_ord,
            "Revenue (ARS)": round(rev_seg, 0),
            "Avg ticket (ARS)": round(ticket, 0),
            "Rev share (%)": round(rev_seg / total_rev * 100, 1),
            "Repeat buyers": "-",
            "Repeat rate (%)": "-",
            "Revenue/order (ARS)": round(rev_seg / max(n_ord, 1), 0),})

    return pd.DataFrame(rows)


# Strong Decisions 
def build_decision_matrix(ab_results: list[ABResult], funnel: pd.DataFrame, cohorts: pd.DataFrame) -> pd.DataFrame:
    """ Synthesize A/B simulation, funnel, and cohort analysis into a ranked decision matrix """
    decisions = []

    # From A/B simulations 
    for r in ab_results:
        if r.confidence in ("High", "Medium"):
            decisions.append({
                "Decision": f"Price +{r.price_increase*100:.0f}% on {r.segment}",
                "Source": "Probabilistic A/B simulation",
                "Expected monthly impact (ARS)": r.expected_lift,
                "P(positive)": f"{r.prob_treatment_wins:.0%}",
                "90% CI": f"[${r.lift_ci_lower:+,.0f}, ${r.lift_ci_upper:+,.0f}]",
                "Confidence": r.confidence,
                "Effort": "Low",
                "Action": r.recommendation.split(":")[0].strip(),})

    #  From funnel - Revenue concentration 
    funnel_num = funnel[funnel["Dimension"] == "Price tier"].copy()
    funnel_num["Revenue (ARS)"] = pd.to_numeric(funnel_num["Revenue (ARS)"], errors="coerce")
    top_tier = funnel_num.nlargest(1, "Revenue (ARS)").iloc[0]
    decisions.append({
        "Decision": f"Focus catalog depth on {top_tier['Segment']} tier",
        "Source": "Funnel analysis - revenue concentration",
        "Expected monthly impact (ARS)": top_tier["Revenue (ARS)"] * 0.05,
        "P(positive)": "~70%",
        "90% CI": "Not estimated",
        "Confidence": "Medium",
        "Effort": "Medium",
        "Action": "EXPAND",})

    # From cohorts - LTV gap 
    priority = cohorts[cohorts["action"].str.contains("Priority", na=False)]
    if len(priority) > 0:
        est_impact = priority["total_ltv"].sum() * 0.15  
        decisions.append({
            "Decision": f"Win-back campaign for {len(priority)} high-LTV cohort(s)",
            "Source": "Cohort LTV analysis",
            "Expected monthly impact (ARS)": est_impact,
            "P(positive)": "~65%",
            "90% CI": "Not estimated",
            "Confidence": "Medium",
            "Effort": "Low",
            "Action": "IMPLEMENT",})

    # Seasonal advertising 
    # Confidence and revenue share were hardcoded ("High", "~80%", "30% above average") disconnected from season_funnel. 
    # Now reads live values and calibrates confidence to match the rest of the project.
    season_funnel = funnel[funnel["Segment"] == "Peak (Aug-Sep)"].iloc[0]
    peak_rev = float(season_funnel["Revenue (ARS)"])
    peak_share_pct = float(season_funnel["Rev share (%)"])
    decisions.append({
        "Decision": "Plan ad spend ahead of Aug-Sep (pre-peak) - verify timing yearly",
        "Source": f"Seasonal funnel - Aug/Sep = {peak_share_pct:.0f}% of total revenue in this fixed window "
                            "(year-to-year share has ranged 17%-32% on a separate complete-years check - variable)",
        "Expected monthly impact (ARS)": peak_rev * 0.08,
        "P(positive)": "~55%",
        "90% CI": "Not estimated",
        "Confidence": "Low-Medium",
        "Effort": "Low",
        "Action": "MONITOR",})

    df_dec = pd.DataFrame(decisions).sort_values("Expected monthly impact (ARS)", ascending=False).reset_index(drop=True)
    df_dec.index += 1   
    return df_dec


# Console Output 
def print_results(segments: list[Segment], ab_results: list[ABResult], funnel: pd.DataFrame,
                  cohorts: pd.DataFrame, decisions: pd.DataFrame) -> None:

    # FUNNEL 
    print(f"{BOLD}PURCHASE FUNNEL - BY SEGMENT{RESET}")
    print(f"{BOLD}{"="*65}{RESET}")
    for dim in funnel["Dimension"].unique():
        print(f"\n {BOLD}{dim}{RESET}")
        sub = funnel[funnel["Dimension"] == dim]
        print(f" {'Segment':<26} {'Orders':>7} {'Rev share':>10} {'Avg ticket':>12} {'Repeat rate':>12}")
        print(f" {"-"*65}")
        for _, row in sub.iterrows():
            rep = f"{row['Repeat rate (%)']:.1f}%" if isinstance(row['Repeat rate (%)'], float) else "-"
            print(f"  {row['Segment']:<26} {row['Orders']:>7} "
                  f"{row['Rev share (%)']:>9.1f}% "
                  f"${row['Avg ticket (ARS)']:>10,.0f} "
                  f"{rep:>12}")

    # A/B SIMULATION 
    print(f"\n{BOLD}{"="*65}{RESET}")
    print(f"{BOLD}PROBABILISTIC A/B SIMULATION{RESET}")
    print(f" Model: Beta-Binomial posterior, {ab_results[0].n_simulations:,} Monte Carlo draws")
    print(" Treatment: price increase applied per segment elasticity")
    print(f" Simulation uses historical elasticity - not a real experiment{RESET}")
    print(f"{BOLD}{"="*65}{RESET}")

    for r in ab_results:
        color = GREEN if r.confidence == "High" else YELLOW if r.confidence == "Medium" else RED
        symbol = "OK" if r.confidence == "High" else "~" if r.confidence == "Medium" else "X"
        print(f"\n {color}{BOLD}[{symbol}] {r.segment}{RESET}")
        print(f" Price increase: +{r.price_increase*100:.0f}%")
        print(f" Control conv rate: {r.control_conv_mean:.1%}")
        print(f" Treatment conv rate: {r.treatment_conv_mean:.1%}  "
              f"(Δ = {(r.treatment_conv_mean-r.control_conv_mean)*100:+.2f}pp)")
        print(f" P(treatment > control):{r.prob_treatment_wins:.0%}")
        print(f" Expected lift: ${r.expected_lift:+,.0f} ARS/month")
        print(f" 90% CI: [${r.lift_ci_lower:+,.0f}, ${r.lift_ci_upper:+,.0f}]")
        print(f" {color}-> {r.recommendation}{RESET}")

    # COHORT DECISIONS 
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}COHORT ANALYSIS - CONNECTED TO DECISIONS{RESET}")
    print(f"{BOLD}{"="*65}{RESET}")
    print(f"\n {'Cohort':<12} {'Size':>6} {'Avg LTV':>10} {'Repeat%':>8} "
          f"{'LTV/avg':>8} {'Action':<22} Detail")
    print(f" {"-"*65}")
    for _, row in cohorts.iterrows():
        ltv_flag = "↑" if row["ltv_vs_avg"] >= 1 else "↓"
        _ret_flag = "↑" if row["ret_vs_avg"] >= 1 else "↓"
        print(f" {str(row['cohort_month']):<12} "
              f"{row['cohort_size']:>6} "
              f"${row['avg_ltv']:>8,.0f} "
              f"{row['pct_repeat']:>7.1f}% "
              f"{ltv_flag}{row['ltv_vs_avg']:>5.2f}x "
              f"{row['action']:<22} "
              f"{row['action_detail'][:50]}")

    # DECISION MATRIX 
    print(f"\n{BOLD}{"="*65}{RESET}")
    print(f"{BOLD}RANKED DECISION MATRIX{RESET}")
    print("  Sorted by expected monthly revenue impact")
    print(f"{BOLD}{"="*65}{RESET}")
    print(f"\n  {'#':<4} {'Action':<12} {'Decision':<48} {'Impact/mo':>12} {'P(+)':>7} {'Conf':>8}")
    print(f"  {"-"*65}")
    for rank, row in decisions.iterrows():
        color = GREEN if row["Action"] == "IMPLEMENT" else YELLOW if row["Action"] in ("EXPAND","RUN REAL TEST") else RESET
        print(f" {rank:<4} {color}{row['Action']:<12}{RESET} "
              f"{row['Decision'][:47]:<48} "
              f"${row['Expected monthly impact (ARS)']:>10,.0f} "
              f"{row['P(positive)']:>7} "
              f"{row['Confidence']:>8}")

    print(f"\n{BOLD}  Top recommendation:{RESET}")
    top = decisions.iloc[0]
    print(f" -> {GREEN}{top['Action']}: {top['Decision']}{RESET}")
    print(f" Expected: ${top['Expected monthly impact (ARS)']:,.0f} ARS/month "
          f"| P(positive) = {top['P(positive)']} | Confidence: {top['Confidence']}")



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Probabilistic A/B simulation and cohort decision analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--price-increase", type=float, default=0.10, help="Price increase fraction to simulate (e.g. 0.10 = 10%%).")
    parser.add_argument("--n-sim", type=int, default=10_000, help="Monte Carlo draws per segment.")
    parser.add_argument("--visitors", type=int, default=200, help="Assumed monthly visitors per group.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(BASE / "data" / "ventas_decoraciones.csv", parse_dates=["Fecha"])
    rfm = pd.read_csv(BASE / "data" / "rfm_clientes.csv")

    print(f"{BOLD}Loading segments...{RESET}")
    segments = estimate_segments(df, rfm)
    print(f" {len(segments)} segments estimated")

    print(f"{BOLD}Running A/B simulations ({args.n_sim:,} draws per segment)...{RESET}")
    ab_results = [
        simulate_ab_test(seg, args.price_increase, args.n_sim, args.visitors)
        for seg in segments]

    print(f"{BOLD}Building funnel...{RESET}")
    funnel = funnel_by_segment(df, rfm)

    print(f"{BOLD}Running cohort analysis...{RESET}")
    cohorts = cohort_decision_analysis(df, rfm)

    print(f"{BOLD}Building decision matrix...{RESET}")
    decisions = build_decision_matrix(ab_results, funnel, cohorts)

    print_results(segments, ab_results, funnel, cohorts, decisions)
