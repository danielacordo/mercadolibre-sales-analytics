# Executive Summary
## MercadoLibre Decoration Business - Pricing & Demand Analysis

**Period analyzed:** January 2023 - April 2026 | 353 orders | 312 customers  
**Business:** Handmade foam rubber decorations, MercadoLibre Argentina

---

## What Made This Hard

This isn't a clean textbook dataset. It's three years of real transaction data from a one-person business operating through Argentina's 2024 hyperinflation, 211% annual CPI, a currency devaluation shock in December 2023, and a customer base where 89% of buyers never returned for a second purchase.

The analytical challenges were real: separating genuine price elasticity from inflation-driven co-movement, detecting seasonality from only three complete years with a shrinking premium, and computing a margin figure that required correcting for off-topic sales that inflated the naive estimate by ~9 percentage points.

The findings below are the ones that survived that scrutiny.

---

## Three Findings That Changed the Recommended Actions

**1. Demand is inelastic, and the business has been leaving money on the table**

Price elasticity: ε = −0.62 (bootstrap 95% CI: −0.79 to −0.45).

A 10% price increase reduces demand by 6.2% and increases net revenue by approximately 3.7%. This held consistently across three years including the December 2023 devaluation shock. At the most conservative CI bound (ε = −0.79), a +10% increase still produces positive revenue growth.

In 2024, prices lagged Argentina's CPI by up to 80 index points. The business absorbed that cost silently, buyers weren't price-sensitive enough to justify it.

**2. The seasonal pattern is real, but smaller and less stable than a first pass suggested**

The initial analysis concluded: "August and September run 30% above average, every year without exception."

That was wrong. Re-running the seasonality index on complete calendar years only (to avoid distortion from partial-year inflation weighting) produced a different answer: September and January are the two strongest months on average, with the seasonal premium narrowing from 32% of annual revenue in 2023 to 17% in 2025.

With three complete years on file, this is a real-but-uncertain pattern, worth planning around, not worth betting a fixed budget on.

**3. Net margin required a methodological correction**

The naive per-row average of `Ingreso_neto / Ingreso_bruto` produced 68.3%. The revenue-weighted aggregate, the correct calculation, is 63.1%. The difference matters for pricing decisions: `AVG(ratio) ≠ ratio(SUM, SUM)` when order sizes vary, which they do significantly here.

The corrected margin is also scoped correctly: ML Official channel only, excluding CNX orders where MercadoLibre doesn't report net revenue.

---

## Cost of Doing Nothing

| Inaction | Estimated cost |
|----------|----------------|
| Holding prices flat for 12 months | Real margin continues eroding against CPI, directionally ~$60–80K ARS, not a precise figure |
| Advertising in August assuming a fixed peak | Missing the pre-decision window if the seasonal pattern continues shifting |
| Not contacting recent first-time buyers | 134 buyers x declining repeat probability each week past the 30-day window |

---

## Three Actions, In Priority Order

**Action 1 - Raise prices 10% across the catalog**

Expected: +3.7% revenue/month. At current monthly volume, approximately +$2,000 ARS/month net after ML fees (63.1% margin).  
Risk: Low. CI lower bound still produces positive revenue growth at +10%.  
Validation: raise one high-volume, near-zero-elasticity SKU by 15% for 30 days and measure volume.

**Action 2 - Contact recent first-time buyers this week**

134 customers in the Potential RFM segment: purchased recently, haven't returned. They represent 59% of total revenue. Academic research (Gupta et al. 2004) shows contacting within 30 days doubles repeat purchase probability.  
Risk: Low. A follow-up message costs nothing and requires no discount.  
Run `python src/decision_layer.py` for the live-computed retention impact, the number moves as the dataset refreshes, so it's not hardcoded here.

**Action 3 - Shift advertising earlier in the calendar**

The August-September premium was real in 2023 but has narrowed each year since. Shifting spend to the month ahead of the observed peak is worth testing, but with moderate expectations.  
Risk: Low-Medium. Reversible. Validate by comparing August 2026 vs August 2025 at equivalent spend.

---

## What This Analysis Cannot Tell You

The data covers completed sales only, not browsers who didn't convert, not competitor pricing, not lost sales from stockouts. The elasticity finding is consistent with standard economics for niche/gift products (Bijmolt et al. 2005 benchmark: ε = −0.65), but the causality question isn't fully settled: inflation and pricing moved together for reasons that may be partially unrelated to demand. A controlled price test on one product would confirm it with certainty.

The seasonal finding in particular should be re-verified each year. Three complete years is not enough to call a pattern stable.

---

## Combined Impact Estimate (3-Month Horizon)

Run `python src/decision_layer.py` for current live-computed figures. The table below gives the structure; ARS amounts are printed by the script from real data and update automatically when the dataset refreshes.

| Action | Driver | Confidence |
|--------|--------|------------|
| Price increase (+10%) | ε = −0.62, current monthly revenue | Medium |
| Retention campaign | 134 buyers × 15% conversion × avg ticket | Medium |
| Advertising timing shift | Peak-month incremental above monthly avg | Low-Medium |
| **Total (3 months)** | **See script output** | **Low-Medium overall** |

These are estimates from 353 orders across 3 complete years. Each action should be measured individually.

---

*Full technical analysis: [github.com/danielacordo/mercadolibre-sales-analytics](https://github.com/danielacordo/mercadolibre-sales-analytics)*  
*Live dashboard: [mercadolibre-analytics.onrender.com](https://mercadolibre-sales-analytics.onrender.com)*
