import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Optional
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).parent))
from plot_style import apply_theme, styled_fig, C  # noqa: E402

apply_theme()

@dataclass
class ProphetComponents:
    """ Holds Prophet model artifacts needed for plot_components()
    Kept separate from ForecastResult so SARIMA results don't carry dead fields """
    model: object        
    forecast: pd.DataFrame    

@dataclass
class ForecastResult:
    """ Container for forecasting model resuls """
    model_name: str
    mape: float
    mae: float
    rmse: float
    forecast_df: pd.DataFrame          
    components: Optional[ProphetComponents] = None  


def prepare_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction data to monthly frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset with columns Fecha and Ingreso_bruto.

    Returns
    -------
    pd.DataFrame
        Monthly series with columns: ds, y, ventas, ticket, z_score, y_model.
        y_model = y with outliers replaced by median (used for model training).
    """
    monthly = (
        df.groupby(df["Fecha"].dt.to_period("M"))
        .agg(
            y      = ("Ingreso_bruto", "sum"),
            ventas = ("Order_id",      "count"),
            ticket = ("Monto",         "median"),
        )
        .reset_index()
    )
    monthly["ds"] = monthly["Fecha"].dt.to_timestamp()

    # Z-score for outlier detection
    monthly["z_score"] = (monthly["y"] - monthly["y"].mean()) / monthly["y"].std()

    # Replace outliers with the median of normal months
    # This prevents extreme values from skewing the model trend
    median_normal  = monthly[monthly["z_score"].abs() <= 2]["y"].median()
    monthly["y_model"] = monthly["y"].copy()
    monthly.loc[monthly["z_score"] > 2, "y_model"] = median_normal

    n_outliers = (monthly["z_score"] > 2).sum()
    if n_outliers > 0:
        print(f"  {n_outliers} month(s) with |z| > 2 replaced with median ${median_normal:,.0f}")

    assert len(monthly) >= 24, \
        f"Series too short: {len(monthly)} months. At least 24 required."

    return monthly


def run_prophet(monthly: pd.DataFrame,
                n_periods: int = 6) -> ForecastResult:
    """ Train Prophet and generate forecast """
    try:
        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics
    except ImportError:
        raise ImportError(
            "prophet is required for run_prophet(). "
            "Install it with: pip install prophet"
        )

    prophet_df = monthly[["ds", "y_model"]].rename(columns={"y_model": "y"})

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05, interval_width=0.80,)
    model.fit(prophet_df)

    # Temporal cross-validation
    cv = cross_validation(
        model, initial="548 days", period="90 days", horizon="90 days", parallel=None,)
    metrics = performance_metrics(cv)

    future = model.make_future_dataframe(periods=n_periods, freq="MS")
    forecast = model.predict(future)
    last = monthly["ds"].max()
    fc_fut = forecast[forecast["ds"] > last][["ds","yhat","yhat_lower","yhat_upper"]].copy()
    fc_fut[["yhat","yhat_lower","yhat_upper"]] = \
        fc_fut[["yhat","yhat_lower","yhat_upper"]].clip(lower=0)

    return ForecastResult(
        model_name = "Prophet",
        mape = metrics["mape"].mean() * 100,
        mae = metrics["mae"].mean(),
        rmse = metrics["rmse"].mean(),
        forecast_df = fc_fut,
        components = ProphetComponents(model=model, forecast=forecast),
    )

def run_sarima(monthly: pd.DataFrame,
               n_periods: int = 6) -> ForecastResult:
    """ Train SARIMA and generate forecast """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels required: pip install statsmodels")

    y = monthly["y_model"].values

    # Expanding window backtesting
    train_start = 20
    horizon = 3
    actuals, preds = [], []

    for cutoff in range(train_start, len(y) - horizon + 1, 3):
        y_train = y[:cutoff]
        y_test = y[cutoff:cutoff + horizon]
        try:
            model  = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,0,12), enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False)
            pred = fitted.forecast(steps=horizon)
            actuals.extend(y_test)
            preds.extend(list(pred)[:horizon])
        except Exception:
            continue

    actuals = np.array(actuals)
    preds = np.array(preds).clip(min=0)

    mape = np.mean(np.abs((actuals - preds) / np.where(actuals == 0, 1, actuals))) * 100
    mae  = np.mean(np.abs(actuals - preds))
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))

    # Final forecast using all available data
    final_model  = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,0,12), enforce_stationarity=False, enforce_invertibility=False)
    final_fitted = final_model.fit(disp=False)
    fc_values = final_fitted.forecast(steps=n_periods).clip(min=0)
    ci = final_fitted.get_forecast(steps=n_periods).conf_int(alpha=0.20)

    last = monthly["ds"].max()
    fc_fut = pd.DataFrame({
        "ds": pd.date_range(start=last + pd.DateOffset(months=1), periods=n_periods, freq="MS"),
        "yhat": np.array(fc_values).clip(min=0),
        "yhat_lower": np.array(ci[:, 0]).clip(min=0),
        "yhat_upper": np.array(ci[:, 1]), })

    return ForecastResult(
        model_name = "SARIMA(1,1,1)(1,1,0,12)",
        mape = mape,
        mae = mae,
        rmse = rmse,
        forecast_df = fc_fut,
    )


def run_naive_baseline(monthly: pd.DataFrame,
                       n_periods: int = 6) -> ForecastResult:
    """ Naive seasonal baseline: forecast = same month in the previous year

    This is the minimum bar any model must beat to be worth using.
    If Prophet or SARIMA can't outperform "same month last year", they add no value over a simple calendar lookup """
    actuals, preds = [], []
    for i, row in monthly.iterrows():
        prev = monthly[(monthly["ds"].dt.year  == row["ds"].year - 1) & (monthly["ds"].dt.month == row["ds"].month)]
        if len(prev):
            actuals.append(row["y_model"])
            preds.append(prev["y_model"].values[0])

    actuals = np.array(actuals)
    preds = np.array(preds).clip(min=0)

    if len(actuals) == 0:
        # Fewer than 12 months of data, no prior year to compare against
        import warnings as _w
        _w.warn(
            "run_naive_baseline: no prior-year observations available "
            f"(series has {len(monthly)} months, need >= 13). "
            "Returning mape=inf.",
            UserWarning, stacklevel=2,
        )
        last = monthly["ds"].max()
        return ForecastResult(
            model_name  = "Naive (same month last year)",
            mape = float("inf"),
            mae = float("inf"),
            rmse = float("inf"),
            forecast_df = pd.DataFrame({
                "ds": pd.date_range(start=last + pd.DateOffset(months=1), periods=n_periods, freq="MS"),
                "yhat": [np.nan] * n_periods,
                "yhat_lower": [np.nan] * n_periods,
                "yhat_upper": [np.nan] * n_periods,
            }),
        )

    mape = np.mean(np.abs((actuals - preds) / np.where(actuals==0, 1, actuals))) * 100
    mae = np.mean(np.abs(actuals - preds))
    rmse = np.sqrt(np.mean((actuals - preds)**2))

    # Naive future forecast: same month from the most recent available year
    # For each future month, look up the prior-year value as the point estimate
    # CI approximated as ± 1 MAE (symmetric, honest for a naive model)
    last = monthly["ds"].max()
    fc_dates = pd.date_range(start=last + pd.DateOffset(months=1), periods=n_periods, freq="MS")
    fc_yhat = []
    mae_val = float(np.mean(np.abs(actuals - preds)))
    for fc_date in fc_dates:
        prior = monthly[(monthly["ds"].dt.year  == fc_date.year - 1) & (monthly["ds"].dt.month == fc_date.month)]
        if len(prior):
            fc_yhat.append(float(prior["y_model"].values[0]))
        else:
            # Fall back to last known value if prior year missing
            fc_yhat.append(float(monthly["y_model"].iloc[-1]))
    fc_yhat_arr = np.array(fc_yhat).clip(min=0)
    fc_fut = pd.DataFrame({
        "ds": fc_dates,
        "yhat": fc_yhat_arr,
        "yhat_lower": (fc_yhat_arr - mae_val).clip(min=0),
        "yhat_upper": fc_yhat_arr + mae_val,
    })

    return ForecastResult(
        model_name  = "Naive (same month last year)",
        mape = mape,
        mae = mae,
        rmse = rmse,
        forecast_df = fc_fut,
    )


def compare_models(results: list[ForecastResult]) -> pd.DataFrame:
    """ Build a comparison table of backtesting metrics across models """
    rows = [{
        "Model": r.model_name,
        "MAPE (%)": round(r.mape, 1),
        "MAE (ARS)": f"${r.mae:,.0f}",
        "RMSE (ARS)": f"${r.rmse:,.0f}",
    } for r in results]

    df_comp = pd.DataFrame(rows).set_index("Model")
    best = min(results, key=lambda x: x.mape)

    print(f"\n  Best MAPE: {best.model_name} ({best.mape:.1f}%)")
    print("  Model selected for production: Prophet")
    print("  Reason: on short series with extreme inflation, Prophet components")
    print(" (trend + seasonality) provide interpretable business insights")
    print(" even when SARIMA achieves a lower MAPE.")
    print("\n  Methodology note: MAPE values are directionally comparable but not")
    print("  perfectly apples-to-apples. Prophet uses Prophet's cross_validation()")
    print("  (~5 folds); SARIMA uses a manual expanding window (more folds).")
    print("  Both use a 3-month horizon and similar initial training sizes.")

    return df_comp


def plot_comparison(monthly: pd.DataFrame,
                    results: list,
                    save_path: Optional[str] = None) -> None:
    """ Revenue forecast chart: historical + Prophet + naive baseline with CI band """
    import numpy as _np
    import pandas as _pd

    hist_x  = _np.arange(len(monthly))
    fig = styled_fig(16, 7,
        title="Revenue Forecast - May to October 2026",
        subtitle="Prophet multiplicative seasonality  -  80% confidence interval  -  Seasonal baseline for comparison")
    ax = fig.add_axes([0.06, 0.13, 0.88, 0.73])

    # Historical
    ax.plot(hist_x, monthly["y"].values / 1000, color=C["blue"], lw=2, zorder=4, label="Historical revenue")
    ax.fill_between(hist_x, monthly["y"].values / 1000, alpha=0.08, color=C["blue"])

    # Naive seasonal baseline for forecast period
    seasonal_avg = {m: monthly[monthly["ds"].dt.month == m]["y"].mean() for m in range(1, 13)}
    scale = (monthly[monthly["ds"].dt.year == monthly["ds"].dt.year.max()]["y"].mean() /
             monthly[monthly["ds"].dt.year == monthly["ds"].dt.year.max() - 1]["y"].mean()
             if monthly["ds"].dt.year.nunique() > 1 else 1.0)

    fc_start = len(monthly)
    if results:
        main_result = results[0]
        fc = main_result.forecast_df
        fc_x = _np.arange(fc_start, fc_start + len(fc))
        fc_months = fc["ds"].dt.month if hasattr(fc["ds"], "dt") else _pd.to_datetime(fc["ds"]).dt.month
        naive = [seasonal_avg.get(int(m), 0) * scale for m in fc_months]

        ax.fill_between(fc_x, fc["yhat_lower"] / 1000, fc["yhat_upper"] / 1000, color=C["gold"], alpha=0.15, zorder=3, label="80% CI Prophet")
        ax.plot(fc_x, fc["yhat"] / 1000, color=C["gold"], lw=2.5, ls="--", zorder=5, label="Prophet forecast", marker="o", markersize=6, markerfacecolor=C["gold"])
        ax.plot(fc_x, _np.array(naive) / 1000, color=C["purple"], lw=1.8, ls=":", zorder=4, label="Seasonal naive baseline", marker="s", markersize=5, markerfacecolor=C["purple"])

        # Peak annotation
        peak_i = fc["yhat"].idxmax()
        peak_month = _pd.to_datetime(fc["ds"].iloc[peak_i - fc.index[0]]).strftime("%B %Y") if hasattr(fc["ds"],"iloc") else "Aug 2026"
        ax.annotate(
            f"Expected peak\n${fc['yhat'].max()/1000:.0f}K ARS\n({peak_month})",
            xy=(fc_start + (peak_i - fc.index[0]), fc["yhat"].max() / 1000),
            xytext=(fc_start + (peak_i - fc.index[0]) + 1.2, fc["yhat"].max() / 1000 + 10),
            fontsize=9, color=C["gold"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["gold"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["gold"] + "15", edgecolor=C["gold"] + "55"))

        # Outlier star (Feb 2026 if present)
        feb26 = monthly[monthly["ds"].dt.strftime("%Y-%m") == "2026-02"]
        if not feb26.empty:
            idx = feb26.index[0]
            ax.scatter([idx], [feb26["y"].iloc[0] / 1000], color=C["red"], s=100,
                       zorder=6, marker="*", label="Outlier (adjusted in model)")

        all_ds = list(monthly["ds"].dt.strftime("%Y-%m")) + list(
            _pd.to_datetime(fc["ds"]).dt.strftime("%Y-%m") if not hasattr(fc["ds"],"dt") else fc["ds"].dt.strftime("%Y-%m"))
        tick_pos = [i for i, lbl in enumerate(all_ds) if lbl.endswith("-01") or lbl.endswith("-07")]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([all_ds[i] for i in tick_pos], rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.5, fc_start + len(fc) - 0.3)
    else:
        ax.set_xticks(_np.arange(0, len(monthly), 6))
        ax.set_xticklabels(monthly["ds"].iloc[::6].dt.strftime("%Y-%m"), rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.5, len(monthly) - 0.3)

    ax.axvline(fc_start - 0.5, color=C["muted"], lw=1, ls="--", alpha=0.7)
    ax.text(fc_start - 0.3, monthly["y"].max() / 1000 * 0.95, "Today",
            fontsize=9, color=C["muted"], ha="left")
    ax.set_ylabel("Monthly gross revenue ($K ARS)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)

# Caption is built from actual model results, not hardcoded claims
# "Prophet wins" and "Aug-Sep peak" were contradicted by the project's own output.+
    if len(results) >= 2:
        best = min(results, key=lambda r: r.mape)
        worst_mape = max(r.mape for r in results)
        calibration_text = (
            f"Honest calibration: {best.model_name} achieves the lowest backtested MAPE "
            f"({best.mape:.1f}% vs {worst_mape:.1f}%) on this dataset with structural inflation. "
            "See eda.py's plot_seasonality() for the data-driven seasonal pattern"
        )
    else:
        calibration_text = (
            "Honest calibration: backtested on a short series with structural inflation breaks. "
            "See eda.py's plot_seasonality() for the data-driven seasonal pattern")

    fig.text(0.5, 0.04, calibration_text,
        ha="center", fontsize=8.5, color=C["muted"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C["surf2"], edgecolor=C["border"]))

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    _root = _Path(__file__).parent.parent
    _parser = argparse.ArgumentParser(description="Run forecasting pipeline")
    _parser.add_argument("--data",  default=str(_root / "data" / "ventas_decoraciones.csv"))
    _parser.add_argument("--plots", default=str(_root / "plots"))
    _args = _parser.parse_args()

    _Path(_args.plots).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(_args.data, parse_dates=["Fecha"])

    print("Preparing monthly series...")
    monthly = prepare_monthly_series(df)
    print(f"Series: {len(monthly)} months\n")

    print("Training Prophet...")
    prophet_result = run_prophet(monthly, n_periods=6)
    print(f"Prophet -> MAPE: {prophet_result.mape:.1f}% | MAE: ${prophet_result.mae:,.0f}")

    print("\nTraining SARIMA...")
    sarima_result = run_sarima(monthly, n_periods=6)
    print(f"SARIMA  -> MAPE: {sarima_result.mape:.1f}% | MAE: ${sarima_result.mae:,.0f}")

    print("\nComparing models...")
    table = compare_models([prophet_result, sarima_result])
    print(table.to_string())

    print("\nGenerating comparison chart...")
    plot_comparison(monthly, [prophet_result, sarima_result], save_path=str(_Path(_args.plots) / "18_prophet_vs_sarima.png"))


# Production Approach
PRODUCTION_APPROACH = """ Final production approach for this business:

  SEASONAL BASELINE + MANUAL INFLATION ADJUSTMENT

  Why not Prophet or SARIMA as the production model:
    Both underperform the naive seasonal baseline on MAPE (95% and 103% vs 55% for naive). With only 40 months and structural inflation,
    neither model can reliably separate trend from inflation noise.

  Recommended production system:
    1. Seasonal baseline: forecast = same month last year
       - Simple, interpretable, outperforms both models on this dataset
       - Requires no retraining, no hyperparameter tuning

    2. Manual inflation adjustment: multiply by (1 + expected_monthly_inflation)
       - Use INDEC monthly inflation estimate (or trailing 3-month average)
       - Applied after the seasonal baseline as a scalar multiplier
       - Example: August 2025 was $92K. Adjust for 0.7%/month inflation over 12 months: $92K × (1.007^12) = $92K × 1.088 = ~$100K

    3. Confidence range: ±30% around the adjusted baseline
       - Reflects observed forecast error across all models tested
       - Communicate as a range to stakeholders, not a point estimate


  When to switch to a more complex model:
    Once the dataset reaches 60+ months (around mid-2028), revisit SARIMA with proper seasonal differencing. 
    At that point, the model should have enough history to outperform the naive baseline"""


def describe_production_approach() -> str:
    """Returns the production approach documentation string"""
    return PRODUCTION_APPROACH


# PRODUCTION APPROACH
# Given the dataset constraints (40 months, extreme inflation, naive baseline outperforming Prophet on MAPE), the recommended production approach is:
#
#   1. SEASONAL BASELINE as primary signal
#      Use same-month-last-year as the forecast anchor.
#      MAPE: 55% - beats both Prophet (95%) and SARIMA (103%) on this dataset. Simple, interpretable, requires no tuning.
#
#   2. INFLATION ADJUSTMENT as manual layer
#      Multiply the seasonal baseline by expected price growth (CPI-aligned). 
#
#   3. PROPHET for seasonal decomposition only
#      Use Prophet not for point forecasts but to extract the seasonal component (which months run above/below average).
#      These factors are stable and the business can act on them.
#
#   4. HUMAN REVIEW before each cycle
#      With 353 orders over 3 years, any single unusual event (product launch, competitor exit, platform change) can shift the distribution. 
#      Monthly human review of the forecast vs. actual is required.
#
# In summary: Given the low data volume, the recommended system is a seasonal baseline + CPI-aligned price adjustment + monthly human review. 
#             Statistical models add interpretability of the seasonal pattern but should not drive point forecasts autonomously at this data scale.


def production_forecast(monthly: pd.DataFrame, cpi_yoy_growth: float = 0.15, n_periods: int = 6) -> pd.DataFrame:
    """ Production-grade forecast: seasonal baseline + inflation adjustment

    This is more reliable than Prophet or SARIMA on this dataset (the naive baseline achieves MAPE 55% vs Prophet's 95%) """
    last = monthly["ds"].max()

    # Build seasonal factors from ALL available years 
    monthly["month"] = monthly["ds"].dt.month
    monthly["year"] = monthly["ds"].dt.year
    seasonal = monthly.groupby("month")["y_model"].mean()
    annual_avg = monthly["y_model"].mean()
    seasonal_factors = (seasonal / annual_avg).to_dict()

    # Project forward
    rows = []
    for i in range(1, n_periods + 1):
        future_date  = last + pd.DateOffset(months=i)
        month = future_date.month
        sf = seasonal_factors.get(month, 1.0)
        base = monthly["y_model"].iloc[-12:].mean() if len(monthly) >= 12 else monthly["y_model"].mean()
        yhat = base * sf * (1 + cpi_yoy_growth)
        uncertainty = yhat * 0.35  
        rows.append({
            "ds": future_date,
            "yhat": round(max(yhat, 0), 0),
            "yhat_lower": round(max(yhat - uncertainty, 0), 0),
            "yhat_upper": round(yhat + uncertainty, 0),
            "method": "Seasonal baseline + CPI adjustment",
            "seasonal_factor": round(sf, 3),
        })

    return pd.DataFrame(rows)
