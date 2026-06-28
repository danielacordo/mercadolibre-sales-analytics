import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import pandas as pd
import numpy as np
import warnings


# Fixtures
def make_monthly(n_months: int = 30, seed: int = 42) -> pd.DataFrame:
    """Builds a synthetic monthly series for testing"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_months, freq="MS")
    y = 30_000 + np.arange(n_months) * 500 + rng.normal(0, 3000, n_months)
    y = np.clip(y, 1000, None)
    df = pd.DataFrame({"ds": dates, "y_model": y, "y": y})
    df["z_score"] = (df["y"] - df["y"].mean()) / df["y"].std()
    return df


def make_transactions(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Builds a minimal transaction DataFrame """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="3D")
    return pd.DataFrame({
        "Order_id": range(n),
        "Fecha": dates,
        "Monto": rng.uniform(500, 15000, n).round(0),
        "Cantidad": np.ones(n),
        "Ingreso_bruto": rng.uniform(500, 15000, n).round(0),
        "Ingreso_neto": rng.uniform(300, 10000, n).round(0),
        "Mes": dates.month,
        "Año": dates.year,
        "Provincia_nombre": rng.choice(["Buenos Aires","Córdoba","Santa Fe"], n),
        "Fuente": rng.choice(["CNX","ML_Oficial"], n),})


def make_rfm(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Builds a minimal RFM DataFrame"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Cliente": [f"buyer_{i}" for i in range(n)],
        "ultima_compra": pd.date_range("2024-01-01", periods=n, freq="7D"),
        "frecuencia": rng.choice([1, 1, 1, 2, 3], n),
        "monto_total": rng.uniform(1000, 30000, n).round(0),
        "recencia": rng.integers(1, 400, n),
        "R_score": rng.integers(1, 5, n),
        "F_score": rng.integers(1, 5, n),
        "M_score": rng.integers(1, 5, n),
        "Segmento": rng.choice(["Potencial","VIP","Leal","Perdido"], n),})


# Forecasting Tests 
def make_raw_transactions(n: int = 80) -> pd.DataFrame:
    """Minimal transaction DataFrame matching prepare_monthly_series() inputs."""
    rng = np.random.default_rng(0)
    # Spread across 30+ months so the 24-month assert passes
    dates = pd.date_range("2022-01-01", periods=n, freq="15D")
    return pd.DataFrame({
        "Fecha": dates,
        "Order_id": range(n),
        "Ingreso_bruto": rng.uniform(1_000, 20_000, n).round(0),
        "Monto": rng.uniform(1_000, 20_000, n).round(0),})

class TestPrepareMonthlySeries:
    """Tests for prepare_monthly_series(), the central ETL step that:
      1. Aggregates raw transactions to monthly frequency
      2. Computes z-scores for outlier detection
      3. Replaces outlier months (|z| > 2) with the median of normal months
      4. Asserts the series is at least 24 months long"""

    def test_returns_dataframe(self):
        from src.forecasting import prepare_monthly_series
        result = prepare_monthly_series(make_raw_transactions())
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        """Output must have the columns downstream modules depend on."""
        from src.forecasting import prepare_monthly_series
        result = prepare_monthly_series(make_raw_transactions())
        for col in ["ds", "y", "y_model", "z_score", "ventas", "ticket"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_ds_is_datetime(self):
        from src.forecasting import prepare_monthly_series
        result = prepare_monthly_series(make_raw_transactions())
        assert pd.api.types.is_datetime64_any_dtype(result["ds"])

    def test_monthly_aggregation(self):
        """3 transactions per month x 30 months -> exactly 30 rows."""
        from src.forecasting import prepare_monthly_series
        rows = []
        for m in range(30):
            base = pd.Timestamp("2022-01-01") + pd.DateOffset(months=m)
            for d in [1, 10, 20]:
                rows.append({
                    "Fecha": base + pd.Timedelta(days=d - 1),
                    "Order_id": m * 3 + d,
                    "Ingreso_bruto": 10_000.0,
                    "Monto": 10_000.0,})
        result = prepare_monthly_series(pd.DataFrame(rows))
        assert len(result) == 30

    def test_raises_on_fewer_than_24_months(self):
        """Series with < 24 monthly buckets must raise AssertionError"""
        from src.forecasting import prepare_monthly_series
        rng = np.random.default_rng(0)
        dates = pd.date_range("2023-01-01", periods=18, freq="7D")
        df = pd.DataFrame({
            "Fecha": dates,
            "Order_id": range(18),
            "Ingreso_bruto": rng.uniform(1_000, 5_000, 18),
            "Monto": rng.uniform(1_000, 5_000, 18),})
        with pytest.raises(AssertionError, match="too short"):
            prepare_monthly_series(df)

    def test_outlier_replacement_keeps_y_model_lower(self):
        """An extreme month (injected 5M ARS order) should have:
          - y  = real sum (very high)
          - y_model = median of normal months (much lower)"""
        from src.forecasting import prepare_monthly_series
        df = make_raw_transactions(120)
        extreme = pd.DataFrame({
            "Fecha": [pd.Timestamp("2025-06-15")],
            "Order_id": [99_999],
            "Ingreso_bruto": [5_000_000.0],
            "Monto": [5_000_000.0],})
        
        result = prepare_monthly_series(pd.concat([df, extreme], ignore_index=True))
        jun25 = result[result["ds"] == pd.Timestamp("2025-06-01")]
        if len(jun25) and jun25["z_score"].values[0] > 2:
            assert jun25["y_model"].values[0] < jun25["y"].values[0], \
                "Outlier month: y_model must be less than raw y after replacement"

    def test_y_model_non_negative(self):
        """Outlier replacement must never produce negative values """
        from src.forecasting import prepare_monthly_series
        result = prepare_monthly_series(make_raw_transactions(120))
        assert (result["y_model"] >= 0).all()


class TestRunNaiveBaseline:
    def test_returns_forecast_result(self):
        from src.forecasting import run_naive_baseline, ForecastResult
        monthly = make_monthly(24)
        result = run_naive_baseline(monthly, n_periods=3)
        assert isinstance(result, ForecastResult)

    def test_mape_finite_with_enough_data(self):
        from src.forecasting import run_naive_baseline
        monthly = make_monthly(24)
        result = run_naive_baseline(monthly)
        assert np.isfinite(result.mape), "MAPE should be finite with 24 months"

    def test_mape_inf_with_short_series(self):
        """Bug 1 fix: series < 12 months returns inf, not NaN."""
        from src.forecasting import run_naive_baseline
        monthly = make_monthly(6)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_naive_baseline(monthly)
            assert result.mape == float("inf"), "MAPE must be inf, not NaN"
            assert len(w) == 1
            assert "no prior-year observations" in str(w[0].message).lower()

    def test_forecast_df_has_required_columns(self):
        from src.forecasting import run_naive_baseline
        monthly = make_monthly(24)
        result = run_naive_baseline(monthly, n_periods=4)
        for col in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
            assert col in result.forecast_df.columns

    def test_forecast_length_matches_n_periods(self):
        from src.forecasting import run_naive_baseline
        monthly = make_monthly(24)
        for n in [3, 6, 12]:
            result = run_naive_baseline(monthly, n_periods=n)
            assert len(result.forecast_df) == n


class TestProductionForecast:
    def test_returns_dataframe(self):
        from src.forecasting import production_forecast
        monthly = make_monthly(30)
        result = production_forecast(monthly, n_periods=6)
        assert isinstance(result, pd.DataFrame)

    def test_all_yhat_positive(self):
        from src.forecasting import production_forecast
        monthly = make_monthly(30)
        result = production_forecast(monthly, n_periods=6)
        assert (result["yhat"] >= 0).all(), "No negative forecasts"

    def test_lower_leq_yhat_leq_upper(self):
        from src.forecasting import production_forecast
        monthly = make_monthly(30)
        result = production_forecast(monthly, n_periods=6)
        assert (result["yhat_lower"] <= result["yhat"]).all()
        assert (result["yhat"] <= result["yhat_upper"]).all()

    def test_seasonal_factor_column_present(self):
        from src.forecasting import production_forecast
        monthly = make_monthly(30)
        result = production_forecast(monthly, n_periods=3)
        assert "seasonal_factor" in result.columns

    def test_uses_all_years_for_seasonal(self):
        """Bug 8 fix: seasonal factors should be from all available years """
        from src.forecasting import production_forecast
        # Build a series where year 2 Sep is anomalously low
        monthly = make_monthly(30)
        monthly["month"] = monthly["ds"].dt.month
        # Artificially suppress September in the last year only
        sep_last = (monthly["ds"].dt.year == monthly["ds"].dt.year.max()) & \
                   (monthly["ds"].dt.month == 9)
        monthly.loc[sep_last, "y_model"] = 1000  # anomalously low
        result = production_forecast(monthly, n_periods=12)
        # Sep should still have positive factor (not collapsed to near zero)
        sep_rows = result[result["ds"].dt.month == 9]
        if len(sep_rows) > 0:
            assert sep_rows["seasonal_factor"].values[0] > 0.3, \
                "Sep factor collapsed - seasonal averaging is biased toward recent years"



# AB Testing Tests 
class TestEstimateSegments:
    def test_returns_list_of_segments(self):
        from src.ab_testing import estimate_segments, Segment
        df = make_transactions()
        rfm = make_rfm()
        segs = estimate_segments(df, rfm)
        assert isinstance(segs, list)
        assert all(isinstance(s, Segment) for s in segs)

    def test_no_crash_with_small_rfm(self):
        """Bug 2 fix: quantile(x > 1) should not crash """
        from src.ab_testing import estimate_segments
        df = make_transactions(20)
        rfm = make_rfm(10)
        segs = estimate_segments(df, rfm)  # should not raise
        assert len(segs) > 0

    def test_elasticities_inherit_sign_from_aggregate(self):
        """Segment elasticities are aggregate * scale. If agg < 0, all price tier elasticities should be < 0 (scale factors are always positive). """
        from src.ab_testing import estimate_segments
        from scipy import stats
        df = make_transactions()
        rfm = make_rfm()
        segs = estimate_segments(df, rfm)
        # Price tier segments have scale > 0, so sign matches aggregate
        monthly_m = df.groupby(df["Fecha"].dt.to_period("M")).agg(quantity=("Order_id","count"), price=("Monto","median")).reset_index()
        monthly_m["ds"] = monthly_m["Fecha"].dt.to_timestamp()
        clean = monthly_m.dropna().query("price > 0 and quantity > 0")
        agg_eps, *_ = stats.linregress(np.log(clean["price"]), np.log(clean["quantity"]))
        tier_segs = [s for s in segs if "Price tier" in s.name]
        for s in tier_segs:
            assert (s.elasticity * agg_eps) > 0 or abs(s.elasticity) < 1e-6, \
                f"Segment {s.name} elasticity sign differs from aggregate"

    def test_revenue_shares_sum_near_one(self):
        from src.ab_testing import estimate_segments
        df = make_transactions()
        rfm = make_rfm()
        segs = estimate_segments(df, rfm)
        # Price tier segments should cover close to 100% of revenue
        tier_segs = [s for s in segs if "Price tier" in s.name]
        total_share = sum(s.revenue_share for s in tier_segs)
        assert 0.9 <= total_share <= 1.1, \
            f"Price tier revenue shares sum to {total_share:.2f}, expected ~1.0"


class TestSimulateABTest:
    def test_returns_ab_result(self):
        from src.ab_testing import simulate_ab_test, Segment, ABResult
        seg = Segment("Test", 100, 0.10, 5000, -0.66, 0.3)
        result = simulate_ab_test(seg, 0.10, n_simulations=500)
        assert isinstance(result, ABResult)

    def test_prob_wins_in_valid_range(self):
        from src.ab_testing import simulate_ab_test, Segment
        seg = Segment("Test", 100, 0.10, 5000, -0.66, 0.3)
        result = simulate_ab_test(seg, 0.10, n_simulations=500)
        assert 0.0 <= result.prob_treatment_wins <= 1.0

    def test_ci_ordering(self):
        from src.ab_testing import simulate_ab_test, Segment
        seg = Segment("Test", 200, 0.15, 8000, -0.40, 0.5)
        result = simulate_ab_test(seg, 0.10, n_simulations=1000)
        assert result.lift_ci_lower <= result.expected_lift <= result.lift_ci_upper, \
            "CI lower must be <= expected_lift <= CI upper"

    def test_highly_inelastic_segment_wins(self):
        """A nearly inelastic segment (ε ≈ 0) should show P(win) > 0.65"""
        from src.ab_testing import simulate_ab_test, Segment
        seg = Segment("Very inelastic", 500, 0.20, 10000, -0.05, 0.4)
        result = simulate_ab_test(seg, 0.10, n_simulations=2000, monthly_visitors=500)
        assert result.prob_treatment_wins > 0.65, \
            f"Near-zero elasticity should show P > 0.65, got P={result.prob_treatment_wins:.2f}"

    def test_elastic_segment_shows_uncertainty(self):
        """A highly elastic segment (ε = -2.0) should show P(win) < 0.45."""
        from src.ab_testing import simulate_ab_test, Segment
        seg = Segment("Very elastic", 200, 0.10, 3000, -2.0, 0.2)
        result = simulate_ab_test(seg, 0.10, n_simulations=2000)
        assert result.prob_treatment_wins < 0.45, \
            f"Highly elastic demand should not win at +10%, got P={result.prob_treatment_wins:.2f}"


class TestFunnelBySegment:
    def test_returns_dataframe(self):
        from src.ab_testing import funnel_by_segment
        df = make_transactions()
        rfm = make_rfm()
        result = funnel_by_segment(df, rfm)
        assert isinstance(result, pd.DataFrame)

    def test_has_three_dimensions(self):
        from src.ab_testing import funnel_by_segment
        df = make_transactions()
        rfm = make_rfm()
        result = funnel_by_segment(df, rfm)
        dims = result["Dimension"].unique()
        assert "Price tier" in dims
        assert "Buyer type" in dims
        assert "Season" in dims

    def test_rev_shares_per_tier_sum_near_100(self):
        from src.ab_testing import funnel_by_segment
        df = make_transactions()
        rfm = make_rfm()
        result = funnel_by_segment(df, rfm)
        tier_share = result[result["Dimension"] == "Price tier"]["Rev share (%)"].astype(float).sum()
        assert 95.0 <= tier_share <= 105.0, \
            f"Price tier rev shares sum to {tier_share:.1f}%, expected ~100%"



# Elasticity Scenario Tests
class TestSimulatePricingScenarios:
    def _make_ci(self, eps=-0.66, lo=-0.83, hi=-0.49):
        return {"epsilon": eps, "ci_lower": lo, "ci_upper": hi, "se_bootstrap": 0.08, "n_months": 40, "n_bootstrap": 2000, "confidence": 0.95}

    def test_returns_dataframe(self):
        from src.elasticity import simulate_pricing_scenarios
        ci = self._make_ci()
        result = simulate_pricing_scenarios(ci, 50_000)
        assert isinstance(result, pd.DataFrame)

    def test_exact_formula_at_10pct(self):
        """Bug 3 fix: uses (1+p)^(1+ε) - 1"""
        from src.elasticity import simulate_pricing_scenarios
        eps = -0.6592
        ci = self._make_ci(eps=eps, lo=eps, hi=eps)
        result = simulate_pricing_scenarios(ci, 100_000, price_increases=[0.10])
        central_pct = result["Central rev Δ (%)"].iloc[0]
        exact = ((1.10) ** (1 + eps) - 1) * 100
        assert abs(central_pct - exact) < 0.1, \
            f"Formula mismatch: got {central_pct:.2f}%, exact = {exact:.2f}%"

    def test_pessimistic_leq_central_leq_optimistic(self):
        from src.elasticity import simulate_pricing_scenarios
        ci = self._make_ci()
        result = simulate_pricing_scenarios(ci, 50_000)
        for _, row in result.iterrows():
            assert row["Pessimistic rev Δ (%)"] <= row["Central rev Δ (%)"], \
                f"Pessimistic > Central for {row['Price increase']}"
            assert row["Central rev Δ (%)"] <= row["Optimistic rev Δ (%)"], \
                f"Central > Optimistic for {row['Price increase']}"

    def test_required_columns_present(self):
        from src.elasticity import simulate_pricing_scenarios
        ci = self._make_ci()
        result = simulate_pricing_scenarios(ci, 50_000)
        required = ["Price increase", "Central rev Δ (%)", "Central rev Δ (ARS/month)", "Pessimistic rev Δ (%)", "Optimistic rev Δ (%)", "Risk"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_worst_case_positive_at_10pct(self):
        """With ε CI = [-0.83, -0.49], the pessimistic revenue delta at +10% should be positive."""
        from src.elasticity import simulate_pricing_scenarios
        ci = self._make_ci()
        result = simulate_pricing_scenarios(ci, 50_000)
        row_10 = result[result["Price increase"] == "+10%"].iloc[0]
        assert row_10["Pessimistic rev Δ (%)"] > 0, \
            f"+10% pessimistic scenario should be revenue-positive, got {row_10['Pessimistic rev Δ (%)']:.2f}%"
