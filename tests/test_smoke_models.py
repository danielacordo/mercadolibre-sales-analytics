import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch


# Shared fixture
def make_monthly(n: int = 34, seed: int = 42) -> pd.DataFrame:
    """Minimal monthly series that satisfies run_prophet / run_sarima inputs"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    y = 30_000 + np.arange(n) * 600 + rng.normal(0, 3_000, n)
    y = np.clip(y, 1_000, None)
    df = pd.DataFrame({
        "ds": dates,
        "y": y,
        "y_model": y,
        "z_score": np.zeros(n),
        "Fecha": pd.PeriodIndex(dates, freq="M"),})
    return df



# Prophet Mock Helpers 
def _make_prophet_mock(monthly: pd.DataFrame, n_periods: int = 6):
    """ Build a mock that makes `from prophet import Prophet` work and returns a minimal forecast DataFrame from model.predict()"""
    _n = len(monthly)
    all_dates = pd.concat([
        monthly[["ds"]],
        pd.DataFrame({"ds": pd.date_range(
            monthly["ds"].max() + pd.DateOffset(months=1),
            periods=n_periods, freq="MS"
        )})
    ], ignore_index=True)

    forecast_df = pd.DataFrame({
        "ds": all_dates["ds"],
        "yhat": np.linspace(30_000, 55_000, len(all_dates)),
        "yhat_lower": np.linspace(20_000, 40_000, len(all_dates)),
        "yhat_upper": np.linspace(40_000, 70_000, len(all_dates)),
        "trend": np.linspace(28_000, 50_000, len(all_dates)),})

    # cv output from cross_validation()
    cv_df = pd.DataFrame({
        "ds": monthly["ds"].iloc[-6:].values,
        "yhat": monthly["y_model"].iloc[-6:].values * 0.9,
        "y": monthly["y_model"].iloc[-6:].values,})

    # performance_metrics() output
    metrics_df = pd.DataFrame({
        "horizon": pd.to_timedelta(["30 days", "60 days", "90 days"]),
        "mape": [0.85, 0.90, 1.02],
        "mae": [18_000, 20_000, 23_000],
        "rmse": [22_000, 25_000, 28_000],})

    mock_model = MagicMock()
    mock_model.fit.return_value = mock_model
    mock_model.make_future_dataframe.return_value = all_dates
    mock_model.predict.return_value = forecast_df

    prophet_module = MagicMock()
    prophet_module.Prophet.return_value = mock_model

    diagnostics_module = MagicMock()
    diagnostics_module.cross_validation.return_value = cv_df
    diagnostics_module.performance_metrics.return_value = metrics_df

    return prophet_module, diagnostics_module, mock_model



# SARIMA Mock Helpers 
def _make_sarima_mock(n_periods: int = 6):
    """ Build a mock that makes `from statsmodels.tsa.statespace.sarimax import SARIMAX` work and returns realistic forecast values """
    fc_values = np.linspace(40_000, 60_000, n_periods)

    ci_array = np.column_stack([
        fc_values * 0.7, # lower bound
        fc_values * 1.3, # upper bound
    ])

    mock_forecast_obj = MagicMock()
    mock_forecast_obj.conf_int.return_value = ci_array

    mock_fitted = MagicMock()
    mock_fitted.forecast.return_value = fc_values
    mock_fitted.get_forecast.return_value = mock_forecast_obj

    mock_model_instance = MagicMock()
    mock_model_instance.fit.return_value = mock_fitted

    mock_sarimax_class = MagicMock(return_value=mock_model_instance)

    sarimax_module = MagicMock()
    sarimax_module.SARIMAX = mock_sarimax_class

    return sarimax_module, mock_fitted


# run_prophet - smoke tests
class TestRunProphetSmoke:
    """Smoke tests for run_prophet() using a mock Prophet.

    These tests verify:
      - The function returns a ForecastResult with all required fields
      - Forecast values are non-negative
      - CI ordering holds (lower <= yhat <= upper)
      - ProphetComponents is attached for downstream plot_components()
      - n_periods controls the length of the forecast"""

    def _run(self, monthly, n_periods=6):
        prophet_mod, diag_mod, _ = _make_prophet_mock(monthly, n_periods)
        with patch.dict("sys.modules", {
            "prophet": prophet_mod,
            "prophet.diagnostics": diag_mod,
        }):
            from src.forecasting import run_prophet
            return run_prophet(monthly, n_periods=n_periods)

    def test_returns_forecast_result(self):
        from src.forecasting import ForecastResult
        monthly = make_monthly()
        result = self._run(monthly)
        assert isinstance(result, ForecastResult)

    def test_model_name_is_prophet(self):
        result = self._run(make_monthly())
        assert "Prophet" in result.model_name

    def test_forecast_length_matches_n_periods(self):
        for n in [3, 6, 9]:
            result = self._run(make_monthly(), n_periods=n)
            assert len(result.forecast_df) == n, \
                f"Expected {n} rows, got {len(result.forecast_df)}"

    def test_forecast_columns_present(self):
        result = self._run(make_monthly())
        for col in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
            assert col in result.forecast_df.columns, f"Missing column: {col}"

    def test_yhat_non_negative(self):
        result = self._run(make_monthly())
        assert (result.forecast_df["yhat"] >= 0).all(), \
            "Prophet forecast must clip negatives to 0"

    def test_ci_ordering(self):
        """yhat_lower <= yhat <= yhat_upper for every forecast row."""
        result = self._run(make_monthly())
        fc = result.forecast_df
        assert (fc["yhat_lower"] <= fc["yhat"]).all(), "yhat_lower > yhat"
        assert (fc["yhat"] <= fc["yhat_upper"]).all(), "yhat > yhat_upper"

    def test_mape_is_positive_finite(self):
        result = self._run(make_monthly())
        assert np.isfinite(result.mape) and result.mape > 0

    def test_prophet_components_attached(self):
        """components field must be set (used by plot_components downstream)"""
        from src.forecasting import ProphetComponents
        result = self._run(make_monthly())
        assert result.components is not None
        assert isinstance(result.components, ProphetComponents)

    def test_mae_and_rmse_positive(self):
        result = self._run(make_monthly())
        assert result.mae > 0
        assert result.rmse > 0


# run_sarima - smoke tests
class TestRunSarimaSmoke:
    """Smoke tests for run_sarima() using a mock SARIMAX"""

    def _run(self, monthly, n_periods=6):
        sarimax_mod, _ = _make_sarima_mock(n_periods)
        with patch.dict("sys.modules", {
            "statsmodels": MagicMock(),
            "statsmodels.tsa": MagicMock(),
            "statsmodels.tsa.statespace": MagicMock(),
            "statsmodels.tsa.statespace.sarimax": sarimax_mod,
        }):
            from src.forecasting import run_sarima
            return run_sarima(monthly, n_periods=n_periods)

    def test_returns_forecast_result(self):
        from src.forecasting import ForecastResult
        result = self._run(make_monthly())
        assert isinstance(result, ForecastResult)

    def test_model_name_contains_sarima(self):
        result = self._run(make_monthly())
        assert "SARIMA" in result.model_name.upper()

    def test_forecast_length_matches_n_periods(self):
        for n in [3, 6, 9]:
            result = self._run(make_monthly(), n_periods=n)
            assert len(result.forecast_df) == n

    def test_forecast_columns_present(self):
        result = self._run(make_monthly())
        for col in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
            assert col in result.forecast_df.columns

    def test_yhat_non_negative(self):
        result = self._run(make_monthly())
        assert (result.forecast_df["yhat"] >= 0).all()

    def test_ci_ordering(self):
        result = self._run(make_monthly())
        fc = result.forecast_df
        assert (fc["yhat_lower"] <= fc["yhat"]).all(), "yhat_lower > yhat"
        assert (fc["yhat"] <= fc["yhat_upper"]).all(), "yhat > yhat_upper"

    def test_mape_is_positive_finite(self):
        result = self._run(make_monthly())
        assert np.isfinite(result.mape) and result.mape > 0

    def test_components_is_none(self):
        """SARIMA does not expose Prophet components - field must be None."""
        result = self._run(make_monthly())
        assert result.components is None, \
            "SARIMA ForecastResult.components must be None"

    def test_mae_and_rmse_positive(self):
        result = self._run(make_monthly())
        assert result.mae > 0
        assert result.rmse > 0


# compare_models - contract test
class TestCompareModelsContract:
    """Verify compare_models() works correctly with ForecastResult objects from both mocked Prophet and SARIMA runs"""

    def _prophet_result(self, mape=95.0):
        from src.forecasting import ForecastResult, ProphetComponents
        fc = pd.DataFrame({
            "ds": pd.date_range("2026-05-01", periods=6, freq="MS"),
            "yhat": [45_000.0] * 6,
            "yhat_lower": [30_000.0] * 6,
            "yhat_upper": [60_000.0] * 6, })
        return ForecastResult(
            model_name="Prophet", mape=mape, mae=18_000.0,
            rmse=22_000.0, forecast_df=fc,
            components=ProphetComponents(model=MagicMock(), forecast=fc),)

    def _sarima_result(self, mape=103.0):
        from src.forecasting import ForecastResult
        fc = pd.DataFrame({
            "ds": pd.date_range("2026-05-01", periods=6, freq="MS"),
            "yhat": [42_000.0] * 6,
            "yhat_lower": [28_000.0] * 6,
            "yhat_upper": [56_000.0] * 6,})
        return ForecastResult(
            model_name="SARIMA(1,1,1)(1,1,0,12)", mape=mape,
            mae=20_000.0, rmse=24_000.0, forecast_df=fc,
        )

    def test_compare_prophet_and_sarima(self):
        from src.forecasting import compare_models
        table = compare_models([self._prophet_result(), self._sarima_result()])
        assert "Prophet" in table.index
        assert "SARIMA(1,1,1)(1,1,0,12)" in table.index

    def test_mape_column_present(self):
        from src.forecasting import compare_models
        table = compare_models([self._prophet_result()])
        assert "MAPE (%)" in table.columns

    def test_three_models_including_naive(self):
        from src.forecasting import compare_models, ForecastResult
        fc = pd.DataFrame({
            "ds": pd.date_range("2026-05-01", periods=6, freq="MS"),
            "yhat": [np.nan] * 6, "yhat_lower": [np.nan] * 6, "yhat_upper": [np.nan] * 6,})
        naive = ForecastResult(
            model_name="Naive (same month last year)", mape=55.2,
            mae=14_000.0, rmse=17_000.0, forecast_df=fc,)
        table = compare_models([
            self._prophet_result(), self._sarima_result(), naive])
        assert len(table) == 3
