import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass


# Stub ForecastResult (mirrors the real dataclass) 
@dataclass
class StubForecastResult:
    model_name: str
    forecast_df: pd.DataFrame
    mape: float
    mae: float
    rmse: float


def make_monthly(n: int = 34, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    y = 30_000 + np.arange(n) * 800 + rng.normal(0, 4000, n)
    return pd.DataFrame({"ds": dates, "y": np.clip(y, 1000, None)})


def make_forecast_result(name: str = "Prophet", n_periods: int = 6) -> StubForecastResult:
    fc_dates = pd.date_range("2026-05-01", periods=n_periods, freq="MS")
    fc_df = pd.DataFrame({
        "ds": fc_dates,
        "yhat": [42000, 38000, 48000, 66000, 54000, 39000],
        "yhat_lower": [25000, 22000, 30000, 45000, 35000, 22000],
        "yhat_upper": [59000, 54000, 66000, 87000, 73000, 56000],})
    
    return StubForecastResult(
        model_name=name, forecast_df=fc_df,
        mape=55.2, mae=18396, rmse=21824,)


# plot_comparison smoke tests 
class TestPlotComparison:
    def test_returns_nothing_no_crash(self):
        """plot_comparison should run without raising any exception """
        from src.forecasting import plot_comparison
        monthly = make_monthly()
        results = [make_forecast_result("Prophet")]
        plot_comparison(monthly, results)  
        plt.close("all")

    def test_saves_file(self, tmp_path):
        from src.forecasting import plot_comparison
        monthly = make_monthly()
        results = [make_forecast_result()]
        out = str(tmp_path / "forecast.png")
        plot_comparison(monthly, results, save_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 10_000
        plt.close("all")

    def test_handles_empty_results(self):
        """Should not crash if results list is empty"""
        from src.forecasting import plot_comparison
        monthly = make_monthly()
        plot_comparison(monthly, [])
        plt.close("all")

    def test_handles_multiple_results(self):
        from src.forecasting import plot_comparison
        monthly = make_monthly()
        results = [make_forecast_result("Prophet"), make_forecast_result("SARIMA")]
        plot_comparison(monthly, results)
        plt.close("all")

    def test_output_file_is_image(self, tmp_path):
        """Saved file should have PNG magic bytes """
        from src.forecasting import plot_comparison
        out = str(tmp_path / "fc.png")
        plot_comparison(make_monthly(), [make_forecast_result()], save_path=out)
        with open(out, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG", "Output is not a valid PNG file"
        plt.close("all")


# run_naive_baseline smoke tests 
class TestRunNaiveBaseline:
    def test_returns_forecast_result(self):
        from src.forecasting import run_naive_baseline
        monthly = make_monthly()
        monthly = monthly.rename(columns={"y": "y_model"})
        monthly["y"] = monthly["y_model"]
        result = run_naive_baseline(monthly, n_periods=6)
        assert hasattr(result, "forecast_df")
        assert hasattr(result, "mape")
        assert len(result.forecast_df) == 6

    def test_naive_yhat_all_positive(self):
        from src.forecasting import run_naive_baseline
        monthly = make_monthly()
        monthly = monthly.rename(columns={"y": "y_model"})
        monthly["y"] = monthly["y_model"]
        result = run_naive_baseline(monthly, n_periods=6)
        assert (result.forecast_df["yhat"] > 0).all()

    def test_naive_mape_finite(self):
        from src.forecasting import run_naive_baseline
        monthly = make_monthly(n=40)
        monthly = monthly.rename(columns={"y": "y_model"})
        monthly["y"] = monthly["y_model"]
        result = run_naive_baseline(monthly, n_periods=6)
        assert np.isfinite(result.mape)


# plot_style smoke tests 
class TestPlotStyle:
    def test_apply_theme_sets_dark_background(self):
        from src.plot_style import apply_theme
        import matplotlib as mpl
        apply_theme()
        assert mpl.rcParams["figure.facecolor"] == "#0A0E17"

    def test_styled_fig_returns_figure(self):
        from src.plot_style import styled_fig
        fig = styled_fig(12, 5, title="Test", subtitle="Sub")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_fmt_ars_formats_thousands(self):
        from src.plot_style import fmt_ars
        assert fmt_ars(45_000) == "$45K"
        assert fmt_ars(1_500_000) == "$1.5M"
        assert fmt_ars(999) == "$999"

    def test_year_colors_all_present(self):
        from src.plot_style import YEAR_C
        for yr in [2023, 2024, 2025, 2026]:
            assert yr in YEAR_C
            assert YEAR_C[yr].startswith("#")

    def test_seg_colors_all_present(self):
        from src.plot_style import SEG_COLORS
        for seg in ["VIP", "Loyal", "Potential", "At risk", "Occasional", "Lost"]:
            assert seg in SEG_COLORS
