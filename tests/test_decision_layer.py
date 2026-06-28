import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.decision_layer import load_inputs, build_pricing_strategy

@pytest.fixture(scope="module")
def strategy():
    """Run the real pipeline once against the real data/ files """
    inputs = load_inputs()
    return build_pricing_strategy(inputs)


class TestLoadInputs:
    def test_loads_without_error(self):
        inputs = load_inputs()
        assert set(inputs) >= {"df", "rfm", "fc", "monthly", "df_ml"}

    def test_rfm_uses_english_segment_values(self):
        """Guards against the Segmento/Potencial (Spanish) regression."""
        inputs = load_inputs()
        assert "Segment" in inputs["rfm"].columns
        assert "Potential" in set(inputs["rfm"]["Segment"])


class TestBuildPricingStrategy:
    def test_runs_without_error(self, strategy):
        assert strategy is not None

    def test_rfm_opportunity_is_nonzero(self, strategy):
        """potential_count was silently 0 under the old Spanish-column bug, a real dataset should always have at least one Potential customer."""
        assert strategy["rfm_opportunity"]["potential_count"] > 0
        assert strategy["rfm_opportunity"]["potential_revenue"] > 0

    def test_segments_cover_all_customers(self, strategy):
        seg = strategy["segments"]
        inputs = load_inputs()
        assert seg["count"].sum() == len(inputs["rfm"])

    def test_segments_revenue_sums_to_100_pct(self, strategy):
        seg = strategy["segments"]
        assert seg["pct_revenue"].sum() == pytest.approx(100.0, abs=0.5)

    def test_forecast_matches_csv(self, strategy):
        inputs = load_inputs()
        expected = inputs["fc"]["Proyeccion_central"].sum()
        assert strategy["forecast"]["central"] == pytest.approx(expected, abs=1)

    def test_net_margin_and_fee_rate_complement(self, strategy):
        cs = strategy["current_state"]
        assert cs["ml_fee_rate"] + cs["net_margin"] == pytest.approx(1.0, abs=1e-6)

    def test_seasonality_keys_present(self, strategy):
        seas = strategy["seasonality"]
        assert "peak_pct_avg" in seas
        assert "peak_months" in seas
        assert "peak_label" in seas
        # With >= 1 complete calendar year on file this should be a real number
        if seas["complete_years"]:
            assert seas["peak_pct_avg"] is not None
            assert 0 <= seas["peak_pct_avg"] <= 100

    def test_peak_months_are_data_driven_not_hardcoded(self, strategy):
        """Regression: peak months are detected dynamically, not hardcoded to Aug-Sep.
        On this dataset the real top-2 are September and January; a regression to hardcoded months would fail here instead of silently conflicting with eda.py and dashboard.py."""
        seas = strategy["seasonality"]
        if seas["complete_years"]:
            assert seas["peak_months"] != [8, 9]
            assert set(seas["peak_months"]) == {9, 1}


class TestPrintStrategyConsole:

    def test_runs_without_error(self, strategy, capsys):
        from src.decision_layer import print_strategy_console
        print_strategy_console(strategy)
        out = capsys.readouterr().out
        assert "PRICING & GROWTH STRATEGY" in out
        assert "$2,690" not in out
