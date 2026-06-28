import sys
import os
import importlib.util
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

BASE = Path(__file__).parent.parent

@pytest.fixture(scope="module")
def dashboard():
    """ Import dashboard.py as a module without running its __main__ block"""
    spec = importlib.util.spec_from_file_location("dashboard_under_test", BASE / "dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

TABS = ["strategy", "overview", "timeseries", "customers", "elasticity"]

class TestModuleLoads:
    def test_imports_without_error(self, dashboard):
        assert dashboard.df is not None
        assert len(dashboard.df) > 0

    def test_seg_colors_keys_match_real_segment_values(self, dashboard):
        """Guards against the Potencial/Leal/... (Spanish) regression, these keys must match rfm['Segment']'s actual values or every non-VIP segment silently renders with the same fallback color."""
        real_segments = set(dashboard.rfm["Segment"].unique())
        assert real_segments.issubset(set(dashboard.seg_colors.keys()))


@pytest.mark.parametrize("tab", TABS)
class TestRenderTab:
    def test_renders_without_error(self, dashboard, tab):
        result = dashboard.render_tab(tab)
        assert result is not None

    def test_potential_count_is_nonzero_in_strategy_tab(self, dashboard, tab):
        if tab != "strategy":
            pytest.skip("only relevant to the strategy tab")
        # Test the underlying RFM filter, not the closure-local _ret_count directly
        assert (dashboard.rfm["Segment"] == "Potential").sum() > 0


CHART_FUNCS = ["fig_price_vs_cpi", "fig_seasonality", "fig_forecast", "fig_geo", "fig_rfm_scatter", "fig_rfm_bars", "fig_elasticity",]


@pytest.mark.parametrize("fn_name", CHART_FUNCS)
def test_chart_function_runs(dashboard, fn_name):
    fn = getattr(dashboard, fn_name)
    fig = fn()
    assert fig is not None


def test_fig_revenue_runs(dashboard):
    fig = dashboard.fig_revenue([2023, 2024, 2025, 2026])
    assert fig is not None
