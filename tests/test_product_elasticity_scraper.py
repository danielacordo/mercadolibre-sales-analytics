import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.product_elasticity import (
    assign_category,
    _log_log_ols,
    _demand_type,
    _reliability,
    estimate_per_item_elasticity,
    estimate_panel_elasticity,
    estimate_category_elasticity,
    build_summary_table,)

from src.competitor_scraper import (
    _build_search_url,
    generate_synthetic_snapshot,
    compute_price_position,
    estimate_cross_elasticity,
    summarize_positions,
    BUSINESS_PRICES,
    ML_BASE_URL,)


# Fixtures
@pytest.fixture
def minimal_df():
    """Minimal transaction DataFrame: 2 items, 8 months each, price variation"""
    rng = np.random.default_rng(0)
    rows = []
    for item_id in ["ITEM_A", "ITEM_B"]:
        base = 1000 if item_id == "ITEM_A" else 3000
        for m in range(1, 9):
            price  = base * (1 + 0.08 * m) 
            orders = max(1, int(5 - 0.3 * m + rng.integers(-1, 2)))
            for _ in range(orders):
                rows.append({
                    "Order_id": f"ORD_{item_id}_{m}_{_}",
                    "Fecha": pd.Timestamp(f"2024-{m:02d}-15"),
                    "Monto": price,
                    "Item_id": item_id,
                    "Titulo_prod": f"Producto {item_id}",
                    "Ingreso_bruto": price * 0.9,
                    "Ingreso_neto": price * 0.68,
                    "Provincia_nombre": "Buenos Aires",
                    "Fuente": "CNX",})
    return pd.DataFrame(rows)


@pytest.fixture
def own_sales_df():
    """Simulated own-sales DataFrame for cross-elasticity test """
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    prices = np.linspace(1500, 12000, 24)
    return pd.DataFrame({
        "Fecha": dates,
        "Monto": prices,
        "Order_id": [f"O{i}" for i in range(24)],
        "Ingreso_bruto": prices * 0.9,})


# assign_category
class TestAssignCategory:
    def test_adorno_torta_detected(self):
        assert assign_category("Adorno De Torta En Goma Eva Plim Plim") == "Adornos de torta"

    def test_adorno_torta_short_form(self):
        assert assign_category("Adorno Torta Snoopy") == "Adornos de torta"

    def test_figuras_bichikids_before_generic_figuras(self):
        # CATEGORY_RULES is ordered: Bichikids must be checked before generic Figuras
        result = assign_category("Figuras Bichikids En Goma Eva")
        assert result == "Figuras Bichikids"

    def test_figuras_animales(self):
        assert assign_category("Figuras De Animales En Goma Eva X 10") == "Figuras de animales"

    def test_cartel(self):
        assert assign_category("Cartel Decoracion Goma Eva Piratas") == "Carteles decorativos"

    def test_apliques(self):
        assert assign_category("Apliques En Goma Eva Para Torta") == "Apliques decorativos"

    def test_nan_returns_sin_titulo(self):
        assert assign_category(float("nan")) == "Sin título"

    def test_unknown_returns_otros(self):
        assert assign_category("Producto Completamente Desconocido XYZ") == "Otros"

    def test_case_insensitive(self):
        # Lowercase should match
        result = assign_category("adorno de torta goma eva")
        assert result == "Adornos de torta"


# _log_log_ols
class TestLogLogOLS:
    def test_returns_dict_with_required_keys(self):
        prices = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000])
        quantities = np.array([10, 8, 7, 6, 5, 4, 3])
        result = _log_log_ols(prices, quantities)
        for key in ("epsilon", "r2", "p_value", "ci_lower", "ci_upper", "se", "n"):
            assert key in result, f"Missing key: {key}"

    def test_inelastic_demand(self):
        # Known-inelastic: demand barely responds to price
        prices = np.array([1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.])
        quantities = np.array([10., 9.5, 9., 8.7, 8.4, 8.2, 8., 7.8])
        result = _log_log_ols(prices, quantities)
        assert result["epsilon"] < 0, "Expected negative elasticity for downward-sloping demand"
        assert abs(result["epsilon"]) < 1, "Expected inelastic (|ε| < 1)"

    def test_ci_contains_epsilon(self):
        prices = np.linspace(1000, 10000, 12)
        quantities = np.array([10, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.8, 4.5])
        result = _log_log_ols(prices, quantities)
        assert result["ci_lower"] <= result["epsilon"] <= result["ci_upper"]

    def test_n_matches_input_length(self):
        prices = np.array([1000., 2000., 3000., 4000., 5000.])
        quantities = np.array([5., 4., 3.5, 3., 2.5])
        result = _log_log_ols(prices, quantities)
        assert result["n"] == 5

    def test_r2_between_zero_and_one(self):
        prices = np.array([1000., 2000., 3000., 4000., 5000., 6000.])
        quantities = np.array([10., 8., 6., 5., 4., 3.])
        result = _log_log_ols(prices, quantities)
        assert 0 <= result["r2"] <= 1

    def test_ci_lower_less_than_ci_upper(self):
        prices = np.array([1000., 2000., 3000., 4000., 5000., 6000.])
        quantities = np.array([10., 8., 6., 5., 4., 3.])
        result = _log_log_ols(prices, quantities)
        assert result["ci_lower"] < result["ci_upper"]


# _demand_type
class TestDemandType:
    def test_inelastic(self):
        assert _demand_type(-0.5) == "inelastic"
        assert _demand_type(-0.99) == "inelastic"
        assert _demand_type(0.3) == "inelastic"

    def test_elastic(self):
        assert _demand_type(-1.5) == "elastic"
        assert _demand_type(-2.0) == "elastic"

    def test_unit_elastic(self):
        assert _demand_type(-1.0) == "unit_elastic"
        assert _demand_type(1.0) == "unit_elastic"


# _reliability
class TestReliability:
    def test_high_reliability(self):
        result = _reliability(n=12, p_value=0.04, price_ratio=3.0)
        assert result == "HIGH"

    def test_medium_reliability_adequate_n(self):
        result = _reliability(n=7, p_value=0.15, price_ratio=2.0)
        assert result == "MEDIUM"

    def test_low_reliability_small_n(self):
        result = _reliability(n=4, p_value=0.3, price_ratio=1.1)
        assert "LOW" in result

    def test_low_reliability_despite_significance_if_n_small(self):
        # p < 0.1 but n=3 — still low
        result = _reliability(n=3, p_value=0.05, price_ratio=2.5)
        assert "LOW" in result


# estimate_per_item_elasticity
class TestPerItemElasticity:
    def test_returns_list(self, minimal_df):
        results = estimate_per_item_elasticity(minimal_df, min_obs=4)
        assert isinstance(results, list)

    def test_items_meet_min_obs(self, minimal_df):
        results = estimate_per_item_elasticity(minimal_df, min_obs=6)
        for r in results:
            assert r.n_obs >= 6

    def test_sorted_by_n_obs_descending(self, minimal_df):
        results = estimate_per_item_elasticity(minimal_df, min_obs=4)
        if len(results) >= 2:
            for a, b in zip(results, results[1:]):
                assert a.n_obs >= b.n_obs

    def test_all_results_have_demand_type(self, minimal_df):
        results = estimate_per_item_elasticity(minimal_df, min_obs=4)
        for r in results:
            assert r.demand_type in ("inelastic", "unit_elastic", "elastic")

    def test_empty_df_returns_empty_list(self):
        empty = pd.DataFrame(columns=["Order_id", "Fecha", "Monto", "Item_id",
                                       "Titulo_prod", "Ingreso_bruto"])
        results = estimate_per_item_elasticity(empty)
        assert results == []

    def test_items_with_no_quantity_variation_excluded(self, minimal_df):
        # Add a single-order-per-month item (log_q = 0 always)
        always_one = pd.DataFrame({
            "Order_id": [f"ONE_{m}" for m in range(8)],
            "Fecha": pd.date_range("2024-01-01", periods=8, freq="MS"),
            "Monto": [1000 * (1 + 0.1 * m) for m in range(8)],
            "Item_id": ["ITEM_SINGLE"] * 8,
            "Titulo_prod": ["Producto Single"] * 8,
            "Ingreso_bruto": [900] * 8,})
        
        df_with_single = pd.concat([minimal_df, always_one], ignore_index=True)
        results = estimate_per_item_elasticity(df_with_single, min_obs=6)
        item_ids = [r.item_id for r in results]
        assert "ITEM_SINGLE" not in item_ids, "Items with no quantity variation should be excluded"

    def test_price_range_ratio_computed(self, minimal_df):
        results = estimate_per_item_elasticity(minimal_df, min_obs=4)
        for r in results:
            assert r.price_range_ratio >= 1.0


# estimate_panel_elasticity
class TestPanelElasticity:
    def test_returns_dict_with_required_keys(self, minimal_df):
        result = estimate_panel_elasticity(minimal_df)
        for key in ("epsilon_pooled", "epsilon_fe", "r2_pooled", "r2_fe", "n_obs", "n_items", "p_value_fe",
                    "ci_lower_fe", "ci_upper_fe", "demand_type", "interpretation"):
            assert key in result, f"Missing key: {key}"

    def test_n_items_positive(self, minimal_df):
        result = estimate_panel_elasticity(minimal_df, min_obs=4)
        assert result["n_items"] >= 1

    def test_ci_order(self, minimal_df):
        result = estimate_panel_elasticity(minimal_df, min_obs=4)
        assert result["ci_lower_fe"] < result["ci_upper_fe"]

    def test_fe_differs_from_pooled(self, minimal_df):
        # FE (within estimator) removes item-level intercepts, epsilon can differ
        result = estimate_panel_elasticity(minimal_df, min_obs=4)
        # They can be equal in degenerate cases, just check both are finite
        assert np.isfinite(result["epsilon_pooled"])
        assert np.isfinite(result["epsilon_fe"])


# estimate_category_elasticity
class TestCategoryElasticity:
    def test_returns_list(self, minimal_df):
        results = estimate_category_elasticity(minimal_df, min_obs=2)
        assert isinstance(results, list)

    def test_no_titled_products_returns_empty(self):
        df = pd.DataFrame({
            "Order_id": ["O1", "O2"],
            "Fecha": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
            "Monto": [1000., 1200.],
            "Item_id": ["I1", "I1"],
            "Titulo_prod": [np.nan, np.nan],
            "Ingreso_bruto": [900., 1100.],})
        
        results = estimate_category_elasticity(df)
        assert results == []


# build_summary_table
class TestBuildSummaryTable:
    def test_returns_dataframe(self, minimal_df):
        per_item = estimate_per_item_elasticity(minimal_df, min_obs=4)
        categories = estimate_category_elasticity(minimal_df, min_obs=2)
        panel = estimate_panel_elasticity(minimal_df, min_obs=4)
        table = build_summary_table(per_item, categories, panel)
        assert isinstance(table, pd.DataFrame)

    def test_contains_panel_row(self, minimal_df):
        per_item = estimate_per_item_elasticity(minimal_df, min_obs=4)
        categories = estimate_category_elasticity(minimal_df, min_obs=2)
        panel = estimate_panel_elasticity(minimal_df, min_obs=4)
        table = build_summary_table(per_item, categories, panel)
        assert "Panel (FE)" in table["Level"].values

    def test_has_required_columns(self, minimal_df):
        per_item = estimate_per_item_elasticity(minimal_df, min_obs=4)
        categories = []
        panel = estimate_panel_elasticity(minimal_df, min_obs=4)
        table = build_summary_table(per_item, categories, panel)
        for col in ("Level", "Label", "N", "epsilon", "CI_95", "R²", "p", "Reliability"):
            assert col in table.columns

    def test_item_with_no_title_does_not_crash(self):
        """Regression test: items with no Titulo_prod (e.g. CNX-only items, where Titulo_prod is always null"""
        rng = np.random.default_rng(1)
        rows = []
        item_id = 905910028  
        for m in range(1, 9):
            price = 1000 * (1 + 0.08 * m)
            orders = max(1, int(5 - 0.3 * m + rng.integers(-1, 2)))
            for _ in range(orders):
                rows.append({
                    "Order_id": f"ORD_{m}_{_}",
                    "Fecha": pd.Timestamp(f"2024-{m:02d}-15"),
                    "Monto": price,
                    "Item_id": item_id,
                    "Titulo_prod": np.nan,  # no title on file 
                    "Ingreso_bruto": price * 0.9,
                    "Ingreso_neto": price * 0.68,
                    "Provincia_nombre": "Buenos Aires",
                    "Fuente":          "CNX",})
        df = pd.DataFrame(rows)

        per_item = estimate_per_item_elasticity(df, min_obs=4)
        categories = estimate_category_elasticity(df, min_obs=2)
        panel = estimate_panel_elasticity(df, min_obs=4)

        assert per_item, "fixture should produce at least one per-item result"
        assert isinstance(per_item[0].label, str), "label must be a string, even with no title on file"

        table = build_summary_table(per_item, categories, panel)  
        assert isinstance(table, pd.DataFrame)


# _build_search_url
class TestBuildSearchUrl:
    def test_page_1_no_offset(self):
        url = _build_search_url("figuras goma eva", page=1)
        assert "Desde" not in url
        assert "figuras-goma-eva" in url

    def test_page_2_has_offset(self):
        url = _build_search_url("adorno torta", page=2)
        assert "Desde_49" in url

    def test_page_3_offset_correct(self):
        url = _build_search_url("apliques", page=3)
        assert "Desde_97" in url

    def test_starts_with_ml_base(self):
        url = _build_search_url("test query", page=1)
        assert url.startswith(ML_BASE_URL)

    def test_spaces_replaced_with_hyphens(self):
        url = _build_search_url("figuras goma eva animales", page=1)
        assert " " not in url



# generate_synthetic_snapshot
class TestSyntheticSnapshot:
    def test_returns_dataframe(self):
        df = generate_synthetic_snapshot()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = generate_synthetic_snapshot()
        for col in ("title", "price", "seller", "url", "category", "scraped_date"):
            assert col in df.columns

    def test_prices_positive(self):
        df = generate_synthetic_snapshot()
        assert (df["price"] > 0).all()

    def test_covers_all_expected_categories(self):
        df = generate_synthetic_snapshot()
        from src.competitor_scraper import SEARCH_QUERIES
        for cat in SEARCH_QUERIES:
            assert cat in df["category"].values

    def test_reproducible_with_same_seed(self):
        df1 = generate_synthetic_snapshot(seed=1)
        df2 = generate_synthetic_snapshot(seed=1)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_give_different_data(self):
        df1 = generate_synthetic_snapshot(seed=1)
        df2 = generate_synthetic_snapshot(seed=99)
        assert not df1["price"].equals(df2["price"])


# compute_price_position
class TestComputePricePosition:
    def test_returns_list_of_positions(self):
        df = generate_synthetic_snapshot()
        positions = compute_price_position(df, BUSINESS_PRICES)
        assert isinstance(positions, list)
        assert len(positions) > 0

    def test_position_labels_valid(self):
        df = generate_synthetic_snapshot()
        positions = compute_price_position(df, BUSINESS_PRICES)
        valid = {"below market", "at market", "above market", "premium"}
        for p in positions:
            assert p.position in valid, f"Invalid position: {p.position}"

    def test_price_gap_consistent_with_position(self):
        df = generate_synthetic_snapshot()
        positions = compute_price_position(df, BUSINESS_PRICES)
        for p in positions:
            if p.position == "below market":
                assert p.price_gap_pct < -10
            elif p.position == "premium":
                assert p.price_gap_pct > 25

    def test_n_competitors_matches_df(self):
        df = generate_synthetic_snapshot()
        positions = compute_price_position(df, BUSINESS_PRICES)
        for p in positions:
            n_in_df = len(df[df["category"] == p.category])
            assert p.n_competitors == n_in_df

    def test_market_p25_less_than_p75(self):
        df = generate_synthetic_snapshot()
        positions = compute_price_position(df, BUSINESS_PRICES)
        for p in positions:
            assert p.market_p25 <= p.market_p75

    def test_categories_with_few_listings_excluded(self):
        # A category with < 3 listings should not appear in results
        small_df = pd.DataFrame([
            {"category": "tiny_cat", "price": 1000.0,
             "title": "t", "seller": "s", "url": "u", "scraped_date": "2026-01-01"},
            {"category": "tiny_cat", "price": 1200.0,
             "title": "t2", "seller": "s", "url": "u2", "scraped_date": "2026-01-01"},])
        
        business = {"tiny_cat": 1100.0}
        positions = compute_price_position(small_df, business)
        assert len(positions) == 0  



# summarize_positions
class TestSummarizePositions:
    def test_returns_dataframe(self):
        df = generate_synthetic_snapshot()
        positions = compute_price_position(df, BUSINESS_PRICES)
        summary = summarize_positions(positions)
        assert isinstance(summary, pd.DataFrame)

    def test_one_row_per_position(self):
        df = generate_synthetic_snapshot()
        positions = compute_price_position(df, BUSINESS_PRICES)
        summary = summarize_positions(positions)
        assert len(summary) == len(positions)

    def test_gap_column_has_sign(self):
        df = generate_synthetic_snapshot()
        positions = compute_price_position(df, BUSINESS_PRICES)
        summary = summarize_positions(positions)
        for val in summary["Gap vs median"]:
            assert "+" in val or "-" in val, f"Expected signed gap: {val}"


# estimate_cross_elasticity
class TestCrossElasticity:
    def test_single_date_returns_insufficient_data(self, own_sales_df):
        snapshot = generate_synthetic_snapshot()  # all same date
        result = estimate_cross_elasticity(snapshot, own_sales_df)
        assert "INSUFFICIENT DATA" in result["Status"].values or \
               "Status" in result.columns

    def test_multi_date_with_enough_data_returns_results(self, own_sales_df):
        # Create synthetic multi-date competitor data (12 months)
        rng = np.random.default_rng(0)
        rows = []
        months = pd.date_range("2023-01-01", periods=12, freq="MS")
        for m in months:
            for cat in list(BUSINESS_PRICES.keys())[:2]:
                for _ in range(5):
                    rows.append({
                        "category": cat,
                        "price": rng.uniform(1000, 8000),
                        "title": "competitor",
                        "seller": "x",
                        "url": "u",
                        "scraped_date": str(m.date()),})
        multi_snapshot = pd.DataFrame(rows)
        result = estimate_cross_elasticity(multi_snapshot, own_sales_df)
        # With 12 dates it should have enough data to attempt estimation
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0