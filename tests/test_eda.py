import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from src.eda import (
    annual_summary,
    monthly_summary,
    geo_summary,
    margin_summary,
    plot_revenue_monthly,
    plot_seasonality,
    plot_geography,)


# Fixtures 
def make_df(n_per_year: int = 10) -> pd.DataFrame:
    """ Build a minimal clean transaction DataFrame for testing """
    records = []
    for year in [2023, 2024, 2025]:
        for i in range(n_per_year):
            month = (i % 12) + 1
            records.append({
                "Order_id": year * 100 + i,
                "Fecha": pd.Timestamp(year, month, 1),
                "Monto": 1000 + year * 100 + i * 50,
                "Cantidad": 1,
                "Ingreso_bruto": 1000 + year * 100 + i * 50,
                "Ingreso_neto": 700  + year * 50  + i * 30
                                   if year >= 2025 else np.nan,
                "Titulo_prod": "Bichikids X7" if year >= 2025 else np.nan,
                "Item_id": 905910028,
                "Provincia_nombre":"Buenos Aires" if i < 7 else "Córdoba",
                "Ciudad": "CABA",
                "Fuente": "CNX" if year < 2025 else "ML_Oficial",
            })
    df = pd.DataFrame(records)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


# annual_summary 
class TestAnnualSummary:
    def test_returns_dataframe(self):
        df = make_df()
        result = annual_summary(df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_year(self):
        df = make_df()
        result = annual_summary(df)
        assert len(result) == 3  # 2023, 2024, 2025
        assert set(result.index) == {2023, 2024, 2025}

    def test_orders_column_correct(self):
        df = make_df(n_per_year=5)
        result = annual_summary(df)
        assert result.loc[2023, "orders"] == 5
        assert result.loc[2024, "orders"] == 5

    def test_revenue_growth_first_year_is_nan(self):
        df = make_df()
        result = annual_summary(df)
        assert pd.isna(result["revenue_growth"].iloc[0])

    def test_revenue_growth_computed_for_non_first_rows(self):
        # Revenue growth is NaN for the first year and a number for subsequent years
        df = make_df()
        result = annual_summary(df)
        assert pd.isna(result["revenue_growth"].iloc[0])
        # All subsequent years should have a numeric (not NaN) growth value
        assert result["revenue_growth"].iloc[1:].notna().all()

    def test_required_columns_present(self):
        df = make_df()
        result = annual_summary(df)
        for col in ["orders", "revenue", "avg_ticket", "median_ticket"]:
            assert col in result.columns, f"Missing column: {col}"


# monthly_summary 
class TestMonthlySummary:
    def test_returns_dataframe(self):
        df = make_df()
        result = monthly_summary(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_ds_column(self):
        df = make_df()
        result = monthly_summary(df)
        assert "ds" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["ds"])

    def test_has_price_index(self):
        df = make_df()
        result = monthly_summary(df)
        assert "price_index" in result.columns
        # First month should be base = 100.0
        assert result["price_index"].iloc[0] == pytest.approx(100.0, abs=0.1)

    def test_year_column(self):
        df = make_df()
        result = monthly_summary(df)
        assert "year" in result.columns
        assert set(result["year"].unique()).issubset({2023, 2024, 2025})

    def test_orders_sum_matches_total(self):
        df = make_df()
        result = monthly_summary(df)
        assert result["orders"].sum() == len(df)


# geo_summary 
class TestGeoSummary:
    def test_returns_dataframe(self):
        df = make_df()
        result = geo_summary(df)
        assert isinstance(result, pd.DataFrame)

    def test_excludes_null_provinces(self):
        df = make_df()
        df.loc[0, "Provincia_nombre"] = np.nan
        result = geo_summary(df)
        assert result["Provincia_nombre"].isna().sum() == 0

    def test_sorted_by_revenue_desc(self):
        df = make_df()
        result = geo_summary(df)
        revenues = result["revenue"].tolist()
        assert revenues == sorted(revenues, reverse=True)

    def test_pct_columns_present(self):
        df = make_df()
        result = geo_summary(df)
        assert "pct_orders" in result.columns
        assert "pct_revenue" in result.columns

    def test_top_n_respected(self):
        df = make_df()
        # The test data has 2 provinces
        result = geo_summary(df, top_n=1)
        assert len(result) == 1

    def test_pct_orders_sums_to_100(self):
        df = make_df()
        result = geo_summary(df)
        assert result["pct_orders"].sum() == pytest.approx(100.0, abs=0.1)


# margin_summary 
class TestMarginSummary:
    def test_only_ml_oficial_rows(self):
        df = make_df()
        result = margin_summary(df)
        assert (result["Fuente"] == "ML_Oficial").all()

    def test_pct_net_column_exists(self):
        df = make_df()
        result = margin_summary(df)
        assert "pct_net" in result.columns

    def test_pct_net_between_0_and_100(self):
        df = make_df()
        result = margin_summary(df)
        assert (result["pct_net"] >= 0).all()
        assert (result["pct_net"] <= 100).all()

    def test_raises_if_no_ml_data(self):
        df = make_df()
        df["Fuente"] = "CNX"  # remove all ML rows
        df["Ingreso_neto"] = np.nan
        with pytest.raises(AssertionError, match="No ML Official rows"):
            margin_summary(df)


# chart functions 
class TestChartFunctions:
    """Smoke tests, verify charts return a Figure and don't raise"""

    def setup_method(self):
        plt.close("all")
        self.df = make_df(n_per_year=12)

    def test_plot_revenue_monthly_returns_figure(self):
        fig = plot_revenue_monthly(self.df)
        assert isinstance(fig, plt.Figure)

    def test_plot_seasonality_returns_figure(self):
        fig = plot_seasonality(self.df)
        assert isinstance(fig, plt.Figure)

    def test_plot_geography_returns_figure(self):
        fig = plot_geography(self.df)
        assert isinstance(fig, plt.Figure)

    def test_charts_dont_raise_with_minimal_data(self):
        # Single year, few rows — should not crash
        small_df = make_df(n_per_year=2)
        for fn in [plot_revenue_monthly, plot_seasonality, plot_geography]:
            plt.close("all")
            fig = fn(small_df)
            assert fig is not None

    def teardown_method(self):
        plt.close("all")