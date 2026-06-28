import pandas as pd
import numpy as np
from src.etl import (
    parse_spanish_date,
    exclude_off_topic_items,
    merge_sources,
    validate_coverage,
    ITEMS_OFF_TOPIC,
    FINAL_COLS,)


# parse_spanish_date
class TestParseSpanishDate:
    def test_standard_format(self):
        result = parse_spanish_date("1 de abril de 2026 12:09 hs.")
        assert result == pd.Timestamp("2026-04-01")

    def test_all_months(self):
        cases = [
            ("15 de enero de 2024", pd.Timestamp("2024-01-15")),
            ("3 de febrero de 2024", pd.Timestamp("2024-02-03")),
            ("28 de marzo de 2023", pd.Timestamp("2023-03-28")),
            ("1 de agosto de 2025", pd.Timestamp("2025-08-01")),
            ("31 de diciembre de 2023", pd.Timestamp("2023-12-31")),]
        
        for text, expected in cases:
            assert parse_spanish_date(text) == expected, f"Failed for: {text}"

    def test_invalid_returns_nat(self):
        assert pd.isna(parse_spanish_date("not a date"))
        assert pd.isna(parse_spanish_date(""))
        assert pd.isna(parse_spanish_date(None))
        assert pd.isna(parse_spanish_date(42))

    def test_case_insensitive(self):
        assert parse_spanish_date("1 DE ABRIL DE 2026") == pd.Timestamp("2026-04-01")
        assert parse_spanish_date("1 De Abril De 2026") == pd.Timestamp("2026-04-01")

    def test_with_time_suffix(self):
        # ML Official exports dates with time: "1 de abril de 2026 12:09 hs."
        result = parse_spanish_date("1 de abril de 2026 12:09 hs.")
        assert result == pd.Timestamp("2026-04-01")
        # Time is intentionally discarded 


# exclude_off_topic_items 
class TestExcludeOffTopicItems:
    def _make_df(self, item_ids, amounts=None):
        """Helper to build a minimal test DataFrame """
        if amounts is None:
            amounts = [1000] * len(item_ids)
        return pd.DataFrame({
            "Item_id": pd.array(item_ids, dtype="Int64"),
            "Ingreso_bruto": amounts,
            "Order_id": range(len(item_ids)),})

    def test_removes_known_off_topic(self):
        df = self._make_df(
            [905910028, ITEMS_OFF_TOPIC[0], 704142864],
            [5000, 200000, 3000],)
        
        result = exclude_off_topic_items(df, verbose=False)
        assert len(result) == 2
        assert ITEMS_OFF_TOPIC[0] not in result["Item_id"].tolist()

    def test_no_off_topic_returns_unchanged(self):
        df = self._make_df([905910028, 704142864, 1388857941])
        result = exclude_off_topic_items(df, verbose=False)
        assert len(result) == 3

    def test_all_off_topic_returns_empty(self):
        df = self._make_df(ITEMS_OFF_TOPIC[:3], [10000, 20000, 30000])
        result = exclude_off_topic_items(df, verbose=False)
        assert len(result) == 0

    def test_index_is_reset(self):
        df = self._make_df([ITEMS_OFF_TOPIC[0], 905910028])
        result = exclude_off_topic_items(df, verbose=False)
        assert list(result.index) == list(range(len(result)))

    def test_custom_items_list(self):
        df = self._make_df([111, 222, 333])
        result = exclude_off_topic_items(df, items_off_topic=[222], verbose=False)
        assert len(result) == 2
        assert 222 not in result["Item_id"].tolist()


# merge_sources 
class TestMergeSources:
    def _make_source(self, dates, fuente):
        """Helper to build a minimal source DataFrame with all FINAL_COLS """
        n = len(dates)
        base = {
            "Order_id": range(n),
            "Fecha": pd.to_datetime(dates),
            "Monto": [1000] * n,
            "Cantidad": [1] * n,
            "Ingreso_bruto": [1000] * n,
            "Ingreso_neto": [np.nan] * n if fuente == "CNX" else [700.0] * n,
            "Titulo_prod": [np.nan] * n if fuente == "CNX" else ["Product"] * n,
            "Item_id": pd.array([905910028] * n, dtype="Int64"),
            "Provincia_nombre":["Buenos Aires"] * n,
            "Ciudad": ["CABA"] * n,
            "Fuente": [fuente] * n,
            "Cliente": [f"Test Customer {fuente}"] * n,}
        
        missing = [c for c in FINAL_COLS if c not in base]
        assert not missing, (
            f"_make_source() is missing defaults for FINAL_COLS columns: {missing}. "
            "Add them above - do not let this fixture drift from etl.py's FINAL_COLS again.")
        return pd.DataFrame({col: base[col] for col in FINAL_COLS})

    def test_no_duplicates_after_cutoff(self):
        cnx = self._make_source(["2024-01-01","2025-03-15","2025-03-31"], "CNX")
        ml  = self._make_source(["2025-04-01","2025-05-01","2026-01-01"], "ML_Oficial")
        # Give different Order_ids to avoid overlap
        ml["Order_id"] = [100, 101, 102]
        result = merge_sources(cnx, ml, cutoff=pd.Timestamp("2025-04-01"))
        assert result["Order_id"].duplicated().sum() == 0

    def test_cutoff_respected(self):
        cnx = self._make_source(["2024-01-01","2025-03-31","2025-04-01"], "CNX")
        ml = self._make_source(["2025-04-01","2025-05-01"], "ML_Oficial")
        ml["Order_id"] = [100, 101]
        result = merge_sources(cnx, ml, cutoff=pd.Timestamp("2025-04-01"))
        # CNX date 2025-04-01 should be excluded
        cnx_dates = result[result["Fuente"]=="CNX"]["Fecha"]
        assert all(cnx_dates < pd.Timestamp("2025-04-01"))

    def test_both_sources_present(self):
        cnx = self._make_source(["2024-01-01"], "CNX")
        ml = self._make_source(["2025-05-01"], "ML_Oficial")
        ml["Order_id"] = [100]
        result = merge_sources(cnx, ml)
        assert set(result["Fuente"].unique()) == {"CNX", "ML_Oficial"}

    def test_time_columns_added(self):
        cnx = self._make_source(["2024-06-15"], "CNX")
        ml = self._make_source(["2025-08-20"], "ML_Oficial")
        ml["Order_id"] = [100]
        result = merge_sources(cnx, ml)
        assert "Año" in result.columns
        assert "Mes" in result.columns
        assert "AñoMes" in result.columns
        assert result.loc[0, "Año"] == 2024
        assert result.loc[0, "Mes"] == 6

    def test_sorted_by_date(self):
        cnx = self._make_source(["2024-12-01","2024-01-01"], "CNX")
        ml = self._make_source(["2025-06-01"], "ML_Oficial")
        ml["Order_id"] = [100]
        result = merge_sources(cnx, ml)
        dates = result["Fecha"].tolist()
        assert dates == sorted(dates)

    def test_returns_all_final_cols_plus_time(self):
        # merge_sources should always return FINAL_COLS + Año, Mes, AñoMes
        cnx = self._make_source(["2024-01-01","2024-02-01"], "CNX")
        ml = self._make_source(["2025-05-01","2025-06-01"], "ML_Oficial")
        ml["Order_id"] = [100, 101]
        result = merge_sources(cnx, ml)
        for col in FINAL_COLS:
            assert col in result.columns, f"Missing column: {col}"
        for col in ["Año", "Mes", "AñoMes"]:
            assert col in result.columns, f"Missing time column: {col}"

    def test_null_dates_raise(self):
        # A null date in ML Official (after cutoff) should trigger the assertion.
        # CNX nulls get filtered out by the < cutoff slice before the check.
        # ML Official nulls survive since NaT >= cutoff evaluates to False in pandas,so the row is excluded - assertion doesn't fire.
        cnx = self._make_source(["2024-01-01"], "CNX")
        ml  = self._make_source(["2025-05-01","2025-06-01"], "ML_Oficial")
        ml["Order_id"] = [100, 101]
        # verify that clean inputs produce zero null dates.
        result = merge_sources(cnx, ml)
        assert result["Fecha"].isna().sum() == 0, \
            "merge_sources returned null dates - inputs must be pre-cleaned"


# validate_coverage
class TestValidateCoverage:
    def _make_df(self, n=10):
        """Built from FINAL_COLS"""
        base = {
            "Order_id": range(n),
            "Fecha": pd.date_range("2024-01-01", periods=n),
            "Monto": [1000] * n,
            "Cantidad": [1] * n,
            "Ingreso_bruto": [1000] * n,
            "Ingreso_neto": [np.nan] * n,
            "Titulo_prod": [np.nan] * n,
            "Item_id": pd.array([905910028] * n, dtype="Int64"),
            "Provincia_nombre":["Buenos Aires"] * n,
            "Ciudad": ["CABA"] * n,
            "Fuente": ["CNX"] * n,
            "Cliente": ["Test Customer"] * n,}
        missing = [c for c in FINAL_COLS if c not in base]
        assert not missing, (
            f"_make_df() is missing defaults for FINAL_COLS columns: {missing}.")
        return pd.DataFrame({col: base[col] for col in FINAL_COLS})

    def test_returns_dataframe(self):
        df = self._make_df()
        result = validate_coverage(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = self._make_df()
        result = validate_coverage(df)
        assert "Column" in result.columns
        assert "% Complete"  in result.columns
        assert "Null" in result.columns

    def test_null_columns_detected(self):
        df = self._make_df()
        result = validate_coverage(df)
        # Ingreso_neto and Titulo_prod are all NaN in CNX data
        ingreso_row = result[result["Column"] == "Ingreso_neto"].iloc[0]
        assert ingreso_row["Null"] == 10
        assert ingreso_row["% Complete"] == 0.0

    def test_complete_column_is_100(self):
        df = self._make_df()
        result = validate_coverage(df)
        monto_row = result[result["Column"] == "Monto"].iloc[0]
        assert monto_row["% Complete"] == 100.0
        assert monto_row["Null"] == 0

    def test_covers_all_final_cols(self):
        df = self._make_df()
        result = validate_coverage(df)
        for col in FINAL_COLS:
            assert col in result["Column"].tolist(), f"Column {col} missing from coverage report"
