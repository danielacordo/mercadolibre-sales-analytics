import pandas as pd
import numpy as np
import re
import os

PROV_MAP: dict[str, str] = {
    "AR-B": "Buenos Aires", "AR-C": "Capital Federal",
    "AR-X": "Córdoba", "AR-S": "Santa Fe",
    "AR-E": "Entre Ríos", "AR-N": "Misiones",
    "AR-Q": "Neuquén", "AR-D": "San Luis",
    "AR-M": "Mendoza", "AR-U": "Chubut",
    "AR-W": "Corrientes", "AR-P": "Formosa",
    "AR-T": "Tucumán", "AR-K": "Catamarca",
}

MESES_ES: dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

# Items confirmed as unrelated to the decoration business
ITEMS_OFF_TOPIC: list[int] = [
    1636648857,  # Gucci perfume - confirmed by product title
    1699566940,  # $149,000 ARS - no title, discontinued ID
    1699553874,  # $25,900  ARS - no title, discontinued ID
    1381829061,  # $20,000  ARS - no title (Sep 2023)
    1381899765,  # $20,000  ARS - no title, same day as above
    1423780045,  # $19,493  ARS - no title, discontinued ID
    1699402488,  # $15,000  ARS - no title, discontinued ID
]

FINAL_COLS: list[str] = [
    "Order_id", "Fecha", "Monto", "Cantidad",
    "Ingreso_bruto", "Ingreso_neto",
    "Titulo_prod", "Item_id",
    "Provincia_nombre", "Ciudad", "Fuente", "Cliente",
]


# Parsing
def parse_spanish_date(text: str) -> pd.Timestamp:
    """ Converts a Spanish-format date string to a Timestamp

    MercadoLibre's official report exports dates as: "1 de abril de 2026 12:09 hs", not parseable by pandas directly """
    if not isinstance(text, str):
        return pd.NaT
    match = re.search(r"(\d+) de (\w+) de (\d{4})", text.lower())
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        year = int(match.group(3))
        month = MESES_ES.get(month_str, 0)
        if month:
            return pd.Timestamp(year, month, day)
    return pd.NaT

# Extract
def load_cnx(path: str) -> pd.DataFrame:
    """ Loads and cleans the CNX file (MercadoLibre BigQuery export)

    The file contains metadata in the first 6 rows (query parameters)
    Actual transaction data starts at row 7 (index 6 in Python) """
    raw = pd.read_excel(path, sheet_name="Respuesta - Reporte Ventas", header=None)
    assert len(raw) > 10, f"CNX file too short: {len(raw)} rows"

    df = raw.iloc[6:].copy().reset_index(drop=True)
    df.columns = [
        "Seller", "Apodo_comprador", "Fecha", "Order_id",
        "Item_id", "Monto", "Cantidad", "Provincia", "Ciudad", "MODE"
    ]

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Monto"] = pd.to_numeric(df["Monto"], errors="coerce")
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce")
    df["Item_id"] = pd.to_numeric(df["Item_id"], errors="coerce").astype("Int64")
    df["Order_id"] = pd.to_numeric(df["Order_id"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["Fecha", "Monto"]).reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped > 0:
        print(f"  [CNX] {dropped} rows dropped (missing date or amount)")

    df["Ingreso_bruto"] = df["Monto"] * df["Cantidad"]
    df["Provincia_nombre"] = df["Provincia"].map(PROV_MAP)
    df["Ingreso_neto"] = np.nan
    df["Titulo_prod"] = np.nan
    df["Fuente"] = "CNX"
    df["Cliente"] = df["Apodo_comprador"]

    assert len(df) > 0, "CNX DataFrame is empty after cleaning"
    assert df["Monto"].min() > 0, "CNX contains negative or zero amounts"
    assert df["Fecha"].isna().sum() == 0, "CNX has null dates after cleaning"

    return df


def load_ml_oficial(path: str) -> pd.DataFrame:
    """ Loads and cleans the official MercadoLibre seller report

    Report structure:
    - Rows 1-4: report metadata
    - Row 5: section headers (Sales, Advertising, Shipping...)
    - Row 6: column sub-headers
    - Row 7+: data

    Columns are mapped by position (known structure of ML export format)
    Column positions:
    col 0 = Order ID
    col 1 = Sale date (Spanish text)
    col 2 = Order status
    col 6 = Units sold
    col 7 = Gross product revenue
    col 18 = Net total (after all ML fees)
    col 22 = Publication ID (Item_id, with MLA prefix)
    col 24 = Product title
    col 26 = Unit sale price
    col 33 = Buyer name
    col 36 = Buyer city
    col 37 = Buyer province/state """
    raw = pd.read_excel(path, sheet_name="Ventas AR", header=None)
    assert len(raw) > 10, f"ML Official file too short: {len(raw)} rows"

    df = raw.iloc[6:].copy().reset_index(drop=True)
    df.columns = range(len(df.columns))

    df["Fecha"] = df[1].apply(parse_spanish_date)
    df["Order_id"] = pd.to_numeric(df[0], errors="coerce")
    df["Estado"] = df[2].astype(str)
    df["Ingreso_bruto"] = pd.to_numeric(df[7],  errors="coerce")
    df["Ingreso_neto"] = pd.to_numeric(df[18], errors="coerce")
    df["Cantidad"] = pd.to_numeric(df[6],  errors="coerce").fillna(1)
    df["Titulo_prod"] = df[24].astype(str).str.strip()
    df["Item_id"] = pd.to_numeric(
                     df[22].astype(str).str.replace("MLA", "", regex=False),
                    errors="coerce"
                    ).astype("Int64")
    df["Monto"] = pd.to_numeric(df[26], errors="coerce")
    df["Provincia_nombre"] = df[37].astype(str).replace({"nan": np.nan, "": np.nan})
    df["Ciudad"] = df[36].astype(str).replace({"nan": np.nan, "": np.nan})
    df["Comprador"] = df[33].astype(str)
    df["Fuente"] = "ML_Oficial"
    df["Cliente"] = df["Comprador"]

    n_before = len(df)
    df = df.dropna(subset=["Fecha", "Ingreso_bruto"])
    # "Paquete de 3 productos" is a grouping row, not an individual sale
    df = df[df["Estado"] != "Paquete de 3 productos"].reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped > 0:
        print(f"  [ML Official] {dropped} rows dropped (nulls or bundle rows)")

    assert len(df) > 0, "ML Official DataFrame is empty after cleaning"
    assert df["Ingreso_bruto"].min() > 0, "ML Official contains negative gross revenue"
    assert df["Ingreso_neto"].isna().sum() < len(df) * 0.5, \
        "More than 50% of net revenues are null, check column 18"

    return df


# Transform
def exclude_off_topic_items(df: pd.DataFrame,
                             items_off_topic: list[int] = ITEMS_OFF_TOPIC,
                             verbose: bool = True) -> pd.DataFrame:
    """ Exclude sales of products unrelated to the decoration business.

    These 7 records represented ~22% of gross revenue and were distorting all downstream analysis """
    mask = df["Item_id"].isin(items_off_topic)
    n_excluded = mask.sum()
    amount_excluded = df.loc[mask, "Ingreso_bruto"].sum()
    pct = amount_excluded / df["Ingreso_bruto"].sum() * 100

    if verbose and n_excluded > 0:
        print(f" Excluded {n_excluded} off-topic sale(s)")
        print(f" Revenue excluded: ${amount_excluded:,.0f} ARS ({pct:.1f}% of original gross)")

    return df[~mask].copy().reset_index(drop=True)


def merge_sources(df_cnx: pd.DataFrame,
                  df_ml: pd.DataFrame,
                  cutoff: pd.Timestamp = pd.Timestamp("2025-04-01")) -> pd.DataFrame:
    """ Merge CNX and ML Official into a single DataFrame without duplicates.

    Both sources overlap in April 2025. To avoid double-counting, CNX is used up to March 2025 and ML Official from April 2025 onwards """
    cnx_pre = df_cnx[df_cnx["Fecha"] < cutoff][FINAL_COLS].copy()
    ml_post = df_ml[df_ml["Fecha"] >= cutoff][FINAL_COLS].copy()

    df = pd.concat([cnx_pre, ml_post], ignore_index=True)
    df = df.sort_values("Fecha").reset_index(drop=True)

    df["Año"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month
    df["AñoMes"] = df["Fecha"].dt.to_period("M")

    # Structural assertions, always valid regardless of dataset size
    assert df["Fecha"].isna().sum() == 0, \
        "Final dataset contains null dates"
    assert df["Monto"].min() > 0, \
        "Final dataset contains zero or negative amounts"
    assert df["Ingreso_bruto"].min() > 0, \
        "Final dataset contains zero or negative gross revenue"

    # Note: business-level assertions (min row count, date range, source count) are checked in run_etl() after the full pipeline runs, not here.
    # This keeps merge_sources() unit-testable with small synthetic DataFrames.
    return df


def validate_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes dataset coverage and data quality

    Documents data limitations for the README and notebook 'Limitations' sections """
    notes = {
        "Ingreso_neto": "Only available in ML Official (Apr 2025 - Apr 2026)",
        "Titulo_prod": "Only available in ML Official (Apr 2025 - Apr 2026)",
        "Provincia_nombre": "CNX does not record province for some buyers",
        "Ciudad": "CNX does not record city for some buyers",
    }
    rows = []
    for col in FINAL_COLS:
        n_null = df[col].isna().sum()
        rows.append({
            "Column": col,
            "Complete": len(df) - n_null,
            "Null": n_null,
            "% Complete": round((1 - n_null / len(df)) * 100, 1),
            "Note": notes.get(col, "-"),
        })
    return pd.DataFrame(rows)


# Main Pipeline
def validate_data_quality(df: pd.DataFrame,
                           verbose: bool = True) -> pd.DataFrame:
    """ Runs data quality checks on the clean dataset

    Checks: negative/zero prices, null dates, duplicate orders, null Monto, implausible quantities, future dates """
    checks = []

    def check(name: str, condition: bool, n_issues: int, note: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        checks.append({"Check": name, "Status": status,
                        "Issues": n_issues, "Note": note})

    # Critical checks, raise on failure
    n_null_dates = int(df["Fecha"].isna().sum())
    check("No null dates", n_null_dates == 0, n_null_dates,
          "Null dates indicate parsing failure in ETL")
    assert n_null_dates == 0, f"Null dates detected: {n_null_dates} rows"

    n_neg_price = int((df["Monto"] <= 0).sum())
    check("No zero/negative prices", n_neg_price == 0, n_neg_price,
          "Prices <= 0 are invalid, likely data entry errors")
    assert n_neg_price == 0, f"Zero/negative prices: {n_neg_price} rows"

    n_dups = int(df["Order_id"].duplicated().sum())
    check("No duplicate Order_ids", n_dups == 0, n_dups,
          "Duplicates indicate merge overlap, check cutoff date")
    assert n_dups == 0, f"Duplicate Order_ids: {n_dups}"

    # Warning checks — do not raise, just flag
    n_null_monto = int(df["Monto"].isna().sum())
    check("No null Monto", n_null_monto == 0, n_null_monto,
          "Null Monto rows should have been dropped in ETL")

    n_qty_zero = int((df["Cantidad"].fillna(1) <= 0).sum())
    check("No zero/negative quantities", n_qty_zero == 0, n_qty_zero,
          "Cantidad <= 0 is unusual, verify source data")

    future_cutoff = pd.Timestamp.now() + pd.Timedelta(days=30)
    n_future = int((df["Fecha"] > future_cutoff).sum())
    check("No implausible future dates", n_future == 0, n_future,
          f"Dates > {future_cutoff.date()} may be data entry errors")

    n_high_ticket = int((df["Monto"] > 50_000).sum())
    check("Ticket sanity (≤ $50K ARS)", n_high_ticket == 0, n_high_ticket,
          "Tickets above $50K ARS are unusual for this business — verify")

    n_both_sources = df["Fuente"].nunique()
    check("Both sources present", n_both_sources == 2, 2 - n_both_sources,
          "Expect exactly CNX and ML_Oficial")

    result = pd.DataFrame(checks)
    if verbose:
        fails = result[result["Status"] == "FAIL"]
        passes = result[result["Status"] == "PASS"]
        print(f" Data quality: {len(passes)}/{len(result)} checks passed")
        if len(fails):
            print(f" Warnings ({len(fails)} checks):")
            for _, row in fails.iterrows():
                print(f" [{row['Status']}] {row['Check']}: {row['Issues']} issues - {row['Note']}")

    return result


def run_etl(path_cnx: str, path_ml: str, output_path: str, cutoff: pd.Timestamp = pd.Timestamp("2025-04-01"),
            verbose: bool = True) -> pd.DataFrame:
    """ Runs the complete ETL pipeline end-to-end """
    if verbose:
        print()
        print("ETL PIPELINE - MercadoLibre Sales Analytics")
        print()

    if verbose:
        print("\n[1/4] Loading CNX...")
    df_cnx = load_cnx(path_cnx)
    if verbose:
        print(f" {len(df_cnx)} orders | "
              f"{df_cnx['Fecha'].min().date()} to {df_cnx['Fecha'].max().date()}")

    if verbose:
        print("\n[2/4] Loading ML Official...")
    df_ml = load_ml_oficial(path_ml)
    if verbose:
        print(f" {len(df_ml)} orders | "
              f"{df_ml['Fecha'].min().date()} to {df_ml['Fecha'].max().date()}")

    if verbose:
        print("\n[3/4] Excluding off-topic sales...")
    df_cnx = exclude_off_topic_items(df_cnx, verbose=verbose)
    df_ml = exclude_off_topic_items(df_ml, verbose=verbose)

    if verbose:
        print("\n[4/4] Merging sources and saving...")
    df = merge_sources(df_cnx, df_ml, cutoff=cutoff)

    # Run data quality checks
    if verbose:
        print("\n Running data quality checks...")
    validate_data_quality(df, verbose=verbose)

    # Business assertions on the full production dataset
    assert len(df) > 300, \
        f"Final dataset has only {len(df)} rows, check date filters"
    assert df["Fuente"].nunique() == 2, \
        "Final dataset does not have exactly 2 sources"
    assert df["Fecha"].max().year >= 2026, \
        "Dataset does not reach 2026, check ML Official loading"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Anonymize customer identifiers before saving
    import hashlib
    df["Cliente"] = df["Cliente"].apply(
    lambda x: "C_" + hashlib.md5(str(x).encode()).hexdigest()[:8]
    if pd.notna(x) else x)
    df.to_csv(output_path, index=False)

    if verbose:
        print("\n")
        print("FINAL DATASET")
        print("")
        print(f" Orders: {len(df)}")
        print(f" Range: {df['Fecha'].min().date()} to {df['Fecha'].max().date()}")
        print(f" Gross revenue: ${df['Ingreso_bruto'].sum():,.0f} ARS")
        print(f" Avg ticket: ${df['Monto'].mean():,.0f} ARS")
        print(f" Median ticket: ${df['Monto'].median():,.0f} ARS")
        print("\n Data coverage:")
        cov = validate_coverage(df)
        print(cov[cov["Null"] > 0][["Column","% Complete","Note"]].to_string(index=False))
        print(f"\n Saved to: {output_path}")
        print(" ETL complete")

    return df


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run the MercadoLibre sales ETL pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cnx", type=str, required=True, help="Path to the CNX BigQuery export .xlsx file.",)
    parser.add_argument("--ml", type=str, required=True, help="Path to the ML Official seller report .xlsx file.",)
    parser.add_argument("--output", type=str, default=str(Path(__file__).parent.parent / "data" / "ventas_decoraciones.csv"), help="Path for the output CSV file",)
    parser.add_argument("--cutoff", type=str, default="2025-04-01", help="Merge cutoff date (YYYY-MM-DD). CNX used before, ML Official from this date",)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.",)
    args = parser.parse_args()

    run_etl(
        path_cnx = args.cnx,
        path_ml = args.ml,
        output_path = args.output,
        cutoff = pd.Timestamp(args.cutoff),
        verbose = not args.quiet,
    )
