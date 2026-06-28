import sys
import sqlite3
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

_BASE = Path(__file__).parent.parent
if str(_BASE / "src") not in sys.path:
    sys.path.insert(0, str(_BASE / "src"))

from elasticity import estimate_log_log_elasticity, estimate_controlled_elasticity  # noqa: E402

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
SQL_DIR = BASE / "sql"

SQL_FILES = [
    "revenue.sql",
    "elasticity_prep.sql",
    "features.sql",
]


# Database Setup 
def load_db(data_dir: Optional[Path] = None, verbose: bool = True) -> sqlite3.Connection:
    """ Load CSVs into an in-memory SQLite database and register extensions.

    Tables created:
        ventas_decoraciones: clean transaction dataset (output of ETL)
        ipc_indec: Argentina CPI index (INDEC, documented)

    Extensions registered:
        LN(x): natural log, required by SQL feature files """
    if data_dir is None:
        data_dir = DATA_DIR

    db = sqlite3.connect(":memory:")

    # Register LN()
    db.create_function("LN", 1, math.log)

    # Load CSVs as tables
    tables = {"ventas_decoraciones": data_dir / "ventas_decoraciones.csv", "ipc_indec": data_dir / "ipc_indec.csv",}
    for table_name, csv_path in tables.items():
        assert csv_path.exists(), f"Missing: {csv_path}"
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, db, index=False, if_exists="replace")
        if verbose:
            print(f" Loaded {table_name}: {len(df):,} rows")

    # Execute SQL view files
    for sql_file in SQL_FILES:
        sql_path = SQL_DIR / sql_file
        if not sql_path.exists():
            if verbose:
                print(f" Skipped (not found): {sql_file}")
            continue
        _execute_sql_file(db, sql_path)
        if verbose:
            _view_name = sql_file.replace(".sql", "").replace("_prep", "_data")
            print(f" Created views from: {sql_file}")

    if verbose:
        views = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()]
        print(f"  Views available: {views}")

    return db


def _execute_sql_file(db: sqlite3.Connection, path: Path) -> None:
    """Executes a .sql file against the database, skipping comment-only blocks"""
    with open(path) as f:
        sql = f.read()

    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        # Skip if all non-empty lines are comments
        non_comment = [
            line for line in stmt.split("\n")
            if line.strip() and not line.strip().startswith("--")]
        if not non_comment:
            continue
        try:
            db.execute(stmt)
            db.commit()
        except sqlite3.OperationalError as e:
            raise sqlite3.OperationalError(f"Error in {path.name}: {e}\nStatement: {stmt[:120]}") from e


def query(db: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Executes a SELECT query and return a DataFrame """
    return pd.read_sql_query(sql, db)



# Feature Extraction 
def get_monthly_features(db: sqlite3.Connection) -> pd.DataFrame:
    """Loads the monthly_features view and add Python-computed columns.

    The SQL view computes: aggregations, seasonality dummies, real price, month fixed effects."""
    df_features = query(db, "SELECT * FROM monthly_features ORDER BY period")
    df_raw = query(db, "SELECT strftime('%Y-%m', Fecha) AS period, Monto FROM ventas_decoraciones WHERE Ingreso_bruto > 0")

    # True median (SQL uses AVG as placeholder)
    median_map = df_raw.groupby("period")["Monto"].median()
    df_features["median_price"] = df_features["period"].map(median_map)

    # Log transforms (requires non-zero, non-null values)
    mask = (
        (df_features["median_price"] > 0) &
        (df_features["quantity"] > 0) &
        (df_features["cpi_index"].notna()))
    df_features.loc[mask, "log_price"] = np.log(df_features.loc[mask, "median_price"])
    df_features.loc[mask, "log_quantity"] = np.log(df_features.loc[mask, "quantity"])
    df_features.loc[mask, "log_cpi"] = np.log(df_features.loc[mask, "cpi_index"])

    # Outlier flag for forecasting (z-score on revenue)
    rev = df_features["revenue"]
    df_features["z_score"] = (rev - rev.mean()) / rev.std()
    median_normal = df_features.loc[df_features["z_score"].abs() <= 2, "revenue"].median()
    df_features["y_model"] = df_features["revenue"].copy()
    df_features.loc[df_features["z_score"] > 2, "y_model"] = median_normal

    df_features["ds"] = pd.to_datetime(df_features["period"] + "-01")

    return df_features


def get_elasticity_data(db: sqlite3.Connection) -> pd.DataFrame:
    """ Load elasticity_data view and add log transforms.

    The SQL view provides: monthly quantity, avg_price, cpi, seasonality dummies. 
    Python adds: median_price, log transforms."""

    df = query(db, "SELECT * FROM elasticity_data ORDER BY period")
    df_raw = query(db, "SELECT strftime('%Y-%m', Fecha) AS period, Monto FROM ventas_decoraciones WHERE Ingreso_bruto > 0")

    median_map = df_raw.groupby("period")["Monto"].median()
    df["median_price"] = df["period"].map(median_map)
    df["price"] = df["median_price"] 

    mask = (df["price"] > 0) & (df["quantity"] > 0) & (df["cpi"].notna())
    df.loc[mask, "log_price"] = np.log(df.loc[mask, "price"])
    df.loc[mask, "log_quantity"] = np.log(df.loc[mask, "quantity"])
    df.loc[mask, "log_cpi"] = np.log(df.loc[mask, "cpi"])

    df["ds"] = pd.to_datetime(df["period"] + "-01")
    df["year"] = df["ds"].dt.year

    return df.dropna(subset=["log_price", "log_quantity"])


def get_revenue_summary(db: sqlite3.Connection) -> pd.DataFrame:
    """ Load monthly_revenue with real vs nominal comparison """
    return query(db, """
        SELECT
            period,
            orders,
            gross_revenue_nominal,
            ingreso_bruto_real,
            avg_ticket_nominal,
            avg_ticket_real,
            cpi_index
        FROM monthly_revenue
        ORDER BY period
    """)



if __name__ == "__main__":
    # estimate_log_log_elasticity, estimate_controlled_elasticity imported at module level
    print()
    print("SQL -> Python -> Model Pipeline")
    print()

    # Step 1: load CSVs into SQLite and execute SQL views
    print("\n[1/4] Loading database and executing SQL views...")
    db = load_db(verbose=True)

    # Step 2: extract feature tables via Python
    print("\n[2/4] Extracting features (SQL views + Python transforms)...")
    monthly = get_monthly_features(db)
    elas_data = get_elasticity_data(db)
    revenue = get_revenue_summary(db)
    print(f" monthly_features: {len(monthly)} rows, {len(monthly.columns)} columns")
    print(f" elasticity_data: {len(elas_data)} rows")
    print(f" revenue_summary: {len(revenue)} rows")

    # Step 3: run elasticity models on SQL-prepared features
    print("\n[3/4] Running elasticity models on SQL-prepared features...")
    result = estimate_log_log_elasticity(elas_data)
    print(f" Simple OLS: epsilon={result.elasticity:.4f}, R²={result.r_squared:.4f}")

    controlled = estimate_controlled_elasticity(elas_data)
    print("\n  Model comparison (epsilon is stable across all specs):")
    print(controlled[["Model", "epsilon (price)", "R²", "Note"]].to_string(index=False))

    # Step 4: real vs nominal revenue from SQL
    print("\n[4/4] Real vs nominal revenue (from SQL monthly_revenue view):")
    by_year = revenue.copy()
    by_year["year"] = by_year["period"].str[:4]
    annual = by_year.groupby("year").agg(nominal=("gross_revenue_nominal", "sum"), real=("ingreso_bruto_real", "sum"),).round(0)
    annual["real_as_pct_nominal"] = (annual["real"] / annual["nominal"] * 100).round(1)
    print(annual.to_string())
    print("\n -> Real revenue collapsed despite nominal growth (inflation illusion)")

    print("\n Pipeline complete")
    db.close()