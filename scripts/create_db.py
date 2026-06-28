
import sqlite3
import pandas as pd
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
SQL_DIR = ROOT / "sql"
DB_PATH = DATA_DIR / "analytics.db"

SQL_FILES = ["revenue.sql", "features.sql", "elasticity_prep.sql", "cohorts.sql",]

CSV_TABLES = {"ventas_decoraciones": DATA_DIR / "ventas_decoraciones.csv",
    "ipc_indec": DATA_DIR / "ipc_indec.csv",
    "rfm_clientes": DATA_DIR / "rfm_clientes.csv",
    "forecast": DATA_DIR / "forecast_6meses.csv",}


def load_csvs(conn: sqlite3.Connection) -> None:
    """Load CSVs into SQLite tables """
    for table_name, path in CSV_TABLES.items():
        if not path.exists():
            print(f" [SKIP] {path.name} not found")
            continue
        df = pd.read_csv(path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f" [OK] {table_name} ({len(df):,} rows)")


def _strip_sql_comments(sql: str) -> str:
    """Strip '--' line comments before statement-splitting """
    lines = []
    for line in sql.splitlines():
        idx = line.find("--")
        lines.append(line[:idx] if idx != -1 else line)
    return "\n".join(lines)


def load_views(conn: sqlite3.Connection) -> None:
    """Execute SQL view definitions """
    for filename in SQL_FILES:
        path = SQL_DIR / filename
        if not path.exists():
            print(f"  [SKIP] {filename} not found")
            continue
        sql = _strip_sql_comments(path.read_text(encoding="utf-8"))
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        any_failed = False
        for stmt in statements:
            try:
                conn.execute(stmt)
            except sqlite3.Error as e:
                print(f"  [WARN] {filename}: {e}")
                any_failed = True
        conn.commit()
        if any_failed:
            print(f" [WARN] {filename}")
        else:
            print(f" [OK] {filename}")


def verify(conn: sqlite3.Connection) -> None:
    """Print a quick summary of what's in the database """
    print("\nDatabase contents:")
    tables = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name").fetchall()
    for name, kind in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
            print(f"  {kind:<6} {name:<35} {count:>6} rows")
        except sqlite3.Error:
            print(f"  {kind:<6} {name:<35}  (view - no row count)")


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing {DB_PATH.name}")

    print(f"\nCreating {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)

    print("\nLoading tables:")
    load_csvs(conn)

    print("\nLoading views:")
    load_views(conn)

    verify(conn)
    conn.close()

    print(f"\nDone. Open with SQLTools in VSCode -> connect to {DB_PATH}")


if __name__ == "__main__":
    main()
