# SQL Queries

SQLite views loaded in-memory via `src/sql_pipeline.py`. All views join `ventas_decoraciones` with `ipc_indec` for inflation-adjusted metrics.

| File | View(s) created | Used by |
|------|-----------------|---------|
| `revenue.sql` | `monthly_revenue`, `annual_revenue`, `net_margin_by_month`, `net_margin_overall` | `src/eda.py`, `src/decision_layer.py` |
| `features.sql` | `monthly_features` | `src/elasticity.py`, `src/forecasting.py` |
| `elasticity_prep.sql` | `elasticity_data` | `src/elasticity.py` (M1/M2/M3 specs) |
| `cohorts.sql` | `cohort_matrix` | `notebooks/07_cohort_analysis.ipynb` |

**Note on `net_margin_overall`:** filters `Ingreso_neto IS NOT NULL` (not `> 0`) - 8 orders have net = 0 (dispute resolutions). Excluding them would inflate the headline margin from 63.1% to 71.4%.
