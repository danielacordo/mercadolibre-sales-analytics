DROP VIEW IF EXISTS monthly_revenue;
CREATE VIEW monthly_revenue AS
SELECT
    strftime('%Y-%m', v.Fecha) AS period,
    CAST(strftime('%Y', v.Fecha) AS INT)  AS year,
    CAST(strftime('%m', v.Fecha) AS INT) AS month,
    COUNT(v.Order_id) AS orders,
    SUM(v.Monto) AS gross_revenue_nominal,
    AVG(v.Monto) AS avg_ticket_nominal,
    SUM(v.Ingreso_bruto) AS ingreso_bruto_nominal,
    -- Real (inflation-adjusted) values, base Jan 2023 = 100
    ROUND(SUM(v.Ingreso_bruto) / i.cpi_index * 100, 2)  AS ingreso_bruto_real,
    ROUND(AVG(v.Monto) / i.cpi_index * 100, 2) AS avg_ticket_real,
    i.cpi_index
FROM ventas_decoraciones v
LEFT JOIN ipc_indec i
    ON strftime('%Y-%m', v.Fecha) = i.period
WHERE v.Ingreso_bruto > 0
GROUP BY
    strftime('%Y-%m', v.Fecha),
    CAST(strftime('%Y', v.Fecha) AS INT),
    CAST(strftime('%m', v.Fecha) AS INT),
    i.cpi_index
ORDER BY period;


-- Annual summary with YoY growth
DROP VIEW IF EXISTS annual_revenue;
CREATE VIEW annual_revenue AS
SELECT
    CAST(strftime('%Y', Fecha) AS INT)  AS year,
    COUNT(Order_id) AS total_orders,
    SUM(Ingreso_bruto) AS gross_revenue_nominal,
    ROUND(AVG(Monto), 0) AS avg_ticket_nominal
FROM ventas_decoraciones
WHERE Ingreso_bruto > 0
GROUP BY CAST(strftime('%Y', Fecha) AS INT)
ORDER BY year;


-- Net margin analysis (ML Official data only)

-- Two methodology notes, consistent with margin_summary() in eda.py and decision_layer.py:
-- 1. Filter IS NOT NULL only, not > 0. Eight ML_Oficial orders have Ingreso_neto = 0 (real sales, zero net payout, likely dispute resolutions). 
--    Excluding them inflates the headline margin from 63.1% to 71.4%, a six-figure-ARS difference.
-- 2. avg_net_margin_pct is an unweighted per-order ratio,  useful for monthly trends. 
--    Use net_margin_overall (revenue-weighted) for that.
DROP VIEW IF EXISTS net_margin_by_month;
CREATE VIEW net_margin_by_month AS
SELECT
    strftime('%Y-%m', Fecha) AS period,
    COUNT(Order_id) AS orders_with_net,
    ROUND(
        AVG(Ingreso_neto / NULLIF(Ingreso_bruto, 0)) * 100
    , 1) AS avg_net_margin_pct,
    ROUND(
        1 - AVG(Ingreso_neto / NULLIF(Ingreso_bruto, 0))
    , 3) AS ml_fee_rate
FROM ventas_decoraciones
WHERE Ingreso_neto IS NOT NULL
  AND Ingreso_bruto > 0
GROUP BY strftime('%Y-%m', Fecha)
ORDER BY period;


-- Net margin, revenue-weighted, single overall figure.

-- Revenue-weighted margin: sum(Ingreso_neto) / sum(Ingreso_bruto).
-- Source of the 63.1% headline figure; matches eda.py and decision_layer.py.
-- Filters IS NOT NULL only
DROP VIEW IF EXISTS net_margin_overall;
CREATE VIEW net_margin_overall AS
SELECT
    COUNT(Order_id) AS orders_with_net,
    ROUND(SUM(Ingreso_neto) * 100.0 / SUM(Ingreso_bruto), 1) AS net_margin_pct_weighted,
    ROUND(1 - SUM(Ingreso_neto) * 1.0 / SUM(Ingreso_bruto), 3) AS ml_fee_rate_weighted
FROM ventas_decoraciones
WHERE Ingreso_neto IS NOT NULL
  AND Ingreso_bruto > 0;
