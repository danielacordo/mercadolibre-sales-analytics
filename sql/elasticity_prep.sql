
DROP VIEW IF EXISTS elasticity_data;
CREATE VIEW elasticity_data AS
SELECT
    strftime('%Y-%m', v.Fecha) AS period,
    CAST(strftime('%Y', v.Fecha) AS INT) AS year,
    CAST(strftime('%m', v.Fecha) AS INT) AS month,
    COUNT(v.Order_id) AS quantity,
    AVG(v.Monto) AS avg_price,
    -- True median_price computed in Python (no MEDIAN in SQLite): df.groupby("period")["Monto"].median()
    i.cpi_index AS cpi,

    -- Seasonality controls
    CASE WHEN CAST(strftime('%m', v.Fecha) AS INT) IN (8, 9)
         THEN 1.0 ELSE 0.0 END AS peak_season,
    CASE WHEN CAST(strftime('%m', v.Fecha) AS INT) IN (6, 7)
         THEN 1.0 ELSE 0.0 END AS low_season

FROM ventas_decoraciones v
LEFT JOIN ipc_indec i
    ON strftime('%Y-%m', v.Fecha) = i.period
WHERE v.Ingreso_bruto > 0
  AND v.Monto > 0
GROUP BY
    strftime('%Y-%m', v.Fecha),
    CAST(strftime('%Y', v.Fecha) AS INT),
    CAST(strftime('%m', v.Fecha) AS INT),
    i.cpi_index
HAVING quantity > 0
   AND cpi IS NOT NULL
ORDER BY period;
