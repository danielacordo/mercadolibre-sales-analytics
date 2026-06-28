
DROP VIEW IF EXISTS monthly_features;
CREATE VIEW monthly_features AS
WITH base AS (
    SELECT
        strftime('%Y-%m', v.Fecha) AS period,
        CAST(strftime('%Y', v.Fecha) AS INT) AS year,
        CAST(strftime('%m', v.Fecha) AS INT)  AS month,
        COUNT(v.Order_id) AS quantity,
        SUM(v.Ingreso_bruto) AS revenue,
        AVG(v.Monto) AS avg_price,
            COUNT(DISTINCT v.Provincia_nombre) AS provinces_active,
        i.cpi_index
    FROM ventas_decoraciones v
    LEFT JOIN ipc_indec i
        ON strftime('%Y-%m', v.Fecha) = i.period
    WHERE v.Ingreso_bruto > 0
    GROUP BY
        strftime('%Y-%m', v.Fecha),
        CAST(strftime('%Y', v.Fecha) AS INT),
        CAST(strftime('%m', v.Fecha) AS INT),
        i.cpi_index)

SELECT
    period,
    year,
    month,
    quantity,
    revenue,
    avg_price,
    cpi_index,
    provinces_active,

    -- Real price (deflated by CPI)
    ROUND(avg_price / cpi_index * 100, 2) AS real_price,

    -- Fixed Aug/Sep and Jun/Jul windows as a feature engineering choice, not a claim about stable peaks. 
    -- Current data shows Sep and Jan are top-2 (17%-32% revenue share, narrowing year to year). 
    -- Currently unused downstream,if wired into a model, derive peak months from data instead. 
    CASE WHEN month IN (8, 9) THEN 1 ELSE 0 END  AS is_peak_season,
    CASE WHEN month IN (6, 7) THEN 1 ELSE 0 END  AS is_low_season,

    -- Month fixed effects (drop month_01 as reference to avoid multicollinearity)
    CASE WHEN month = 2 THEN 1 ELSE 0 END AS fe_m02,
    CASE WHEN month = 3 THEN 1 ELSE 0 END AS fe_m03,
    CASE WHEN month = 4 THEN 1 ELSE 0 END AS fe_m04,
    CASE WHEN month = 5 THEN 1 ELSE 0 END AS fe_m05,
    CASE WHEN month = 6 THEN 1 ELSE 0 END AS fe_m06,
    CASE WHEN month = 7 THEN 1 ELSE 0 END AS fe_m07,
    CASE WHEN month = 8 THEN 1 ELSE 0 END AS fe_m08,
    CASE WHEN month = 9 THEN 1 ELSE 0 END AS fe_m09,
    CASE WHEN month = 10 THEN 1 ELSE 0 END AS fe_m10,
    CASE WHEN month = 11 THEN 1 ELSE 0 END AS fe_m11,
    CASE WHEN month = 12 THEN 1 ELSE 0 END AS fe_m12

FROM base
ORDER BY period;
