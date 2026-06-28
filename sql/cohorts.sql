
-- Step 1: identify each customer's acquisition (first purchase) cohort
DROP VIEW IF EXISTS customer_cohorts;
CREATE VIEW customer_cohorts AS
SELECT
    v.Cliente AS customer,
    MIN(strftime('%Y-%m', v.Fecha)) AS cohort_month,
    CAST(strftime('%Y', MIN(v.Fecha)) AS INT) AS cohort_year,
    COUNT(DISTINCT strftime('%Y-%m', v.Fecha)) AS active_months,
    COUNT(v.Order_id) AS total_orders,
    SUM(v.Ingreso_bruto) AS lifetime_value,
    MIN(v.Fecha) AS first_purchase,
    MAX(v.Fecha) AS last_purchase,
    ROUND(JULIANDAY(MAX(v.Fecha)) - JULIANDAY(MIN(v.Fecha))) AS tenure_days
FROM ventas_decoraciones v
WHERE v.Ingreso_bruto > 0
  AND v.Cliente IS NOT NULL
GROUP BY 1;


-- Step 2: month-by-month retention matrix
-- Each row = cohort × months_since_first_purchase
DROP VIEW IF EXISTS cohort_retention;
CREATE VIEW cohort_retention AS
SELECT
    c.cohort_month,
    strftime('%Y-%m', v.Fecha) AS activity_month,
    CAST(
        ROUND(
            (JULIANDAY(date(strftime('%Y-%m', v.Fecha) || '-01')) - JULIANDAY(date(c.cohort_month || '-01'))) / 30.44) AS INT)                                           AS months_since_first,
    COUNT(DISTINCT v.Cliente) AS active_customers,
    SUM(v.Ingreso_bruto) AS cohort_revenue
FROM ventas_decoraciones v
JOIN customer_cohorts c
    ON v.Cliente = c.customer
WHERE v.Ingreso_bruto > 0
  AND v.Cliente IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 1, 3;


-- Step 3: cohort size (denominator for retention rate)
DROP VIEW IF EXISTS cohort_sizes;
CREATE VIEW cohort_sizes AS
SELECT
    cohort_month,
    COUNT(DISTINCT customer) AS cohort_size,
    SUM(lifetime_value) AS cohort_ltv,
    AVG(lifetime_value) AS avg_ltv
FROM customer_cohorts
GROUP BY 1
ORDER BY 1;


-- Step 4: retention rate per cohort × month 
DROP VIEW IF EXISTS retention_rates;
CREATE VIEW retention_rates AS
SELECT
    r.cohort_month,
    r.months_since_first,
    r.active_customers,
    s.cohort_size,
    ROUND(r.active_customers * 100.0 / s.cohort_size, 1) AS retention_pct
FROM cohort_retention r
JOIN cohort_sizes s
    ON r.cohort_month = s.cohort_month
ORDER BY r.cohort_month, r.months_since_first;
