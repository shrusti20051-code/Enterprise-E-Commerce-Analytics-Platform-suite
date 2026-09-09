DROP TABLE IF EXISTS olist_orders

-- Creating the Orders Table to track the lifecycle of every purchase
CREATE TABLE olist_orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    order_purchase_timestamp TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP,
    order_status VARCHAR(20)
);

--Identifying "Late" Deliveries
--The Issue: The business noticed a drop in review scores. I suspected that orders arriving after the estimated date were the primary cause. I needed to find exactly how many orders were late and by how many days.

WITH delivery_performance AS (
    SELECT 
        order_id,
        order_delivered_customer_date,
        order_estimated_delivery_date,
        EXTRACT(DAY FROM (order_delivered_customer_date - order_estimated_delivery_date)) AS days_diff
    FROM olist_orders
    WHERE order_status = 'delivered'
)
SELECT 
    COUNT(order_id) AS total_late_orders,
    ROUND(AVG(days_diff), 2) AS avg_days_late
FROM delivery_performance
WHERE days_diff > 0;

--Finding the "Bottleneck" Region
--The Issue: Now that I knew orders were late, I needed to know where. Are certain states or cities performing worse than others? This helps the company know which warehouse or carrier to fix.

SELECT 
    c.customer_state, 
    COUNT(o.order_id) AS late_order_count,
    ROUND(AVG(EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date))), 2) AS avg_delay
FROM olist_orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_delivered_customer_date > o.order_estimated_delivery_date
GROUP BY c.customer_state
ORDER BY avg_delay DESC
LIMIT 10;

--You can still analyze late deliveries overall (without state info) using only olist_orders

SELECT 
    -- COUNT(order_id) AS late_order_count,
    ROUND(AVG(EXTRACT(DAY FROM order_delivered_customer_date - order_estimated_delivery_date)), 2) AS avg_delay
FROM olist_orders
WHERE order_delivered_customer_date > order_estimated_delivery_date;

--What is the total order volume?

SELECT COUNT(*) AS total_orders
FROM olist_orders;

--Business question: How many orders are delivered, canceled, shipped, etc.?

SELECT
    order_status,
    COUNT(*) AS total_orders
FROM olist_orders
GROUP BY order_status
ORDER BY total_orders DESC;

--What percentage of orders were delivered?
--Instead of only knowing the number, let's calculate the percentage.

SELECT
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE order_status = 'delivered')
        / COUNT(*),
        2
    ) AS delivered_percentage
FROM olist_orders;

--What percentage of orders were canceled?
SELECT
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE order_status = 'canceled')
        / COUNT(*),
        2
    ) AS canceled_percentage
FROM olist_orders;

--How many orders were placed each year?

SELECT
    EXTRACT(YEAR FROM order_purchase_timestamp) AS year,
    COUNT(*) AS total_orders
FROM olist_orders
GROUP BY year
ORDER BY year;


--How many orders were placed each month?

SELECT
    DATE_TRUNC('month', order_purchase_timestamp) AS month,
    COUNT(*) AS total_orders
FROM olist_orders
GROUP BY month
ORDER BY month;

What day of the week receives the most orders?

SELECT
    TO_CHAR(order_purchase_timestamp, 'Day') AS day_of_week,
    COUNT(*) AS total_orders
FROM olist_orders
GROUP BY day_of_week
ORDER BY total_orders DESC;

--What hour receives the most orders?

SELECT
    EXTRACT(HOUR FROM order_purchase_timestamp) AS order_hour,
    COUNT(*) AS total_orders
FROM olist_orders
GROUP BY order_hour
ORDER BY total_orders DESC;

--How long does it take to deliver an order?

SELECT
    order_id,
    order_purchase_timestamp,
    order_delivered_customer_date,
    order_delivered_customer_date - order_purchase_timestamp
        AS delivery_time
FROM olist_orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL;

--Now we can calculate the average.

SELECT
    ROUND(
        AVG(
            EXTRACT(
                EPOCH FROM
                (order_delivered_customer_date - order_purchase_timestamp)
            ) / 86400
        ),
        2
    ) AS average_delivery_days
FROM olist_orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL;

--How many orders were delivered late?

SELECT
    COUNT(*) AS late_orders
FROM olist_orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date >
      order_estimated_delivery_date;

--What percentage of delivered orders were late?
SELECT
    ROUND(
        100.0 *
        COUNT(*) FILTER (
            WHERE order_delivered_customer_date >
                order_estimated_delivery_date
        )
        / COUNT(*),
        2
    ) AS late_delivery_percentage
FROM olist_orders
WHERE order_status = 'delivered'
AND order_delivered_customer_date IS NOT NULL
AND order_estimated_delivery_date IS NOT NULL;

--How many days late are orders?

SELECT
    order_id,
    ROUND(
        EXTRACT(
            EPOCH FROM
            (order_delivered_customer_date -
             order_estimated_delivery_date)
        ) / 86400,
        2
    ) AS days_late
FROM olist_orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date >
      order_estimated_delivery_date
ORDER BY days_late DESC;

--Now find the 10 most delayed orders

SELECT
    order_id,
    ROUND(
        EXTRACT(
            EPOCH FROM
            (order_delivered_customer_date -
             order_estimated_delivery_date)
        ) / 86400,
        2
    ) AS days_late
FROM olist_orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date >
      order_estimated_delivery_date
ORDER BY days_late DESC
LIMIT 10;

--How many orders have missing delivery

SELECT
    COUNT(*) FILTER (WHERE order_purchase_timestamp IS NULL)
        AS missing_purchase_date,

    COUNT(*) FILTER (WHERE order_approved_at IS NULL)
        AS missing_approval_date,

    COUNT(*) FILTER (WHERE order_delivered_carrier_date IS NULL)
        AS missing_carrier_date,

    COUNT(*) FILTER (WHERE order_delivered_customer_date IS NULL)
        AS missing_customer_delivery_date,

    COUNT(*) FILTER (WHERE order_estimated_delivery_date IS NULL)
        AS missing_estimated_date
FROM olist_orders;

--How long does order approval take?

SELECT
    ROUND(
        AVG(
            EXTRACT(
                EPOCH FROM
                (order_approved_at - order_purchase_timestamp)
            ) / 3600
        ),
        2
    ) AS avg_approval_hours
FROM olist_orders
WHERE order_approved_at IS NOT NULL;

--How long does it take to hand orders to the carrier?

SELECT
    ROUND(
        AVG(
            EXTRACT(
                EPOCH FROM
                (order_delivered_carrier_date -
                 order_approved_at)
            ) / 24 / 3600
        ),
        2
    ) AS avg_days_to_carrier
FROM olist_orders
WHERE order_approved_at IS NOT NULL
  AND order_delivered_carrier_date IS NOT NULL;

--Find orders that took more than 30 days to deliver

SELECT
    order_id,
    ROUND(
        EXTRACT(
            EPOCH FROM
            (order_delivered_customer_date -
             order_purchase_timestamp)
        ) / 86400,
        2
    ) AS delivery_days
FROM olist_orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
  AND order_delivered_customer_date -
      order_purchase_timestamp > INTERVAL '30 days'
ORDER BY delivery_days DESC;

--Categorize delivery performance

SELECT
    CASE
        WHEN order_delivered_customer_date <=
             order_estimated_delivery_date
            THEN 'On Time'


        WHEN order_delivered_customer_date >
             order_estimated_delivery_date
            THEN 'Late'


        ELSE 'Unknown'
    END AS delivery_performance,
    COUNT(*) AS total_orders
FROM olist_orders
WHERE order_status = 'delivered'
GROUP BY delivery_performance
ORDER BY total_orders DESC;

--What is the order success rate?

SELECT
    ROUND(
        100.0 *
        COUNT(*) FILTER (WHERE order_status = 'delivered')
        / COUNT(*),
        2
    ) AS success_rate
FROM olist_orders;

-- Which order statuses should management investigate?

SELECT
    order_status,
    COUNT(*) AS total_orders,
    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage_of_orders
FROM olist_orders
GROUP BY order_status
ORDER BY total_orders DESC;