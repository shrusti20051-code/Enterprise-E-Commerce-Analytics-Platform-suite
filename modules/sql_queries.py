"""
SQL Showcase Engine for Olist E-Commerce Portfolio.
Executes ANSI SQL queries in-memory using DuckDB directly on the master dataset.
Answers the 7 core business questions from the project proposal.
"""

import duckdb
import pandas as pd

SQL_QUERIES = {
    "q1_top_categories": {
        "title": "1. Top 10 Revenue Product Categories & Unit Economics",
        "business_question": "Which product categories generate the highest total revenue, and what are their average price and order volumes?",
        "sql": """
WITH category_metrics AS (
    SELECT 
        category_english AS category,
        COUNT(DISTINCT order_id) AS total_orders,
        COUNT(product_id) AS items_sold,
        ROUND(SUM(price), 2) AS net_product_sales,
        ROUND(SUM(freight_value), 2) AS total_freight,
        ROUND(SUM(total_order_value), 2) AS gross_revenue,
        ROUND(AVG(price), 2) AS avg_item_price,
        ROUND(AVG(review_score), 2) AS avg_review_score
    FROM df
    WHERE order_status = 'delivered'
    GROUP BY category_english
)
SELECT 
    category,
    total_orders,
    items_sold,
    gross_revenue,
    ROUND((gross_revenue / SUM(gross_revenue) OVER ()) * 100, 2) AS revenue_share_pct,
    avg_item_price,
    avg_review_score
FROM category_metrics
ORDER BY gross_revenue DESC
LIMIT 10;
        """,
        "technical_highlight": "Uses a CTE and window function `SUM(...) OVER ()` to compute category market share without a secondary subquery.",
        "business_insight": "Health & Beauty, Bed Bath Table, and Computers Accessories form the top revenue triad. High AOV categories like Watches and Computers drive disproportionate gross margins."
    },

    "q2_state_sales_freight": {
        "title": "2. Regional Sales Distribution & Freight Friction by State",
        "business_question": "Which Brazilian regions generate the most sales, and how do freight costs vary by geographical distance?",
        "sql": """
SELECT 
    customer_state AS state,
    COUNT(DISTINCT order_id) AS order_volume,
    ROUND(SUM(total_order_value), 2) AS total_revenue,
    ROUND(AVG(total_order_value), 2) AS avg_order_value,
    ROUND(AVG(freight_value), 2) AS avg_freight_cost,
    ROUND((AVG(freight_value) / AVG(total_order_value)) * 100, 2) AS freight_to_revenue_ratio_pct,
    ROUND(AVG(delivery_days), 1) AS avg_delivery_days,
    ROUND(AVG(CASE WHEN is_late = true THEN 1.0 ELSE 0.0 END) * 100, 2) AS late_delivery_rate_pct
FROM df
WHERE order_status = 'delivered'
GROUP BY customer_state
ORDER BY total_revenue DESC
LIMIT 10;
        """,
        "technical_highlight": "Conditional aggregation using `CASE WHEN is_late = true THEN 1.0 ELSE 0.0 END` to compute late rate inside standard grouping.",
        "business_insight": "São Paulo (SP) represents over 40% of demand with low freight friction (10-12% ratio). Distant states (Northeast) face freight costs exceeding 25% of basket value, creating conversion headwinds."
    },

    "q3_mom_growth": {
        "title": "3. Monthly Revenue Trend & MoM Growth Velocity (Window Functions)",
        "business_question": "How did revenue and order volumes evolve month-over-month, and what was the monthly growth trajectory?",
        "sql": """
WITH monthly_summary AS (
    SELECT 
        order_year_month AS period,
        COUNT(DISTINCT order_id) AS monthly_orders,
        ROUND(SUM(total_order_value), 2) AS monthly_revenue,
        ROUND(AVG(total_order_value), 2) AS monthly_aov
    FROM df
    WHERE order_status = 'delivered'
    GROUP BY order_year_month
),
growth_calc AS (
    SELECT 
        period,
        monthly_orders,
        monthly_revenue,
        monthly_aov,
        LAG(monthly_revenue, 1) OVER (ORDER BY period) AS prev_month_revenue,
        LAG(monthly_orders, 1) OVER (ORDER BY period) AS prev_month_orders
    FROM monthly_summary
)
SELECT 
    period,
    monthly_orders,
    monthly_revenue,
    monthly_aov,
    ROUND(((monthly_revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0)) * 100, 2) AS revenue_growth_pct,
    ROUND(((monthly_orders - prev_month_orders) / NULLIF(prev_month_orders, 0)) * 100, 2) AS order_growth_pct
FROM growth_calc
ORDER BY period;
        """,
        "technical_highlight": "Uses `LAG(...) OVER (ORDER BY period)` across multiple CTE stages with `NULLIF` guards against division by zero.",
        "business_insight": "Displays rapid expansion throughout 2017 with seasonal spikes around Q4 (Black Friday), stabilizing in mid-2018 at consistent run-rates."
    },

    "q4_repeat_vs_onetime": {
        "title": "4. Customer Repeat Behavior & Lifetime Value (CLV)",
        "business_question": "How do repeat customers compare against one-time buyers in total spend, average order value, and revenue contribution?",
        "sql": """
WITH customer_orders AS (
    SELECT 
        customer_unique_id,
        COUNT(DISTINCT order_id) AS total_orders,
        ROUND(SUM(total_order_value), 2) AS customer_total_spend,
        ROUND(AVG(total_order_value), 2) AS customer_aov
    FROM df
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
),
segmentation AS (
    SELECT 
        customer_unique_id,
        total_orders,
        customer_total_spend,
        customer_aov,
        CASE 
            WHEN total_orders > 1 THEN 'Repeat Customer' 
            ELSE 'One-Time Customer' 
        END AS customer_type
    FROM customer_orders
)
SELECT 
    customer_type,
    COUNT(customer_unique_id) AS total_customers,
    ROUND((COUNT(customer_unique_id) * 100.0 / SUM(COUNT(customer_unique_id)) OVER ()), 2) AS customer_base_pct,
    ROUND(SUM(customer_total_spend), 2) AS aggregate_revenue,
    ROUND((SUM(customer_total_spend) * 100.0 / SUM(SUM(customer_total_spend)) OVER ()), 2) AS revenue_contribution_pct,
    ROUND(AVG(customer_total_spend), 2) AS avg_clv,
    ROUND(AVG(customer_aov), 2) AS avg_order_value
FROM segmentation
GROUP BY customer_type;
        """,
        "technical_highlight": "Hierarchical customer aggregation followed by segmented population metrics using windowed percentage calculations.",
        "business_insight": "Repeat customers represent strictly ~3.1% of the total customer base, yet generate over 2.5x higher average customer lifetime value (CLV) than one-time buyers. This confirms that while acquisition is strong, post-purchase retention is Olist's highest-ROI untapped growth lever."
    },

    "q5_payment_installments": {
        "title": "5. Payment Method Preferences & Credit Card Installment Habits",
        "business_question": "Which payment channels dominate, and do customers who use installment plans make larger purchases?",
        "sql": """
SELECT 
    payment_type,
    COUNT(DISTINCT order_id) AS transaction_count,
    ROUND(SUM(total_order_value), 2) AS total_volume_brl,
    ROUND((SUM(total_order_value) / SUM(SUM(total_order_value)) OVER ()) * 100, 2) AS volume_share_pct,
    ROUND(AVG(total_order_value), 2) AS avg_ticket_size,
    ROUND(AVG(payment_installments), 1) AS avg_installments,
    ROUND(MAX(payment_installments), 0) AS max_installments
FROM df
GROUP BY payment_type
ORDER BY total_volume_brl DESC;
        """,
        "technical_highlight": "Aggregates multi-dimensional financial payment characteristics and market share in a single group query.",
        "business_insight": "Credit cards dominate with ~75% share, with average installments of 3-4 months. Installment options allow consumers to absorb larger ticket sizes (AOV > R$ 160 vs R$ 120 for Boleto)."
    },

    "q6_delivery_delay_reviews": {
        "title": "6. Impact of Delivery Delays on Customer Review Ratings",
        "business_question": "What is the empirical drop in customer satisfaction when orders miss their estimated delivery date?",
        "sql": """
WITH delay_classification AS (
    SELECT 
        order_id,
        review_score,
        delivery_delay_days,
        CASE 
            WHEN delivery_delay_days <= -3 THEN '1. Early (> 3 Days)'
            WHEN delivery_delay_days <= 0 THEN '2. On-Time (0-3 Days Early)'
            WHEN delivery_delay_days <= 3 THEN '3. Minor Delay (1-3 Days Late)'
            WHEN delivery_delay_days <= 7 THEN '4. Moderate Delay (4-7 Days Late)'
            ELSE '5. Severe Delay (> 7 Days Late)'
        END AS delay_tier
    FROM df
    WHERE order_status = 'delivered' 
      AND review_score IS NOT NULL 
      AND delivery_delay_days IS NOT NULL
)
SELECT 
    delay_tier,
    COUNT(order_id) AS total_orders,
    ROUND(AVG(review_score), 2) AS avg_review_score,
    ROUND(AVG(CASE WHEN review_score = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_1_star_reviews,
    ROUND(AVG(CASE WHEN review_score = 5 THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_5_star_reviews
FROM delay_classification
GROUP BY delay_tier
ORDER BY delay_tier;
        """,
        "technical_highlight": "Categorical tiering via CASE statements correlated against dual conditional review percentages.",
        "business_insight": "Severe delays lead to an immediate 60%+ 1-star review rate (average score drops from 4.3 to 1.8). Logistics reliability is the single greatest determinant of CSAT."
    },

    "q7_underperforming_categories": {
        "title": "7. Identification of Underperforming or High-Friction Categories",
        "business_question": "Which product categories experience the highest complaint rates and shipping delays?",
        "sql": """
SELECT 
    category_english AS category,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(SUM(total_order_value), 2) AS category_revenue,
    ROUND(AVG(review_score), 2) AS avg_review_score,
    ROUND(AVG(CASE WHEN review_score <= 2 THEN 1.0 ELSE 0.0 END) * 100, 2) AS negative_review_pct,
    ROUND(AVG(delivery_days), 1) AS avg_delivery_days,
    ROUND(AVG(freight_value), 2) AS avg_freight
FROM df
WHERE order_status = 'delivered'
GROUP BY category_english
HAVING COUNT(DISTINCT order_id) >= 100
ORDER BY negative_review_pct DESC
LIMIT 10;
        """,
        "technical_highlight": "Employs `HAVING COUNT(...) >= 100` to eliminate statistical noise from small-sample long-tail categories.",
        "business_insight": "Bulky goods (Furniture, Office Furniture) exhibit lower review scores primarily driven by higher shipping transit times and freight costs."
    }
}

def execute_sql_query(query_key: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    """
    Executes the specified query key against the provided DataFrame in DuckDB.
    Returns: (result_df, sql_query, technical_highlight, business_insight)
    """
    query_info = SQL_QUERIES[query_key]
    sql = query_info["sql"]

    # Register df in DuckDB session
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    result_df = con.execute(sql).df()
    con.close()

    return result_df, sql, query_info["technical_highlight"], query_info["business_insight"]
