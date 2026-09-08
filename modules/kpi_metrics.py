"""
KPI Calculation and Analytics Engine for Olist E-Commerce Portfolio.
Computes:
- Executive headline KPIs and MoM trends
- RFM (Recency, Frequency, Monetary) Customer Segmentation
- Pareto (80/20) Product Category Analysis
- Logistics & Review Score Correlation Metrics
"""

import pandas as pd
import numpy as np

def compute_overview_kpis(df: pd.DataFrame) -> dict:
    """Computes headline KPIs for the executive summary."""
    delivered_df = df[df["order_status"] == "delivered"] if "delivered" in df["order_status"].values else df

    total_revenue = float(delivered_df["total_order_value"].sum())
    product_revenue = float(delivered_df["price"].sum())
    freight_revenue = float(delivered_df["freight_value"].sum())
    total_orders = int(delivered_df["order_id"].nunique())
    unique_customers = int(delivered_df["customer_unique_id"].nunique())
    aov = float(total_revenue / total_orders) if total_orders > 0 else 0.0

    avg_review = float(delivered_df["review_score"].mean()) if "review_score" in delivered_df.columns else 0.0
    avg_delivery_days = float(delivered_df["delivery_days"].dropna().mean()) if "delivery_days" in delivered_df.columns else 0.0

    late_orders = delivered_df[delivered_df["is_late"] == True]["order_id"].nunique()
    late_rate = (late_orders / total_orders * 100.0) if total_orders > 0 else 0.0
    ontime_rate = 100.0 - late_rate

    # Exact Repeat customer count and rate
    customer_order_counts = delivered_df.groupby("customer_unique_id")["order_id"].nunique()
    repeat_customers = int((customer_order_counts > 1).sum())
    one_time_customers = int((customer_order_counts == 1).sum())
    repeat_rate = (repeat_customers / unique_customers * 100.0) if unique_customers > 0 else 0.0

    # Calculate MoM revenue growth for latest complete months
    monthly_rev = delivered_df.groupby("order_year_month")["total_order_value"].sum().sort_index()
    if len(monthly_rev) >= 2:
        latest_rev = monthly_rev.iloc[-1]
        prev_rev = monthly_rev.iloc[-2]
        mom_growth = ((latest_rev - prev_rev) / prev_rev * 100.0) if prev_rev > 0 else 0.0
    else:
        mom_growth = 0.0

    return {
        "total_revenue": total_revenue,
        "product_revenue": product_revenue,
        "freight_revenue": freight_revenue,
        "total_orders": total_orders,
        "unique_customers": unique_customers,
        "repeat_customers": repeat_customers,
        "one_time_customers": one_time_customers,
        "aov": aov,
        "avg_review": avg_review,
        "avg_delivery_days": avg_delivery_days,
        "ontime_rate": ontime_rate,
        "late_rate": late_rate,
        "repeat_rate": repeat_rate,
        "mom_growth": mom_growth,
    }

def compute_rfm_segments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs RFM (Recency, Frequency, Monetary) segmentation.
    Returns:
    - customer_rfm_df: individual customer scores and segment labels
    - segment_summary_df: aggregate metrics with formatted presentation columns
    """
    valid_df = df.dropna(subset=["order_purchase_timestamp", "customer_unique_id"]).copy()
    ref_date = valid_df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = valid_df.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": lambda x: (ref_date - x.max()).days,
        "order_id": "nunique",
        "total_order_value": "sum"
    }).reset_index()

    rfm.columns = ["customer_unique_id", "recency", "frequency", "monetary"]

    # Calculate quartiles for scoring
    r_labels = [4, 3, 2, 1]
    m_labels = [1, 2, 3, 4]

    rfm["R_score"] = pd.qcut(rfm["recency"].rank(method="first"), q=4, labels=r_labels).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=4, labels=m_labels).astype(int)

    def score_frequency(f):
        if f >= 3:
            return 4
        elif f == 2:
            return 3
        else:
            return 2  # Baseline single order

    rfm["F_score"] = rfm["frequency"].apply(score_frequency)
    rfm["RFM_Score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)

    def assign_segment(row):
        r = row["R_score"]
        f = row["F_score"]
        m = row["M_score"]

        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        elif r >= 2 and f >= 3:
            return "Loyal Customers"
        elif r >= 3 and f == 2 and m >= 3:
            return "Potential Loyalists"
        elif r >= 3 and f == 2 and m <= 2:
            return "Recent Buyers"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f == 2 and m >= 3:
            return "Cant Lose Them"
        elif r <= 2 and f == 2 and m <= 2:
            return "Hibernating"
        else:
            return "Lost / Inactive"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    seg_summary = rfm.groupby("Segment").agg(
        Customer_Count=("customer_unique_id", "count"),
        Avg_Recency=("recency", "mean"),
        Avg_Frequency=("frequency", "mean"),
        Avg_Spend=("monetary", "mean"),
        Total_Revenue=("monetary", "sum")
    ).reset_index()

    total_customers = rfm["customer_unique_id"].nunique()
    total_rev = rfm["monetary"].sum()

    seg_summary["Customer_Pct"] = (seg_summary["Customer_Count"] / total_customers * 100.0).round(1)
    seg_summary["Revenue_Pct"] = (seg_summary["Total_Revenue"] / total_rev * 100.0).round(1)
    seg_summary["Avg_Recency"] = seg_summary["Avg_Recency"].round(0)
    seg_summary["Avg_Spend"] = seg_summary["Avg_Spend"].round(2)
    seg_summary["Total_Revenue"] = seg_summary["Total_Revenue"].round(2)

    strategies = {
        "Champions": "VIP loyalty perks, early access to new lines, ambassador referrals.",
        "Loyal Customers": "Upsell premium bundles, personalized category recommendations.",
        "Potential Loyalists": "Offer membership incentives, engage with post-purchase rewards.",
        "Recent Buyers": "Provide high-touch onboarding, trigger second-purchase discount in 14 days.",
        "At Risk": "Aggressive win-back campaigns, limited-time renewal coupons, feedback survey.",
        "Cant Lose Them": "Direct outreach, premium concierge support, high-value retention voucher.",
        "Hibernating": "Low-cost email drip re-engagement campaigns, seasonal flash clearance.",
        "Lost / Inactive": "Exclude from high-cost ad targeting; seasonal reactivation attempts only."
    }
    seg_summary["Strategic_Action"] = seg_summary["Segment"].map(strategies).fillna("Standard engagement")
    seg_summary = seg_summary.sort_values(by="Total_Revenue", ascending=False)

    return rfm, seg_summary

def compute_pareto_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Pareto (80/20) analysis on product categories.
    Identifies which categories generate the top 80% of total revenue.
    """
    cat_rev = df.groupby("category_english")["total_order_value"].sum().reset_index()
    cat_rev.columns = ["Category", "Revenue"]
    cat_rev = cat_rev.sort_values(by="Revenue", ascending=False).reset_index(drop=True)

    total_rev = cat_rev["Revenue"].sum()
    cat_rev["Cumulative_Revenue"] = cat_rev["Revenue"].cumsum()
    cat_rev["Cumulative_Pct"] = (cat_rev["Cumulative_Revenue"] / total_rev * 100.0).round(2)
    cat_rev["Revenue_Share_Pct"] = (cat_rev["Revenue"] / total_rev * 100.0).round(2)
    cat_rev["Is_Top_80"] = cat_rev["Cumulative_Pct"] <= 80.0

    return cat_rev

def compute_logistics_satisfaction_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes delivery delay vs review score correlations.
    Returns:
    - delay_bins_df: review score distribution across delay buckets
    - state_logistics_df: delivery performance and review ratings by Brazilian state
    """
    clean_df = df.dropna(subset=["delivery_delay_days", "review_score"]).copy()

    def categorize_delay(days):
        if days <= -3:
            return "Early (>3 days)"
        elif days <= 0:
            return "On-Time (0-3 days early)"
        elif days <= 3:
            return "Slight Delay (1-3 days late)"
        elif days <= 7:
            return "Moderate Delay (4-7 days late)"
        else:
            return "Severe Delay (>7 days late)"

    clean_df["Delay_Bucket"] = clean_df["delivery_delay_days"].apply(categorize_delay)

    bucket_order = [
        "Early (>3 days)",
        "On-Time (0-3 days early)",
        "Slight Delay (1-3 days late)",
        "Moderate Delay (4-7 days late)",
        "Severe Delay (>7 days late)"
    ]

    delay_bins = clean_df.groupby("Delay_Bucket").agg(
        Order_Count=("order_id", "nunique"),
        Avg_Review_Score=("review_score", "mean"),
        Pct_1_Star=("review_score", lambda x: (x == 1).mean() * 100.0),
        Pct_5_Star=("review_score", lambda x: (x == 5).mean() * 100.0)
    ).reindex(bucket_order).reset_index()

    delay_bins["Avg_Review_Score"] = delay_bins["Avg_Review_Score"].round(2)
    delay_bins["Pct_1_Star"] = delay_bins["Pct_1_Star"].round(1)
    delay_bins["Pct_5_Star"] = delay_bins["Pct_5_Star"].round(1)

    # State performance
    state_logistics = clean_df.groupby("customer_state").agg(
        Total_Orders=("order_id", "nunique"),
        Total_Revenue=("total_order_value", "sum"),
        Avg_Delivery_Days=("delivery_days", "mean"),
        Late_Delivery_Pct=("is_late", lambda x: x.mean() * 100.0),
        Avg_Freight_Cost=("freight_value", "mean"),
        Avg_Review_Score=("review_score", "mean")
    ).reset_index()

    state_logistics["Avg_Delivery_Days"] = state_logistics["Avg_Delivery_Days"].round(1)
    state_logistics["Late_Delivery_Pct"] = state_logistics["Late_Delivery_Pct"].round(1)
    state_logistics["Avg_Freight_Cost"] = state_logistics["Avg_Freight_Cost"].round(2)
    state_logistics["Avg_Review_Score"] = state_logistics["Avg_Review_Score"].round(2)
    state_logistics["Total_Revenue"] = state_logistics["Total_Revenue"].round(2)
    state_logistics = state_logistics.sort_values(by="Total_Revenue", ascending=False)

    return delay_bins, state_logistics
