"""
Visualizer Engine for Olist E-Commerce Portfolio.
Generates polished, responsive, and recruiter-ready Plotly visualizations
utilizing modern color palettes, typography, and clear annotations.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Executive color palette
PALETTE = {
    "primary": "#3B82F6",      # Bright Blue
    "secondary": "#10B981",    # Emerald
    "accent": "#F59E0B",       # Amber
    "danger": "#EF4444",       # Rose Red
    "purple": "#8B5CF6",       # Violet
    "slate": "#64748B",        # Slate
    "bg_card": "rgba(255, 255, 255, 0.05)",
    "grid": "rgba(150, 150, 150, 0.15)"
}

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified"
)

def plot_monthly_revenue_and_orders(df: pd.DataFrame) -> go.Figure:
    """Creates a dual-axis interactive chart for monthly gross revenue and order volume."""
    monthly = df.groupby("order_year_month").agg(
        Revenue=("total_order_value", "sum"),
        Orders=("order_id", "nunique")
    ).reset_index().sort_values("order_year_month")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Revenue Bars
    fig.add_trace(
        go.Bar(
            x=monthly["order_year_month"],
            y=monthly["Revenue"],
            name="Gross Revenue (R$)",
            marker=dict(
                color=monthly["Revenue"],
                colorscale="Blues",
                showscale=False,
                line=dict(width=0)
            ),
            hovertemplate="<b>%{x}</b><br>Revenue: R$ %{y:,.2f}<extra></extra>"
        ),
        secondary_y=False,
    )

    # Order Volume Line
    fig.add_trace(
        go.Scatter(
            x=monthly["order_year_month"],
            y=monthly["Orders"],
            name="Delivered Orders",
            mode="lines+markers",
            line=dict(color="#10B981", width=3),
            marker=dict(size=7, color="#10B981"),
            hovertemplate="Orders: %{y:,}<extra></extra>"
        ),
        secondary_y=True,
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=None,
        margin=dict(l=40, r=40, t=45, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(title="Month", showgrid=False, tickangle=-45),
        yaxis=dict(title="Revenue (R$)", showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis2=dict(title="Order Count", showgrid=False, overlaying="y", side="right")
    )
    return fig

def plot_state_distribution(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing top 10 states by revenue and freight friction."""
    state_df = df.groupby("customer_state").agg(
        Revenue=("total_order_value", "sum"),
        Orders=("order_id", "nunique"),
        Avg_Freight=("freight_value", "mean")
    ).reset_index().sort_values("Revenue", ascending=True).tail(10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=state_df["customer_state"],
        x=state_df["Revenue"],
        orientation="h",
        marker=dict(
            color=state_df["Avg_Freight"],
            colorscale="Viridis",
            colorbar=dict(title="Avg Freight (R$)", len=0.8),
            showscale=True
        ),
        hovertemplate="<b>State: %{y}</b><br>Revenue: R$ %{x:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=None,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="Total Revenue (R$)", showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(title="State")
    )
    return fig

def plot_rfm_treemap(seg_summary: pd.DataFrame) -> go.Figure:
    """Treemap of RFM customer segments."""
    fig = px.treemap(
        seg_summary,
        path=["Segment"],
        values="Total_Revenue",
        color="Avg_Spend",
        color_continuous_scale="Blues",
        hover_data=["Customer_Count", "Customer_Pct", "Revenue_Pct", "Avg_Spend"]
    )
    fig.update_layout(**LAYOUT_DEFAULTS, margin=dict(l=20, r=20, t=20, b=20))
    fig.update_traces(
        textinfo="label+value+percent root",
        hovertemplate="<b>%{label}</b><br>Revenue: R$ %{value:,.2f}<br>Avg Spend: R$ %{color:.2f}<extra></extra>"
    )
    return fig

def plot_rfm_scatter(rfm_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of Recency vs Monetary value colored by Segment."""
    # Subsample to 2,000 for fast browser rendering if needed
    sample_df = rfm_df.sample(min(2000, len(rfm_df)), random_state=42) if len(rfm_df) > 2000 else rfm_df

    color_map = {
        "Champions": "#10B981",
        "Loyal Customers": "#3B82F6",
        "Potential Loyalists": "#06B6D4",
        "Recent Buyers": "#8B5CF6",
        "At Risk": "#F59E0B",
        "Cant Lose Them": "#EC4899",
        "Hibernating": "#64748B",
        "Lost / Inactive": "#EF4444"
    }

    fig = px.scatter(
        sample_df,
        x="recency",
        y="monetary",
        color="Segment",
        color_discrete_map=color_map,
        log_y=True,
        opacity=0.75,
        hover_data=["customer_unique_id", "frequency"]
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        margin=dict(l=40, r=130, t=20, b=40),
        xaxis=dict(title="Recency (Days Since Last Order - Lower is More Recent)", showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(title="Monetary Spend (R$ Log Scale)", showgrid=True, gridcolor=PALETTE["grid"]),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
    )
    return fig

def plot_pareto_curve(pareto_df: pd.DataFrame) -> go.Figure:
    """Pareto 80/20 cumulative revenue distribution curve."""
    top_n = pareto_df.head(20).copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=top_n["Category"],
            y=top_n["Revenue"],
            name="Category Revenue (R$)",
            marker=dict(color="#3B82F6"),
            hovertemplate="<b>%{x}</b><br>Revenue: R$ %{y:,.2f}<extra></extra>"
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=top_n["Category"],
            y=top_n["Cumulative_Pct"],
            name="Cumulative %",
            mode="lines+markers",
            line=dict(color="#EF4444", width=3),
            marker=dict(size=6, color="#EF4444"),
            hovertemplate="Cumulative: %{y:.1f}%<extra></extra>"
        ),
        secondary_y=True
    )

    # 80% threshold line
    fig.add_hline(
        y=80.0,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="80% Threshold",
        annotation_position="bottom right",
        secondary_y=True
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=None,
        margin=dict(l=40, r=40, t=45, b=95),
        xaxis=dict(title="Product Category", showgrid=False, tickangle=-45),
        yaxis=dict(title="Revenue (R$)", showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis2=dict(title="Cumulative %", showgrid=False, range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

def plot_delivery_delay_impact(delay_bins: pd.DataFrame) -> go.Figure:
    """Grouped chart illustrating review score and 1-star review spikes vs delivery delays."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1-star percentage bars
    fig.add_trace(
        go.Bar(
            x=delay_bins["Delay_Bucket"],
            y=delay_bins["Pct_1_Star"],
            name="1-Star Reviews (%)",
            marker=dict(color="#EF4444"),
            hovertemplate="1-Star Complaints: %{y:.1f}%<extra></extra>"
        ),
        secondary_y=False
    )

    # 5-star percentage bars
    fig.add_trace(
        go.Bar(
            x=delay_bins["Delay_Bucket"],
            y=delay_bins["Pct_5_Star"],
            name="5-Star Reviews (%)",
            marker=dict(color="#10B981"),
            hovertemplate="5-Star Delight: %{y:.1f}%<extra></extra>"
        ),
        secondary_y=False
    )

    # Average review score line
    fig.add_trace(
        go.Scatter(
            x=delay_bins["Delay_Bucket"],
            y=delay_bins["Avg_Review_Score"],
            name="Avg Review Score (⭐)",
            mode="lines+markers",
            line=dict(color="#F59E0B", width=4),
            marker=dict(size=10, symbol="diamond", color="#F59E0B"),
            hovertemplate="Avg Score: %{y:.2f} ⭐<extra></extra>"
        ),
        secondary_y=True
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        barmode="group",
        title=None,
        margin=dict(l=40, r=40, t=45, b=65),
        xaxis=dict(title="Delivery Timeliness Relative to Estimated Date", showgrid=False),
        yaxis=dict(title="% of Total Reviews in Tier", showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis2=dict(title="Average Review Score (Stars)", range=[1, 5.2], showgrid=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    return fig

def plot_payment_breakdown(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """Donut chart of payment methods + histogram of installment counts."""
    # Donut
    pay_agg = df.groupby("payment_type")["total_order_value"].sum().reset_index()
    donut_fig = px.pie(
        pay_agg,
        names="payment_type",
        values="total_order_value",
        hole=0.55,
        color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    )
    donut_fig.update_layout(
        **LAYOUT_DEFAULTS,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    donut_fig.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Total: R$ %{value:,.2f}<br>Share: %{percent}<extra></extra>"
    )

    # Installment distribution for credit card
    cc_df = df[df["payment_type"] == "credit_card"].copy()
    cc_df["payment_installments"] = pd.to_numeric(cc_df["payment_installments"], errors="coerce").fillna(1).astype(int)
    
    # Ensure every installment from 1 to 10 is present in the distribution
    inst_series = cc_df["payment_installments"].value_counts().reindex(range(1, 11), fill_value=0)
    inst_counts = inst_series.reset_index()
    inst_counts.columns = ["Installments", "Count"]
    inst_counts["Label"] = inst_counts["Installments"].astype(str) + "x"
    total_cc_orders = max(1, inst_counts["Count"].sum())
    inst_counts["Pct"] = (inst_counts["Count"] / total_cc_orders * 100.0).round(1)

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Bar(
        x=inst_counts["Label"],
        y=inst_counts["Count"],
        marker=dict(
            color=inst_counts["Count"],
            colorscale="Blues",
            line=dict(color="#3B82F6", width=1)
        ),
        hovertemplate="<b>%{x} Installments</b><br>Orders: %{y:,}<br>Share: %{customdata:.1f}%<extra></extra>",
        customdata=inst_counts["Pct"]
    ))
    hist_fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=None,
        margin=dict(l=40, r=30, t=30, b=40),
        xaxis=dict(title="Installment Plan (1x to 10x)", showgrid=False),
        yaxis=dict(title="Number of Orders", showgrid=True, gridcolor=PALETTE["grid"])
    )

    return donut_fig, hist_fig
