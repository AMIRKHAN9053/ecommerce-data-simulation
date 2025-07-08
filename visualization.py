import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊E-Commerce Visual Insights")

# Load Data
df = pd.read_csv("D:\destop file\get info\python\imarticus project\Imarticus Data Science Internship - Assessment_by_Amir_Khan\ecommerce_data (Generated data).csv", parse_dates=["timestamp"])
df["date"] = pd.to_datetime(df["timestamp"]).dt.date

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Your View")
category_filter = st.sidebar.multiselect("Product Category", df["product_category"].unique(), default=df["product_category"].unique())
location_filter = st.sidebar.multiselect("Location", df["location"].unique(), default=df["location"].unique())
segment_filter = st.sidebar.multiselect("Customer Segment", df["customer_segment"].unique(), default=df["customer_segment"].unique())
start_date = st.sidebar.date_input("Start Date", df["date"].min())
end_date = st.sidebar.date_input("End Date", df["date"].max())

# Filter Data
filtered_df = df[
    (df["product_category"].isin(category_filter)) &
    (df["location"].isin(location_filter)) &
    (df["customer_segment"].isin(segment_filter)) &
    (df["date"] >= start_date) & (df["date"] <= end_date)
]

# --- KPIs ---
st.subheader("📈 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Revenue", f"₹{filtered_df['total_paid'].sum():,.0f}")
col2.metric("Conversion Rate", f"{filtered_df['purchase_made'].mean():.2%}")
col3.metric("Return Rate", f"{filtered_df['return_requested'].mean():.2%}")
col4.metric("Avg CLV", f"₹{filtered_df['clv'].mean():,.0f}")
col5.metric("Avg Session Duration", f"{filtered_df['session_duration'].mean():.0f}s")

# --- Sales Over Time ---
st.subheader("📅 Revenue Over Time")

# Ensure data has purchases only and correctly grouped by date
revenue_trend = (
    filtered_df[filtered_df["purchase_made"] == True]
    .groupby("date", as_index=False)["total_paid"]
    .sum()
    .sort_values("date")
)

# Handle empty data gracefully
if not revenue_trend.empty:
    fig_revenue = px.line(
        revenue_trend,
        x="date",
        y="total_paid",
        markers=True,
        title="Revenue Trend Over Time"
    )
    fig_revenue.update_layout(xaxis_title="Date", yaxis_title="Total Revenue (₹)")
    st.plotly_chart(fig_revenue, use_container_width=True)
else:
    st.info("No revenue data available for selected filters. Adjust your filters to see the revenue trend.")


# --- Geo Sales by Location ---
st.subheader("📍 Sales by Location")
location_sales = filtered_df[filtered_df["purchase_made"] == True].groupby("location")["total_paid"].sum().reset_index()
fig_map = px.bar(location_sales, x="location", y="total_paid", color="total_paid", title="Revenue by Location", color_continuous_scale="greens")
st.plotly_chart(fig_map, use_container_width=True)

# --- Cart vs Purchase Analysis ---
st.subheader("🛒 Cart Abandonment vs Purchases")
cart_stats = filtered_df.groupby("product_category").agg(
    carts=("added_to_cart", "sum"),
    purchases=("purchase_made", "sum")
).reset_index()
fig_cart = px.bar(cart_stats, x="product_category", y=["carts", "purchases"], barmode="group", title="Cart Additions vs Purchases")
st.plotly_chart(fig_cart, use_container_width=True)

st.subheader("🎯 Average Price vs Purchase by Product Category")
price_vs_purchase = df.groupby("product_category")[["price", "purchase_made"]].mean().reset_index()
fig_price_vs_purchase = px.scatter(price_vs_purchase, x="price", y="purchase_made", color="product_category", 
                                   title="Average Price vs Purchase Likelihood by Product Category", color_continuous_scale="cividis")
st.plotly_chart(fig_price_vs_purchase, use_container_width=True)


# --- Returns vs Purchases ---
st.subheader("🔁 Returns vs Purchases by Category")
return_stats = filtered_df.groupby("product_category").agg(
    purchases=("purchase_made", "sum"),
    returns=("return_requested", "sum")
).reset_index()
fig_return = px.bar(return_stats, x="product_category", y=["purchases", "returns"], barmode="group", title="Returns vs Purchases")
st.plotly_chart(fig_return, use_container_width=True)

st.markdown("---")
st.caption("📊 Built by Amir khan | Imarticus Data Science Internship Project")
