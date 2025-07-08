import streamlit as st
import pandas as pd
import random
import uuid
import numpy as np
import time
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()

# Static Pools (with realistic skewing)
device_types_pool = ['mobile', 'desktop', 'tablet']
payment_methods = ['card', 'UPI', 'COD', 'wallet']
product_categories = [
    'Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Toys',
    'Beauty & Personal Care', 'Sports & Fitness', 'Grocery', 'Automotive', 'Health & Wellness'
]
customer_segments_pool = ['new', 'returning', 'loyal', 'high-spender', 'inactive']
locations_pool = [
    'Jaipur', 'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Pune', 'Chandigarh', 'Lucknow', 'Ahmedabad', 'Kolkata'
]

products = [
    {'product_id': f"P{str(i).zfill(3)}", 'category': random.choice(product_categories)}
    for i in range(1, 51)
]

# Review Pools
positive_reviews = [
    "Absolutely love this product!", "Great value for the price.", "Super fast delivery and well packed.",
    "Exceeded my expectations.", "Top quality, will buy again.", "Works perfectly, no complaints.",
    "Five stars — highly recommend.", "Exactly what I needed, thanks!", "Amazing experience from start to finish.",
    "Build quality is excellent.", "Delivered before time, very satisfied.", "High-end feel, worth the price.",
    "Great packaging and fast delivery.", "Fits perfectly and looks great.", "Performs better than expected.",
    "Setup was a breeze.", "Customer support was fantastic.", "Highly durable and stylish.",
    "Very reliable and efficient.", "Battery life is amazing.", "Truly plug and play.",
    "Incredible sound quality.", "Comfortable and light weight.", "Super intuitive interface.",
    "Color exactly as shown.", "Loved the extras included.", "A+ on delivery and service.",
    "Fast and secure checkout.", "Will recommend to friends.", "Great gift item.",
    "Fantastic resolution.", "Stunning build quality.", "Best purchase of the month.",
    "Value for money.", "Top tier performance.", "User-friendly and powerful.",
    "Just what I needed.", "Highly useful in daily life.", "Appreciated the updates.",
    "Exactly as described.", "Practical and elegant.", "Delivery was surprisingly fast.",
    "Hassle-free experience.", "Loved the finish.", "Good customer service.",
    "Very compact yet efficient.", "Sturdy and dependable.", "Handles multitasking well.",
    "A joy to use.", "No regrets buying this."
]

neutral_reviews = [
    "It's okay, nothing special.", "Average experience overall.", "Product is fine, but not amazing.",
    "Does what it says, no more no less.", "Decent product for short-term use.", "Not bad, but not great either.",
    "Packaging could’ve been better.", "Arrived as expected, nothing extra.", "Satisfactory but can be improved.",
    "Fairly standard, nothing to rave about.", "Works as intended.", "Quality meets expectations.",
    "Design is average.", "Had to tweak a few things.", "Looks okay.", "Affordable, but basic.",
    "Not the best, not the worst.", "Some minor flaws.", "Usable but not ideal.",
    "Acceptable at this price.", "Met my minimum needs.", "Bare bones but works.",
    "Just functional.", "Could be better.", "Color is slightly off.",
    "Shipping was a bit slow.", "Expected a bit more.", "Not as sleek as shown.",
    "Needed some adjustments.", "Neutral about this.", "Okay for casual use.",
    "Sound is average.", "Display is fine.", "Speed is not too bad.",
    "Durability is questionable.", "Needs better packaging.", "Looks a bit outdated.",
    "Performance fluctuates.", "Nothing impressive.", "It's passable.",
    "Standard product.", "Average UI experience.", "No strong feelings.",
    "Works but could be better.", "Didn’t wow me.", "Minimalist design."
]

negative_reviews = [
    "Very poor quality, not recommended.", "Disappointed, expected better.", "Stopped working after a week.",
    "Looks used, not new.", "Customer service was unhelpful.", "Waste of money — avoid this.",
    "Cheap build, definitely not durable.", "Received damaged product.", "Battery drains too fast.",
    "Instructions were unclear.", "Faulty unit received.", "Keeps disconnecting.",
    "Colors look faded.", "Extremely noisy operation.", "Material feels flimsy.",
    "Doesn’t perform as advertised.", "Overheats quickly.", "Returned it same day.",
    "Delivery was delayed a week.", "Had to call support multiple times.",
    "Not compatible as claimed.", "Felt like a downgrade.", "Full of bugs.",
    "Missing parts in the package.", "Heats up too much.", "Laggy interface.",
    "Not suitable for daily use.", "Wrong item delivered.", "No support documentation.",
    "Way overpriced.", "Doesn't hold charge.", "Build is very weak.",
    "Sound is distorted.", "Display flickers.", "Gets stuck frequently.",
    "Looks different than pictures.", "Hard to install.", "Stopped charging.",
    "Keeps freezing.", "Too many software issues.", "Plastic feels cheap.",
    "Doesn’t live up to the hype.", "Lacks essential features.", "Shipping box was crushed.",
    "Awful customer experience.", "Low-quality components."
]

# --- Timestamp State ---
if 'start_time' not in st.session_state:
    st.session_state['start_time'] = datetime.now() - timedelta(days=30)
if 'row_index' not in st.session_state:
    st.session_state['row_index'] = 0
if 'time_range' not in st.session_state:
    st.session_state['time_range'] = (datetime.now() - st.session_state['start_time']).total_seconds()

# --- Generate Row ---
def generate_row(existing_combinations):
    while True:
        product = random.choice(products)
        session_id = str(uuid.uuid4())
        if (product['product_id'], session_id) not in existing_combinations:
            existing_combinations.add((product['product_id'], session_id))
            break

    location = random.choices(locations_pool, weights=[0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03, 0.04, 0.04])[0]
    device_type = random.choices(device_types_pool, weights=[0.65, 0.25, 0.1])[0]
    customer_segment = random.choices(customer_segments_pool, weights=[0.35, 0.4, 0.15, 0.05, 0.05])[0]

    if product['category'] == 'Electronics':
        price = round(random.uniform(500, 50000), 2)
    elif product['category'] == 'Grocery':
        price = round(random.uniform(50, 500), 2)
    else:
        price = round(random.uniform(200, 20000), 2)

    discount = round(random.triangular(0.2, 0.5, 0.3), 2)
    final_price = round(price * (1 - discount), 2)
    added_to_cart = random.choice([True, False])
    purchase_made = random.choices([True, False], weights=[0.35, 0.65])[0]

    rating = None
    review_text = None
    sentiment = None
    review_length = 0

    if purchase_made:
        sentiment = random.choices(['positive', 'neutral', 'negative'], weights=[0.5, 0.3, 0.2])[0]
        if sentiment == 'positive':
            review_text = random.choice(positive_reviews)
            rating = random.choices([4, 5], weights=[0.3, 0.7])[0]
        elif sentiment == 'neutral':
            review_text = random.choice(neutral_reviews)
            rating = random.choices([2, 3], weights=[0.4, 0.6])[0]
        else:
            review_text = random.choice(negative_reviews)
            rating = random.choices([1, 2], weights=[0.7, 0.3])[0]
        review_length = len(review_text) if review_text else 0

    if location in ['Jaipur', 'Lucknow', 'Chandigarh']:
        delivery_time = random.randint(5, 10)
        shipping_cost = round(random.uniform(100, 500), 2)
    else:
        delivery_time = random.randint(1, 7)
        shipping_cost = round(random.uniform(20, 200), 2)

    total_paid = round(final_price + shipping_cost, 2) if purchase_made else 0

    return_rate_map = {
        'Clothing': (0.1, 0.4, 0.25),
        'Electronics': (0.0, 0.2, 0.1),
        'Books': (0.0, 0.1, 0.05),
        'Automotive': (0.0, 0.05, 0.01),
        'Sports & Fitness': (0.05, 0.15, 0.08)
    }
    product_return_rate = round(random.triangular(*return_rate_map.get(product['category'], (0.0, 0.3, 0.1))), 2)
    purchase_prob = round(0.4 + 0.6 * discount, 2) if discount > 0.3 else round(0.2 + 0.5 * discount, 2)

    session_duration = random.randint(30, 1200)
    user_activity_score = round(session_duration / 100 + random.uniform(1, 10), 2)
    user_interaction_score = round(random.uniform(1, 10) + session_duration / 300, 2)

    category_trend = {
        'Electronics': round(random.uniform(-0.1, 0.1), 2),
        'Clothing': round(random.uniform(-0.2, 0.2), 2),
        'Sports & Fitness': round(random.uniform(-0.3, 0.3), 2)
    }.get(product['category'], round(random.uniform(-0.1, 0.1), 2))

    engagement_score = 0  # Initialize engagement_score
    if rating is not None:
        engagement_score = round(random.uniform(0.5, 1.5) * (rating + random.randint(1, 5)), 2)
    else:
        engagement_score = round(random.uniform(0.5, 1.5) * random.randint(1, 5), 2) # Or some other default logic
    clv = round(final_price * random.randint(1, 10), 2)
    discount_sensitivity = round(discount * 100, 2)
    price_sensitivity = round(random.uniform(0.2, 1.0), 2)
    product_rating_to_review_ratio = round((rating or 0) / (review_length / 20) if review_length > 0 else 0, 2)
    session_id_count = random.randint(1, 5)
    product_views = random.randint(1, 20)
    purchase_frequency = random.randint(1, 10)
    customer_engagement = round(engagement_score * 1.2, 2)
    delivery_status = random.choices(['on-time', 'delayed'], weights=[0.85, 0.15])[0]
    cart_to_purchase_ratio = round(random.uniform(0.3, 0.9), 2)
    review_quality_score = round(review_length * 0.1, 2)

    new_timestamp = st.session_state['start_time'] + timedelta(
        seconds=int(st.session_state['row_index'] * (st.session_state['time_range'] / 100000)))

    return {
        'timestamp': new_timestamp,
        'session_id': session_id,
        'user_id': f"U{random.randint(1000, 9999)}",
        'location': location,
        'device_type': device_type,
        'product_id': product['product_id'],
        'product_category': product['category'],
        'price': price,
        'discount': discount,
        'final_price': final_price,
        'shipping_cost': shipping_cost if purchase_made else 0,
        'total_paid': total_paid,
        'views': product_views,
        'added_to_cart': added_to_cart,
        'cart_duration_sec': random.randint(0, 900) if added_to_cart else 0,
        'purchase_made': purchase_made,
        'payment_method': random.choice(payment_methods) if purchase_made else 'N/A',
        'return_requested': random.choice([True, False]) if purchase_made else False,
        'delivery_time_days': delivery_time if purchase_made else 0,
        'customer_segment': customer_segment,
        'rating': rating,
        'review_text': review_text,
        'review_sentiment': sentiment,
        'engagement_score': engagement_score,
        'session_duration': session_duration,
        'clv': clv,
        'discount_sensitivity': discount_sensitivity,
        'price_sensitivity': price_sensitivity,
        'product_rating_to_review_ratio': product_rating_to_review_ratio,
        'user_interaction_score': user_interaction_score,
        'session_id_count': session_id_count,
        'purchase_prob': purchase_prob,
        'product_return_rate': product_return_rate,
        'purchase_frequency': purchase_frequency,
        'customer_engagement': customer_engagement,
        'category_trend': category_trend,
        'delivery_status': delivery_status if purchase_made else 'N/A',
        'cart_to_purchase_ratio': cart_to_purchase_ratio,
        'user_activity_score': user_activity_score,
        'review_length': review_length,
        'review_quality_score': review_quality_score
    }

# Init session state
if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False
if 'stream_buffer' not in st.session_state:
    st.session_state['stream_buffer'] = []
if 'existing_combinations' not in st.session_state:
    st.session_state['existing_combinations'] = set()

# --- Title ---
st.title("📡 Live E-Commerce Data Stream")

# --- Placeholders ---
kpi_placeholder = st.empty()
live_placeholder = st.empty()

# --- Controls ---
col1, col2 = st.columns(2)
row_delay = col1.slider("⏱ Delay per row (sec)", 0.1, 2.0, 0.5, step=0.1)
total_rows = col2.slider("📦 Total rows to stream", 1, 100, 20)

col1.button("▶️ Start", on_click=lambda: st.session_state.update({'is_running': True}), disabled=st.session_state['is_running'])
col2.button("⏹️ Stop", on_click=lambda: st.session_state.update({'is_running': False}), disabled=not st.session_state['is_running'])

# --- KPI function ---
def calculate_kpis(df):
    total_sales = df['total_paid'].sum()
    total_purchases = df['purchase_made'].sum()
    avg_engagement = df['customer_engagement'].mean()
    avg_rating = df['rating'].mean()
    return total_sales, total_purchases, avg_engagement, avg_rating

# --- File path ---
csv_path = r"D:\destop file\get info\python\imarticus project\Imarticus Data Science Internship - Assessment_by_Amir_Khan\ecommerce_data (Generated data).csv"

# --- Streaming logic ---
if st.session_state['is_running']:
    for i in range(total_rows):
        if not st.session_state['is_running']:
            st.warning("⏹️ Stopped by user.")
            break

        row = generate_row(st.session_state['existing_combinations'])
        st.session_state['row_index'] += 1
        row_df = pd.DataFrame([row])

        # Append to CSV
        if os.path.exists(csv_path):
            row_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            row_df.to_csv(csv_path, index=False)

        st.session_state['stream_buffer'].insert(0, row)
        stream_df = pd.DataFrame(st.session_state['stream_buffer'][:10])

        # ✅ Safer read_csv
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df_full = pd.read_csv(csv_path)
            if 'timestamp' in df_full.columns:
                df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], errors='coerce')
        else:
            df_full = pd.DataFrame()

        total_sales, total_purchases, avg_engagement, avg_rating = calculate_kpis(df_full)

        with kpi_placeholder.container():
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("💰 Total Sales", f"₹{total_sales:,.2f}")
            kpi2.metric("🛒 Total Purchases", f"{int(total_purchases)}")
            kpi3.metric("🔥 Avg. Engagement", f"{avg_engagement:.2f}")
            kpi4.metric("⭐ Avg. Rating", f"{avg_rating:.2f}")

        live_placeholder.markdown("### 🆕 Newest Streamed Rows")
        live_placeholder.dataframe(stream_df, use_container_width=True)

        time.sleep(row_delay)

    st.success("✅ Streaming complete.")
    st.session_state['is_running'] = False

# --- When simulation not running, load recent
elif os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
    df_full = pd.read_csv(csv_path)
    if 'timestamp' in df_full.columns:
        df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], errors='coerce')

    total_sales, total_purchases, avg_engagement, avg_rating = calculate_kpis(df_full)

    with kpi_placeholder.container():
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("💰 Total Sales", f"₹{total_sales:,.2f}")
        kpi2.metric("🛒 Total Purchases", f"{int(total_purchases)}")
        kpi3.metric("🔥 Avg. Engagement", f"{avg_engagement:.2f}")
        kpi4.metric("⭐ Avg. Rating", f"{avg_rating:.2f}")

    live_placeholder.markdown("### 🆕 Last Streamed Rows")
    history = df_full.tail(10)
    live_placeholder.dataframe(history, use_container_width=True)

else:
    st.info("Click ▶️ Start to begin live data generation.")
