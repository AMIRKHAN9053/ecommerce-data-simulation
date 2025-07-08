import streamlit as st
from streamlit_lottie import st_lottie
import requests

# ---- Helper to Load Lottie ----
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---- Page Config ----
st.set_page_config(page_title="E-Commerce ML Dashboard", layout="wide")

# ---- Animated Title ----
st.markdown("""
    <style>
    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(-10px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: bold;
        animation: fadeIn 1s ease-in-out;
        color: #4CAF50;
    }
    </style>
    <div class='main-title'>📦 Smart Retail ML Dashboard</div>
""", unsafe_allow_html=True)

# ---- Sidebar with Animation ----
lottie_dashboard = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")
with st.sidebar:
    st_lottie(lottie_dashboard, height=140, key="dash")
    st.markdown("---")
    selected = st.radio("📍 Navigate to", ["Live Data", "Visual Insights", "Predictions", "Chatbot"])

# ---- Route Pages ----
if selected == "Live Data":
    st.markdown("""
        <h2 style='color:#2E86C1;'>📊 Live Sensor Simulation & Real-Time Anomalies</h2>
    """, unsafe_allow_html=True)
    with open("pages/ecommerce-model.py", encoding="utf-8") as f:
        code_live = f.read().splitlines()
        exec("\n".join([line for line in code_live if not line.strip().startswith("st.set_page_config")]), globals())


elif selected == "Visual Insights":
    st.markdown("""
        <h2 style='color:#8E44AD;'>📈 Visual Exploration of Product & User Trends</h2>
    """, unsafe_allow_html=True)
    with open("pages/visualization.py", encoding="utf-8") as f:
        code_vis = f.read().splitlines()
        exec("\n".join([line for line in code_vis if not line.strip().startswith("st.set_page_config")]), globals())



elif selected == "Predictions":
    st.markdown("""
        <h2 style='color:#28B463;'>🎯 Predictive Modeling & Customer Segmentation</h2>
    """, unsafe_allow_html=True)
    with open("pages/prediction.py", encoding="utf-8") as f:
        code_pred = f.read().splitlines()
        exec("\n".join([line for line in code_pred if not line.strip().startswith("st.set_page_config")]), globals())


elif selected == "Chatbot":
    st.markdown("""
        <h2 style='color:#E67E22;'>🤖 Chat with Documentation (RAG Bot)</h2>
    """, unsafe_allow_html=True)
    with open("pages/RAG_Chatbot.py", encoding="utf-8") as f:
        code_chat = f.read().splitlines()
        exec("\n".join([line for line in code_chat if not line.strip().startswith("st.set_page_config")]), globals())


