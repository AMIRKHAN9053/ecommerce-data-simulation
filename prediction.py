import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import plotly.express as px

# --- Streamlit Setup ---
st.set_page_config(layout="wide")
st.title("🧠 Purchase Prediction Dashboard")

# --- Load Data ---
df = pd.read_csv("D:\destop file\get info\python\imarticus project\Imarticus Data Science Internship - Assessment_by_Amir_Khan\ecommerce_data (Generated data).csv")
df.fillna(0, inplace=True)

# --- Define Safe Features (Excludes added_to_cart, cart_duration_sec) ---
safe_features = [
    "location", "device_type", "product_category", "customer_segment",
    "price", "discount", "views", "rating",
    "session_duration", "user_activity_score", "user_interaction_score"
]

# --- Encode Categorical Columns ---
categorical_cols = ["location", "device_type", "product_category", "customer_segment"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Train/Test Split ---
X = df[safe_features]
y = df["purchase_made"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# --- Train Model ---
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
cv_mean = np.mean(cv_scores)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Generate classification report as dict
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame for display
report_df = pd.DataFrame(report_dict).transpose()

# Round for better readability
report_df = report_df.round(2)

# Display metrics as header row
st.subheader("📊 Model Performance")

# Accuracy & CV as separate table
summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "Cross-Validation (5-fold)"],
    "Score": [f"{accuracy_score(y_test, y_pred)*100:.2f}%", f"{cv_mean*100:.2f}%"]
})
st.table(summary_df)

# Full classification report as table
st.subheader("📋 Classification Report")
st.dataframe(report_df, use_container_width=True)


# --- Distribution Checks ---
st.write("🎯 Actual Class Distribution:")
st.write(y.value_counts(normalize=True).rename({0: "No Purchase", 1: "Purchase"}))

st.write("🔍 Prediction Distribution:")
st.write(pd.Series(y_pred).value_counts(normalize=True).rename({0: "No Purchase", 1: "Purchase"}))

# --- Feature Importance ---
importances = model.feature_importances_
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

order_option = st.selectbox("📊 Order By", ["Descending", "Ascending"], index=0)

# Apply sorting based on selection
ascending = True if order_option == "Ascending" else False
feature_df = feature_df.sort_values(by="Importance", ascending=ascending)

st.subheader("📊 Feature Importances")
fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (XGBoost)')
st.plotly_chart(fig, use_container_width=True)



# --- Anomaly Detection ---
st.subheader("🚨 Anomaly Detection")
anomaly_features = ["views", "rating", "session_duration", "user_activity_score", "user_interaction_score"]
iso = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_score"] = iso.fit_predict(df[anomaly_features])
anomalies = df[df["anomaly_score"] == -1]

st.write(f"⚠️ Detected {len(anomalies)} anomalies in the data.")
st.dataframe(anomalies[["timestamp", "user_id", "product_id", "rating", "purchase_made"]])

# --- Visual Insights ---
st.subheader("📊 Visual Insights Based on Model")
insight_1 = df.groupby("rating")["purchase_made"].mean().reset_index()
insight_2 = df.groupby("discount")["purchase_made"].mean().reset_index()
insight_3 = df.groupby("product_category")["price"].mean().reset_index()

st.subheader("🎯 Purchase Likelihood by Rating")
insight_1 = insight_1[insight_1["rating"] > 0]
fig1 = px.bar(insight_1, x="rating", y="purchase_made", title="Purchase Likelihood by Rating")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🎯 Purchase Likelihood by Discount")
fig2 = px.line(insight_2, x="discount", y="purchase_made", title="Purchase Likelihood by Discount")
st.plotly_chart(fig2, use_container_width=True)


# --- Live Prediction Form ---
st.subheader("🎯 Try a Live Prediction")

with st.form("predict_form"):
    form_data = {}

    for col in categorical_cols:
        options = list(label_encoders[col].classes_)
        selected = st.selectbox(f"{col.replace('_', ' ').title()}", options)
        form_data[col] = label_encoders[col].transform([selected])[0]

    form_data["price"] = st.slider("Price", 100, 50000, 3000)
    form_data["discount"] = st.slider("Discount", 0.0, 0.5, 0.2)
    form_data["views"] = st.slider("Views", 1, 30, 5)
    form_data["rating"] = st.slider("Rating", 1, 5, 3)
    form_data["session_duration"] = st.slider("Session Duration (sec)", 10, 3600, 600)
    form_data["user_activity_score"] = st.slider("User Activity Score", 0.0, 50.0, 12.5)
    form_data["user_interaction_score"] = st.slider("Interaction Score", 0.0, 30.0, 10.0)

    if st.form_submit_button("Predict"):
        user_input = pd.DataFrame([form_data])
        prediction = model.predict(user_input)[0]
        prob = model.predict_proba(user_input)[0][1]

        st.markdown("#### 📢 Prediction Result")
        if prediction:
            st.success(f"🟢 Purchase Likely (Confidence: {prob:.2%})")
        else:
            st.error(f"🔴 Purchase Not Likely (Confidence: {1 - prob:.2%})")

        st.markdown("#### 🧾 Input Summary")
        st.dataframe(user_input.T.rename(columns={0: "Value"}), use_container_width=True)

# --- Footer ---
st.markdown("---")
st.caption("🚀 Built from clean behavioral data | No leakage | Balanced & explainable predictions")
