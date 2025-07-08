# Smart Retail ML Dashboard

## Project Overview
This project presents a full-featured **Streamlit** dashboard built during the Imarticus Data Science Internship. It simulates e-commerce behavior in real time, visualizes customer/product trends, predicts purchase likelihood using machine learning, and features a PDF-based chatbot using Retrieval-Augmented Generation (RAG).

---

## Dataset
- **Source**: Simulated using Python’s Faker library.
- **File Used**: `ecommerce_data.csv`
- **Records**: Continuously appended live during simulation.
- **Columns**:
  1. `user_id`: Simulated customer identifier.
  2. `product_id`: Simulated product identifier.
  3. `category`: Product category.
  4. `location`: Customer location.
  5. `action`: Behavior type (view, cart, purchase).
  6. `timestamp`: Interaction timestamp.
  7. Additional engineered features used for model training.

---

## Key Tasks and Steps

### 1. Importing Libraries
Key Python libraries used include:
- `streamlit`
- `pandas`, `numpy`
- `xgboost`, `scikit-learn`
- `plotly`
- `faiss`, `PyPDF2`, `langchain`, `openai`

### 2. Real-Time Data Simulation
- Used Faker to simulate e-commerce user behavior.
- Simulated actions (view, cart, purchase) saved in `ecommerce_data.csv`.
- KPIs like total users, conversions, and bounce rate displayed live.

### 3. Visualizations and Insights

#### a. Revenue and Behavior Trends
- Line plots and bar charts show revenue and user behavior patterns over time.
- Filtered by date, category, or region.
- **Insight**: Conversion peaks seen during specific hours and product types.

#### b. Cart Abandonment and Returns
- Analyzed actions to identify patterns in cart abandonment and returns.
- **Result**: Return rates are higher in specific regions and categories.

#### c. Top Categories and Locations
- Showed top-performing product categories and cities by purchase volume.
- **Conclusion**: Electronics and fashion lead across major metros.

### 4. Machine Learning Predictions

#### a. Model Training and Evaluation
- Used XGBoost to predict purchase likelihood from user behavior.
- Included encoding, scaling, stratified train-test split.
- Evaluated with accuracy, precision, recall, and F1-score.

#### b. Live Prediction
- Inputs simulated or uploaded, model predicts likelihood of purchase.
- **Result**: Accuracy ~87%, live predictions shown in dashboard.

### 5. PDF-Based Chatbot with RAG

#### a. PDF Upload and Indexing
- PDFs uploaded by user are processed and embedded using FAISS.

#### b. Q&A Interface
- Uses OpenAI or Together AI for intelligent responses to PDF questions.
- Chat export supported.

---

## Results and Conclusions
1. **User Insight**: Conversion and abandonment rates vary by category and hour.
2. **ML Performance**: XGBoost offers strong predictive power on behavioral data.
3. **Visualization**: Clear, filterable views for business metrics.
4. **Chatbot**: Practical PDF Q&A with vector search and LLMs.

---

## How to Use This Project
1. Clone the repository and ensure all required files (e.g., `ecommerce_data.csv`) are present.
2. Install dependencies with:
   ```bash
   pip install -r requirements.txt
