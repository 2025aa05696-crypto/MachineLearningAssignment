import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost import train_xgboost


# ===============================
# Helper: Load uploaded test data
# ===============================
def load_uploaded_test_data(uploaded_file):
    data = pd.read_csv(uploaded_file)

    if "url" in data.columns:
        data = data.drop(columns=["url"])

    X_test = data.drop(columns=["status"])
    y_test = data["status"].map({"legitimate": 0, "phishing": 1})

    return X_test, y_test


# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Phishing Website Detection – Prem Garg",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# Custom CSS Styling
# ===============================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #888;
        margin-top: -0.5rem;
    }
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.8rem;
        font-weight: 700;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #38bdf8;
    }
    /* Author card */
    .author-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px;
        margin-top: 10px;
        text-align: center;
    }
    .author-card h4 {
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .author-card p {
        color: #cbd5e1;
        margin: 2px 0;
        font-size: 0.92rem;
    }
    /* Table styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        padding: 20px 0 10px 0;
        border-top: 1px solid #334155;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Sidebar – Controls & Author Info
# ===============================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Dataset Upload
    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader(
        "Upload Test Dataset (CSV)",
        type=["csv"],
        help="Upload a CSV file with the same feature columns as the training data.",
    )

    # Download Dataset Button
    dataset_path = "dataset_phishing.csv"
    try:
        dataset_df = pd.read_csv(dataset_path)
        st.download_button(
            label="⬇️ Download Dataset CSV",
            data=dataset_df.to_csv(index=False).encode("utf-8"),
            file_name="dataset_phishing.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except FileNotFoundError:
        st.warning("Dataset file not found.")

    st.markdown("")

    # Model Selection
    st.markdown("### 🤖 Model")
    model_name = st.selectbox(
        "Select ML Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost",
        ],
    )

    MODEL_DESCRIPTIONS = {
        "Logistic Regression": "Linear model using sigmoid function for binary classification.",
        "Decision Tree": "Tree-based model that splits data on feature thresholds.",
        "KNN": "Instance-based learner using nearest-neighbour voting.",
        "Naive Bayes": "Probabilistic classifier based on Bayes' theorem.",
        "Random Forest": "Ensemble of decision trees with bagging.",
        "XGBoost": "Gradient-boosted ensemble with regularisation.",
    }
    st.info(MODEL_DESCRIPTIONS[model_name])

    st.markdown("")
    run_clicked = st.button("🚀 Run Model", use_container_width=True)

    # Author Info in Sidebar
    st.markdown("---")
    st.markdown("### 👤 Author")
    st.markdown("""
    **Prem Garg**  
    BITS ID: `2025aa05696`  
    📧 2025aa05696@wilp.bits-pilani.ac.in  
    🎓 M.Tech – BITS Pilani (WILP)
    """)

# ===============================
# Main Content Area – Header
# ===============================
st.markdown("""
<div class="main-header">
    <h1>🛡️ Phishing Website Detection</h1>
    <p>Machine Learning Assignment 2 — Model Evaluation Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ===============================
# Run Model
# ===============================
if run_clicked:

    if uploaded_file is not None:
        X_ext, y_ext = load_uploaded_test_data(uploaded_file)
    else:
        X_ext, y_ext = None, None

    with st.spinner(f"Training **{model_name}** … please wait"):
        # ===============================
        # Model Execution
        # ===============================
        if model_name == "Logistic Regression":
            metrics, y_test, y_pred = train_logistic_regression(X_ext, y_ext)
        elif model_name == "Decision Tree":
            metrics, y_test, y_pred = train_decision_tree(X_ext, y_ext)
        elif model_name == "KNN":
            metrics, y_test, y_pred = train_knn(X_ext, y_ext)
        elif model_name == "Naive Bayes":
            metrics, y_test, y_pred = train_naive_bayes(X_ext, y_ext)
        elif model_name == "Random Forest":
            metrics, y_test, y_pred = train_random_forest(X_ext, y_ext)
        elif model_name == "XGBoost":
            metrics, y_test, y_pred = train_xgboost(X_ext, y_ext)

    st.success(f"✅ {model_name} evaluation complete!")
    st.markdown("")

    # ===============================
    # KPI Metrics
    # ===============================
    st.markdown("### 📊 Key Evaluation Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    col2.metric("AUC", f"{metrics['AUC']:.3f}")
    col3.metric("Precision", f"{metrics['Precision']:.3f}")
    col4.metric("Recall", f"{metrics['Recall']:.3f}")
    col5.metric("F1 Score", f"{metrics['F1']:.3f}")
    col6.metric("MCC", f"{metrics['MCC']:.3f}")

    st.markdown("")

    # ===============================
    # Results – Two Columns
    # ===============================
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### 🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Legitimate", "Actual Phishing"],
            columns=["Pred. Legitimate", "Pred. Phishing"],
        )
        st.dataframe(cm_df, use_container_width=True)

    with right_col:
        st.markdown("### 📄 Classification Report")
        report_df = pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                target_names=["Legitimate", "Phishing"],
                output_dict=True,
            )
        ).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("")

    # ===============================
    # Metric Bar Chart
    # ===============================
    st.markdown("### 📈 Metrics Overview")
    chart_data = pd.DataFrame(
        {"Metric": list(metrics.keys()), "Score": list(metrics.values())}
    ).set_index("Metric")
    st.bar_chart(chart_data, height=350)

else:
    # ===============================
    # Landing state – instructions
    # ===============================
    st.markdown("")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown("#### 1️⃣ Upload")
        st.write("Upload your test CSV dataset using the sidebar (optional).")
    with info_col2:
        st.markdown("#### 2️⃣ Select")
        st.write("Choose a machine learning model from the sidebar dropdown.")
    with info_col3:
        st.markdown("#### 3️⃣ Evaluate")
        st.write("Click **Run Model** to view evaluation metrics & reports.")

# ===============================
# Footer with Author Details
# ===============================
st.markdown("""
<div class="footer">
    Developed by <strong>Prem Garg</strong> (2025aa05696) · M.Tech, BITS Pilani (WILP) ·
    <a href="mailto:2025aa05696@wilp.bits-pilani.ac.in" style="color:#38bdf8;">2025aa05696@wilp.bits-pilani.ac.in</a>
</div>
""", unsafe_allow_html=True)
