"""
Maternal Health Risk Predictor — Unified Production Version
Combines: Advanced ML Insights + Clean UX Design
"""

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING (clean + modern + dark-mode safe)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        opacity: 0.7;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid;
    }
    .high { border-color: #e74c3c; background: rgba(231,76,60,0.08); }
    .low { border-color: #27ae60; background: rgba(39,174,96,0.08); }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def risk_label(p):
    if p < 0.4:
        return "🟢 Low Risk"
    elif p < 0.7:
        return "🟠 Moderate Risk"
    return "🔴 High Risk"


def warn_inputs(age, bmi, sys, dia):
    warnings = []
    if age < 15:
        warnings.append("Very young maternal age detected.")
    if bmi < 12 or bmi > 55:
        warnings.append("Abnormal BMI range.")
    if sys > 180 or dia > 120:
        warnings.append("Severely high blood pressure detected.")
    return warnings


# ----------------------------------------------------------------------------
# LOAD ARTIFACTS
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/metrics.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


@st.cache_data
def load_dataset():
    return pd.read_csv("data/maternal_health_data.csv")


model, scaler, metadata = load_artifacts()

FEATURES = metadata["feature_columns"]
BEST_MODEL = metadata["best_model_name"]

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", ["Predict Risk", "Model Insights", "About"])

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
**Model:** {BEST_MODEL}  
**Dataset:** {metadata['dataset_size']} patients  
**Class Split:** {metadata['class_distribution']['Low']} Low / {metadata['class_distribution']['High']} High
"""
)

# ----------------------------------------------------------------------------
# PAGE 1 — PREDICT
# ----------------------------------------------------------------------------
if page == "Predict Risk":

    st.markdown('<div class="main-header">Maternal Health Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered pregnancy risk estimation tool</div>', unsafe_allow_html=True)

    with st.form("form"):

        with st.expander("🧬 Vitals", expanded=True):
            c1, c2, c3 = st.columns(3)
            age = c1.slider("Age", 10, 65, 25)
            sys = c2.slider("Systolic BP", 70, 200, 120)
            dia = c3.slider("Diastolic BP", 40, 140, 80)

            hr = c1.slider("Heart Rate", 55, 110, 76)
            bmi = c2.slider("BMI", 12.0, 45.0, 23.0)

        with st.expander("🧪 Lab Values"):
            bs = st.slider("Blood Sugar", 3.0, 19.0, 7.0)
            temp = st.slider("Body Temperature", 97.0, 103.0, 98.0)

        with st.expander("🩺 History"):
            prev = st.toggle("Previous Complications")
            dm = st.toggle("Preexisting Diabetes")
            gdm = st.toggle("Gestational Diabetes")
            mh = st.toggle("Mental Health Concerns")

        submit = st.form_submit_button("🔍 Predict Risk", type="primary")

    if submit:

        # ---------------- warnings ----------------
        for w in warn_inputs(age, bmi, sys, dia):
            st.warning(w)

        X_input = pd.DataFrame([{
            "Age": age,
            "Systolic BP": sys,
            "Diastolic": dia,
            "BS": bs,
            "Body Temp": temp,
            "BMI": bmi,
            "Heart Rate": hr,
            "Previous Complications": int(prev),
            "Preexisting Diabetes": int(dm),
            "Gestational Diabetes": int(gdm),
            "Mental Health": int(mh),
        }])[FEATURES]

        X = scaler.transform(X_input) if metadata["needs_scaling"] else X_input

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        high = proba[1]

        st.markdown("---")

        # ---------------- RESULT ----------------
        col1, col2 = st.columns([1.2, 1])

        with col1:
            label = risk_label(high)
            cls = "high" if pred == 1 else "low"

            st.markdown(
                f"""
                <div class="card {cls}">
                    <h2>{label}</h2>
                    <p>High risk probability: <b>{high:.1%}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(float(high))

            m1, m2, m3 = st.columns(3)
            m1.metric("High Risk", f"{high:.1%}")
            m2.metric("Low Risk", f"{proba[0]:.1%}")
            m3.metric("Confidence", f"{max(proba):.1%}")

        # ---------------- GAUGE ----------------
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=high * 100,
                title={"text": "Risk Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if pred == 1 else "#27ae60"},
                    "steps": [
                        {"range": [0, 40], "color": "#e8f8f0"},
                        {"range": [40, 70], "color": "#fdf2d0"},
                        {"range": [70, 100], "color": "#fceae9"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- FEATURE IMPORTANCE ----------------
        st.markdown("### 🧠 Key Contributing Factors")
        fi = pd.DataFrame(metadata["feature_importance"]).head(6)

        st.plotly_chart(
            px.bar(fi, x="Importance", y="Feature",
                   orientation="h",
                   color="Importance",
                   color_continuous_scale="Blues"),
            use_container_width=True
        )

# ----------------------------------------------------------------------------
# PAGE 2 — MODEL INSIGHTS
# ----------------------------------------------------------------------------
elif page == "Model Insights":

    st.markdown("## 📊 Model Performance Dashboard")

    metrics = pd.DataFrame(metadata["all_metrics"])

    st.dataframe(metrics, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            px.bar(metrics, x="Model", y="Accuracy", text_auto=".3f"),
            use_container_width=True
        )

    with c2:
        melted = metrics.melt(
            id_vars="Model",
            value_vars=["Precision", "Recall", "F1 Score"],
            var_name="Metric",
            value_name="Score"
        )
        st.plotly_chart(
            px.bar(melted, x="Model", y="Score", color="Metric", barmode="group"),
            use_container_width=True
        )

    # ---------------- CONFUSION MATRICES ----------------
    st.markdown("### 🔍 Confusion Matrices")

    cols = st.columns(len(metadata["confusion_matrices"]))

    for col, (name, cm) in zip(cols, metadata["confusion_matrices"].items()):
        with col:
            st.plotly_chart(
                px.imshow(
                    np.array(cm),
                    text_auto=True,
                    x=["Low", "High"],
                    y=["Low", "High"],
                    color_continuous_scale="Blues"
                ),
                use_container_width=True
            )

    # ---------------- FEATURE IMPORTANCE ----------------
    st.markdown("### 🧠 Feature Importance (Best Model)")
    st.plotly_chart(
        px.bar(pd.DataFrame(metadata["feature_importance"]),
               x="Importance", y="Feature",
               orientation="h",
               color="Importance",
               color_continuous_scale="Blues"),
        use_container_width=True
    )

    # ---------------- DATASET ----------------
    df = load_dataset()

    st.markdown("### 📂 Dataset Overview")

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(px.histogram(df, x="Risk Level"), use_container_width=True)

    with c2:
        st.plotly_chart(px.box(df, x="Risk Level", y="Age"), use_container_width=True)

    with st.expander("📊 Correlation Heatmap"):
        corr = df.copy()
        corr["Risk Level"] = corr["Risk Level"].map({"Low": 0, "High": 1})
        st.plotly_chart(
            px.imshow(corr.corr(numeric_only=True),
                      color_continuous_scale="RdBu_r",
                      text_auto=".2f"),
            use_container_width=True
        )

# ----------------------------------------------------------------------------
# PAGE 3 — ABOUT
# ----------------------------------------------------------------------------
else:

    st.markdown("## ℹ️ About This Project")

    st.write(
        """
        This application predicts maternal health risk using machine learning.

        **Features:**
        - Multiple ML models compared
        - Real-time risk prediction
        - Model interpretability (feature importance)
        - Interactive dashboards

        **Tech Stack:** Python, Scikit-learn, Streamlit, Plotly

        ⚠️ Educational use only — not for clinical decision-making.
        """
    )