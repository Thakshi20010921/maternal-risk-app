"""
app.py
------
Maternal Health Risk Predictor — Streamlit app (Improved UX/UI version)
"""

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Dark-mode safe styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0;
    }

    .sub-header {
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 1.2rem;
    }

    .risk-card-high {
        background: rgba(231, 76, 60, 0.10);
        border-left: 6px solid #e74c3c;
        border-radius: 10px;
        padding: 1.5rem;
    }

    .risk-card-low {
        background: rgba(39, 174, 96, 0.10);
        border-left: 6px solid #27ae60;
        border-radius: 10px;
        padding: 1.5rem;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Risk badge helper
# ----------------------------------------------------------------------------
def risk_badge(prob):
    if prob < 0.4:
        return "🟢 Low Risk"
    elif prob < 0.7:
        return "🟠 Moderate Risk"
    else:
        return "🔴 High Risk"


# ----------------------------------------------------------------------------
# Cached loading
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
NEEDS_SCALING = metadata["needs_scaling"]
BEST_MODEL_NAME = metadata["best_model_name"]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", ["Predict Risk", "Model Insights", "About"])

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
**Model:** {BEST_MODEL_NAME}  
**Dataset size:** {metadata['dataset_size']}  
**Class split:** {metadata['class_distribution']['Low']} Low / {metadata['class_distribution']['High']} High
"""
)

st.sidebar.caption(
    "⚠️ Educational tool only — not medical advice."
)

# ----------------------------------------------------------------------------
# PAGE 1: Predict
# ----------------------------------------------------------------------------
if page == "Predict Risk":

    st.markdown("## 🩺 Maternal Health Risk Predictor")
    st.markdown(
        f"Enter patient vitals to estimate risk using **{BEST_MODEL_NAME}**"
    )

    with st.form("prediction_form"):

        # ---------------- VITALS ----------------
        with st.expander("🧬 Vitals", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", 10, 65, 25)
                systolic = st.slider("Systolic BP", 70, 200, 120)
            with col2:
                diastolic = st.slider("Diastolic BP", 40, 140, 80)
                heart_rate = st.slider("Heart Rate", 55, 110, 76)
            with col3:
                bmi = st.slider("BMI", 12.0, 45.0, 23.0)

        # ---------------- LABS ----------------
        with st.expander("🧪 Lab Values", expanded=True):
            bs = st.slider("Blood Sugar (mmol/L)", 3.0, 19.0, 7.0, step=0.1)
            body_temp = st.slider("Body Temperature (°F)", 97.0, 103.0, 98.0, step=0.1)

        # ---------------- HISTORY ----------------
        with st.expander("🩺 Medical History", expanded=True):
            prev_complications = st.toggle("Previous Complications")
            preexisting_diabetes = st.toggle("Preexisting Diabetes")
            gestational_diabetes = st.toggle("Gestational Diabetes")
            mental_health = st.toggle("Mental Health Concerns")

        submitted = st.form_submit_button("🔍 Predict Risk", type="primary")

    # ----------------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------------
    if submitted:

        warnings = []

        if bmi < 12 or bmi > 60:
            warnings.append("BMI is unusually low/high.")
        if systolic > 180 or diastolic > 120:
            warnings.append("Blood pressure is in a critical range.")
        if age < 15:
            warnings.append("Very young maternal age detected.")

        for w in warnings:
            st.warning(w)

        input_dict = {
            "Age": age,
            "Systolic BP": systolic,
            "Diastolic": diastolic,
            "BS": bs,
            "Body Temp": body_temp,
            "BMI": bmi,
            "Previous Complications": int(prev_complications),
            "Preexisting Diabetes": int(preexisting_diabetes),
            "Gestational Diabetes": int(gestational_diabetes),
            "Mental Health": int(mental_health),
            "Heart Rate": heart_rate,
        }

        input_df = pd.DataFrame([input_dict])[FEATURES]

        X = scaler.transform(input_df) if NEEDS_SCALING else input_df

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        high = proba[1]
        low = proba[0]

        st.markdown("---")

        colA, colB = st.columns([1.2, 1])

        # ---------------- RESULT ----------------
        with colA:

            badge = risk_badge(high)

            if pred == 1:
                st.markdown(
                    f"""
                    <div class="risk-card-high">
                        <h3>{badge}</h3>
                        <p>High risk probability: <b>{high:.1%}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-card-low">
                        <h3>{badge}</h3>
                        <p>Low risk probability: <b>{low:.1%}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.progress(float(high))
            st.caption("High-risk probability")

            m1, m2, m3 = st.columns(3, gap="large")
            m1.metric("High Risk", f"{high:.1%}")
            m2.metric("Low Risk", f"{low:.1%}")
            m3.metric("Confidence", f"{max(proba):.1%}")

        # ---------------- GAUGE ----------------
        with colB:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=high * 100,
                    title={"text": "Risk %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#e74c3c" if pred == 1 else "#27ae60"},
                        "steps": [
                            {"range": [0, 40], "color": "#e8f8f0"},
                            {"range": [40, 70], "color": "#fdf2d0"},
                            {"range": [70, 100], "color": "#fceae9"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- FEATURE IMPORTANCE ----------------
        st.markdown("### 🧠 Key Contributing Factors")
        importance_df = pd.DataFrame(metadata["feature_importance"]).head(5)

        fig2 = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig2.update_layout(height=300, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE 2: MODEL INSIGHTS
# ----------------------------------------------------------------------------
elif page == "Model Insights":

    st.markdown("## 📊 Model Insights")

    metrics_df = pd.DataFrame(metadata["all_metrics"])

    st.dataframe(metrics_df, use_container_width=True)

    fig = px.bar(metrics_df, x="Model", y="Accuracy", text_auto=".3f")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Feature Importance")
    importance_df = pd.DataFrame(metadata["feature_importance"])

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📂 Dataset Overview")

    df = load_dataset()

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(px.histogram(df, x="Risk Level"), use_container_width=True)

    with c2:
        st.plotly_chart(px.box(df, x="Risk Level", y="Age"), use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE 3: ABOUT
# ----------------------------------------------------------------------------
else:

    st.markdown("## ℹ️ About")

    st.write(
        """
        Maternal Health Risk Predictor using machine learning.

        Built with:
        - Python
        - Scikit-learn
        - Streamlit
        - Plotly

        Educational project only — not for clinical use.
        """
    )