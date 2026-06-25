"""
app.py
------
Maternal Health Risk Predictor — Streamlit app.

Two views:
  1. Predict  -> enter patient vitals, get a Low/High risk prediction with confidence
  2. Model Insights -> compare the 5 trained models, view confusion matrices,
     feature importance, and dataset summary.
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
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #5a6b7d;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #fff1f0 0%, #ffe1de 100%);
        border-left: 6px solid #e74c3c;
        border-radius: 10px;
        padding: 1.6rem;
        margin-top: 1rem;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #f0fff4 0%, #e1f8e7 100%);
        border-left: 6px solid #27ae60;
        border-radius: 10px;
        padding: 1.6rem;
        margin-top: 1rem;
    }
    .risk-title-high { color: #c0392b; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .risk-title-low { color: #1e8449; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .metric-pill {
        background: #f0f3f7;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Cached resource loading
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

# Clinically sensible slider bounds (overrides raw data min where data has artifacts,
# e.g. BMI of 0 in the source data is not physiologically possible)
SLIDER_BOUNDS = {
    "Age": (10, 65, 25),
    "Systolic BP": (70, 200, 120),
    "Diastolic": (40, 140, 80),
    "BS": (3.0, 19.0, 7.0),
    "Body Temp": (97.0, 103.0, 98.0),
    "BMI": (12.0, 45.0, 23.0),
    "Heart Rate": (55, 95, 76),
}

FEATURE_HELP = {
    "Age": "Patient's age in years",
    "Systolic BP": "Systolic blood pressure (mmHg)",
    "Diastolic": "Diastolic blood pressure (mmHg)",
    "BS": "Blood sugar level (mmol/L)",
    "Body Temp": "Body temperature (°F)",
    "BMI": "Body Mass Index",
    "Heart Rate": "Resting heart rate (bpm)",
    "Previous Complications": "History of complications in a previous pregnancy",
    "Preexisting Diabetes": "Diagnosed with diabetes before this pregnancy",
    "Gestational Diabetes": "Diabetes diagnosed during this pregnancy",
    "Mental Health": "Current mental health concerns (e.g. stress, anxiety, depression)",
}


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", ["Predict Risk", "Model Insights", "About"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Active model:** {BEST_MODEL_NAME}
    **Dataset size:** {metadata['dataset_size']} patients
    **Class split:** {metadata['class_distribution']['Low']} Low / {metadata['class_distribution']['High']} High
    """
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ This tool is for educational/portfolio purposes only and is **not** a substitute "
    "for professional medical advice or diagnosis."
)

# ----------------------------------------------------------------------------
# PAGE 1: Predict Risk
# ----------------------------------------------------------------------------
if page == "Predict Risk":
    st.markdown('<p class="main-header">Maternal Health Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Enter patient vitals below to estimate pregnancy risk level '
        f'using a trained {BEST_MODEL_NAME} model.</p>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Vitals**")
            age = st.slider("Age (years)", *SLIDER_BOUNDS["Age"], help=FEATURE_HELP["Age"])
            systolic = st.slider("Systolic BP (mmHg)", *SLIDER_BOUNDS["Systolic BP"], help=FEATURE_HELP["Systolic BP"])
            diastolic = st.slider("Diastolic BP (mmHg)", *SLIDER_BOUNDS["Diastolic"], help=FEATURE_HELP["Diastolic"])

        with col2:
            st.markdown("**Lab values**")
            bs = st.slider("Blood Sugar (mmol/L)", *SLIDER_BOUNDS["BS"], step=0.1, help=FEATURE_HELP["BS"])
            body_temp = st.slider("Body Temp (°F)", *SLIDER_BOUNDS["Body Temp"], step=0.1, help=FEATURE_HELP["Body Temp"])
            bmi = st.slider("BMI", *SLIDER_BOUNDS["BMI"], step=0.1, help=FEATURE_HELP["BMI"])
            heart_rate = st.slider("Heart Rate (bpm)", *SLIDER_BOUNDS["Heart Rate"], help=FEATURE_HELP["Heart Rate"])

        with col3:
            st.markdown("**Medical history**")
            prev_complications = st.toggle("Previous Complications", help=FEATURE_HELP["Previous Complications"])
            preexisting_diabetes = st.toggle("Preexisting Diabetes", help=FEATURE_HELP["Preexisting Diabetes"])
            gestational_diabetes = st.toggle("Gestational Diabetes", help=FEATURE_HELP["Gestational Diabetes"])
            mental_health = st.toggle("Mental Health Concerns", help=FEATURE_HELP["Mental Health"])

        submitted = st.form_submit_button("🔍 Predict Risk Level", use_container_width=True, type="primary")

    if submitted:
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

        if NEEDS_SCALING:
            input_for_model = scaler.transform(input_df)
        else:
            input_for_model = input_df

        pred = model.predict(input_for_model)[0]
        proba = model.predict_proba(input_for_model)[0]
        high_risk_prob = proba[1]
        low_risk_prob = proba[0]

        st.markdown("---")
        result_col, gauge_col = st.columns([1.3, 1])

        with result_col:
            if pred == 1:
                st.markdown(
                    f"""
                    <div class="risk-card-high">
                        <p class="risk-title-high">⚠️ High Risk</p>
                        <p style="margin-top:0.5rem; color:#7b241c;">
                        The model estimates a <b>{high_risk_prob:.1%}</b> probability of high risk
                        based on the provided vitals. Clinical follow-up is recommended.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-card-low">
                        <p class="risk-title-low">✅ Low Risk</p>
                        <p style="margin-top:0.5rem; color:#196f3d;">
                        The model estimates a <b>{low_risk_prob:.1%}</b> probability of low risk
                        based on the provided vitals.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("####  ")
            m1, m2, m3 = st.columns(3)
            m1.metric("High Risk Probability", f"{high_risk_prob:.1%}")
            m2.metric("Low Risk Probability", f"{low_risk_prob:.1%}")
            m3.metric("Model Confidence", f"{max(proba):.1%}")

        with gauge_col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=high_risk_prob * 100,
                    title={"text": "High Risk Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#c0392b" if pred == 1 else "#27ae60"},
                        "steps": [
                            {"range": [0, 40], "color": "#e8f8f0"},
                            {"range": [40, 70], "color": "#fdf2d0"},
                            {"range": [70, 100], "color": "#fceae9"},
                        ],
                        "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Why this prediction? (Top contributing factors)")
        importance_df = pd.DataFrame(metadata["feature_importance"]).head(5)
        fig2 = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig2.update_layout(height=280, showlegend=False, yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "These are the model's overall most influential features (from training), "
            "not a per-patient explanation."
        )

# ----------------------------------------------------------------------------
# PAGE 2: Model Insights
# ----------------------------------------------------------------------------
elif page == "Model Insights":
    st.markdown('<p class="main-header">Model Insights & Comparison</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Performance comparison across 5 trained classifiers '
        "(Logistic Regression, Decision Tree, Random Forest, KNN, SVM).</p>",
        unsafe_allow_html=True,
    )

    metrics_df = pd.DataFrame(metadata["all_metrics"])

    st.markdown("#### Performance metrics")
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], color="#d4f4dd"
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = px.bar(
            metrics_df,
            x="Model",
            y="Accuracy",
            color="Model",
            title="Accuracy by Model",
            text_auto=".3f",
        )
        fig.update_layout(showlegend=False, yaxis_range=[metrics_df["Accuracy"].min() - 0.05, 1.0])
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        melted = metrics_df.melt(
            id_vars="Model", value_vars=["Precision", "Recall", "F1 Score"], var_name="Metric", value_name="Score"
        )
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", title="Precision / Recall / F1")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Confusion matrices")
    cm_cols = st.columns(len(metadata["confusion_matrices"]))
    for col, (name, cm) in zip(cm_cols, metadata["confusion_matrices"].items()):
        with col:
            cm_arr = np.array(cm)
            fig = px.imshow(
                cm_arr,
                text_auto=True,
                x=["Low", "High"],
                y=["Low", "High"],
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
            )
            fig.update_layout(title=name, height=260, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature importance (Random Forest)")
    importance_df = pd.DataFrame(metadata["feature_importance"])
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Dataset overview")
    df = load_dataset()
    d1, d2 = st.columns(2)
    with d1:
        fig = px.histogram(df, x="Risk Level", color="Risk Level", title="Risk Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        fig = px.box(df, x="Risk Level", y="Age", color="Risk Level", title="Age Distribution by Risk Level")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View correlation heatmap"):
        numeric_df = df.copy()
        numeric_df["Risk Level"] = numeric_df["Risk Level"].map({"Low": 0, "High": 1})
        corr = numeric_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 3: About
# ----------------------------------------------------------------------------
else:
    st.markdown('<p class="main-header">About this project</p>', unsafe_allow_html=True)
    st.markdown(
        """
        This app predicts **maternal health risk level** (Low / High) from a set of
        vitals and medical history features, using machine learning models trained on
        a maternal health dataset.

        **Pipeline:**
        1. Data cleaning (missing values, duplicates)
        2. Exploratory data analysis (distributions, correlations)
        3. Trained and compared 5 classifiers: Logistic Regression, Decision Tree,
           Random Forest, KNN (tuned via grid search), and SVM (tuned via grid search)
        4. Selected the best model by F1 score for deployment

        **Tech stack:** Python, scikit-learn, pandas, Streamlit, Plotly

        ---
        *Built as part of a data science project. For educational and portfolio purposes only —
        not intended for real clinical use.*
        """
    )

    st.markdown("#### Feature reference")
    ref_df = pd.DataFrame(
        [{"Feature": k, "Description": v} for k, v in FEATURE_HELP.items()]
    )
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
