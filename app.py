"""
Maternal Health Risk Predictor — Unified Production Version
Combines: ML Insights + Light/Dark Theme Toggle + SHAP Explanations + PDF Report Export
"""

import io
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# PAGE CONFIG

st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# THEME STATE

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg": "#0e1117",
        "card_bg": "#1c212c",
        "text": "#f5f5f5",
        "text_muted": "#b8c0cc",
        "border": "#343b48",
        "plot_template": "plotly_dark",
        "plot_bg": "#161a23",
        "high_bg": "rgba(231,76,60,0.18)",
        "low_bg": "rgba(39,174,96,0.18)",
        "high_border": "#e74c3c",
        "low_border": "#2ecc71",
        "accent": "#5b8def",
    },
    "light": {
        "bg": "#ffffff",
        "card_bg": "#f7f8fa",
        "text": "#1a1a1a",
        "text_muted": "#5a6b7d",
        "border": "#e1e4e8",
        "plot_template": "plotly_white",
        "plot_bg": "#ffffff",
        "high_bg": "rgba(231,76,60,0.08)",
        "low_bg": "rgba(39,174,96,0.08)",
        "high_border": "#e74c3c",
        "low_border": "#27ae60",
        "accent": "#1e3a5f",
    },
}
T = THEMES[st.session_state.theme]
PLOT_TEMPLATE = T["plot_template"]


# GLOBAL PLOTLY LAYOUT DEFAULTS — keeps charts on-theme (no white boxes)

def themed(fig, height=None):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=T["plot_bg"],
        plot_bgcolor=T["plot_bg"],
        font=dict(color=T["text"], size=13),
        legend=dict(font=dict(color=T["text"])),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(color=T["text"], gridcolor=T["border"], title_font=dict(color=T["text"]))
    fig.update_yaxes(color=T["text"], gridcolor=T["border"], title_font=dict(color=T["text"]))
    return fig



# STYLING

st.markdown(
    f"""
    <style>
    html, body, .stApp {{
        background-color: {T['bg']} !important;
        color: {T['text']} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {T['card_bg']} !important;
    }}

    /* Headings */
    .main-header {{
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: {T['text']} !important;
    }}
    .sub-header {{
        color: {T['text_muted']} !important;
        margin-bottom: 1rem;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {T['text']} !important;
    }}

    /* General text everywhere (markdown, captions, labels) */
    p, span, label, li, div[data-testid="stMarkdownContainer"] {{
        color: {T['text']} !important;
    }}
    [data-testid="stCaptionContainer"] {{
        color: {T['text_muted']} !important;
    }}

    /* Cards */
    .card {{
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid;
        background-color: {T['card_bg']};
        color: {T['text']} !important;
    }}
    .card h2, .card p {{ color: {T['text']} !important; }}
    .high {{ border-color: {T['high_border']}; background: {T['high_bg']}; }}
    .low {{ border-color: {T['low_border']}; background: {T['low_bg']}; }}

    .info-box {{
        background-color: {T['card_bg']};
        border: 1px solid {T['border']};
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: {T['text']} !important;
    }}
    .info-box b {{ color: {T['text']} !important; }}

    /* Sliders: track labels, min/max, value bubble */
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] p {{
        color: {T['text']} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"],
    div[data-testid="stTickBarMin"],
    div[data-testid="stTickBarMax"] {{
        color: {T['text']} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stThumbValue"] {{
        color: #ffffff !important;
        font-weight: 700 !important;
        background-color: {T['accent']} !important;
    }}

    /* Expander headers + body */
    details {{
        background-color: {T['card_bg']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 8px;
    }}
    details summary {{
        color: {T['text']} !important;
    }}
    details summary span, details summary p, details summary div {{
        color: {T['text']} !important;
        font-weight: 600 !important;
    }}

    /* Toggles */
    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] p,
    div[data-testid="stToggle"] span {{
        color: {T['text']} !important;
    }}

    /* Radio buttons (sidebar nav) */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p,
    div[data-testid="stRadio"] span {{
        color: {T['text']} !important;
    }}

    /* Buttons */
    button[kind="primary"], button[kind="secondary"] {{
        color: #ffffff !important;
    }}
    div[data-testid="stDownloadButton"] button {{
        color: {T['text']} !important;
        background-color: {T['card_bg']} !important;
        border: 1px solid {T['border']} !important;
    }}
    div[data-testid="stDownloadButton"] button p {{
        color: {T['text']} !important;
    }}
    div[data-testid="stDownloadButton"] button:hover {{
        border-color: {T['accent']} !important;
        color: {T['accent']} !important;
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background-color: {T['card_bg']};
        border: 1px solid {T['border']};
        border-radius: 10px;
        padding: 0.6rem;
    }}
    div[data-testid="stMetricLabel"] p {{ color: {T['text_muted']} !important; }}
    div[data-testid="stMetricValue"] {{ color: {T['text']} !important; }}

    /* Dataframes */
    div[data-testid="stDataFrame"] {{
        background-color: {T['card_bg']};
    }}

    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)



# HELPERS

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



# PDF GENERATION ENGINE

def get_personalized_recommendations(raw_values, prev, dm, gdm, mh):
    """Generate specific recommendations based on which values are abnormal."""
    recs = []

    if raw_values["sys"] > 140 or raw_values["dia"] > 90:
        recs.append("HIGH BLOOD PRESSURE detected — reduce salt intake, rest adequately, and consult a doctor immediately.")
    if raw_values["sys"] > 160 or raw_values["dia"] > 110:
        recs.append("CRITICALLY HIGH BP — seek emergency medical care without delay.")
    if raw_values["bs"] > 7.8:
        recs.append("ELEVATED BLOOD SUGAR — monitor glucose levels daily, follow a low-GI diet, and consult an endocrinologist.")
    if raw_values["bmi"] > 30:
        recs.append("HIGH BMI — adopt a balanced diet and light, pregnancy-safe exercise. Consult a dietitian.")
    if raw_values["bmi"] < 18.5:
        recs.append("LOW BMI / UNDERWEIGHT — increase nutritional intake with iron-rich and protein-rich foods.")
    if raw_values["hr"] > 100:
        recs.append("ELEVATED HEART RATE — avoid caffeine, rest frequently, and report to your doctor if persistent.")
    if raw_values["temp"] > 99.5:
        recs.append("ELEVATED BODY TEMPERATURE — monitor for fever; seek medical evaluation if above 100.4°F.")
    if raw_values["age"] < 18:
        recs.append("ADOLESCENT PREGNANCY — additional monitoring and nutritional support are especially important.")
    if raw_values["age"] > 40:
        recs.append("ADVANCED MATERNAL AGE — schedule more frequent antenatal visits and consider genetic counselling.")
    if prev:
        recs.append("PREVIOUS COMPLICATIONS history — inform your obstetric team of all prior pregnancy issues.")
    if dm:
        recs.append("PREEXISTING DIABETES — tight glucose control is critical; work closely with your endocrinologist throughout pregnancy.")
    if gdm:
        recs.append("GESTATIONAL DIABETES — monitor blood sugar before and after meals; follow your dietitian's meal plan.")
    if mh:
        recs.append("MENTAL HEALTH CONCERNS — connect with a perinatal mental health counsellor; emotional support is part of safe care.")

    # Always include these baseline recommendations
    recs.append("Attend all scheduled prenatal visits and do not skip follow-up appointments.")
    recs.append("Seek immediate medical care if you experience severe headache, vision changes, chest pain, or reduced fetal movement.")

    return recs


def check_vitals_status(raw_values):
    """Return dict of {label: (value_str, is_abnormal)} for the vitals table."""
    return {
        "Age":               (str(raw_values["age"]),     raw_values["age"] < 15 or raw_values["age"] > 45),
        "Systolic BP":       (str(raw_values["sys"]),     raw_values["sys"] > 140 or raw_values["sys"] < 90),
        "Diastolic BP":      (str(raw_values["dia"]),     raw_values["dia"] > 90  or raw_values["dia"] < 60),
        "Heart Rate":        (str(raw_values["hr"]),      raw_values["hr"] > 100  or raw_values["hr"] < 60),
        "BMI":               (str(raw_values["bmi"]),     raw_values["bmi"] > 30  or raw_values["bmi"] < 18.5),
        "Blood Sugar":       (str(raw_values["bs"]),      raw_values["bs"] > 7.8  or raw_values["bs"] < 3.9),
        "Body Temperature":  (str(raw_values["temp"]),    raw_values["temp"] > 99.5),
    }


def generate_pdf(pred_text, high, low, inputs, raw_values=None, patient_name=None, hospital_name=None, top_factors=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.alignment = TA_CENTER
    heading = styles["Heading2"]
    normal = styles["BodyText"]
    story = []

    # --- Header ---
    story.append(Paragraph("MATERNAL HEALTH RISK ASSESSMENT REPORT", title_style))
    if hospital_name:
        story.append(Paragraph(f"<b>{hospital_name}</b>", styles["Heading3"]))
    story.append(Paragraph("<b>AI Assisted Clinical Decision Support Report</b>", styles["Heading3"]))
    story.append(Spacer(1, 10))

    # --- Emergency alert banner ---
    if raw_values and (raw_values.get("sys", 0) > 160 or raw_values.get("dia", 0) > 110):
        alert_data = [["⚠️  CRITICAL: Severely elevated blood pressure detected. Seek emergency care immediately."]]
        alert_table = Table(alert_data, colWidths=[480])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FF0000")),
            ("TEXTCOLOR",  (0, 0), (-1, -1), colors.white),
            ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 11),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ]))
        story.append(alert_table)
        story.append(Spacer(1, 10))

    # --- Patient info ---
    story.append(Paragraph("<b>Patient Information</b>", heading))
    patient_data = [["Parameter", "Value"]]
    if patient_name:
        patient_data.append(["Patient Name / ID", patient_name])
    for k, v in inputs.items():
        patient_data.append([k, str(v)])
    patient_table = Table(patient_data, colWidths=[240, 240])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.whitesmoke),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 15))

    # --- Color-coded vitals status table ---
    if raw_values:
        story.append(Paragraph("<b>Vital Signs Status</b>", heading))
        story.append(Paragraph(
            "<font size=9 color='gray'>Values highlighted in red are outside the normal reference range.</font>",
            normal,
        ))
        story.append(Spacer(1, 6))
        vitals_status = check_vitals_status(raw_values)
        vitals_data = [["Vital Sign", "Value", "Status"]]
        vitals_style_cmds = [
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]
        for i, (label, (val, abnormal)) in enumerate(vitals_status.items(), start=1):
            status_text = "ABNORMAL" if abnormal else "Normal"
            vitals_data.append([label, val, status_text])
            if abnormal:
                vitals_style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FDDEDE")))
                vitals_style_cmds.append(("TEXTCOLOR",  (2, i), (2, i), colors.HexColor("#CC0000")))
                vitals_style_cmds.append(("FONTNAME",   (2, i), (2, i), "Helvetica-Bold"))
            else:
                vitals_style_cmds.append(("TEXTCOLOR",  (2, i), (2, i), colors.HexColor("#207520")))
        vitals_table = Table(vitals_data, colWidths=[200, 140, 140])
        vitals_table.setStyle(TableStyle(vitals_style_cmds))
        story.append(vitals_table)
        story.append(Spacer(1, 15))

    # --- Prediction result ---
    story.append(Paragraph("<b>Prediction Result</b>", heading))
    if high > 0.7:
        result_color = colors.red
    elif high > 0.4:
        result_color = colors.orange
    else:
        result_color = colors.green
    result_table = Table([
        ["Prediction",          pred_text],
        ["High Risk Probability", f"{high:.1%}"],
        ["Low Risk Probability",  f"{low:.1%}"],
    ], colWidths=[240, 240])
    result_table.setStyle(TableStyle([
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D9EAD3")),
        ("BACKGROUND", (1, 0), (1, 0),  result_color),
        ("TEXTCOLOR",  (1, 0), (1, 0),  colors.white),
        ("FONTNAME",   (1, 0), (1, 0),  "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 15))

    # --- Top SHAP factors ---
    if top_factors:
        story.append(Paragraph("<b>Top Factors for This Patient (SHAP)</b>", heading))
        factor_data = [["Feature", "Effect on Risk"]]
        for feat, val in top_factors:
            direction = "increases risk" if val > 0 else "decreases risk"
            factor_data.append([feat, direction])
        factor_table = Table(factor_data, colWidths=[240, 240])
        factor_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(factor_table)
        story.append(Spacer(1, 15))

    # --- Clinical interpretation ---
    story.append(Paragraph("<b>Clinical Interpretation</b>", heading))
    if high > 0.7:
        interp = ("The patient demonstrates HIGH maternal health risk. "
                  "Immediate medical evaluation and close antenatal monitoring are strongly recommended.")
    elif high > 0.4:
        interp = "Moderate maternal risk detected. Follow-up examinations and regular monitoring are advised."
    else:
        interp = ("Low maternal health risk detected. Continue routine antenatal care "
                  "and maintain healthy lifestyle practices.")
    story.append(Paragraph(interp, normal))
    story.append(Spacer(1, 15))

    # --- Personalized recommendations ---
    story.append(Paragraph("<b>Personalised Recommendations</b>", heading))
    if raw_values:
        prev_val = inputs.get("Previous Complications") == "Yes"
        dm_val   = inputs.get("Preexisting Diabetes") == "Yes"
        gdm_val  = inputs.get("Gestational Diabetes") == "Yes"
        mh_val   = inputs.get("Mental Health Concerns") == "Yes"
        recs = get_personalized_recommendations(raw_values, prev_val, dm_val, gdm_val, mh_val)
        for rec in recs:
            story.append(Paragraph(f"• {rec}", normal))
    else:
        story.append(Paragraph("• Maintain scheduled prenatal visits.", normal))
        story.append(Paragraph("• Monitor blood pressure regularly.", normal))
        story.append(Paragraph("• Maintain blood glucose control.", normal))
        story.append(Paragraph("• Follow a balanced diet.", normal))
        story.append(Paragraph("• Seek immediate medical care if severe symptoms occur.", normal))

    story.append(Spacer(1, 20))

    # --- Signature + footer ---
    story.append(Paragraph("Doctor Signature: __________________________", normal))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}", normal))
    story.append(Paragraph("Report ID: MH-" + datetime.now().strftime("%Y%m%d%H%M"), normal))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<font size=9 color='gray'>"
        "This report is generated using an AI model and is intended for educational purposes only. "
        "It should not replace professional medical diagnosis."
        "</font>",
        normal,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# LOAD ARTIFACTS

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/metrics.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_dataset():
    return pd.read_csv("data/maternal_health_data.csv")


model, scaler, metadata = load_artifacts()
explainer = load_shap_explainer(model)

FEATURES = metadata["feature_columns"]
BEST_MODEL = metadata["best_model_name"]


# SIDEBAR

st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", ["Predict Risk", "Model Insights", "About"])

st.sidebar.markdown("---")
theme_choice = st.sidebar.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
new_theme = "dark" if theme_choice else "light"
if new_theme != st.session_state.theme:
    st.session_state.theme = new_theme
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="info-box">
    <b>Model:</b> {BEST_MODEL}<br>
    <b>Dataset:</b> {metadata['dataset_size']} patients<br>
    <b>Class Split:</b> {metadata['class_distribution']['Low']} Low / {metadata['class_distribution']['High']} High
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Educational/portfolio project — not a substitute for professional medical advice."
)


# PAGE 1 — PREDICT

if page == "Predict Risk":

    st.markdown('<div class="main-header">Maternal Health Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Enter patient vitals to estimate risk using a trained {BEST_MODEL} model.</div>',
        unsafe_allow_html=True,
    )

    with st.form("form"):
        with st.expander("👤 Patient Info", expanded=True):
            pi1, pi2 = st.columns(2)
            patient_name = pi1.text_input("Patient Name / ID", placeholder="e.g. P-00123 or Jane D.")
            hospital_name = pi2.text_input("Hospital / Clinic Name", placeholder="e.g. Colombo General Hospital")

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
        low = proba[0]

        st.markdown("---")
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
            m2.metric("Low Risk", f"{low:.1%}")
            m3.metric("Confidence", f"{max(proba):.1%}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=high * 100,
                title={"text": "Risk Probability", "font": {"color": T["text"]}},
                number={"font": {"color": T["text"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": T["text"]},
                    "bar": {"color": "#e74c3c" if pred == 1 else "#2ecc71"},
                    "bgcolor": T["plot_bg"],
                    "steps": [
                        {"range": [0, 40], "color": T["low_bg"]},
                        {"range": [40, 70], "color": "rgba(243,156,18,0.18)"},
                        {"range": [70, 100], "color": T["high_bg"]},
                    ],
                },
            ))
            fig = themed(fig, height=280)
            fig.update_layout(margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- SHAP per-patient explanation ----------------
        st.markdown("### 🧠 Why this prediction? (Patient-specific explanation)")
        st.caption(
            "Generated with SHAP — shows how each of this patient's specific values pushed the "
            "prediction toward Low or High risk, not just which features matter in general."
        )

        shap_values = explainer.shap_values(X_input)
        patient_shap = shap_values[0, :, 1]

        shap_df = pd.DataFrame({
            "Feature": FEATURES,
            "Value": X_input.iloc[0].values,
            "Effect": patient_shap,
        }).sort_values("Effect", key=lambda s: s.abs(), ascending=False)

        shap_df["Direction"] = np.where(shap_df["Effect"] > 0, "Increases risk", "Decreases risk")

        fig_shap = px.bar(
            shap_df,
            x="Effect",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={"Increases risk": "#e74c3c", "Decreases risk": "#2ecc71"},
            hover_data={"Value": True},
        )
        fig_shap = themed(fig_shap, height=380)
        fig_shap.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10),
            xaxis_title="Contribution to High Risk Probability",
            legend_title_text="Direction",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        top_factors = list(zip(shap_df["Feature"].head(5), shap_df["Effect"].head(5)))

        # ---------------- PDF report download ----------------
        st.markdown("### 📄 Download Report")
        pdf_buffer = generate_pdf(
            pred_text=label,
            high=high,
            low=low,
            patient_name=patient_name,
            hospital_name=hospital_name,
            raw_values={
                "age": age, "sys": sys, "dia": dia,
                "hr": hr, "bmi": bmi, "bs": bs, "temp": temp,
            },
            inputs={
                "Age": age,
                "Systolic BP": sys,
                "Diastolic BP": dia,
                "Heart Rate": hr,
                "BMI": bmi,
                "Blood Sugar": bs,
                "Body Temperature": temp,
                "Previous Complications": "Yes" if prev else "No",
                "Preexisting Diabetes": "Yes" if dm else "No",
                "Gestational Diabetes": "Yes" if gdm else "No",
                "Mental Health Concerns": "Yes" if mh else "No",
            },
            top_factors=top_factors,
        )
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"maternal_health_report_{datetime.now().strftime('%Y%m%d%H%M')}.pdf",
            mime="application/pdf",
        )


# PAGE 2 — MODEL INSIGHTS

elif page == "Model Insights":

    st.markdown('<div class="main-header">Model Performance Dashboard</div>', unsafe_allow_html=True)

    metrics = pd.DataFrame(metadata["all_metrics"])
    cv_rows = []
    for row in metadata["all_metrics"]:
        if "CV" in row:
            cv_rows.append({
                "Model": row["Model"],
                "CV Mean Accuracy": row["CV"]["cv_mean"],
                "CV Std Dev": row["CV"]["cv_std"],
            })
    metrics_display = metrics.drop(columns=["CV"], errors="ignore")
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(metrics_display, x="Model", y="Accuracy", text_auto=".3f", color="Model")
        fig.update_layout(template=PLOT_TEMPLATE, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        melted = metrics_display.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1 Score"],
                               var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group")
        fig.update_layout(template=PLOT_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Cross-validation ----------------
    if cv_rows:
        st.markdown("### 🔁 5-Fold Cross-Validation")
        st.caption(
            "A single train/test split can get lucky. Cross-validation re-splits the data "
            "5 times and reports the average — a more honest estimate of how each model "
            "generalizes, and a check against overfitting."
        )
        cv_df = pd.DataFrame(cv_rows).sort_values("CV Mean Accuracy", ascending=False)
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=cv_df["Model"],
            y=cv_df["CV Mean Accuracy"],
            error_y=dict(type="data", array=cv_df["CV Std Dev"], visible=True),
            marker_color=T["accent"],
        ))
        fig_cv.update_layout(
            template=PLOT_TEMPLATE,
            height=380,
            yaxis_title="Mean Accuracy (5-fold CV)",
            yaxis_range=[max(0, cv_df["CV Mean Accuracy"].min() - 0.05), 1.0],
        )
        st.plotly_chart(fig_cv, use_container_width=True)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

    # ---------------- Calibration curve ----------------
    if metadata.get("calibration_curve"):
        st.markdown(f"### 🎯 Calibration Curve ({metadata['best_model_name']})")
        st.caption(
            "When the model says '80% confidence', is it actually right about 80% of the "
            "time? Points on the diagonal line mean the model's probabilities are well "
            "calibrated, not just confidently overfit."
        )
        cal = metadata["calibration_curve"]
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
            line=dict(dash="dash", color=T["text_muted"]),
        ))
        fig_cal.add_trace(go.Scatter(
            x=cal["mean_predicted_prob"], y=cal["fraction_of_positives"],
            mode="lines+markers", name=metadata["best_model_name"],
            line=dict(color=T["accent"]), marker=dict(size=8),
        ))
        fig_cal.update_layout(
            template=PLOT_TEMPLATE,
            height=400,
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of actual positives",
            xaxis_range=[0, 1], yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # ---------------- Precision-Recall curve ----------------
    if metadata.get("pr_curve"):
        st.markdown(f"### ⚖️ Precision-Recall Tradeoff ({metadata['best_model_name']})")
        st.caption(
            "For a health-risk model, missing a high-risk patient (low recall) and "
            "wrongly flagging a low-risk patient (low precision) have different costs. "
            "This shows the tradeoff as the decision threshold changes."
        )
        pr = metadata["pr_curve"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"], mode="lines+markers",
            line=dict(color=T["accent"]), marker=dict(size=6),
        ))
        fig_pr.update_layout(
            template=PLOT_TEMPLATE,
            height=400,
            xaxis_title="Recall (sensitivity to High-risk patients)",
            yaxis_title="Precision",
            xaxis_range=[0, 1.02], yaxis_range=[0, 1.02],
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("### 🔍 Confusion Matrices")
    
    cols = st.columns(len(metadata["confusion_matrices"]))
    for col, (name, cm) in zip(cols, metadata["confusion_matrices"].items()):
        with col:
            fig = px.imshow(np.array(cm), text_auto=True, x=["Low", "High"], y=["Low", "High"],
                             color_continuous_scale="Blues")
            fig = themed(fig, height=260)
            fig.update_layout(title=name, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Feature Importance (Best Model)")
    fig = px.bar(pd.DataFrame(metadata["feature_importance"]), x="Importance", y="Feature",
                 orientation="h", color="Importance", color_continuous_scale="Blues")
    fig = themed(fig, height=420)
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    df = load_dataset()
    st.markdown("### 📂 Dataset Overview")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Risk Level", color="Risk Level")
        fig = themed(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, x="Risk Level", y="Age", color="Risk Level")
        fig = themed(fig)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Correlation Heatmap"):
        corr = df.copy()
        corr["Risk Level"] = corr["Risk Level"].map({"Low": 0, "High": 1})
        fig = px.imshow(corr.corr(numeric_only=True), color_continuous_scale="RdBu_r", text_auto=".2f")
        fig = themed(fig, height=550)
        st.plotly_chart(fig, use_container_width=True)


# PAGE 3 — ABOUT

else:
    st.markdown('<div class="main-header">About This Project</div>', unsafe_allow_html=True)
    st.write(
        """
        This application predicts maternal health risk using machine learning.

        **Features:**
        - 5 ML models trained and compared (Logistic Regression, Decision Tree, Random Forest, KNN, SVM)
        - Real-time risk prediction with confidence scores
        - Patient-specific explanations via SHAP (not just global feature importance)
        - Downloadable PDF clinical report
        - Light/dark theme toggle
        - Interactive model comparison dashboards

        **Tech Stack:** Python, scikit-learn, SHAP, Streamlit, Plotly, ReportLab

        ⚠️ Educational use only — not for clinical decision-making.
        """
    )