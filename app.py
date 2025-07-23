# Telco Customer Churn Prediction (Streamlit-Only Version)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# Load saved artifacts

model = joblib.load("models/lgbm_model_tuned.pkl")
scaler = joblib.load("models/scaler.pkl")
scaled_cols = joblib.load("models/scaled_columns.pkl")
all_features = joblib.load("models/feature_columns.pkl")


# Streamlit UI

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("Telco Customer Churn Predictor")
st.markdown("This app predicts whether a customer will churn using a trained ML model.")

# Collect user input
with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", ["Mailed check", "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check"])

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, step=1.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, step=1.0)

    submit_btn = st.form_submit_button("Predict Churn")


# Feature Engineering

def engineer_features(input_df):
    df = input_df.copy()

    # tenure_group
    def tenure_bucket(x):
        if x <= 12: return "0-1 yr"
        elif x <= 24: return "1-2 yrs"
        elif x <= 48: return "2-4 yrs"
        else: return "4+ yrs"
    df["tenure_group"] = df["tenure"].apply(tenure_bucket)

    # interaction terms
    df["charges_ratio"] = df["monthly_charges"] / (df["total_charges"] + 1)
    df["tenure_monthly"] = df["tenure"] * df["monthly_charges"]
    df["log_total_charges"] = np.log1p(df["total_charges"])

    # contract score
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_score"] = df["contract"].map(contract_map)

    return df


# Predict & Explain

if submit_btn:
    raw_input = pd.DataFrame([{
        "gender": gender,
        "senior_citizen": 1 if senior_citizen == "Yes" else 0,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "contract": contract,
        "payment_method": payment_method,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "streaming_tv": streaming_tv,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }])

    # Feature engineering
    fe_input = engineer_features(raw_input)

    # One-hot encode
    fe_input = pd.get_dummies(fe_input)

    # Align with training features
    for col in all_features:
        if col not in fe_input.columns:
            fe_input[col] = 0
    fe_input = fe_input[all_features]

    # Scale numeric columns
    fe_input[scaled_cols] = scaler.transform(fe_input[scaled_cols])

    # Predict
    prediction = model.predict(fe_input)[0]
    prob = model.predict_proba(fe_input)[0][1]

    st.markdown("---")
    st.subheader(" Prediction Result")
    if prediction == 1:
        st.error(f"❌ Customer is likely to churn! (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Customer is likely to stay. (Confidence: {1 - prob:.2%})")

    # SHAP Explanation
    st.markdown("---")
    st.subheader(" Model Explanation (SHAP)")
    shap.initjs()

    explainer = shap.Explainer(model)
    shap_values = explainer(fe_input)



    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)
