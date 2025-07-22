# app.py
import streamlit as st
import requests

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

st.title(" Telco Customer Churn Predictor")

st.markdown("Fill in the customer information below:")

# -----------------------------
# Step 1: User Input Section
# -----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# -----------------------------
# Step 2: API Call on Submit
# -----------------------------
if st.button(" Predict Churn"):
    with st.spinner("Contacting ML model..."):

        # Prepare input data
        input_data = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges
        }

        # Send POST request to FastAPI backend
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"input": input_data}
            )
            if response.status_code == 200:
                result = response.json()
                pred = result['prediction']
                prob = result['probability']

                st.success(f"✅ Prediction: {'Churn' if pred == 1 else 'No Churn'}")
                st.info(f" Probability of churn: {prob*100:.2f}%")
            else:
                st.error("❌ Something went wrong. Try again.")
        except Exception as e:
            st.error(f"❌ API error: {e}")
