import streamlit as st
import pandas as pd
import joblib
import os

# --- Load model and training columns using relative paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "logistic_model.pkl")
columns_path = os.path.join(BASE_DIR, "..", "models", "training_columns.pkl")

# Load model
model = joblib.load(model_path)

# Load training columns
with open(columns_path, "rb") as f:
    training_columns = joblib.load(f)

# --- Streamlit UI ---
st.title("Customer Churn Prediction")

# Input fields
st.header("Enter Customer Details:")
input_data = {
    "gender": st.selectbox("Gender", ["Male", "Female"]),
    "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.selectbox("Partner", ["Yes", "No"]),
    "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
    "tenure": st.slider("Tenure (months)", 0, 72, 12),
    "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
    "MultipleLines": st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
    "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
    "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
    "TechSupport": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
    "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
    "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
    "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": st.selectbox("Paperless Billing", ["Yes", "No"]),
    "PaymentMethod": st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    "MonthlyCharges": st.number_input("Monthly Charges", value=70.0),
    "TotalCharges": st.number_input("Total Charges", value=200.0)
}

# --- Predict ---
if st.button("Predict Churn"):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    # Align with training columns
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    st.subheader("Prediction Result:")
    st.write("Churn:" if prediction == 1 else "No Churn")
    st.write(f"Churn Probability: {probability:.2%}")
