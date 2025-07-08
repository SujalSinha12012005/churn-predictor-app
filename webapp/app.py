# app.py
import streamlit as st
import pandas as pd
import joblib

# --- Load model and columns ---
model = joblib.load(r"C:\\Users\Sujit Kumar Sinha\\ml_project\\models\\logistic_model.pkl")
columns = joblib.load(r"C:\\Users\\Sujit Kumar Sinha\\ml_project\\models\\training_columns.pkl")

st.title("Customer Churn Predictor")

# --- UI for inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

# --- Predict button ---
if st.button("Predict Churn"):
    # Input to DataFrame
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phoneservice,
        "MultipleLines": multiplelines,
        "InternetService": internet,
        "OnlineSecurity": onlinesecurity,
        "OnlineBackup": onlinebackup,
        "DeviceProtection": deviceprotection,
        "TechSupport": techsupport,
        "StreamingTV": streamingtv,
        "StreamingMovies": streamingmovies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)

    # Align with training columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    # Make prediction
    pred = model.predict(df)[0]
    label = "Yes" if pred == 1 else "No"
    st.subheader(f"🔮 Churn Prediction: **{label}**")
