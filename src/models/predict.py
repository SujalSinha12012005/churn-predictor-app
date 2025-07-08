import pandas as pd
import pickle

# Load the trained model
with open("C:\\Users\\Sujit Kumar Sinha\\ml_project\\models\\logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training columns
with open("C:\\Users\\Sujit Kumar Sinha\\ml_project\\models\\training_columns.pkl", "rb") as f:
    training_columns = pickle.load(f)

# Example raw input
input_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 139.65
}

# Convert to DataFrame
df = pd.DataFrame([input_data])

# One-hot encode
df = pd.get_dummies(df)

# Align with training columns
for col in training_columns:
    if col not in df.columns:
        df[col] = 0  # add missing column
df = df[training_columns]  # reorder columns

# Predict
prediction = model.predict(df)
print("Predicted Churn:", prediction[0])
