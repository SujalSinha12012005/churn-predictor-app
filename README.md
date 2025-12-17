# Customer Churn Prediction Web App

🚀 **Live App:**  
https://sujalsinha12012005-churn-predictor-app-webappapp-z2vs8j.streamlit.app/

## 📌 Overview
This project is a **Customer Churn Prediction Web Application** built using **Machine Learning** and deployed with **Streamlit**.  
The app predicts whether a customer is likely to **churn (leave the service)** based on various customer attributes.

It is designed to be **interactive, user-friendly, and suitable for real-world business use cases**, helping companies take proactive actions to retain customers.

---

## 🧠 Machine Learning Model
- The model is trained on customer-related features such as:
  - Demographics
  - Account information
  - Service usage details
- After training, the model predicts:
  - **Churn** ❌
  - **No Churn** ✅

> The trained model is saved and loaded inside the Streamlit app for real-time predictions.

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** – for web application
- **Scikit-learn** – for ML model training
- **Pandas & NumPy** – data processing
- **Pickle / Joblib** – model serialization

---

## 🎯 Features
- Interactive UI for entering customer details
- Real-time churn prediction
- Clean and simple design
- Deployed and accessible online

---

## 🧪 How It Works
1. User enters customer details using the web interface.
2. Input data is preprocessed.
3. The trained ML model predicts churn probability.
4. Result is displayed instantly on the screen.

---

## ▶️ Run Locally
Follow these steps to run the app on your local machine:

```bash
# Clone the repository
git clone https://github.com/your-username/churn-predictor-app.git

# Navigate to the project directory
cd churn-predictor-app

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
