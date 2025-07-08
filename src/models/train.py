import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# --- File paths ---
data_path = r"C:\\Users\Sujit Kumar Sinha\\ml_project\data\\processed\\customers_clean.csv"
model_path = r"C:\\Users\\Sujit Kumar Sinha\\ml_project\\models\\logistic_model.pkl"
columns_path = r"C:\\Users\\Sujit Kumar Sinha\\ml_project\\models\\training_columns.pkl"

# --- Load data ---
df = pd.read_csv(data_path)
print("Initial DataFrame shape:", df.shape)

# --- Fix Churn column if needed ---
print("Churn value counts BEFORE:\n", df['Churn'].value_counts(dropna=False))
if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].str.strip().map({'Yes': 1, 'No': 0})
print("Churn value counts AFTER:\n", df['Churn'].value_counts(dropna=False))

# --- Drop rows where Churn is NaN (after mapping) ---
df = df.dropna(subset=['Churn'])

# --- Drop customerID if exists ---
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# --- Separate features and target ---
X = df.drop('Churn', axis=1)
y = df['Churn']

# --- One-hot encode categorical variables ---
X = pd.get_dummies(X)

# --- Check for nulls ---
print("Final shape of X:", X.shape)
print("Final shape of y:", y.shape)
print("Any nulls in X?", X.isnull().any().any())
print("Any nulls in y?", y.isnull().any())

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate model ---
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Save model and columns ---
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
joblib.dump(X.columns.tolist(), columns_path)
print(f"Model saved to: {model_path}")
print(f"Training columns saved to: {columns_path}")      #train.py