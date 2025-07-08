import pandas as pd
import os

def preprocess(input_path, output_path):
    # Load CSV
    df = pd.read_csv(input_path)

    # Drop customerID if it exists
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Clean TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", pd.NA), errors="coerce")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.factorize(df[col])[0]

    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    input_path = "C:\\Users\\Sujit Kumar Sinha\\ml_project\\data\\raw\\customers.csv"
    output_path = "C:\\Users\\Sujit Kumar Sinha\\ml_project\\data\\processed\\customers_clean.csv"
    preprocess(input_path, output_path)
