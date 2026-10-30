import pandas as pd
from src.config import RAW_DATA_PATH

def clean_data(df):
    """Clean and validate the customer churn dataset."""

    df = df.copy()

    print("\nStarting data cleaning...")

    # 1. Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    print("Duplicate rows found:", duplicate_count)

    df = df.drop_duplicates()

    # 2. Remove duplicate customer IDs, if present
    if "customerID" in df.columns:
        duplicate_ids = df["customerID"].duplicated().sum()
        print("Duplicate customer IDs found:", duplicate_ids)

        df = df.drop_duplicates(subset=["customerID"])

    # 3. Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        # Blank TotalCharges for customers with zero tenure
    
        zero_tenure = (
            df["tenure"].eq(0)
            if "tenure" in df.columns
            else pd.Series(False, index=df.index)
        )

        df.loc[
            zero_tenure & df["TotalCharges"].isna(),
            "TotalCharges"
        ] = 0

    # 4. Remove rows with missing target values
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])

    # 5. Display remaining missing values
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    print("\nCleaned dataset shape:", df.shape)
    print("Data cleaning completed!")

    return df
    
if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    clean_data(df)