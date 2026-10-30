import pandas as pd

from src.config import RAW_DATA_PATH


def load_data():
    """Load the raw customer churn dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "Please place raw_churn.csv inside the data folder."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    return df
if __name__ == "__main__":
    load_data()
