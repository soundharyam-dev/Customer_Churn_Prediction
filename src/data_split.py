
from sklearn.model_selection import train_test_split
import pandas as pd

from src.config import RAW_DATA_PATH

def split_data(X, y, test_size=0.2, random_state=42):
    """Split features and target into training and testing sets."""

    print("\nSplitting data into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("Training features shape:", X_train.shape)
    print("Testing features shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)

    return X_train, X_test, y_train, y_test
