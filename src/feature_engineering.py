
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(df, target_column="Churn"):
    """Separate features and target, then build a preprocessing pipeline."""

    df = df.copy()

    # 1. Separate target from input features
    X = df.drop(columns=[target_column])
    y = df[target_column].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "Target column must contain only 'Yes' and 'No' values."
        )

    # 2. Remove customer ID because it is an identifier
    X = X.drop(columns=["customerID"], errors="ignore")

    # 3. Identify numeric and categorical columns
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # 4. Preprocess numeric features
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 5. Preprocess categorical features
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 6. Apply the correct preprocessing to each column type
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    return X, y, preprocessor
