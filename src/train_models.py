
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models(X_train, y_train, preprocessor):
    """Train three machine learning models for customer churn prediction."""

    print("\nStarting model training...")

    # 1. Logistic Regression
    logistic_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 2. Random Forest
    random_forest_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # 3. XGBoost
    xgboost_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Store all models in a dictionary
    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgboost_model
    }

    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        print(f"{name} training completed!")

    return models
