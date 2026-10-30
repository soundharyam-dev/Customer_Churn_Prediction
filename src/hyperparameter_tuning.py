
from sklearn.model_selection import RandomizedSearchCV


def tune_models(models, X_train, y_train):
    """Tune model hyperparameters using cross-validation."""

    print("\nStarting hyperparameter tuning...")

    # Settings to test for Logistic Regression
    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs"]
        },

        # Settings to test for Random Forest
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        },

        # Settings to test for XGBoost
        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1]
        }
    }

    tuned_models = {}

    for name, model in models.items():
        print(f"\nTuning {name}...")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[name],
            n_iter=4,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train)

        tuned_models[name] = search.best_estimator_

        print("Best parameters:", search.best_params_)
        print("Best cross-validation ROC-AUC:", round(search.best_score_, 4))

    print("\nHyperparameter tuning completed!")

    return tuned_models
