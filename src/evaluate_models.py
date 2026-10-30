
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_models(models, X_test, y_test):
    """Evaluate trained models on unseen test data."""

    print("\nEvaluating models...")

    results = []

    for name, model in models.items():

        # Predict churn class: 0 or 1
        y_pred = model.predict(X_test)

        # Predict probability of churn (class 1)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
        roc_auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4)
        })

        print(f"\n{name}")
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))
        print("ROC-AUC:", round(roc_auc, 4))

    # Convert results into a table
    results_df = pd.DataFrame(results)

    # Sort by ROC-AUC, highest first
    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    return results_df
