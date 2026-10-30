
import pandas as pd

from src.config import OUTPUT_DIR


def save_predictions(model, X_test, y_test):
    """Predict customer churn and save results to a CSV file."""

    print("\nGenerating final predictions...")

    # Predict churn: 0 = No, 1 = Yes
    predictions = model.predict(X_test)

    # Predict probability of churn
    probabilities = model.predict_proba(X_test)[:, 1]

    # Create a results table
    results_df = pd.DataFrame({
        "Actual_Churn": y_test.to_numpy(),
        "Predicted_Churn": predictions,
        "Churn_Probability": probabilities.round(4)
    })

    # Convert numeric labels back to readable values
    results_df["Actual_Churn"] = results_df["Actual_Churn"].map({
        0: "No",
        1: "Yes"
    })

    results_df["Predicted_Churn"] = results_df["Predicted_Churn"].map({
        0: "No",
        1: "Yes"
    })

    # Save predictions to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "churn_predictions.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFirst five predictions:")
    print(results_df.head())

    print("\nPredictions saved to:", output_path)

    return results_df
