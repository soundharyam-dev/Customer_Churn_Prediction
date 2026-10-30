
from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.eda import perform_eda
from src.feature_engineering import prepare_features
from src.data_split import split_data
from src.train_models import train_models
from src.hyperparameter_tuning import tune_models
from src.evaluate_models import evaluate_models
from src.feature_importance import plot_feature_importance
from src.predict import save_predictions
import joblib

from src.config import MODEL_DIR
from src.config import OUTPUT_DIR
def main():
    # Step 1: Load raw data
    df = load_data()

    # Step 2: Clean data
    cleaned_df = clean_data(df)

    # Step 3: Perform EDA
    perform_eda(cleaned_df)

    print("\nFinal cleaned dataset preview:")
    # print(cleaned_df.head())

    #step 4 After loading and cleaning the dataset:
    X, y, preprocessor = prepare_features(cleaned_df)

    print("\nFeature data shape:", X.shape)
    print("Target data shape:", y.shape)
    print("\nTarget preview:")
    print(y.head())

    # Step 5: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 6: Train the models
    models = train_models(X_train, y_train, preprocessor)

    # Step 7: Tune the models
    tuned_models = tune_models(models, X_train, y_train)
    # Step 8: Evaluate tuned models
    results_df = evaluate_models(
    tuned_models,
    X_test,
    y_test
    )
    # Save model comparison results
    comparison_path = OUTPUT_DIR / "model_comparison.csv"

    results_df.to_csv(comparison_path, index=False)

    print("Model comparison saved to:", comparison_path)
    print(results_df)

    # Step 9: Visualize important features
    plot_feature_importance(tuned_models)
 
    # Step 10: Select the best model using ROC-AUC
    best_model_name = results_df.iloc[0]["Model"]
    best_model = tuned_models[best_model_name]

    print("\nBest model:", best_model_name)

    # Step 11: Generate and save test predictions
    predictions_df = save_predictions(
    best_model,
    X_test,
    y_test
)
    # Select the best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = tuned_models[best_model_name]

    print("Best model:", best_model_name)

    # Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "best_churn_model.joblib"

    joblib.dump(best_model, model_path)

    print("Best model saved successfully!")
    print("Saved location:", model_path)

    

if __name__ == "__main__":
    main()
   