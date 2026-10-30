
import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_DIR


def plot_feature_importance(models, top_n=15):
    """Plot the most important features for tree-based models."""

    importance_dir = OUTPUT_DIR / "feature_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)

    for name in ["Random Forest", "XGBoost"]:

        if name not in models:
            continue

        model = models[name]

        # Get the fitted preprocessing step
        preprocessor = model.named_steps["preprocessor"]

        # Get the trained classifier
        classifier = model.named_steps["model"]

        # Get feature names after encoding
        feature_names = preprocessor.get_feature_names_out()

        # Get importance scores from the tree-based model
        importances = classifier.feature_importances_

        # Create a table of features and their importance
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })

        # Select the top features
        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(top_n)

        # Plot the results
        plt.figure(figsize=(10, 7))
        plt.barh(
            importance_df["Feature"][::-1],
            importance_df["Importance"][::-1]
        )

        plt.title(f"Top {top_n} Feature Importances - {name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save the chart
        safe_name = name.lower().replace(" ", "_")
        chart_path = importance_dir / f"{safe_name}_importance.png"
        plt.savefig(chart_path)
        plt.show()
        plt.close()

        print(f"\nTop features for {name}:")
        print(importance_df.to_string(index=False))
        print("Chart saved to:", chart_path)

    print("\nFeature importance analysis completed!")
