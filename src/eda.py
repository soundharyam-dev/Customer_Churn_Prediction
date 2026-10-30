
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUTPUT_DIR


def perform_eda(df):
    """Explore customer churn data and save visualizations."""

    # Create folder for charts
    chart_dir = OUTPUT_DIR / "eda_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting Exploratory Data Analysis...")

    # 1. Churn count plot
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Count")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(chart_dir / "churn_count.png")
    plt.show()
    plt.close()

    # 2. Churn by contract type
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Customer Churn by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(chart_dir / "churn_by_contract.png")
    plt.show()
    plt.close()

   
    

    # 3. Correlation heatmap
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        plt.figure(figsize=(9, 6))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )
        plt.title("Numeric Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(chart_dir / "correlation_heatmap.png")
        plt.show()
        plt.close()

    # Summary statistics
    print("\nNumeric Summary:")
    print(df.describe())

    print("\nChurn Distribution:")
    print(df["Churn"].value_counts())

    print("\nEDA completed!")
    print("Charts saved in:", chart_dir)
