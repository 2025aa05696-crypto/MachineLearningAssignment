import pandas as pd

from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost import train_xgboost


def create_comparison_table():
    results = []

    # Logistic Regression
    lr_metrics = train_logistic_regression()
    lr_metrics["Model"] = "Logistic Regression"
    results.append(lr_metrics)

    # Decision Tree
    dt_metrics = train_decision_tree()
    dt_metrics["Model"] = "Decision Tree"
    results.append(dt_metrics)

    # KNN
    knn_metrics = train_knn()
    knn_metrics["Model"] = "KNN"
    results.append(knn_metrics)

    # Naive Bayes
    nb_metrics = train_naive_bayes()
    nb_metrics["Model"] = "Naive Bayes"
    results.append(nb_metrics)

    # Random Forest
    rf_metrics = train_random_forest()
    rf_metrics["Model"] = "Random Forest"
    results.append(rf_metrics)

    # XGBoost
    xgb_metrics = train_xgboost()
    xgb_metrics["Model"] = "XGBoost"
    results.append(xgb_metrics)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns as per assignment
    df = df[
        [
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC",
        ]
    ]

    return df


if __name__ == "__main__":
    comparison_df = create_comparison_table()

    print("\nMODEL COMPARISON TABLE\n")
    print(comparison_df)

    # Save to CSV (useful for README & Streamlit)
    comparison_df.to_csv("model_comparison_results.csv", index=False)
