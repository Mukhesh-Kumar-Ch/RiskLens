"""
Model Evaluation Module
Contains evaluation utilities separated from training logic.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test) -> None:
    """Evaluate a trained model using standard classification outputs.

    Evaluation is intentionally separated from training so model fitting and
    performance assessment remain modular, reusable, and easy to test.
    """
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Plot and save a confusion matrix for the trained model.

    Visualizing the confusion matrix separately keeps evaluation modular while
    making it easy to inspect class-specific errors for the risk prediction
    baseline.
    """
    matrix = confusion_matrix(y_test, y_pred)

    figure_path = Path("results/figures/confusion_matrix.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_labels = ["Not At Risk", "At Risk"]
    plt.xticks(range(len(tick_labels)), tick_labels)
    plt.yticks(range(len(tick_labels)), tick_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    threshold = matrix.max() / 2.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


def show_feature_coefficients(model, feature_names) -> None:
    """Print Logistic Regression coefficients for interpretable risk analysis.

    Coefficient interpretation matters because it shows how each feature
    directionally contributes to predicted academic risk, supporting a
    transparent baseline model.
    """
    coefficients_df = _build_coefficients_dataframe(model, feature_names)

    print("Feature Coefficients (descending):")
    print(coefficients_df.to_string(index=False))


def plot_feature_coefficients(model, feature_names, output_path: str = "results/figures/feature_coefficients.png") -> None:
    """Plot Logistic Regression coefficients and save the figure.

    Visualizing coefficients helps compare feature direction and magnitude at a
    glance, which is useful when interpreting an explainable baseline model.
    """
    coefficients_df = _build_coefficients_dataframe(model, feature_names)

    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.barh(coefficients_df["Feature"], coefficients_df["Coefficient"], color="steelblue")
    plt.title("Logistic Regression Feature Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature Name")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


def _build_coefficients_dataframe(model, feature_names) -> pd.DataFrame:
    """Build a sorted DataFrame of Logistic Regression coefficients."""
    return pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_[0],
        }
    ).sort_values(by="Coefficient", ascending=False)
