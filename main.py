"""Main execution pipeline for student academic risk prediction."""

import pandas as pd

from src.feature_engineering import create_features
from src.data_preprocessing import prepare_data
from src.train_models import train_logistic_regression
from src.evaluate_models import evaluate_model, show_feature_coefficients, plot_feature_coefficients
from create_pipeline_diagram import create_pipeline_diagram


def run_pipeline(dataset_path: str, dataset_name: str, coefficient_plot_path: str) -> None:
	"""Run the end-to-end pipeline for a single dataset."""
	print("\n==============================")
	print(f"{dataset_name} DATASET RESULTS")
	print("==============================")

	# This experiment evaluates whether predictive and interpretability patterns
	# remain stable across academic subjects.

	# Stage A: Load dataset
	df = pd.read_csv(dataset_path, sep=";")

	# Ensure target exists for downstream preprocessing module.
	if "at_risk" not in df.columns:
		df["at_risk"] = (df["G3"] < 10).astype(int)

	# Stage B: Apply feature engineering
	df = create_features(df)

	# Stage C: Prepare train/test data
	X_train_scaled, X_test_scaled, y_train, y_test, _ = prepare_data(df)

	# Stage D: Train model
	model = train_logistic_regression(X_train_scaled, y_train)

	# Stage E: Evaluate model
	evaluate_model(model, X_test_scaled, y_test)

	# Stage F: Display coefficients and save coefficient visualization
	show_feature_coefficients(model, X_train_scaled.columns)
	plot_feature_coefficients(model, X_train_scaled.columns, output_path=coefficient_plot_path)


def main() -> None:
	"""Run comparative pipeline experiments across both academic subjects."""
	run_pipeline(
		dataset_path="data/raw/student-mat.csv",
		dataset_name="MATHEMATICS",
		coefficient_plot_path="results/figures/math_feature_coefficients.png",
	)

	run_pipeline(
		dataset_path="data/raw/student-por.csv",
		dataset_name="PORTUGUESE",
		coefficient_plot_path="results/figures/portuguese_feature_coefficients.png",
	)

	# Generate a static architecture diagram once for project documentation.
	create_pipeline_diagram()


if __name__ == "__main__":
	main()
