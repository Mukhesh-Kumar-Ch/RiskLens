# AI-Driven Student Performance & Risk Prediction System

A machine learning project to predict student risk factors and outcomes.

## Problem Statement

Academic institutions often struggle to identify students who are at risk of poor academic performance early enough for effective intervention. Traditional approaches rely on manual monitoring and final examination results, which limits the ability to provide timely support.

This project aims to build a data-driven system that predicts whether a student is at academic risk using historical academic and behavioral data. The goal is not just to achieve high predictive accuracy, but to generate actionable insights that can support early academic interventions.

## Project Objective

The objective of this project is to develop a supervised learning model that classifies students as "at risk" or "not at risk" based on academic performance indicators and related attributes.

The system is designed to:
- Identify students who are likely to underperform academically
- Analyze key factors contributing to academic risk
- Support data-driven decision-making for early intervention strategies

## Prediction Target

The target variable for this project is a binary classification label:
- **At Risk (1):** Students likely to score below an acceptable academic threshold
- **Not At Risk (0):** Students expected to meet or exceed the threshold

The risk label is derived from final academic performance scores, simulating a real-world academic evaluation scenario.

## Machine Learning Formulation

This problem is formulated as a supervised classification task. Historical student data with known academic outcomes is used to train models that learn patterns associated with academic risk.

Supervised learning is appropriate because:
- Labeled outcome data is available
- The objective is to predict a discrete risk category
- Model performance can be evaluated using standard classification metrics

## Project Scope and Assumptions

- The model focuses on academic and behavioral features available during the academic term
- The system is designed for early risk identification, not final academic judgment
- The project does not aim to replace human decision-making but to assist it

## Dataset Description

This project uses the Student Performance Dataset from the UCI Machine Learning Repository. The dataset contains academic, demographic, and behavioral attributes of secondary school students.

Key features include study time, number of past failures, absences, family support indicators, and prior academic performance.

## Target Variable Definition

The target variable `at_risk` is derived from the final grade (G3):

- **at_risk = 1:** G3 < 10 (student at academic risk)
- **at_risk = 0:** G3 ≥ 10 (student not at risk)

This formulation reflects a realistic academic risk identification scenario.

## Project Structure

```
student-risk-prediction/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration and analysis
├── src/
│   ├── data_preprocessing.py       # Data loading and cleaning
│   ├── feature_engineering.py      # Feature creation and transformation
│   ├── train_models.py             # Model training
│   ├── evaluate_models.py          # Model evaluation
│   └── utils.py                    # Utility functions
├── models/               # Saved model files
├── results/
│   ├── figures/          # Visualization outputs
│   └── metrics/          # Performance metrics
├── main.py               # Main execution script
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

Run the main script to preprocess data, train models, and evaluate performance:
```bash
python main.py
```

## Requirements

- Python 3.8+
- See requirements.txt for detailed dependencies

## Author

Ch. Mukhesh Kumar
B.Tech Computer Science & Engineering

## License

MIT License
