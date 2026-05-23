"""
Data Preprocessing Module
Handles data loading, cleaning, and preparation
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple



def prepare_data(df: pd.DataFrame) -> Tuple:
    """Prepare model-ready train/test sets from engineered student risk data.

    Steps:
    1. Select final features and target.
    2. Encode `higher` as binary (`yes` -> 1, `no` -> 0).
    3. Split data with stratification on target.
    4. Scale selected numeric columns.
    """
    # Some engineered/behavioral features were temporarily excluded after coefficient-level
    # instability analysis during Logistic Regression interpretation.
    feature_columns = [
        "failures",
        "G2",
        "age",
        "higher",
        "goout",
        # "grade_change",  # Temporarily excluded: multicollinearity with G2
        # "studytime",  # Temporarily excluded: reverse causality effects
        # "famsup_effective",  # Temporarily excluded: unstable directional behavior
    ]
    target_column = "at_risk"

    prepared_df = df.copy()
    prepared_df["higher"] = prepared_df["higher"].map({"yes": 1, "no": 0})

    X = prepared_df[feature_columns].copy()
    y = prepared_df[target_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    # Scale only the remaining continuous features used in the final feature set.
    scale_columns = ["G2", "age"]
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[scale_columns] = scaler.fit_transform(X_train[scale_columns])
    X_test_scaled[scale_columns] = scaler.transform(X_test[scale_columns])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
