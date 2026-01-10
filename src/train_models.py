"""
Model Training Module
Handles model training and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from typing import Any


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train logistic regression model"""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100) -> RandomForestClassifier:
    """Train random forest model"""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100) -> GradientBoostingClassifier:
    """Train gradient boosting model"""
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model
