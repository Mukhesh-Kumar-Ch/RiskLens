"""
Feature Engineering Module
Handles feature creation and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns"""
    df_copy = df.copy()
    # Add feature engineering logic here
    return df_copy


def scale_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Scale numerical features"""
    scaler = StandardScaler()
    df_copy = df.copy()
    df_copy[columns] = scaler.fit_transform(df[columns])
    return df_copy


def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Encode categorical features"""
    df_copy = df.copy()
    df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
    return df_copy
