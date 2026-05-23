"""
Feature Engineering Module
Handles feature creation and transformation
"""

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the full set of engineered features for the dataset.
    """
    df_copy = df.copy()

    df_copy = add_grade_change(df_copy)
    df_copy = add_famsup_effective(df_copy)

    return df_copy


def add_grade_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create grade_change feature using:
    grade_change = G2 - G1
    """

    df_copy = df.copy()

    df_copy["grade_change"] = (
        df_copy["G2"] - df_copy["G1"]
    )

    return df_copy


def add_famsup_effective(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create famsup_effective feature.

    Logic:
    If family educational support exists:
        famsup_effective = Medu + Fedu
    Else:
        famsup_effective = 0
    """

    df_copy = df.copy()

    famsup_mask = (
        df_copy["famsup"] == "yes"
    )

    df_copy["famsup_effective"] = (
        (df_copy["Medu"] + df_copy["Fedu"])
        .where(famsup_mask, 0)
    )

    return df_copy