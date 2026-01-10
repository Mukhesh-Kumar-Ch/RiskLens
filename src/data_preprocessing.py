"""
Data Preprocessing Module
Handles data loading, cleaning, and preparation
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file"""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and handle missing values"""
    df = df.dropna()
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test
