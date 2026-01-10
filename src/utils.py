"""
Utility Functions
Helper functions for the project
"""

import os
import json
import pickle
from typing import Any


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model: Any, filepath: str) -> None:
    """Save model to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """Load model from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_metrics(metrics: dict, filepath: str) -> None:
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
