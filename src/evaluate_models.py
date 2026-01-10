"""
Model Evaluation Module
Handles model evaluation and metrics calculation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def get_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
    """Get confusion matrix"""
    return confusion_matrix(y_true, y_pred)


def compare_models(models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Compare multiple models"""
    return pd.DataFrame(models_results).T
