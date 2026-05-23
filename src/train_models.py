"""
Model Training Module
Contains the interpretable baseline training function.
"""

from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train an interpretable Logistic Regression baseline model.

    Logistic Regression is used as the baseline because its coefficients are
    straightforward to interpret, which supports transparent academic risk
    analysis.

    Returns:
        A trained ``LogisticRegression`` model fitted on ``X_train`` and
        ``y_train``.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model
