"""
Logistic model implementation for intent classification
"""

from sklearn.linear_model import LogisticRegression


def train(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        verbose=1
    )
    model.fit(X_train, y_train)
    return model


def predict(model: LogisticRegression, X):
    return model.predict(X)