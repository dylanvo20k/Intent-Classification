"""
Random forest implementation for intent classification
"""

from sklearn.ensemble import RandomForestClassifier


def train(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    model.fit(X_train, y_train)
    return model


def predict(model: RandomForestClassifier, X):
    return model.predict(X)