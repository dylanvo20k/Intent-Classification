"""
Naive-bayes model implementation for intent classification
"""

from sklearn.naive_bayes import MultinomialNB


def train(X_train, y_train) -> MultinomialNB:
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def predict(model: MultinomialNB, X):
    return model.predict(X)