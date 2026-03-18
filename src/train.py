"""
Trains all models and saves predictions for evaluation.
"""

import os
import pickle
import numpy as np
from preprocess import get_tfidf_data
from models import logistic, naive_bayes, random_forest, ffn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "eval", "results")


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading and preprocessing data")
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, le = get_tfidf_data()

    # save label encoder for evaluate.py
    save(le, os.path.join(RESULTS_DIR, "label_encoder.pkl"))
    save(y_test, os.path.join(RESULTS_DIR, "y_test.pkl"))

    # Logistic Regression
    print("\nTraining Logistic Regression")
    lr_model = logistic.train(X_train, y_train)
    lr_preds = logistic.predict(lr_model, X_test)
    save(lr_preds, os.path.join(RESULTS_DIR, "preds_logistic.pkl"))

    # Naive Bayes
    print("\nTraining Naive Bayes")
    nb_model = naive_bayes.train(X_train, y_train)
    nb_preds = naive_bayes.predict(nb_model, X_test)
    save(nb_preds, os.path.join(RESULTS_DIR, "preds_naive_bayes.pkl"))

    # Random Forest
    print("\nTraining Random Forest")
    rf_model = random_forest.train(X_train, y_train)
    rf_preds = random_forest.predict(rf_model, X_test)
    save(rf_preds, os.path.join(RESULTS_DIR, "preds_random_forest.pkl"))

    # Feedforward Neural Network 
    print("\nTraining Feedforward Neural Network")
    ffn_model = ffn.train(X_train, y_train, X_val, y_val)
    ffn_preds = ffn.predict(ffn_model, X_test)
    save(ffn_preds, os.path.join(RESULTS_DIR, "preds_ffn.pkl"))

    print("\nAll models trained. Predictions saved to eval/results/")


if __name__ == "__main__":
    main()