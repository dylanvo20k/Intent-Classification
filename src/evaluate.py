"""
Evaluate all models and save metrics + figures
"""
import os
import sys
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from preprocess import get_tfidf_data
from models import logistic, naive_bayes, random_forest, ffn, distilbert

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "eval", "results")
FIGURES_DIR = os.path.join(ROOT, "eval", "figures")
APPLE_SDK_INF_TIME_MS = 1434868.3
MODELS = ["Logistic Regression", "Naive Bayes", "Random Forest", "FFN", "DistilBERT", "Apple SDK"]


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def measure_inference_time(model_name, model_fn, X_test, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model_fn(X_test)
        times.append(time.perf_counter() - start)
    avg_ms = (sum(times) / n_runs) * 1000
    print(f"  {model_name} inference time: {avg_ms:.1f} ms (avg over {n_runs} runs)")
    return avg_ms


def plot_confusion_matrix(y_test, preds, label_names, model_name):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(FIGURES_DIR, f"confusion_matrix_{fname}.png"), dpi=150)
    plt.close()


def plot_accuracy_comparison(accuracies):
    plt.figure(figsize=(12, 5))
    bars = plt.bar(MODELS, accuracies, color="steelblue", edgecolor="black")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{acc:.3f}", ha="center", fontsize=10)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()


def plot_f1_comparison(f1_scores):
    plt.figure(figsize=(12, 5))
    bars = plt.bar(MODELS, f1_scores, color="steelblue", edgecolor="black")
    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{f1:.3f}", ha="center", fontsize=10)
    plt.ylim(0, 1.0)
    plt.ylabel("Macro F1 Score")
    plt.title("Macro F1 Score by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "f1_comparison.png"), dpi=150)
    plt.close()


def plot_inference_time(times):
    plt.figure(figsize=(12, 5))
    bars = plt.bar(MODELS, times, color="steelblue", edgecolor="black")
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{t:.1f}ms", ha="center", fontsize=10)
    plt.yscale("log")
    plt.ylabel("Avg Inference Time (ms)")
    plt.title("Inference Time by Model (CPU, full test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "inference_time.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data and predictions")
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, le = get_tfidf_data()
    label_names = le.classes_

    # raw utterances for DistilBERT
    test_utts    = load(os.path.join(RESULTS_DIR, "test_utts.pkl"))
    db_tokenizer = load(os.path.join(RESULTS_DIR, "distilbert_tokenizer.pkl"))
    train_utts   = load(os.path.join(RESULTS_DIR, "train_utts.pkl"))  
    val_utts     = load(os.path.join(RESULTS_DIR, "val_utts.pkl"))    


    preds = {
        "Logistic Regression": load(os.path.join(RESULTS_DIR, "preds_logistic.pkl")),
        "Naive Bayes":         load(os.path.join(RESULTS_DIR, "preds_naive_bayes.pkl")),
        "Random Forest":       load(os.path.join(RESULTS_DIR, "preds_random_forest.pkl")),
        "FFN":                 load(os.path.join(RESULTS_DIR, "preds_ffn.pkl")),
        "DistilBERT":          load(os.path.join(RESULTS_DIR, "preds_distilbert.pkl")),
        "Apple SDK":      load(os.path.join(RESULTS_DIR, "preds_apple_sdk.pkl")),
    }

    # retrain lightweight models for inference timing
    print("\nRetraining models for inference timing")
    lr_model = logistic.train(X_train, y_train)
    nb_model = naive_bayes.train(X_train, y_train)
    rf_model = random_forest.train(X_train, y_train)
    ffn_model = ffn.train(X_train, y_train, X_val, y_val)
    db_model, _ = distilbert.train(train_utts, y_train, val_utts, y_val)

    print("\nMetrics ")
    accuracies, f1_scores, inf_times = [], [], []

    model_fns = {
        "Logistic Regression": lambda X: logistic.predict(lr_model, X),
        "Naive Bayes":         lambda X: naive_bayes.predict(nb_model, X),
        "Random Forest":       lambda X: random_forest.predict(rf_model, X),
        "FFN":                 lambda X: ffn.predict(ffn_model, X),
        "DistilBERT":          lambda X: distilbert.predict(db_model, db_tokenizer, test_utts), 
    }

    for name in MODELS:
        y_pred = preds[name]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        if name == "Apple SDK":
            inf_time = APPLE_SDK_INF_TIME_MS
            print(f"  Apple SDK inference time: {inf_time:.1f} ms (hardcoded wall-clock)")
        else: 
            inf_time = measure_inference_time(name, model_fns[name], X_test)
                                              
        accuracies.append(acc)
        f1_scores.append(f1)
        inf_times.append(inf_time)

        print(f"\n{name}")
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  Macro F1:   {f1:.4f}")

        plot_confusion_matrix(y_test, y_pred, label_names, name)

    # summary plots
    plot_accuracy_comparison(accuracies)
    plot_f1_comparison(f1_scores)
    plot_inference_time(inf_times)

    print("\nAll figures saved to eval/figures/")


if __name__ == "__main__":
    main() 