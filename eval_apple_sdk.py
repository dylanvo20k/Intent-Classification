"""
Apple SDK evaluation script.
Run this separately — requires macOS 26+, Xcode 26+, Apple Intelligence.
Saves preds_apple_sdk.pkl to eval/results/ for use in evaluate.py.
"""
import os
import pickle
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# add src/ to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import get_tfidf_data
from models import apple_sdk

ROOT        = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "eval", "results")


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    print("Loading data...")
    _, _, _, _, _, y_test, _, le = get_tfidf_data()
    test_utts = load(os.path.join(RESULTS_DIR, "test_utts.pkl"))

    print(f"Running Apple SDK on {len(test_utts)} test utterances...")
    start = time.perf_counter()
    preds = apple_sdk.predict(test_utts, le)
    elapsed_ms = (time.perf_counter() - start) * 1000

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="macro", zero_division=0)

    print(f"\nApple SDK Results")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Macro F1:       {f1:.4f}")
    print(f"  Total time:     {elapsed_ms:.1f} ms ({elapsed_ms/len(test_utts):.1f} ms/sample)")

    save(preds, os.path.join(RESULTS_DIR, "preds_apple_sdk.pkl"))
    print(f"\nSaved to eval/results/preds_apple_sdk.pkl")
    print("You can now add 'Apple SDK' to MODELS in evaluate.py to include it in plots.")


if __name__ == "__main__":
    main()