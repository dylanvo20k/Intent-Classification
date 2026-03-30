"""
Apple Foundation Models SDK baseline (macOS 26+, Xcode 26+, Apple Intelligence required)
Calls a Swift subprocess to run on-device inference, then maps predictions back to label indices.
"""
import os
import json
import subprocess
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SWIFT_SCRIPT = os.path.join(ROOT, "apple_sdk_runner.swift")
RESULTS_DIR  = os.path.join(ROOT, "eval", "results")


def predict(utterances: list[str], le: LabelEncoder) -> np.ndarray:
    """
    Run Apple SDK inference on utterances.
    Returns np.ndarray of integer class indices (same format as other models).
    """
    input_json  = os.path.join(RESULTS_DIR, "apple_sdk_input.json")
    output_json = os.path.join(RESULTS_DIR, "apple_sdk_output.json")

    # Write utterances to JSON for Swift to read
    with open(input_json, "w") as f:
        json.dump(utterances, f)

    print("  Running Swift/Foundation Models inference (this may take a while)...")
    result = subprocess.run(
        ["swift", SWIFT_SCRIPT, input_json, output_json],
        capture_output=False,  # let progress print to terminal
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Swift script failed. Make sure you have macOS 26+, "
                           f"Xcode 26+, and Apple Intelligence enabled.")

    with open(output_json, "r") as f:
        raw_preds = json.load(f)  

    # Map string labels -> integer indices, handle unknowns
    known = set(le.classes_)
    cleaned = [p if p in known else le.classes_[0] for p in raw_preds]
    return le.transform(cleaned)