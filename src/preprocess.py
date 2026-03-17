"""
Text cleaning, TF-IDF, tokenization 
"""
import re
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

LOCALE = "en-US"
MAX_TFIDF_FEATURES = 10_000
MAX_SEQ_LEN = 32  #

# Load data 
def load_raw_data(locale: str = LOCALE):
    dataset = load_dataset("AmazonScience/massive", LOCALE)
    return dataset["train"], dataset["validation"], dataset["test"]

# Cleaning function 
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# TF-IDF
def get_tfidf_data(locale: str = LOCALE):
    train_ds, val_ds, test_ds = load_raw_data(locale)

    X_train_raw = [clean_text(x) for x in train_ds["utt"]]
    X_val_raw   = [clean_text(x) for x in val_ds["utt"]]
    X_test_raw  = [clean_text(x) for x in test_ds["utt"]]

    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val   = vectorizer.transform(X_val_raw)
    X_test  = vectorizer.transform(X_test_raw)

    le = LabelEncoder()
    y_train = le.fit_transform(train_ds["intent"])
    y_val   = le.transform(val_ds["intent"])
    y_test  = le.transform(test_ds["intent"])

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, le



