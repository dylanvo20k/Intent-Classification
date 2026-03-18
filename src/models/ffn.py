"""
Feedforward Neural Network for intent classification
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_CLASSES = 60


class IntentFFN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def _to_tensor_dataset(X, y):
    # X is a sparse scipy matrix from TF-IDF, convert to dense tensor
    X_dense = torch.tensor(X.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_dense, y_tensor)


def train(X_train, y_train, X_val, y_val) -> IntentFFN:
    train_loader = DataLoader(
        _to_tensor_dataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        _to_tensor_dataset(X_val, y_val),
        batch_size=BATCH_SIZE
    )

    input_dim = X_train.shape[1]
    model = IntentFFN(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        # training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}")

    return model


def predict(model: IntentFFN, X) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        _to_tensor_dataset(X, np.zeros(X.shape[0], dtype=int)),
        batch_size=BATCH_SIZE
    )
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds)