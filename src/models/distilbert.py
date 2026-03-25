"""
DistilBERT fine-tuned for intent classification
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 2e-5
NUM_CLASSES = 60


class IntentDataset(Dataset):
    def __init__(self, utterances, labels, tokenizer):
        self.encodings = tokenizer(
            utterances,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }


def train(train_utterances, y_train, val_utterances, y_val):
    print(f"  Using device: {DEVICE}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(
        IntentDataset(train_utterances, y_train, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        IntentDataset(val_utterances, y_val, tokenizer),
        batch_size=BATCH_SIZE
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(EPOCHS):
        # training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)
                preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = train_loss / len(train_loader)
        val_acc = correct / total
        print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}")

    return model, tokenizer


def predict(model, tokenizer, utterances) -> np.ndarray:
    model.eval()
    dataset = IntentDataset(utterances, [0] * len(utterances), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)