"""
Shared training/evaluation loop. Used by run_comparison.py to train the
GRU and the Transformer under identical conditions (data, optimizer,
epochs, batch size) so the comparison is fair.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, lengths, labels in loader:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits).detach().cpu()
            all_probs.extend(probs.tolist())
            all_preds.extend((probs > 0.5).long().tolist())
            all_labels.extend(labels.cpu().long().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs),
    }
    return metrics, all_labels, all_preds


def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3,
                 name="model"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "epoch_time": []}
    best_val_f1, best_state = -1, None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics, _, _ = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        epoch_time = time.time() - t0

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["epoch_time"].append(epoch_time)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{name}] epoch {epoch}/{epochs} "
              f"train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1={val_metrics['f1']:.4f} "
              f"({epoch_time:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def measure_inference_latency(model, loader, device, n_batches=20):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (x, lengths, labels) in enumerate(loader):
            if i >= n_batches:
                break
            x, lengths = x.to(device), lengths.to(device)
            t0 = time.time()
            _ = model(x, lengths)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) / x.size(0))
    return sum(times) / len(times) * 1000  # ms per example
