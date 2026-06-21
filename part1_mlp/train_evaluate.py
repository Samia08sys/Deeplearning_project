"""
Part I — Training Loop & Evaluation
Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part1")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)
        all_preds .extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / total
    acc      = correct / total
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


def train(
    model,
    loaders: dict,
    n_epochs:    int   = 50,
    lr:          float = 1e-3,
    weight_decay:float = 1e-4,
    label:       str   = "model",
):
    """Full training loop with validation tracking.

    Returns
    -------
    history : dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  Training: {label}  |  Device: {DEVICE}  |  Epochs: {n_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, DEVICE)
        vl_loss, vl_acc, _, _ = evaluate(model, loaders["val"], criterion, DEVICE)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"] .append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
                f"({elapsed:.1f}s)"
            )

    print(f"  Best val accuracy: {best_val_acc:.4f}\n")
    return history


def compute_metrics(y_true, y_pred, class_names=None, label=""):
    """Print and return classification metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'─'*50}")
    print(f"  Metrics for: {label}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(cm)
    print(f"{'─'*50}\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(histories: dict, save_path=None):
    """Plot loss and accuracy curves for multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for idx, (lbl, h) in enumerate(histories.items()):
        c = colors[idx % len(colors)]
        axes[0].plot(h["train_loss"], linestyle="--", color=c, alpha=0.6, label=f"{lbl} train")
        axes[0].plot(h["val_loss"],   linestyle="-",  color=c,            label=f"{lbl} val")
        axes[1].plot(h["train_acc"],  linestyle="--", color=c, alpha=0.6, label=f"{lbl} train")
        axes[1].plot(h["val_acc"],    linestyle="-",  color=c,            label=f"{lbl} val")

    axes[0].set_title("Loss Curves");  axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Accuracy Curves"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(OUT_DIR, "learning_curves.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[Plot] Learning curves saved → {path}")


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Plot and SAVE the confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    path = save_path or os.path.join(OUT_DIR, "confusion_matrix.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()

    # ── CONFIRMED: Confusion matrix printed to console ────────────────────────
    print(f"\n[✓] Confusion Matrix ({title}):")
    header = "        " + "  ".join(f"{c:>6}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:>6}  " + "  ".join(f"{v:>6}" for v in row)
        print(row_str)
    print(f"[Plot] Confusion matrix saved → {path}\n")
