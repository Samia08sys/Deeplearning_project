"""
Part II — CNN Experiments
Grid search over padding, stride, and number of filters.
Records train/val accuracy and prints a comparison table.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part2")


# ── Tiny configurable CNN for experiments ────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, n_classes=10, n_channels=1, img_size=28,
                 n_filters=16, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_filters,
                              kernel_size=3, padding=padding, stride=stride)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat = n_filters * 4 * 4
        self.fc  = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


def train_epoch(model, loader, opt, crit):
    model.train()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        correct += (model(X).argmax(1) == y).sum().item()
        total   += len(y)
    return correct / total


@torch.no_grad()
def val_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        correct += (model(X).argmax(1) == y).sum().item()
        total   += len(y)
    return correct / total


def run_experiments(train_loader, val_loader,
                    n_classes=10, n_epochs=5):
    """Grid search over n_filters, padding, stride.

    Returns
    -------
    results : list of dicts
    """
    configs = [
        {"n_filters": 8,  "padding": 0, "stride": 1},
        {"n_filters": 16, "padding": 0, "stride": 1},
        {"n_filters": 32, "padding": 0, "stride": 1},
        {"n_filters": 16, "padding": 1, "stride": 1},
        {"n_filters": 16, "padding": 2, "stride": 1},
        {"n_filters": 16, "padding": 0, "stride": 2},
        {"n_filters": 16, "padding": 1, "stride": 2},
    ]

    results = []
    for cfg in configs:
        model = TinyCNN(n_classes=n_classes, **cfg).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        crit  = nn.CrossEntropyLoss()
        for _ in range(n_epochs):
            tr_acc = train_epoch(model, train_loader, opt, crit)
        vl_acc = val_accuracy(model, val_loader)
        row = {**cfg, "train_acc": round(tr_acc, 4), "val_acc": round(vl_acc, 4)}
        results.append(row)
        print(f"  filters={cfg['n_filters']:2d}  padding={cfg['padding']}  "
              f"stride={cfg['stride']}  → val_acc={vl_acc:.4f}")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  CNN EXPERIMENT TABLE")
    print("="*65)
    print(f"  {'Filters':>8}  {'Padding':>8}  {'Stride':>7}  {'Train Acc':>10}  {'Val Acc':>9}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*9}")
    for r in results:
        print(f"  {r['n_filters']:>8}  {r['padding']:>8}  {r['stride']:>7}"
              f"  {r['train_acc']:>10.4f}  {r['val_acc']:>9.4f}")
    print("="*65 + "\n")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    labels    = [f"f{r['n_filters']},p{r['padding']},s{r['stride']}" for r in results]
    val_accs  = [r["val_acc"]  for r in results]
    train_accs = [r["train_acc"] for r in results]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - 0.2 for i in x], train_accs, width=0.4, label="Train", alpha=0.8)
    ax.bar([i + 0.2 for i in x], val_accs,   width=0.4, label="Val",   alpha=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_title("CNN Experiment: Filters / Padding / Stride")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cnn_experiments.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[Plot] Experiment chart saved → {path}")

    return results
