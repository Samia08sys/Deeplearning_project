"""
Part II — CNN Main Orchestrator
Full pipeline: manual ops demo → LeNet training on Fashion-MNIST → experiments → feature map visualization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manual_ops import manual_conv2d, manual_maxpool2d, manual_avgpool2d, output_size
from lenet      import LeNet5, ImprovedLeNet
from experiments import run_experiments
from visualize  import visualize_sample

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part2")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS     = 8
BATCH_SIZE = 64
DATASET    = "FashionMNIST"   # or "MNIST"


# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(name: str, batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    data_root = os.path.join(os.path.dirname(__file__), "..", "outputs", "data")

    Dataset = getattr(torchvision.datasets, name)
    train_ds = Dataset(data_root, train=True,  download=True, transform=transform)
    test_ds  = Dataset(data_root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # small val split from test
    val_size  = 2000
    val_ds, _ = torch.utils.data.random_split(test_ds, [val_size, len(test_ds) - val_size])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[Data] {name}  train={len(train_ds)}  val={val_size}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
def train_cnn(model, train_loader, val_loader, n_epochs: int, label: str):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward(); optimizer.step()
            tr_loss   += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total   += len(y)

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                loss   = criterion(logits, y)
                vl_loss    += loss.item() * len(y)
                vl_correct += (logits.argmax(1) == y).sum().item()
                vl_total   += len(y)

        tr_acc = tr_correct / tr_total
        vl_acc = vl_correct / vl_total
        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"]  .append(vl_loss / vl_total)
        history["train_acc"] .append(tr_acc)
        history["val_acc"]   .append(vl_acc)

        print(f"  [{label}] Epoch {epoch:2d}/{n_epochs}  "
              f"train_loss={tr_loss/tr_total:.4f}  train_acc={tr_acc:.4f}  "
              f"val_acc={vl_acc:.4f}")

    return history


def plot_cnn_curves(histories: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10.colors
    for idx, (lbl, h) in enumerate(histories.items()):
        c = colors[idx % len(colors)]
        axes[0].plot(h["train_loss"], "--", color=c, alpha=0.6, label=f"{lbl} train")
        axes[0].plot(h["val_loss"],   "-",  color=c,            label=f"{lbl} val")
        axes[1].plot(h["train_acc"],  "--", color=c, alpha=0.6, label=f"{lbl} train")
        axes[1].plot(h["val_acc"],    "-",  color=c,            label=f"{lbl} val")
    for ax, title in zip(axes, ["Loss", "Accuracy"]):
        ax.set_title(f"CNN {title} Curves"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()
    print(f"[Plot] CNN learning curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print(f"  PART II — CNN on {DATASET}")
    print("="*60)

    # ── SECTION A: Manual Operations Demo ────────────────────────────────────
    print("\n─── SECTION A: Manual Convolution & Pooling Demo ───")
    rng = np.random.RandomState(0)
    X_demo = rng.randn(1, 8, 8)
    K_demo = rng.randn(1, 3, 3)

    for padding, stride in [(0,1), (1,1), (0,2)]:
        Y = manual_conv2d(X_demo[0], K_demo[0], padding=padding, stride=stride)
        exp = output_size(8, 3, padding, stride)
        print(f"  conv2d  p={padding} s={stride}  "
              f"→ output {Y.shape}  [formula: {exp}×{exp}]")

    Ymax = manual_maxpool2d(X_demo, kernel_size=2, stride=2)
    Yavg = manual_avgpool2d(X_demo, kernel_size=2, stride=2)
    print(f"  maxpool2d k=2 s=2 → {Ymax.shape}")
    print(f"  avgpool2d k=2 s=2 → {Yavg.shape}")

    # ── SECTION B: Load Dataset ───────────────────────────────────────────────
    print(f"\n─── SECTION B: Loading {DATASET} ───")
    train_loader, val_loader, test_loader = load_dataset(DATASET, BATCH_SIZE)

    # ── SECTION C: Train LeNet-5 and Improved LeNet ───────────────────────────
    print("\n─── SECTION C: Training LeNet-5 vs Improved LeNet ───")
    lenet    = LeNet5(n_classes=10)
    improved = ImprovedLeNet(n_classes=10)
    print(f"  LeNet-5 params    : {lenet.count_parameters():,}")
    print(f"  ImprovedLeNet params: {improved.count_parameters():,}")

    h_lenet    = train_cnn(lenet,    train_loader, val_loader, EPOCHS, "LeNet-5")
    h_improved = train_cnn(improved, train_loader, val_loader, EPOCHS, "Improved")
    plot_cnn_curves(
        {"LeNet-5": h_lenet, "Improved": h_improved},
        save_path=os.path.join(OUT_DIR, "cnn_curves.png"),
    )

    # ── SECTION D: Hyperparameter Experiments ─────────────────────────────────
    print("\n─── SECTION D: Filter/Padding/Stride Experiments ───")
    run_experiments(train_loader, val_loader, n_classes=10, n_epochs=3)

    # ── SECTION E: Feature Map Visualization ──────────────────────────────────
    print("\n─── SECTION E: Feature Map Visualization ───")
    sample_imgs, _ = next(iter(test_loader))
    sample_img = sample_imgs[0]  # single image (C, H, W)
    paths = visualize_sample(improved, sample_img, max_channels=16)
    print(f"[✓] {len(paths)} feature map image(s) rendered and saved.")

    print("\n[✓] Part II complete. Outputs saved to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
