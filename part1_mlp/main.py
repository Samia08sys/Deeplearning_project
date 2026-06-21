"""
Part I — Main Orchestrator
Runs the full MLP pipeline on the Star Classification dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from data_preparation import load_and_preprocess
from mlp_sequential   import build_mlp_sequential, count_parameters
from mlp_custom       import MLPCustom
from weight_init      import apply_init, STRATEGIES
from train_evaluate   import (
    train, evaluate, compute_metrics, plot_history, plot_confusion_matrix, DEVICE
)
import torch.nn as nn

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part1")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS     = 60
LR         = 1e-3
BATCH_SIZE = 32

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  PART I — MLP on Star Classification")
    print("="*60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    loaders, meta = load_and_preprocess(batch_size=BATCH_SIZE)
    n_feat   = meta["n_features"]
    n_cls    = meta["n_classes"]
    cls_names = meta["class_names"]

    print(f"\n[Meta] Features: {n_feat}  Classes: {n_cls}  → {cls_names}")

    criterion = nn.CrossEntropyLoss()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION A: Sequential vs Custom MLP
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  SECTION A: Sequential vs Custom MLP (Xavier init)")
    print("─"*60)

    seq_model  = apply_init(build_mlp_sequential(n_feat, n_cls), "Xavier")
    cust_model = apply_init(MLPCustom(n_feat, n_cls), "Xavier")

    print(f"  Sequential params  : {count_parameters(seq_model):,}")
    print(f"  Custom params      : {cust_model.count_parameters():,}")
    cust_model.print_state_dict_summary()

    hist_seq  = train(seq_model,  loaders, n_epochs=EPOCHS, lr=LR, label="Sequential")
    hist_cust = train(cust_model, loaders, n_epochs=EPOCHS, lr=LR, label="Custom")

    plot_history(
        {"Sequential": hist_seq, "Custom": hist_cust},
        save_path=os.path.join(OUT_DIR, "curves_seq_vs_custom.png"),
    )

    # Evaluate on test set
    for lbl, mdl in [("Sequential", seq_model), ("Custom", cust_model)]:
        _, _, preds, labels = evaluate(mdl, loaders["test"], criterion, DEVICE)
        m = compute_metrics(labels, preds, cls_names, label=lbl)
        plot_confusion_matrix(
            m["cm"], cls_names,
            title=f"Confusion Matrix — {lbl}",
            save_path=os.path.join(OUT_DIR, f"cm_{lbl.lower()}.png"),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION B: Weight Initialization Comparison
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("  SECTION B: Weight Initialization Comparison")
    print("─"*60)

    base_model = MLPCustom(n_feat, n_cls)
    init_histories = {}
    init_metrics   = {}

    for strategy_name in STRATEGIES:
        mdl = apply_init(base_model, strategy_name)
        h   = train(mdl, loaders, n_epochs=EPOCHS, lr=LR, label=strategy_name)
        init_histories[strategy_name] = h

        _, _, preds, labels = evaluate(mdl, loaders["test"], criterion, DEVICE)
        m = compute_metrics(labels, preds, cls_names, label=strategy_name)
        init_metrics[strategy_name] = m

        # Per-strategy confusion matrix
        plot_confusion_matrix(
            m["cm"], cls_names,
            title=f"Confusion Matrix — {strategy_name} Init",
            save_path=os.path.join(OUT_DIR, f"cm_init_{strategy_name.lower()}.png"),
        )

    plot_history(
        init_histories,
        save_path=os.path.join(OUT_DIR, "curves_weight_init.png"),
    )

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY TABLE — Initialization Strategies (Test Set)")
    print("="*60)
    print(f"  {'Strategy':12s}  {'Accuracy':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for s, m in init_metrics.items():
        print(
            f"  {s:12s}  {m['accuracy']:>10.4f}  {m['precision']:>10.4f}"
            f"  {m['recall']:>10.4f}  {m['f1']:>10.4f}"
        )
    print("="*60)

    print("\n[✓] Part I complete. Outputs saved to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
