"""
Part III — Training & Evaluation
BPTT, gradient clipping, Perplexity, BLEU score,
and RNN vs LSTM vs GRU learning curve comparison.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part3")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment classification training
# ─────────────────────────────────────────────────────────────────────────────
def train_classifier(
    model, train_loader, val_loader,
    n_epochs=10, lr=1e-3, clip=1.0, label="model"
) -> dict:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"  Training classifier: {label}  |  Epochs: {n_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X, y, lengths in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X, lengths)
            loss = criterion(out, y)
            loss.backward()
            # ── Gradient Clipping (BPTT mitigation) ───────────────────────────
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            tr_loss    += loss.item() * len(y)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total   += len(y)

        # ── val ───────────────────────────────────────────────────────────────
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y, lengths in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out  = model(X, lengths)
                loss = criterion(out, y)
                vl_loss    += loss.item() * len(y)
                vl_correct += (out.argmax(1) == y).sum().item()
                vl_total   += len(y)

        tr_acc = tr_correct / tr_total
        vl_acc = vl_correct / vl_total
        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"]  .append(vl_loss / vl_total)
        history["train_acc"] .append(tr_acc)
        history["val_acc"]   .append(vl_acc)

        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{n_epochs}  "
                  f"train_loss={tr_loss/tr_total:.4f}  train_acc={tr_acc:.4f}  "
                  f"val_acc={vl_acc:.4f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Language model training
# ─────────────────────────────────────────────────────────────────────────────
def train_language_model(
    model, train_loader, val_loader,
    n_epochs=1, lr=1e-3, clip=1.0, label="language_model", pad_idx=0
) -> dict:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    history = {"train_loss": [], "val_loss": []}

    print(f"\n{'='*60}")
    print(f"  Training language model: {label}  |  Epochs: {n_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss, tr_tokens = 0.0, 0

        for X, y, lengths in train_loader:
            X = X.to(DEVICE)
            optimizer.zero_grad()

            inp = X[:, :-1]
            tgt = X[:, 1:]
            logits, _ = model(inp, None)

            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            n_tok = (tgt != pad_idx).sum().item()
            tr_loss += loss.item() * n_tok
            tr_tokens += n_tok

        model.eval()
        vl_loss, vl_tokens = 0.0, 0
        with torch.no_grad():
            for X, y, lengths in val_loader:
                X = X.to(DEVICE)
                inp = X[:, :-1]
                tgt = X[:, 1:]
                logits, _ = model(inp, None)

                B, T, V = logits.shape
                loss = criterion(logits.reshape(B * T, V), tgt.reshape(-1))
                n_tok = (tgt != pad_idx).sum().item()
                vl_loss += loss.item() * n_tok
                vl_tokens += n_tok

        history["train_loss"].append(tr_loss / max(tr_tokens, 1))
        history["val_loss"].append(vl_loss / max(vl_tokens, 1))

        print(
            f"  Epoch {epoch:2d}/{n_epochs}  "
            f"train_loss={history['train_loss'][-1]:.4f}  "
            f"val_loss={history['val_loss'][-1]:.4f}"
        )

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity evaluation
# ─────────────────────────────────────────────────────────────────────────────
def compute_perplexity(model, loader, pad_idx=0) -> float:
    """Compute perplexity of an LSTMLanguageModel on a data loader.

    Perplexity = exp(mean cross-entropy loss over all non-pad tokens)
    """
    model.eval().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for X, y, lengths in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            # For LM: input=X[:, :-1], target=X[:, 1:]
            inp = X[:, :-1]
            tgt = X[:, 1:]
            logits, _ = model(inp, None)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), tgt.reshape(-1))
            n_tok = (tgt != pad_idx).sum().item()
            total_loss   += loss.item()
            total_tokens += n_tok

    perplexity = math.exp(total_loss / max(total_tokens, 1))
    return perplexity


# ─────────────────────────────────────────────────────────────────────────────
# BLEU score
# ─────────────────────────────────────────────────────────────────────────────
def compute_bleu(references: list, hypotheses: list) -> float:
    """Compute corpus-level BLEU score.

    Parameters
    ----------
    references  : list of list of str (token sequences)
    hypotheses  : list of list of str

    Returns
    -------
    bleu : float (0-100)
    """
    try:
        import sacrebleu
        hyp_str  = [" ".join(h) for h in hypotheses]
        ref_strs = [[" ".join(r) for r in references]]
        bleu = sacrebleu.corpus_bleu(hyp_str, ref_strs)
        return bleu.score
    except ImportError:
        # fallback: rough n-gram precision
        from collections import Counter

        def ngrams(seq, n):
            return Counter(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))

        total_match, total_count = 0, 0
        for ref, hyp in zip(references, hypotheses):
            for n in range(1, 5):
                ref_ng = ngrams(ref, n)
                hyp_ng = ngrams(hyp, n)
                for gram, cnt in hyp_ng.items():
                    total_match += min(cnt, ref_ng.get(gram, 0))
                    total_count += cnt
        if total_count == 0:
            return 0.0
        return 100.0 * total_match / total_count


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(histories: dict, save_path: str = None):
    """Plot loss & accuracy comparison for RNN vs LSTM vs GRU."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    for idx, (lbl, h) in enumerate(histories.items()):
        c = colors[idx % len(colors)]
        axes[0].plot(h["train_loss"], "--", color=c, alpha=0.5, label=f"{lbl} train")
        axes[0].plot(h["val_loss"],   "-",  color=c,            label=f"{lbl} val")
        axes[1].plot(h["train_acc"],  "--", color=c, alpha=0.5, label=f"{lbl} train")
        axes[1].plot(h["val_acc"],    "-",  color=c,            label=f"{lbl} val")

    for ax, t in zip(axes, ["Loss Curves", "Accuracy Curves"]):
        ax.set_title(t); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(OUT_DIR, "rnn_comparison.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"[Plot] RNN/LSTM/GRU comparison curves → {path}")
