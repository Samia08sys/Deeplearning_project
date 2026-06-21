"""
Part III — Data Preparation for NLP
Vocabulary building, tokenization, padding, and IMDB dataset loading.
"""

import os
import re
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── special tokens ────────────────────────────────────────────────────────────
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


# ─────────────────────────────────────────────────────────────────────────────
class Vocabulary:
    """Simple word-level vocabulary."""

    def __init__(self, min_freq: int = 2):
        self.min_freq  = min_freq
        self.word2idx  = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.idx2word  = {i: t for t, i in self.word2idx.items()}

    def build(self, texts: list):
        """Build vocabulary from a list of tokenized sentences (list of str)."""
        counter = Counter()
        for tokens in texts:
            counter.update(tokens)
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        print(f"[Vocab] Size: {len(self.word2idx)}")

    def encode(self, tokens: list, add_sos=False, add_eos=False) -> list:
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        if add_sos: ids = [SOS_IDX] + ids
        if add_eos: ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids: list) -> list:
        return [self.idx2word.get(i, "<unk>") for i in ids]

    def __len__(self):
        return len(self.word2idx)


def tokenize(text: str) -> list:
    """Simple whitespace + lowercase tokenizer."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic sentiment dataset (IMDB fallback)
# ─────────────────────────────────────────────────────────────────────────────
POS_TEMPLATES = [
    "this movie is absolutely amazing and wonderful",
    "i loved every single moment of this great film",
    "brilliant acting and beautiful story it was fantastic",
    "one of the best films i have ever seen outstanding",
    "wonderful experience heartwarming and deeply moving masterpiece",
    "the director did an incredible job with this gem",
    "spectacular visual effects combined with superb performances",
]
NEG_TEMPLATES = [
    "this movie was terrible and very boring waste of time",
    "awful acting and poor story i hated every minute",
    "worst film i have ever seen completely disappointing",
    "terrible dialogue and bad direction total disaster film",
    "nothing works in this movie it is just bad",
    "complete waste of money and time so boring",
    "the plot was incoherent and the actors mediocre here",
]


def generate_imdb_synthetic(n_samples: int = 1000):
    rng = random.Random(SEED)
    data = []
    for _ in range(n_samples // 2):
        t = rng.choice(POS_TEMPLATES)
        # add slight variation
        words = t.split()
        rng.shuffle(words[2:5])
        data.append((" ".join(words), 1))
    for _ in range(n_samples // 2):
        t = rng.choice(NEG_TEMPLATES)
        words = t.split()
        rng.shuffle(words[2:5])
        data.append((" ".join(words), 0))
    rng.shuffle(data)
    return data


def load_imdb(n_samples: int = 2000):
    """Download the real IMDB dataset using Hugging Face datasets. Fall back to synthetic data."""
    try:
        from datasets import load_dataset
        print("[Data] Téléchargement du dataset IMDB réel via HuggingFace (datasets)...")
        dataset = load_dataset("imdb", split="train")
        # dataset has ['text', 'label'] where label is 0 or 1
        # take a random sample
        dataset = dataset.shuffle(seed=SEED).select(range(n_samples))
        data = [(row["text"], row["label"]) for row in dataset]
        print(f"[Data] ✓ {len(data)} vrais avis IMDB chargés avec succès.")
    except ImportError:
        print("[Data] ⚠ Librairie 'datasets' non installée (pip install datasets). Utilisation données synthétiques.")
        data = generate_imdb_synthetic(n_samples)
    except Exception as e:
        print(f"[Data] ⚠ Erreur lors du chargement de IMDB via datasets ({e}). Utilisation données synthétiques.")
        data = generate_imdb_synthetic(n_samples)
    return data


# ─────────────────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, samples: list, vocab: Vocabulary, max_len: int = 64):
        self.samples = []
        for text, label in samples:
            ids = vocab.encode(tokenize(text))[:max_len]
            self.samples.append((ids, label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    seqs, labels = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded  = [s + [PAD_IDX] * (max_len - len(s)) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return (
        torch.tensor(padded,  dtype=torch.long),
        torch.tensor(labels,  dtype=torch.long),
        lengths,
    )


def prepare_sentiment_data(batch_size=32, max_len=64, n_samples=2000):
    """Load IMDB, build vocabulary, return loaders + vocab."""
    raw_data  = load_imdb(n_samples)
    split     = int(0.8 * len(raw_data))
    train_raw = raw_data[:split]
    test_raw  = raw_data[split:]

    # Build vocabulary from training set only
    vocab = Vocabulary(min_freq=1)
    vocab.build([tokenize(t) for t, _ in train_raw])

    train_ds = SentimentDataset(train_raw, vocab, max_len)
    test_ds  = SentimentDataset(test_raw,  vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"[Data] Train: {len(train_ds)}  Test: {len(test_ds)}  Vocab: {len(vocab)}")
    return train_loader, test_loader, vocab


if __name__ == "__main__":
    train_loader, test_loader, vocab = prepare_sentiment_data()
    X, y, lengths = next(iter(train_loader))
    print("Batch X:", X.shape, "y:", y.shape)
