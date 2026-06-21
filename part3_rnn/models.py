"""
Part III — RNN, LSTM, GRU Models
Implementations both manual (from scratch) and using nn.LSTM/nn.GRU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Manual RNN Cell
# ─────────────────────────────────────────────────────────────────────────────
class ManualRNNCell(nn.Module):
    """Single vanilla RNN cell: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_xh = nn.Linear(input_size,  hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev))


class VanillaRNN(nn.Module):
    """Vanilla RNN for sequence classification / language modelling."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 n_layers: int = 1, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cells = nn.ModuleList()
        for i in range(n_layers):
            in_sz = embed_dim if i == 0 else hidden_size
            self.cells.append(ManualRNNCell(in_sz, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor, lengths=None):
        """
        x : (B, T) token indices
        Returns (B, n_classes) logits
        """
        B, T  = x.shape
        emb   = self.embedding(x)           # (B, T, E)
        h     = [torch.zeros(B, self.hidden_size, device=x.device)
                 for _ in range(self.n_layers)]

        for t in range(T):
            inp = emb[:, t, :]
            for l, cell in enumerate(self.cells):
                h[l] = cell(inp, h[l])
                inp   = self.dropout(h[l])

        return self.fc(h[-1])               # (B, n_classes)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Model (using nn.LSTM)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """LSTM-based classifier using torch nn.LSTM."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 n_layers: int = 2, n_classes: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_size   = hidden_size
        self.n_layers      = n_layers
        self.bidirectional = bidirectional
        self.D             = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * self.D, n_classes)

    def forward(self, x: torch.Tensor, lengths=None):
        emb = self.dropout(self.embedding(x))
        if lengths is not None:
            emb = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h_n, _) = self.lstm(emb)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Take last hidden layer (both directions if bidirectional)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        return self.fc(self.dropout(h_last))


# ─────────────────────────────────────────────────────────────────────────────
# GRU Model (using nn.GRU)
# ─────────────────────────────────────────────────────────────────────────────
class GRUClassifier(nn.Module):
    """GRU-based classifier using torch nn.GRU."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 n_layers: int = 2, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor, lengths=None):
        emb = self.dropout(self.embedding(x))
        if lengths is not None:
            emb = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(emb)
        h_last = h_n[-1]                    # last layer hidden state
        return self.fc(self.dropout(h_last))


# ─────────────────────────────────────────────────────────────────────────────
# Language Model variant (for next-token prediction / perplexity)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMLanguageModel(nn.Module):
    """LSTM language model: projects hidden state to vocab logits at each step."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(self.dropout(out))
        return logits, hidden

    def init_hidden(self, batch_size: int, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return h, c


if __name__ == "__main__":
    V, E, H = 5000, 128, 256
    x = torch.randint(0, V, (4, 50))
    rnn  = VanillaRNN(V, E, H, n_classes=2)
    lstm = LSTMClassifier(V, E, H, n_classes=2)
    gru  = GRUClassifier(V, E, H, n_classes=2)
    print("VanillaRNN  output:", rnn(x).shape)
    print("LSTM        output:", lstm(x).shape)
    print("GRU         output:", gru(x).shape)
