"""
Part III — Seq2Seq Encoder-Decoder (without attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """LSTM Encoder: encodes source sequence into context vector."""

    def __init__(self, src_vocab_size: int, embed_dim: int,
                 hidden_size: int, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor):
        """
        src : (B, T_src)
        Returns:
          outputs : (B, T_src, H)
          hidden  : (h_n, c_n)
        """
        emb = self.dropout(self.embedding(src))
        outputs, hidden = self.lstm(emb)
        return outputs, hidden


class Decoder(nn.Module):
    """LSTM Decoder: generates target sequence one token at a time."""

    def __init__(self, tgt_vocab_size: int, embed_dim: int,
                 hidden_size: int, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token: torch.Tensor, hidden):
        """
        token  : (B,) — current input token
        hidden : (h, c) from encoder or previous decoder step
        Returns:
          logits : (B, tgt_vocab_size)
          hidden : updated (h, c)
        """
        emb = self.dropout(self.embedding(token.unsqueeze(1)))   # (B, 1, E)
        out, hidden = self.lstm(emb, hidden)                     # (B, 1, H)
        logits = self.fc(out.squeeze(1))                         # (B, V)
        return logits, hidden


class Seq2Seq(nn.Module):
    """Seq2Seq wrapper combining Encoder and Decoder."""

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 sos_idx: int, eos_idx: int, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5):
        """
        src : (B, T_src)  source token indices
        tgt : (B, T_tgt)  target token indices (with SOS at pos 0)
        Returns:
          outputs : (B, T_tgt-1, vocab_size) — one logit per step
        """
        B, T_tgt = tgt.shape
        tgt_vocab = self.decoder.fc.out_features

        _, hidden = self.encoder(src)
        input_tok = tgt[:, 0]                          # SOS tokens
        outputs   = []

        for t in range(1, T_tgt):
            logits, hidden = self.decoder(input_tok, hidden)
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing
            import random
            if random.random() < teacher_forcing_ratio:
                input_tok = tgt[:, t]
            else:
                input_tok = logits.argmax(-1)

        return torch.cat(outputs, dim=1)   # (B, T_tgt-1, vocab)


if __name__ == "__main__":
    SRC_V, TGT_V, E, H, L = 100, 80, 32, 64, 1
    enc = Encoder(SRC_V, E, H, n_layers=L)
    dec = Decoder(TGT_V, E, H, n_layers=L)
    s2s = Seq2Seq(enc, dec, sos_idx=1, eos_idx=2)
    src = torch.randint(3, SRC_V, (4, 10))
    tgt = torch.randint(3, TGT_V, (4, 8))
    out = s2s(src, tgt)
    print("Seq2Seq output:", out.shape)   # (4, 7, 80)
