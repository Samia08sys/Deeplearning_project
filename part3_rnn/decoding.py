"""
Part III — Greedy Decoding & Beam Search Decoding
"""

import torch
import heapq


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def greedy_decode(
    model,            # Seq2Seq model
    src: torch.Tensor,   # (1, T_src)
    max_len: int = 50,
    sos_idx: int = 1,
    eos_idx: int = 2,
    device=None,
) -> list:
    """Greedily decode one source sequence.

    At each step, choose the token with the highest probability (argmax).

    Returns
    -------
    tokens : list of int (decoded token indices, without SOS)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    src = src.to(device)

    _, hidden = model.encoder(src)
    token = torch.tensor([sos_idx], device=device)
    decoded = []

    for _ in range(max_len):
        logits, hidden = model.decoder(token, hidden)
        next_tok = logits.argmax(-1)                # (1,)
        idx = next_tok.item()
        if idx == eos_idx:
            break
        decoded.append(idx)
        token = next_tok

    return decoded


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def beam_search_decode(
    model,
    src: torch.Tensor,     # (1, T_src)
    beam_width: int = 3,
    max_len:    int = 50,
    sos_idx:    int = 1,
    eos_idx:    int = 2,
    device=None,
) -> list:
    """Beam search decoding.

    Maintains beam_width hypotheses simultaneously and returns the most
    probable complete sequence.

    A hypothesis is represented as:
      (neg_log_prob, token_list, hidden_state)

    Returns
    -------
    best_tokens : list of int
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    src = src.to(device)

    _, init_hidden = model.encoder(src)
    start_tok = torch.tensor([sos_idx], device=device)

    # Each entry: [score, sequence, hidden]
    beams = [(0.0, [sos_idx], init_hidden)]
    completed = []

    for step in range(max_len):
        candidates = []
        for score, seq, hidden in beams:
            last_tok = torch.tensor([seq[-1]], device=device)
            logits, new_hidden = model.decoder(last_tok, hidden)
            log_probs = torch.log_softmax(logits, dim=-1)[0]  # (V,)

            # Expand top beam_width tokens
            top_probs, top_idxs = log_probs.topk(beam_width)
            for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
                new_score = score - prob          # negative log-prob (min-heap)
                new_seq   = seq + [idx]
                candidates.append((new_score, new_seq, new_hidden))

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0])
        beams = []
        for cand in candidates[:beam_width]:
            if cand[1][-1] == eos_idx:
                completed.append(cand)
            else:
                beams.append(cand)

        if len(beams) == 0:
            break

    if not completed:
        completed = beams            # fallback if EOS never reached

    # Best = lowest neg-log-prob
    best = min(completed, key=lambda x: x[0])
    tokens = [t for t in best[1][1:] if t != eos_idx]  # strip SOS/EOS
    return tokens
