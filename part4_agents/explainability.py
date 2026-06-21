"""
Very small explainability helpers.

This module keeps a SHAP/LIME-like spirit without making the project complex.
If you later want the real libraries, you can replace these helpers with SHAP/LIME.
"""

from typing import List, Sequence, Tuple


def top_items(names: Sequence[str], values: Sequence[float], top_k: int = 5) -> List[Tuple[str, float]]:
    pairs = list(zip(names, values))
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return pairs[:top_k]


def explain_tabular(features: Sequence[str], values: Sequence[float], top_k: int = 5) -> str:
    items = top_items(features, values, top_k=top_k)
    formatted = ", ".join(f"{name} ({value:+.3f})" for name, value in items)
    return f"Variables les plus influentes: {formatted}."


def explain_image(channels: Sequence[str], scores: Sequence[float], top_k: int = 4) -> str:
    items = top_items(channels, scores, top_k=top_k)
    formatted = ", ".join(f"{name} ({value:+.3f})" for name, value in items)
    return f"Régions/canaux les plus actifs: {formatted}."


def explain_text(tokens: Sequence[str], scores: Sequence[float], top_k: int = 6) -> str:
    items = top_items(tokens, scores, top_k=top_k)
    formatted = ", ".join(f"{token} ({value:+.3f})" for token, value in items)
    return f"Mots les plus importants: {formatted}."
