"""
Part II — Feature Map Visualization
Uses forward hooks to capture and display intermediate CNN activations.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "part2")
os.makedirs(OUT_DIR, exist_ok=True)


class FeatureMapExtractor:
    """Register forward hooks on named Conv2d layers to capture activations."""

    def __init__(self, model: torch.nn.Module):
        self.model        = model
        self.activations  = {}
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        import torch.nn as nn
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def extract(self, x: torch.Tensor) -> dict:
        """Run a forward pass and return captured activations."""
        self.activations.clear()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            self.model(x.to(device))
        return dict(self.activations)


def plot_feature_maps(activations: dict, max_channels: int = 16, save_prefix: str = "featuremap"):
    """Plot and save feature maps for each captured conv layer.

    Parameters
    ----------
    activations  : dict {layer_name: tensor (B, C, H, W)}
    max_channels : maximum number of channels to display per layer
    save_prefix  : filename prefix for saved images
    """
    saved_paths = []
    for layer_name, act in activations.items():
        # take first sample in batch
        fmaps = act[0]  # (C, H, W)
        n_show = min(fmaps.shape[0], max_channels)

        cols = 8
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
        axes = np.array(axes).flatten()

        for i in range(n_show):
            fm = fmaps[i].numpy()
            axes[i].imshow(fm, cmap="viridis", aspect="auto")
            axes[i].set_title(f"ch {i}", fontsize=7)
            axes[i].axis("off")
        for j in range(n_show, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Feature Maps — {layer_name}", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        safe_name = layer_name.replace(".", "_")
        path = os.path.join(OUT_DIR, f"{save_prefix}_{safe_name}.png")
        plt.savefig(path, dpi=110)
        plt.close()
        saved_paths.append(path)
        # ── CONFIRMED: Feature maps rendered and saved ────────────────────────
        print(f"[✓] Feature maps rendered → {path}  (layer: {layer_name}, channels shown: {n_show})")

    return saved_paths


def visualize_sample(model, sample_img: torch.Tensor, max_channels: int = 16):
    """High-level helper: extract + plot feature maps for one sample image."""
    extractor = FeatureMapExtractor(model)
    activations = extractor.extract(sample_img.unsqueeze(0))
    extractor.remove_hooks()
    paths = plot_feature_maps(activations, max_channels=max_channels)
    return paths
