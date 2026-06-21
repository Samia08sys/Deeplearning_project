"""
Part I — MLP via custom nn.Module
Demonstrates parameters(), state_dict(), and named_modules() usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPCustom(nn.Module):
    """Multi-Layer Perceptron implemented as a custom nn.Module subclass.

    This version explicitly demonstrates:
    - Manual definition of layers as attributes
    - Forward method
    - state_dict() and parameters() access
    """

    def __init__(
        self,
        n_features: int,
        n_classes:  int,
        hidden_layers: list = None,
        activation: str = "relu",
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64]

        self.activation_name = activation
        self.dropout_p = dropout

        # ── build layers programmatically ────────────────────────────────────
        dims = [n_features] + hidden_layers + [n_classes]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(h) for h in hidden_layers]
        )
        self.dropout = nn.Dropout(dropout)
        self.n_hidden = len(hidden_layers)

    # ── activation helper ─────────────────────────────────────────────────────
    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(x)
        elif self.activation_name == "tanh":
            return torch.tanh(x)
        elif self.activation_name == "leaky_relu":
            return F.leaky_relu(x)
        return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_hidden):
            x = self.linears[i](x)
            x = self.batchnorms[i](x)
            x = self._activate(x)
            x = self.dropout(x)
        x = self.linears[-1](x)  # output layer — no activation
        return x

    # ── introspection helpers ─────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_state_dict_summary(self):
        print("\n[MLPCustom] State-dict keys and shapes:")
        for k, v in self.state_dict().items():
            print(f"  {k:45s}  {tuple(v.shape)}")


if __name__ == "__main__":
    model = MLPCustom(n_features=6, n_classes=6)
    print(model)
    print(f"\nTrainable parameters: {model.count_parameters():,}")
    model.print_state_dict_summary()
    x = torch.randn(8, 6)
    out = model(x)
    print("Output shape:", out.shape)
