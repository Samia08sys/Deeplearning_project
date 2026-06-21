"""
Part I — MLP via nn.Sequential
"""

import torch
import torch.nn as nn


def build_mlp_sequential(
    n_features: int,
    n_classes:  int,
    hidden_layers: list = None,
    activation: str = "relu",
    dropout: float = 0.3,
) -> nn.Sequential:
    """Build a configurable MLP using nn.Sequential.

    Parameters
    ----------
    n_features    : number of input features
    n_classes     : number of output classes
    hidden_layers : list of hidden layer sizes, e.g. [128, 64]
    activation    : 'relu' | 'tanh' | 'leaky_relu'
    dropout       : dropout probability (0 = no dropout)
    """
    if hidden_layers is None:
        hidden_layers = [128, 64]

    act_map = {
        "relu":       nn.ReLU,
        "tanh":       nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    act_cls = act_map.get(activation, nn.ReLU)

    layers = []
    in_dim = n_features
    for h in hidden_layers:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.BatchNorm1d(h))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, n_classes))

    model = nn.Sequential(*layers)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_mlp_sequential(n_features=6, n_classes=6)
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")
    x = torch.randn(8, 6)
    print("Output shape:", model(x).shape)
