"""
Part I — Weight Initialization Strategies
Gaussian, Constant, and Xavier initialization.
"""

import torch
import torch.nn as nn
from copy import deepcopy


def init_gaussian(model: nn.Module, mean: float = 0.0, std: float = 0.01) -> nn.Module:
    """Initialize all Linear weights with a Gaussian (Normal) distribution."""
    m = deepcopy(model)
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=mean, std=std)
            nn.init.zeros_(layer.bias)
    return m


def init_constant(model: nn.Module, val: float = 0.01) -> nn.Module:
    """Initialize all Linear weights with a constant value."""
    m = deepcopy(model)
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight, val)
            nn.init.zeros_(layer.bias)
    return m


def init_xavier(model: nn.Module) -> nn.Module:
    """Initialize all Linear weights using Xavier Uniform initialization."""
    m = deepcopy(model)
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return m


STRATEGIES = {
    "Gaussian": init_gaussian,
    "Constant":  init_constant,
    "Xavier":    init_xavier,
}


def apply_init(model: nn.Module, strategy: str) -> nn.Module:
    """Apply a named initialization strategy to a copy of model.

    Parameters
    ----------
    model    : nn.Module to copy and initialize
    strategy : 'Gaussian' | 'Constant' | 'Xavier'
    """
    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGIES)}")
    return fn(model)


if __name__ == "__main__":
    from mlp_custom import MLPCustom
    base = MLPCustom(n_features=6, n_classes=6)
    for name, fn in STRATEGIES.items():
        m = fn(base)
        w0 = list(m.parameters())[0]
        print(f"[{name:10s}] first weight row: mean={w0.mean():.4f}  std={w0.std():.4f}")
