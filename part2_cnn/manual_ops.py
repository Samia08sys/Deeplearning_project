"""
Part II — Manual Implementations of Convolution & Pooling Operations
(No high-level PyTorch layers used in the core computations)
"""

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
def output_size(in_size: int, kernel: int, padding: int, stride: int) -> int:
    """Compute the output spatial dimension for a convolution/pooling layer.

    Formula:  out = floor((in + 2*padding - kernel) / stride) + 1
    """
    return (in_size + 2 * padding - kernel) // stride + 1


# ─────────────────────────────────────────────────────────────────────────────
def manual_conv2d(
    X: np.ndarray,
    K: np.ndarray,
    padding: int = 0,
    stride:  int = 1,
) -> np.ndarray:
    """2-D cross-correlation (no PyTorch conv layers).

    Parameters
    ----------
    X       : (H, W) or (C, H, W) input feature map
    K       : (kH, kW) or (C, kH, kW) kernel
    padding : zero-padding added to both sides of H and W
    stride  : step size

    Returns
    -------
    Y : (out_H, out_W) output
    """
    # Ensure 3-D
    if X.ndim == 2:
        X = X[np.newaxis, ...]
    if K.ndim == 2:
        K = K[np.newaxis, ...]

    C, H, W   = X.shape
    Ck, kH, kW = K.shape
    assert C == Ck, "Input channels must match kernel channels"

    if padding > 0:
        X = np.pad(X, ((0, 0), (padding, padding), (padding, padding)))

    out_H = output_size(H, kH, padding, stride)
    out_W = output_size(W, kW, padding, stride)

    Y = np.zeros((out_H, out_W), dtype=np.float64)
    for i in range(out_H):
        for j in range(out_W):
            patch = X[:, i*stride : i*stride+kH, j*stride : j*stride+kW]
            Y[i, j] = np.sum(patch * K)
    return Y


# ─────────────────────────────────────────────────────────────────────────────
def manual_maxpool2d(
    X: np.ndarray,
    kernel_size: int = 2,
    stride:      int = 2,
) -> np.ndarray:
    """2-D max-pooling (no PyTorch pooling layers).

    Parameters
    ----------
    X           : (H, W) or (C, H, W)
    kernel_size : pooling window size (square)
    stride      : step size

    Returns
    -------
    Y : (C, out_H, out_W) or (out_H, out_W)
    """
    single = (X.ndim == 2)
    if single:
        X = X[np.newaxis, ...]
    C, H, W = X.shape

    out_H = output_size(H, kernel_size, 0, stride)
    out_W = output_size(W, kernel_size, 0, stride)
    Y = np.zeros((C, out_H, out_W), dtype=X.dtype)

    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                patch = X[c,
                          i*stride : i*stride+kernel_size,
                          j*stride : j*stride+kernel_size]
                Y[c, i, j] = patch.max()
    return Y[0] if single else Y


# ─────────────────────────────────────────────────────────────────────────────
def manual_avgpool2d(
    X: np.ndarray,
    kernel_size: int = 2,
    stride:      int = 2,
) -> np.ndarray:
    """2-D average-pooling (no PyTorch pooling layers).

    Same signature as manual_maxpool2d.
    """
    single = (X.ndim == 2)
    if single:
        X = X[np.newaxis, ...]
    C, H, W = X.shape

    out_H = output_size(H, kernel_size, 0, stride)
    out_W = output_size(W, kernel_size, 0, stride)
    Y = np.zeros((C, out_H, out_W), dtype=X.dtype)

    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                patch = X[c,
                          i*stride : i*stride+kernel_size,
                          j*stride : j*stride+kernel_size]
                Y[c, i, j] = patch.mean()
    return Y[0] if single else Y


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng  = np.random.RandomState(0)
    X    = rng.randn(1, 8, 8)
    K    = rng.randn(1, 3, 3)

    print("Input shape  :", X.shape)
    print("Kernel shape :", K.shape)

    configs = [
        {"padding": 0, "stride": 1},
        {"padding": 1, "stride": 1},
        {"padding": 0, "stride": 2},
    ]
    for cfg in configs:
        Y = manual_conv2d(X[0], K[0], **cfg)
        expected_h = output_size(8, 3, cfg["padding"], cfg["stride"])
        print(f"  conv2d  padding={cfg['padding']} stride={cfg['stride']}  "
              f"→ output ({Y.shape[0]}, {Y.shape[1]})  "
              f"[formula says {expected_h}x{expected_h}]")

    Xp = manual_maxpool2d(X, kernel_size=2, stride=2)
    print(f"\n  maxpool2d k=2 s=2  → {Xp.shape}")
    Xa = manual_avgpool2d(X, kernel_size=2, stride=2)
    print(f"  avgpool2d k=2 s=2  → {Xa.shape}")
