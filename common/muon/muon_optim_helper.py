import torch
from .polar_express import coeffs_list
from itertools import repeat


@torch.compile
def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7, using_pe=False, ns_dtype: str = "bf16"):
    """
    Newton-Schulz iteration to compute the zeroth power of the matrix G.
    Approximates UV^T where G = USV^T.

    Args:
        G: Input matrix
        steps: Number of Newton-Schulz iterations
        eps: Small constant for numerical stability
        using_pe: Whether to use Polar Express coefficients
        ns_dtype: Computation dtype. Options: "bf16", "fp32", "fp64" (default: "bf16")
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)

    # Convert to specified computation dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    compute_dtype = dtype_map.get(ns_dtype)

    original_dtype = G.dtype
    X = G.to(compute_dtype)

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    # limit to avoid div by zero
    scaling = 1.1 if using_pe else 1.
    X = X / (X.norm(dim=(-2, -1), keepdim=True)*scaling + eps)

    # Perform the NS iterations
    if using_pe:
        hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list)))
    else:
        hs = [(a, b, c)] * steps

    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(original_dtype)
