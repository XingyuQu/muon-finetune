"""
Polar Express coefficients for Newton-Schulz iteration.

The coefficients are used by zeropower_via_newtonschulz5 and zeropower_via_newtonschulz5_lowrank
when using_pe=True. This provides better convergence properties than standard NS coefficients.
"""
from math import inf, sqrt

import numpy as np


def optimal_quintic(l, u):
    """Solve for (a, b, c) of the equioscillating quintic p(x)=ax+bx^3+cx^5 on [l, u]."""
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        # Above this threshold, the equioscillating polynomials
        # is numerically equal to...
        return (15/8)/u, (-10/8)/(u**3), (3/8)/(u**5)
    # This initialization becomes exact as l -> u
    q = (3*l + 1) / 4
    r = (l + 3) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt((-3*b + np.array([-1, 1]) *
                        sqrt(9*b**2 - 20*a*c)) / (10*c))
    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, safety_factor_eps=0, cushion=0):
    """Build `num_iters` Polar-Express NS coefficient triples by composing optimal quintics on [l, 1]."""
    u = 1
    assert 0 <= l <= u
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for iter in range(num_iters):
        a, b, c = optimal_quintic(max(l, cushion*u), u)
        if cushion*u > l:
            # Due to cushioning, this may be centered around 1 with
            # respect to 0.024*u, u. Recenter it around 1 with respect
            # to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
            pl = a*l + b*l**3 + c*l**5
            pu = a*u + b*u**3 + c*u**5
            rescaler = 2/(pl + pu)
            a *= rescaler; b *= rescaler; c *= rescaler
        # Optionally incorporate safety factor here:
        if iter < num_iters - 1:  # don't apply to last polynomial
            a /= safety_factor; b /= safety_factor**3; c /= safety_factor**5
        coefficients.append((a, b, c))
        l = a*l + b*l**3 + c*l**5
        u = 2 - l
    return coefficients


# Pre-computed Polar Express coefficients
# Used by zeropower_via_newtonschulz5 and zeropower_via_newtonschulz5_lowrank when using_pe=True
coeffs_list = optimal_composition(l=1e-3, num_iters=10, safety_factor_eps=1e-2, cushion=0.02)
