"""Quasi-Newton: BFGS (notes section 5, p.26): p = -B^-1 grad f(x), B built from gradients.

    bfgs_update          p.27  B = approximate Hessian, rank-two update

with s = x_(k+1) - x_k, the step just taken, and y = grad f(x_(k+1)) - grad f(x_k), how
much the gradient changed over it. Each update makes the new matrix agree with that pair
(the secant equation B s = y), so it learns the curvature without ever computing H.
y's is y transposed times s, a dot product.

The notes do not give the starting matrix. B0 = I makes the first step a
gradient-descent step, which is what the figures on p.30 show. alpha0 = 1 always (p.29).

Each update keeps the matrix positive definite only while y's > 0 (p.27 asks for exactly
that), and a positive definite matrix is what guarantees p points downhill. A Wolfe line
search, which also checks the slope at the new point, would guarantee y's > 0;
backtracking does not. On fun3 from (-2, -2), one step breaks it and the next direction
points uphill. So when y's <= 1e-10 |s| |y|, which is y's <= 0 up to round-off, the update
is skipped and the matrix stays as it was: the standard safeguard (Nocedal and Wright,
2nd ed., section 6.1).
"""

from __future__ import annotations

import numpy as np

from line_search import minimize

__all__ = ["bfgs", "bfgs_update"]

CURVATURE = 1e-10  # skip the update unless y's exceeds this fraction of |s||y|


def bfgs_update(B, s, y):
    """B - B s s' B/(s'B s) + y y'/(y's)."""
    Bs = B @ s
    return B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(y, y) / (y @ s)


def _quasi_newton(f, grad, x0, update, solve, **options):
    """Run ``minimize`` with a matrix that is updated from each accepted step."""
    M = np.eye(np.size(x0))
    last = None  # (x, grad) where the previous direction was computed

    def direction(x, g):
        nonlocal M, last
        if last is not None:
            s, y = x - last[0], g - last[1]
            if y @ s > CURVATURE * np.linalg.norm(s) * np.linalg.norm(y):
                M = update(M, s, y)
        last = (x.copy(), g.copy())
        return solve(M, g)

    return minimize(f, grad, x0, direction, alpha0="one", **options)


def bfgs(f, grad, x0, **options):
    """BFGS."""
    return _quasi_newton(
        f, grad, x0, bfgs_update, lambda B, g: np.linalg.solve(B, -g), **options
    )
