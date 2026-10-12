"""Quasi-Newton methods (notes section 5, p.26): p = -B^-1 grad f(x), B built from gradients.

    dfp_update           p.27  C = approximate inverse Hessian, rank-two update
    bfgs_update          p.27  B = approximate Hessian, rank-two update
    bfgs_inverse_update  p.28  F = B^-1 updated directly (Sherman-Morrison-Woodbury)

with s = x_(k+1) - x_k and y = grad f(x_(k+1)) - grad f(x_k). The notes do not give the
starting matrix; B0 = C0 = I makes the first step a gradient-descent step, which is what the
figures on p.30 show. alpha0 = 1 always (p.29).

Every update keeps B positive definite only while y's > 0 (p.27 asks for exactly that).
A Wolfe line search guarantees it; backtracking alone does not, and on fun3 from (-2, -2)
one step breaks it and the next direction points uphill. So a step with
y's <= 1e-10 |s| |y|, which is y's <= 0 up to round-off, leaves the matrix unchanged: the
standard safeguard (Nocedal and Wright, 2nd ed., section 6.1).
"""

from __future__ import annotations

import numpy as np

from line_search import minimize

__all__ = ["bfgs", "bfgs_inverse_update", "bfgs_update", "dfp", "dfp_update"]

CURVATURE = 1e-10  # skip the update unless y's exceeds this fraction of |s||y|


def dfp_update(C, s, y):
    """C + s s'/(y's) - C y y' C/(y'C y)."""
    Cy = C @ y
    return C - np.outer(Cy, Cy) / (y @ Cy) + np.outer(s, s) / (y @ s)


def bfgs_update(B, s, y):
    """B - B s s' B/(s'B s) + y y'/(y's)."""
    Bs = B @ s
    return B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(y, y) / (y @ s)


def bfgs_inverse_update(F, s, y):
    """F + y'(F y + s)/(y's)^2 s s' - (s y'F + F y s')/(y's)  (p.28)."""
    ys = y @ s
    Fy = F @ y
    return F + (y @ (Fy + s)) / ys**2 * np.outer(s, s) - (np.outer(s, Fy) + np.outer(Fy, s)) / ys


def _quasi_newton(f, grad, x0, update, solve, **options):
    """Run ``minimize`` with a matrix that is updated from each accepted step."""
    M = np.eye(np.size(x0))
    last = None

    def direction(x, g):
        nonlocal M, last
        if last is not None:
            s, y = x - last[0], g - last[1]
            if y @ s > CURVATURE * np.linalg.norm(s) * np.linalg.norm(y):
                M = update(M, s, y)
        last = (x.copy(), g.copy())
        return solve(M, g)

    return minimize(f, grad, x0, direction, alpha0="one", **options)


def bfgs(f, grad, x0, inverse=False, **options):
    """BFGS. ``inverse=True`` updates F = B^-1 (p.28) and skips the linear solve."""
    if inverse:
        return _quasi_newton(f, grad, x0, bfgs_inverse_update, lambda F, g: -F @ g, **options)
    return _quasi_newton(f, grad, x0, bfgs_update, lambda B, g: np.linalg.solve(B, -g), **options)


def dfp(f, grad, x0, **options):
    """DFP, which updates C = approximate H^-1, so p = -C grad f(x)."""
    return _quasi_newton(f, grad, x0, dfp_update, lambda C, g: -C @ g, **options)
