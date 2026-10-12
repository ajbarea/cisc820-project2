"""Newton's method (notes section 4, p.21): solve H(x) p = -grad f(x), alpha0 = 1 always.

Plain Newton needs H positive definite wherever it is evaluated (p.23). ``modified=True``
applies the further-reading fix on p.24, Greenstadt's method: rebuild H from its
eigenvectors with every eigenvalue lifted to max(|lambda|, delta), so H + E is symmetric
positive definite and not close to singular.
"""

from __future__ import annotations

import numpy as np

from line_search import minimize

__all__ = ["greenstadt", "newton", "newton_direction"]

DELTA = 1e-8  # smallest eigenvalue the modified Hessian may have


def greenstadt(H, delta=DELTA):
    """H + E with every eigenvalue replaced by max(|lambda|, delta)  (p.24)."""
    lam, V = np.linalg.eigh(H)
    return (V * np.maximum(np.abs(lam), delta)) @ V.T


def newton_direction(H, g):
    """p solving H p = -g."""
    return np.linalg.solve(H, -g)


def newton(f, grad, hess, x0, modified=False, **options):
    """Newton with backtracking from alpha0 = 1. ``options`` pass through to ``minimize``."""

    def direction(x, g):
        H = hess(x)
        return newton_direction(greenstadt(H) if modified else H, g)

    return minimize(f, grad, x0, direction, alpha0="one", **options)
