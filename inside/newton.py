"""Newton's method (notes section 4, p.21): solve H(x) p = -grad f(x), alpha0 = 1 always.

Newton needs H positive definite wherever it is evaluated (p.23); otherwise the direction
may point uphill and the line search raises.
"""

from __future__ import annotations

import numpy as np

from line_search import minimize

__all__ = ["newton", "newton_direction"]


def newton_direction(H, g):
    """p solving H p = -g."""
    return np.linalg.solve(H, -g)


def newton(f, grad, hess, x0, **options):
    """Newton with backtracking from alpha0 = 1. ``options`` pass through to ``minimize``."""

    def direction(x, g):
        H = hess(x)
        return newton_direction(H, g)

    return minimize(f, grad, x0, direction, alpha0="one", **options)
