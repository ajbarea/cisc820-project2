"""Numerical gradient and Hessian by finite differences.

Implements the two exercises in topic-04 (Introduction to Optimization), section 1.2:

    gradient   p.8   order 1, 2 and 4 difference formulas, one coordinate at a time
    hessian    p.9   difference of gradients, forward or central, from any gradient function
    step_size  p.8   the notes' choice of h, sqrt(eps) * |x|_inf

The notes call all three gradient formulas "central difference". Order 1 is one-sided
(it only steps forward, x + h); orders 2 and 4 step both ways.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EPS", "gradient", "hessian", "step_size"]

EPS = np.finfo(np.float64).eps


def step_size(x):
    """h = sqrt(eps) * |x|_inf (notes p.8).

    Falls back to sqrt(eps) at x = 0, where the rule itself gives h = 0.
    """
    scale = np.max(np.abs(x))
    return np.sqrt(EPS) * (scale if scale > 0 else 1.0)


def _unit(n, i):
    """e_i: the i-th unit vector, only the i-th element nonzero and equal to 1."""
    e = np.zeros(n)
    e[i] = 1.0
    return e


def gradient(f, x, order=2, h=None):
    """Approximate grad f(x) with the order-1, 2 or 4 formula, one coordinate e_i at a time.

    order 1:  (f(x + h e_i) - f(x)) / h
    order 2:  (f(x + h e_i) - f(x - h e_i)) / 2h
    order 4:  (-f(x + 2h e_i) + 8 f(x + h e_i) - 8 f(x - h e_i) + f(x - 2h e_i)) / 12h
    """
    x = np.asarray(x, dtype=np.float64)
    h = step_size(x) if h is None else h
    n = x.size
    g = np.empty(n)
    fx = f(x) if order == 1 else None
    for i in range(n):
        d = h * _unit(n, i)
        if order == 1:
            g[i] = (f(x + d) - fx) / h
        elif order == 2:
            g[i] = (f(x + d) - f(x - d)) / (2 * h)
        elif order == 4:
            g[i] = (-f(x + 2 * d) + 8 * f(x + d) - 8 * f(x - d) + f(x - 2 * d)) / (12 * h)
        else:
            raise ValueError(f"order must be 1, 2 or 4, got {order}")
    return g


def hessian(grad, x, h=None, central=False):
    """Approximate H(x) from a gradient function: column j is how grad f changes along e_j.

    forward:  h_ij = (grad_i(x + h e_j) - grad_i(x)) / h
    central:  h_ij = (grad_i(x + h e_j) - grad_i(x - h e_j)) / 2h

    ``grad`` can be the true gradient or a finite-difference one, the two versions the
    exercise compares.
    """
    x = np.asarray(x, dtype=np.float64)
    h = step_size(x) if h is None else h
    n = x.size
    H = np.empty((n, n))
    gx = None if central else np.asarray(grad(x), dtype=np.float64)
    for j in range(n):
        d = h * _unit(n, j)
        if central:
            H[:, j] = (np.asarray(grad(x + d)) - np.asarray(grad(x - d))) / (2 * h)
        else:
            H[:, j] = (np.asarray(grad(x + d)) - gx) / h
    return H
