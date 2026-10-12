"""Gradient descent (notes section 3, p.17): p = -grad f(x), with a backtracking line search.

alpha0="one" starts every line search at 1; alpha0="previous" uses p.18's rule from the
second iteration on, which is what makes the 10x-scaled example converge in half the steps.
"""

from __future__ import annotations

from line_search import minimize

__all__ = ["gradient_descent"]


def gradient_descent(f, grad, x0, **options):
    """Steepest descent. ``options`` pass through to ``line_search.minimize``."""
    return minimize(f, grad, x0, lambda x, g: -g, **options)
