"""The two test functions of topic-04, p.6, and the starting points topics 05 to 08 use them from.

    QUADRATIC     x1^2 + 10 x2^2 from (50, 50)
    EXPONENTIAL   e^(x1+3x2-0.1) + e^(x1-3x2-0.1) + e^(-x1-0.1) from (2, 1)
    SCALED        10 times the exponential, from (2, 1)   (topic-06, p.18)

Each problem is ``(f, grad, hess, x0)``. The notes print no numbers for the optimizer
examples, only figures, so the tests check each method against what those figures show.
"""

from __future__ import annotations

import numpy as np


def f1(x):
    """y = x1^2 + 10 x2^2 (notes p.6)."""
    return x[0] ** 2 + 10 * x[1] ** 2


def grad1(x):
    return np.array([2 * x[0], 20 * x[1]])


def hess1(_x):
    return np.array([[2.0, 0.0], [0.0, 20.0]])


def _terms(x):
    return np.exp(x[0] + 3 * x[1] - 0.1), np.exp(x[0] - 3 * x[1] - 0.1), np.exp(-x[0] - 0.1)


def f2(x):
    """y = e^(x1+3x2-0.1) + e^(x1-3x2-0.1) + e^(-x1-0.1) (notes p.6)."""
    return sum(_terms(x))


def grad2(x):
    a, b, c = _terms(x)
    return np.array([a + b - c, 3 * a - 3 * b])


def hess2(x):
    a, b, c = _terms(x)
    return np.array([[a + b + c, 3 * a - 3 * b], [3 * a - 3 * b, 9 * a + 9 * b]])


QUADRATIC = (f1, grad1, hess1, np.array([50.0, 50.0]))
EXPONENTIAL = (f2, grad2, hess2, np.array([2.0, 1.0]))
SCALED = (
    lambda x: 10 * f2(x),
    lambda x: 10 * grad2(x),
    lambda x: 10 * hess2(x),
    np.array([2.0, 1.0]),
)
