"""``derivatives.py`` against the two test functions of topic-04, p.6, whose gradient and
Hessian the notes print. Each order is checked for the accuracy it claims, by halving h and
reading the exponent off the error ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from derivatives import EPS, gradient, hessian, step_size


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


X0 = np.array([1.0, 1.0])


def error(approx, true):
    return float(np.max(np.abs(np.asarray(approx) - np.asarray(true))))


def test_step_size_is_the_notes_rule():
    assert step_size(np.array([2.0, -3.0])) == pytest.approx(np.sqrt(EPS) * 3.0)


def test_step_size_falls_back_at_the_origin():
    # sqrt(eps) * |x|_inf is 0 at x = 0, which would divide by zero.
    assert step_size(np.zeros(3)) == pytest.approx(np.sqrt(EPS))


# Largest error either test function produces at the notes' h. Order 1 is one-sided, so it
# carries the first-derivative truncation term and lands two decades behind the other two.
GRADIENT_TOLERANCE = {1: 1e-5, 2: 1e-6, 4: 1e-6}


@pytest.mark.parametrize(("f", "grad"), [(f1, grad1), (f2, grad2)])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_gradient_matches_the_true_gradient(f, grad, order):
    assert error(gradient(f, X0, order), grad(X0)) < GRADIENT_TOLERANCE[order]


@pytest.mark.parametrize("order", [1, 2, 4])
def test_gradient_shows_the_order_of_accuracy_it_claims(order):
    """err(h) / err(h/2) should be about 2^order. f2 is used because no truncation term
    of it vanishes; on the quadratic f1 orders 2 and 4 are already exact."""
    h = 0.1
    ratio = error(gradient(f2, X0, order, h), grad2(X0)) / error(
        gradient(f2, X0, order, h / 2), grad2(X0)
    )
    assert np.log2(ratio) == pytest.approx(order, abs=0.35)


def test_orders_2_and_4_are_exact_on_a_quadratic():
    # Their error terms carry the 3rd and 5th derivatives, which are zero here.
    for order in (2, 4):
        assert error(gradient(f1, X0, order, 0.1), grad1(X0)) < 1e-10


@pytest.mark.parametrize(("grad", "hess"), [(grad1, hess1), (grad2, hess2)])
@pytest.mark.parametrize("central", [False, True])
def test_hessian_from_the_true_gradient(grad, hess, central):
    assert error(hessian(grad, X0, central=central), hess(X0)) < 1e-4


@pytest.mark.parametrize(("f", "hess"), [(f1, hess1), (f2, hess2)])
def test_hessian_from_a_finite_difference_gradient(f, hess):
    """The exercise's second version: differencing an approximate gradient.

    Two layers of finite difference, so the error is judged relative to the size of H
    rather than in absolute terms: f2's Hessian entries reach 446 and f1's reach 20.
    """

    def fd_grad(z):
        return gradient(f, z, order=2)

    H = hess(X0)
    assert error(hessian(fd_grad, X0, h=1e-4), H) / np.max(np.abs(H)) < 1e-3


def test_gradient_rejects_an_order_it_does_not_have():
    with pytest.raises(ValueError, match="order must be 1, 2 or 4"):
        gradient(f1, X0, order=3)
