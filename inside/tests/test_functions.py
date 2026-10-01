"""Every analytic derivative in ``functions.py``, checked against a finite difference of the
function itself. A sign slip or a transpose in the gradient or the Hessian shows up here
rather than as an optimizer that quietly fails to converge.
"""

from __future__ import annotations

import numpy as np
import pytest

from derivatives import gradient, hessian
from functions import (
    DATA,
    PROBLEMS,
    fun1,
    fun1_grad,
    fun1_hess,
    fun2,
    fun2_grad,
    fun2_hess,
    fun2_slack,
    fun3,
    fun3_grad,
    fun3_hess,
    load_fun2,
)

# Points to differentiate at: away from the minimum, where every term is active.
X1 = np.linspace(-1.0, 1.0, 100)
X2 = np.zeros(100)
X3 = np.array([-1.2, 1.0])


@pytest.mark.parametrize(
    ("f", "grad", "x"),
    [(fun1, fun1_grad, X1), (fun2, fun2_grad, X2), (fun3, fun3_grad, X3)],
)
def test_gradient_matches_finite_difference(f, grad, x):
    approx = gradient(f, x, order=4)
    assert np.allclose(approx, grad(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("grad", "hess", "x"),
    [(fun1_grad, fun1_hess, X1), (fun2_grad, fun2_hess, X2), (fun3_grad, fun3_hess, X3)],
)
def test_hessian_matches_difference_of_gradients(grad, hess, x):
    approx = hessian(grad, x, central=True)
    assert np.allclose(approx, hess(x), rtol=1e-4, atol=1e-4)


# --- fun 1 ------------------------------------------------------------------------------


def test_fun1_minimum_is_zero_at_the_origin():
    assert fun1(np.zeros(100)) == 0.0
    assert np.all(fun1_grad(np.zeros(100)) == 0.0)


def test_fun1_hessian_is_constant_with_condition_number_100():
    a, b = fun1_hess(np.zeros(100)), fun1_hess(np.ones(100))
    assert np.array_equal(a, b)
    assert np.linalg.cond(a) == pytest.approx(100.0)


# --- fun 2 ------------------------------------------------------------------------------


def test_fun2_shapes():
    A, b, c = load_fun2()
    assert (A.shape, b.shape, c.shape) == ((500, 100), (500,), (100,))


def test_origin_is_feasible_because_every_b_is_positive():
    _, b, _ = load_fun2()
    assert np.all(b > 0)
    assert np.all(fun2_slack(np.zeros(100)) > 0)
    assert np.isfinite(fun2(np.zeros(100)))


def test_fun2_is_infinite_outside_its_domain():
    A, b, _ = load_fun2()
    # Step far enough along a row of A that its own slack goes negative.
    x = A[0] * (b[0] / (A[0] @ A[0]) + 1.0)
    assert fun2_slack(x)[0] < 0
    assert fun2(x) == np.inf


def test_fun2_hessian_is_positive_definite_at_the_origin():
    # Convexity on the domain: the Cholesky factorisation exists only if H is p.d.
    np.linalg.cholesky(fun2_hess(np.zeros(100)))


def test_column_major_and_row_major_build_different_problems():
    """The reshape order changes the objective, so the ReadMe's column-major order matters.

    Both orders are feasible at x = 0 and both give a convex problem; nothing in the data
    itself rules either out, only ``data/ReadMe.rtf`` does.
    """
    column_major = load_fun2(order="F")
    row_major = load_fun2(order="C")
    assert not np.array_equal(column_major[0], row_major[0])

    x = np.full(100, 0.01)
    assert fun2(x, coefficients=column_major) != fun2(x, coefficients=row_major)


# --- fun 3 ------------------------------------------------------------------------------


def test_fun3_minimum_is_zero_at_one_one():
    assert fun3([1.0, 1.0]) == 0.0
    assert np.all(fun3_grad([1.0, 1.0]) == 0.0)


def test_fun3_hessian_is_indefinite_on_the_valley_floor():
    # x2 > x1^2 + 1/200 makes the (0, 0) entry negative, so the Newton direction there
    # is not guaranteed to be a descent direction.
    H = fun3_hess([0.0, 1.0])
    assert np.min(np.linalg.eigvalsh(H)) < 0


# --- the registry -----------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(PROBLEMS))
def test_every_problem_starts_somewhere_finite(key):
    p = PROBLEMS[key]
    assert p.x0.size == p.n
    assert np.isfinite(p.f(p.x0))
    assert p.grad(p.x0).shape == (p.n,)
    assert p.hess(p.x0).shape == (p.n, p.n)


def test_data_files_ship_with_the_repository():
    assert sorted(f.name for f in DATA.glob("*.txt")) == ["fun2_A.txt", "fun2_b.txt", "fun2_c.txt"]
