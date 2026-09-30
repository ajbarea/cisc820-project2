"""The three objective functions of Project 2, each with its analytic gradient and Hessian.

    fun1  sum(i * x_i^2), i = 1..100        quadratic, convex, n = 100, cond(H) = 100
    fun2  c'x - sum(log(b - Ax))            log barrier, convex, m = 500, n = 100
    fun3  100(x2 - x1^2)^2 + (1 - x1)^2     Rosenbrock, non-convex, n = 2

The derivative formulas for fun2 are the ones the project description gives in MATLAB:

    f(x) = c'x - sum(log(b - Ax))
    grad = c + A'(1 ./ (b - Ax))
    H    = A' diag(1 ./ (b - Ax).^2) A

``PROBLEMS`` holds all three behind one interface, so an optimizer written against
``(f, grad, hess, x0)`` runs on every function without knowing which it has.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DATA",
    "PROBLEMS",
    "Problem",
    "fun1",
    "fun1_grad",
    "fun1_hess",
    "fun2",
    "fun2_grad",
    "fun2_hess",
    "fun2_slack",
    "fun3",
    "fun3_grad",
    "fun3_hess",
    "load_fun2",
]

DATA = Path(__file__).resolve().parent / "data"


# --- fun 1: sum(i * x_i^2) --------------------------------------------------------------

N1 = 100
_I = np.arange(1, N1 + 1, dtype=np.float64)  # the weights i = 1..100


def fun1(x):
    """f(x) = sum(i * x_i^2). Minimum f = 0 at x = 0, the one optimum the description gives."""
    x = np.asarray(x, dtype=np.float64)
    return float(_I[: x.size] @ (x**2))


def fun1_grad(x):
    """grad_i = 2 i x_i."""
    x = np.asarray(x, dtype=np.float64)
    return 2 * _I[: x.size] * x


def fun1_hess(x):
    """H = diag(2i), constant. Eigenvalues 2..200, so cond(H) = 100 whatever x is."""
    x = np.asarray(x, dtype=np.float64)
    return np.diag(2 * _I[: x.size])


# --- fun 2: c'x - sum(log(b - Ax)) ------------------------------------------------------

_FUN2: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def load_fun2(directory=DATA, order="F"):
    """Read A (500, 100), b (500,) and c (100,) from ``data/fun2_{A,b,c}.txt``.

    Each file is one whitespace-separated run of numbers, so A's 50,000 entries have to be
    folded back into a matrix. ``order="F"`` folds column by column, which is how MATLAB
    writes a matrix out; ``order="C"`` folds row by row. Both build a valid problem with a
    different minimum, so the choice is an assumption, not a detail -- see issue #9.
    """
    A = np.loadtxt(Path(directory) / "fun2_A.txt").reshape((500, 100), order=order)
    b = np.loadtxt(Path(directory) / "fun2_b.txt")
    c = np.loadtxt(Path(directory) / "fun2_c.txt")
    return A, b, c


def _coefficients():
    global _FUN2
    if _FUN2 is None:
        _FUN2 = load_fun2()
    return _FUN2


def fun2_slack(x, coefficients=None):
    """b - Ax. Every entry must stay positive: it is what the logarithm is taken of."""
    A, b, _ = coefficients or _coefficients()
    return b - A @ np.asarray(x, dtype=np.float64)


def fun2(x, coefficients=None):
    """f(x) = c'x - sum(log(b - Ax)), or +inf outside the domain b - Ax > 0.

    Returning +inf rather than raising keeps a backtracking line search working: an
    infeasible trial step fails the sufficient-decrease test and gets halved like any
    other rejected step.
    """
    A, b, c = coefficients or _coefficients()
    x = np.asarray(x, dtype=np.float64)
    slack = b - A @ x
    if np.any(slack <= 0):
        return np.inf
    return float(c @ x - np.sum(np.log(slack)))


def fun2_grad(x, coefficients=None):
    """grad = c + A'(1 / (b - Ax))."""
    A, b, c = coefficients or _coefficients()
    return c + A.T @ (1.0 / (b - A @ np.asarray(x, dtype=np.float64)))


def fun2_hess(x, coefficients=None):
    """H = A' diag(1 / (b - Ax)^2) A.

    Formed as A' @ (w[:, None] * A), which scales the rows of A directly rather than
    materialising the 500 x 500 diagonal matrix.
    """
    A, b, _ = coefficients or _coefficients()
    slack = b - A @ np.asarray(x, dtype=np.float64)
    return A.T @ ((1.0 / slack**2)[:, None] * A)


# --- fun 3: Rosenbrock ------------------------------------------------------------------


def fun3(x):
    """f(x) = 100(x2 - x1^2)^2 + (1 - x1)^2."""
    x = np.asarray(x, dtype=np.float64)
    return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def fun3_grad(x):
    """grad = (-400 x1 (x2 - x1^2) - 2(1 - x1), 200(x2 - x1^2))."""
    x = np.asarray(x, dtype=np.float64)
    r = x[1] - x[0] ** 2
    return np.array([-400 * x[0] * r - 2 * (1 - x[0]), 200 * r])


def fun3_hess(x):
    """H = [[-400(x2 - x1^2) + 800 x1^2 + 2, -400 x1], [-400 x1, 200]].

    Indefinite wherever x2 > x1^2 + 1/200, which is most of the valley floor: Newton needs
    a modification there, because the Newton direction is not a descent direction.
    """
    x = np.asarray(x, dtype=np.float64)
    r = x[1] - x[0] ** 2
    return np.array([[-400 * r + 800 * x[0] ** 2 + 2, -400 * x[0]], [-400 * x[0], 200.0]])


# --- one interface over the three -------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One objective function with everything an optimizer needs to start on it.

    ``minimum`` is the optimal value where it is known and None where the instructor is
    withholding it until after the project.
    """

    name: str
    f: Callable
    grad: Callable
    hess: Callable
    n: int
    x0: np.ndarray
    minimum: float | None
    note: str


PROBLEMS = {
    "fun1": Problem(
        name="sum(i * x_i^2), i = 1..100",
        f=fun1,
        grad=fun1_grad,
        hess=fun1_hess,
        n=N1,
        x0=np.ones(N1),
        minimum=0.0,
        note="Convex quadratic. The description gives f_min = 0, so this is the function "
        "the convergence curves are drawn on.",
    ),
    "fun2": Problem(
        name="c'x - sum(log(b - Ax)), m = 500, n = 100",
        f=fun2,
        grad=fun2_grad,
        hess=fun2_hess,
        n=100,
        x0=np.zeros(100),
        minimum=None,
        note="Convex on the open set b - Ax > 0 and +inf off it. x = 0 is feasible "
        "because every b_i is positive, which makes it the safe starting point.",
    ),
    "fun3": Problem(
        name="100(x2 - x1^2)^2 + (1 - x1)^2",
        f=fun3,
        grad=fun3_grad,
        hess=fun3_hess,
        n=2,
        x0=np.array([-1.2, 1.0]),
        minimum=0.0,
        note="Rosenbrock. Non-convex, minimum 0 at (1, 1) -- textbook, not given by the "
        "instructor. (-1.2, 1.0) is the standard starting point.",
    ),
}
