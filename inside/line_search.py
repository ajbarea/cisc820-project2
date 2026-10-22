"""Line search and the iteration every descent method shares (notes section 2.2).

    backtracking         p.14  shrink alpha by rho until sufficient decrease holds
    interpolated_step    p.15  minimiser of the quadratic through phi(0), phi'(0), phi(alpha0)
    previous_step_alpha0 p.18  alpha0 that repeats the last step's first-order change
    quadratic_alpha0     p.18  alpha0 from a quadratic through the last two f values
    minimize             p.12  "until x is a good enough solution": find p, step, repeat

Every method here picks a search direction p, then the line search picks a step length
alpha and moves x to x + alpha p. phi(alpha) = f(x + alpha p) is f along that line, and
phi'(0) = p'grad f(x) is its slope at the start (p' is p transposed, so this is a dot
product). p is a descent direction when that slope is negative.

The notes never fix "good enough". Stopping once |f(x_k) - f(x_(k-1))| < 1e-8 reproduces
the iteration counts of every convergence figure in topics 06 to 08, so that is the default.
A run also stops when the line search returns alpha = 0, meaning no step can move x.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Result",
    "backtracking",
    "interpolated_step",
    "minimize",
    "previous_step_alpha0",
    "quadratic_alpha0",
]

# The notes' examples all use c = 0.1 and rho = 1/2.
C = 0.1  # how much of the predicted decrease a step must achieve to be accepted
RHO = 0.5  # factor alpha shrinks by after each rejected trial
TOL = 1e-8
MAX_ITER = 10_000


def backtracking(f, x, p, grad_x, alpha0=1.0, c=C, rho=RHO, fx=None):
    """Largest alpha = alpha0 * rho^j with f(x + alpha p) <= f(x) + c alpha p'grad f(x).

    That inequality is the sufficient-decrease (Armijo) condition: the step must lower f by
    at least a fraction c of what the slope at x predicts. Starting from alpha0, alpha is
    multiplied by rho until it holds. ``fx`` is f(x) if the caller already has it.

    A trial point where f is +inf or overflows fails the test like any other, which is what
    keeps the search inside fun2's domain. Returns 0 once alpha * p is too small to move x
    in floating point: no step can decrease f, so x is as good as this search can make it.
    """
    fx = f(x) if fx is None else fx
    slope = float(p @ grad_x)
    if slope > 0 or (slope == 0 and np.any(p)):
        raise ValueError(f"p is not a descent direction: p'grad f = {slope:.3g} >= 0")
    if not np.any(p):
        return 0.0  # grad f = 0 exactly: a stationary point
    alpha = alpha0
    with np.errstate(over="ignore", invalid="ignore"):
        while True:
            trial = x + alpha * p
            if np.array_equal(trial, x):  # every component rounds back: nothing left to try
                return 0.0
            if f(trial) <= fx + c * alpha * slope:  # False for nan and +inf
                return alpha
            alpha *= rho


def interpolated_step(phi0, dphi0, alpha0, phi_alpha0):
    """alpha1 = -phi'(0) alpha0^2 / (2 (phi(alpha0) - phi(0) - alpha0 phi'(0)))  (p.15)."""
    return -dphi0 * alpha0**2 / (2 * (phi_alpha0 - phi0 - alpha0 * dphi0))


def previous_step_alpha0(alpha_prev, p_prev, grad_prev, p, grad):
    """alpha0 = alpha_(k-1) p_(k-1)'grad f(x_(k-1)) / p_k'grad f(x_k)  (p.18, first rule)."""
    return alpha_prev * float(p_prev @ grad_prev) / float(p @ grad)


def quadratic_alpha0(f_prev, f_curr, dphi0):
    """alpha0 = 2 (f(x_k) - f(x_(k-1))) / phi'(0)  (p.18, second rule).

    The notes write phi'(0) = p_(k-1)'grad f(x_(k-1)); pass whichever slope is wanted.
    """
    return 2 * (f_curr - f_prev) / dphi0


@dataclass
class Result:
    """Where a run ended, and every iteration on the way (path[0] is the start)."""

    x: np.ndarray
    f: float
    iterations: int
    path: list[np.ndarray] = field(repr=False)
    values: list[float] = field(repr=False)
    alphas: list[float] = field(repr=False)

    @property
    def decreases(self):
        """f(x_(k-1)) - f(x_k) for k = 1..n: what the notes' convergence figures plot."""
        return -np.diff(self.values)


def minimize(
    f,
    grad,
    x0,
    direction: Callable,
    *,
    alpha0="one",
    c=C,
    rho=RHO,
    tol=TOL,
    max_iter=MAX_ITER,
):
    """The line-search loop of p.12, with ``direction(x, grad f(x)) -> p`` supplying p.

    ``alpha0`` is "one" (every iteration starts from 1, as Newton and quasi-Newton must) or
    "previous" (p.18's first rule from the second iteration on).
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    fx = f(x)
    path, values, alphas = [x.copy()], [fx], []
    prev = None  # (alpha, p, grad) of the last iteration, for the "previous" alpha0 rule
    for _ in range(max_iter):
        g = np.asarray(grad(x), dtype=np.float64)
        p = direction(x, g)
        start = 1.0
        if alpha0 == "previous" and prev is not None:
            start = previous_step_alpha0(*prev, p, g)
        elif alpha0 not in ("one", "previous"):
            raise ValueError(f"alpha0 must be 'one' or 'previous', got {alpha0!r}")
        alpha = backtracking(f, x, p, g, start, c, rho, fx)
        prev = (alpha, p, g)
        x = x + alpha * p
        f_new = f(x)
        path.append(x.copy())
        values.append(f_new)
        alphas.append(alpha)
        done = abs(fx - f_new) < tol or alpha == 0  # alpha = 0: no step can move x
        fx = f_new
        if done:
            break
    return Result(x, fx, len(alphas), path, values, alphas)
