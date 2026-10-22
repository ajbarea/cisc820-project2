"""``line_search.py`` against what topic-05, section 2.2, states about backtracking, on the notes'
quadratic and exponential examples.
"""

from __future__ import annotations

import numpy as np
import pytest

from line_search import RHO, C, backtracking, minimize
from notes_examples import EXPONENTIAL, QUADRATIC


def test_backtracking_accepts_sufficient_decrease_and_rejected_the_step_before():
    """p.14: the accepted alpha meets the test; alpha/rho, the previous trial, did not."""
    f, grad, _, x0 = EXPONENTIAL
    g = grad(x0)
    p = -g
    alpha = backtracking(f, x0, p, g)
    assert f(x0 + alpha * p) <= f(x0) + C * alpha * (p @ g)
    with np.errstate(over="ignore"):
        rejected = f(x0 + alpha / RHO * p)
    assert rejected > f(x0) + C * alpha / RHO * (p @ g)


def test_backtracking_refuses_an_ascent_direction():
    """p.23: without p.d. H, Newton "may take ascent directions"; the search must not loop."""
    f, grad, _, x0 = EXPONENTIAL
    with pytest.raises(ValueError, match="not a descent direction"):
        backtracking(f, x0, grad(x0), grad(x0))


def test_backtracking_keeps_going_while_any_component_can_move():
    """The line search gives up only when x + alpha p rounds to x in every component."""

    def f(x):
        return 1e12 * x[1] ** 2 + (x[0] - 1e6) ** 2

    x = np.array([1e6, 1e-10])
    g = np.array([0.0, 2e12 * x[1]])
    alpha = backtracking(f, x, -g, g)
    assert alpha > 0 and f(x - alpha * g) < f(x)


def test_backtracking_returns_zero_once_no_step_moves_x():
    f, grad, _, _ = QUADRATIC
    x = np.array([1.0, 1.0])
    p = np.array([-1e-300, -1e-300])  # a descent direction far below round-off at x
    assert backtracking(f, x, p, grad(x)) == 0


def test_minimize_records_every_iterate_and_stops_on_the_tolerance():
    f, grad, _, x0 = QUADRATIC
    r = minimize(f, grad, x0, lambda x, g: -g, tol=1e-8)
    assert len(r.path) == len(r.values) == r.iterations + 1
    assert r.path[0] == pytest.approx(x0)
    assert abs(r.values[-2] - r.values[-1]) < 1e-8
    assert np.all(r.decreases >= 0)


def test_minimize_rejects_an_unknown_alpha0_rule():
    f, grad, _, x0 = QUADRATIC
    with pytest.raises(ValueError, match="alpha0 must be"):
        minimize(f, grad, x0, lambda x, g: -g, alpha0="two")
