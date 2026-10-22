"""``newton.py`` against topic-07's example, Figure 4.1 (p.25), and what section 4 states.

c = 0.1, rho = 1/2, alpha0 = 1 always. ``FIGURE`` holds what the figure shows, read off
the axes. On the quadratic it plots f - f* at iterations 1 and 2: 27500 at the start, 0
after one step.
"""

from __future__ import annotations

import numpy as np
import pytest

from newton import newton
from notes_examples import EXPONENTIAL, QUADRATIC

RUNS = {"4.1 quadratic": QUADRATIC, "4.1 exponential": EXPONENTIAL}

FIGURE = {
    "4.1 quadratic": dict(iterations=2, decreases=(27500, 0), path={1: (0, 0)}),
    "4.1 exponential": dict(
        iterations=9, decreases=(85, 31, 11, 4), path={1: (1.17, 0.95), 2: (0.75, 0.75)}
    ),
}

PATH_TOL = 0.05  # a fraction of one axis tick


@pytest.mark.parametrize("name", FIGURE)
def test_figure_iteration_count(name):
    assert newton(*RUNS[name]).iterations == FIGURE[name]["iterations"]


@pytest.mark.parametrize("name", FIGURE)
def test_figure_decreases(name):
    want = FIGURE[name]["decreases"]
    assert newton(*RUNS[name]).decreases[: len(want)] == pytest.approx(
        want, rel=0.06, abs=0.5
    )


@pytest.mark.parametrize("name", FIGURE)
def test_figure_path(name):
    r = newton(*RUNS[name])
    for k, xk in FIGURE[name]["path"].items():
        assert r.path[k] == pytest.approx(xk, abs=PATH_TOL)


def test_newton_solves_a_quadratic_in_one_step():
    """p.21: "if f is actually a quadratic ... x + p will be the global minimum"."""
    r = newton(*QUADRATIC)
    assert r.path[1] == pytest.approx([0, 0], abs=1e-12)
    assert r.alphas[0] == 1


def test_newton_converges_quadratically():
    """p.22: e_(k+1) <= c e_k^2 near the minimiser, so log e roughly doubles each step."""
    r = newton(*EXPONENTIAL, tol=0)
    x_star = r.path[-1]
    e = [np.linalg.norm(x - x_star) for x in r.path]
    tail = [k for k in range(1, len(e) - 1) if 1e-10 < e[k] < 1e-2 and e[k + 1] > 0]
    assert tail and all(e[k + 1] <= 10 * e[k] ** 2 for k in tail)
