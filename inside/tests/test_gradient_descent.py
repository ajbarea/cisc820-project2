"""``gradient_descent.py`` against topic-06's examples, Figures 3.1 (p.19) and 3.2 (p.20).

c = 0.1, rho = 1/2, and alpha0 = 1 except where Figure 3.2 uses p.18's previous-step rule.
The figures print no numbers, so ``FIGURE`` holds what they show, read off the axes: the
last iteration plotted, the first decreases f(x_(k-1)) - f(x_k), and iterates marked on the
contour plots.
"""

from __future__ import annotations

import pytest

from gradient_descent import gradient_descent
from notes_examples import EXPONENTIAL, QUADRATIC, SCALED

RUNS = {
    "3.1 quadratic": (QUADRATIC, "one"),
    "3.1 exponential": (EXPONENTIAL, "one"),
    "3.2 scaled, alpha0 = 1": (SCALED, "one"),
    "3.2 scaled, previous-step alpha0": (SCALED, "previous"),
}

FIGURE = {
    "3.1 quadratic": dict(iterations=46, decreases=(2.4e4, 1.9e3), path={1: (44, -12)}),
    "3.1 exponential": dict(
        iterations=24, decreases=(112, 12), path={1: (1.5, -0.57), 2: (1.13, 0.42)}
    ),
    "3.2 scaled, alpha0 = 1": dict(
        iterations=39, decreases=(1250,), path={1: (1.67, 0.02), 2: (-1.28, -0.49)}
    ),
    "3.2 scaled, previous-step alpha0": dict(
        iterations=20, decreases=(1250,), path={1: (1.67, 0.02), 2: (-1.13, -0.46)}
    ),
}

# How far an iterate may sit from the figure: about a fraction of one axis tick.
PATH_TOL = {"3.1 quadratic": 1.0}
DEFAULT_PATH_TOL = 0.05


def run(name):
    (f, grad, _, x0), alpha0 = RUNS[name]
    return gradient_descent(f, grad, x0, alpha0=alpha0)


@pytest.mark.parametrize("name", FIGURE)
def test_figure_iteration_count(name):
    """|f(x_k) - f(x_(k-1))| < 1e-8 stops each run on the figure's last plotted iteration."""
    assert run(name).iterations == FIGURE[name]["iterations"]


@pytest.mark.parametrize("name", FIGURE)
def test_figure_decreases(name):
    want = FIGURE[name]["decreases"]
    assert run(name).decreases[: len(want)] == pytest.approx(want, rel=0.06, abs=0.5)


@pytest.mark.parametrize("name", FIGURE)
def test_figure_path(name):
    r = run(name)
    for k, xk in FIGURE[name]["path"].items():
        assert r.path[k] == pytest.approx(xk, abs=PATH_TOL.get(name, DEFAULT_PATH_TOL))


def test_previous_step_alpha0_halves_the_iterations_on_the_scaled_problem():
    """p.18 / Figure 3.2: 39 iterations with alpha0 = 1, 20 with the previous-step rule."""
    fixed = run("3.2 scaled, alpha0 = 1").iterations
    previous = run("3.2 scaled, previous-step alpha0").iterations
    assert previous * 2 <= fixed + 1
