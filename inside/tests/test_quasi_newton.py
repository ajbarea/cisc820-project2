"""``quasi_newton.py`` against topic-08's BFGS example, Figure 5.1 (p.30), and section 5.

c = 0.1, rho = 1/2, alpha0 = 1 always, B0 = I. ``FIGURE`` holds what the figure shows,
read off the axes.

The figure's quadratic run does not follow from c = 0.1. Its second iterate sits at
(-1, -6), half a step along the BFGS direction. At c = 0.1 the full step passes the
sufficient-decrease test (the actual decrease is 0.1247 of the predicted one) and lands at
(-45.8, 0.46) instead. Any c in (0.1247, 0.3806] reproduces the figure; above 0.3806 the
first step changes too. Both reach the minimum in the figure's 7 iterations.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_descent import gradient_descent
from newton import newton
from notes_examples import EXPONENTIAL, QUADRATIC
from quasi_newton import bfgs, bfgs_update

RUNS = {"5.1 quadratic": QUADRATIC, "5.1 exponential": EXPONENTIAL}

FIGURE = {
    "5.1 quadratic": dict(
        iterations=7, decreases=(2.4e4, 3.1e3), path={1: (44, -12), 2: (-1, -6)}
    ),
    "5.1 exponential": dict(
        iterations=9, decreases=(112, 20), path={1: (1.5, -0.57), 2: (-0.78, -0.02)}
    ),
}

# How many of the figure's iterates and decreases a run at c = 0.1 reproduces. The quadratic's
# second iterate needs a larger c; see the module docstring and the c tests below.
MATCHED_AT_C_0_1 = {"5.1 quadratic": 1, "5.1 exponential": 2}
PATH_TOL = {
    "5.1 quadratic": 1.0,
    "5.1 exponential": 0.05,
}  # a fraction of one axis tick


def run(name, **options):
    f, grad, _, x0 = RUNS[name]
    return bfgs(f, grad, x0, **options)


@pytest.mark.parametrize("name", FIGURE)
def test_figure_iteration_count(name):
    assert run(name).iterations == FIGURE[name]["iterations"]


@pytest.mark.parametrize("name", FIGURE)
def test_figure_decreases(name):
    want = FIGURE[name]["decreases"][: MATCHED_AT_C_0_1[name]]
    assert run(name).decreases[: len(want)] == pytest.approx(want, rel=0.06, abs=0.5)


@pytest.mark.parametrize("name", FIGURE)
def test_figure_path(name):
    r = run(name)
    for k, xk in FIGURE[name]["path"].items():
        if k <= MATCHED_AT_C_0_1[name]:
            assert r.path[k] == pytest.approx(xk, abs=PATH_TOL[name])


def test_bfgs_quadratic_figure_needs_larger_c():
    """Figure 5.1's second iterate (-1, -6) is half a step; c = 0.1 accepts the full step."""
    stated = run("5.1 quadratic")
    figure = run("5.1 quadratic", c=0.15)
    assert stated.alphas[1] == 1 and stated.path[2] == pytest.approx(
        (-45.76, 0.458), abs=0.01
    )
    assert figure.alphas[1] == 0.5 and figure.path[2] == pytest.approx(
        (-1, -6), abs=0.05
    )
    assert stated.iterations == figure.iterations == 7


@pytest.mark.parametrize(
    "c,matches", [(0.1246, False), (0.1248, True), (0.38, True), (0.381, False)]
)
def test_bfgs_quadratic_figure_c_range_is_0_1247_to_0_3806(c, matches):
    """Below 0.1247 the full second step passes; above 0.3806 the first step changes."""
    r = run("5.1 quadratic", c=c)
    figure = r.path[1] == pytest.approx((43.75, -12.5)) and r.path[2] == pytest.approx(
        (-1.007, -6.021), abs=0.01
    )
    assert figure is matches


@pytest.mark.parametrize("name", FIGURE)
def test_bfgs_first_step_is_a_gradient_descent_step(name):
    """B0 = I, which Figure 5.1 shows: the first BFGS iterate is Figure 3.1's."""
    f, grad, _, x0 = RUNS[name]
    assert run(name).path[1] == pytest.approx(gradient_descent(f, grad, x0).path[1])


def test_bfgs_update_satisfies_the_secant_equation():
    """p.26: B_(k+1) s = y."""
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=3), rng.normal(size=3)
    y = y if y @ s > 0 else -y
    assert bfgs_update(np.eye(3), s, y) @ s == pytest.approx(y)


def test_bfgs_reaches_newtons_minimiser():
    f, grad, hess, x0 = EXPONENTIAL
    assert bfgs(f, grad, x0).x == pytest.approx(newton(f, grad, hess, x0).x, abs=1e-4)
