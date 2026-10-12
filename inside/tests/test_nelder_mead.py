"""``nelder_mead.py`` against topic-09's example, Figure 6.2 (p.33).

    f(x, y) = x^2 - 4x + y^2 - y - xy from V1 = (0, 0), V2 = (1.2, 0), V3 = (0, 0.8)
    alpha = 1, gamma = 2, rho = 1/2, sigma = 1/2

The minimum is at (3, 2), f = -7, where grad f = (2x - 4 - y, 2y - 1 - x) = 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from nelder_mead import _order, nelder_mead

START = [(0.0, 0.0), (1.2, 0.0), (0.0, 0.8)]
MINIMUM = (np.array([3.0, 2.0]), -7.0)


def f(v):
    x, y = v
    return x**2 - 4 * x + y**2 - y - x * y


# Figure 6.2. T1 to T6 sit on the axis ticks and read exactly. T7 to T10 are read from the
# zoomed figure; T7 onward only appears once the T6 tie is broken as the notes do.
FIGURE = [
    {(0, 0), (1.2, 0), (0, 0.8)},
    {(1.2, 0), (0, 0.8), (1.8, 1.2)},
    {(1.2, 0), (1.8, 1.2), (3.0, 0.4)},
    {(1.8, 1.2), (3.0, 0.4), (3.6, 1.6)},
    {(1.8, 1.2), (3.6, 1.6), (2.4, 2.4)},
    {(3.6, 1.6), (2.4, 2.4), (2.4, 1.6)},
    {(2.4, 2.4), (2.4, 1.6), (3.0, 1.8)},
    {(2.4, 1.6), (3.0, 1.8), (2.55, 2.05)},
    {(3.0, 1.8), (2.55, 2.05), (3.15, 2.25)},
    {(3.0, 1.8), (3.15, 2.25), (2.8, 2.05)},
]
EXACT = 6  # T1 to T6 are exact; T7 to T10 are figure readings, good to about 0.05


def _as_set(simplex, decimals):
    return {tuple(np.round(v, decimals)) for v in simplex}


def test_nelder_mead_reproduces_the_first_six_triangles_exactly():
    _, history = nelder_mead(f, START)
    for k in range(EXACT):
        assert _as_set(history[k], 10) == _as_set(FIGURE[k], 10), f"T{k + 1}"


@pytest.mark.parametrize(
    "k", range(EXACT, len(FIGURE)), ids=[f"T{k + 1}" for k in range(EXACT, len(FIGURE))]
)
def test_nelder_mead_triangles_7_to_10_match_the_figure(k):
    _, history = nelder_mead(f, START)
    got = sorted(map(tuple, history[k]))
    want = sorted(FIGURE[k])
    assert np.array(got) == pytest.approx(np.array(want), abs=0.05)


def test_nelder_mead_finds_the_analytic_minimum():
    x, _ = nelder_mead(f, START)
    x_star, f_star = MINIMUM
    assert x == pytest.approx(x_star, abs=1e-4)
    assert f(x) == pytest.approx(f_star, abs=1e-9)


def test_nelder_mead_t6_tie_is_real_but_lost_to_round_off():
    """T6's two worst vertices both have f = -6.24; computed, they differ in the last bits.

    So a plain sort picks the worst by round-off, and the run leaves Figure 6.2 at T7.
    """
    _, history = nelder_mead(f, START)
    worst_two = [f(v) for v in history[5][1:]]
    assert worst_two == pytest.approx([-6.24, -6.24], abs=1e-12)
    assert worst_two[0] != worst_two[1]


def test_nelder_mead_needs_no_derivatives():
    """Section 6: only f is evaluated; it still solves the Rosenbrock function from fun3's start."""

    def rosenbrock(v):
        return 100 * (v[1] - v[0] ** 2) ** 2 + (1 - v[0]) ** 2

    x, _ = nelder_mead(rosenbrock, [(-1.2, 1.0), (-1.0, 1.0), (-1.2, 1.2)], tol=1e-14)
    assert x == pytest.approx([1, 1], abs=1e-4)


def test_nelder_mead_ties_are_relative_and_newest_first():
    """Values within a relative 1e-12 tie, whatever their size; the newer point ranks first."""
    for scale in (1e-3, 1.0, 1e3, 1e8):
        values = {0.0: scale, 1.0: scale * (1 + 1e-14), 2.0: scale * (1 + 1e-9)}

        def g(v, values=values):
            return values[float(v[0])]

        old, new, worse = np.array([0.0]), np.array([1.0]), np.array([2.0])
        ranked = _order(g, [new, old, worse])  # new first, as nelder_mead inserts it
        assert [float(v[0]) for v in ranked] == [1.0, 0.0, 2.0], scale
