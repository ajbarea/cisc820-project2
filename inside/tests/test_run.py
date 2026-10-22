"""``run.py``: every method on every function, one table row per pair."""

from __future__ import annotations

import numpy as np

from functions import PROBLEMS
from run import METHODS, solve_all, table


def test_every_method_reaches_the_rosenbrock_minimum():
    results = solve_all(["fun3"])
    assert set(results) == {("fun3", m) for m in METHODS}
    for (_, method), (r, _) in results.items():
        assert np.allclose(r.x, [1.0, 1.0], atol=1e-3), method
        assert r.f < 1e-6, method


def test_table_has_one_row_per_pair():
    results = solve_all(["fun1"])
    rows = table(results).splitlines()[2:]
    assert len(rows) == len(METHODS)
    assert all(row.startswith("| fun1 |") for row in rows)
    assert PROBLEMS["fun1"].minimum == 0.0
