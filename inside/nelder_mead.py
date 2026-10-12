"""Nelder-Mead simplex method (notes section 6, p.31), steps 1 to 6 exactly as listed.

    alpha = 1 reflection, gamma = 2 expansion, rho = 1/2 contraction, sigma = 1/2 reduction

The notes give no stopping rule; this stops once the simplex's f values span less than tol.
Nor do they break ties in step 1. Figure 6.2 needs one at T6, where (3.6, 1.6) and
(2.4, 2.4) both have f = -6.24 exactly; it treats the older vertex as worse. Computed,
the two differ in the last bits, so values within a relative 1e-12 count as tied, and
within a tie the more recently accepted point ranks better.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nelder_mead"]

ALPHA, GAMMA, RHO, SIGMA = 1.0, 2.0, 0.5, 0.5
TOL = 1e-10
TIE = 1e-12  # relative: f values this close are ties, not round-off winners
MAX_ITER = 10_000


def _order(f, pts):
    """Sort by f; runs of values within TIE of each other keep their incoming order."""
    ranked = sorted(pts, key=f)  # stable
    groups = [[ranked[0]]]
    for p in ranked[1:]:
        f0 = f(groups[-1][0])
        if abs(f(p) - f0) <= TIE * max(1.0, abs(f0)):
            groups[-1].append(p)
        else:
            groups.append([p])
    order = {id(p): i for i, p in enumerate(pts)}
    return [p for g in groups for p in sorted(g, key=lambda p: order[id(p)])]


def nelder_mead(
    f, simplex, alpha=ALPHA, gamma=GAMMA, rho=RHO, sigma=SIGMA, tol=TOL, max_iter=MAX_ITER
):
    """Minimise f from an initial simplex of n + 1 points. Returns (best x, every simplex).

    ``history[k]`` is the simplex after k iterations, ordered best first.
    """
    pts = [np.asarray(v, dtype=np.float64) for v in simplex]
    history = []
    for _ in range(max_iter + 1):
        pts = _order(f, pts)  # 1. ordering
        history.append(np.array(pts))
        if f(pts[-1]) - f(pts[0]) < tol:
            break
        best, second_worst, worst = pts[0], pts[-2], pts[-1]
        xo = np.mean(pts[:-1], axis=0)  # 2. midpoint of all but the worst
        xr = xo + alpha * (xo - worst)  # 3. reflection
        new = None
        if f(best) <= f(xr) < f(second_worst):
            new = xr
        elif f(xr) < f(best):  # 4. expansion
            xe = xo + gamma * (xo - worst)
            new = xe if f(xe) < f(xr) else xr
        else:  # 5. contraction, f(xr) >= f(x_n)
            xc = worst + rho * (xo - worst)
            if f(xc) < f(worst):
                new = xc
            else:  # 6. reduction toward the best point
                pts = [best] + [best + sigma * (p - best) for p in pts[1:]]
        if new is not None:
            pts = [new] + pts[:-1]  # newest first: _order keeps that order inside a tie
    return pts[0], history
