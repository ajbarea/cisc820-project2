"""Run every method on every project function and print the solutions table.

    python inside/run.py                  # all three functions
    python inside/run.py fun3 --tol 1e-8  # one function, the notes' stopping rule

Each row gives f at the solution, the gradient norm there, the iterations and the time.
The default tolerance is 1e-12, because the notes' 1e-8 stops gradient descent early on
fun3 (Rosenbrock), long before it reaches the minimum.
"""

from __future__ import annotations

import os

# Pin BLAS to one thread before numpy loads: at n = 100 threads only contend, and
# unpinned runs of BFGS on fun2 swing from a fraction of a second to many seconds.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from functions import PROBLEMS  # noqa: E402
from gradient_descent import gradient_descent  # noqa: E402
from newton import newton  # noqa: E402
from quasi_newton import bfgs  # noqa: E402

TOL = 1e-12

METHODS = {
    "gd": lambda p, **kw: gradient_descent(p.f, p.grad, p.x0, **kw),
    "newton": lambda p, **kw: newton(p.f, p.grad, p.hess, p.x0, **kw),
    "bfgs": lambda p, **kw: bfgs(p.f, p.grad, p.x0, **kw),
}


def solve_all(functions=None, methods=None, tol=TOL, max_iter=100_000):
    """{(function, method): (Result, seconds)} for every requested pair."""
    out = {}
    for fname in functions or PROBLEMS:
        problem = PROBLEMS[fname]
        for mname, run in methods.items():
            t0 = time.perf_counter()
            result = run(problem, tol=tol, max_iter=max_iter)
            out[fname, mname] = (result, time.perf_counter() - t0)
    return out


def table(results):
    """Markdown table: f(x*), norm of the gradient at x*, iterations, seconds."""
    lines = [
        "| Function | Method |       f(x*)       | norm grad f(x*) | Iterations | Seconds |",
        "|----------|--------|-------------------|-----------------|------------|---------|",
    ]
    for (fname, mname), (r, sec) in results.items():
        g = np.linalg.norm(PROBLEMS[fname].grad(r.x))
        lines.append(f"| {fname:<8} | {mname:<6} | {r.f:>17.10g} | {g:>15.2e} | {r.iterations:>10} | {sec:>7.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-functions", nargs="*", help=f"any of {', '.join(PROBLEMS)} (default: all)")
    parser.add_argument("-methods", nargs="*", help=f"any of {', '.join(METHODS.keys())} (default: all)", default=list(METHODS.keys()))
    parser.add_argument("--tol", type=float, default=TOL, help="stop when |f_k - f_(k-1)| < tol")
    args = parser.parse_args()
    unknown_funcs = sorted(set(args.functions) - set(PROBLEMS))
    if unknown_funcs:
        parser.error(f"unknown function(s): {', '.join(unknown_funcs)}")
    unknown_methods = sorted(set(args.methods) - set(METHODS.keys()))
    if unknown_methods:
        parser.error(f"unknown method(s): {', '.join(unknown_methods)}")
    
    methods = {k: v for k, v in METHODS.items() if k in args.methods}
    
    print(f"Stopping rule: |f(x_k) - f(x_(k-1))| < {args.tol:g}\n")
    print(table(solve_all(args.functions or None, methods, args.tol)))


if __name__ == "__main__":
    main()
