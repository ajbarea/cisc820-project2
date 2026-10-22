"""Run gradient descent, Newton and BFGS on the three project functions.

    python inside/run.py                                 # every method on every function
    python inside/run.py -functions fun3 -methods bfgs  # a subset
    python inside/run.py -functions fun3 --tol 1e-8      # the notes' stopping rule

Prints one row per function and method (f at the solution, the gradient norm there, the
iterations and the time) and saves one plot per function to ``inside/results/``.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from functions import PROBLEMS  # noqa: E402
from gradient_descent import gradient_descent  # noqa: E402
from newton import newton  # noqa: E402
from quasi_newton import bfgs  # noqa: E402

TOL = 1e-12
RESULTS = Path(__file__).resolve().parent / "results"

METHODS = {
    "gd": lambda p, **kw: gradient_descent(p.f, p.grad, p.x0, **kw),
    "newton": lambda p, **kw: newton(p.f, p.grad, p.hess, p.x0, **kw),
    "bfgs": lambda p, **kw: bfgs(p.f, p.grad, p.x0, **kw),
}


def solve_all(functions=None, methods=None, tol=TOL, max_iter=100_000):
    """{(function, method): (Result, seconds)} for every requested pair (default: all)."""
    out = {}
    for fname in functions or PROBLEMS:
        problem = PROBLEMS[fname]
        for mname, run in (methods or METHODS).items():
            t0 = time.perf_counter()
            # Some numpy builds on macOS raise spurious divide/overflow flags inside matmul.
            with np.errstate(all="ignore"):
                result = run(problem, tol=tol, max_iter=max_iter)
            out[fname, mname] = (result, time.perf_counter() - t0)
    return out


def table(results):
    """Markdown table: f(x*), norm of the gradient at x*, iterations, seconds."""
    header = ["Function", "Method", "f(x*)", "norm grad f(x*)", "Iterations", "Seconds"]
    rows = []
    for (fname, mname), (r, sec) in results.items():
        with np.errstate(all="ignore"):
            g = np.linalg.norm(PROBLEMS[fname].grad(r.x))
        rows.append(
            [fname, mname, f"{r.f:.10g}", f"{g:.2e}", str(r.iterations), f"{sec:.3f}"]
        )
    # Pad every column to its widest cell: text left-aligned, numbers right-aligned.
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    text_columns = 2  # Function and Method; the rest are numbers

    def line(cells):
        padded = [
            c.ljust(widths[i]) if i < text_columns else c.rjust(widths[i])
            for i, c in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |"

    dashes = [
        "-" * (w + 2) if i < text_columns else "-" * (w + 1) + "-"
        for i, w in enumerate(widths)
    ]
    rule = "|" + "|".join(dashes) + "|"
    return "\n".join([line(header), rule, *(line(row) for row in rows)])


def plot(results, directory=RESULTS):
    """One figure per function, one line per method, iteration k on the x-axis.

    The project gives the true minimum only for fun1, "so that, for this particular
    function, you can observe and show the convergence curve": fun1 gets f(x_k) - f_min.
    fun2 and fun3 have their optima withheld, so they get the gradient norm |grad f(x_k)|,
    which shows progress toward a stationary point without assuming a minimum.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(exist_ok=True)
    paths = []
    for fname in dict.fromkeys(f for f, _ in results):
        problem = PROBLEMS[fname]
        runs = {m: r for (f, m), (r, _) in results.items() if f == fname}
        fig, ax = plt.subplots(figsize=(6, 4))
        for mname, r in runs.items():
            if problem.minimum is not None:
                y = np.asarray(r.values) - problem.minimum
                y[y <= 0] = (
                    np.nan
                )  # f_min reached exactly: nothing to draw on a log axis
            else:
                with np.errstate(all="ignore"):
                    y = np.array([np.linalg.norm(problem.grad(x)) for x in r.path])
            ax.semilogy(y, label=f"{mname} ({r.iterations} it)")
        if problem.minimum is not None:
            ylabel, kind = (
                f"f(x_k) - f_min  (f_min = {problem.minimum:g})",
                "convergence",
            )
        else:
            ylabel, kind = "norm of grad f(x_k)", "gradient_norm"
        ax.set(xlabel="iteration k", ylabel=ylabel, title=f"{fname}: {problem.name}")
        ax.set_xscale("symlog", linthresh=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = directory / f"{fname}_{kind}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-functions",
        nargs="*",
        default=[],
        help=f"any of {', '.join(PROBLEMS)} (default: all)",
    )
    parser.add_argument(
        "-methods",
        nargs="*",
        default=[],
        help=f"any of {', '.join(METHODS)} (default: all)",
    )
    parser.add_argument(
        "--tol", type=float, default=TOL, help="stop when |f_k - f_(k-1)| < tol"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip saving convergence plots"
    )
    args = parser.parse_args()
    unknown_funcs = sorted(set(args.functions) - set(PROBLEMS))
    if unknown_funcs:
        parser.error(f"unknown function(s): {', '.join(unknown_funcs)}")
    unknown_methods = sorted(set(args.methods) - set(METHODS))
    if unknown_methods:
        parser.error(f"unknown method(s): {', '.join(unknown_methods)}")
    methods = {k: v for k, v in METHODS.items() if k in args.methods} or None

    print(f"Stopping rule: |f(x_k) - f(x_(k-1))| < {args.tol:g}\n")
    results = solve_all(args.functions or None, methods, args.tol)
    print(table(results))
    if not args.no_plot:
        print("\nSaved:", *plot(results), sep="\n  ")


if __name__ == "__main__":
    main()
