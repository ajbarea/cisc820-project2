# CISC-820 Project 2: Unconstrained Optimization

Group 1: AJ Barea, Hem Raj Pandeya, Jinting Liu, Max Frohman

Implement gradient descent, Newton, and quasi-Newton, run each on three functions, and
analyze each.

## Setup
```bash
python3 -m venv .venv          # Python 3.12 or newer
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## The three functions

| | f(x) | n | Start |
|---|---|---|---|
| fun1 | `sum(i * x_i^2)`, i = 1..100 | 100 | `ones(100)` |
| fun2 | `c'x - sum(log(b - Ax))` | 100 | `zeros(100)` |
| fun3 | `100(x2 - x1^2)^2 + (1 - x1)^2` (Rosenbrock) | 2 | `(-1.2, 1.0)` |

They are already written, with their gradients and Hessians, in `inside/functions.py`.
Each one comes with everything an optimizer needs:

```python
from functions import PROBLEMS

p = PROBLEMS["fun1"]
p.f(p.x0), p.grad(p.x0), p.hess(p.x0)
```

## The methods

| File | Method |
|---|---|
| `inside/line_search.py` | Backtracking line search and the descent loop the methods share |
| `inside/gradient_descent.py` | Gradient descent |
| `inside/newton.py` | Newton |
| `inside/quasi_newton.py` | Quasi-Newton: BFGS |

`line_search.minimize(f, grad, x0, direction)` runs the loop for any method that supplies
`direction(x, g) -> p`. It stops once f changes by less than `tol` (1e-8 by default, which
reproduces the iteration counts in the notes' figures) and returns every iterate in
`Result.path`.

## Run every method

```bash
python inside/run.py                                # every method on every function
python inside/run.py -functions fun3 -methods bfgs  # a subset (methods: gd, newton, bfgs)
```

It prints one row per function and method: f at the solution, the gradient norm there,
the iterations and the time, and saves the plots to `inside/results/`:
`fun1_convergence.png` (f(x_k) - f_min, the convergence curve the project asks for, since
fun1 is the only function whose minimum is given) and `fun2_gradient_norm.png`,
`fun3_gradient_norm.png` (norm of grad f(x_k), which needs no minimum). `--tol` changes
the stopping rule (default 1e-12); `--no-plot` skips the figures.

| Function | Method | f(x*) | norm grad f(x*) | Iterations |
|---|---|---|---|---|
| fun1 | gd | 4.918e-11 | 3.66e-05 | 533 |
| fun1 | newton | 2.233e-62 | 2.70e-30 | 2 |
| fun1 | bfgs | 1.830e-17 | 5.35e-08 | 115 |
| fun2 | gd | -2443.247657 | 1.68e-04 | 6724 |
| fun2 | newton | -2443.247657 | 9.17e-11 | 11 |
| fun2 | bfgs | -2443.247657 | 2.91e-05 | 205 |
| fun3 | gd | 1.458e-09 | 9.76e-05 | 6277 |
| fun3 | newton | 7.682e-24 | 1.22e-10 | 21 |
| fun3 | bfgs | 2.746e-17 | 8.83e-08 | 34 |
