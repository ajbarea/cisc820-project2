# CISC-820 Project 2: Unconstrained Optimization

Group 1: AJ Barea, Hem Raj Pandeya, Jinting Liu, Max Frohman

Implement gradient descent, Newton, and quasi-Newton, run each on three functions, and
analyze where each method works and where it breaks down.

| Folder | What goes there |
|---|---|
| `inside/` | The in-class project: the instructor's three functions |
| `outside/` | The out-of-class project: a problem we pick ourselves |

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

`inside/derivatives.py` has finite-difference gradients and Hessians, handy for checking
your own derivatives.

## The methods

| File | Method |
|---|---|
| `inside/line_search.py` | Backtracking line search and the descent loop the methods share |
| `inside/gradient_descent.py` | Gradient descent, alpha0 = 1 or p.18's previous-step rule |
| `inside/newton.py` | Newton, with p.24's Greenstadt modification as an option (`modified=True`) |
| `inside/quasi_newton.py` | BFGS (`inverse=True` for p.28's inverse form) and DFP |

`line_search.minimize(f, grad, x0, direction)` runs the loop for any method that supplies
`direction(x, g) -> p`. It stops once f changes by less than `tol` (1e-8 by default, which
reproduces the iteration counts in the notes' figures) and returns every iterate in
`Result.path`.

## What's left

Everything still to do is an [issue](https://github.com/ajbarea/cisc820-project2/issues).

- [#10](https://github.com/ajbarea/cisc820-project2/issues/10) Choose the outside project
- [#11](https://github.com/ajbarea/cisc820-project2/issues/11) Nelder-Mead (optional)

CI runs the tests on every push.
