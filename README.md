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

## What's left

Everything still to do is an [issue](https://github.com/ajbarea/cisc820-project2/issues).

- [#5](https://github.com/ajbarea/cisc820-project2/issues/5) Shared line search and stopping rule (optional)
- [#6](https://github.com/ajbarea/cisc820-project2/issues/6) Gradient descent
- [#7](https://github.com/ajbarea/cisc820-project2/issues/7) Newton's method
- [#8](https://github.com/ajbarea/cisc820-project2/issues/8) Quasi-Newton (BFGS)
- [#9](https://github.com/ajbarea/cisc820-project2/issues/9) Ask the instructor for `ReadMe.txt` and the full description
- [#10](https://github.com/ajbarea/cisc820-project2/issues/10) Choose the outside project
- [#11](https://github.com/ajbarea/cisc820-project2/issues/11) Nelder-Mead (optional)

CI runs the tests on every push.
