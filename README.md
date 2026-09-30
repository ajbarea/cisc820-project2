# CISC-820 Project 2: Unconstrained Optimization

Group 1: AJ Barea, Hem Raj Pandeya, Jinting Liu, Max Frohman

Implement gradient descent, Newton, and quasi-Newton, apply each to three objective
functions, and analyze where each method is strong and where it breaks down. Graded on the
accuracy of the solutions and on the depth of the analysis, so the write-up counts as much
as the numbers.

The repository has two halves, and the course keeps them separate:

| | |
|---|---|
| `inside/` | The in-class project, defined by the instructor: three functions, three algorithms |
| `outside/` | The out-of-class project, a problem we bring ourselves on the same topic |

Nothing in `outside/` reads the instructor's data, and nothing in `inside/` depends on
whatever we choose for the outside half. Project 1 kept the in-class half loose at the
repository root; splitting both into named folders means one `pytest`, one CI run, and no
ambiguity about which half a file belongs to.

## Setup

```bash
python3 -m venv .venv          # 3.12 or newer
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                      # 38 tests, the shared foundation
```

Checked on 3.12 (CI) and 3.14 (numpy 2.5.3). If your `python3` is older than 3.12, name a
newer one explicitly: `python3.12 -m venv .venv`.

## The three functions

| | f(x) | n | Type | Start |
|---|---|---|---|---|
| fun1 | `sum(i * x_i^2)`, i = 1..100 | 100 | Convex quadratic, cond(H) = 100 | `ones(100)` |
| fun2 | `c'x - sum(log(b - Ax))`, m = 500 | 100 | Convex log barrier, +inf off its domain | `zeros(100)` |
| fun3 | `100(x2 - x1^2)^2 + (1 - x1)^2` | 2 | Rosenbrock, non-convex | `(-1.2, 1.0)` |

`fun1` is the only one whose optimum the instructor gives (f = 0 at the origin), so it is
the function the convergence curves are drawn on: error against the true minimum at each
iteration, all three algorithms on one plot. `fun3` is textbook Rosenbrock, minimum 0 at
(1, 1), which is useful for checking our own code even though the instructor is not
handing it to us.

All three are in `inside/functions.py` behind one `Problem` interface, so an optimizer
written against `(f, grad, hess, x0)` runs on every function without special-casing:

```python
from functions import PROBLEMS

p = PROBLEMS["fun2"]
p.f(p.x0), p.grad(p.x0), p.hess(p.x0)
```

## What is already here

The shared foundation, so nobody writes it three times. All of it is tested.

| File | Contents |
|---|---|
| `inside/derivatives.py` | Numerical gradient (order 1, 2, 4) and finite-difference Hessian, from the topic-04 exercises |
| `inside/functions.py` | The three functions, their analytic gradients and Hessians, and the `fun2` loader |
| `inside/data/` | `fun2_{A,b,c}.txt` as the instructor posted them |
| `inside/tests/` | Every analytic derivative checked against a finite difference of the function itself |

`derivatives.py` is what makes the tests worth anything: a sign slip or a transpose in an
analytic gradient shows up as a failing test rather than as an optimizer that silently
fails to converge. Use it the same way on the algorithms.

## What we still have to write

Three algorithms, one each, plus a driver. Nothing is stubbed, so pick a file and start.

| File | Method | Notes say |
|---|---|---|
| `inside/gradient_descent.py` | `p = -grad f(x)` | Backtracking line search, c = 0.1, rho = 1/2. Compare a fixed `alpha0 = 1` against the previous-step rule on p.18 |
| `inside/newton.py` | `H(x) p = -grad f(x)` | `alpha0 = 1` always. Needs H positive definite everywhere it is evaluated, which fails on Rosenbrock |
| `inside/quasi_newton.py` | BFGS, and DFP if we have room | Updates B (or its inverse) from the secant condition. Keep the update positive definite |
| `inside/main.py` | Driver | Every method on every function, into the solutions table and the fun1 convergence plot |

Two things are shared and should be written once, by whoever gets there first, rather than
three times:

- **Backtracking line search.** All three methods use it with the same c and rho, and only
  the initial `alpha0` differs. One `line_search.py` taking `alpha0` as an argument.
- **The stopping rule.** The notes say "until x is a good enough solution" and never fix a
  tolerance. Pick one convention (`norm(grad f) < tol`, plus an iteration cap) and have
  all three use it, or the iteration counts in our table are not comparable.

## Open questions, in the order they will bite us

1. **How `fun2_A.txt` gets folded into a matrix.** The file is 50,000 numbers on one line
   and the description points at a `ReadMe.txt` that MyCourses does not post. 500 x 100
   column-major (MATLAB's own order, `order="F"`) and 500 x 100 row-major both give a
   feasible problem at x = 0 and both converge. They converge to *different* minima, and
   the grade is partly the accuracy of that number. `load_fun2` defaults to column-major
   and takes `order` so we can switch in one place, and a test keeps the choice visible.
   Someone should ask the instructor for the `ReadMe.txt` before we commit to an answer.
2. **The description PDF is truncated.** It ends mid-sentence at "Please submit a table
   summarizing the solutions" on page 1. We know from the syllabus that a project is a
   technical report plus code plus a presentation, one zipped submission per team, but we
   do not have the instructor's actual deliverable list. Worth asking for alongside the
   `ReadMe.txt`.
3. **Three algorithms or four.** The description says "the 3 or 4 algorithms" when it
   describes the convergence curve. Nelder-Mead (topic-09) is the obvious fourth and needs
   no derivatives at all, which makes it a clean contrast. Optional, and only if the three
   required ones are finished.
4. **What the outside project is.** See below. This is the one with a real lead time,
   because it needs data and a problem before any of us can write code against it.

## The outside project

Same topic, our own problem. It has to be something where the three methods actually
behave differently, or the analysis has nothing to say. Candidates, roughly in order of
how much work they are:

- **Logistic regression by maximum likelihood** on a public classification dataset. Convex,
  the Hessian is `X' diag(p(1-p)) X`, and Newton here is exactly iteratively reweighted
  least squares. Scales from cheap to expensive just by picking a wider dataset, and it
  connects straight back to Project 1.
- **Nonlinear least squares** on a real curve fit, where Gauss-Newton and
  Levenberg-Marquardt drop out as modifications of what we already built.
- **A standard benchmark set.** The Rosenbrock family and the CUTEst unconstrained
  problems run from 2 to 1000 dimensions and are what the literature benchmarks against,
  so our numbers would sit beside published ones. More breadth, less story.
- **Minimizing a small neural network's loss**, which is non-convex and high-dimensional
  and is where gradient descent wins and Newton is unusable. Good contrast, most work.

Bring an opinion to the next meeting and we will pick one.

## Conventions

- `main` is protected by convention, not by settings. Each change starts as an issue with
  a label and an assignee. Branch, push, and open a pull request whose description says
  `Closes #<issue>`. Merging closes the issue and deletes the branch.
- `ruff check . && ruff format .` and `pytest -q` before pushing. CI runs both and reports
  lint without blocking a merge.
- Numbers in the report come from a script in the repository, never retyped from a
  terminal. If a number moves, it moves in one place.
- Every algorithm gets a test that pins it to something known: fun1 from any start, fun3
  from (-1.2, 1.0), Newton terminating in one step on a quadratic.
