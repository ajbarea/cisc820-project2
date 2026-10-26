# Outside project

The out-of-class half: our own optimization problem, on the same topic as `inside/`.

We recover the parameters of a forced mass-spring-damper from its simulated motion,
by least squares over an ODE solve, and compare gradient descent, Newton and BFGS
on it ([#10](https://github.com/ajbarea/cisc820-project2/issues/10) has the other
candidates we considered).

| File | Contents |
|---|---|
| `system_identification_project.ipynb` | Hem's notebook: model, forward solve, objective, the three methods and their results |
| `mass_spring_damper_intro.gif` | Animation of the system, for the report and slides introduction |

Data for it can live in `outside/data/`, and tests in `outside/tests/` run in CI with the rest.
