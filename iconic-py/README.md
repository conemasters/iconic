# iconicpy

**iconicpy** is a from-scratch convex optimization and mixed-integer programming
solver written in Rust, exposed to Python with a [CVXPY](https://www.cvxpy.org/)
backend. It solves LP, QP, SOCP, SDP, exponential-cone, and MIP problems in the
canonical conic standard form via a regularized interior-point method.

> The distribution is named **`iconicpy`**; the import name is **`iconic`**
> (like `scikit-learn` → `sklearn`).

## Install

```sh
pip install iconicpy            # core solver
pip install iconicpy[cvxpy]     # + CVXPY backend
```

## Use

```python
import iconic

sol = iconic.solve(
    p=[[2.0, 0.0], [0.0, 2.0]], q=[-2.0, -3.0],
    a=[[1.0, 1.0]], b=[1.0], cones=[("zero", 1)],
)
print(sol.status, sol.obj_val, sol.x)
```

With CVXPY:

```python
import cvxpy as cp
from iconic._backend import register

register()
prob.solve(solver="ICONIC")     # LP / QP / SOCP / SDP / EXP / MIP
```

See `ARCHITECTURE.md` in the repository for the full
solver design, benchmarks, and architecture.

## License

Apache-2.0.
