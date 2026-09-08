"""Minimal usage of the `iconic` native module (build with `maturin develop`).

This is the native surface a CVXPY backend class calls into: the canonical conic
form min ½xᵀPx + qᵀx s.t. Ax + s = b, s ∈ K, with cones given as (kind, dim) pairs.
"""

import math
import iconic

# Quadratic program with an equality (Zero cone) and an inequality (Nonnegative):
#   min ½(x0² + x1²)  s.t.  x0 + x1 = 2,  x0 ≤ 1.5   ->  x = [1, 1]
sol = iconic.solve(
    p=[[1.0, 0.0], [0.0, 1.0]],
    q=[0.0, 0.0],
    a=[[1.0, 1.0], [1.0, 0.0]],
    b=[2.0, 1.5],
    cones=[("zero", 1), ("nonneg", 1)],
)
print("QP  ", sol.status, "x =", [round(v, 4) for v in sol.x])

# Second-order cone program: project c = (1, 2, 2) onto Q3.
sol = iconic.solve(
    p=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    q=[-1.0, -2.0, -2.0],
    a=[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    b=[0.0, 0.0, 0.0],
    cones=[("soc", 3)],
)
print("SOCP", sol.status, "x0 =", round(sol.x[0], 5),
      "(closed form", round((1 + 2 * math.sqrt(2)) / 2, 5), ")")
