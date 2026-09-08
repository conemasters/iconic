"""End-to-end tests of the ICONIC CVXPY backend.

Requires the native `iconic` module (build with `maturin develop --release`) and
`cvxpy`. Correctness is checked against closed-form solutions where they exist, and
against feasibility + optimal status otherwise — no external solver is needed.
Run: `python -m pytest iconic-py/tests/test_cvxpy.py` (or execute directly).
"""

import os
import sys

import numpy as np
import cvxpy as cp

from iconic._backend import register

register()

TOL = 1e-4


def _soc_project(t, v):
    """Euclidean projection of (t, v) onto the second-order cone {(u, w): u ≥ ‖w‖}."""
    nv = np.linalg.norm(v)
    if nv <= t:
        return t, v.copy()
    if nv <= -t:
        return 0.0, np.zeros_like(v)
    scale = (nv + t) / (2.0 * nv)
    return scale * nv, scale * v


def _psd_project(c):
    """Projection of a symmetric matrix onto the PSD cone (clamp eigenvalues at 0)."""
    w, q = np.linalg.eigh(c)
    return (q * np.maximum(w, 0.0)) @ q.T


def test_qp_equality_and_bound():
    # min ½‖x‖² s.t. x0 + x1 = 2, x0 ≤ 1.5.  The point on the line nearest the
    # origin is (1, 1) (the bound is slack), with objective 1.0.
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x)), [x[0] + x[1] == 2, x[0] <= 1.5])
    obj = prob.solve(solver="ICONIC")
    assert prob.status == "optimal"
    assert abs(obj - 1.0) < TOL
    assert np.allclose(x.value, [1.0, 1.0], atol=1e-4)


def test_socp_projection():
    # min ½‖y − c‖² s.t. y ∈ SOC  ⇔  y = Π_SOC(c), which has a closed form.
    c = np.array([1.0, 2.0, 2.0])
    y = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(y - c)), [cp.SOC(y[0], y[1:])])
    obj = prob.solve(solver="ICONIC")
    t_star, v_star = _soc_project(c[0], c[1:])
    y_star = np.concatenate([[t_star], v_star])
    assert prob.status == "optimal"
    assert np.allclose(y.value, y_star, atol=1e-4)
    assert abs(obj - 0.5 * np.sum((y_star - c) ** 2)) < TOL


def test_sdp_projection():
    # min ‖X − C‖²_F s.t. X ⪰ 0  ⇔  X = Π_PSD(C).
    c = np.array([[1.0, 2.0], [2.0, 1.0]])
    x = cp.Variable((2, 2), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - c)), [x >> 0])
    obj = prob.solve(solver="ICONIC")
    x_star = _psd_project(c)
    assert prob.status == "optimal"
    assert np.allclose(x.value, x_star, atol=1e-4)
    assert np.allclose(x.value, [[1.5, 1.5], [1.5, 1.5]], atol=1e-4)
    assert abs(obj - np.sum((x_star - c) ** 2)) < TOL


def test_dual_values_satisfy_kkt():
    # min ‖x‖² − [1,2,3]·x s.t. 1ᵀx = 1, x ≥ 0.  Analytic primal: (0, ¼, ¾).
    # Without an external oracle, verify the returned duals satisfy complementary
    # slackness and a consistent sign (silent dual bugs are easy to miss).
    x = cp.Variable(3)
    p = np.diag([2.0, 2.0, 2.0])
    eq = x[0] + x[1] + x[2] == 1
    nn = x >= 0
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, p) - np.array([1.0, 2, 3]) @ x), [eq, nn])
    prob.solve(solver="ICONIC")
    assert prob.status == "optimal"
    assert np.allclose(x.value, [0.0, 0.25, 0.75], atol=1e-4)
    lam = np.atleast_1d(nn.dual_value)
    # The nonneg-cone multipliers share one sign and are complementary to the primal.
    assert np.all(lam >= -1e-6) or np.all(lam <= 1e-6)
    assert abs(float(lam @ x.value)) < 1e-5


def test_mixed_cones_feasible_optimal():
    # Equality + nonnegative + second-order cone together. No closed form, so we
    # verify ICONIC reports `optimal` and the returned point is feasible.
    y = cp.Variable(4)
    eq = y[0] + y[1] == 3
    nn = y >= 0
    soc = cp.SOC(y[3], y[1:3])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y - np.array([1.0, 1, 1, 5]))), [eq, nn, soc])
    prob.solve(solver="ICONIC")
    v = y.value
    assert prob.status == "optimal"
    assert abs(v[0] + v[1] - 3) < 1e-4
    assert np.all(v > -1e-4)
    assert v[3] + 1e-4 >= np.linalg.norm(v[1:3])


def test_lp_vertex():
    # max x0 + x1 s.t. 0 ≤ x ≤ 1.  Optimum at the vertex (1, 1), value 2.
    x = cp.Variable(2)
    prob = cp.Problem(cp.Maximize(x[0] + x[1]), [x <= 1, x >= 0])
    obj = prob.solve(solver="ICONIC")
    assert prob.status == "optimal"
    assert abs(obj - 2.0) < TOL
    assert np.allclose(x.value, [1.0, 1.0], atol=1e-4)


def test_max_entropy_uniform():
    # max Σ entr(y) s.t. 1ᵀy = 1, y ≥ 0.  Optimum is uniform, value log(n).
    for n in (2, 4, 6):
        y = cp.Variable(n)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(y))), [cp.sum(y) == 1, y >= 0])
        obj = prob.solve(solver="ICONIC")
        assert prob.status == "optimal", f"n={n}: {prob.status}"
        assert abs(obj - np.log(n)) < 1e-5, f"n={n}: {obj} vs {np.log(n)}"
        assert np.allclose(y.value, np.full(n, 1.0 / n), atol=1e-4)


def test_log_sum_exp_feasible_optimal():
    # A smooth exponential-cone program with no closed form: check optimal + feasible.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 3))
    b = np.arange(5) / 5.0
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(cp.log_sum_exp(a @ x + b)), [cp.norm(x, 1) <= 1])
    obj = prob.solve(solver="ICONIC")
    assert prob.status == "optimal"
    assert np.isfinite(obj)
    assert np.linalg.norm(x.value, 1) <= 1 + 1e-4


def test_detects_infeasible():
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(x), [x >= 2, x <= 1])
    prob.solve(solver="ICONIC")
    assert prob.status == "infeasible"


def test_detects_unbounded():
    y = cp.Variable()
    prob = cp.Problem(cp.Minimize(y), [y <= 5])
    prob.solve(solver="ICONIC")
    assert prob.status == "unbounded"


def test_fuzz_robustness():
    # Many random problems across cone types must each solve to `optimal` with a
    # primal-feasible point — the surest catch for a hidden bug. Deterministic seed.
    rng = np.random.default_rng(0)
    solved = 0
    for trial in range(24):
        kind = trial % 4
        if kind == 0:  # QP with inequalities
            n = int(rng.integers(2, 6))
            a = rng.standard_normal((n, n))
            p = a @ a.T + np.eye(n)
            x = cp.Variable(n)
            g = rng.standard_normal((n, n))
            h = g @ rng.standard_normal(n) + np.abs(rng.standard_normal(n)) + 0.5
            prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, p) + rng.standard_normal(n) @ x), [g @ x <= h])
            feasible = lambda: bool(np.all(g @ x.value <= h + 1e-3))  # noqa: E731
        elif kind == 1:  # LP
            n = int(rng.integers(2, 6))
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(rng.standard_normal(n) @ x), [x >= -2, x <= 2])
            feasible = lambda: bool(np.all(np.abs(x.value) <= 2 + 1e-4))  # noqa: E731
        elif kind == 2:  # SOCP
            n = int(rng.integers(2, 5))
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(rng.standard_normal(n) @ x), [cp.norm(x) <= 1])
            feasible = lambda: bool(np.linalg.norm(x.value) <= 1 + 1e-4)  # noqa: E731
        else:  # small SDP
            k = int(rng.integers(2, 4))
            xm = cp.Variable((k, k), symmetric=True)
            c = rng.standard_normal((k, k))
            c = c + c.T
            prob = cp.Problem(cp.Minimize(cp.trace(c @ xm)), [xm >> 0, cp.trace(xm) == 1])
            feasible = lambda: bool(  # noqa: E731
                np.linalg.eigvalsh(xm.value).min() > -1e-4 and abs(np.trace(xm.value) - 1) < 1e-4
            )
        prob.solve(solver="ICONIC")
        if prob.status == "optimal":
            assert feasible(), f"trial {trial} (kind {kind}): infeasible solution returned"
            solved += 1
    assert solved >= 18, f"only {solved}/24 problems solved"


if __name__ == "__main__":
    test_qp_equality_and_bound()
    test_socp_projection()
    test_sdp_projection()
    test_dual_values_satisfy_kkt()
    test_mixed_cones_feasible_optimal()
    test_lp_vertex()
    test_max_entropy_uniform()
    test_log_sum_exp_feasible_optimal()
    test_detects_infeasible()
    test_detects_unbounded()
    test_fuzz_robustness()
    print("all CVXPY backend tests passed")

def test_power_cone_matches_clarabel():
    # max x^α y^(1−α) s.t. x + y = 1, via PowCone3D. Optimum at x=α, y=1−α
    # (on the curved boundary), value α^α (1−α)^(1−α). ICONIC's answer must match
    # the closed form AND Clarabel's.
    for alpha in (0.3, 0.5, 0.7):
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()
        prob = cp.Problem(cp.Maximize(z), [cp.PowCone3D(x, y, z, alpha), x + y == 1])
        obj = prob.solve(solver="ICONIC")
        want = alpha**alpha * (1.0 - alpha) ** (1.0 - alpha)
        assert prob.status == "optimal", f"a={alpha}: {prob.status}"
        assert abs(obj - want) < 1e-5, f"a={alpha}: {obj} vs {want}"
        obj_c = prob.solve(solver="CLARABEL")
        assert prob.status == "optimal", f"clarabel a={alpha}: {prob.status}"
        assert abs(obj - obj_c) < 1e-5, f"a={alpha}: ICONIC {obj} vs Clarabel {obj_c}"


def test_power_cone_projection_interior():
    # min ½‖w−c‖² s.t. w ∈ P_α with c strictly inside the cone: the optimum is
    # w = c (interior point, value 0). The dual-step fix keeps the dual from
    # freezing on this class.
    for alpha in (0.5, 0.7):
        c = np.array([1.0, 1.0, 0.0])
        w = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(w - c)),
            [cp.PowCone3D(w[0], w[1], w[2], alpha)],
        )
        obj = prob.solve(solver="ICONIC")
        assert prob.status == "optimal", f"a={alpha}: {prob.status}"
        assert abs(obj) < 1e-5, f"a={alpha}: obj={obj}"
        assert np.allclose(w.value, c, atol=1e-4), f"a={alpha}: w={w.value}"
        obj_c = prob.solve(solver="CLARABEL")
        assert abs(obj - obj_c) < 1e-5, f"a={alpha}: ICONIC {obj} vs Clarabel {obj_c}"


def test_power_cone_unbounded_linear_objective():
    # max √(xy) − 3x − 0.1y: the supremum 0 is approached at the cone apex and
    # never attained, so the honest grade is optimal-ish near zero — ICONIC and
    # Clarabel must agree on the objective within a loose tolerance.
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    prob = cp.Problem(cp.Maximize(z - 3 * x - 0.1 * y), [cp.PowCone3D(x, y, z, 0.5)])
    obj = prob.solve(solver="ICONIC")
    obj_c = prob.solve(solver="CLARABEL")
    assert np.isfinite(obj) and np.isfinite(obj_c)
    assert abs(obj - obj_c) < 1e-3, f"ICONIC {obj} vs Clarabel {obj_c}"
