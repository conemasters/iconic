"""CVXPY backend for ICONIC (``iconic._backend``).

Registers ICONIC as a CVXPY *conic* solver, *QP* solver, and *MIP* solver.
The heavy lifting is done by the native ``libiconic_c`` shared library, called
through ctypes for zero-copy numpy ↔ Rust data transfer.

Usage::

    import cvxpy as cp
    from iconic._backend import register
    register()
    prob.solve(solver="ICONIC")
"""

import ctypes as _ct
import os as _os
import sys
import types

import numpy as np

# -- cvxpy import -------------------------------------------------------
# Requires CVXPY ≥ 1.7.
try:
    import cvxpy.settings as s
    from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
    from cvxpy.reductions.solution import Solution, failure_solution
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
except ImportError:
    raise ImportError(
        "cvxpy ≥ 1.7 is required for the ICONIC CVXPY backend. "
        "Install: pip install iconic[cvxpy]"
    )

# -- load the native C library ------------------------------------------
# Searches the package directory for libiconic_c.{so,dylib,dll} (built from
# the iconic-c crate).  In a wheel install this is bundled alongside this file;
# in dev it's built separately: `cargo build -p iconic-c --release`.
_here = _os.path.dirname(_os.path.abspath(__file__))
_c_api = None
for _name in ("libiconic_c.so", "libiconic_c.dylib", "iconic_c.dll"):
    _path = _os.path.join(_here, _name)
    if _os.path.exists(_path):
        try:
            _c_api = _ct.CDLL(_path)
            break
        except OSError:
            continue

if _c_api is None:
    raise RuntimeError(
        "libiconic_c not found next to _backend.py — "
        "build it with: cargo build -p iconic-c --release"
    )

class _IconicSettings(_ct.Structure):
    """C mirror of iconic-c's versioned `IconicSettings` struct.

    Sentinel values mean "use the default": `0` for ``max_iters`` (the
    C-layer default is 800) and the ``eps_*`` fields, `-1` for the int32
    switches (``presolve``/``hsd``/``sparse_kkt``), `0` for
    ``blas_threads``.  Layout and field order must match iconic-c's
    `#[repr(C)] IconicSettings` exactly.
    """

    _fields_ = [
        ("version", _ct.c_int32),
        ("max_iters", _ct.c_int32),
        ("eps_abs", _ct.c_double),
        ("eps_rel", _ct.c_double),
        ("eps_gap", _ct.c_double),
        ("presolve", _ct.c_int32),
        ("hsd", _ct.c_int32),
        ("sparse_kkt", _ct.c_int32),
        ("blas_threads", _ct.c_int32),
    ]


def _build_iconic_settings(settings):
    """Build the C ``IconicSettings`` struct from a parsed solver_opts dict.

    The backend always passes explicit values (defaults included) rather
    than sentinels, so the CVXPY-facing defaults are the documented ones in
    ICONIC_DEFAULTS (e.g. max_iters 200) — independent of the C layer's NULL
    default of 800.  Fields the backend does not expose (sparse_kkt,
    blas_threads) stay sentinel (auto).
    """
    cs = _IconicSettings()
    cs.version = 1
    cs.max_iters = int(settings.get("max_iters", ICONIC_DEFAULTS["max_iters"]))
    cs.eps_abs = float(settings.get("eps_abs", ICONIC_DEFAULTS["eps_abs"]))
    cs.eps_rel = float(settings.get("eps_rel", ICONIC_DEFAULTS["eps_rel"]))
    cs.eps_gap = float(settings.get("eps_gap", ICONIC_DEFAULTS["eps_gap"]))
    cs.presolve = 1 if settings.get("presolve", True) else 0
    cs.hsd = 1 if settings.get("hsd", True) else 0
    cs.sparse_kkt = -1
    cs.blas_threads = 0
    return cs


# iconic_solve (CSC path)
_c_api.iconic_solve.argtypes = [
    _ct.c_int, _ct.c_int,  # n, m
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_double),  # P CSC
    _ct.POINTER(_ct.c_double),  # q
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_double),  # A CSC
    _ct.POINTER(_ct.c_double),  # b
    _ct.POINTER(_ct.c_char_p), _ct.c_int, _ct.c_int,  # cones, n_cones, presolve
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # x, s, z
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),  # obj, iters, status
]
_c_api.iconic_solve.restype = _ct.c_int

# iconic_solve_dense (dense row-major path)
_c_api.iconic_solve_dense.argtypes = [
    _ct.c_int, _ct.c_int,  # n, m
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # P, q
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # A, b
    _ct.POINTER(_ct.c_char_p), _ct.c_int, _ct.c_int,  # cones, n_cones, presolve
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # x, s, z
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),  # obj, iters, status
]
_c_api.iconic_solve_dense.restype = _ct.c_int

# iconic_solve_with_settings (CSC path + trailing IconicSettings*)
_c_api.iconic_solve_with_settings.argtypes = [
    _ct.c_int, _ct.c_int,  # n, m
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_double),  # P CSC
    _ct.POINTER(_ct.c_double),  # q
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_double),  # A CSC
    _ct.POINTER(_ct.c_double),  # b
    _ct.POINTER(_ct.c_char_p), _ct.c_int, _ct.c_int,  # cones, n_cones, presolve
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # x, s, z
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),  # obj, iters, status
    _ct.POINTER(_IconicSettings),  # settings (NULL = legacy)
]
_c_api.iconic_solve_with_settings.restype = _ct.c_int

# iconic_solve_dense_with_settings (dense row-major path + trailing IconicSettings*)
_c_api.iconic_solve_dense_with_settings.argtypes = [
    _ct.c_int, _ct.c_int,  # n, m
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # P, q
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # A, b
    _ct.POINTER(_ct.c_char_p), _ct.c_int, _ct.c_int,  # cones, n_cones, presolve
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # x, s, z
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),  # obj, iters, status
    _ct.POINTER(_IconicSettings),  # settings (NULL = legacy)
]
_c_api.iconic_solve_dense_with_settings.restype = _ct.c_int

# iconic_solve_mip_csc (sparse CSC MIP path — zero-copy) — optional
try:
    _c_api.iconic_solve_mip_csc.argtypes = [
    _ct.c_int, _ct.c_int,  # n, m
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_double),  # P CSC
    _ct.POINTER(_ct.c_double),  # q
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_double),  # A CSC
    _ct.POINTER(_ct.c_double),  # b
    _ct.POINTER(_ct.c_char_p), _ct.c_int,  # cones, n_cones
    _ct.POINTER(_ct.c_int),  # var_types
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # lb, ub
    _ct.c_int, _ct.c_double, _ct.c_double, _ct.c_double,  # max_nodes, max_time, abs_gap, rel_gap
    _ct.POINTER(_ct.c_double),  # x_out
    _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double), _ct.POINTER(_ct.c_double),  # obj, best_bound, gap
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),  # nodes, simplex_iters, status
]
    _c_api.iconic_solve_mip_csc.restype = _ct.c_int
except AttributeError:
    pass  # iconic_solve_mip_csc not yet available in this build

_c_api.iconic_version.restype = _ct.c_char_p

_STATUS_TO_STR = {
    0: "Solved", 1: "SolvedInaccurate", 2: "PrimalInfeasible",
    3: "DualInfeasible", 4: "MaxIterations", 5: "TimeLimit",
    6: "NumericalError", -1: "Unsolved",
}

_MIP_STATUS_TO_STR = {
    10: "Optimal", 11: "Feasible", 12: "Infeasible",
    13: "Unbounded", 14: "NodeLimit", 15: "TimeLimit",
    16: "NoSolution", 17: "NumericalError", -1: "Unsolved",
}

_MIP_STATUS_TO_CVXPY = {
    "Optimal": s.OPTIMAL, "Feasible": s.OPTIMAL_INACCURATE,
    "Infeasible": s.INFEASIBLE, "Unbounded": s.UNBOUNDED,
    "NodeLimit": s.USER_LIMIT, "TimeLimit": s.USER_LIMIT,
    "NoSolution": s.SOLVER_ERROR, "NumericalError": s.SOLVER_ERROR,
    "Unsolved": s.SOLVER_ERROR,
}


def _call_c_dense(n, m, p_dense, q, a_dense, b, cones, presolve, settings=None):
    """Call iconic_solve_dense_with_settings via ctypes. All arrays must be
    contiguous f64. `settings` is the parsed solver_opts dict (None = the
    documented defaults from ICONIC_DEFAULTS)."""
    import time
    t0 = time.perf_counter()

    cone_bytes = [f"{k}:{d}".encode() for k, d in cones]
    cone_ptrs = (_ct.c_char_p * len(cone_bytes))()
    for i, cb in enumerate(cone_bytes):
        cone_ptrs[i] = cb

    cs = _build_iconic_settings(settings or ICONIC_DEFAULTS)

    x = np.empty(n, dtype=np.float64)
    s_out = np.empty(m, dtype=np.float64)
    z = np.empty(m, dtype=np.float64)
    obj_val = _ct.c_double()
    iters = _ct.c_int()
    status = _ct.c_int()

    ret = _c_api.iconic_solve_dense_with_settings(
        _ct.c_int(n), _ct.c_int(m),
        np.ascontiguousarray(p_dense, dtype=np.float64).ctypes.data_as(_ct.POINTER(_ct.c_double)) if p_dense is not None else None,
        np.ascontiguousarray(q, dtype=np.float64).ctypes.data_as(_ct.POINTER(_ct.c_double)),
        np.ascontiguousarray(a_dense, dtype=np.float64).ctypes.data_as(_ct.POINTER(_ct.c_double)),
        np.ascontiguousarray(b, dtype=np.float64).ctypes.data_as(_ct.POINTER(_ct.c_double)),
        cone_ptrs, _ct.c_int(len(cones)), _ct.c_int(1 if presolve else 0),
        x.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        s_out.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        z.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        _ct.byref(obj_val), _ct.byref(iters), _ct.byref(status),
        _ct.byref(cs),
    )
    if ret != 0:
        if _os.environ.get("ICONIC_DEBUG"):
            print("DEBUG n=", n, "m=", m, "cones=", cones, "b=", b, "a_shape=", a_dense.shape if a_dense is not None else None)
        raise RuntimeError(f"iconic_solve_dense_with_settings returned {ret}")

    return types.SimpleNamespace(
        x=x, z=z,
        status=_STATUS_TO_STR.get(status.value, "Unsolved"),
        obj_val=obj_val.value,
        iterations=iters.value,
        solve_time=time.perf_counter() - t0,
    )


def _iconic_solve(P, A, q, b, cones, presolve, n, m, settings=None):
    """Call the native solver through the zero-copy C ABI (ctypes).

    `settings` is the parsed solver_opts dict (None = the documented defaults
    from ICONIC_DEFAULTS)."""
    import time
    import scipy.sparse as sp

    f64 = lambda v: np.ascontiguousarray(v, dtype=np.float64)
    t0 = time.perf_counter()

    # Dense path: P and A are already numpy arrays (caller did todense).
    if isinstance(P, np.ndarray) and isinstance(A, np.ndarray):
        raw = _call_c_dense(n, m, P, q, A, b, cones, presolve, settings)
        raw.solve_time = time.perf_counter() - t0
        return raw

    # Small-problem fast path: for n < 64, convert CSC to dense in Python (numpy
    # todense is ~1µs for tiny matrices) and use the dense C ABI.  This skips the
    # Rust-side CSC→dense conversion and all sparse-path overhead (density proxy,
    # AMD ordering, symbolic analysis) which dominates small solves.
    if n < 64:
        Pd = P.toarray().ravel() if P is not None else np.zeros(n * n, dtype=np.float64)
        Ad = A.toarray().ravel() if (A is not None and A.nnz > 0) else np.zeros(m * n, dtype=np.float64)
        raw = _call_c_dense(n, m, f64(Pd), f64(q), f64(Ad), f64(b), cones, presolve, settings)
        raw.solve_time = time.perf_counter() - t0
        return raw

    # CSC path
    Pc = sp.csc_matrix((n, n)) if P is None else sp.csc_matrix(P)
    Ac = sp.csc_matrix((m, n)) if (A is None or m == 0) else sp.csc_matrix(A)

    i32 = lambda v: np.ascontiguousarray(v, dtype=np.int32)

    cs = _build_iconic_settings(settings or ICONIC_DEFAULTS)

    cone_bytes = [f"{k}:{d}".encode() for k, d in cones]
    cone_ptrs = (_ct.c_char_p * len(cone_bytes))()
    for i, cb in enumerate(cone_bytes):
        cone_ptrs[i] = cb

    x = np.empty(n, dtype=np.float64)
    s_out = np.empty(m, dtype=np.float64)
    z = np.empty(m, dtype=np.float64)
    obj_val = _ct.c_double()
    iters = _ct.c_int()
    status = _ct.c_int()

    ret = _c_api.iconic_solve_with_settings(
        _ct.c_int(n), _ct.c_int(m),
        i32(Pc.indptr).ctypes.data_as(_ct.POINTER(_ct.c_int)) if Pc.nnz > 0 else None,
        i32(Pc.indices).ctypes.data_as(_ct.POINTER(_ct.c_int)) if Pc.nnz > 0 else None,
        Pc.data.ctypes.data_as(_ct.POINTER(_ct.c_double)) if Pc.nnz > 0 else None,
        f64(q).ctypes.data_as(_ct.POINTER(_ct.c_double)),
        i32(Ac.indptr).ctypes.data_as(_ct.POINTER(_ct.c_int)),
        i32(Ac.indices).ctypes.data_as(_ct.POINTER(_ct.c_int)),
        Ac.data.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        f64(b).ctypes.data_as(_ct.POINTER(_ct.c_double)),
        cone_ptrs, _ct.c_int(len(cones)), _ct.c_int(1 if presolve else 0),
        x.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        s_out.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        z.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        _ct.byref(obj_val), _ct.byref(iters), _ct.byref(status),
        _ct.byref(cs),
    )
    if ret != 0:
        raise RuntimeError(f"iconic_solve_with_settings returned {ret}")

    return types.SimpleNamespace(
        x=x, z=z,
        status=_STATUS_TO_STR.get(status.value, "Unsolved"),
        obj_val=obj_val.value,
        iterations=iters.value,
        solve_time=time.perf_counter() - t0,
    )


def _call_c_mip_csc(n, m, data, var_types, lb, ub, cones,
                    max_nodes=0, max_time=0.0, abs_gap=0.0, rel_gap=0.0):
    """Call iconic_solve_mip_csc via ctypes — sparse CSC path, zero-copy."""
    import time, scipy.sparse as sp
    t0 = time.perf_counter()

    f64 = lambda v: np.ascontiguousarray(v, dtype=np.float64) if v is not None else None
    i32 = lambda v: np.ascontiguousarray(v, dtype=np.int32) if v is not None else None

    # Extract CSC data from CVXPY data dict
    P = data.get('P')  # CVXPY key for P matrix
    A = data.get('A')  # CVXPY key for A matrix

    # P CSC data
    Pc = sp.csc_matrix((n, n)) if P is None else sp.csc_matrix(P)
    # A CSC data
    Ac = sp.csc_matrix((m, n)) if (A is None or m == 0) else sp.csc_matrix(A)

    cone_bytes = [f"{k}:{d}".encode() for k, d in cones]
    cone_ptrs = (_ct.c_char_p * len(cone_bytes))()
    for i, cb in enumerate(cone_bytes):
        cone_ptrs[i] = cb

    x = np.empty(n, dtype=np.float64)
    obj_val = _ct.c_double()
    best_bound = _ct.c_double()
    gap = _ct.c_double()
    nodes = _ct.c_int()
    simplex_iters = _ct.c_int()
    status = _ct.c_int()

    ret = _c_api.iconic_solve_mip_csc(
        _ct.c_int(n), _ct.c_int(m),
        i32(Pc.indptr).ctypes.data_as(_ct.POINTER(_ct.c_int)) if Pc.nnz > 0 else None,
        i32(Pc.indices).ctypes.data_as(_ct.POINTER(_ct.c_int)) if Pc.nnz > 0 else None,
        Pc.data.ctypes.data_as(_ct.POINTER(_ct.c_double)) if Pc.nnz > 0 else None,
        f64(data.get('c')).ctypes.data_as(_ct.POINTER(_ct.c_double)),
        i32(Ac.indptr).ctypes.data_as(_ct.POINTER(_ct.c_int)),
        i32(Ac.indices).ctypes.data_as(_ct.POINTER(_ct.c_int)),
        Ac.data.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        f64(data.get('b')).ctypes.data_as(_ct.POINTER(_ct.c_double)),
        cone_ptrs, _ct.c_int(len(cones)),
        i32(var_types).ctypes.data_as(_ct.POINTER(_ct.c_int)) if var_types is not None else None,
        f64(lb).ctypes.data_as(_ct.POINTER(_ct.c_double)) if lb is not None else None,
        f64(ub).ctypes.data_as(_ct.POINTER(_ct.c_double)) if ub is not None else None,
        _ct.c_int(max_nodes), _ct.c_double(max_time),
        _ct.c_double(abs_gap), _ct.c_double(rel_gap),
        x.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        _ct.byref(obj_val), _ct.byref(best_bound),
        _ct.byref(gap), _ct.byref(nodes), _ct.byref(simplex_iters),
        _ct.byref(status),
    )
    if ret != 0:
        raise RuntimeError(f"iconic_solve_mip_csc returned {ret}")

    return types.SimpleNamespace(
        x=x,
        status=_MIP_STATUS_TO_STR.get(status.value, "Unsolved"),
        obj_val=obj_val.value,
        best_bound=best_bound.value,
        gap=gap.value,
        nodes=nodes.value,
        simplex_iters=simplex_iters.value,
        solve_time=time.perf_counter() - t0,
    )


# ── solver settings ──────────────────────────────────────────────────────
# Documented defaults that match iconic-core `Settings::default()`.
ICONIC_DEFAULTS = {
    # -- Tolerances ---------------------------------------------------------
    # ICONIC declares convergence when all three criteria are met:
    #   1. Primal residual:  ‖Ax+s−b‖∞ ≤ ε_abs + ε_rel·max(‖Ax‖∞,‖s‖∞,‖b‖∞)
    #   2. Dual residual:    ‖Px+q+Aᵀz‖∞ ≤ ε_abs + ε_rel·max(‖Px‖∞,‖q‖∞,‖Aᵀz‖∞)
    #   3. Duality gap:      |xᵀPx+qᵀx+bᵀz| ≤ ε_gap·(1+|primal_obj|+|dual_obj|)
    "eps_abs": 1e-8,     # Absolute tolerance for residuals & gap
    "eps_rel": 1e-8,     # Relative tolerance for residuals
    "eps_gap": 1e-8,     # Duality gap tolerance
    #
    # -- Iterations ---------------------------------------------------------
    # Passed explicitly to the native layer (the C ABI's NULL default is 800;
    # the backend's CVXPY-facing default is 200, matching iconic-core).
    "max_iters": 200,    # Max IPM iterations
    #
    # -- Presolve -----------------------------------------------------------
    "presolve": True,    # Ruiz equilibration + structural reduction passes
    #
    # -- Algorithm selection ------------------------------------------------
    "hsd": True,         # Homogeneous Self-Dual embedding (robust infeasibility)
    "verbose": False,    # Print per-iteration progress
}

# Usable parameter names (the ones the CVXPY backend actually passes through
# to the native layer via the IconicSettings struct).  Other keys in
# ICONIC_DEFAULTS are documented for reference / future use.
_ICONIC_PARAMS = {"presolve", "max_iters", "eps_abs", "eps_rel", "eps_gap", "hsd"}


class ICONIC(ConicSolver):
    """ICONIC — a convex optimizer in Rust (LP / QP / SOCP / SDP / ExpCone / MIP).

    ICONIC implements a regularized proximal interior-point method (IPM) with
    Mehrotra predictor-corrector steps, Gondzio multiple centrality correctors,
    and quasidefinite LDLᵀ factorization.  It supports the full symmetric-cone
    product (zero / nonnegative / second-order / PSD) plus the exponential cone.

    Parameters
    ----------
    Pass solver-specific options via CVXPY's ``solver_opts`` dict:

        prob.solve(solver="ICONIC", solver_opts={"presolve": False})

    Available options (all passed through to the native layer via the
    versioned ``IconicSettings`` struct):

    ``presolve`` (bool, default ``True``)
        Run Ruiz equilibration + structural reduction passes before the IPM
        loop.  Disable when debugging or when the problem is known to be
        well-scaled and small.

    ``eps_abs``, ``eps_rel``, ``eps_gap``: float, default ``1e-8``
        Absolute/relative residual and duality-gap tolerances.

    ``max_iters``: int, default ``200``
        Maximum IPM iterations.  A small value stops the solve early: the
        returned status is ``optimal_inaccurate`` / ``user_limit`` and
        ``num_iters`` reflects the cap.

    ``hsd``: bool, default ``True``
        Use the Homogeneous Self-Dual embedding for infeasibility detection.

    ``verbose`` (bool, default ``False``)
        Print per-iteration progress to stdout.  Can also be set via the
        top-level CVXPY ``verbose`` argument:
        ``prob.solve(solver="ICONIC", verbose=True)``.

    Inspect all defaults at runtime::

        >>> from iconic._backend import ICONIC_DEFAULTS
        >>> print(ICONIC_DEFAULTS)
    """

    MIP_CAPABLE = True
    _SUPPORTED = [SOC, ExpCone, PowCone3D]
    _SUPPORTED.append(PSD)
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + _SUPPORTED
    EXP_CONE_ORDER = [0, 1, 2]

    STATUS_MAP = {
        "Solved": s.OPTIMAL,
        "SolvedInaccurate": s.OPTIMAL_INACCURATE,
        "PrimalInfeasible": s.INFEASIBLE,
        "DualInfeasible": s.UNBOUNDED,
        "MaxIterations": s.USER_LIMIT,
        "TimeLimit": s.USER_LIMIT,
        "NumericalError": s.SOLVER_ERROR,
        "Unsolved": s.SOLVER_ERROR,
    }

    def name(self):
        return "ICONIC"

    def import_solver(self):
        pass  # ctypes load is at module level

    def supports_quad_obj(self) -> bool:
        return True

    def cite(self, data):
        return "ICONIC: a convex optimizer in Rust."

    @staticmethod
    def parse_solver_opts(verbose, opts):
        """Parse CVXPY solver_opts into a ICONIC settings dict.

        Settings start from documented defaults, apply ``solver_opts``
        overrides, then set ``verbose`` from the top-level CVXPY flag.
        Unknown keys are silently ignored.

        Note on the CVXPY 1.7 argument shape: ``solve_via_data`` receives the
        full kwargs dict of the ``prob.solve(...)`` call, with the user's
        options namespaced under the ``'solver_opts'`` key (the same shape
        SCIPY's interface reads via ``'scipy_options'``).  Accept both that
        and a bare options dict.
        """
        settings = dict(ICONIC_DEFAULTS)
        if opts:
            if isinstance(opts, dict) and isinstance(opts.get("solver_opts"), dict):
                opts = opts["solver_opts"]
            for key in _ICONIC_PARAMS:
                if key in opts:
                    settings[key] = opts[key]
        settings["verbose"] = verbose
        return settings

    def apply(self, problem):
        """Extract boolean/integer indices from the problem before reduction."""
        data, inv_data = super().apply(problem)
        # Extract integer/boolean variable indices from the problem variable
        var = problem.x
        if hasattr(var, 'boolean_idx') and var.boolean_idx:
            data[s.BOOL_IDX] = np.asarray([int(t[0]) for t in var.boolean_idx], dtype=np.int32)
        else:
            data[s.BOOL_IDX] = np.array([], dtype=np.int32)
        if hasattr(var, 'integer_idx') and var.integer_idx:
            data[s.INT_IDX] = np.asarray([int(t[0]) for t in var.integer_idx], dtype=np.int32)
        else:
            data[s.INT_IDX] = np.array([], dtype=np.int32)
        inv_data['is_mip'] = bool(len(data[s.BOOL_IDX]) or len(data[s.INT_IDX]))
        return data, inv_data

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        # CVXPY 1.7 passes the full kwargs dict with the user's options under
        # the 'solver_opts' key; unwrap once so both parse_solver_opts and the
        # MIP option reads below see the user's dict.
        user_opts = solver_opts or {}
        if isinstance(user_opts, dict) and isinstance(user_opts.get("solver_opts"), dict):
            user_opts = user_opts["solver_opts"]
        settings = self.parse_solver_opts(verbose, user_opts)

        A = data[s.A]
        b = np.ascontiguousarray(np.asarray(data[s.B], dtype=np.float64).ravel())
        q = np.ascontiguousarray(np.asarray(data[s.C], dtype=np.float64).ravel())
        dims = data[ConicSolver.DIMS]
        n = q.size
        m = b.size

        cones = []
        if dims.zero > 0:
            cones.append(("zero", int(dims.zero)))
        if dims.nonneg > 0:
            cones.append(("nonneg", int(dims.nonneg)))
        for d in dims.soc:
            cones.append(("soc", int(d)))
        for k in dims.psd:
            # ICONIC expects the svec length k(k+1)/2, but CVXPY's A matrix
            # has k² rows for the PSD cone (full matrix including duplicate
            # lower-triangle entries).  We drop the duplicates here and pad
            # the duals back below.
            cones.append(("psd", int(k) * (int(k) + 1) // 2))
        for _ in range(dims.exp):
            cones.append(("exp", 3))
        for alpha in getattr(dims, 'p3d', []):
            cones.append(("power", f"3:{alpha}"))

        # CVXPY 1.7+ arranges PSD cone rows in column-major order over
        # the k×k matrix (k² rows), but ICONIC's PsdTriangle expects the
        # svec length k(k+1)/2 with √2 scaling on off-diagonals.
        #
        # We apply row-scaling and a keep mask so that:
        #   - Off-diagonal upper-triangle rows are scaled by √2.
        #   - Strict lower-triangle rows are dropped.
        # After solving, duals are mapped back (z_svec / √2 for off-diags,
        # copied to both upper and lower full-matrix positions).
        _psd_blocks = []  # list of (ki, svec_len) for dual reconstruction
        _sqrt2 = np.sqrt(2.0)
        if dims.psd:
            _psd_offset = dims.zero + dims.nonneg + sum(dims.soc)
            keep = np.ones(m, dtype=bool)
            row_scale = np.ones(m)  # scaling factor per row
            pos = _psd_offset
            for k in dims.psd:
                ki = int(k)
                svec_len = ki * (ki + 1) // 2
                _psd_blocks.append((ki, svec_len))
                for col in range(ki):
                    for row in range(ki):
                        idx = pos + col * ki + row
                        if row <= col:
                            if row < col:
                                row_scale[idx] = _sqrt2
                        else:
                            keep[idx] = False
                pos += ki * ki
            # Apply row scaling: scale_mat = diag(row_scale)
            from scipy.sparse import diags as sp_diags
            A = sp_diags(row_scale) @ A
            b = row_scale * b
            A = A[keep]
            b = b[keep]
            m = A.shape[0]

        # MIP path: integer or binary variables → branch-and-bound
        has_mip = bool(
            (s.BOOL_IDX in data and data[s.BOOL_IDX] is not None and len(data[s.BOOL_IDX]) > 0)
            or (s.INT_IDX in data and data[s.INT_IDX] is not None and len(data[s.INT_IDX]) > 0)
        )

        if has_mip:
            var_types = np.zeros(n, dtype=np.int32)
            lb = np.full(n, -1e20, dtype=np.float64)
            ub = np.full(n, 1e20, dtype=np.float64)
            if s.BOOL_IDX in data and data[s.BOOL_IDX] is not None and len(data[s.BOOL_IDX]) > 0:
                bool_idx = np.asarray(data[s.BOOL_IDX], dtype=np.int32)
                var_types[bool_idx] = 2
                lb[bool_idx] = 0.0
                ub[bool_idx] = 1.0
            if s.INT_IDX in data and data[s.INT_IDX] is not None and len(data[s.INT_IDX]) > 0:
                int_idx = np.asarray(data[s.INT_IDX], dtype=np.int32)
                var_types[int_idx] = 1

            max_nodes = int(user_opts.get("max_nodes", 100000))
            max_time = float(user_opts.get("max_time", 60.0))
            abs_gap = float(user_opts.get("abs_gap", 1e-6))
            rel_gap = float(user_opts.get("rel_gap", 1e-4))

            raw = _call_c_mip_csc(
                n, m, data, var_types, lb, ub, cones,
                max_nodes=max_nodes, max_time=max_time,
                abs_gap=abs_gap, rel_gap=rel_gap,
            )
            raw._is_mip = True
            raw._mip_nodes = raw.nodes
            return raw

        presolve = settings.get("presolve", True)
        P = data[s.P] if (s.P in data and data[s.P] is not None) else None
        raw = _iconic_solve(P, A, q, b, cones, presolve, n, A.shape[0], settings)

        # Reconstruct the full k² dual vector from the svec duals,
        # applying inverse √2 scaling for off-diagonals and copying
        # to both upper and lower full-matrix positions.
        z_padded = raw.z
        if _psd_blocks:
            z_parts = []
            pos = 0
            if dims.zero > 0:
                z_parts.append(z_padded[pos:pos + dims.zero])
                pos += dims.zero
            if dims.nonneg > 0:
                z_parts.append(z_padded[pos:pos + dims.nonneg])
                pos += dims.nonneg
            for d in dims.soc:
                z_parts.append(z_padded[pos:pos + d])
                pos += d
            for ki, svec_len in _psd_blocks:
                duals_ut = z_padded[pos:pos + svec_len]  # upper-triangle duals
                full_block = np.zeros(ki * ki)
                ut_idx = 0
                for col in range(ki):
                    for row in range(ki):
                        if row <= col:  # upper triangle
                            z_val = duals_ut[ut_idx]
                            if row < col:  # off-diagonal: z_svec / √2
                                z_val = z_val / _sqrt2
                            full_block[col * ki + row] = z_val
                            ut_idx += 1
                        else:  # lower triangle: copy from symmetric upper entry
                            # Upper entry is at (col, row) = (this_row, this_col)
                            # In column-major: upper_pos = row*ki + col
                            z_val = full_block[row * ki + col]
                            full_block[col * ki + row] = z_val
                z_parts.append(full_block)
                pos += svec_len
            exp_count = dims.exp * 3
            if exp_count > 0:
                z_parts.append(z_padded[pos:pos + exp_count])
                pos += exp_count
            p3d_count = sum(1 for _ in getattr(dims, 'p3d', [])) * 3
            if p3d_count > 0:
                z_parts.append(z_padded[pos:pos + p3d_count])
                pos += p3d_count
            z_padded = np.concatenate(z_parts)

        # Return a dict in the standard CVXPY ConicSolver format so the base
        # class's invert() can split duals correctly.
        return {
            "status": self.STATUS_MAP.get(str(raw.status), s.SOLVER_ERROR),
            "value": raw.obj_val,
            "primal": raw.x,
            "eq_dual": z_padded[:dims.zero] if dims.zero > 0 else np.array([]),
            "ineq_dual": z_padded[dims.zero:] if dims.zero < len(z_padded) else np.array([]),
            "_solve_time": raw.solve_time,
            "_iters": raw.iterations,
        }

    def invert(self, solution, inverse_data):
        is_mip = getattr(solution, '_is_mip', False)
        if is_mip:
            attr = {s.SOLVE_TIME: solution.solve_time, s.NUM_ITERS: solution.nodes}
            status = _MIP_STATUS_TO_CVXPY.get(str(solution.status), s.SOLVER_ERROR)
            if status in s.SOLUTION_PRESENT:
                opt_val = solution.obj_val + inverse_data[s.OFFSET]
                primal_vars = {inverse_data[self.VAR_ID]: solution.x}
                return Solution(status, opt_val, primal_vars, {}, attr)
            return failure_solution(status, attr)

        # Non-MIP: delegate to the base class which correctly splits duals.
        result = super().invert(solution, inverse_data)
        result.attr[s.SOLVE_TIME] = solution.get("_solve_time", 0.0)
        result.attr[s.NUM_ITERS] = solution.get("_iters", 0)
        return result


def register():
    """Register ICONIC with CVXPY so ``solver="ICONIC"`` works for conic and MIP."""
    import cvxpy.reductions.solvers.defines as defines

    conic = ICONIC()
    defines.SOLVER_MAP_CONIC[conic.name()] = conic
    for lst in (
        defines.CONIC_SOLVERS,
        defines.MI_SOLVERS,
        defines.INSTALLED_SOLVERS,
        defines.INSTALLED_CONIC_SOLVERS,
        defines.INSTALLED_MI_SOLVERS,
    ):
        if conic.name() not in lst:
            lst.append(conic.name())
    return conic
