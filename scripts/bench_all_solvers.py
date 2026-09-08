#!/usr/bin/env python3
"""ICONIC vs every installed open-source CVXPY solver — QP/LP + MIP.

Problem instances are *not* defined here. They're generated once, in Rust, by
`iconic-bench`'s `build_suite`/`build_mip_suite` (the same catalogue the native
`iconic-bench` binary times ICONIC against), and exported as JSON so this script drives
CVXPY solvers over the exact same data — no separate Python problem generators to
keep in sync.

ICONIC's own numbers are never driven through this script: they're merged in from a
native `iconic-bench run --mip` JSONL instead (see `--iconic-jsonl`). The C ABI (the
ctypes path `iconic-py`/`iconic-c` ship) was tried here first and, at the time, disagreed
badly with the native solve path on perfectly ordinary instances -- e.g. a 50-
variable `qp_random` case that the native path solves in 8 iterations hit 799
iterations and `MaxIterations` through the C ABI, and a `lp_random` instance hung
outright. Root-caused and fixed (a beta-scaling bug in the faer dsyrk fallback that
only iconic-c's build configuration exercised, corrupting the condensed Hessian on
reused per-iteration scratch buffers, plus a missing escape hatch in the
factorization-escalation loop for equality-free QPs) -- the C ABI now matches the
native path exactly on the same instances. Kept merging native numbers anyway: it
avoids ctypes call overhead entirely and there's no reason to route ICONIC's own
measurement through an extra FFI hop when the native run already produces the
number directly.

Usage:
    cargo run -p iconic-bench --release -- export-qp  --out /tmp/qp_suite.json  --max-n 200
    cargo run -p iconic-bench --release -- export-mip --out /tmp/mip_suite.json --max-n 150
    cargo run -p iconic-bench --release -- run --mip --out data.jsonl   # ICONIC's own timings

    python3 scripts/bench_all_solvers.py \
        --qp-json /tmp/qp_suite.json --mip-json /tmp/mip_suite.json \
        --iconic-jsonl data.jsonl --out iconic-bench/baselines/multisolv.jsonl

Only the open-source CVXPY solvers listed in `QP_SOLVERS`/`MIP_SOLVERS` below are
driven here — an explicit allowlist, rather than "everything CVXPY reports installed".
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time

import numpy as np

# ── CVXPY solver wrapper ───────────────────────────────────────────────────
import cvxpy as cp

# Explicit open-source allowlist -- never "whatever CVXPY reports installed".
QP_SOLVERS = ["CLARABEL", "PIQP", "OSQP", "CVXOPT", "ECOS", "SCS", "HIGHS"]
MIP_SOLVERS = ["SCIP", "HIGHS", "GLPK_MI", "CBC"]


def _handler(sig, frame):
    raise TimeoutError("timeout")


def cvxpy_qp(P, q, A_eq, b_eq, A_in, b_in, solver, timeout=8):
    n = len(q)
    x = cp.Variable(n)
    constraints = []
    if A_in.shape[0]:
        constraints.append(A_in @ x <= b_in)
    if A_eq.shape[0]:
        constraints.append(A_eq @ x == b_eq)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x), constraints)
    t0 = time.perf_counter()
    old = signal.signal(signal.SIGALRM, _handler); signal.alarm(timeout)
    try:
        prob.solve(solver=solver, verbose=False, ignore_dpp=True)
        elapsed = (time.perf_counter() - t0) * 1000
        s = prob.status
        return ("Solved" if s == cp.OPTIMAL else "SolvedInaccurate" if s == cp.OPTIMAL_INACCURATE
                else "PrimalInfeasible" if s == cp.INFEASIBLE else "DualInfeasible" if s == cp.UNBOUNDED
                else str(s)), (prob.value or 0), elapsed
    except (TimeoutError, Exception) as e:
        return f"ERR:{type(e).__name__}", 0, (time.perf_counter() - t0) * 1000
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)


def cvxpy_mip(q, A_eq, b_eq, A_in, b_in, lb, ub, var_types, solver, timeout=12):
    """Solve `min qᵀx  s.t.  A_eq x = b_eq, A_in x <= b_in, lb<=x<=ub`, with `var_types[j]`
    one of 'continuous'/'integer'/'binary'. Discrete variables in one instance are always
    uniformly binary or uniformly general-integer (never mixed) in this suite, so a single
    discrete CVXPY Variable with the right flag suffices; continuous and discrete blocks are
    kept as separate Variables and column-sliced back into the constraint matrices, avoiding
    an extra linking-equality per discrete variable.
    """
    n = len(q)
    cont_idx = np.array([j for j, vt in enumerate(var_types) if vt == "continuous"], dtype=int)
    disc_idx = np.array([j for j, vt in enumerate(var_types) if vt != "continuous"], dtype=int)
    all_binary = len(disc_idx) == 0 or all(var_types[j] == "binary" for j in disc_idx)

    xc = cp.Variable(len(cont_idx)) if len(cont_idx) else None
    xd = cp.Variable(len(disc_idx), boolean=bool(all_binary), integer=not all_binary) if len(disc_idx) else None

    def combine(A):
        expr = 0
        if xc is not None:
            expr = expr + A[:, cont_idx] @ xc
        if xd is not None:
            expr = expr + A[:, disc_idx] @ xd
        return expr

    obj = 0
    if xc is not None:
        obj = obj + q[cont_idx] @ xc
    if xd is not None:
        obj = obj + q[disc_idx] @ xd

    constraints = []
    if A_in.shape[0]:
        constraints.append(combine(A_in) <= b_in)
    if A_eq.shape[0]:
        constraints.append(combine(A_eq) == b_eq)
    if xc is not None:
        constraints += [xc >= lb[cont_idx], xc <= ub[cont_idx]]
    if xd is not None:
        constraints += [xd >= lb[disc_idx], xd <= ub[disc_idx]]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    t0 = time.perf_counter()
    old = signal.signal(signal.SIGALRM, _handler); signal.alarm(timeout)
    try:
        prob.solve(solver=solver, verbose=False, ignore_dpp=True)
        elapsed = (time.perf_counter() - t0) * 1000
        s = prob.status
        return ("Solved" if s == cp.OPTIMAL else "SolvedInaccurate" if s == cp.OPTIMAL_INACCURATE
                else "PrimalInfeasible" if s == cp.INFEASIBLE else str(s)), (prob.value or 0), elapsed
    except (TimeoutError, Exception) as e:
        return f"ERR:{type(e).__name__}", 0, (time.perf_counter() - t0) * 1000
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)


# ── caching layer ────────────────────────────────────────────────────────────

def _solver_versions(solvers):
    """Return {name_lower: version_string} for installed solvers via pip freeze."""
    try:
        out = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout
    except Exception:
        return {}
    versions = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            continue
        pkg, ver = line.split("==", 1)
        for s in solvers:
            if pkg.lower() == s.lower() or pkg.lower().startswith(s.lower() + "-"):
                versions[s.lower()] = ver
                break
    return versions


def _problem_hash(**arrays):
    """SHA256 of the problem data as JSON (deterministic per instance)."""
    payload = {}
    for k, arr in arrays.items():
        if isinstance(arr, np.ndarray):
            payload[k] = arr.tolist()
        elif isinstance(arr, list):
            payload[k] = arr
        else:
            payload[k] = arr
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_cache(path):
    """Read cache from disk.  Returns (entries, stored_versions) — empty on miss/corrupt."""
    if not os.path.exists(path):
        return {}, {}
    try:
        with open(path) as f:
            data = json.load(f)
        entries = {k: v for k, v in data.items() if k != "_versions"}
        versions = data.get("_versions", {})
        return entries, versions
    except Exception:
        return {}, {}


def _save_cache(path, entries, versions):
    """Atomically write cache to disk."""
    data = dict(entries)
    data["_versions"] = versions
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, sort_keys=True)
    os.replace(tmp, path)


# ── loading Rust-exported instances ─────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def qp_from_record(r):
    n, me, mi = r["n"], r["m_eq"], r["m_in"]
    P = np.array(r["P"], dtype=np.float64).reshape(n, n)
    q = np.array(r["q"], dtype=np.float64)
    A_eq = np.array(r["A_eq"], dtype=np.float64).reshape(me, n)
    b_eq = np.array(r["b_eq"], dtype=np.float64)
    A_in = np.array(r["A_in"], dtype=np.float64).reshape(mi, n)
    b_in = np.array(r["b_in"], dtype=np.float64)
    return P, q, A_eq, b_eq, A_in, b_in


def mip_from_record(r):
    n, m = r["n"], r["m"]
    q = np.array(r["q"], dtype=np.float64)
    A = np.array(r["A"], dtype=np.float64).reshape(m, n)
    b = np.array(r["b"], dtype=np.float64)
    lb = np.array(r["lb"], dtype=np.float64)
    ub = np.array(r["ub"], dtype=np.float64)
    var_types = r["var_types"]
    eq_rows, in_rows, off = [], [], 0
    for kind, dim in r["cones"]:
        rows = list(range(off, off + dim))
        (eq_rows if kind == "zero" else in_rows).extend(rows)
        off += dim
    A_eq, b_eq = A[eq_rows], b[eq_rows]
    A_in, b_in = A[in_rows], b[in_rows]
    return q, A_eq, b_eq, A_in, b_in, lb, ub, var_types


def load_iconic_rows(path, mode, wanted_names):
    """Read `mode=="<mode>", solver=="iconic"` rows from a iconic-bench `run` JSONL (native
    Rust timing). Used for both QP (mode="presolve") and MIP (mode="mip") -- see the
    module docstring for why ICONIC is never driven through this script directly."""
    rows = []
    if not path:
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("mode") == mode and r.get("solver") == "iconic" and r.get("name") in wanted_names:
                rows.append(r)
    return rows


# ── main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qp-json", help="QP/LP instances from `iconic-bench export-qp`")
    ap.add_argument("--mip-json", help="MIP instances from `iconic-bench export-mip`")
    ap.add_argument("--iconic-jsonl", help="iconic-bench `run --mip` JSONL, to merge ICONIC's own QP+MIP timings in")
    ap.add_argument("--out", default="/tmp/multisolv.jsonl")
    ap.add_argument("--qp-timeout", type=int, default=8)
    ap.add_argument("--mip-timeout", type=int, default=12)
    ap.add_argument("--cache", default="/tmp/iconic_solver_cache.json",
                    help="Cache file for solver results (default: %(default)s)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Skip cache entirely — always re-run solvers")
    args = ap.parse_args()

    installed = cp.installed_solvers()
    qp_solvers = [s for s in QP_SOLVERS if s in installed]
    mip_solvers = [s for s in MIP_SOLVERS if s in installed]
    print(f"QP: {qp_solvers}  MIP: {mip_solvers}  (+ ICONIC, merged from --iconic-jsonl)")

    # ── cache init ─────────────────────────────────────────────────────────
    cache_entries, cache_versions = ({}, {}) if args.no_cache else _load_cache(args.cache)
    all_sl = [s.lower() for s in qp_solvers + mip_solvers]
    current_versions = _solver_versions(all_sl) if not args.no_cache else {}
    stale_solvers = set()
    for s in all_sl:
        if s in cache_versions and s not in current_versions:
            stale_solvers.add(s)
        elif s in cache_versions and s in current_versions and cache_versions[s] != current_versions[s]:
            stale_solvers.add(s)
    if stale_solvers:
        for key in list(cache_entries):
            for ss in stale_solvers:
                cache_entries[key].pop(ss, None)
            if not cache_entries[key]:
                del cache_entries[key]
        for s in stale_solvers:
            cache_versions.pop(s, None)
        print(f"Cache: invalidated {len(stale_solvers)} solver(s) due to version change ({', '.join(sorted(stale_solvers))})")
    else:
        print(f"Cache: {len(cache_entries)} entries loaded")
    cache_hits, cache_misses = 0, 0

    records = []
    f = open(args.out, "w")
    def write(r):
        records.append(r); f.write(json.dumps(r) + "\n"); f.flush()

    # ── QP/LP benchmarks ────────────────────────────────────────────────
    if args.qp_json and qp_solvers:
        qp_cases = load_json(args.qp_json)
        print(f"\nQP/LP: {len(qp_cases)} instances from {args.qp_json}")
        iconic_qp_rows = {r["name"]: r for r in load_iconic_rows(args.iconic_jsonl, "presolve", {c["name"] for c in qp_cases})}
        print(f"Merged {len(iconic_qp_rows)} ICONIC QP rows from {args.iconic_jsonl}")

        for r in qp_cases:
            cname, name = r["category"], r["name"]
            P, q, A_eq, b_eq, A_in, b_in = qp_from_record(r)
            n, m = r["n"], r["m_eq"] + r["m_in"]

            iconic_row = iconic_qp_rows.get(name)
            bt_c = iconic_row["time_ms"] if iconic_row else float("inf")
            write(iconic_row if iconic_row else {
                "category": cname, "name": name, "solver": "iconic", "mode": "presolve",
                "n": n, "m": m, "status": "Unsolved", "iters": 0, "time_ms": 0.0, "kkt_res": 0.0})

            parts = [f"iconic={bt_c:.1f}ms"]
            for sol in qp_solvers:
                sol_key = sol.lower()
                phash = (None if args.no_cache else
                         _problem_hash(P=P, q=q, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in))
                bt, bs, bo = float("inf"), "Unsolved", 0.0
                if phash and phash in cache_entries and sol_key in cache_entries[phash]:
                    cached = cache_entries[phash][sol_key]
                    if cached["status"] == "Solved":
                        bt, bs, bo = cached["time_ms"], cached["status"], cached.get("obj_val", 0.0)
                        cache_hits += 1
                if bs != "Solved":
                    cache_misses += 1
                    for _ in range(2):
                        s, o, t = cvxpy_qp(P, q, A_eq, b_eq, A_in, b_in, sol, timeout=args.qp_timeout)
                        if t < bt and s == "Solved":
                            bt, bs, bo = t, s, o
                        elif t < bt and bs == "Unsolved":
                            bt, bs, bo = t, s, o
                    if phash:
                        cache_entries.setdefault(phash, {})[sol_key] = {
                            "status": bs, "obj_val": bo, "time_ms": bt}
                ratio = bt / bt_c if bt_c > 0 and bt > 0 and bs == "Solved" else 0
                parts.append(f"{sol[:4]}={bt:.0f}ms" + (f"({ratio:.1f}x)" if ratio > 0 else "?"))
                write({"category": cname, "name": name, "solver": sol.lower(), "mode": "presolve",
                       "n": n, "m": m, "status": bs, "iters": 0, "time_ms": round(bt, 5), "kkt_res": 0.0})
            print(f"  {cname}/{name} n={n} m={m}: " + " ".join(parts))

    # ── MIP benchmarks ──────────────────────────────────────────────────
    if args.mip_json and mip_solvers:
        mip_cases = load_json(args.mip_json)
        print(f"\nMIP: {len(mip_cases)} instances from {args.mip_json}")
        for r in mip_cases:
            cname, name = r["category"], r["name"]
            q, A_eq, b_eq, A_in, b_in, lb, ub, var_types = mip_from_record(r)
            n, m = r["n"], r["m"]

            parts = []
            for sol in mip_solvers:
                sol_key = sol.lower()
                phash = (None if args.no_cache else
                         _problem_hash(q=q, A_eq=A_eq, b_eq=b_eq, A_in=A_in, b_in=b_in,
                                       lb=lb, ub=ub, var_types=var_types))
                bt, bs, bo = float("inf"), "Unsolved", 0.0
                if phash and phash in cache_entries and sol_key in cache_entries[phash]:
                    cached = cache_entries[phash][sol_key]
                    if cached["status"] == "Solved":
                        bt, bs, bo = cached["time_ms"], cached["status"], cached.get("obj_val", 0.0)
                        cache_hits += 1
                if bs != "Solved":
                    cache_misses += 1
                    for _ in range(2):
                        s, o, t = cvxpy_mip(q, A_eq, b_eq, A_in, b_in, lb, ub, var_types, sol, timeout=args.mip_timeout)
                        if t < bt and s == "Solved":
                            bt, bs, bo = t, s, o
                        elif t < bt and bs == "Unsolved":
                            bt, bs, bo = t, s, o
                    if phash:
                        cache_entries.setdefault(phash, {})[sol_key] = {
                            "status": bs, "obj_val": bo, "time_ms": bt}
                parts.append(f"{sol[:4]}={bt:.1f}ms {bs}")
                write({"category": cname, "name": name, "solver": sol.lower(), "mode": "mip",
                       "n": n, "m": m, "status": bs, "iters": 0, "time_ms": round(bt, 5), "kkt_res": 0.0})
            print(f"  {cname}/{name} n={n} m={m}: " + " ".join(parts))

        # Merge in ICONIC's own (Rust-timed) MIP rows for the same instances, so the Dolan
        # curve compares every solver on identical problems from a single output file.
        wanted = {r["name"] for r in mip_cases}
        iconic_rows = load_iconic_rows(args.iconic_jsonl, "mip", wanted)
        print(f"\nMerged {len(iconic_rows)} ICONIC MIP rows from {args.iconic_jsonl}")
        for r in iconic_rows:
            write(r)

    f.close()
    print(f"\nWrote {len(records)} records to {args.out}")

    # ── cache save & stats ──────────────────────────────────────────────────
    if not args.no_cache and cache_misses:
        # Merge in newly observed solver versions before saving
        for s, v in current_versions.items():
            cache_versions.setdefault(s, v)
        _save_cache(args.cache, cache_entries, cache_versions)
    total = cache_hits + cache_misses
    if total:
        print(f"Cache: {cache_hits} hits / {cache_misses} misses ({cache_hits * 100 // total}% hit) "
              f"[{args.cache}{' (disabled)' if args.no_cache else ''}]")
    elif not args.no_cache:
        print(f"Cache: no lookups [{args.cache}]")

    from math import log, exp
    for mode_name, mode_val in [("QP", "presolve"), ("MIP", "mip")]:
        recs = [r for r in records if r["mode"] == mode_val and r["status"] == "Solved"]
        solvers_in_mode = sorted(set(r["solver"] for r in recs))
        if len(solvers_in_mode) <= 1:
            continue
        print(f"\n{mode_name} summary:")
        iconic_ts = [r["time_ms"] for r in recs if r["solver"] == "iconic"]
        iconic_sgm = exp(sum(log(t + 0.1) for t in iconic_ts) / len(iconic_ts)) - 0.1 if iconic_ts else 0
        print(f"{'Solver':<12} {'OK':>5} {'SGM(ms)':>10} {'vsICONIC':>8}")
        print("-" * 38)
        for s in solvers_in_mode:
            ts = [r["time_ms"] for r in recs if r["solver"] == s]
            ok = len(ts)
            sgm = exp(sum(log(t + 0.1) for t in ts) / ok) - 0.1 if ok else float("inf")
            print(f"{s:<12} {ok:>5} {sgm:>10.1f} {sgm/iconic_sgm:>7.2f}x" if iconic_sgm > 0 and ok else f"{s:<12} {ok:>5} {'-':>10} {'-':>8}")


if __name__ == "__main__":
    main()
