#!/usr/bin/env python3
"""Generate publication-quality Dolan-More performance profiles from iconic-bench data.

For a proper multi-solver Dolan-More chart (all solvers compared), use the
combined output from bench_all_solvers.py:

    # 1. Export problem instances (Rust):
    cargo run -p iconic-bench --release -- export-qp --out /tmp/qp.json --max-n 200

    # 2. Run ICONIC natively for timing:
    cargo run -p iconic-bench --release -- run --reps 3 --out data.jsonl

    # 3. Drive all CVXPY solvers + merge ICONIC timings:
    python3 scripts/bench_all_solvers.py --qp-json /tmp/qp.json \\
        --iconic-jsonl data.jsonl --out combined.jsonl

    # 4. Generate Dolan-More charts:
    python3 scripts/plot_benchmarks.py combined.jsonl -o scripts/charts/

For ICONIC-only data (single solver), a CDF speed curve is produced instead.
When 2+ solvers are present, a proper Dolan-More performance profile is shown.

Produces:
  - dolan_more_qp.png           Dolan–Moré profile on QP/LP
  - dolan_more_mip.png          Dolan–Moré profile on MIP

Comparison strategy (auto-detected from JSONL):
  - Multiple solvers → compare by solver (ICONIC vs Clarabel vs …)
  - Single solver, multiple modes → compare by mode (raw vs presolve, old JSONL)
  - Single solver, single mode → compare each problem against the per-problem
    best time across all its records (still produces a meaningful step curve
    when there are multiple reps per problem).

All charts use a colourblind-friendly palette and are saved at 150 DPI.
"""

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]

SOLVER_COLORS = {
    # QP/LP chart: 8 solvers, one PALETTE color each (no two ever collide in that chart).
    "iconic": "#0072B2", "clarabel": "#D55E00", "piqp": "#009E73",
    "osqp": "#CC79A7", "cvxopt": "#F0E442", "ecos": "#56B4E9",
    "scs": "#E69F00", "highs": "#000000",
    # MIP chart: iconic/highs keep their QP-chart colors (same solver, same color across
    # charts); scip/cbc/glpk_mi reuse colors from solvers that never appear in the MIP
    # chart, so there's still no collision within it.
    "scip": "#D55E00", "cbc": "#009E73", "glpk_mi": "#CC79A7",
}

SOLVER_LABELS = {
    "iconic": "ICONIC", "clarabel": "Clarabel", "piqp": "PIQP",
    "osqp": "OSQP", "cvxopt": "CVXOPT", "ecos": "ECOS",
    "scs": "SCS",
    "scip": "SCIP", "cbc": "CBC", "highs": "HiGHS", "glpk_mi": "GLPK_MI",
}

MODE_COLORS = {
    "presolve": "#0072B2",
    "raw":       "#D55E00",
    "cone":      "#009E73",
    "mip":       "#E69F00",
}

MODE_LABELS = {
    "presolve": "ICONIC (presolve)",
    "raw":       "ICONIC (raw engine)",
    "cone":      "ICONIC (conic)",
    "mip":       "ICONIC (MIP)",
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
}

plt.rcParams.update(STYLE)

# ── data model ──────────────────────────────────────────────────────────────

@dataclass
class Record:
    category: str
    name: str
    solver: str
    mode: str
    n: int
    m: int
    status: str
    iters: int
    time_ms: float
    kkt_res: float

    @classmethod
    def from_jsonl(cls, line: str) -> "Record":
        d = json.loads(line)
        return cls(
            category=d.get("category", ""),
            name=d.get("name", ""),
            solver=d.get("solver", "iconic"),
            mode=d.get("mode", "presolve"),
            n=d.get("n", 0),
            m=d.get("m", 0),
            status=d.get("status", "Unsolved"),
            iters=d.get("iters", 0),
            time_ms=d.get("time_ms", 0.0),
            kkt_res=d.get("kkt_res", 0.0),
        )


def load_records(path: str) -> List[Record]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(Record.from_jsonl(line))
    return records


# ── Dolan–Moré step-curve builder (shared) ───────────────────────────────────

def _build_dm_step(xs_raw: List[float], n_problems: int) -> Tuple[List[float], List[float]]:
    """Build the Dolan–Moré step function from sorted performance ratios.

    Returns (xs, ys) for `ax.plot`.
    """
    xs_raw.sort()
    xs, ys = [], []
    prev_x = 1.0
    for i, r in enumerate(xs_raw):
        r_clamped = max(r, 1.0)
        if r_clamped > prev_x + 1e-9:
            xs.append(prev_x)
            ys.append(i / n_problems)
        xs.append(r_clamped)
        ys.append(i / n_problems)
        xs.append(r_clamped)
        ys.append((i + 1) / n_problems)
        prev_x = r_clamped
    xs.append(max(prev_x, xs[-1] if xs else 1.0))
    ys.append(1.0)
    return xs, ys


def _cdf_solve_times(
    solved_groups: list,     # list of dict[str, Record]
    gk: str,                 # group key
    group_colors: dict,
    group_labels: dict,
    title: str,
    out_path: str,
):
    """CDF of absolute solve times — fallback for single-group data."""
    times = []
    for g in solved_groups:
        if gk in g:
            times.append(g[gk].time_ms)
    times.sort()

    if not times:
        print(f"  [skip] {out_path}: no solve times")
        return

    n = len(times)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    color = group_colors.get(gk, "#0072B2")
    label = group_labels.get(gk, gk)

    xs, ys = [], []
    for i, t in enumerate(times):
        xs.append(t)
        ys.append(i / n)
        xs.append(t)
        ys.append((i + 1) / n)
    # Extend to right edge
    xs.append(times[-1])
    ys.append(1.0)

    ax.plot(xs, ys, color=color, linewidth=2.0, label=label, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Solve time (ms, log₂)")
    ax.set_ylabel("Fraction of problems solved")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.set_ylim(0, 1.02)

    # SGM annotation
    sgm = math.exp(sum(math.log(t + 0.1) for t in times) / n) - 0.1
    ax.axvline(sgm, color=color, linewidth=1.0, linestyle="--", alpha=0.6)
    ax.text(sgm, 0.5, f" SGM = {sgm:.1f} ms", color=color, fontsize=9,
            va="center", alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path} ({n} problems, CDF — single-group fallback)")


def _dolan_more_core(
    records: List[Record],
    group_keys: List[str],
    group_colors: dict,
    group_labels: dict,
    key_fn,          # Record -> str  (extract group key, e.g. .solver or .mode)
    title: str,
    out_path: str,
    log_x: bool = True,
):
    """Generic Dolan–Moré: group by *key_fn*, compute ratio vs best-in-problem.

    Only problems where **every** group key is present and ``Solved`` are included.
    """
    # Partition records by (category, name) → dict[key_fn(r), Record]
    problem_groups: dict[Tuple[str, str], dict[str, Record]] = defaultdict(dict)
    for r in records:
        k = key_fn(r)
        if k in group_keys:
            problem_groups[(r.category, r.name)][k] = r

    # Keep only problems where every group key is present and Solved
    solved: dict[Tuple[str, str], dict[str, Record]] = {}
    for pkey, g in problem_groups.items():
        if len(g) == len(group_keys) and all(
            r.status == "Solved" for r in g.values()
        ):
            solved[pkey] = g

    if not solved:
        print(f"  [skip] {out_path}: no problems with all groups Solved")
        return

    n = len(solved)

    # A Dolan–Moré profile with a single group is meaningless (every problem IS
    # trivially the best → vertical line at τ=1).  Require ≥2 groups.  To get
    # meaningful multi-solver charts, run bench_all_solvers.py first.
    if len(group_keys) <= 1:
        print(f"  [skip] {out_path}: only one group — need >=2 for Dolan-Moré profile")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for gk in group_keys:
        color = group_colors.get(gk, "#999999")
        label = group_labels.get(gk, gk)

        ratios = []
        for g in solved.values():
            best = min(r.time_ms for r in g.values())
            own = g[gk].time_ms
            ratios.append(own / best if best > 0 else 1.0)

        xs, ys = _build_dm_step(ratios, n)
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label, zorder=5)

    if log_x:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Performance ratio τ (log₂)")
        # Tick positions at powers of two, including fractional values.
        # LogLocator with base=2 gives ticks at …, ¼, ½, 1, 2, 4, …
        ax.xaxis.set_major_locator(mticker.LogLocator(base=2))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _:
                f"{int(v)}" if v >= 1 and v == int(v) else
                f"1/{int(1/v)}" if v < 1 and abs(v * round(1/v) - 1) < 0.01 else
                f"{v:.2g}"
            )
        )
    else:
        ax.set_xlabel("Performance ratio τ")
    ax.set_xlim(left=1.0)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction of problems solved")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path} ({n} problems, groups: {', '.join(group_keys)})")


# ── public entry points ──────────────────────────────────────────────────────

def dolan_more_by_solver(
    records: List[Record],
    solvers: List[str],
    title: str,
    out_path: str,
    log_x: bool = True,
    filter_mode: Optional[str] = None,
):
    recs = [r for r in records if filter_mode is None or r.mode == filter_mode]
    _dolan_more_core(
        recs, solvers, SOLVER_COLORS, SOLVER_LABELS,
        key_fn=lambda r: r.solver,
        title=title, out_path=out_path, log_x=log_x,
    )


def dolan_more_by_mode(
    records: List[Record],
    modes: List[str],
    title: str,
    out_path: str,
    log_x: bool = True,
):
    """Dolan–Moré grouped by mode (e.g. raw vs presolve)."""
    recs = [r for r in records if r.mode in modes]
    _dolan_more_core(
        recs, modes, MODE_COLORS, MODE_LABELS,
        key_fn=lambda r: r.mode,
        title=title, out_path=out_path, log_x=log_x,
    )


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts from iconic-bench JSONL.")
    parser.add_argument("jsonl", help="Path to benchmark JSONL file")
    parser.add_argument("-o", "--out-dir", default="scripts/charts",
                        help="Output directory for PNG charts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.jsonl}...")
    records = load_records(args.jsonl)
    print(f"  {len(records)} records")

    solvers = sorted(set(r.solver for r in records))
    modes = sorted(set(r.mode for r in records))
    print(f"  solvers: {solvers}  modes: {modes}")

    # ── Decide strategy ──
    # Solver-based comparison whenever 2+ solvers reported results for a mode; otherwise
    # fall back to mode-based (raw vs presolve, from older JSONL) or a single-solver CDF.
    # The solver *set* is derived from what's actually in the data, not a hardcoded
    # whitelist — a newly installed or newly benchmarked solver shows up automatically
    # without editing this script.
    qp_records = [r for r in records if r.mode != "cone" and r.mode != "mip"]
    qp_solvers = sorted(set(r.solver for r in qp_records if r.mode == "presolve"))
    qp_modes = [m for m in ["raw", "presolve"] if m in modes]

    print("\nDolan–Moré QP/LP:")
    if len(qp_solvers) >= 2:
        dolan_more_by_solver(
            qp_records,
            solvers=qp_solvers,
            title="Dolan–Moré performance profile — QP / LP",
            out_path=str(out_dir / "dolan_more_qp.png"),
            filter_mode="presolve",
        )
    elif len(qp_modes) >= 2:
        # Old JSONL with raw + presolve: compare modes
        dolan_more_by_mode(
            qp_records,
            modes=qp_modes,
            title="Dolan–Moré performance profile — QP / LP",
            out_path=str(out_dir / "dolan_more_qp.png"),
        )
    elif "presolve" in modes:
        # Single solver: Dolan-Moré profile is meaningless with one solver
        # (every instance's "best" is trivially that solver → vertical line).
        # Run bench_all_solvers.py to produce proper multi-solver charts.
        print("  [skip] only ICONIC data present — run bench_all_solvers.py for multi-solver charts")
    else:
        print("  [skip] no QP data")

    # ── MIP ────────────────────────────────────────────────────────────
    print("\nDolan–Moré MIP:")
    mip_records = [r for r in records if r.mode == "mip"]
    mip_solvers = sorted(set(r.solver for r in mip_records))
    if len(mip_solvers) >= 2:
        dolan_more_by_solver(
            mip_records,
            solvers=mip_solvers,
            title="Dolan–Moré performance profile — MIP",
            out_path=str(out_dir / "dolan_more_mip.png"),
            filter_mode="mip",
        )
    elif mip_records:
        # Single solver: Dolan-Moré profile is meaningless with one solver.
        # Run bench_all_solvers.py to produce proper multi-solver charts.
        print("  [skip] only ICONIC data present — run bench_all_solvers.py for multi-solver charts")
    else:
        print("  [skip] no MIP data")

    print(f"\nDone — charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
