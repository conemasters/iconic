# iconic-bench

In-process benchmark harness for ICONIC. One Rust runner solves a catalogue of problem
*families* at several sizes and reports status, iterations, wall-clock time, and KKT
accuracy — so improvements are measurable and regressions are caught.

## Running

```sh
# Run the suite and print the grouped report + per-category summaries:
cargo run -p iconic-bench --release

# Run and also save machine-readable JSONL (one record per case/mode):
cargo run -p iconic-bench --release -- run --reps 5 --out results.jsonl

# Include the MIP suite (branch-and-bound; separate from the QP/conic families below):
cargo run -p iconic-bench --release -- run --reps 5 --mip --out results.jsonl
cargo run -p iconic-bench --release -- mip --out mip_results.jsonl   # MIP suite only

# Diff two runs: flags status downgrades, accuracy loss, and iteration regressions.
cargo run -p iconic-bench --release -- compare baseline.jsonl results.jsonl

# Export the exact QP/LP or MIP instances as JSON (dense arrays), size-capped via
# --max-n, so another language/tool (e.g. scripts/bench_all_solvers.py) can drive
# other solvers over the *same* instances instead of redefining them:
cargo run -p iconic-bench --release -- export-qp  --out qp.json  --max-n 200
cargo run -p iconic-bench --release -- export-mip --out mip.json --max-n 150
```

`compare` exits non-zero if it finds any regression, so it can gate CI.

## What the suite covers

Each family is run at several sizes. QP/LP families are solved in two **modes** — `raw`
(core engine, no presolve) and `presolve` (the full presolve + equilibration pipeline,
the real user path) — so a change is visible in both. Conic families use the `cone`
engine.

| Family | What it stresses |
|---|---|
| `qp_random` | generic well-scaled convex QP |
| `qp_badscaled` | data spanning ~10 orders of magnitude (equilibration target) |
| `qp_banded` | genuinely sparse QP (sparse vs dense KKT) |
| `qp_illcond` | Hessian condition number ~1e8 (flat directions; regularization + refinement) |
| `lp_random` | pure LP, `P = 0` (rank-deficient (x,x) block) |
| `lasso` | ℓ₁ regression — rank-deficient PSD Hessian |
| `nnls` | non-negative least squares |
| `portfolio` | Markowitz QP (equality budget + bounds) |
| `svm` | soft-margin linear SVM (dense curvature on `w` only, margin + nonneg inequalities) |
| `huber` | Huber robust regression (epigraph split folded into a dense rank-deficient Hessian) |
| `mpc` | condensed model-predictive-control QP (block-Toeplitz Hessian + box bounds) |
| `qp_linked` | equality doubletons `x_{2i}=x_{2i+1}` (doubleton presolve halves n) |
| `degen_redundant_eq` | **degenerate**: rank-deficient equality block |
| `degen_dominated_ineq` | **degenerate**: dominated / parallel inequalities |
| `degen_primal_vertex` | **degenerate**: optimum at a vertex with a redundant active set |
| `socp` | second-order cone programs |
| `sdp` | semidefinite programs (PSD cone) |

The MIP suite (`mip.rs`, run separately via `--mip`/`mip`) covers ~66 instances across
knapsack, set covering/packing, facility location, generalized assignment, scheduling,
and deliberately adversarial stress families (weak-LP-relaxation lot-sizing, TSP via
Miller–Tucker–Zemlin, capacitated facility location) designed to expose
branch-and-bound weaknesses rather than all be expected to solve.

## Metrics

- **status** — correctness is the first signal; a fast *wrong* answer must never win.
- **iters / time_ms** — best-of-`reps` wall-clock. Aggregated as a **shifted geometric
  mean** (robust to outliers and near-zero entries).
- **kkt_res** — max KKT violation in original units (cone complementarity is measured by
  the cone inner product, not the elementwise product).

## Baselines

`baselines/main.jsonl` is a committed reference run. Status / iteration / accuracy fields
are reproducible across machines and are what `compare` gates on; the `time_ms` field is
machine-specific and only informs the SGM-time summary line.
