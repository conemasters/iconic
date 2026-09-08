//! Time the transport-LP presolve path phase-by-phase.
//! Run: cargo run --release -p iconic-bench --example diag_transport_presolve
use iconic_bench::lp_transport;
use iconic_presolve::reductions::*;
use std::time::Instant;

fn t(label: &str, t0: Instant) -> Instant {
    eprintln!("[{label}] {:>9.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    Instant::now()
}

fn main() {
    let prob = lp_transport(60, 80, 61); // 4800 x 4940, 99.94% sparse
    let n = prob.q.len();
    let mi = prob.b_in.len();
    let total = n * mi;
    let nnz = prob.a_in.data.iter().filter(|&&v| v != 0.0).count();
    eprintln!(
        "transport 60x80: n={n} mi={mi} cells={total} nnz={nnz} density={:.5}%",
        100.0 * nnz as f64 / total as f64
    );
    eprintln!("dense A_in bytes = {:.0} MB", (total * 8) as f64 / 1e6);
    eprintln!("dense P bytes    = {:.0} MB", (n * n * 8) as f64 / 1e6);

    // 1. SparseAIn build (the one dense extraction)
    let t0 = Instant::now();
    let sp = SparseAIn::build(&prob);
    let _t0 = t("SparseAIn::build (one O(mi*n) scan)", t0);

    // 2. The no-op pass chain: each returns prob.clone() (O(1) — the matrices
    //    are Arc-shared since the copy-on-write change; the remaining time is
    //    the passes' own scans/grouping)
    let t0 = Instant::now();
    let (pair_reduced, _pairred) = fold_negated_pairs(&prob, &sp).unwrap();
    let _t0 = t("fold_negated_pairs (no-op chain (O(1) clones))", t0);

    let t0 = Instant::now();
    let (col_reduced, _colred) = eliminate_empty_cols(&pair_reduced, &sp).unwrap();
    let _t0 = t("eliminate_empty_cols (no-op chain (O(1) clones))", t0);

    let t0 = Instant::now();
    let (reduced, _rowred) = reduce_rows(&col_reduced, &sp).unwrap();
    let _t0 = t("reduce_rows (no-op chain (O(1) clones))", t0);

    let t0 = Instant::now();
    let (_redun_reduced, _redunred) = remove_redundant_ineqs(&reduced, &sp).unwrap();
    let _t0 = t("remove_redundant_ineqs (no-op chain (O(1) clones))", t0);

    // 3. Full solve_presolved (includes Ruiz + ipm + ipm-side csr rebuild)
    let t0 = Instant::now();
    let settings = iconic_core::Settings::<f64>::default();
    let sol = iconic_presolve::solve_presolved(&prob, &settings);
    let _t0 = t("solve_presolved total", t0);
    eprintln!(
        "status={:?} iters={} obj={:.10} kkt={:.3e}",
        sol.status,
        sol.iters,
        sol.obj_val,
        iconic_bench::kkt_residual(&prob, &sol)
    );

    // 4. Solve with presolve disabled, for the presolve overhead itself
    let t0 = Instant::now();
    let s2 = iconic_core::Settings::<f64> {
        presolve: false,
        ..iconic_core::Settings::<f64>::default()
    };
    let sol2 = iconic_ipm::solve_qp_with_termination(
        &prob,
        &s2,
        &iconic_ipm::TermScale {
            dual: vec![1.0; n],
            prim_eq: vec![],
            prim_in: vec![1.0; mi],
            comp: 1.0,
        },
    );
    let _t0 = t("solve with presolve=false (ipm only)", t0);
    eprintln!("status={:?} iters={}", sol2.status, sol2.iters);
}
