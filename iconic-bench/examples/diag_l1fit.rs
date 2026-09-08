//! Diagnostic: reproduce lp_l1fit rows60_cols12 / rows120_cols20 and print the
//! KKT-residual components + NaN audit of the returned solution. The suite's
//! records show status=Solved with kkt_res exactly 1.0 at 156-916 iters while
//! `inf_norm` ignores NaN entries — checking whether the solution hides NaN.
//!
//! Run: cargo run --release -p iconic-bench --example diag_l1fit -- 60 12

use iconic_bench::lp_l1fit;
use iconic_presolve::solve_presolved;

fn inf_norm(v: &[f64]) -> f64 {
    let mut m = 0.0f64;
    for &x in v {
        if x > m {
            m = x;
        }
    }
    m
}

fn main() {
    let rows: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let cols: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let prob = lp_l1fit(rows, cols, 89 + rows as u64);
    let sol = solve_presolved(&prob, &iconic_core::Settings::default());

    println!(
        "status={:?} iters={} obj={:.10}",
        sol.status, sol.iters, sol.obj_val
    );
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();

    let px = prob.p.matvec(&sol.x);
    let aty = prob.a_eq.matvec_t(&sol.y);
    let atz = prob.a_in.matvec_t(&sol.z);
    let rd: Vec<f64> = (0..n)
        .map(|i| px[i] + prob.q[i] + aty[i] + atz[i])
        .collect();
    let aeqx = prob.a_eq.matvec(&sol.x);
    let rb: Vec<f64> = (0..me).map(|i| aeqx[i] - prob.b_eq[i]).collect();
    let ainx = prob.a_in.matvec(&sol.x);
    let rh: Vec<f64> = (0..mi).map(|i| ainx[i] + sol.s[i] - prob.b_in[i]).collect();
    let comp: Vec<f64> = (0..mi).map(|i| (sol.s[i] * sol.z[i]).abs()).collect();

    println!(
        "inf_norm: rd={:.3e} rb={:.3e} rh={:.3e} comp={:.3e}",
        inf_norm(&rd),
        inf_norm(&rb),
        inf_norm(&rh),
        inf_norm(&comp)
    );
    let nan_count = |v: &[f64]| v.iter().filter(|x| !x.is_finite()).count();
    println!(
        "non-finite: x={} y={} s={} z={} (n={} me={} mi={})",
        nan_count(&sol.x),
        nan_count(&sol.y),
        nan_count(&sol.s),
        nan_count(&sol.z),
        n,
        me,
        mi
    );
    // Where do the non-finite live?
    for (i, v) in sol.s.iter().enumerate() {
        if !v.is_finite() {
            println!("  s[{}] = {:?} (z[{}] = {:?})", i, v, i, sol.z[i]);
        }
    }
    // The suite compares against the presolved-problem optimum via a dense reference?
    // Print the first few x components for sanity.
    println!("x[0..5] = {:?}", &sol.x[..5.min(n)]);
}
