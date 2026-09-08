//! Trace lp_random n100_m60 through the QP path (the pure-LP IPM path
//! takes 15-27 iterations there). Replicates iconic-bench.s random_lp
//! generator bit-for-bit (same Lcg + seed) so the trace observes the exact
//! suite instance. Run with ICONIC_TRACE_LP=1 for per-iteration mu/residuals.
//!
//! Run: ICONIC_TRACE_LP=1 cargo run --release -p iconic-bench --example diag_lp_random -- 100 60

use iconic_core::rng::Lcg;
use iconic_core::Settings;
use iconic_ipm::QpProblem;
use iconic_linalg::DenseMatrix;
use iconic_presolve::solve_presolved;

fn random_lp(n: usize, m_in: usize, seed: u64) -> iconic_ipm::QpProblem<f64> {
    let mut rng = Lcg::new(seed);
    let q: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
    let x0: Vec<f64> = (0..n).map(|_| 0.5 * rng.signed()).collect();
    let big = 10.0;

    let rows = m_in + 2 * n;
    let mut a_in = DenseMatrix::zeros(rows, n);
    let mut b_in = vec![0.0; rows];
    for r in 0..m_in {
        for j in 0..n {
            a_in.set(r, j, rng.signed());
        }
    }
    let ax0 = a_in.matvec(&x0);
    for r in 0..m_in {
        b_in[r] = ax0[r] + 0.5 + rng.unit();
    }
    for j in 0..n {
        a_in.set(m_in + j, j, 1.0);
        b_in[m_in + j] = big;
        a_in.set(m_in + n + j, j, -1.0);
        b_in[m_in + n + j] = big;
    }
    QpProblem {
        p: DenseMatrix::zeros(n, n),
        q,
        a_eq: DenseMatrix::zeros(0, n),
        b_eq: vec![],
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr: None,
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let m: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let prob = random_lp(n, m, 31);
    let sol = solve_presolved(&prob, &Settings::default());
    println!(
        "status={:?} iters={} obj={:.10}",
        sol.status, sol.iters, sol.obj_val
    );
}
