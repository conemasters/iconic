#![allow(clippy::needless_range_loop)]
#![allow(clippy::field_reassign_with_default)]
//! Diagnostic: does the global BLAS toggle (used by the me==0 escalation
//! regression test) change the illcond n80 solve's outcome?
//!
//! The escalation test sets `set_blas_enabled(false)` process-wide while
//! tests run in parallel; any concurrent solve dispatches to faer instead of
//! OpenBLAS and takes a different numeric path. This reproduces the gate
//! run's one-off `illcond_qp_baseline_failures_now_solved` failure (kkt >=
//! 1000) in isolation: solve the same instance with BLAS on vs off.

use iconic_bench::kkt_residual;
use iconic_core::rng::Lcg;
use iconic_core::Settings;
use iconic_ipm::{solve_qp, QpProblem};
use iconic_linalg::DenseMatrix;

fn ill_conditioned_qp(n: usize, log10_cond: f64, m_in: usize, seed: u64) -> QpProblem<f64> {
    let mut rng = Lcg::new(seed);
    // Random orthogonal Q: fill row-major with signed values, orthonormalize
    // the COLUMNS (matching iconic-bench's random_orthogonal).
    let mut q_orth = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            q_orth[i][j] = rng.signed();
        }
    }
    for j in 0..n {
        for k in 0..j {
            let mut dotp = 0.0;
            for i in 0..n {
                dotp += q_orth[i][k] * q_orth[i][j];
            }
            for i in 0..n {
                q_orth[i][j] -= dotp * q_orth[i][k];
            }
        }
        let mut nrm = 0.0;
        for i in 0..n {
            nrm += q_orth[i][j] * q_orth[i][j];
        }
        nrm = nrm.sqrt().max(1e-12);
        for i in 0..n {
            q_orth[i][j] /= nrm;
        }
    }
    let lam: Vec<f64> = (0..n)
        .map(|i| {
            let t = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            10f64.powf(-log10_cond * t)
        })
        .collect();
    let mut p = DenseMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for t in 0..n {
                acc += q_orth[i][t] * lam[t] * q_orth[j][t];
            }
            p.set(i, j, acc);
        }
    }
    let q: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
    let mut a_in = DenseMatrix::zeros(m_in, n);
    for r in 0..m_in {
        for j in 0..n {
            a_in.set(r, j, rng.signed());
        }
    }
    let x0: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
    let ax0: Vec<f64> = (0..m_in)
        .map(|r| (0..n).map(|j| a_in.get(r, j) * x0[j]).sum())
        .collect();
    let b_in: Vec<f64> = (0..m_in).map(|i| ax0[i] + 0.5 + rng.unit()).collect();
    QpProblem {
        p,
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
    let prob = ill_conditioned_qp(80, 4.0, 20, 23);

    let run = |label: &str| {
        let sol = solve_qp(&prob, &Settings::<f64>::default());
        println!(
            "{}: status={:?} iters={} kkt={:.2e}",
            label,
            sol.status,
            sol.iters,
            kkt_residual(&prob, &sol)
        );
    };

    // BLAS on (default with the use-blas feature).
    iconic_linalg::blas::set_blas_enabled(true);
    run("blas on ");

    // BLAS off (the escalation test's process-wide toggle).
    iconic_linalg::blas::set_blas_enabled(false);
    run("blas off");

    iconic_linalg::blas::set_blas_enabled(true);
    run("blas on2");
}
