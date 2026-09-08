//! Benchmark: dense augmented KKT (faer Ldlt/Lblt) vs sparse augmented KKT
//! (supernodal LDLT) on dense, rank-deficient QPs (`mi < n`, dense `P`/`A_in`).
//!
//! These are exactly the problems the condensed-gram path struggles with: the
//! gram `A_inᵀ(Z/S)A_in` has rank ≤ mi < n, so its nullspace relies entirely on
//! static regularization. Routing through the quasidefinite augmented system
//! instead avoids that rank deficiency altogether.
//!
//! Run with: `cargo run --release --example bench_rankdef -p iconic-ipm`

use iconic_core::Settings;
use iconic_ipm::{solve_qp_with_termination, QpProblem, TermScale};
use iconic_linalg::DenseMatrix;
use std::time::Instant;

fn rng_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A dense random QP `min ½xᵀPx + qᵀx s.t. A_in·x ≤ b_in` with `mi < n` (fewer
/// inequality rows than variables), so the condensed gram is rank-deficient.
fn make_problem(n: usize, mi: usize, seed: u64) -> QpProblem<f64> {
    let mut state = seed;
    let mut l = vec![0.0f64; n * n];
    for v in l.iter_mut() {
        *v = rng_next(&mut state) * 0.1;
    }
    let mut p = DenseMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += l[i * n + k] * l[j * n + k];
            }
            p.set(i, j, acc);
        }
        p.set(i, i, p.get(i, i) + 0.01);
    }
    let q: Vec<f64> = (0..n).map(|_| rng_next(&mut state)).collect();
    let mut a_in = DenseMatrix::<f64>::zeros(mi, n);
    for i in 0..mi {
        for j in 0..n {
            a_in.set(i, j, rng_next(&mut state));
        }
    }
    let b_in: Vec<f64> = (0..mi)
        .map(|_| rng_next(&mut state).abs() * 5.0 + 1.0)
        .collect();
    QpProblem {
        p,
        q,
        a_eq: DenseMatrix::<f64>::zeros(0, n),
        b_eq: vec![],
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr: None,
    }
}

fn main() {
    println!(
        "{:>6} {:>6} {:>14} {:>14} {:>8}",
        "n", "mi", "dense (ms)", "sparse (ms)", "speedup"
    );
    for &(n, mi) in &[
        (100usize, 80usize),
        (150, 100),
        (200, 150),
        (300, 200),
        (400, 250),
        (600, 400),
        (800, 500),
    ] {
        let prob = make_problem(n, mi, 42 + n as u64);
        let term = TermScale::identity(n, 0, mi);

        // Dense augmented path: auto-enabled since mi < n.
        let settings_dense = Settings::<f64> {
            sparse_kkt: false,
            ..Default::default()
        };
        let t0 = Instant::now();
        let sol_dense = solve_qp_with_termination(&prob, &settings_dense, &term);
        let t_dense = t0.elapsed();

        // Forced sparse supernodal path (pre-existing behavior).
        let settings_sparse = Settings::<f64> {
            sparse_kkt: true,
            ..Default::default()
        };
        let t0 = Instant::now();
        let sol_sparse = solve_qp_with_termination(&prob, &settings_sparse, &term);
        let t_sparse = t0.elapsed();

        assert_eq!(
            sol_dense.status, sol_sparse.status,
            "status mismatch at n={n}"
        );
        assert!(
            (sol_dense.obj_val - sol_sparse.obj_val).abs()
                < 1e-4 * sol_dense.obj_val.abs().max(1.0),
            "objective mismatch at n={n}: dense={} sparse={}",
            sol_dense.obj_val,
            sol_sparse.obj_val
        );

        let td_ms = t_dense.as_secs_f64() * 1000.0;
        let ts_ms = t_sparse.as_secs_f64() * 1000.0;
        println!(
            "{n:>6} {mi:>6} {td_ms:>14.2} {ts_ms:>14.2} {:>7.2}x",
            ts_ms / td_ms.max(1e-9)
        );
    }
}
