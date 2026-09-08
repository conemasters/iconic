#![allow(clippy::field_reassign_with_default)]
//! The low-rank (Woodbury) QP path disagrees with the dense path on the factor-model
//! QP. Check whether the disagreement depends on platform-BLAS dispatch, and report
//! where the two solutions diverge.

use iconic_bench::{factor_model_qp, kkt_residual};
use iconic_core::Settings;
use iconic_ipm::{solve_qp, solve_qp_lowrank};

fn main() {
    for blas in [true, false] {
        iconic_linalg::blas::set_blas_enabled(blas);
        println!("=== platform BLAS enabled: {blas} ===");
        let settings = Settings::<f64>::default();
        for &(n, r) in &[(300usize, 15usize), (700, 35)] {
            let prob = factor_model_qp(n, r, 0.1, 1234 + n as u64);
            let dense = solve_qp(&prob, &settings);
            match solve_qp_lowrank(&prob, &settings) {
                None => println!("  n={n} r={r}: low-rank path did not engage"),
                Some(low) => {
                    let dx = dense
                        .x
                        .iter()
                        .zip(low.x.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    println!(
                        "  n={n} r={r}: dense obj={:.8} ({:?}) kkt={:.2e} | low obj={:.8} ({:?}) kkt={:.2e} | |dx|inf={dx:.3e}",
                        dense.obj_val, dense.status, kkt_residual(&prob, &dense),
                        low.obj_val, low.status, kkt_residual(&prob, &low)
                    );
                }
            }
        }
    }
}
