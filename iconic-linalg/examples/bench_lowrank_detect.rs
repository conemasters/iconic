//! Benchmark: `low_rank_plus_diag`'s detection cost on a genuinely full-rank matrix
//! (the common case — most dense `P` are not factor models, so the detector runs and
//! correctly bails on nearly every dense QP with n >= 64).
//!
//! Run with: `cargo run --release --example bench_lowrank_detect -p iconic-linalg`

use iconic_linalg::dense::DenseMatrix;
use iconic_linalg::lowrank::low_rank_plus_diag;
use std::time::Instant;

fn rng_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A genuinely full-rank dense PSD matrix `P = L Lᵀ + 0.01 I` with `L` an `n×n` random
/// matrix (not `n×r`, r≪n) — the detector should quickly conclude "not low-rank" here.
fn full_rank_psd(n: usize, seed: u64) -> DenseMatrix<f64> {
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
    p
}

fn main() {
    println!("{:>6} {:>12} {:>10}", "n", "time (ms)", "result");
    for &n in &[64usize, 100, 150, 200, 300, 400] {
        let p = full_rank_psd(n, 42 + n as u64);
        let max_rank = (n / 8).max(4).min(n.saturating_sub(1));
        let t0 = Instant::now();
        let result = low_rank_plus_diag(&p, 1e-10, max_rank);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        // Mirror the caller's actual gate (iconic_ipm::solve_qp_lowrank): a genuine
        // factor-model fit requires offdiag_rel below ~1e-9; the raw Some/None from
        // low_rank_plus_diag alone doesn't reflect whether the fit is trustworthy.
        let verdict = match &result {
            Some(lr) if lr.offdiag_rel < 1e-9 => "GENUINE FIT",
            Some(_) => "rejected (offdiag_rel too large — correct, not a factor model)",
            None => "None",
        };
        let offdiag = result.as_ref().map(|lr| lr.offdiag_rel).unwrap_or(0.0);
        println!("{n:>6} {elapsed:>12.3}   offdiag_rel={offdiag:>10.3e}   {verdict}");
    }
}
