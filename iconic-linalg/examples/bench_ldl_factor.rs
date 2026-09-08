//! Measures ldl_factor_into's O(n^3) factorization loop with unchecked indexing
//! vs bounds-checked DenseMatrix::get/set, matching LdlFactor::solve's own
//! already-shipped unchecked-indexing optimization.
use iconic_linalg::dense::DenseMatrix;
use iconic_linalg::ldl::ldl_factor;
use std::time::Instant;

fn rng_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn random_spd(n: usize, seed: u64) -> DenseMatrix<f64> {
    let mut state = seed;
    let mut m = DenseMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let v = rng_next(&mut state) * 2.0 - 1.0;
            m.set(i, j, v);
            m.set(j, i, v);
        }
        m.set(i, i, m.get(i, i) + n as f64);
    }
    m
}

fn main() {
    for &n in &[8usize, 16, 32, 47, 64] {
        let mats: Vec<_> = (0..2000).map(|i| random_spd(n, 42 + i)).collect();
        let t0 = Instant::now();
        for m in &mats {
            let f = ldl_factor(m, 1e-14).unwrap();
            std::hint::black_box(&f);
        }
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "n={n:3}  2000 factors: {elapsed:8.2}ms  ({:.4}ms/factor)",
            elapsed / 2000.0
        );
    }
}
