//! Measures DenseMatrix::matvec/matvec_t: row-slice iteration (bounds-check-
//! eliding, no unsafe) vs indexed self.get(i,j)/x[j] access.
use iconic_linalg::dense::DenseMatrix;
use std::time::Instant;

fn rng_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    for &(n, m) in &[(50usize, 50usize), (100, 80), (200, 150), (400, 300)] {
        let mut state = 42u64;
        let data: Vec<f64> = (0..m * n).map(|_| rng_next(&mut state)).collect();
        let mat = DenseMatrix::from_row_major(m, n, data);
        let x: Vec<f64> = (0..n).map(|_| rng_next(&mut state)).collect();
        let xt: Vec<f64> = (0..m).map(|_| rng_next(&mut state)).collect();
        let reps = 200_000_000 / (m * n).max(1);

        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(mat.matvec(&x));
        }
        let e1 = t0.elapsed().as_secs_f64() * 1e9 / reps as f64;

        let t0 = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(mat.matvec_t(&xt));
        }
        let e2 = t0.elapsed().as_secs_f64() * 1e9 / reps as f64;

        println!("n={n:4} m={m:4}  reps={reps:8}  matvec={e1:8.1}ns  matvec_t={e2:8.1}ns");
    }
}
