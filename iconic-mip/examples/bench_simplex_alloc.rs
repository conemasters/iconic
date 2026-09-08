//! Measures the effect of eliminating iconic-simplex's per-iteration `bt_rho` heap
//! allocation and the `rho` clone in `dual_loop`'s iterative refinement — every
//! other LU scratch buffer in `DualSolver` is already a pre-allocated struct
//! field; these two were the exception.
//!
//! Run with: `cargo run --release --example bench_simplex_alloc -p iconic-mip`

use iconic_core::Cone;
use iconic_linalg::DenseMatrix;
use iconic_mip::{solve_mip, MipProblem, MipSettings, VarType};
use std::time::Instant;

fn rng_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Multi-dimensional 0/1 knapsack (k constraints) — defeats the single-constraint
/// DP presolve shortcut, so B&B genuinely explores the tree (thousands of dual
/// simplex iterations across thousands of nodes).
fn gen_multi_knapsack(n: usize, k: usize, seed: u64) -> MipProblem<f64> {
    let mut state = seed;
    let profits: Vec<f64> = (0..n).map(|_| 10.0 + rng_next(&mut state) * 90.0).collect();
    let mut a_data = vec![0.0; k * n];
    let mut capacities = vec![0.0; k];
    for i in 0..k {
        let mut row_sum = 0.0;
        for j in 0..n {
            let w = 1.0 + rng_next(&mut state) * 29.0;
            a_data[i * n + j] = w;
            row_sum += w;
        }
        capacities[i] = row_sum * 0.4;
    }
    MipProblem {
        p: DenseMatrix::zeros(n, n),
        q: profits.iter().map(|&p| -p).collect(),
        a: DenseMatrix::from_row_major(k, n, a_data),
        b: capacities,
        cones: vec![Cone::NonNegative(k)],
        var_types: vec![VarType::Binary; n],
        lb: vec![0.0; n],
        ub: vec![1.0; n],
        warm_start: None,
    }
}

fn main() {
    for &(n, k) in &[(30usize, 5usize), (40, 6), (60, 15)] {
        let prob = gen_multi_knapsack(n, k, 42 + n as u64);
        let settings = MipSettings::<f64>::default();
        let mut best = f64::INFINITY;
        let mut nodes = 0;
        for _ in 0..3 {
            let t0 = Instant::now();
            let sol = solve_mip(&prob, &settings);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            if elapsed < best {
                best = elapsed;
                nodes = sol.nodes;
            }
        }
        println!("n={n:3} k={k:2}  best={best:9.2}ms  nodes={nodes}");
    }
}
