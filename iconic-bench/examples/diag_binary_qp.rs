//! Where does an unconstrained binary quadratic lose its optimum?
//!
//! Reports, for a small indefinite instance, the enumerated optimum alongside what
//! `solve_mip` returns with and without the convexifying reformulation, plus whether the
//! returned point is even the best one the search saw.

use iconic_core::Cone;
use iconic_linalg::DenseMatrix;
use iconic_mip::{solve_mip, MipProblem, MipSettings, VarType};

fn fixture(n: usize, rows: bool) -> MipProblem<f64> {
    let mut p = DenseMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        p.set(i, i, -2.0 - (i as f64) * 0.5);
        for j in (i + 1)..n {
            let v = 1.0 + ((i * 7 + j * 3) % 5) as f64 * 0.25;
            p.set(i, j, v);
            p.set(j, i, v);
        }
    }
    // Optionally add a genuinely binding row (sum x <= 2), so no column is empty and
    // the bound-derived reductions have something to work with.
    let (m, a, b, cones) = if rows {
        let mut a = DenseMatrix::<f64>::zeros(1, n);
        for j in 0..n {
            a.set(0, j, 1.0);
        }
        (1usize, a, vec![2.0], vec![Cone::NonNegative(1)])
    } else {
        (0usize, DenseMatrix::<f64>::zeros(0, n), vec![], vec![])
    };
    let _ = m;
    MipProblem {
        p,
        q: (0..n)
            .map(|j| {
                if std::env::var_os("ZEROQ").is_some() {
                    0.0
                } else {
                    0.75 - (j as f64) * 0.3
                }
            })
            .collect(),
        a,
        b,
        cones,
        var_types: vec![VarType::Binary; n],
        lb: vec![0.0; n],
        ub: vec![1.0; n],
        warm_start: None,
    }
}

fn objective(p: &DenseMatrix<f64>, q: &[f64], x: &[f64]) -> f64 {
    let n = x.len();
    let mut quad = 0.0;
    for i in 0..n {
        for j in 0..n {
            quad += x[i] * p.get(i, j) * x[j];
        }
    }
    0.5 * quad + (0..n).map(|j| q[j] * x[j]).sum::<f64>()
}

fn main() {
    for rows in [false, true] {
        for n in [4usize, 6, 8] {
            let prob = fixture(n, rows);
            let mut brute = f64::INFINITY;
            let mut arg = vec![];
            for mask in 0u32..(1u32 << n) {
                let x: Vec<f64> = (0..n).map(|j| ((mask >> j) & 1) as f64).collect();
                let v = objective(&prob.p, &prob.q, &x);
                if v < brute {
                    brute = v;
                    arg = x;
                }
            }
            let settings = MipSettings::<f64> {
                max_nodes: 20_000,
                max_time: 30.0,
                ..Default::default()
            };
            let sol = solve_mip(&prob, &settings);
            let at_x = if sol.x.len() == n {
                objective(&prob.p, &prob.q, &sol.x)
            } else {
                f64::NAN
            };
            println!(
                "rows={rows} n={n}: brute={brute:.6} at {arg:?}\n   solve_mip -> {:?} obj_val={:.6} q(x)={:.6} bound={:.6} x={:?}",
                sol.status, sol.obj_val, at_x, sol.best_bound, sol.x
            );
        }
    }
}
