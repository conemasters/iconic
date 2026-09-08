#![allow(clippy::needless_range_loop)]
#![allow(clippy::field_reassign_with_default)]
//! Does the MIP's quadratic objective reach the solver the way `compute_objective` reads it?
//!
//! `compute_objective` evaluates `½ xᵀPx + qᵀx` over the *full* matrix. If the relaxation
//! interprets the same `P` differently -- upper triangle only, or without the ½ -- then the
//! search optimises one function while the incumbent is scored by another, and every bound
//! it derives is a bound on the wrong problem.
//!
//! Each case below is a convex box-QP whose minimiser is known by hand, run through
//! `solve_mip` with continuous variables so nothing but the QP path is exercised.

use iconic_linalg::DenseMatrix;
use iconic_mip::{solve_mip, MipProblem, MipSettings, VarType};

fn run(name: &str, p: DenseMatrix<f64>, q: Vec<f64>, lb: Vec<f64>, ub: Vec<f64>, expect: f64) {
    let n = q.len();
    let prob = MipProblem {
        p,
        q,
        a: DenseMatrix::<f64>::zeros(0, n),
        b: vec![],
        cones: vec![],
        var_types: vec![VarType::Continuous; n],
        lb,
        ub,
        warm_start: None,
    };
    let mut settings = MipSettings::<f64>::default();
    settings.max_time = 10.0;
    let sol = solve_mip(&prob, &settings);
    let ok = (sol.obj_val - expect).abs() < 1e-6;
    println!(
        "{name:34} expect {expect:9.5}  got {:9.5}  {:?}  x={:?}  {}",
        sol.obj_val,
        sol.status,
        sol.x
            .iter()
            .map(|v| (v * 1e5).round() / 1e5)
            .collect::<Vec<_>>(),
        if ok { "OK" } else { "<-- MISMATCH" }
    );
}

fn main() {
    // 1. Diagonal only: min ½(2x0² + 2x1²) - 3x0 - 3x1 over [0,1]².
    //    Unconstrained min at x = 1.5 each, so the box optimum is (1,1): ½·4 - 6 = -4.
    let mut p = DenseMatrix::<f64>::zeros(2, 2);
    p.set(0, 0, 2.0);
    p.set(1, 1, 2.0);
    run(
        "diagonal P",
        p,
        vec![-3.0, -3.0],
        vec![0.0; 2],
        vec![1.0; 2],
        -4.0,
    );

    // 2. With an off-diagonal: P = [[2,1],[1,2]], q = [-3,-3].
    //    Px = -q gives x = (1,1), inside the box; value ½·6 - 6 = -3.
    //    Reading only the upper triangle would make the cross term half as strong and the
    //    answer differ, so this is the case that separates the conventions.
    let mut p = DenseMatrix::<f64>::zeros(2, 2);
    p.set(0, 0, 2.0);
    p.set(1, 1, 2.0);
    p.set(0, 1, 1.0);
    p.set(1, 0, 1.0);
    run(
        "symmetric P, off-diagonal",
        p,
        vec![-3.0, -3.0],
        vec![0.0; 2],
        vec![1.0; 2],
        -3.0,
    );

    // 3. Interior optimum, so the answer is not simply a corner: P = [[2,1],[1,2]],
    //    q = [-1,-1] gives x = (1/3, 1/3), value ½·(2/3) - 2/3 = -1/3.
    let mut p = DenseMatrix::<f64>::zeros(2, 2);
    p.set(0, 0, 2.0);
    p.set(1, 1, 2.0);
    p.set(0, 1, 1.0);
    p.set(1, 0, 1.0);
    run(
        "interior optimum",
        p,
        vec![-1.0, -1.0],
        vec![0.0; 2],
        vec![1.0; 2],
        -1.0 / 3.0,
    );

    // 4. Nearly singular PSD, which is what convexifying an indefinite objective produces
    //    (the shift drives the smallest eigenvalue to ~0): P = [[1,1],[1,1]], q = [0,-1].
    //    Minimise ½(x0+x1)² - x1: at x0 = 0 it is ½x1² - x1, least at x1 = 1, value -1/2.
    let mut p = DenseMatrix::<f64>::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            p.set(i, j, 1.0);
        }
    }
    run(
        "singular PSD",
        p,
        vec![0.0, -1.0],
        vec![0.0; 2],
        vec![1.0; 2],
        -0.5,
    );

    // 5. The convexified form of the indefinite fixture, as a *continuous* QP. Its box
    //    minimum is -2.0781016468 (computed independently). Run here to separate "the QP
    //    relaxation is wrong on this matrix" from "the branch-and-bound around it is".
    let n = 4usize;
    let mut p = DenseMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        p.set(i, i, -2.0 - (i as f64) * 0.5);
        for j in (i + 1)..n {
            let v = 1.0 + ((i * 7 + j * 3) % 5) as f64 * 0.25;
            p.set(i, j, v);
            p.set(j, i, v);
        }
    }
    let lam = iconic_linalg::eig::min_eigenvalue(&p);
    let shift = -lam * (1.0 + 1e-9) + 1e-12;
    let mut pc = p.clone();
    let mut qc: Vec<f64> = (0..n).map(|j| 0.75 - (j as f64) * 0.3).collect();
    for j in 0..n {
        pc.set(j, j, pc.get(j, j) + shift);
        qc[j] -= shift * 0.5;
    }
    println!("   (lambda_min {lam:.6}, shift {shift:.6})");
    run(
        "convexified fixture, continuous",
        pc,
        qc,
        vec![0.0; n],
        vec![1.0; n],
        -2.0781016468,
    );
}
