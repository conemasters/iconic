#![allow(clippy::needless_range_loop)]
#![allow(clippy::field_reassign_with_default)]
//! Minimal repro for the equality-tied singleton-column block behavior.
//!
//! Shape (the huber epigraph split, without the algebraic-elimination
//! workaround): `min ½‖u‖² + δ·1ᵀ(p+n)  s.t.  u + p − n = Cx − d,  p,n ≥ 0`.
//! Columns of `u`, `p`, `n` are all singleton columns of the equality block
//! (each appears in exactly one row); `u` carries diagonal curvature `P=1`,
//! `x` carries none (bounded by the `p,n ≥ 0` slacks).
//!
//! Behavior: the raw QP path converges to
//! a self-consistent fixed point on this shape with the active-set duals
//! frozen at ~σ·z scale, pinning the stationarity residual at ~2.3e-8 — just
//! above the tight tolerance — while feasibility, complementarity and the
//! Newton direction itself are all exact. Honest `SolvedInaccurate`, exact
//! objective. The user-facing API path is unaffected: presolve's
//! `eliminate_auxiliary_vars` removes the singleton `u` block exactly and the
//! reduced problem converges (see the
//! `huber_shape_equality_singletons_solve_via_api` regression test).
//!
//! Run: `cargo run --release -p iconic-bench --example diag_singleton_eq -- 5`

use iconic_core::rng::SplitMix;
use iconic_core::{Cone, Settings};
use iconic_ipm::{solve_qp, QpProblem};
use iconic_linalg::dense::DenseMatrix;

fn huber_shape_qp(n_x: usize, m: usize, delta: f64, seed: u64) -> QpProblem<f64> {
    let mut rng = SplitMix::new(seed);
    // Variables: [x (n_x), u (m), p (m), n (m)]
    let n = n_x + 3 * m;
    // P: curvature on u only.
    let mut p = DenseMatrix::zeros(n, n);
    for i in n_x..n_x + m {
        p.set(i, i, 1.0);
    }
    // q: δ on p and n, zero elsewhere.
    let mut q = vec![0.0; n];
    for i in n_x + m..n {
        q[i] = delta;
    }
    // A_eq = [-C | I | I | -I], b_eq = -d. C is m×n_x, d random.
    let mut a_eq = DenseMatrix::zeros(m, n);
    let mut d = vec![0.0; m];
    for i in 0..m {
        d[i] = rng.signed();
        for j in 0..n_x {
            a_eq.set(i, j, -rng.signed());
        }
        a_eq.set(i, n_x + i, 1.0); // u_i
        a_eq.set(i, n_x + m + i, 1.0); // p_i
        a_eq.set(i, n_x + 2 * m + i, -1.0); // n_i
    }
    let b_eq: Vec<f64> = d.iter().map(|&di| -di).collect();
    // A_in: -p ≤ 0 and -n ≤ 0 (2m rows).
    let mut a_in = DenseMatrix::zeros(2 * m, n);
    for i in 0..m {
        a_in.set(i, n_x + m + i, -1.0);
        a_in.set(m + i, n_x + 2 * m + i, -1.0);
    }
    QpProblem {
        p,
        q,
        a_eq,
        b_eq,
        a_in,
        b_in: vec![0.0; 2 * m],
        a_eq_csr: None,
        a_in_csr: None,
    }
}

fn main() {
    let n_x: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    println!("n_x = {n_x}, sweeping m (huber-shape equality rows):");
    for m in (4..=32).step_by(2) {
        let prob = huber_shape_qp(n_x, m, 0.5, 42);
        let sol = solve_qp(&prob, &Settings::<f64>::default());
        // Also route through the conic engine (Zero + NonNeg cones).
        let n = prob.q.len();
        let me = prob.b_eq.len();
        let mi = prob.b_in.len();
        let mut a = DenseMatrix::zeros(me + mi, n);
        let mut b = vec![0.0f64; me + mi];
        for r in 0..me {
            for j in 0..n {
                a.set(r, j, prob.a_eq.get(r, j));
            }
            b[r] = prob.b_eq[r];
        }
        for r in 0..mi {
            for j in 0..n {
                a.set(me + r, j, prob.a_in.get(r, j));
            }
            b[me + r] = prob.b_in[r];
        }
        use iconic_api::ConeProgram;
        let cones = vec![Cone::Zero(me), Cone::NonNegative(mi)];
        let cp = ConeProgram {
            p: prob.p.clone(),
            q: prob.q.clone(),
            a,
            b,
            cones,
            a_csc: None,
        };
        let csol = iconic_api::solve(&cp, &Settings::<f64>::default());
        let (st, it, obj) = match csol {
            Ok(s) => (format!("{:?}", s.status), s.iters, s.obj_val),
            Err(e) => (format!("ERR {e:?}"), 0, f64::NAN),
        };
        // Direct conic-engine solve (bypasses the QP path entirely).
        let cones_d = vec![iconic_ipm::conic::Cone::NonNeg(mi)]; // equalities from a_eq
        let cd = iconic_ipm::conic::solve_cone_qp(&prob, &cones_d, &Settings::<f64>::default());
        println!("  m={m:3}: QP {:?} it={:>3} obj={:.6}  |  API {st:<12} it={it:>3} obj={obj:.6}  |  cone {:?} it={:>3} obj={:.6}",
            sol.status, sol.iters, sol.obj_val, cd.status, cd.iters, cd.obj_val);
    }
}
