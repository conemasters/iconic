//! Deterministic generators for exponential- / power-cone benchmark programs.
//!
//! The generators live here (not in a test module) so both the `iconic-ipm` test
//! battery and the `iconic-bench` suite construct the *same* instances from the
//! *same* code — one copy, one seed table. The seeds are fixed and load-bearing:
//! the boxed log-sum-exp family's optima are Clarabel-verified against exactly
//! these draws, so changing a seed would invalidate the reference table.

use crate::nonsym::NsCone;
use crate::QpProblem;
use iconic_core::rng::Lcg;
use iconic_linalg::DenseMatrix;

/// Build the maximum-entropy program with `n` variables:
/// `max Σ −xᵢ·log xᵢ s.t. Σxᵢ = 1`. Variables `(x₁..xₙ, t₁..tₙ)`; cone i is
/// `(tᵢ, xᵢ, 1) ∈ K_exp ⇔ −xᵢ log xᵢ ≥ tᵢ`. The objective `min −Σtᵢ` then maximizes
/// the entropy. The unique optimum is the uniform distribution `xᵢ = 1/n`, entropy
/// `log n`, objective `−log n`.
pub fn max_entropy(n: usize) -> (QpProblem<f64>, Vec<NsCone>) {
    let nv = 2 * n; // x₁..xₙ, t₁..tₙ
    let mi = 3 * n;
    let mut a_in = DenseMatrix::<f64>::zeros(mi, nv);
    let mut b_in = vec![0.0; mi];
    for i in 0..n {
        // cone i rows: s0 = tᵢ, s1 = xᵢ, s2 = 1.
        a_in.set(3 * i, n + i, -1.0); // s0 = tᵢ
        a_in.set(3 * i + 1, i, -1.0); // s1 = xᵢ
        b_in[3 * i + 2] = 1.0; // s2 = 1
    }
    let mut q = vec![0.0; nv];
    for i in 0..n {
        q[n + i] = -1.0; // min −Σtᵢ
    }
    let mut a_eq = DenseMatrix::<f64>::zeros(1, nv);
    for i in 0..n {
        a_eq.set(0, i, 1.0); // Σxᵢ = 1
    }
    let prob = QpProblem {
        p: DenseMatrix::zeros(nv, nv),
        q,
        a_eq,
        b_eq: vec![1.0],
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr: None,
    };
    let cones = vec![NsCone::Exp; n];
    (prob, cones)
}

/// `max −Σ xᵢ log(xᵢ/wᵢ) s.t. Σxᵢ = 1` with `wᵢ = e^{skew·i}`: the optimum is
/// `x ∝ w`, objective `−ln Σwᵢ` (closed form).
pub fn weighted_entropy(n: usize, skew: f64) -> (QpProblem<f64>, Vec<NsCone>) {
    let nv = 2 * n;
    let mi = 3 * n;
    let mut a_in = DenseMatrix::<f64>::zeros(mi, nv);
    let mut b_in = vec![0.0; mi];
    for i in 0..n {
        a_in.set(3 * i, n + i, -1.0);
        a_in.set(3 * i + 1, i, -1.0);
        b_in[3 * i + 2] = 1.0;
    }
    let mut q = vec![0.0; nv];
    for i in 0..n {
        q[n + i] = -1.0;
    }
    let mut a_eq = DenseMatrix::<f64>::zeros(1, nv);
    for i in 0..n {
        a_eq.set(0, i, 1.0);
    }
    for i in 0..n {
        let w = (skew * i as f64).exp();
        q[i] = -(w.ln());
    }
    let prob = QpProblem {
        p: DenseMatrix::zeros(nv, nv),
        q,
        a_eq,
        b_eq: vec![1.0],
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr: None,
    };
    (prob, vec![NsCone::Exp; n])
}

/// Log-sum-exp with a box on x: `min Σⱼ tⱼ s.t. exp(aⱼᵀx + bⱼ) ≤ tⱼ`,
/// `−1 ≤ xᵢ ≤ 1` (well-posed; the unboxed form has infimum 0 at infinity).
/// Same generator as the Clarabel-verified probe (seed 1000 + n + m — fixed).
pub fn log_sum_exp(n: usize, m: usize, scale: f64) -> (QpProblem<f64>, Vec<NsCone>) {
    let nv = n + m;
    let mi = 3 * m + 2 * n;
    let mut a_in = DenseMatrix::<f64>::zeros(mi, nv);
    let mut b_in = vec![0.0; mi];
    let mut cones: Vec<NsCone> = Vec::new();
    let mut rng = Lcg::new(1000 + n as u64 + m as u64);
    for j in 0..m {
        cones.push(NsCone::Exp);
        for i in 0..n {
            a_in.set(3 * j, i, -rng.signed() * scale);
        }
        b_in[3 * j] = rng.signed() * 0.5;
        b_in[3 * j + 1] = 1.0;
        a_in.set(3 * j + 2, n + j, -1.0);
    }
    cones.push(NsCone::NonNeg(2 * n));
    let base = 3 * m;
    for i in 0..n {
        b_in[base + 2 * i] = 1.0;
        a_in.set(base + 2 * i, i, 1.0);
        b_in[base + 2 * i + 1] = 1.0;
        a_in.set(base + 2 * i + 1, i, -1.0);
    }
    let mut q = vec![0.0; nv];
    for j in 0..m {
        q[n + j] = 1.0;
    }
    let prob = QpProblem {
        p: DenseMatrix::zeros(nv, nv),
        q,
        a_eq: DenseMatrix::zeros(0, nv),
        b_eq: vec![],
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr: None,
    };
    (prob, cones)
}

/// Projection onto the power cone: `min ½‖x−c‖²` over `x ∈ K_α` with `c = (1,1,0)`
/// (outside the cone for every α — the boundary-active case). In `Ax+s=b` form
/// (`A = −I, b = 0` so `s = x`), `q = −c`. The closed form is `obj = −½‖c‖² = −1`.
pub fn pow_proj(alpha: f64) -> (QpProblem<f64>, Vec<NsCone>) {
    let a_in =
        DenseMatrix::from_row_major(3, 3, vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]);
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        q: vec![-1.0, -1.0, 0.0],
        a_eq: DenseMatrix::zeros(0, 3),
        b_eq: vec![],
        a_in,
        b_in: vec![0.0, 0.0, 0.0],
        a_eq_csr: None,
        a_in_csr: None,
    };
    (prob, vec![NsCone::Power(alpha)])
}

/// A boundary-active power-cone equality program: `max x^α y^(1−α) s.t. x+y = 1`
/// (`min −z s.t. (x, y, z) ∈ K_α, x+y = 1`). The optimum sits on the curved
/// boundary of the dual cone; closed form `obj = −α^α (1−α)^(1−α)`.
/// `α = 0.5` is the boundary-active dual-boundary case (`obj = −0.5`).
pub fn pow_eq(alpha: f64) -> (QpProblem<f64>, Vec<NsCone>) {
    let a_in =
        DenseMatrix::from_row_major(3, 3, vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]);
    let prob = QpProblem {
        p: DenseMatrix::zeros(3, 3),
        q: vec![0.0, 0.0, -1.0],
        a_eq: DenseMatrix::from_row_major(1, 3, vec![1.0, 1.0, 0.0]),
        b_eq: vec![1.0],
        a_in,
        b_in: vec![0.0, 0.0, 0.0],
        a_eq_csr: None,
        a_in_csr: None,
    };
    (prob, vec![NsCone::Power(alpha)])
}

/// GenPower geometric-mean cone, non-binding: `min ½‖x−c‖²` with `c` strictly
/// INSIDE the cone (base `(c₀·…·c_{k−1})^{1/k} > max tail`), so the unconstrained
/// minimizer `x* = c` is optimal, `obj = −½‖c‖²`. In `Ax+s=b` form (`A = −I, b = 0`
/// so `s = x`). `α = (1/k, …, 1/k)` uniform over the base of dimension `k`, tail of
/// dimension `tail`. Exercises a live GenPower cone end-to-end (P ≠ 0, the
/// nonsymmetric IPM must respect the cone's shape throughout, it just does not need
/// to reach the boundary).
pub fn genpow_geomean(k: usize, tail: usize) -> (QpProblem<f64>, Vec<NsCone>) {
    let d = k + tail;
    let mut c: Vec<f64> = vec![8.0; k];
    c.extend(vec![2.0; tail]);
    let mut a_in = DenseMatrix::<f64>::zeros(d, d);
    for i in 0..d {
        a_in.set(i, i, -1.0);
    }
    let mut p = DenseMatrix::<f64>::zeros(d, d);
    for i in 0..d {
        p.set(i, i, 1.0);
    }
    let alpha: Vec<f64> = vec![1.0 / k as f64; k];
    let prob = QpProblem {
        p,
        q: c.iter().map(|&v| -v).collect(),
        a_eq: DenseMatrix::zeros(0, d),
        b_eq: vec![],
        a_in,
        b_in: vec![0.0; d],
        a_eq_csr: None,
        a_in_csr: None,
    };
    (prob, vec![NsCone::GenPower(alpha, tail)])
}

/// Moment-constrained entropy: `max −Σxᵢlog xᵢ s.t. Σxᵢ = 1, Σvᵢxᵢ = c`,
/// `vᵢ = i/(n−1)`. The optimum is the exponential-tilt distribution
/// `xᵢ ∝ e^{λvᵢ}`; for `c` near the boundary of the feasible moment range
/// `[0, 1]` the tilt is extreme (many near-zero atoms) — the historical
/// "many active exp cones" stall class. No closed form; the engine stalls
/// honestly (`MaxIterations`, best iterate in a sane range).
pub fn moment_entropy(n: usize, c: f64) -> (QpProblem<f64>, Vec<NsCone>) {
    let nv = 2 * n;
    let mi = 3 * n;
    let mut a_in = DenseMatrix::<f64>::zeros(mi, nv);
    let mut b_in = vec![0.0; mi];
    for i in 0..n {
        a_in.set(3 * i, n + i, -1.0);
        a_in.set(3 * i + 1, i, -1.0);
        b_in[3 * i + 2] = 1.0;
    }
    let mut q = vec![0.0; nv];
    for i in 0..n {
        q[n + i] = -1.0;
    }
    let mut a_eq = DenseMatrix::<f64>::zeros(2, nv);
    for i in 0..n {
        a_eq.set(0, i, 1.0);
        a_eq.set(1, i, i as f64 / (n as f64 - 1.0));
    }
    let prob = QpProblem {
        p: DenseMatrix::zeros(nv, nv),
        q,
        a_eq,
        b_eq: vec![1.0, c],
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr: None,
    };
    (prob, vec![NsCone::Exp; n])
}
