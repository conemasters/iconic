//! Power-cone primitives for a nonsymmetric interior-point treatment.
//!
//! The 3-D power cone with parameter `α ∈ (0, 1)` is
//! `K_α = cl { (x,y,z) : x^α·y^(1-α) ≥ |z|,  x > 0, y > 0 }`.
//! CVXPY's `PowCone3D` maps to this with `α` taken from the constraint.
//!
//! The logarithmically-homogeneous self-concordant barrier (parameter ν = 2) is
//! `f(x,y,z) = −log(x^{2α}·y^{2(1-α)} − z²) − (1-α)·log(x) − α·log(y)`.
//!
//! Like the exponential cone, the power cone is not self-dual and has no
//! Nesterov–Todd scaling; the nonsymmetric IPM works directly with its barrier.

use iconic_core::Scalar;

/// Centrality function: `φ = x^{2α}·y^{2(1-α)} − z²`.  Strictly interior iff `φ > 0, x > 0, y > 0`.
fn phi<T: Scalar>(s: &[T], alpha: T) -> T {
    let one = T::one();
    let two = T::from_f64(2.0).expect("scalar literal");
    let a2 = two * alpha;
    let b2 = two * (one - alpha);
    s[0].powf(a2) * s[1].powf(b2) - s[2] * s[2]
}

/// Primal-cone margin: `min(φ, x, y)`.
pub fn margin<T: Scalar>(s: &[T], alpha: T) -> T {
    if s[0] <= T::zero() || s[1] <= T::zero() {
        return s[0].min(s[1]);
    }
    phi(s, alpha).min(s[0]).min(s[1])
}

/// Is `s` in the primal power cone (with tolerance `tol`)?
pub fn in_cone<T: Scalar>(s: &[T], alpha: T, tol: T) -> bool {
    s[0] > tol && s[1] > tol && phi(s, alpha) > tol
}

/// Largest `α ≥ 0` keeping `s + α·ds` in the primal power cone, via bisection.
pub fn max_step<T: Scalar>(s: &[T], ds: &[T], alpha: T) -> T {
    crate::nonsym::curved_max_step(|p| margin(p, alpha), s, ds)
}

/// Barrier value `f(s) = −log(φ) − (1−α)·log(x) − α·log(y)` (requires `s` strictly
/// interior).
pub fn barrier<T: Scalar>(s: &[T], alpha: T) -> T {
    let one = T::one();
    -phi(s, alpha).ln() - (one - alpha) * s[0].ln() - alpha * s[1].ln()
}

/// Signed margin in the **dual** power cone. The dual of
/// `K_α = {(x,y,z) : x^α·y^(1−α) ≥ |z|}` is the *larger* cone
/// `K*_α = {(u,v,w) : u ≥ 0, v ≥ 0, (u/α)^α·(v/(1−α))^(1−α) ≥ |w|}`
/// (derived from the boundary critical-point conditions
/// `u = |w|·α·x^(α−1)y^(1−α)`, `v = |w|·(1−α)·x^α·y^(−α)`, which exist exactly
/// when `(u/α)^α(v/(1−α))^(1−α) = |w|`; verified numerically: `−∇f(s) ∈ int K*`
/// for every interior `s`). Positive iff strictly inside.
fn dual_margin<T: Scalar>(d: &[T], alpha: T) -> T {
    let zero = T::zero();
    let one = T::one();
    let (u, v, w) = (d[0], d[1], d[2]);
    if u <= zero || v <= zero {
        return u.min(v);
    }
    let lhs = (u / alpha).powf(alpha) * (v / (one - alpha)).powf(one - alpha);
    lhs - w.abs()
}

/// Largest `α ≥ 0` keeping the ray `d + α·dd` in the **dual** power cone, via
/// bisection on [`dual_margin`] (the curved dual boundary has no closed-form
/// crossing in general). The step must use the dual cone's own boundary: the
/// primal cone `K_α ⊂ K*_α`, so the primal-boundary step (what the engine used
/// before) is sound but conservative — it collapses to zero on rays the true
/// dual step permits, freezing the dual iterate on boundary-active programs.
pub fn max_step_dual<T: Scalar>(d: &[T], dd: &[T], alpha: T) -> T {
    crate::nonsym::curved_max_step(|p| dual_margin(p, alpha), d, dd)
}

/// Barrier gradient `∇f(s)` (3-vector).
pub fn grad<T: Scalar>(s: &[T], alpha: T) -> [T; 3] {
    let one = T::one();
    let two = T::from_f64(2.0).expect("scalar literal");
    let a2 = two * alpha;
    let b2 = two * (one - alpha);
    let phi_val = phi(s, alpha);
    let inv_phi = one / phi_val;
    // ∂φ/∂x = 2α·x^{2α-1}·y^{2(1-α)}
    let dphi_dx = a2 * s[0].powf(a2 - one) * s[1].powf(b2);
    // ∂φ/∂y = 2(1-α)·x^{2α}·y^{2(1-α)-1}
    let dphi_dy = b2 * s[0].powf(a2) * s[1].powf(b2 - one);
    // ∂φ/∂z = −2z
    let dphi_dz = -two * s[2];
    [
        -inv_phi * dphi_dx - (one - alpha) / s[0],
        -inv_phi * dphi_dy - alpha / s[1],
        -inv_phi * dphi_dz,
    ]
}

/// Barrier Hessian `∇²f(s)` (symmetric 3×3, row-major).
pub fn hess<T: Scalar>(s: &[T], alpha: T) -> [[T; 3]; 3] {
    let one = T::one();
    let two = T::from_f64(2.0).expect("scalar literal");
    let a2 = two * alpha;
    let b2 = two * (one - alpha);
    let phi_val = phi(s, alpha);
    let inv_phi = one / phi_val;
    let inv_phi2 = inv_phi * inv_phi;

    // First derivatives of φ
    let dphi_dx = a2 * s[0].powf(a2 - one) * s[1].powf(b2);
    let dphi_dy = b2 * s[0].powf(a2) * s[1].powf(b2 - one);
    let dphi_dz = -two * s[2];

    // Second derivatives: ∂²φ/∂x², ∂²φ/∂x∂y, etc.
    let d2phi_dx2 = a2 * (a2 - one) * s[0].powf(a2 - two) * s[1].powf(b2);
    let d2phi_dy2 = b2 * (b2 - one) * s[0].powf(a2) * s[1].powf(b2 - two);
    let d2phi_dxy = a2 * b2 * s[0].powf(a2 - one) * s[1].powf(b2 - one);
    let d2phi_dz2 = -two;

    // H_ab = −φ_ab/φ + φ_a·φ_b/φ²  (+ diag contributions from −log(x), −log(y))
    let h00 = -inv_phi * d2phi_dx2 + inv_phi2 * dphi_dx * dphi_dx + (one - alpha) / (s[0] * s[0]);
    let h01 = -inv_phi * d2phi_dxy + inv_phi2 * dphi_dx * dphi_dy;
    let h02 = inv_phi2 * dphi_dx * dphi_dz; // φ_xz = 0
    let h11 = -inv_phi * d2phi_dy2 + inv_phi2 * dphi_dy * dphi_dy + alpha / (s[1] * s[1]);
    let h12 = inv_phi2 * dphi_dy * dphi_dz; // φ_yz = 0
    let h22 = -inv_phi * d2phi_dz2 + inv_phi2 * dphi_dz * dphi_dz; // no −log(z) term

    [[h00, h01, h02], [h01, h11, h12], [h02, h12, h22]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membership() {
        let alpha = 0.5;
        // (0.25, 0.25, 0) — x^0.5 * y^0.5 = 0.5 > 0 = |z|
        assert!(in_cone::<f64>(&[0.25, 0.25, 0.0], alpha, 1e-9));
        // (0.25, 0.25, 1.0) — 0.5 < 1.0 — outside
        assert!(!in_cone::<f64>(&[0.25, 0.25, 1.0], alpha, 1e-9));
    }

    #[test]
    fn test_gradient_fd() {
        let alpha = 0.3;
        let s = [0.5_f64, 0.8, 0.1];
        let g = grad(&s, alpha);
        let h = 1e-6;
        for k in 0..3 {
            let mut sp = s;
            let mut sm = s;
            sp[k] += h;
            sm[k] -= h;
            let barrier = |x: &[f64]| {
                let p = phi(x, alpha);
                -p.ln() - (1.0 - alpha) * x[0].ln() - alpha * x[1].ln()
            };
            let fd = (barrier(&sp) - barrier(&sm)) / (2.0 * h);
            assert!((g[k] - fd).abs() < 1e-5, "grad[{k}]={} fd={fd}", g[k]);
        }
    }

    #[test]
    fn test_hessian_fd() {
        let alpha = 0.3;
        let s = [0.5_f64, 0.8, 0.1];
        let hmat = hess(&s, alpha);
        let h = 1e-6;
        for a in 0..3 {
            for b in 0..3 {
                let mut sp = s;
                let mut sm = s;
                sp[b] += h;
                sm[b] -= h;
                let fd = (grad(&sp, alpha)[a] - grad(&sm, alpha)[a]) / (2.0 * h);
                assert!(
                    (hmat[a][b] - fd).abs() < 1e-4,
                    "H[{a}][{b}]={} fd={fd}",
                    hmat[a][b]
                );
            }
        }
    }

    #[test]
    fn test_step_stays_in_cone() {
        let alpha = 0.5;
        let s = [0.5_f64, 0.5, 0.0];
        let ds = [0.1, 0.0, -0.1];
        let a = max_step(&s, &ds, alpha);
        assert!(a > 0.0 && a < 1e10);
        let inside = [
            s[0] + 0.99 * a * ds[0],
            s[1] + 0.99 * a * ds[1],
            s[2] + 0.99 * a * ds[2],
        ];
        assert!(in_cone::<f64>(&inside, alpha, 1e-9));
    }

    /// The dual-boundary step keeps the dual ray inside the true dual cone
    /// `(u/α)^α(v/(1−α))^(1−α) ≥ |w|` and crosses the boundary between 0.99·step
    /// and 1.01·step (bisection reference). Also: on rays where the old
    /// primal-boundary map returned zero — the dual-freeze failure mode — the
    /// true dual step is strictly positive.

    use iconic_core::rng::Lcg;

    #[test]
    fn dual_step_tracks_dual_boundary() {
        let mut rng = Lcg::new(5);
        let mut positive_where_primal_zero = 0usize;
        for _ in 0..20000 {
            let alpha = 0.05 + rng.unit() * 0.9;
            // Random interior dual point: u, v > 0, (u/a)^a (v/(1-a))^(1-a) > |w|.
            let u = 0.1 + rng.unit() * 2.0;
            let v = 0.1 + rng.unit() * 2.0;
            let lhs = (u / alpha).powf(alpha) * (v / (1.0 - alpha)).powf(1.0 - alpha);
            let w = (rng.unit() * 2.0 - 1.0) * 0.9 * lhs;
            let d = [u, v, w];
            let dd = [rng.signed(), rng.signed(), rng.signed()];
            let a = max_step_dual(&d, &dd, alpha);
            // 0.99a strictly inside, 1.01a outside (or both outside if a == 0).
            let inside = |t: f64| {
                let p = [d[0] + t * dd[0], d[1] + t * dd[1], d[2] + t * dd[2]];
                dual_margin(&p, alpha) > 0.0
            };
            if a > 0.0 && a < 1e10 {
                assert!(inside(0.99 * a), "0.99a outside: a={a}");
                assert!(!inside(1.01 * a), "1.01a inside: a={a}");
            } else if a >= 1e10 {
                assert!(inside(0.99 * a), "unbounded ray: 0.99a outside");
            }
            // Dual-freeze rays (where a degenerate map would return step 0)
            // still have a positive true dual step whenever the dual
            // direction has interior content.
            let primal_map = max_step(&d, &dd, alpha);
            if primal_map == 0.0 && a > 1e-9 {
                positive_where_primal_zero += 1;
            }
        }
        assert!(
            positive_where_primal_zero > 1000,
            "dual-freeze rays: {positive_where_primal_zero}"
        );
    }
}
