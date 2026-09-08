//! Generalized power-cone primitives for a nonsymmetric interior-point treatment.
//!
//! The generalized power cone with exponents `α ∈ R^n_+` (`Σαᵢ = 1`) and tail
//! dimension `m` is
//! `K_α = cl { (x,z) ∈ R^n_+ × R^m : ∏ᵢ xᵢ^{αᵢ} ≥ ‖z‖₂,  x > 0 }`.
//! It reduces to the ordinary 3-D power cone (`n=2, m=1`) in [`crate::pow`]
//! when `α = (α₁, 1-α₁)`.
//!
//! The barrier below is the direct `n`-base/`m`-tail generalization of
//! [`crate::pow`]'s `f(x,y,z) = −log(x^{2α}·y^{2(1-α)} − z²) − (1-α)log(x) − α log(y)`,
//! re-derived from the same public convex-analysis structure:
//! `φ(x,z) = ∏ᵢ xᵢ^{2αᵢ} − ‖z‖² = P(x) − ‖z‖²`, and
//! `f(x,z) = −log(φ(x,z)) − Σᵢ(1-αᵢ)·log(xᵢ)`.
//! Substituting `n=2` recovers the 3-D case exactly (`1-α₂ = α₁`).

use iconic_core::Scalar;

/// `P(x) = ∏ᵢ xᵢ^{2αᵢ}` — the product term alone (before subtracting `‖z‖²`).
fn prod_term<T: Scalar>(x: &[T], alpha: &[T]) -> T {
    let two = T::from_f64(2.0).expect("scalar literal");
    let mut p = T::one();
    for i in 0..x.len() {
        p *= x[i].powf(two * alpha[i]);
    }
    p
}

/// Centrality function: `φ = P(x) − ‖z‖²`. Strictly interior iff `φ > 0, x > 0`.
fn phi<T: Scalar>(s: &[T], alpha: &[T]) -> T {
    let n = alpha.len();
    let p = prod_term(&s[..n], alpha);
    let mut zz = T::zero();
    for &zj in &s[n..] {
        zz += zj * zj;
    }
    p - zz
}

/// Primal-cone margin: `min(φ, x₁, …, xₙ)`.
pub fn margin<T: Scalar>(s: &[T], alpha: &[T]) -> T {
    let n = alpha.len();
    let mut min_x = s[0];
    for &xi in &s[1..n] {
        min_x = min_x.min(xi);
    }
    if min_x <= T::zero() {
        return min_x;
    }
    phi(s, alpha).min(min_x)
}

/// Is `s` in the primal generalized power cone (with tolerance `tol`)?
pub fn in_cone<T: Scalar>(s: &[T], alpha: &[T], tol: T) -> bool {
    let n = alpha.len();
    (0..n).all(|i| s[i] > tol) && phi(s, alpha) > tol
}

/// Largest `a ≥ 0` keeping `s + a·ds` in the primal cone, via the shared
/// curved-boundary bisection ([`crate::nonsym::curved_max_step`]).
pub fn max_step<T: Scalar>(s: &[T], ds: &[T], alpha: &[T]) -> T {
    crate::nonsym::curved_max_step(|p| margin(p, alpha), s, ds)
}

/// Signed margin in the **dual** generalized power cone. The dual of
/// `K_α = {(x,z) : ∏ᵢ xᵢ^{αᵢ} ≥ ‖z‖₂}` is the larger cone
/// `K*_α = {(u,w) : ∏ᵢ (uᵢ/αᵢ)^{αᵢ} ≥ ‖w‖₂, u ≥ 0}` (the direct generalization
/// of the 3-D power-cone dual in [`crate::pow`]; verified numerically that
/// `−∇f(s) ∈ int K*` for every interior `s`). Positive iff strictly inside.
fn dual_margin<T: Scalar>(d: &[T], alpha: &[T]) -> T {
    let zero = T::zero();
    let n = alpha.len();
    let m = d.len() - n;
    let mut min_u = d[0];
    for &u in &d[1..n] {
        min_u = min_u.min(u);
    }
    if min_u <= zero {
        return min_u;
    }
    let mut lhs = T::one();
    for i in 0..n {
        lhs *= (d[i] / alpha[i]).powf(alpha[i]);
    }
    let mut nw = T::zero();
    for &w in &d[n..n + m] {
        nw += w * w;
    }
    lhs - nw.sqrt()
}

/// Largest `a ≥ 0` keeping the ray `d + a·dd` in the **dual** generalized power
/// cone, via bisection on [`dual_margin`] (see [`crate::pow::max_step_dual`] for
/// why the dual cone's own boundary — not the primal's — is the correct step map).
pub fn max_step_dual<T: Scalar>(d: &[T], dd: &[T], alpha: &[T]) -> T {
    crate::nonsym::curved_max_step(|p| dual_margin(p, alpha), d, dd)
}

/// Barrier value `f(s) = −log(φ) − Σᵢ(1−αᵢ)·log(xᵢ)` (requires `s` strictly interior).
pub fn barrier<T: Scalar>(s: &[T], alpha: &[T]) -> T {
    let n = alpha.len();
    let one = T::one();
    let mut v = -phi(s, alpha).ln();
    for i in 0..n {
        v -= (one - alpha[i]) * s[i].ln();
    }
    v
}

/// Barrier gradient `∇f(s)` (length `n + m`).
pub fn grad<T: Scalar>(s: &[T], alpha: &[T]) -> Vec<T> {
    let n = alpha.len();
    let d = s.len();
    let two = T::from_f64(2.0).expect("scalar literal");
    let one = T::one();
    let p = prod_term(&s[..n], alpha);
    let phi_val = phi(s, alpha);
    let inv_phi = one / phi_val;
    let mut g = vec![T::zero(); d];
    for i in 0..n {
        // ∂P/∂xᵢ = 2αᵢ·P/xᵢ (log-derivative: log P = Σⱼ 2αⱼ log xⱼ)
        let dphi_dxi = two * alpha[i] * p / s[i];
        g[i] = -inv_phi * dphi_dxi - (one - alpha[i]) / s[i];
    }
    for j in n..d {
        // ∂φ/∂zⱼ = −2zⱼ
        let dphi_dzj = -two * s[j];
        g[j] = -inv_phi * dphi_dzj;
    }
    g
}

/// Barrier Hessian `∇²f(s)` (dense `d×d`, row-major, `d = n+m`).
///
/// Off-diagonal `x`-`x` cross terms are the genuinely new structure versus the
/// 3-D case: `∂²P/∂xᵢ∂xₖ = 4αᵢαₖ·P/(xᵢxₖ)` for `i≠k`.
pub fn hess<T: Scalar>(s: &[T], alpha: &[T]) -> Vec<T> {
    let n = alpha.len();
    let d = s.len();
    let two = T::from_f64(2.0).expect("scalar literal");
    let one = T::one();
    let p = prod_term(&s[..n], alpha);
    let phi_val = phi(s, alpha);
    let inv_phi = one / phi_val;
    let inv_phi2 = inv_phi * inv_phi;

    // First derivatives of φ
    let mut dphi = vec![T::zero(); d];
    for i in 0..n {
        dphi[i] = two * alpha[i] * p / s[i];
    }
    for j in n..d {
        dphi[j] = -two * s[j];
    }

    let mut h = vec![T::zero(); d * d];
    for a in 0..d {
        for b in 0..d {
            // Second derivative of φ (not of the barrier f) at (a,b)
            let phi_ab = if a < n && b < n {
                if a == b {
                    // ∂²P/∂xᵢ² = 2αᵢ·(2αᵢ-1)·P/xᵢ²
                    two * alpha[a] * (two * alpha[a] - one) * p / (s[a] * s[a])
                } else {
                    // ∂²P/∂xᵢ∂xₖ = 4αᵢαₖ·P/(xᵢxₖ), i≠k
                    two * two * alpha[a] * alpha[b] * p / (s[a] * s[b])
                }
            } else if a >= n && b >= n {
                if a == b {
                    -two
                } else {
                    T::zero()
                }
            } else {
                T::zero() // no x-z cross term in phi
            };
            let mut hab = -inv_phi * phi_ab + inv_phi2 * dphi[a] * dphi[b];
            if a == b && a < n {
                // −log(xᵢ) barrier term's own contribution: (1-αᵢ)/xᵢ²
                hab += (one - alpha[a]) / (s[a] * s[a]);
            }
            h[a * d + b] = hab;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 3-D reduction (n=2, m=1) must match [`crate::pow`] exactly — both
    /// the barrier value and its derivatives — confirming the generalization
    /// is consistent with the already-shipped, already-tested special case.
    #[test]
    fn reduces_to_pow_3d() {
        let alpha1 = 0.3_f64;
        let alpha = [alpha1, 1.0 - alpha1];
        let s = [0.5_f64, 0.8, 0.1];

        let phi_gen = phi(&s, &alpha);
        let phi_pow = {
            let a2 = 2.0 * alpha1;
            let b2 = 2.0 * (1.0 - alpha1);
            s[0].powf(a2) * s[1].powf(b2) - s[2] * s[2]
        };
        assert!((phi_gen - phi_pow).abs() < 1e-12);

        let g_gen = grad(&s, &alpha);
        let g_pow = crate::pow::grad(&s, alpha1);
        for k in 0..3 {
            assert!(
                (g_gen[k] - g_pow[k]).abs() < 1e-10,
                "grad[{k}] gen={} pow={}",
                g_gen[k],
                g_pow[k]
            );
        }

        let h_gen = hess(&s, &alpha);
        let h_pow = crate::pow::hess(&s, alpha1);
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (h_gen[a * 3 + b] - h_pow[a][b]).abs() < 1e-10,
                    "H[{a}][{b}] gen={} pow={}",
                    h_gen[a * 3 + b],
                    h_pow[a][b]
                );
            }
        }
    }

    #[test]
    fn test_membership() {
        let alpha = [0.2_f64, 0.3, 0.5];
        assert!(in_cone::<f64>(&[1.0, 1.0, 1.0, 0.0], &alpha, 1e-9));
        assert!(!in_cone::<f64>(&[1.0, 1.0, 1.0, 1.5], &alpha, 1e-9));
        assert!(!in_cone::<f64>(&[-1.0, 1.0, 1.0, 0.0], &alpha, 1e-9));
    }

    #[test]
    fn test_gradient_fd_n3_m2() {
        let alpha = [0.2_f64, 0.3, 0.5];
        let s = [0.6_f64, 0.9, 0.4, 0.15, -0.1];
        let g = grad(&s, &alpha);
        let h = 1e-6;
        let n = alpha.len();
        let barrier = |x: &[f64]| {
            let p = phi(x, &alpha);
            let mut acc = -p.ln();
            for i in 0..n {
                acc -= (1.0 - alpha[i]) * x[i].ln();
            }
            acc
        };
        for k in 0..s.len() {
            let mut sp = s;
            let mut sm = s;
            sp[k] += h;
            sm[k] -= h;
            let fd = (barrier(&sp) - barrier(&sm)) / (2.0 * h);
            assert!((g[k] - fd).abs() < 1e-5, "grad[{k}]={} fd={fd}", g[k]);
        }
    }

    #[test]
    fn test_hessian_fd_n3_m2() {
        let alpha = [0.2_f64, 0.3, 0.5];
        let s = [0.6_f64, 0.9, 0.4, 0.15, -0.1];
        let d = s.len();
        let hmat = hess(&s, &alpha);
        let h = 1e-6;
        for a in 0..d {
            for b in 0..d {
                let mut sp = s;
                let mut sm = s;
                sp[b] += h;
                sm[b] -= h;
                let fd = (grad(&sp, &alpha)[a] - grad(&sm, &alpha)[a]) / (2.0 * h);
                assert!(
                    (hmat[a * d + b] - fd).abs() < 1e-4,
                    "H[{a}][{b}]={} fd={fd}",
                    hmat[a * d + b]
                );
            }
        }
    }

    #[test]
    fn test_step_stays_in_cone() {
        let alpha = [0.4_f64, 0.6];
        let s = [0.5_f64, 0.5, 0.0];
        let ds = [0.1, 0.0, -0.1];
        let a = max_step(&s, &ds, &alpha);
        assert!(a > 0.0 && a < 1e10);
        let inside: Vec<f64> = (0..3).map(|i| s[i] + 0.99 * a * ds[i]).collect();
        assert!(in_cone::<f64>(&inside, &alpha, 1e-9));
    }

    /// Larger instance (n=4 base, m=3 tail, d=7): gradient/Hessian FD checks
    /// at a second, more asymmetric point.
    #[test]
    fn test_gradient_hessian_fd_larger_instance() {
        let alpha = [0.1_f64, 0.15, 0.25, 0.5];
        let s = [1.3_f64, 0.7, 2.1, 0.55, 0.3, -0.2, 0.1];
        let d = s.len();
        let g = grad(&s, &alpha);
        let h = 1e-6;
        let n = alpha.len();
        let barrier = |x: &[f64]| {
            let p = phi(x, &alpha);
            let mut acc = -p.ln();
            for i in 0..n {
                acc -= (1.0 - alpha[i]) * x[i].ln();
            }
            acc
        };
        for k in 0..d {
            let mut sp = s;
            let mut sm = s;
            sp[k] += h;
            sm[k] -= h;
            let fd = (barrier(&sp) - barrier(&sm)) / (2.0 * h);
            assert!((g[k] - fd).abs() < 1e-5, "grad[{k}]={} fd={fd}", g[k]);
        }
        let hmat = hess(&s, &alpha);
        for a in 0..d {
            for b in 0..d {
                let mut sp = s;
                let mut sm = s;
                sp[b] += h;
                sm[b] -= h;
                let fd = (grad(&sp, &alpha)[a] - grad(&sm, &alpha)[a]) / (2.0 * h);
                assert!(
                    (hmat[a * d + b] - fd).abs() < 1e-4,
                    "H[{a}][{b}]={} fd={fd}",
                    hmat[a * d + b]
                );
            }
        }
    }
}
