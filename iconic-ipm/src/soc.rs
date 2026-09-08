//! Second-order (Lorentz) cone primitives for the interior-point engine.
//!
//! `Q_m = { u = (u₀, u₁) ∈ ℝ × ℝ^{m−1} : u₀ ≥ ‖u₁‖₂ }`.
//!
//! These are the building blocks the cone-aware IPM needs: the Jordan-algebra
//! product and identity, the cone margin (membership test), and the exact
//! step-to-boundary along a direction. Nesterov–Todd scaling and the KKT
//! integration build on top of these.

use iconic_core::Scalar;

/// Euclidean norm of the tail `u₁` (all but the first component).
fn tail_norm<T: Scalar>(u: &[T]) -> T {
    u[1..].iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt()
}

/// Cone margin `u₀ − ‖u₁‖`. Positive in the interior, zero on the boundary.
pub fn margin<T: Scalar>(u: &[T]) -> T {
    u[0] - tail_norm(u)
}

/// Whether `u` lies in the (closed) cone within a tolerance.
pub fn in_cone<T: Scalar>(u: &[T], tol: T) -> bool {
    margin(u) >= -tol
}

/// Jordan-algebra identity `e = (1, 0, …, 0)`.
pub fn identity<T: Scalar>(m: usize) -> Vec<T> {
    let mut e = vec![T::zero(); m];
    e[0] = T::one();
    e
}

/// Jordan product `u ∘ v = (uᵀv, u₀·v₁ + v₀·u₁)`.
pub fn jordan<T: Scalar>(u: &[T], v: &[T]) -> Vec<T> {
    let m = u.len();
    let mut w = vec![T::zero(); m];
    let mut dot = T::zero();
    for i in 0..m {
        dot += u[i] * v[i];
    }
    w[0] = dot;
    for i in 1..m {
        w[i] = u[0] * v[i] + v[0] * u[i];
    }
    w
}

/// Largest `α ≥ 0` with `u + α·du ∈ Q`, or `+∞` if the ray never leaves the cone.
///
/// The boundary condition `(u₀+αdu₀) ≥ ‖u₁+αdu₁‖` squares to the quadratic
/// `q(α) = aα² + 2bα + c ≥ 0` (with `a = du₀²−‖du₁‖²`, `b = u₀du₀−u₁·du₁`,
/// `c = u₀²−‖u₁‖² ≥ 0`), combined with `u₀+αdu₀ ≥ 0`. Since the margin is concave
/// and positive at `α = 0`, the limiting step is the first positive crossing.
/// [`jordan`] into a caller buffer (same accumulation order).
pub fn jordan_into<T: Scalar>(u: &[T], v: &[T], out: &mut [T]) {
    let m = u.len();
    let mut dot = T::zero();
    for i in 0..m {
        dot += u[i] * v[i];
    }
    out[0] = dot;
    for i in 1..m {
        out[i] = u[0] * v[i] + v[0] * u[i];
    }
}

/// [`arrow_inverse_apply`] into a caller buffer.
pub fn arrow_inverse_apply_into<T: Scalar>(u: &[T], v: &[T], out: &mut [T]) {
    let m = u.len();
    let u1v1 = u[1..]
        .iter()
        .zip(&v[1..])
        .fold(T::zero(), |acc, (&a, &b)| acc + a * b);
    let u1u1 = u[1..].iter().fold(T::zero(), |acc, &a| acc + a * a);
    let d = u[0] * u[0] - u1u1;
    // Reciprocals: `d`, `u[0]` and `u[0]·d` are loop-invariant (3 divisions per
    // call instead of per element; <1 ulp difference).
    let inv_d = d.recip();
    let inv_u0 = u[0].recip();
    let inv_u0d = inv_u0 * inv_d;
    out[0] = (u[0] * v[0] - u1v1) * inv_d;
    for i in 1..m {
        out[i] = -u[i] * v[0] * inv_d + v[i] * inv_u0 + u[i] * u1v1 * inv_u0d;
    }
}

/// [`band_project`] into a caller buffer.
pub fn band_project_into<T: Scalar>(v: &[T], lo: T, hi: T, out: &mut [T]) {
    let n = v.len();
    let clamp = |x: T| {
        if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    };
    let nv1 = tail_norm(v);
    if nv1 <= T::epsilon() {
        out[0] = clamp(v[0]);
        for i in 1..n {
            out[i] = T::zero();
        }
        return;
    }
    let half = T::from_f64(0.5).expect("scalar literal");
    let lam_p = clamp(v[0] + nv1);
    let lam_m = clamp(v[0] - nv1);
    // w = λ₊·(e + v̂) + λ₋·(e − v̂), halved.
    out[0] = half * (lam_p + lam_m);
    let scale = half * (lam_p - lam_m) / nv1;
    for i in 1..n {
        out[i] = scale * v[i];
    }
}

pub fn max_step<T: Scalar>(u: &[T], du: &[T]) -> T {
    let zero = T::zero();
    let two = T::from_f64(2.0).expect("scalar literal");

    let mut alpha = T::infinity();

    // Constraint u₀ + α·du₀ ≥ 0.
    if du[0] < zero {
        alpha = alpha.min(-u[0] / du[0]);
    }

    let dot_tail = u[1..]
        .iter()
        .zip(&du[1..])
        .fold(zero, |acc, (&a, &b)| acc + a * b);
    let ndu1_sq = du[1..].iter().fold(zero, |acc, &v| acc + v * v);
    let nu1_sq = u[1..].iter().fold(zero, |acc, &v| acc + v * v);

    let a = du[0] * du[0] - ndu1_sq;
    let b = u[0] * du[0] - dot_tail;
    let c = u[0] * u[0] - nu1_sq;

    if a == zero {
        // q(α) = 2bα + c; only bounded when b < 0.
        if b < zero {
            alpha = alpha.min(-c / (two * b));
        }
    } else {
        let disc = b * b - a * c;
        if disc >= zero {
            let sq = disc.sqrt();
            if a > zero {
                // Opens up; q(0) = c ≥ 0 → limiting root is (−b − √disc)/a when positive.
                let root = (-b - sq) / a;
                if root > zero {
                    alpha = alpha.min(root);
                }
            } else {
                // Opens down; roots straddle 0 (product c/a < 0) → take the positive one.
                let r1 = (-b + sq) / a;
                let r2 = (-b - sq) / a;
                let root = r1.max(r2);
                if root > zero {
                    alpha = alpha.min(root);
                }
            }
        }
    }

    // Fixed boundary margin: a step computed exactly to the cone boundary can
    // round past it, leaving a non-interior iterate (the documented SOC NaN
    // history); backing off a fixed 1e-13 keeps the point strictly interior
    // with negligible step cost (the caller's fraction-to-boundary eta does the
    // bulk backoff).
    let margin = T::from_f64(1e-13).expect("scalar literal");
    (alpha - margin).max(zero)
}

/// J-norm `√(u₀² − ‖u₁‖²)` (the "determinant" square-root), defined for `u ∈ Q`.
fn jnorm<T: Scalar>(u: &[T]) -> T {
    let tail_sq = u[1..].iter().fold(T::zero(), |acc, &v| acc + v * v);
    // Clamp the determinant to a tiny positive: a near-boundary iterate can round it
    // slightly negative, which would otherwise produce a NaN scaling.
    let det = u[0] * u[0] - tail_sq;
    det.max(T::from_f64(1e-300).expect("scalar literal")).sqrt()
}

/// Nesterov–Todd scaling for the second-order cone.
///
/// Returns `(η², w̄)` defining the symmetric scaling block `H = η²(2 w̄w̄ᵀ − J)`
/// (with `J = diag(1, −1, …, −1)`), which satisfies `H z = s` — the cone analogue
/// of the nonnegative block `diag(s/z)` (which also maps `z → s`). Requires
/// `s, z ∈ int(Q)`.
pub fn nt_scaling<T: Scalar>(s: &[T], z: &[T]) -> (T, Vec<T>) {
    let m = s.len();
    let two = T::from_f64(2.0).expect("scalar literal");

    let js = jnorm(s);
    let jz = jnorm(z);
    let inv_js = js.recip();
    let inv_jz = jz.recip();
    let sbar: Vec<T> = s.iter().map(|&v| v * inv_js).collect();
    let zbar: Vec<T> = z.iter().map(|&v| v * inv_jz).collect();

    // γ = √((1 + s̄ᵀz̄)/2).
    let mut sdotz = T::zero();
    for i in 0..m {
        sdotz += sbar[i] * zbar[i];
    }
    let gamma = ((T::one() + sdotz) / two).sqrt();

    // w̄ = (s̄ + J z̄)/(2γ).
    let inv_2g = (two * gamma).recip();
    let mut wbar = vec![T::zero(); m];
    wbar[0] = (sbar[0] + zbar[0]) * inv_2g;
    for i in 1..m {
        wbar[i] = (sbar[i] - zbar[i]) * inv_2g;
    }

    // η² = jnorm(s)/jnorm(z) so that H z = s.
    (js / jz, wbar)
}

/// Apply the SOC scaling block `H = η²(2 w̄w̄ᵀ − J)` to a vector `v`.
pub fn apply_nt<T: Scalar>(eta_sq: T, wbar: &[T], v: &[T]) -> Vec<T> {
    let m = v.len();
    let two = T::from_f64(2.0).expect("scalar literal");
    let mut wdotv = T::zero();
    for i in 0..m {
        wdotv += wbar[i] * v[i];
    }
    let mut out = vec![T::zero(); m];
    // −J v = (−v₀, v₁), so 2w̄(w̄ᵀv) − J v has component 0 = 2w̄₀(w̄·v) − v₀.
    out[0] = eta_sq * (two * wbar[0] * wdotv - v[0]);
    for i in 1..m {
        out[i] = eta_sq * (two * wbar[i] * wdotv + v[i]);
    }
    out
}

/// Apply the SOC scaling matrix `W = η·W̄` to `v`, where `η = √(η²)` and
/// `W̄ = [[w̄₀, w̄₁ᵀ], [w̄₁, I + w̄₁w̄₁ᵀ/(1+w̄₀)]]` is the (det-1) scaling matrix.
/// `W² = H` (the NT block) and `W z = W⁻¹ s = λ`, the scaled point.
pub fn apply_w<T: Scalar>(eta: T, wbar: &[T], v: &[T]) -> Vec<T> {
    let m = v.len();
    let one = T::one();
    let w1v1 = wbar[1..]
        .iter()
        .zip(&v[1..])
        .fold(T::zero(), |acc, (&w, &x)| acc + w * x);
    let coef = w1v1 / (one + wbar[0]);
    let mut out = vec![T::zero(); m];
    out[0] = eta * (wbar[0] * v[0] + w1v1);
    for i in 1..m {
        out[i] = eta * (wbar[i] * v[0] + v[i] + wbar[i] * coef);
    }
    out
}

/// Apply the inverse scaling `W⁻¹ = (1/η)·W̄⁻¹`, with `W̄⁻¹ = J W̄ J` (the couplings
/// to the first component flip sign).
pub fn apply_w_inv<T: Scalar>(eta: T, wbar: &[T], v: &[T]) -> Vec<T> {
    let m = v.len();
    let one = T::one();
    let w1v1 = wbar[1..]
        .iter()
        .zip(&v[1..])
        .fold(T::zero(), |acc, (&w, &x)| acc + w * x);
    let coef = w1v1 / (one + wbar[0]);
    let mut out = vec![T::zero(); m];
    out[0] = (wbar[0] * v[0] - w1v1) / eta;
    for i in 1..m {
        out[i] = (-wbar[i] * v[0] + v[i] + wbar[i] * coef) / eta;
    }
    out
}

/// Jordan inverse-operator apply `Arw(u)⁻¹ v`, the unique `w` with `u ∘ w = v`
/// (for `u ∈ int(Q)`). `Arw(u) = [[u₀, u₁ᵀ], [u₁, u₀I]]`, and its inverse uses the
/// Jordan determinant `d = u₀² − ‖u₁‖²`.
pub fn arrow_inverse_apply<T: Scalar>(u: &[T], v: &[T]) -> Vec<T> {
    let m = u.len();
    let u1v1 = u[1..]
        .iter()
        .zip(&v[1..])
        .fold(T::zero(), |acc, (&a, &b)| acc + a * b);
    let u1u1 = u[1..].iter().fold(T::zero(), |acc, &a| acc + a * a);
    let d = u[0] * u[0] - u1u1;

    let mut out = vec![T::zero(); m];
    out[0] = (u[0] * v[0] - u1v1) / d;
    for i in 1..m {
        out[i] = -u[i] * v[0] / d + v[i] / u[0] + u[i] * u1v1 / (u[0] * d);
    }
    out
}

/// Project the Jordan spectrum of `v` into `[lo, hi]`, used by the Gondzio centrality
/// corrector. A second-order-cone element has eigenvalues `v0 ± ‖v1‖` with eigenvectors
/// `½(1, ±v1/‖v1‖)`; clamp each eigenvalue to the band and reconstruct.
pub fn band_project<T: Scalar>(v: &[T], lo: T, hi: T) -> Vec<T> {
    let n = v.len();
    let clamp = |x: T| {
        if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    };
    let nv1 = tail_norm(v);
    let mut out = vec![T::zero(); n];
    if nv1 <= T::epsilon() {
        out[0] = clamp(v[0]); // v = v0·e, a repeated eigenvalue
        return out;
    }
    let half = T::from_f64(0.5).expect("scalar literal");
    let lam_p = clamp(v[0] + nv1);
    let lam_m = clamp(v[0] - nv1);
    out[0] = half * (lam_p + lam_m);
    let scale = half * (lam_p - lam_m) / nv1;
    for i in 1..n {
        out[i] = scale * v[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_and_membership() {
        assert!((margin::<f64>(&[2.0, 1.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!(in_cone::<f64>(&[1.0, 0.6, 0.8], 1e-9)); // ‖(0.6,0.8)‖ = 1.0
        assert!(!in_cone::<f64>(&[0.5, 0.6, 0.8], 1e-9));
    }

    #[test]
    fn jordan_product_and_identity() {
        let e = identity::<f64>(3);
        let u = vec![2.0, 1.0, -1.0];
        // u ∘ e = u.
        assert_eq!(jordan(&u, &e), u);
        // u ∘ u = (uᵀu, 2u₀u₁).
        let uu = jordan(&u, &u);
        assert!((uu[0] - 6.0).abs() < 1e-12); // 4+1+1
        assert!((uu[1] - 4.0).abs() < 1e-12); // 2*2*1
        assert!((uu[2] + 4.0).abs() < 1e-12); // 2*2*(-1)
    }

    #[test]
    fn step_to_boundary() {
        // From (2,1,0) along (0,1,0): exits at α=1 where 2 = |1+α|.
        assert!((max_step::<f64>(&[2.0, 1.0, 0.0], &[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-9);
        // From (2,0,0) along (0,3,0): exits at α = 2/3.
        assert!((max_step::<f64>(&[2.0, 0.0, 0.0], &[0.0, 3.0, 0.0]) - 2.0 / 3.0).abs() < 1e-9);
        // From (1,0,0) along (-1,0,0): hits the apex at α=1.
        assert!((max_step::<f64>(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn step_unbounded_when_ray_stays_inside() {
        // Moving deeper into the cone never exits.
        assert!(max_step::<f64>(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).is_infinite());
        assert!(max_step::<f64>(&[2.0, 0.5, 0.0], &[5.0, 0.0, 0.0]).is_infinite());
    }

    #[test]
    fn nt_scaling_maps_z_to_s() {
        // For s, z ∈ int(Q), the NT scaling block H satisfies H s = z and H z⁻¹...
        // here we check the defining identity H s = z on several interior points.
        let cases: [(Vec<f64>, Vec<f64>); 3] = [
            (vec![2.0, 1.0, 0.0], vec![3.0, 0.0, 1.0]),
            (vec![5.0, 1.0, -2.0], vec![2.0, 0.5, 0.5]),
            (vec![1.5, 0.3, 0.4], vec![4.0, -1.0, 2.0]),
        ];
        for (s, z) in cases {
            assert!(margin(&s) > 0.0 && margin(&z) > 0.0);
            let (eta_sq, wbar) = nt_scaling(&s, &z);
            // Defining property: the scaling block maps z → s.
            let hz = apply_nt(eta_sq, &wbar, &z);
            for i in 0..s.len() {
                assert!(
                    (hz[i] - s[i]).abs() < 1e-9,
                    "H z ≠ s at {i}: {} vs {}",
                    hz[i],
                    s[i]
                );
            }
            // The block is symmetric PD: w̄ ∈ int(Q) and η² > 0.
            assert!(margin(&wbar) > 0.0 && eta_sq > 0.0);
        }
    }

    #[test]
    fn arrow_inverse_undoes_jordan_product() {
        let us: [Vec<f64>; 3] = [
            vec![3.0, 1.0, 0.5],
            vec![5.0, -1.0, 2.0],
            vec![2.0, 0.3, 0.1],
        ];
        let v = vec![1.0_f64, 2.0, -0.5];
        for u in us {
            // u ∘ (Arw(u)⁻¹ v) = v.
            let w = arrow_inverse_apply(&u, &v);
            let back = jordan(&u, &w);
            for i in 0..3 {
                assert!((back[i] - v[i]).abs() < 1e-9, "u∘(Arw⁻¹v) ≠ v at {i}");
            }
        }
    }

    #[test]
    fn scaling_matrix_squares_to_block_and_maps_points() {
        let cases: [(Vec<f64>, Vec<f64>); 3] = [
            (vec![2.0, 1.0, 0.0], vec![3.0, 0.0, 1.0]),
            (vec![5.0, 1.0, -2.0], vec![2.0, 0.5, 0.5]),
            (vec![1.5, 0.3, 0.4], vec![4.0, -1.0, 2.0]),
        ];
        for (s, z) in cases {
            let (eta_sq, wbar) = nt_scaling(&s, &z);
            let eta = eta_sq.sqrt();
            // W² v = H v on a probe vector.
            let v = vec![1.0_f64, -2.0, 0.5];
            let wwv = apply_w(eta, &wbar, &apply_w(eta, &wbar, &v));
            let hv = apply_nt(eta_sq, &wbar, &v);
            for i in 0..3 {
                assert!((wwv[i] - hv[i]).abs() < 1e-9, "W²≠H at {i}");
            }
            // W z = W⁻¹ s = λ (the scaled point).
            let wz = apply_w(eta, &wbar, &z);
            let wis = apply_w_inv(eta, &wbar, &s);
            for i in 0..3 {
                assert!((wz[i] - wis[i]).abs() < 1e-9, "W z ≠ W⁻¹ s at {i}");
            }
            // W and W⁻¹ are mutually inverse.
            let roundtrip = apply_w_inv(eta, &wbar, &apply_w(eta, &wbar, &v));
            for i in 0..3 {
                assert!((roundtrip[i] - v[i]).abs() < 1e-9, "W⁻¹W ≠ I at {i}");
            }
        }
    }
}
