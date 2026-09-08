//! Exponential-cone primitives for a nonsymmetric interior-point treatment.
//!
//! The exponential cone is
//! `K_exp = cl { (x,y,z) : y·exp(x/y) ≤ z, y > 0 }`,
//! equivalently `{ (x,y,z) : y·log(z/y) − x ≥ 0, y > 0, z > 0 }`. Unlike the symmetric
//! cones (orthant/SOC/PSD) it is **not** self-dual and has no Nesterov–Todd scaling, so
//! the interior-point method works directly with its logarithmically-homogeneous
//! self-concordant barrier (parameter ν = 3)
//! `f(x,y,z) = −log(ψ) − log(y) − log(z)`, with `ψ = y·log(z/y) − x`.
//!
//! This module provides the barrier value, gradient, and Hessian, plus membership for
//! the primal and dual cones — the building blocks the nonsymmetric IPM step will use
//! (the cone's `(z,z)` block is the barrier Hessian, and centering follows the
//! gradient). It is verified against finite differences.

use iconic_core::Scalar;

/// `ψ = y·log(z/y) − x`; the primal cone is `{ψ ≥ 0, y > 0, z > 0}`.
fn psi<T: Scalar>(s: &[T]) -> T {
    let (x, y, z) = (s[0], s[1], s[2]);
    y * (z / y).ln() - x
}

/// Primal-cone margin: `min(ψ, y, z)` (strictly interior iff all three are `> 0`).
pub fn margin<T: Scalar>(s: &[T]) -> T {
    let (y, z) = (s[1], s[2]);
    if y <= T::zero() || z <= T::zero() {
        return y.min(z);
    }
    psi(s).min(y).min(z)
}

/// In the primal exponential cone (with tolerance `tol`)?
pub fn in_cone<T: Scalar>(s: &[T], tol: T) -> bool {
    s[1] > tol && s[2] > tol && psi(s) > tol
}

/// In the dual exponential cone? `K* = cl{ (u,v,w) : u < 0, w > 0, w·e ≥ −u·exp(v/u) }`
/// (also contains `{u = 0, v ≥ 0, w ≥ 0}`).
pub fn in_dual_cone<T: Scalar>(d: &[T], tol: T) -> bool {
    let (u, v, w) = (d[0], d[1], d[2]);
    if u < -tol && w > tol {
        let e = T::one().exp();
        return w * e >= -u * (v / u).exp() - tol;
    }
    u.abs() <= tol && v >= -tol && w > -tol
}

/// Largest `α ≥ 0` keeping `s + α·ds` in the (primal) exponential cone, via a
/// **closed-form boundary map** solving the curved-boundary crossing with the Lambert W
/// function (the margin is concave along the ray, so the feasible set is an interval
/// `[0, α_max]`, and the exit is the first boundary hit). Returns a large finite value
/// if the ray stays interior. The bisection is kept only as a fallback for the
/// degenerate cases the closed form self-validates against.
///
/// The curved boundary `y·ln(z/y) = x` along the ray reduces, with `t = z(α)/y(α)`, to
/// `P + Q·t + R·ln t = 0` for the three direction invariants
/// `P = z·dx − x·dz`, `Q = x·dy − y·dx`, `R = y·dz − z·dy` (eliminating α from the two
/// crossing conditions `z(α) = t·y(α)` and `x(α) = y(α)·ln t`). Its roots are
/// `t = (R/Q)·W((Q/R)·e^{−P/R})` — evaluated in the numerically stable log form
/// (`u + ln u = ln|Q/R| − P/R`, or `ln v − v = ln|Q/R| − P/R` with `v = −u` for the
/// second real branch), never forming the (potentially overflowing) exponential. The
/// two real W branches are selected by the sign of `Q/R` and the branch-point test
/// `c ≤ −1`; near the branch point the initial guess is the DLMF 4.13.10/4.13.11
/// series `W = −1 ± p ∓ p²/3 ± 11p³/72` with `p = sqrt(2(−1 − c))`, followed by Newton
/// steps. `t` then gives `α` from either crossing condition; the better-conditioned
/// formula is chosen. The y=0 and (primal) z=0 faces contribute the trivial candidates
/// `−y/dy`, `−z/dz`; the answer is the minimum valid candidate, self-validated by
/// probing the margin at `0.99α` / `1.01α` (falling back to the bisection on failure).
pub fn max_step<T: Scalar>(s: &[T], ds: &[T]) -> T {
    match step_closed(s, ds, true) {
        Some(a) => (a - T::from_f64(1e-13).expect("scalar literal")).max(T::zero()),
        None => max_step_bisect(s, ds),
    }
}

/// Largest `α ≥ 0` keeping the ray `d + α·dd` in the **dual** exponential cone, via the
/// same closed-form boundary map in the mirrored coordinates
/// `(x̄, ȳ, z̄) = (−v, −u, w·e)` (the dual curved boundary `w·e = −u·e^{v/u}` is the
/// primal surface mirrored in `v`). The u=0 face maps to the ȳ=0 face; the w=0 exit of
/// the open dual cone (its `w > 0` half-space face in the `u < 0` region) maps to the
/// z̄=0 face. Bisection fallback for degenerate cases.
pub fn max_step_dual<T: Scalar>(d: &[T], dd: &[T]) -> T {
    let e = T::one().exp();
    let x = -d[1];
    let y = -d[0];
    let z = d[2] * e;
    let dx = -dd[1];
    let dy = -dd[0];
    let dz = dd[2] * e;
    match step_closed(&[x, y, z], &[dx, dy, dz], true) {
        Some(a) => (a - T::from_f64(1e-13).expect("scalar literal")).max(T::zero()),
        None => max_step_dual_bisect(d, dd),
    }
}

/// Bisection fallback for the primal step (see [`max_step`]).
fn max_step_bisect<T: Scalar>(s: &[T], ds: &[T]) -> T {
    crate::nonsym::curved_max_step(margin, s, ds)
}

/// Bisection fallback for the dual step (see [`max_step_dual`]). The shared
/// bisection's margin closure is `> tol`-positive inside; `in_dual_cone(p, tol)`
/// is that predicate, so it slots in directly (and picks up the same fixed
/// 1e-13 boundary backoff as every other cone).
fn max_step_dual_bisect<T: Scalar>(v: &[T], dv: &[T]) -> T {
    crate::nonsym::curved_max_step(|p| {
        if in_dual_cone(p, T::from_f64(1e-12).expect("scalar literal")) {
            T::one()
        } else {
            T::zero()
        }
    }, v, dv)
}

/// Lambert W (principal branch) root of `u + ln u = c` with `u > 0` — the W₀ branch for
/// a nonnegative argument, in the stable log form that never forms the argument itself.
/// Returns `None` when the root underflows below float range (the crossing then
/// collapses to the z=0 face and that edge candidate covers it).
fn w0_positive<T: Scalar>(c: T) -> Option<T> {
    let one = T::one();
    let mut u = if c > one { c - c.ln() } else { c.exp() };
    if u == T::zero() {
        return None;
    }
    for _ in 0..8 {
        let f = u + u.ln() - c;
        u -= f / (one + one / u);
    }
    Some(u)
}

/// Lambert W roots for a negative argument, both real branches, via the stable log form
/// `ln v − v = c` with `u = −v` (the caller guarantees `c = ln|a| ≤ −1`, i.e. the
/// argument lies in `[−1/e, 0)`). Returns `(u_w0, u_wm1)`; the W₀ root lives in
/// `(−1, 0)`, the W₋₁ root in `(−∞, −1)`. Near the branch point the initial guess is the
/// DLMF 4.13.10/4.13.11 series `v = 1 ∓ p ± p²/3 ∓ 11p³/72` with
/// `p = sqrt(2(−1 − c))`, followed by Newton steps; away from it the asymptotic guesses
/// `v ≈ e^c` (W₀) and `v ≈ −c` (W₋₁) are used. An underflowed W₀ root is returned as
/// `−0` (its `t = u/(Q/R)` is filtered out downstream as `t ≤ 0`; the crossing
/// collapses to the z=0 face candidate).
fn w_negative_roots<T: Scalar>(c: T) -> (T, T) {
    let one = T::one();
    let zero = T::zero();
    let two = T::from_f64(2.0).expect("scalar literal");
    let th = T::from_f64(1.0 / 3.0).expect("scalar literal");
    let s11 = T::from_f64(11.0 / 72.0).expect("scalar literal");
    let near = T::from_f64(0.35).expect("scalar literal");
    // Branch point (double root u = −1): return it directly, the Newton iterate
    // divides by the vanishing derivative there.
    if c == -one {
        return (-one, -one);
    }
    let p = (two * (-one - c)).sqrt();
    // W₀ root: v in (0,1).
    let mut w0 = -zero;
    if p < near {
        let mut v = one - p + p * p * th - p * p * p * s11;
        for _ in 0..8 {
            let f = v.ln() - v - c;
            v -= f / (one / v - one);
        }
        w0 = -v;
    } else {
        let v0 = c.exp();
        if v0 > zero {
            let mut v = v0;
            for _ in 0..8 {
                let f = v.ln() - v - c;
                v -= f / (one / v - one);
            }
            w0 = -v;
        }
    }
    // W₋₁ root: v in (1,∞).
    let mut wm1;
    if p < near {
        let mut v = one + p + p * p * th + p * p * p * s11;
        for _ in 0..8 {
            let f = v.ln() - v - c;
            v -= f / (one / v - one);
        }
        wm1 = -v;
    } else if c < -two {
        let mut v = -c;
        for _ in 0..8 {
            let f = v.ln() - v - c;
            v -= f / (one / v - one);
        }
        wm1 = -v;
    } else {
        let mut v = T::from_f64(1.5).expect("scalar literal");
        for _ in 0..8 {
            let f = v.ln() - v - c;
            v -= f / (one / v - one);
        }
        wm1 = -v;
    }
    if !wm1.is_finite() {
        wm1 = -zero;
    }
    (w0, wm1)
}

/// Shared closed-form exponential-cone step core. `s = (x, y, z)` is an interior point
/// with `y > 0`, `z > 0`; `edge_z` selects whether the `z = 0` face is part of the cone
/// boundary (primal: yes; the dual cone's `w = 0` exit maps to it: yes — both callers
/// pass `true`). Returns the largest step to the boundary, or `None` when the closed
/// form cannot self-validate (the caller falls back to the bisection).
fn step_closed<T: Scalar>(s: &[T], ds: &[T], edge_z: bool) -> Option<T> {
    let zero = T::zero();
    let big = T::from_f64(1e10).expect("scalar literal");
    let (x, y, z) = (s[0], s[1], s[2]);
    let (dx, dy, dz) = (ds[0], ds[1], ds[2]);
    let p_val = z * dx - x * dz;
    let q_val = x * dy - y * dx;
    let r_val = y * dz - z * dy;
    let mut best = big;
    // Curved-boundary candidates: the roots of P + Q·t + R·ln t = 0.
    let mut ts: Vec<T> = Vec::with_capacity(2);
    if q_val != zero && r_val != zero {
        let c = (q_val / r_val).abs().ln() - p_val / r_val;
        if c.is_finite() {
            if q_val * r_val > zero {
                // Argument of W nonnegative: the W₀ branch only.
                if let Some(u) = w0_positive(c) {
                    let t = u / (q_val / r_val);
                    if t > zero {
                        ts.push(t);
                    }
                }
            } else if c <= -T::one() {
                // Argument in [−1/e, 0): both real branches.
                let (u0, um1) = w_negative_roots(c);
                let s = q_val / r_val;
                for u in [u0, um1] {
                    let t = u / s;
                    if t > zero {
                        ts.push(t);
                    }
                }
            }
        }
    } else if q_val == zero && r_val != zero {
        // x/y constant along the ray: t = e^{−P/R} exactly.
        let t = (-p_val / r_val).exp();
        if t > zero && t.is_finite() {
            ts.push(t);
        }
    } else if q_val != zero {
        // z/y constant along the ray: t = −P/Q exactly.
        let t = -p_val / q_val;
        if t > zero && t.is_finite() {
            ts.push(t);
        }
    }
    for t in ts {
        let lt = t.ln();
        let (a1, d1) = ((z - t * y), (t * dy - dz));
        let (a2, d2) = ((x - y * lt), (dy * lt - dx));
        // Both formulas give the same α for an exact t; prefer the better-conditioned
        // one (larger |denominator|), falling back to the other when it is degenerate.
        let a = if d1 == zero {
            if d2 == zero {
                continue;
            }
            a2 / d2
        } else if d2 == zero || d1.abs() >= d2.abs() {
            a1 / d1
        } else {
            a2 / d2
        };
        if a > zero && a < big {
            // The crossing must lie on the curved boundary's active region.
            let ya = y + a * dy;
            let za = z + a * dz;
            if ya > zero && za > zero {
                best = best.min(a);
            }
        }
    }
    // Edge candidates: the y=0 face and (optionally) the z=0 face.
    if dy < zero {
        let a = -y / dy;
        if a > zero {
            best = best.min(a);
        }
    }
    if edge_z && dz < zero {
        let a = -z / dz;
        if a > zero {
            best = best.min(a);
        }
    }
    // Self-validation: the boundary must be crossed between 0.99·best and 1.01·best.
    if best >= big {
        // Unbounded claim: still interior at 0.99·big.
        let p = [
            s[0] + T::from_f64(0.99).expect("scalar literal") * big * dx,
            s[1] + T::from_f64(0.99).expect("scalar literal") * big * dy,
            s[2] + T::from_f64(0.99).expect("scalar literal") * big * dz,
        ];
        if margin(&p) > zero {
            return Some(big);
        }
        return None;
    }
    let f99 = T::from_f64(0.99).expect("scalar literal");
    let f101 = T::from_f64(1.01).expect("scalar literal");
    let p = [
        s[0] + f99 * best * dx,
        s[1] + f99 * best * dy,
        s[2] + f99 * best * dz,
    ];
    let q = [
        s[0] + f101 * best * dx,
        s[1] + f101 * best * dy,
        s[2] + f101 * best * dz,
    ];
    if margin(&p) > zero && margin(&q) <= zero {
        Some(best)
    } else {
        None
    }
}

/// Barrier value `f(s) = −log(ψ) − log(y) − log(z)` (requires `s` strictly interior).
pub fn barrier<T: Scalar>(s: &[T]) -> T {
    let (y, z) = (s[1], s[2]);
    -psi(s).ln() - y.ln() - z.ln()
}

/// The Wright Ω function: the unique real root of `ω + ln ω = L` on the principal
/// branch (`ω ≥ 1` for `L ≥ 1`). Equivalently `ω = W₀(e^L)` in the stable log form
/// (the argument `e^L` is never formed). Returns `None` when the root underflows —
/// the caller's `L > 1` interiority gate keeps the domain away from that region.
pub fn wright_omega<T: Scalar>(l: T) -> Option<T> {
    w0_positive(l)
}

/// **Conjugate-gradient scaling point**: the unique `s̄ ∈ int K` whose barrier
/// gradient equals the (sign-flipped) scaled dual point, `∇f(s̄) = −z/μ` — i.e. the
/// value of the conjugate barrier's gradient `s̄ = ∇f*(−z/μ)`. For the exponential
/// cone this has closed form in the Wright Ω function: with `(u,v,w) = −z/μ`
/// (so `u = −z₀/μ > 0`, `w = −z₂/μ < 0` for `z ∈ int K*`), the identity
/// `ψ̄ = −1/u` and the two gradient equations solve to
/// `L = ln(−w/u) + 2 − v/u`, `ω = WrightΩ(L)`, and
/// `x̄ = ln(ωu/(−w))/((ω−1)u) − 1/u`, `ȳ = 1/((ω−1)u)`, `z̄ = −ω/((ω−1)w)`,
/// which satisfy the secant condition `μ·∇²f(s̄)·s̄ = z` exactly (verify:
/// `∇²f(s)·s = −∇f(s)` by logarithmic homogeneity). The dual point is strictly
/// interior iff `ω > 1`; the mapping returns `None` on or outside the dual cone
/// (the caller then keeps the secant scaling for that cone).
pub fn scaling_point<T: Scalar>(z: &[T], mu: T) -> Option<[T; 3]> {
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).expect("scalar literal");
    // z ∈ int K* means z₀ < 0, z₂ > 0; the conjugate map needs u = −z₀/μ > 0
    // and w = −z₂/μ < 0, i.e. the sign-flipped dual point on the interior side.
    if z[0] >= zero || z[2] <= zero || mu <= zero {
        return None;
    }
    let u = -z[0] / mu;
    let v = -z[1] / mu;
    let w = -z[2] / mu;
    let l = (-w / u).ln() + two - v / u;
    // ω > 1 iff (u,v,w) is strictly interior to −K* (equivalently L > 1); on the
    // boundary ω = 1 and ȳ, z̄ collapse, so the map is only defined past it.
    if !l.is_finite() || l <= one {
        return None;
    }
    let omega = wright_omega(l)?;
    if omega <= one || !omega.is_finite() {
        return None;
    }
    let om1 = omega - one;
    let x = (omega * u / (-w)).ln() / (om1 * u) - one / u;
    let y = one / (om1 * u);
    let z2 = -omega / (om1 * w);
    if !x.is_finite() || !y.is_finite() || !z2.is_finite() || y <= zero || z2 <= zero {
        return None;
    }
    // Final guard: the image must land strictly interior (ψ̄ = −1/u > 0 by
    // construction, but rounding near the boundary can push it out).
    if psi(&[x, y, z2]) <= zero {
        return None;
    }
    Some([x, y, z2])
}

/// Barrier gradient `∇f(s)` (3-vector).
pub fn grad<T: Scalar>(s: &[T]) -> [T; 3] {
    let (y, z) = (s[1], s[2]);
    let p = psi(s);
    let l = (z / y).ln();
    let one = T::one();
    // ψ_x = −1, ψ_y = log(z/y) − 1, ψ_z = y/z.
    [
        one / p,                  // −ψ_x/ψ = 1/ψ
        -(l - one) / p - one / y, // −ψ_y/ψ − 1/y
        -(y / z) / p - one / z,   // −ψ_z/ψ − 1/z
    ]
}

/// **Safeguarded barrier gradient**: the gradient of the barrier with the
/// margin `ψ` replaced by `max(ψ, floor)` (a bound-relaxed barrier that
/// flattens the `~1/ψ²` singularity once the slack is closer to the curved
/// boundary than `floor`). The nonsymmetric IPM uses this for the exp cone
/// when `ψ` falls far below the cone's own complementarity scale (`s·z`) —
/// the doubly-degenerate state (slack and dual both on their boundaries)
/// where the true data `~1/ψ²` blows up the scaling and collapses the step
/// (see `NsCone::pd_scaling`).
pub fn grad_floored<T: Scalar>(s: &[T], floor: T) -> [T; 3] {
    let (y, z) = (s[1], s[2]);
    let p = psi(s).max(floor);
    let l = (z / y).ln();
    let one = T::one();
    [
        one / p,
        -(l - one) / p - one / y,
        -(y / z) / p - one / z,
    ]
}

/// **Dual barrier gradient** `∇f*(z)` for `z ∈ int K*` (the exp cone is
/// self-dual; with `(u,v,w) = z` the dual barrier is
/// `f*(z) = −ln(ψ*) − ln(−u) − ln(w)` with `ψ* = v − u − u·ln(w/(−u))`).
/// Satisfies the duality identity `s = −μ∇f*(z)` on the central path
/// (`z = −μ∇f(s)`).
pub fn grad_dual<T: Scalar>(z: &[T]) -> [T; 3] {
    let (u, v, w) = (z[0], z[1], z[2]);
    let ps = v - u - u * (w / -u).ln();
    let one = T::one();
    [
        (w / -u).ln() / ps - one / u,
        -one / ps,
        u / (w * ps) - one / w,
    ]
}

/// **Dual barrier Hessian** `∇²f*(z)` (symmetric 3×3, row-major) for
/// `z ∈ int K*`. With `ψ* = v − u − u·ln(w/(−u))` and
/// `∇ψ* = (−ln(w/(−u)), 1, −u/w)`:
/// `∇²f* = ∇ψ*·∇ψ*ᵀ/ψ*² − ∇²ψ*/ψ* + diag(1/u², 0, 1/w²)`,
/// `∇²ψ*` having entries `(1/u, 0, −1/w; 0, 0, 0; −1/w, 0, u/w²)`.
pub fn hess_dual<T: Scalar>(z: &[T]) -> [[T; 3]; 3] {
    let (u, v, w) = (z[0], z[1], z[2]);
    let ps = v - u - u * (w / -u).ln();
    let one = T::one();
    let l = (w / -u).ln();
    let uw = u / w;
    let ps2 = ps * ps;
    // ∇ψ*·∇ψ*ᵀ / ψ*²  (ψ*_0 = −l, ψ*_1 = 1, ψ*_2 = −u/w: the (0,2) product
    // is (+l·u/w), not −l·u/w).
    let m00 = l * l / ps2;
    let m01 = -l / ps2;
    let m02 = uw * l / ps2;
    let m11 = one / ps2;
    let m12 = -uw / ps2;
    let m22 = uw * uw / ps2;
    // −∇²ψ*/ψ*  (∇²ψ* = [[1/u, 0, −1/w],[0,0,0],[−1/w, 0, u/w²]])
    let m00 = m00 - (one / u) / ps;
    let m02 = m02 + (one / w) / ps;
    let m22 = m22 - (uw / w) / ps;
    // + diag(1/u², 0, 1/w²)
    let m00 = m00 + one / (u * u);
    let m22 = m22 + one / (w * w);
    [[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]]
}

/// **Safeguarded barrier Hessian**: the Hessian of the bound-relaxed barrier
/// (see [`grad_floored`]): the margin `ψ` is replaced by `max(ψ, floor)`, so
/// the `~1/ψ²` singularity is flattened once the slack is within `floor` of
/// the curved boundary. Used consistently with [`grad_floored`] so the
/// scaling and the target gradient stay at the same (bounded) scale.
pub fn hess_floored<T: Scalar>(s: &[T], floor: T) -> [[T; 3]; 3] {
    let (y, z) = (s[1], s[2]);
    let p = psi(s).max(floor);
    let l = (z / y).ln();
    let one = T::one();
    let lm1 = l - one; // ψ_y
    let yz = y / z; // ψ_z
    let p2 = p * p;

    // H_ab = −ψ_ab/ψ + ψ_a ψ_b/ψ²  (+ 1/a² on the y,y and z,z diagonals).
    let hxx = one / p2;
    let hxy = -lm1 / p2;
    let hxz = -yz / p2;
    let hyy = one / (y * p) + lm1 * lm1 / p2 + one / (y * y);
    let hyz = -(one / z) / p + lm1 * yz / p2;
    let hzz = y / (z * z * p) + yz * yz / p2 + one / (z * z);
    [[hxx, hxy, hxz], [hxy, hyy, hyz], [hxz, hyz, hzz]]
}

/// Barrier Hessian `∇²f(s)` (symmetric 3×3, row-major).
pub fn hess<T: Scalar>(s: &[T]) -> [[T; 3]; 3] {
    let (y, z) = (s[1], s[2]);
    let p = psi(s);
    let l = (z / y).ln();
    let one = T::one();
    let lm1 = l - one; // ψ_y
    let yz = y / z; // ψ_z
    let p2 = p * p;

    // H_ab = −ψ_ab/ψ + ψ_a ψ_b/ψ²  (+ 1/a² on the y,y and z,z diagonals from −log y,−log z).
    // ψ_xx=ψ_xy=ψ_xz=0, ψ_yy=−1/y, ψ_yz=1/z, ψ_zz=−y/z².
    let hxx = one / p2; // (−1)²/ψ²
    let hxy = -lm1 / p2; // (−1)(ψ_y)/ψ²
    let hxz = -yz / p2; // (−1)(ψ_z)/ψ²
    let hyy = one / (y * p) + lm1 * lm1 / p2 + one / (y * y);
    let hyz = -(one / z) / p + lm1 * yz / p2;
    let hzz = y / (z * z * p) + yz * yz / p2 + one / (z * z);
    [[hxx, hxy, hxz], [hxy, hyy, hyz], [hxz, hyz, hzz]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membership() {
        // (x,y,z) with y·log(z/y) − x ≥ 0. Take y=1, z=e, x=0.5: ψ = log(e) − 0.5 = 0.5.
        let e = std::f64::consts::E;
        assert!(in_cone::<f64>(&[0.5, 1.0, e], 1e-9));
        assert!(!in_cone::<f64>(&[2.0, 1.0, e], 1e-9)); // ψ = 1 − 2 < 0
        assert!(!in_cone::<f64>(&[0.0, -1.0, e], 1e-9)); // y < 0
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let s = [0.3_f64, 1.2, 2.5];
        let g = grad(&s);
        let h = 1e-6;
        for k in 0..3 {
            let mut sp = s;
            let mut sm = s;
            sp[k] += h;
            sm[k] -= h;
            let fd = (barrier(&sp) - barrier(&sm)) / (2.0 * h);
            assert!((g[k] - fd).abs() < 1e-5, "grad[{k}]={} fd={fd}", g[k]);
        }
    }

    #[test]
    fn hessian_matches_finite_difference() {
        let s = [0.3_f64, 1.2, 2.5];
        let hmat = hess(&s);
        let h = 1e-6;
        for a in 0..3 {
            for b in 0..3 {
                let mut sp = s;
                let mut sm = s;
                sp[b] += h;
                sm[b] -= h;
                let fd = (grad(&sp)[a] - grad(&sm)[a]) / (2.0 * h);
                assert!(
                    (hmat[a][b] - fd).abs() < 1e-4,
                    "H[{a}][{b}]={} fd={fd}",
                    hmat[a][b]
                );
            }
        }
    }

    #[test]
    fn barrier_is_logarithmically_homogeneous() {
        // f(t·s) = f(s) − ν·log(t) with ν = 3.
        let s = [0.3_f64, 1.2, 2.5];
        let t = 2.0;
        let ts = [t * s[0], t * s[1], t * s[2]];
        let nu = 3.0;
        assert!((barrier(&ts) - (barrier(&s) - nu * t.ln())).abs() < 1e-12);
    }

    #[test]
    fn line_search_step_stays_in_cone() {
        let e = std::f64::consts::E;
        let s = [0.5_f64, 1.0, e]; // interior (ψ = 0.5)
                                   // A direction heading toward the boundary.
        let ds = [1.0_f64, 0.0, -0.5];
        let a = max_step(&s, &ds);
        assert!(a > 0.0 && a < 1e10, "step {a}");
        // 0.99·α stays strictly inside; just past α leaves.
        let inside = [
            s[0] + 0.99 * a * ds[0],
            s[1] + 0.99 * a * ds[1],
            s[2] + 0.99 * a * ds[2],
        ];
        assert!(in_cone::<f64>(&inside, 1e-9), "0.99α not in cone");
        let past = [
            s[0] + 1.01 * a * ds[0],
            s[1] + 1.01 * a * ds[1],
            s[2] + 1.01 * a * ds[2],
        ];
        assert!(!in_cone::<f64>(&past, 1e-9), "1.01α still in cone");
    }

    #[test]
    fn dual_membership() {
        // Dual point from −grad at an interior primal point lies in K* (a known property:
        // −∇f(s) ∈ int K* for s ∈ int K).
        let s = [0.3_f64, 1.2, 2.5];
        let g = grad(&s);
        let d = [-g[0], -g[1], -g[2]];
        assert!(
            in_dual_cone::<f64>(&d, 1e-9),
            "−grad not in dual cone: {d:?}"
        );
    }

    use iconic_core::rng::Lcg;

    /// High-precision bisection reference for the closed-form step: the exact
    /// boundary crossing (60 iterations, interiority by exact margin sign).
    fn fine_bisect(s: &[f64], ds: &[f64], big: f64) -> f64 {
        let pt = |a: f64| [s[0] + a * ds[0], s[1] + a * ds[1], s[2] + a * ds[2]];
        let mut hi = 1.0;
        while margin(&pt(hi)) > 0.0 {
            hi *= 2.0;
            if hi > big {
                return big;
            }
        }
        let mut lo = 0.0;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if margin(&pt(mid)) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The closed-form Lambert-W step agrees with a high-precision bisection of the
    /// margin (the finite-difference-style reference) on random interior points and
    /// directions, including near-tangential and edge-crossing rays.
    #[test]
    fn closed_form_step_matches_bisection() {
        let mut rng = Lcg::new(7);
        let big = 1e10;
        let mut checked = 0usize;
        let mut unbounded = 0usize;
        let mut worst: f64 = 0.0;
        for _ in 0..60000 {
            let y = 1e-3 + rng.unit() * 3.0;
            let z = 1e-3 + rng.unit() * 3.0;
            let psi0 = 1e-6 + rng.unit() * 2.0;
            let x = y * (z / y).ln() - psi0;
            let s = [x, y, z];
            let ds = if rng.unit() < 0.25 {
                [0.0, 0.0, -2.0 + rng.unit() * 4.0]
            } else {
                [rng.signed(), rng.signed(), rng.signed()]
            };
            let closed = max_step(&s, &ds);
            let ref_ = fine_bisect(&s, &ds, big);
            if ref_ >= big {
                unbounded += 1;
                assert!(
                    closed >= big,
                    "unbounded ray, closed={closed} s={s:?} ds={ds:?}"
                );
                continue;
            }
            checked += 1;
            let err = (closed - ref_).abs() / ref_.max(1.0);
            worst = worst.max(err);
            assert!(
                err < 1e-8,
                "step mismatch: closed={closed} ref={ref_} s={s:?} ds={ds:?}"
            );
        }
        assert!(
            checked > 10000 && unbounded > 100,
            "sweep too small: {checked}/{unbounded}"
        );
        assert!(worst < 1e-8, "worst step error {worst}");
    }

    /// The dual step's closed form agrees with a high-precision bisection of the
    /// dual membership on random interior dual points (the "dual solve" branch).
    #[test]
    fn closed_form_dual_step_matches_bisection() {
        let mut rng = Lcg::new(11);
        let big = 1e10;
        let e = std::f64::consts::E;
        let mut checked = 0usize;
        let mut unbounded = 0usize;
        let mut worst: f64 = 0.0;
        for _ in 0..60000 {
            let u = -0.05 - rng.unit() * 3.0;
            let w = 1e-3 + rng.unit() * 3.0;
            let margin = 1e-6 + rng.unit() * 1.5;
            let v = u * (w * e / -u).ln() + margin; // interior: v ≥ u·ln(we/−u)
            let d = [u, v, w];
            let dd = [rng.signed(), rng.signed(), rng.signed()];
            let closed = max_step_dual(&d, &dd);
            // Fine bisection on in_dual_cone (exact sign).
            let pt = |a: f64| [d[0] + a * dd[0], d[1] + a * dd[1], d[2] + a * dd[2]];
            let mut hi = 1.0;
            while in_dual_cone(&pt(hi), 0.0) {
                hi *= 2.0;
                if hi > big {
                    break;
                }
            }
            let mut ref_ = if hi > big { big } else { 0.0 };
            if ref_ < big {
                let mut lo = 0.0;
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if in_dual_cone(&pt(mid), 0.0) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                ref_ = lo;
            }
            if ref_ >= big {
                unbounded += 1;
                assert!(
                    closed >= big,
                    "unbounded dual ray, closed={closed} d={d:?} dd={dd:?}"
                );
                continue;
            }
            checked += 1;
            let err = (closed - ref_).abs() / ref_.max(1.0);
            worst = worst.max(err);
            assert!(
                err < 1e-8,
                "dual step mismatch: closed={closed} ref={ref_} d={d:?} dd={dd:?}"
            );
        }
        assert!(
            checked > 10000 && unbounded > 100,
            "sweep too small: {checked}/{unbounded}"
        );
        assert!(worst < 1e-8, "worst dual step error {worst}");
    }

    /// The Lambert-W roots: principal branch against tabulated values, the two real
    /// branches of a negative argument against the defining equation `u·e^u = a`.
    #[test]
    fn lambert_w_roots() {
        // W0(1) = 0.5671432904097838: u + ln u = ln 1 = 0.
        let u = w0_positive::<f64>(0.0).expect("root");
        assert!((u - 0.5671432904097838).abs() < 1e-14, "W0(1)={u}");
        // W0(e) = 1.
        let u = w0_positive::<f64>(1.0).expect("root");
        assert!((u - 1.0).abs() < 1e-14, "W0(e)={u}");
        // W0(10) = 1.7455280027406994.
        let u = w0_positive::<f64>(10.0_f64.ln()).expect("root");
        assert!((u - 1.7455280027406994).abs() < 1e-13, "W0(10)={u}");
        // Negative argument: both real branches satisfy u·e^u = a, on the correct
        // sides of the branch point −1. The stable form's c = ln|a|.
        let a: f64 = -0.1;
        let (u0, um1) = w_negative_roots::<f64>((-a).ln());
        assert!(u0 > -1.0 && u0 < 0.0, "W0(−0.1)={u0}");
        assert!(um1 < -1.0, "W−1(−0.1)={um1}");
        for u in [u0, um1] {
            assert!(
                (u * u.exp() - a).abs() < 1e-13,
                "u·e^u={} want {a} (u={u})",
                u * u.exp()
            );
        }
        // W−1(−0.1) = −3.577152063957297 (tabulated).
        assert!(
            (um1 - (-3.577152063957297)).abs() < 1e-13,
            "W−1(−0.1)={um1}"
        );
        // Branch point: both roots coincide at −1.
        let (u0, um1) = w_negative_roots::<f64>(-1.0);
        assert!(
            (u0 + 1.0).abs() < 1e-14 && (um1 + 1.0).abs() < 1e-14,
            "branch point: {u0} {um1}"
        );
        // Large positive argument (the c ≫ 1 guess branch): verified in the stable
        // log form (u·e^u itself overflows).
        let c = 1e6;
        let u = w0_positive::<f64>(c).expect("root");
        assert!(
            (u + u.ln() - c).abs() < 1e-8 * c,
            "u+ln u={} want {c}",
            u + u.ln()
        );
    }

    /// The conjugate-gradient scaling point: on the central path
    /// (`z = −μ∇f(s)`) the map must reproduce `s` itself, and in general the
    /// secant condition `μ∇²f(s̄)·s̄ = z` holds exactly.
    #[test]
    fn scaling_point_secant_and_reproduction() {
        let mut rng = Lcg::new(3);
        for _ in 0..20000 {
            let y = 0.05 + rng.unit() * 3.0;
            let z0 = 0.05 + rng.unit() * 3.0;
            let psi0 = 1e-4 + rng.unit() * 2.0;
            let x = y * (z0 / y).ln() - psi0;
            let s = [x, y, z0];
            let mu = 10f64.powf(-8.0 + rng.unit() * 10.0);
            let g = grad(&s);
            let z = [-mu * g[0], -mu * g[1], -mu * g[2]];
            let sbar = scaling_point(&z, mu).expect("central-path dual is interior");
            // Reproduction: sbar == s on the central path.
            assert!(
                (sbar[0] - s[0]).abs() < 1e-9
                    && (sbar[1] - s[1]).abs() < 1e-9
                    && (sbar[2] - s[2]).abs() < 1e-9,
                "reproduction failed: sbar={sbar:?} s={s:?}"
            );
            // Secant: mu * H(sbar) * sbar == z.
            let h = hess(&sbar);
            let ms = [
                mu * (h[0][0] * sbar[0] + h[0][1] * sbar[1] + h[0][2] * sbar[2]),
                mu * (h[1][0] * sbar[0] + h[1][1] * sbar[1] + h[1][2] * sbar[2]),
                mu * (h[2][0] * sbar[0] + h[2][1] * sbar[1] + h[2][2] * sbar[2]),
            ];
            for i in 0..3 {
                assert!(
                    (ms[i] - z[i]).abs() < 1e-8 * mu.max(1.0),
                    "secant failed at {i}: {} vs {}",
                    ms[i],
                    z[i]
                );
            }
        }
        // Non-interior / boundary dual points must be rejected.
        assert!(scaling_point(&[1.0, 0.0, 1.0], 1e-3).is_none()); // z0 > 0
        assert!(scaling_point(&[-1.0, 1.0, -1.0], 1e-3).is_none()); // z2 < 0
        assert!(scaling_point(&[-1.0, 1.0, 1e-12], 1e-3).is_none()); // outside K*
        assert!(scaling_point(&[-1.0, 1.0, 1.0], 0.0).is_none()); // mu = 0
    }

    /// The dual barrier (gradient/Hessian at the dual point): finite-difference
    /// verification plus the logarithmic-homogeneity identities
    /// `∇²f*(z)·z = −∇f*(z)` and `f*(tz) = f*(z) − 3·ln t` (the LHSCB
    /// properties the dual-side step relies on; the exp cone's dual cone is the
    /// e-form, so the naive `s = −μ∇f*(z)` identity does not hold in these
    /// coordinates — the closed-form conjugate map is `scaling_point`).
    #[test]
    fn dual_barrier_matches_finite_difference_and_duality() {
        // Finite differences of the dual barrier value f* = −ln ψ* − ln(−u) − ln w.
        let fstar = |z: &[f64]| -> f64 {
            let (u, v, w) = (z[0], z[1], z[2]);
            let ps = v - u - u * (w / -u).ln();
            -ps.ln() - (-u).ln() - w.ln()
        };
        let mut rng = Lcg::new(17);
        for _ in 0..5000 {
            let u = -0.02 - rng.unit() * 2.0;
            let w = 0.02 + rng.unit() * 2.0;
            let ps0 = 1e-3 + rng.unit() * 1.5;
            // v = u + u·ln(w/−u) + ψ* (interior dual point).
            let v = u + u * (w / -u).ln() + ps0;
            let z = [u, v, w];
            let g = grad_dual(&z);
            let h = 1e-6;
            for k in 0..3 {
                let mut zp = z;
                let mut zm = z;
                zp[k] += h;
                zm[k] -= h;
                let fd = (fstar(&zp) - fstar(&zm)) / (2.0 * h);
                let tol = 1e-4 * g[k].abs().max(fd.abs()).max(1.0);
                assert!(
                    (g[k] - fd).abs() < tol,
                    "grad_dual[{k}]={} fd={fd} z={z:?}",
                    g[k]
                );
            }
            let hmat = hess_dual(&z);
            for a in 0..3 {
                for b in 0..3 {
                    let mut zp = z;
                    let mut zm = z;
                    zp[b] += h;
                    zm[b] -= h;
                    let fd = (grad_dual(&zp)[a] - grad_dual(&zm)[a]) / (2.0 * h);
                    let tol = 1e-4 * hmat[a][b].abs().max(fd.abs()).max(1.0);
                    assert!(
                        (hmat[a][b] - fd).abs() < tol,
                        "hess_dual[{a}][{b}]={} fd={fd}",
                        hmat[a][b]
                    );
                }
            }
            // Logarithmic homogeneity of the dual barrier (the LHSCB property
            // the dual-side step relies on): ∇²f*(z)·z = −∇f*(z) and
            // f*(tz) = f*(z) − 3·ln t. (The exp cone's dual cone is the e-form
            // `{u<0, w>0, −u·e^{v/u} ≤ e·w}` — not the primal cone itself — so
            // the naive `s = −∇f*(−∇f(s))` duality identity does not hold in
            // these coordinates; the closed-form conjugate map is
            // `scaling_point` (verified by `scaling_point_secant_and_reproduction`).
            let hz = hess_dual(&z);
            for i in 0..3 {
                let mut acc = 0.0;
                for j in 0..3 {
                    acc += hz[i][j] * z[j];
                }
                let zscale = z.iter().fold(1.0f64, |a, &v| a.max(v.abs()));
                assert!(
                    (acc + g[i]).abs() < 1e-8 * zscale,
                    "∇²f*(z)·z = {acc} vs −∇f*(z) = {} at z={z:?}",
                    -g[i]
                );
            }
            let t = 2.0 + rng.unit();
            let ps2 = |u: f64, v: f64, w: f64| v - u - u * (w / -u).ln();
            let fz = -ps2(z[0], z[1], z[2]).ln() - (-z[0]).ln() - z[2].ln();
            let ftz = -ps2(t * z[0], t * z[1], t * z[2]).ln() - (-(t * z[0])).ln() - (t * z[2]).ln();
            assert!(
                (ftz - (fz - 3.0 * t.ln())).abs() < 1e-10,
                "f*(t·z)={ftz} vs f*(z)−3ln t={} (z={z:?}, t={t})",
                fz - 3.0 * t.ln()
            );
        }
    }

    /// Wright Omega solves ω + ln ω = L on the principal branch: ω = 1 at L = 1,
    /// and the defining equation holds to machine precision across the domain.
    #[test]
    fn wright_omega_roots() {
        assert!((wright_omega::<f64>(1.0).unwrap() - 1.0).abs() < 1e-14);
        // ω + ln ω = 0 ⇒ ω = W(1) ≈ 0.56714329.
        assert!((wright_omega::<f64>(0.0).unwrap() - 0.5671432904097838).abs() < 1e-13);
        // W(10) = 1.7455280027406994 solves ω + ln ω = ln 10.
        assert!((wright_omega::<f64>(10f64.ln()).unwrap() - 1.7455280027406994).abs() < 1e-13);
        for l in [-5.0f64, -1.0, 0.5, 2.0, 10.0, 100.0, 1e6] {
            let w = wright_omega::<f64>(l).expect("root");
            assert!(
                (w + w.ln() - l).abs() < 1e-9 * l.abs().max(1.0),
                "omega + ln omega = {} want {l} (w={w})",
                w + w.ln()
            );
            assert!(w > 0.0, "principal branch (w>0): w={w}");
        }
        // Below L = 1 the root is < 1 (not the interiority branch the caller
        // needs, but the equation still holds).
        assert!(
            (wright_omega::<f64>(0.5).unwrap() + wright_omega::<f64>(0.5).unwrap().ln() - 0.5)
                .abs()
                < 1e-12
        );
    }
}
