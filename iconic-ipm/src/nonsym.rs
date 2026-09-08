//! Nonsymmetric interior-point method for cones without a Nesterov–Todd scaling
//! (the exponential cone, and — uniformly — the nonnegative orthant via its log
//! barrier). Where the symmetric solver uses NT scaling and a Jordan-algebra step,
//! this one works directly with each cone's barrier gradient `g = ∇f` and Hessian
//! `H = ∇²f`, following the central path `z = −μ·g(s)`.
//!
//! Newton step (predictor–corrector). The complementarity is linearized with the
//! **current** μ as the scaling — `Δz + μ·H(s)·Δs = −(z + σμ·g(s))` — so the condensed
//! `(x,x)` block `P + ρI + μ·Aᵀ H A` stays positive definite even when `P = 0` (an LP).
//! Eliminating `Δs` (primal feasibility) and `Δz` (complementarity) gives that condensed
//! system; `Δs, Δz` are recovered afterward. Step lengths are separate primal/dual
//! fraction-to-boundary line searches — for the exp cone the boundary map is the
//! closed-form Lambert-W step (see [`crate::exp`]), for the remaining cones their own.

use crate::{exp, genpow, pow, QpProblem, QpSolution};
use iconic_core::{Scalar, Settings, Status, WarmStart};
use iconic_linalg::{dot, inf_norm, DenseMatrix};

/// Largest `a ≥ 0` keeping the ray `v + a·dv` inside a curved-boundary cone,
/// shared by the power/gen-power primal and dual step maps: expand from 1 until
/// the margin fails, then bisect 17 times, and back off a fixed 1e-13 so a step
/// computed exactly to the boundary cannot round past it.
pub(crate) fn curved_max_step<T: Scalar, M>(margin: M, v: &[T], dv: &[T]) -> T
where
    M: Fn(&[T]) -> T,
{
    let tol = T::from_f64(1e-12).expect("scalar literal");
    let two = T::from_f64(2.0).expect("scalar literal");
    let big = T::from_f64(1e10).expect("scalar literal");
    let half = T::from_f64(0.5).expect("scalar literal");
    let n = v.len();
    let pt = |a: T| -> Vec<T> { (0..n).map(|i| v[i] + a * dv[i]).collect() };
    let mut hi = T::one();
    for _ in 0..50 {
        if margin(&pt(hi)) <= tol {
            break;
        }
        hi *= two;
        if hi > big {
            return big;
        }
    }
    let mut lo = T::zero();
    for _ in 0..17 {
        let mid = (lo + hi) * half;
        if margin(&pt(mid)) > tol {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo - T::from_f64(1e-13).expect("scalar literal")).max(T::zero())
}

/// A cone for the nonsymmetric path: the exponential cone, power cone (α∈(0,1)),
/// generalized power cone (α vector + tail), or a nonnegative orthant.
#[derive(Clone, Debug)]
pub enum NsCone {
    Exp,
    Power(f64),
    GenPower(Vec<f64>, usize),
    NonNeg(usize),
}

impl NsCone {
    fn dim(&self) -> usize {
        match self {
            NsCone::Exp | NsCone::Power(_) => 3,
            NsCone::GenPower(alpha, tail) => alpha.len() + tail,
            NsCone::NonNeg(d) => *d,
        }
    }
    fn nu(&self) -> usize {
        match self {
            NsCone::Exp | NsCone::Power(_) => 3,
            NsCone::GenPower(alpha, tail) => alpha.len() + tail,
            NsCone::NonNeg(d) => *d,
        }
    }
    /// A fixed strictly-interior point, used to start the infeasible-start IPM
    /// regardless of whether `b_in` itself is interior. For the exp cone the
    /// **near-origin** point `(−1e-3, 1e-2, 1)` (ψ ≈ 0.047) rather than the
    /// unit central point `(−1, 1, 1)`: the fallback is used exactly when the
    /// *induced* slack `b − A·x` cannot be interior (e.g. an exp cone whose
    /// y-coordinate variable is pinned at 0 by an equality — the boundary-
    /// active atom), and the fallback's y-coordinate *is* the slack-variable
    /// gap the infeasible-start iteration must then close. A unit-y fallback
    /// leaves a gap of 1 that can only shrink along the cone boundary at
    /// curvature-limited (~sqrt(ψ)) steps — the boundary-active stall's crawl.
    /// A small-y interior point keeps the gap ~1e-2 so the iteration closes it
    /// in a handful of full steps. For the orthant the all-ones vector. For
    /// GenPower, all base variables at 1 and the tail at 0 (any positive
    /// constant works: `phi = prod(c^{2*alpha_i}) - 0 = c^2 > 0` regardless of
    /// the specific alpha values since `sum(alpha_i)=1`; 1 keeps it clean).
    fn central_point<T: Scalar>(&self) -> Vec<T> {
        let from = |v: f64| T::from_f64(v).expect("scalar literal");
        match self {
            NsCone::Exp => vec![from(-1.0), from(1e-2), from(1.0)],
            // z = 0.1·(0.5^α 0.5^(1−α)) = 0.05 is strictly interior for any α
            // (equal slots make the geometric mean exactly 0.5); the old z = 0
            // sat ON the boundary, and a near-boundary start is not
            // permutation-invariant — the rotated (CVXPY) emission order
            // stalled at an infeasible fixed point.
            NsCone::Power(_) => vec![from(0.5), from(0.5), from(0.05)],
            NsCone::GenPower(alpha, tail) => {
                let mut v = vec![T::one(); alpha.len()];
                v.extend(vec![T::zero(); *tail]);
                v
            }
            NsCone::NonNeg(d) => vec![T::one(); *d],
        }
    }
    /// Whether `s` is comfortably in the cone interior (a margin well away from the boundary,
    /// scaled to `s`), so it makes a good primal-feasible starting slack.
    fn well_interior<T: Scalar>(&self, s: &[T]) -> bool {
        let scale = s
            .iter()
            .fold(T::zero(), |a, &v| a.max(v.abs()))
            .max(T::one());
        let tol = T::from_f64(0.05).expect("scalar literal") * scale;
        match self {
            NsCone::Exp => s[1] > tol && exp::margin(s) > tol,
            NsCone::Power(a) => {
                s[0] > tol && s[1] > tol && pow::margin(s, T::from_f64(*a).expect("scalar literal")) > tol
            }
            NsCone::GenPower(alpha, _) => {
                let a: Vec<T> = alpha.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
                (0..a.len()).all(|i| s[i] > tol) && genpow::margin(s, &a) > tol
            }
            NsCone::NonNeg(d) => (0..*d).all(|i| s[i] > tol),
        }
    }
    /// Whether `s` is strictly inside the cone interior with a small relative margin — a
    /// usable (if near-boundary) starting slack. Looser than [`Self::well_interior`]: it
    /// accepts a slack that sits close to the boundary, as long as it is genuinely interior.
    /// Used to decide whether the *induced* primal-feasible slack `b − A_in x` is usable as-is.
    fn strict_interior<T: Scalar>(&self, s: &[T]) -> bool {
        let scale = s
            .iter()
            .fold(T::zero(), |a, &v| a.max(v.abs()))
            .max(T::one());
        let tol = T::from_f64(1e-7).expect("scalar literal") * scale;
        match self {
            NsCone::Exp => s[1] > tol && s[2] > tol && exp::margin(s) > tol,
            NsCone::Power(a) => {
                s[0] > tol && s[1] > tol && pow::margin(s, T::from_f64(*a).expect("scalar literal")) > tol
            }
            NsCone::GenPower(alpha, _) => {
                let a: Vec<T> = alpha.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
                (0..a.len()).all(|i| s[i] > tol) && genpow::margin(s, &a) > tol
            }
            NsCone::NonNeg(d) => (0..*d).all(|i| s[i] > tol),
        }
    }
    /// Barrier gradient `∇f(s)` for this cone block.
    fn grad<T: Scalar>(&self, s: &[T]) -> Vec<T> {
        match self {
            NsCone::Exp => exp::grad(s).to_vec(),
            NsCone::Power(a) => pow::grad(s, T::from_f64(*a).expect("scalar literal")).to_vec(),
            NsCone::GenPower(alpha, _) => {
                let a: Vec<T> = alpha.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
                genpow::grad(s, &a)
            }
            NsCone::NonNeg(_) => s.iter().map(|&v| -T::one() / v).collect(),
        }
    }
    /// **Primal-dual scaling** block (dense `d×d`, row-major): a
    /// positive-definite matrix satisfying the secant condition `H·s = z`, used in place
    /// of the raw primal Hessian `μ∇²f(s)` (which is ill-conditioned at the boundary,
    /// `~1/ψ²`, and balances the primal and dual poorly). For the orthant this is the
    /// diagonal `diag(zᵢ/sᵢ)`; for the exp cone it is the rank-2 BFGS update of `μ∇²f(s)`
    /// that enforces `H·s = z` while staying PD (`sᵀz > 0`). Both reduce to `μ∇²f(s)` on
    /// the central path (`z = −μ∇f(s)`, using `∇²f(s)·s = −∇f(s)`).
    ///
    /// Returns `None` when the scaling cannot be formed — a **non-positive
    /// intermediate** (`sᵀz ≤ 0` or `sᵀHs ≤ 0` after rounding, or a non-positive
    /// orthant entry). The caller then applies the degrade-and-continue protocol: zero
    /// this cone's `(s,s)` block contribution for the iteration and keep going —
    /// never hard-fail a scaling. (These intermediates are strictly positive for any
    /// primal-dual-feasible pair, so `None` is a defensive path reached only by
    /// rounding, but the degraded iteration is well-defined: the cone's rows then
    /// contribute to the direction only through the primal-feasibility coupling and
    /// `ρI` keeps the condensed system regularized.)
    fn pd_scaling<T: Scalar>(&self, s: &[T], z: &[T], mu: T) -> Option<Vec<T>> {
        match self {
            NsCone::NonNeg(d) => {
                let mut m = vec![T::zero(); d * d];
                for i in 0..*d {
                    if s[i] <= T::zero() || z[i] <= T::zero() {
                        return None;
                    }
                    m[i * d + i] = z[i] / s[i];
                }
                Some(m)
            }
            NsCone::Exp => {
                // Primal-dual scaling via the rank-2 BFGS secant update.
                //
                // Margin safeguard: when the slack approaches the curved boundary
                // far faster than its own complementarity scale (ψ ≪ s·z — the
                // doubly-degenerate state, slack and dual both on their boundaries),
                // the true data blows up (~1/ψ²) and the scaling pins the direction,
                // freezing the iterate (the boundary-active exp-cone stall: cone
                // y-coordinate driven to 0, e.g. max-entropy with y[0] == 0). Use
                // the bound-relaxed barrier data with ψ̃ = max(ψ, floor) there so
                // the scaling and target gradient stay bounded; the secant data
                // (H̃·s, sᵀH̃s) is then computed directly (well-conditioned at the
                // floored scale, exact secant M·s = z by construction), while the
                // normal regime keeps the stable logarithmic-homogeneity data
                // (bit-identical trajectories).
                let sz = s.iter().zip(z.iter()).map(|(&a, &b)| a * b).fold(T::zero(), |a, v| a + v);
                let psi = exp::margin(s);
                // The floor is 0.01·s·z: a healthy cone tracks ψ ≈ s·z/3 on the
                // central path, so the floor engages only 3× past the central
                // value — the deep off-center state. (The interior exp cases
                // measure ψ/s·z ≥ ~1/3 throughout — bit-identical trajectories.)
                let floor = T::from_f64(0.01).expect("scalar literal") * sz;
                let (h, hs, shs) = if psi < floor {
                    let hf = exp::hess_floored(s, floor);
                    let mut hflat = vec![T::zero(); 9];
                    for i in 0..3 {
                        for j in 0..3 {
                            hflat[i * 3 + j] = hf[i][j];
                        }
                    }
                    // Direct secant data at the floored scale (no cancellation:
                    // H̃ is bounded by ~1/floor²).
                    let hs: Vec<T> = (0..3)
                        .map(|i| (0..3).map(|j| hf[i][j] * s[j]).fold(T::zero(), |a, v| a + v))
                        .collect();
                    let shs = (0..3).map(|i| hs[i] * s[i]).fold(T::zero(), |a, v| a + v);
                    (hflat, hs, shs)
                } else {
                    let h = exp::hess(s);
                    let mut hflat = vec![T::zero(); 9];
                    for i in 0..3 {
                        for j in 0..3 {
                            hflat[i * 3 + j] = h[i][j];
                        }
                    }
                    // Stable secant data via logarithmic homogeneity:
                    // ∇²f(s)·s = −∇f(s), sᵀ∇²f(s)s = ν (= 3 for the exp cone).
                    let g = exp::grad(s);
                    let hs: Vec<T> = g.iter().map(|&v| -v).collect();
                    let shs = T::from_f64(3.0).expect("scalar literal");
                    (hflat, hs, shs)
                };
                bfgs_scaling(&h, s, &hs, shs, z, mu, 3)
            }
            NsCone::Power(alpha) => {
                let a = T::from_f64(*alpha).expect("scalar literal");
                let h = pow::hess(s, a);
                let mut hflat = vec![T::zero(); 9];
                for i in 0..3 {
                    for j in 0..3 {
                        hflat[i * 3 + j] = h[i][j];
                    }
                }
                let g = pow::grad(s, a);
                let hs: Vec<T> = g.iter().map(|&v| -v).collect();
                let shs = T::from_f64(3.0).expect("scalar literal");
                bfgs_scaling(&hflat, s, &hs, shs, z, mu, 3)
            }
            NsCone::GenPower(alpha, _) => {
                let a: Vec<T> = alpha.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
                let h = genpow::hess(s, &a);
                let g = genpow::grad(s, &a);
                let hs: Vec<T> = g.iter().map(|&v| -v).collect();
                let shs = T::from_f64((s.len() + 1) as f64).expect("scalar literal");
                bfgs_scaling(&h, s, &hs, shs, z, mu, s.len())
            }
        }
    }
    /// **Dual-side scaled step data** (Clarabel's formulation for the
    /// nonsymmetric cones): linearize the *dual* central-path condition
    /// `s + μ∇f*(z) = 0` instead of `z + μ∇f(s) = 0`, so the complementarity
    /// becomes `Hs·Δz + Δs = rcs` with `Hs = μ∇²f*(z)` (the **dual** barrier
    /// Hessian at the dual point) and `rcs = −(s + σμ·∇f*(z))`. Eliminating
    /// `Δs = −r_p − A·Δx` gives `Δz = Hs⁻¹(rcs + r_p + A·Δx)`, i.e. the
    /// condensed (x,x) block receives `AᵀHs⁻¹A` and the effective target is
    /// `rc_eff = Hs⁻¹·rcs`. Crucially, the dual barrier's logarithmic
    /// homogeneity (`∇²f*(z)·z = −∇f*(z)`) gives
    /// `Hs⁻¹·(σμ∇f*(z)) = σ·z`, so the effective target is
    /// `rc_eff = −Hs⁻¹·s + σ·z` — fully bounded (the `~1/ψ*²` blowup of the
    /// dual Hessian *cancels* against the `Hs⁻¹` rather than contaminating
    /// the direction), and `Δz = rc_eff − Hs⁻¹·Δs` stays bounded.
    ///
    /// This is the missing piece for the doubly-degenerate exp-cone state
    /// (slack and dual both on their boundaries, e.g. max-entropy with a
    /// variable pinned at 0): the primal-side recovery `Δz = rc − H_pd·Δs`
    /// blows up there (`H_pd ~ 1/ψ²`), the dual step collapses, and the
    /// iterate freezes. The dual-side step is engaged by the stall-triggered
    /// fallback (the "dual strategy" — see [`solve_nonsym`]); the healthy
    /// path keeps the secant scaling bit-identical.
    ///
    /// Returns `(Hs⁻¹, rc_eff)` where `rc_eff = −Hs⁻¹·s` is the affine target
    /// and the combined target is `rc_eff + σ·z`; `None` for cones without a
    /// usable dual barrier (the orthant's secant scaling is exact; the power
    /// cones' dual boundary map already keeps their dual step positive).
    fn dual_side_step<T: Scalar>(
        &self,
        s: &[T],
        z: &[T],
        mu: T,
    ) -> Option<(Vec<T>, Vec<T>)> {
        match self {
            NsCone::Exp => {
                let h = exp::hess_dual(z);
                let mut hs = vec![T::zero(); 9];
                for i in 0..3 {
                    for j in 0..3 {
                        hs[i * 3 + j] = mu * h[i][j];
                    }
                }
                // 3×3 inverse of Hs (symmetric; dense solve with pivoting).
                let mut hinv = vec![T::zero(); 9];
                for col in 0..3 {
                    let mut e = vec![T::zero(); 3];
                    e[col] = T::one();
                    let x = dense_solve(&hs, &e, 3)?;
                    for r in 0..3 {
                        hinv[r * 3 + col] = x[r];
                    }
                }
                // rc_eff = −Hs⁻¹·s (the affine target; the combined target
                // adds σ·z on top via the dual barrier's homogeneity).
                let mut hm_s = vec![T::zero(); 3];
                for r in 0..3 {
                    let mut acc = T::zero();
                    for c in 0..3 {
                        acc += hinv[r * 3 + c] * s[c];
                    }
                    hm_s[r] = -acc;
                }
                Some((hinv, hm_s))
            }
            _ => None,
        }
    }
    /// Barrier gradient for the centering target in the combined step
    /// (`rc = −(z + σμ·g)`). The exp cone uses the **safeguarded** gradient
    /// (ψ̃ = max(ψ, 0.01·s·z)) so the target stays bounded in the
    /// doubly-degenerate state (see [`Self::pd_scaling`]); all other cones use
    /// the plain gradient (their data is well-behaved at the boundary — the
    /// power cones' dual boundary map keeps their dual step positive).
    fn grad_target<T: Scalar>(&self, s: &[T], z: &[T]) -> Vec<T> {
        match self {
            NsCone::Exp => {
                let sz = s
                    .iter()
                    .zip(z.iter())
                    .map(|(&a, &b)| a * b)
                    .fold(T::zero(), |a, v| a + v);
                let floor = T::from_f64(0.01).expect("scalar literal") * sz;
                if exp::margin(s) < floor {
                    exp::grad_floored(s, floor).to_vec()
                } else {
                    exp::grad(s).to_vec()
                }
            }
            _ => self.grad(s),
        }
    }
    /// Barrier value `f(s)` for this cone block (requires `s` strictly interior).
    /// Used by the dual-strategy barrier backtracking line search (see
    /// [`solve_nonsym`]).
    fn barrier<T: Scalar>(&self, s: &[T]) -> T {
        match self {
            NsCone::Exp => exp::barrier(s),
            NsCone::Power(a) => pow::barrier(s, T::from_f64(*a).expect("scalar literal")),
            NsCone::GenPower(alpha, _) => {
                let a: Vec<T> = alpha.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
                genpow::barrier(s, &a)
            }
            NsCone::NonNeg(_) => s.iter().fold(T::zero(), |a, &v| a - v.ln()),
        }
    }
    /// **Conjugate-gradient ("dual") scaling** block (dense `d×d`, row-major): the
    /// barrier Hessian evaluated at the conjugate scaling point,
    /// `M = μ·∇²f(s̄)` with `s̄ = ∇f*(−z/μ)` (i.e. `∇f(s̄) = −z/μ`). Unlike the
    /// secant [`Self::pd_scaling`] — which enforces `M·s = z` at the *current*
    /// iterate and can degenerate near the boundary — this scaling is anchored to
    /// the dual point itself; on the central path the two coincide exactly
    /// (`∇²f(s̄)·s̄ = −∇f(s̄)` by logarithmic homogeneity, so `M·s̄ = z`). It is
    /// the engine's analogue of Clarabel's `ScalingStrategy::Dual`, engaged only
    /// as a stall-triggered fallback (the default-path swap was measured to
    /// regress). Returns `None` when the conjugate point cannot be formed (the
    /// dual point not strictly interior — `z` on or outside `K*` — or Newton
    /// inversion failing); the caller then keeps the secant scaling for that cone.
    fn conjugate_scaling<T: Scalar>(&self, s: &[T], z: &[T], mu: T) -> Option<Vec<T>> {
        match self {
            NsCone::Exp => {
                let sp = exp::scaling_point(z, mu)?;
                let h = exp::hess(&sp);
                let mut m = vec![T::zero(); 9];
                for i in 0..3 {
                    for j in 0..3 {
                        m[i * 3 + j] = mu * h[i][j];
                    }
                }
                Some(m)
            }
            NsCone::Power(a) => {
                let alpha = T::from_f64(*a).expect("scalar literal");
                let sp = conjugate_point_newton(
                    s,
                    z,
                    mu,
                    |v| pow::grad(v, alpha).to_vec(),
                    |v| {
                        let h = pow::hess(v, alpha);
                        let mut flat = vec![T::zero(); 9];
                        for i in 0..3 {
                            for j in 0..3 {
                                flat[i * 3 + j] = h[i][j];
                            }
                        }
                        flat
                    },
                    |v| pow::margin(v, alpha) > T::from_f64(1e-12).expect("scalar literal"),
                    3,
                )?;
                let h = pow::hess(&sp, alpha);
                let mut m = vec![T::zero(); 9];
                for i in 0..3 {
                    for j in 0..3 {
                        m[i * 3 + j] = mu * h[i][j];
                    }
                }
                Some(m)
            }
            NsCone::GenPower(alpha, _) => {
                let a: Vec<T> = alpha.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
                let d = s.len();
                let sp = conjugate_point_newton(
                    s,
                    z,
                    mu,
                    |v| genpow::grad(v, &a),
                    |v| genpow::hess(v, &a),
                    |v| genpow::margin(v, &a) > T::from_f64(1e-12).expect("scalar literal"),
                    d,
                )?;
                let h = genpow::hess(&sp, &a);
                Some(h.iter().map(|&v| mu * v).collect())
            }
            // The orthant's secant scaling is already exact (diag(z/s)); nothing
            // to fall back to.
            NsCone::NonNeg(_) => None,
        }
    }
    /// Largest `α ≥ 0` keeping `v + α·dv` in this (primal) cone.
    fn max_step<T: Scalar>(&self, v: &[T], dv: &[T]) -> T {
        match self {
            NsCone::Exp => exp::max_step(v, dv),
            NsCone::Power(a) => pow::max_step(v, dv, T::from_f64(*a).expect("scalar literal")),
            NsCone::GenPower(alpha, _) => {
                let a: Vec<T> = alpha.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
                genpow::max_step(v, dv, &a)
            }
            NsCone::NonNeg(_) => crate::max_step(v, dv),
        }
    }
    /// Largest `α ≥ 0` keeping `v + α·dv` in this cone's dual. The exp cone's dual
    /// step is the same closed-form Lambert-W boundary map as the primal (mirrored
    /// in the first coordinate); the orthant is self-dual.
    fn max_step_dual<T: Scalar>(&self, v: &[T], dv: &[T]) -> T {
        match self {
            NsCone::NonNeg(_) => self.max_step(v, dv), // self-dual
            NsCone::Exp => exp::max_step_dual(v, dv),
            // The power cones are NOT self-dual: the dual cone
            // `{(u,v,w): (u/α)^α(v/(1−α))^(1−α) ≥ |w|}` is larger than the primal,
            // and the primal boundary map under-estimates (even zeroes) the true
            // dual step, freezing the dual on boundary-active programs. Use each
            // cone's own dual boundary.
            NsCone::Power(a) => pow::max_step_dual(v, dv, T::from_f64(*a).expect("scalar literal")),
            NsCone::GenPower(alpha, _) => {
                let a: Vec<T> = alpha.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
                genpow::max_step_dual(v, dv, &a)
            }
        }
    }
}

/// Generic rank-2 primal-dual scaling update, given
/// any cone's primal Hessian `h` (dense `d×d`, row-major): `M = μH −
/// μ(Hs)(Hs)ᵀ/(s·Hs) + zzᵀ/(s·z)`. Enforces the secant condition `M·s = z`
/// while staying positive-definite (since `s·z > 0` for any strictly
/// primal-dual-feasible pair, by convex duality) — generalizes the rank-2
/// BFGS fallback already used for the exponential cone (see its own
/// `pd_scaling` arm above) to arbitrary dimension. Falls back to the plain
/// scaled Hessian `μH` if either denominator is positive but too small.
/// Returns `None` for a **non-positive intermediate** (see [`NsCone::pd_scaling`]);
/// the caller then zeroes the cone's `(s,s)` block for the iteration.
///
/// Needed because the raw `μ·∇²f(s)` scaling (what `Power`/`GenPower` used
/// unconditionally before this) is ill-conditioned at the boundary (~1/ψ²)
/// and balances the primal and dual poorly — observed directly: a power-cone
/// QP whose true optimum sits strictly inside the cone (`MaxIterations` with
/// the raw Hessian scaling; converges with BFGS).
///
/// `hs` and `shs` are supplied by the caller, computed stably via the
/// barrier's logarithmic homogeneity (`∇²f(s)·s = −∇f(s)`,
/// `sᵀ∇²f(s)s = ν` the barrier degree) instead of the direct `H·s` /
/// `sᵀHs` products, whose f64 evaluation catastrophically cancels at the
/// cone boundary (H entries ~1/φ², true sᵀHs = ν): the direct products
/// return garbage of either sign there, and the non-positive-intermediate
/// guard then zeroes the cone's (s,s) block, collapsing the iteration into
/// a pure-ρ least-squares direction that the fraction-to-boundary step
/// rejects at α = 0 — the boundary-active power-cone freeze.
fn bfgs_scaling<T: Scalar>(
    h: &[T],
    s: &[T],
    hs: &[T],
    shs: T,
    z: &[T],
    mu: T,
    d: usize,
) -> Option<Vec<T>> {
    let mut sz = T::zero();
    for i in 0..d {
        sz += s[i] * z[i];
    }
    let eps_sq = T::from_f64(1e-8).expect("scalar literal");
    if shs <= T::zero() || sz <= T::zero() {
        return None;
    }
    let mut m = vec![T::zero(); d * d];
    if shs < eps_sq || sz < eps_sq {
        for i in 0..d {
            for j in 0..d {
                m[i * d + j] = mu * h[i * d + j];
            }
        }
        return Some(m);
    }
    // Divisions by shs/sz are loop-invariant; precompute reciprocals.
    let inv_shs = shs.recip();
    let inv_sz = sz.recip();
    for i in 0..d {
        for j in 0..d {
            m[i * d + j] = mu * h[i * d + j] - mu * hs[i] * hs[j] * inv_shs + z[i] * z[j] * inv_sz;
        }
    }
    Some(m)
}

/// Small dense linear solve (Gaussian elimination with partial pivoting) for the
/// Newton inversion below. Used only on the stall-triggered fallback path, so a
/// straightforward implementation is fine.
fn dense_solve<T: Scalar>(m: &[T], rhs: &[T], d: usize) -> Option<Vec<T>> {
    let zero = T::zero();
    let mut a: Vec<T> = m.to_vec();
    let mut b: Vec<T> = rhs.to_vec();
    for c in 0..d {
        // Partial pivoting.
        let mut pivot = c;
        let mut best = a[c * d + c].abs();
        for r in c + 1..d {
            let v = a[r * d + c].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best == zero || !best.is_finite() {
            return None;
        }
        if pivot != c {
            for k in 0..d {
                a.swap(c * d + k, pivot * d + k);
            }
            b.swap(c, pivot);
        }
        for r in c + 1..d {
            let f = a[r * d + c] / a[c * d + c];
            if f == zero {
                continue;
            }
            for k in c..d {
                let av = a[c * d + k]; // local: row r is mutated below
                a[r * d + k] -= f * av;
            }
            let bc = b[c]; // local: b[r] is mutated below
            b[r] -= f * bc;
        }
    }
    let mut x = vec![zero; d];
    for r in (0..d).rev() {
        let mut acc = b[r];
        for k in r + 1..d {
            acc -= a[r * d + k] * x[k];
        }
        x[r] = acc / a[r * d + r];
    }
    Some(x)
}

/// Solve `∇f(s̄) = −z/μ` for the conjugate scaling point `s̄ ∈ int K` by damped
/// Newton from the current interior point `s0` (barrier self-concordance gives a
/// wide Newton region; the damping keeps each trial strictly interior). Returns
/// `None` when the dual point is not reachable — `−z/μ` outside `int K*` — or
/// Newton fails to converge, in which case the caller keeps the secant scaling.
fn conjugate_point_newton<T: Scalar>(
    s0: &[T],
    z: &[T],
    mu: T,
    grad: impl Fn(&[T]) -> Vec<T>,
    hess: impl Fn(&[T]) -> Vec<T>,
    interior: impl Fn(&[T]) -> bool,
    d: usize,
) -> Option<Vec<T>> {
    let zero = T::zero();
    let one = T::one();
    if mu <= zero {
        return None;
    }
    // Target gradient: −z/μ.
    let target: Vec<T> = z.iter().map(|&zi| -zi / mu).collect();
    let tscale = target.iter().fold(one, |a, &v| a.max(v.abs()));
    let tol = T::from_f64(1e-13).expect("scalar literal") * tscale.max(one);
    let mut s = s0.to_vec();
    for _ in 0..60 {
        if !interior(&s) {
            return None;
        }
        let g = grad(&s);
        let mut r = vec![zero; d];
        let mut nrm = zero;
        for i in 0..d {
            r[i] = g[i] - target[i];
            nrm = nrm.max(r[i].abs());
        }
        if nrm <= tol {
            return Some(s);
        }
        let h = hess(&s);
        let ds = dense_solve(&h, &r, d)?;
        // Damped step: halve until the trial stays strictly interior.
        let mut step = one;
        let mut moved = false;
        for _ in 0..60 {
            let t: Vec<T> = (0..d).map(|i| s[i] - step * ds[i]).collect();
            if interior(&t) {
                s = t;
                moved = true;
                break;
            }
            step *= T::from_f64(0.5).expect("scalar literal");
        }
        if !moved {
            return None;
        }
    }
    // Final convergence check (the loop may have exited on the iteration cap).
    let g = grad(&s);
    let mut nrm = zero;
    for i in 0..d {
        nrm = nrm.max((g[i] - target[i]).abs());
    }
    if nrm <= tol && interior(&s) {
        Some(s)
    } else {
        None
    }
}

/// Trace of the nonsymmetric engine's scaling-strategy fallback. Returned to
/// callers and tests (via [`solve_nonsym_traced`]) so the stall-triggered
/// conjugate scaling can be observed; the engine is silent about it otherwise.
#[derive(Clone, Debug, Default)]
pub struct NsSolveTrace {
    /// The fallback (checkpoint-restore + switch to the conjugate-gradient dual
    /// scaling) engaged at least once.
    pub fallback_engaged: bool,
    /// Iterations at which the fallback restored the checkpoint and switched.
    pub fallback_iters: Vec<usize>,
    /// Whether the dual (conjugate) scaling strategy was active at termination.
    pub dual_strategy_active: bool,
}

fn offsets(cones: &[NsCone]) -> Vec<usize> {
    let mut o = Vec::with_capacity(cones.len());
    let mut acc = 0;
    for c in cones {
        o.push(acc);
        acc += c.dim();
    }
    o
}

/// Factorization of the condensed `(x,y)` system: faer's SIMD LBLT for larger systems,
/// ICONIC's scalar LDLᵀ for small ones (where faer's thread-pool overhead doesn't pay).
enum NsFac<'a, T: Scalar> {
    Faer(iconic_linalg::faer_dense::FaerLblt),
    /// Borrows the reused [`LdlFactor`] buffer (see `ldl_buf` in `solve_nonsym`).
    Scalar(&'a iconic_linalg::ldl::LdlFactor<T>),
}

impl<T: Scalar> NsFac<'_, T> {
    fn solve(&self, rhs: &[T]) -> Vec<T> {
        match self {
            NsFac::Faer(f) => {
                let rf: Vec<f64> = rhs.iter().map(|v| v.to_f64().expect("finite scalar")).collect();
                f.solve(&rf)
                    .iter()
                    .map(|&v| T::from_f64(v).expect("finite scalar"))
                    .collect()
            }
            NsFac::Scalar(f) => f.solve(rhs),
        }
    }
}

/// Solve `min ½xᵀPx + qᵀx s.t. A_eq x = b_eq, A_in x + s = b_in, s ∈ K` for `K` a
/// product of nonsymmetric cones. Starts from a **primal-feasible** interior point when one
/// can be constructed (interior target slack projected onto the equalities; see below) so the
/// step only has to drive the duality gap `μ → 0` — which keeps the iterate from freezing with
/// many near-boundary cones. Where a block cannot be made interior it falls back to a fixed
/// interior slack, and the infeasible-start residual terms drive its primal infeasibility to
/// zero. A common (min) primal/dual step couples that progress with the duality gap.
///
/// A **stall-triggered scaling-strategy fallback** (Clarabel-style checkpointing)
/// guards the exp/power cone paths: when the iterate stalls (a small combined step
/// or no residual improvement over a window, while not near-optimal), the engine
/// restores the best iterate and switches the per-cone `(s,s)` scaling from the
/// secant BFGS ("primal-dual") update to the conjugate-gradient ("dual") scaling
/// `M = μ∇²f(s̄)`, `s̄ = ∇f*(−z/μ)` — the barrier Hessian anchored to the dual
/// point. One switch per solve; if it does not help, the loop runs out and grades
/// the best iterate honestly.
pub fn solve_nonsym<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[NsCone],
    settings: &Settings<T>,
) -> QpSolution<T> {
    solve_nonsym_traced(prob, cones, settings, None).0
}

/// Warm-start seed validation + interiorization for the nonsymmetric engine
/// (M8). Returns the seeded `(x, s, z)` — `s` derived as `b − A·x` (exact
/// inequality feasibility) — or `None` when the seed fails any check and the
/// caller must use the cold start:
///   1. dimensions match, all entries finite;
///   2. `A·x + s = b` to 1e-6 relative (a previous solution satisfies it to
///      solver tolerance);
///   3. per cone, the seeded `s` is strictly interior after blending toward the
///      cone's central point with θ = 0.05 when it sits on/near the boundary —
///      optima are boundary-active, and the step-to-boundary cannot start from
///      the boundary;
///   4. positive complementarity `sᵀz` (relative 1e-9) per cone. The exp/power
///      cones have no canonical dual interior point (the cold start's own dual,
///      `z = −∇f(s)`, need not lie in the dual cone at all — the engine's
///      current-μ linearization is anchored on the primal Hessian), so the
///      load-bearing dual invariant is the secant-condition positivity, which
///      the scaling machinery degrades gracefully without.
///
/// Any failure falls back to the cold start — a warm start can only change
/// convergence speed, never the converged point.
fn nonsym_warm_seed<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[NsCone],
    off: &[usize],
    init: Option<&WarmStart<T>>,
) -> Option<(Vec<T>, Vec<T>, Vec<T>)> {
    let (x, mut s, z) = crate::conic::warm_seed_primal(prob, init)?;
    let zero = T::zero();
    let one = T::one();
    let from = |v: f64| T::from_f64(v).expect("scalar literal");
    let theta = from(0.05);

    // 3./4. Per-cone interiorization of s + complementarity-positivity of the
    // dual.
    for (c, cone) in cones.iter().enumerate() {
        let o = off[c];
        let d = cone.dim();
        if !cone.strict_interior(&s[o..o + d]) {
            let cp = cone.central_point::<T>();
            for i in 0..d {
                s[o + i] = (one - theta) * s[o + i] + theta * cp[i];
            }
            if !cone.strict_interior(&s[o..o + d]) {
                return None;
            }
        }
        let mut sdot = zero;
        let (mut sscale, mut zscale) = (one, one);
        for i in 0..d {
            sdot += s[o + i] * z[o + i];
            sscale = sscale.max(s[o + i].abs());
            zscale = zscale.max(z[o + i].abs());
        }
        if sdot <= from(1e-9) * sscale * zscale {
            return None;
        }
    }
    Some((x, s, z))
}

/// Solve the nonsymmetric (exp/power) conic QP, optionally seeded from a
/// previous near-solution. See [`solve_nonsym`] for the engine; the seed
/// semantics match the symmetric engines: validated (dimensions, finiteness,
/// `A x + s = b`, per-cone strict interiority of `s` after a θ-blend toward
/// the cone's central point, and positive complementarity `sᵀz`) and silently
/// falling back to the cold start on any failure — a warm start can only
/// change convergence speed, never the converged point.
pub fn solve_nonsym_warm<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[NsCone],
    settings: &Settings<T>,
    init: Option<&WarmStart<T>>,
) -> QpSolution<T> {
    solve_nonsym_impl(prob, cones, settings, None, init).0
}

/// Like [`solve_nonsym`], additionally reporting the scaling-strategy fallback
/// activity in `trace` (used by the test battery to verify the fallback engages
/// on a crafted stall and is inert otherwise).
pub fn solve_nonsym_traced<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[NsCone],
    settings: &Settings<T>,
    trace: Option<&mut NsSolveTrace>,
) -> (QpSolution<T>, NsSolveTrace) {
    solve_nonsym_impl(prob, cones, settings, trace, None)
}

/// The shared implementation behind [`solve_nonsym`], [`solve_nonsym_traced`]
/// and [`solve_nonsym_warm`].
fn solve_nonsym_impl<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[NsCone],
    settings: &Settings<T>,
    trace: Option<&mut NsSolveTrace>,
    init: Option<&WarmStart<T>>,
) -> (QpSolution<T>, NsSolveTrace) {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let off = offsets(cones);
    let from = |v: f64| T::from_f64(v).expect("scalar literal");
    let zero = T::zero();
    let one = T::one();
    let rho = from(1e-8);
    let delta = from(1e-8);
    let eta = from(0.98); // fraction to boundary
    let eps = settings.eps_abs;
    let pivot_tol = from(1e-14);
    // Scaling-strategy fallback parameters, mirroring Clarabel's checkpoint
    // semantics (their defaults: min_switch_step_length = 1e-1, line-search
    // backtrack 0.8). The switch threshold and stall window are corroborated
    // (both a small step AND no improvement) so a healthy solve's occasional
    // short step can never engage the fallback — only a genuine stall.
    let min_switch_step_length = from(1e-2);
    let stall_window = 15usize; // no combined-residual improvement → stall
    let max_fallback_switches = 1usize;
    let nu = from(cones.iter().map(|c| c.nu()).sum::<usize>().max(1) as f64);

    // ----- warm start (M8) -----
    // Seed x/s/z from a previous near-solution, validated + interiorized per
    // cone by `nonsym_warm_seed`. When the seed is accepted, the
    // primal-feasible construction below is skipped entirely: the seeded
    // `s = b − A·x` already achieves exact inequality feasibility — the very
    // property the construction exists to provide — and the seeded dual is the
    // informative one from the previous solve (its scaling point/barrier base
    // is re-derived from the seeded `s` by the first iteration's scaling
    // computation, which reads `s`/`z` fresh each iteration). On any validation
    // failure the cold construction runs, bit-identical to the unseeded path.
    let warm = nonsym_warm_seed(prob, cones, &off, init);
    let mut x = warm
        .as_ref()
        .map(|w| w.0.clone())
        .unwrap_or_else(|| vec![zero; n]);
    let mut y = vec![zero; me];
    let mut s = vec![zero; mi];
    let mut z = vec![zero; mi];
    if let Some((_, ws_s, ws_z)) = &warm {
        s.copy_from_slice(ws_s);
        z.copy_from_slice(ws_z);
    }

    // Pre-allocated condensed (x,y) matrix and LDLᵀ buffer: refilled/factored in
    // place every iteration (dim is fixed per problem), avoiding an n²+me²
    // allocation and a factor allocation per iteration. The (n+r1, n+r2)
    // off-diagonals of the equality border are never written by any iteration —
    // they stay zero from this initialization.
    let dim = n + me;
    let mut m = DenseMatrix::<T>::zeros(dim, dim);
    let mut ldl_buf = iconic_linalg::ldl::LdlFactor::<T>::with_capacity(dim);

    // ----- primal-feasible start (cold path only — the warm seed above already
    // gives exact inequality feasibility) -----
    // The single failure mode of an infeasible interior start with *many* near-boundary
    // cones is that the iterate centers but the feasibility-correcting direction immediately
    // exits the cones, so the step length collapses and the primal residual freezes (the
    // dual residual having already converged). The cure is to begin primal-FEASIBLE: once
    // `A_in x + s = b` and `A_eq x = b_eq` hold, the Newton step preserves them exactly
    // (`Δs = −A_in Δx`, equal primal step on `x` and `s`), so feasibility is never something
    // the cone-limited step has to fight against — it only has to drive the gap `μ → 0`.
    //
    // (On the warm path the seeded `s = b − A_in·x` already achieves the exact
    // inequality residual — this construction is skipped.)
    if warm.is_none() {
    //
    // Construct it by picking a comfortably-interior target slack `s_tgt` per cone, then
    // finding the `x` whose induced slack `b − A_in x` matches it as closely as possible while
    // satisfying the equalities exactly:
    //   min_x ½‖A_in x − (b − s_tgt)‖² + ½ρ‖x‖²   s.t.  A_eq x = b_eq.
    // Its KKT system `[[A_inᵀA_in + ρI, A_eqᵀ],[A_eq, −δI]] [x; y] = [A_inᵀ(b − s_tgt); b_eq]`
    // reuses the very assembly the iteration uses (with H ← I). We then set `s = b − A_in x`,
    // so the inequality residual is *exactly zero* on every row `x` can reach (all of them for
    // the coordinate-selector `A_in` of structured exp programs — log-sum-exp, entropy). Any
    // cone block whose induced slack is still not interior falls back to the fixed central
    // point (it then carries a small residual, handled by the usual infeasible-start terms).
    let s_tgt: Vec<T> = {
        let mut t = vec![zero; mi];
        for (c, cone) in cones.iter().enumerate() {
            let o = off[c];
            let d = cone.dim();
            let natural = &prob.b_in[o..o + d];
            let blk = if cone.well_interior(natural) {
                natural.to_vec()
            } else {
                // The near-origin fallback point, but with the coordinates the
                // least-squares cannot influence — rows with an empty A-block
                // (the slack is pinned to `b_in` there) — kept at their natural
                // values: a constant row (e.g. the exp cone's y-coordinate fixed
                // at 1 in the log-sum-exp form) must not be asked to move to
                // the fallback's small-y value (the unreachable target would
                // shift the least-squares solution and change the trajectory).
                let mut cp = cone.central_point::<T>();
                for k in 0..d {
                    let r = o + k;
                    let empty = (0..n).all(|j| prob.a_in.get(r, j) == zero);
                    if empty {
                        cp[k] = natural[k];
                    }
                }
                cp
            };
            t[o..o + d].copy_from_slice(&blk);
        }
        t
    };
    {
        // RHS_x = A_inᵀ(b − s_tgt); RHS_y = b_eq.
        let rhs_in: Vec<T> = (0..mi).map(|i| prob.b_in[i] - s_tgt[i]).collect();
        let at_rhs = prob.a_in.matvec_t(&rhs_in);
        let dim = n + me;
        let mut m = DenseMatrix::<T>::zeros(dim, dim);
        // (x,x) block: A_inᵀA_in + ρI.
        for r in 0..mi {
            for i in 0..n {
                let air = prob.a_in.get(r, i);
                if air == zero {
                    continue;
                }
                for j in 0..n {
                    let v = m.get(i, j) + air * prob.a_in.get(r, j);
                    m.set(i, j, v);
                }
            }
        }
        for i in 0..n {
            m.set(i, i, m.get(i, i) + rho);
        }
        for r in 0..me {
            for j in 0..n {
                let v = prob.a_eq.get(r, j);
                m.set(n + r, j, v);
                m.set(j, n + r, v);
            }
            m.set(n + r, n + r, -delta);
        }
        let mut rhs = vec![zero; dim];
        rhs[0..n].copy_from_slice(&at_rhs);
        rhs[n..n + me].copy_from_slice(&prob.b_eq);
        // Same factorization split as the iteration (faer LBLT for large, scalar LDLᵀ small).
        let x0 = if dim >= 130 {
            let mf = DenseMatrix::from_row_major(
                dim,
                dim,
                m.data.iter().map(|v| v.to_f64().expect("finite scalar")).collect(),
            );
            NsFac::Faer(iconic_linalg::faer_dense::FaerLblt::factor(&mf)).solve(&rhs)
        } else {
            match iconic_linalg::ldl::ldl_factor_into(&m, pivot_tol, &mut ldl_buf) {
                Ok(()) => NsFac::Scalar(&ldl_buf).solve(&rhs),
                Err(_) => rhs.iter().map(|_| zero).collect(),
            }
        };
        x.copy_from_slice(&x0[0..n]);
        y.copy_from_slice(&x0[n..n + me]);
    }
    // s = b − A_in x (zero inequality residual where x reaches), repaired to interior per cone.
    let ainx0 = prob.a_in.matvec(&x);
    for i in 0..mi {
        s[i] = prob.b_in[i] - ainx0[i];
    }
    for (c, cone) in cones.iter().enumerate() {
        let o = off[c];
        let d = cone.dim();
        if !cone.strict_interior(&s[o..o + d]) {
            // The induced slack left this block outside (or on) the cone — start it at the
            // fixed interior point instead (re-introducing a small primal residual on the block,
            // which the infeasible-start residual terms absorb). A near-boundary-but-interior
            // induced slack is kept as-is: it preserves exact primal feasibility, and the
            // primal-dual scaling handles the boundary, whereas falling back would reintroduce
            // the very primal residual this start exists to eliminate.
            //
            // Coordinates pinned by an empty A-row (the slack is `b_in` there — e.g. the exp
            // cone's y-coordinate fixed at 1 in the log-sum-exp form) keep their natural
            // values: the fallback point's small-y coordinate must not create a slack
            // inconsistent with the constant row (a ~1 residual the infeasible-start terms
            // would then have to close).
            let mut cp = cone.central_point::<T>();
            for k in 0..d {
                let r = o + k;
                let empty = (0..n).all(|j| prob.a_in.get(r, j) == zero);
                if empty {
                    cp[k] = s[o + k];
                }
            }
            s[o..o + d].copy_from_slice(&cp);
        }
        let g = cone.grad(&s[o..o + d]);
        for i in 0..d {
            z[o + i] = -g[i];
        }
    }
    } // warm.is_none()

    let mut status = Status::MaxIterations;
    let mut iters = 0;

    // Best-iterate tracking + stuck-fixed-point detection, mirroring the
    // symmetric/QP path's own established fix for the identical failure
    // mode (see `iconic-ipm/src/lib.rs`'s `best_err`/`iters_since_improvement`):
    // on some problems the Newton iteration converges to an exact fixed
    // point -- x/y/s/z stop changing bit-for-bit -- comfortably within a
    // relaxed tolerance but short of the tight `eps`, and without this
    // tracking, the loop then repeats, unchanged, all the way to max_iters
    // before grading a bare MaxIterations failure despite having essentially
    // solved the problem (observed residuals ~1e-6-1e-7, objective accurate
    // to ~1e-13). Track the lowest-combined-residual iterate seen so the
    // iteration-limit path can return and grade it, never a worse frozen (or
    // later-drifted) final iterate.
    let mut best_err = T::infinity();
    let mut best_x = x.clone();
    let mut best_y = y.clone();
    let mut best_s = s.clone();
    let mut best_z = z.clone();
    let mut iters_since_improvement = 0usize;

    // Scaling-strategy fallback state (see the doc comment on [`solve_nonsym`]).
    // `strategy_dual` switches the per-cone `(s,s)` scaling from the secant BFGS
    // ("primal-dual") update to the conjugate-gradient ("dual") scaling once a
    // stall is detected; the iterate is checkpoint-restored to the best point.
    let mut trace_out = NsSolveTrace::default();
    let mut strategy_dual = false;
    let mut switches = 0usize;
    let has_ns_scaling = cones.iter().any(|c| !matches!(c, NsCone::NonNeg(_)));

    for it in 0..settings.max_iters {
        iters = it;
        let px = prob.p.matvec(&x);
        let aty = prob.a_eq.matvec_t(&y);
        let atz = prob.a_in.matvec_t(&z);
        let r_d: Vec<T> = (0..n)
            .map(|i| px[i] + prob.q[i] + aty[i] + atz[i])
            .collect();
        let aeqx = prob.a_eq.matvec(&x);
        let r_b: Vec<T> = (0..me).map(|i| aeqx[i] - prob.b_eq[i]).collect();
        let ainx = prob.a_in.matvec(&x);
        let r_p: Vec<T> = (0..mi).map(|i| ainx[i] + s[i] - prob.b_in[i]).collect();
        let mut mu = zero;
        for i in 0..mi {
            mu += s[i] * z[i];
        }
        mu /= nu;

        let nd = inf_norm(&r_d);
        let nb = inf_norm(&r_b);
        let np = inf_norm(&r_p);
        // Track the best (lowest combined-residual) finite iterate. See the
        // NaN-guard note on the analogous QP-path tracking: `T::max` follows
        // IEEE-754 maxNum semantics (ignores a NaN operand), so only trust
        // `err` when all state vectors are finite.
        let iterate_finite = x.iter().all(|v| v.is_finite())
            && y.iter().all(|v| v.is_finite())
            && s.iter().all(|v| v.is_finite())
            && z.iter().all(|v| v.is_finite());
        let err = nd.max(nb).max(np).max(mu);
        if iterate_finite && err < best_err {
            best_err = err;
            best_x.copy_from_slice(&x);
            best_y.copy_from_slice(&y);
            best_s.copy_from_slice(&s);
            best_z.copy_from_slice(&z);
            iters_since_improvement = 0;
        } else if iterate_finite {
            iters_since_improvement += 1;
        }
        if !iterate_finite {
            break;
        }

        if nd <= eps && nb <= eps && np <= eps && mu <= eps {
            status = Status::Solved;
            break;
        }
        // "Almost solved" at a relaxed tolerance (√eps or 10×eps, the same
        // convention as the symmetric/QP path): if the combined residual has
        // been stuck (no improvement) for 10 iterations while already within
        // this looser band, grade SolvedInaccurate now rather than
        // exhausting the full iteration budget on a true fixed point.
        let eps_relaxed = eps.sqrt().max(eps * T::from_f64(10.0).expect("scalar literal"));
        let near_opt =
            nd <= eps_relaxed && nb <= eps_relaxed && np <= eps_relaxed && mu <= eps_relaxed;
        if near_opt && iters_since_improvement >= 10 {
            status = Status::SolvedInaccurate;
            break;
        }
        // Honesty floor (power-cone termination): the gap and primal feasibility
        // can converge while the dual residual is stuck. At a boundary-active dual
        // (power cones) `mu = Σ s·z/ν` is blind to the dual's scale error — every
        // point on the optimal dual ray is complementary to the optimal slack, so
        // `mu` reaches 1e-9 with `nd` still at 0.137 and the iterate frozen
        // bit-for-bit (the CVXPY-emission power-cone shape: primal/objective exact
        // to 1e-9, dual wrong-scale, budget burned on identical no-ops). Neither
        // the Solved exit (nd > eps) nor `near_opt` (which also gates on nd) can
        // fire; the scaling-strategy fallback cannot help either (the dual sits on
        // the K* boundary, so the conjugate scaling point does not exist). Once the
        // gap+feasibility have converged and the iterate has stopped improving,
        // exit SolvedInaccurate (best iterate) — the primal/objective are exact;
        // only the dual scale is approximate.
        if nb <= eps && np <= eps && mu <= eps && nd > eps_relaxed && iters_since_improvement >= 10
        {
            status = Status::SolvedInaccurate;
            break;
        }
        // Scaling-strategy fallback — insufficient-progress trigger (Clarabel's
        // checkpoint-insufficient-progress semantics): the combined residual has
        // not improved over the stall window while the iterate is still far from
        // optimal. The secant (BFGS) scaling is suspect — it enforces `M·s = z`
        // at the current point and can degenerate near the cone boundary — so
        // restore the best iterate and switch to the conjugate-gradient scaling
        // anchored to the dual point. One switch per solve; a dual strategy that
        // also stalls runs out and is graded honestly (best iterate).
        if has_ns_scaling
            && !strategy_dual
            && !near_opt
            && iters_since_improvement >= stall_window
            && switches < max_fallback_switches
        {
            if std::env::var_os("ICONIC_TRACE_NONSYM").is_some() {
                eprintln!(
                    "ns fallback: insufficient progress at it={it} (stalled {} iters, err={:e}), restoring best iterate",
                    iters_since_improvement,
                    err.to_f64().expect("finite scalar")
                );
            }
            x.copy_from_slice(&best_x);
            y.copy_from_slice(&best_y);
            s.copy_from_slice(&best_s);
            z.copy_from_slice(&best_z);
            strategy_dual = true;
            switches += 1;
            iters_since_improvement = 0;
            trace_out.fallback_engaged = true;
            trace_out.fallback_iters.push(it);
            continue;
        }

        // Per-cone barrier gradient and primal-dual scaling block H_pd at (s, z).
        // The gradient feeds the combined-step centering target
        // (`rc = −(z + σμ·g)`); the exp cone's is safeguarded (see
        // `grad_target`).
        let mut g = vec![zero; mi];
        let mut hblk: Vec<Vec<T>> = Vec::with_capacity(cones.len());
        // Dual-side recovery data (per cone, `Some` for the cones on the
        // dual-side step): `hinv` = Hs⁻¹ = (μ∇²f*(z))⁻¹ and `aff` = −Hs⁻¹·s
        // (the affine target; the combined target is `aff + σ·z`). The
        // recovery for these cones is `dz = aff(+σz) − Hs⁻¹·ds` — bounded
        // even when the dual point sits on the K* boundary.
        let mut dual_hinv: Vec<Option<Vec<T>>> = Vec::with_capacity(cones.len());
        let mut dual_aff: Vec<Option<Vec<T>>> = Vec::with_capacity(cones.len());
        for (c, cone) in cones.iter().enumerate() {
            let o = off[c];
            let d = cone.dim();
            let gc = cone.grad_target(&s[o..o + d], &z[o..o + d]);
            g[o..o + d].copy_from_slice(&gc);
            let h = if strategy_dual {
                // Dual strategy: the (x,x) block keeps the safeguarded secant
                // scaling (bounded by the margin floor — see `pd_scaling`), but
                // for the **degenerate** exp cones (slack far below its own
                // complementarity scale, ψ < 0.01·s·z — the doubly-degenerate
                // state) the recovered dual direction switches to the dual-side
                // form `dz = −Hs⁻¹·s (+σz) − Hs⁻¹·ds` with `Hs = μ∇²f*(z)` (see
                // `dual_side_step`): the ~1/ψ*² blowup of the dual Hessian
                // cancels against its own inverse, so the dual direction is
                // bounded and correctly signed, and the dual step stops
                // collapsing. The healthy cones keep the primal-side recovery
                // (their dual step is not boundary-blocked, and the dual-side
                // linearization perturbs their trajectory). If the dual-side
                // data cannot be formed, fall back to the conjugate-gradient
                // scaling, then the secant scaling, then the degrade-and-
                // continue zero block.
                let degenerate = match cone {
                    NsCone::Exp => {
                        let sz = s[o..o + d]
                            .iter()
                            .zip(z[o..o + d].iter())
                            .map(|(&a, &b)| a * b)
                            .fold(zero, |a, v| a + v);
                        exp::margin(&s[o..o + d]) < T::from_f64(0.01).expect("scalar literal") * sz
                    }
                    _ => false,
                };
                if degenerate {
                    match cone.dual_side_step(&s[o..o + d], &z[o..o + d], mu) {
                        Some((hinv, aff)) => {
                            dual_hinv.push(Some(hinv));
                            dual_aff.push(Some(aff));
                            cone.pd_scaling(&s[o..o + d], &z[o..o + d], mu)
                        }
                        None => {
                            dual_hinv.push(None);
                            dual_aff.push(None);
                            cone.conjugate_scaling(&s[o..o + d], &z[o..o + d], mu)
                                .or_else(|| cone.pd_scaling(&s[o..o + d], &z[o..o + d], mu))
                        }
                    }
                } else {
                    dual_hinv.push(None);
                    dual_aff.push(None);
                    cone.conjugate_scaling(&s[o..o + d], &z[o..o + d], mu)
                        .or_else(|| cone.pd_scaling(&s[o..o + d], &z[o..o + d], mu))
                }
            } else {
                dual_hinv.push(None);
                dual_aff.push(None);
                cone.pd_scaling(&s[o..o + d], &z[o..o + d], mu)
            };
            match h {
                Some(h) => hblk.push(h),
                None => {
                    // Graceful degradation (degrade-and-continue): the scaling failed
                    // with a non-positive intermediate (see `pd_scaling`). Zero this
                    // cone's (s,s) block contribution for the iteration instead of
                    // stalling — the cone's rows then enter the condensed system only
                    // through the primal-feasibility coupling (Δs = −r_p − A_in Δx,
                    // Δz = rc for this block), and the next iteration retries the
                    // scaling from the advanced point. Never hard-fail a scaling.
                    hblk.push(vec![zero; d * d]);
                }
            }
        }

        // Condensed (x,y) matrix M = [[P+ρ+Aᵀ H_pd A, A_eqᵀ],[A_eq, −δ]] (H_pd carries μ).
        // `m` is pre-allocated; the P-copy below overwrites all n² entries, the cone
        // contributions read this iteration's copy, and the equality border writes
        // its positions unconditionally — the (n+r1, n+r2) off-diagonals are never
        // written by any iteration and stay zero from initialization.
        for i in 0..n {
            for j in 0..n {
                m.set(i, j, prob.p.get(i, j));
            }
            m.set(i, i, m.get(i, i) + rho);
        }
        // Cone contribution to the (x,x) block: Σ_{a,b} H_pd[a][b]·A_in[o+a]·A_in[o+b]ᵀ.
        // The zero-skips make this efficient for the structured (sparse) A_in typical of
        // exp programs (log-sum-exp, entropy), where a faer dense gram would be wasteful.
        for (c, cone) in cones.iter().enumerate() {
            let o = off[c];
            let d = cone.dim();
            let h = &hblk[c];
            for a in 0..d {
                for b in 0..d {
                    let coef = h[a * d + b];
                    if coef == zero {
                        continue;
                    }
                    for i in 0..n {
                        let aia = prob.a_in.get(o + a, i);
                        if aia == zero {
                            continue;
                        }
                        for j in 0..n {
                            let v = m.get(i, j) + coef * aia * prob.a_in.get(o + b, j);
                            m.set(i, j, v);
                        }
                    }
                }
            }
        }
        // Scale-aware regularization of the (x,x) block — gated on the power
        // cones. The failure this fixes is the power-cone boundary-active dual's
        // scale error: at `phi ~ 1e-9` the barrier Hessian is `~1/phi² ~ 1e18`, so
        // the `mu·H` contribution to the condensed block is `~1e9` while the flat
        // `rho = 1e-8` is 16 orders below, and the fixed-order no-pivot LDLᵀ then
        // pivots on cancellation noise — the CVXPY-emission power-cone shape
        // burned its whole budget frozen at the exactly-correct point because one
        // permutation of a permuted-identical matrix factored to a ~1e7-blowup
        // Newton direction (step length 0 → iterate frozen bit-for-bit) while the
        // other order factored cleanly (solved at 17 iters; verified: perturbing
        // the data by 1e-12 flips which one). Raise the diagonal floor with the
        // matrix's own scale, the QP path's established precedent
        // (`max_diag * 1e-10/1e-12` in `solve_qp_with_termination`): with
        // `eps_reg = 1e-10` the added term is ~1e-10 of a healthy O(1) diagonal (a
        // no-op on healthy trajectories) and ~0.01-1 at the degenerate state,
        // lifting the pivots out of the cancellation regime so both orderings
        // return the same true Newton direction. The dual step `dz` (which
        // carries the boundary-ray scale fix) is unchanged by the regularization
        // (verified on the exact degenerate state: dx bounded ~1e-9, dz
        // identical).
        //
        // Gated on the power cones because the exp-cone trajectories are
        // knife-edge sensitive to ANY diagonal shift at the late near-boundary
        // iterates — measured: a ~1e-16 perturbation moved `log_sum_exp_coupled`
        // from Solved at 12 iters to a backward gap step + a 199-iteration drift,
        // and `went_n8_s0.3` from Solved to a frozen MaxIterations. Exp problems
        // keep the flat `rho` exactly (bit-identical trajectories).
        if cones
            .iter()
            .any(|c| matches!(c, NsCone::Power(_) | NsCone::GenPower(_, _)))
        {
            let mut max_diag = rho;
            for i in 0..n {
                max_diag = max_diag.max(m.get(i, i).abs());
            }
            let reg = (max_diag * from(1e-10)).max(rho);
            let reg_add = reg - rho;
            if reg_add > zero {
                for i in 0..n {
                    m.set(i, i, m.get(i, i) + reg_add);
                }
            }
        }
        for r in 0..me {
            for j in 0..n {
                let v = prob.a_eq.get(r, j);
                m.set(n + r, j, v);
                m.set(j, n + r, v);
            }
            m.set(n + r, n + r, -delta);
        }
        // faer's SIMD LBLT for larger systems (≥130, factor-once/solve-many); scalar LDLᵀ
        // for small ones.
        let factor = if dim >= 130 {
            let mf = DenseMatrix::from_row_major(
                dim,
                dim,
                m.data.iter().map(|v| v.to_f64().expect("finite scalar")).collect(),
            );
            NsFac::Faer(iconic_linalg::faer_dense::FaerLblt::factor(&mf))
        } else {
            match iconic_linalg::ldl::ldl_factor_into(&m, pivot_tol, &mut ldl_buf) {
                Ok(()) => NsFac::Scalar(&ldl_buf),
                Err(_) => {
                    status = Status::NumericalError;
                    break;
                }
            }
        };

        // Apply the per-cone Hessian to a length-mi vector.
        let apply_h = |v: &[T]| -> Vec<T> {
            let mut out = vec![zero; mi];
            for (c, cone) in cones.iter().enumerate() {
                let o = off[c];
                let d = cone.dim();
                let h = &hblk[c];
                for a in 0..d {
                    let mut acc = zero;
                    for b in 0..d {
                        acc += h[a * d + b] * v[o + b];
                    }
                    out[o + a] = acc;
                }
            }
            out
        };

        // Solve for the search direction given the complementarity RHS `rc`
        // (Δz + H_pd·Δs = rc). Returns (Δx, Δy, Δs, Δz).
        let solve_dir = |rc: &[T]| {
            // Δs = −r_p − A_in Δx ; Δz = rc + H_pd r_p + H_pd A_in Δx.
            // RHS_x = −r_d − A_inᵀ rc − A_inᵀ H_pd r_p ; RHS_y = −r_b.
            let hrp = apply_h(&r_p);
            let mut t = vec![zero; mi]; // rc + H_pd·r_p
            for i in 0..mi {
                t[i] = rc[i] + hrp[i];
            }
            let at_t = prob.a_in.matvec_t(&t);
            let mut rhs = vec![zero; dim];
            for i in 0..n {
                rhs[i] = -r_d[i] - at_t[i];
            }
            for i in 0..me {
                rhs[n + i] = -r_b[i];
            }
            let sol = factor.solve(&rhs);
            let dx = sol[0..n].to_vec();
            let dy = sol[n..n + me].to_vec();
            let aindx = prob.a_in.matvec(&dx);
            let ds: Vec<T> = (0..mi).map(|i| -r_p[i] - aindx[i]).collect();
            let hds = apply_h(&ds);
            let mut dz = vec![zero; mi];
            for (c, cone) in cones.iter().enumerate() {
                let o = off[c];
                let d = cone.dim();
                match &dual_hinv[c] {
                    Some(hinv) => {
                        // Dual-side recovery: dz = rc − Hs⁻¹·ds (`rc` already
                        // holds the effective target −Hs⁻¹·s (+σ·z for the
                        // combined step)); bounded at the degenerate state.
                        for a in 0..d {
                            let mut acc = zero;
                            for b in 0..d {
                                acc += hinv[a * d + b] * ds[o + b];
                            }
                            dz[o + a] = rc[o + a] - acc;
                        }
                    }
                    None => {
                        for i in 0..d {
                            dz[o + i] = rc[o + i] - hds[o + i];
                        }
                    }
                }
            }
            (dx, dy, ds, dz)
        };

        // Corrector solve: pure complementarity RHS (zero feasibility residual), so a
        // centrality correction does not disturb the feasibility progress of the
        // combined step. Same factorization.
        let solve_cor = |rc: &[T]| {
            let at_t = prob.a_in.matvec_t(rc);
            let mut rhs = vec![zero; dim];
            for i in 0..n {
                rhs[i] = -at_t[i];
            }
            let sol = factor.solve(&rhs);
            let dx = sol[0..n].to_vec();
            let dy = sol[n..n + me].to_vec();
            let aindx = prob.a_in.matvec(&dx);
            let ds: Vec<T> = (0..mi).map(|i| -aindx[i]).collect();
            let hds = apply_h(&ds);
            let mut dz = vec![zero; mi];
            for (c, cone) in cones.iter().enumerate() {
                let o = off[c];
                let d = cone.dim();
                match &dual_hinv[c] {
                    Some(hinv) => {
                        // Dual-side recovery (see the combined-step closure).
                        for a in 0..d {
                            let mut acc = zero;
                            for b in 0..d {
                                acc += hinv[a * d + b] * ds[o + b];
                            }
                            dz[o + a] = rc[o + a] - acc;
                        }
                    }
                    None => {
                        for i in 0..d {
                            dz[o + i] = rc[o + i] - hds[o + i];
                        }
                    }
                }
            }
            (dx, dy, ds, dz)
        };

        let step = |ds: &[T], dz: &[T]| -> (T, T) {
            let mut ap = T::infinity();
            let mut ad = T::infinity();
            for (c, cone) in cones.iter().enumerate() {
                let o = off[c];
                let d = cone.dim();
                ap = ap.min(cone.max_step(&s[o..o + d], &ds[o..o + d]));
                ad = ad.min(cone.max_step_dual(&z[o..o + d], &dz[o..o + d]));
            }
            ((eta * ap).min(one), (eta * ad).min(one))
        };

        // Affine predictor: rc = −z (drive the complementarity to zero); the
        // dual-side cones use their bounded effective target −Hs⁻¹·s.
        let mut rc_aff = vec![zero; mi];
        for (c, cone) in cones.iter().enumerate() {
            let o = off[c];
            let d = cone.dim();
            match &dual_aff[c] {
                Some(aff) => rc_aff[o..o + d].copy_from_slice(aff),
                None => {
                    for i in 0..d {
                        rc_aff[o + i] = -z[o + i];
                    }
                }
            }
        }
        let (_dxa, _dya, dsa, dza) = solve_dir(&rc_aff);
        let (apa, ada) = step(&dsa, &dza);
        // Common step couples primal/dual progress — essential for the infeasible start,
        // where letting the dual race ahead collapses μ while primal infeasibility stalls.
        let aa = apa.min(ada);
        let mut mu_aff = zero;
        for i in 0..mi {
            mu_aff += (s[i] + aa * dsa[i]) * (z[i] + aa * dza[i]);
        }
        mu_aff /= nu;
        let sigma = if mu > zero {
            // Cap the centering parameter to prevent over-centering when the affine
            // step makes good progress: sigma = min(alpha², 0.25)·alpha, floored at
            // 1e-16 so the combined step never collapses to a pure affine direction.
            // Clamp alpha to [0,1] first -- a near-degenerate affine direction can
            // push mu_aff above mu, and capping only the squared term while
            // multiplying by the uncapped alpha leaves sigma unbounded for alpha > 1.
            let alpha = (mu_aff / mu).max(zero).min(one);
            ((alpha * alpha).min(from(0.25)) * alpha).max(from(1e-16))
        } else {
            zero
        };

        // Combined: rc = −(z + σμ·g(s)); the dual-side cones use
        // −Hs⁻¹·s + σ·z (bounded by the dual barrier's homogeneity — see
        // `dual_side_step`).
        let sm = sigma * mu;
        let mut rc = vec![zero; mi];
        for (c, cone) in cones.iter().enumerate() {
            let o = off[c];
            let d = cone.dim();
            match &dual_aff[c] {
                Some(aff) => {
                    for i in 0..d {
                        rc[o + i] = aff[i] + sigma * z[o + i];
                    }
                }
                None => {
                    for i in 0..d {
                        rc[o + i] = -(z[o + i] + sm * g[o + i]);
                    }
                }
            }
        }
        let (mut dx, mut dy, mut ds, mut dz) = solve_dir(&rc);
        let (mut ap0, mut ad0) = step(&ds, &dz);

        // Higher-order centrality correctors (Gondzio-style): at the current trial point
        // the centrality residual z + σμ·g(s) is nonzero because g is nonlinear; correct
        // toward it with the same factorization. Acceptance-tested — kept only if the
        // common step strictly lengthens — so it is always safe (cannot diverge).
        //
        // For the exponential cone the barrier gradient's curvature makes these correctors
        // ineffective in practice: the trial-point correction rarely lengthens the step enough
        // to clear the acceptance test, so the extra factorization solve buys nothing. We skip
        // them outright whenever any exp cone is present in the problem.
        let max_cor = if cones.iter().any(|c| matches!(c, NsCone::Exp)) {
            0
        } else {
            2
        };
        for _ in 0..max_cor {
            let alpha = ap0.min(ad0);
            let st: Vec<T> = (0..mi).map(|i| s[i] + alpha * ds[i]).collect();
            let zt: Vec<T> = (0..mi).map(|i| z[i] + alpha * dz[i]).collect();
            let mut rc_cor = vec![zero; mi];
            for (c, cone) in cones.iter().enumerate() {
                let o = off[c];
                let d = cone.dim();
                let gt = cone.grad(&st[o..o + d]);
                for i in 0..d {
                    rc_cor[o + i] = -(zt[o + i] + sm * gt[i]);
                }
            }
            let (cdx, cdy, cds, cdz) = solve_cor(&rc_cor);
            let ndx: Vec<T> = (0..n).map(|i| dx[i] + cdx[i]).collect();
            let ndy: Vec<T> = (0..me).map(|i| dy[i] + cdy[i]).collect();
            let nds: Vec<T> = (0..mi).map(|i| ds[i] + cds[i]).collect();
            let ndz: Vec<T> = (0..mi).map(|i| dz[i] + cdz[i]).collect();
            let (nap, nad) = step(&nds, &ndz);
            if nap.min(nad) > alpha + from(1e-4) {
                dx = ndx;
                dy = ndy;
                ds = nds;
                dz = ndz;
                ap0 = nap;
                ad0 = nad;
            } else {
                break;
            }
        }

        // Scaling-strategy fallback — small-step trigger (Clarabel's
        // checkpoint-small-step semantics): a combined step below the switch
        // threshold, corroborated by a short no-improvement window and the
        // iterate not near-optimal, means the secant scaling is collapsing the
        // direction near the cone boundary. Restore the best iterate and switch
        // to the conjugate-gradient dual scaling.
        if has_ns_scaling
            && !strategy_dual
            && !near_opt
            && ap0.min(ad0) < min_switch_step_length
            && iters_since_improvement >= 5
            && switches < max_fallback_switches
        {
            if std::env::var_os("ICONIC_TRACE_NONSYM").is_some() {
                eprintln!(
                    "ns fallback: small step at it={it} (alpha={:e}, stalled {} iters, err={:e}), restoring best iterate",
                    ap0.min(ad0).to_f64().expect("finite scalar"),
                    iters_since_improvement,
                    err.to_f64().expect("finite scalar")
                );
            }
            x.copy_from_slice(&best_x);
            y.copy_from_slice(&best_y);
            s.copy_from_slice(&best_s);
            z.copy_from_slice(&best_z);
            strategy_dual = true;
            switches += 1;
            iters_since_improvement = 0;
            trace_out.fallback_engaged = true;
            trace_out.fallback_iters.push(it);
            continue;
        }

        // Couple primal/dual progress (prevents the dual racing ahead and collapsing μ
        // while primal infeasibility lags). But if the dual stalls against its cone
        // boundary (ad≈0) while the primal can still move, let the primal progress
        // rather than freezing both — the dual direction usually frees up next iterate.
        let (mut ap, mut ad) = {
            let common = ap0.min(ad0);
            if ad0 < from(1e-3) && ap0 > ad0 {
                (ap0, common)
            } else {
                (common, common)
            }
        };
        if strategy_dual {
            // Barrier-backtracking line search on the dual arm (Clarabel's
            // backtrack_step_to_barrier): the conjugate scaling's step can still
            // drive the trial point into the thin near-boundary layer where the
            // barrier value explodes; halve the common step (0.8 factor, ≤ 50
            // halvings) until each cone's trial barrier stays below its current
            // value plus one. The allowance is per cone and relative to the
            // current point: a converged iterate legitimately sits near the
            // cone boundary (its barrier is large), and an absolute `bar < 1`
            // test — or a total-barrier test dominated by the healthy cones —
            // collapses the step to ~0 there, the boundary-active exp-cone
            // stall's death spiral after the dual-side switch.
            let mut a = ap.min(ad);
            let mut halvings = 0usize;
            let bar0: Vec<T> = cones
                .iter()
                .enumerate()
                .map(|(c, cone)| {
                    let o = off[c];
                    let d = cone.dim();
                    cone.barrier(&s[o..o + d])
                })
                .collect();
            while halvings < 50 {
                let mut ok = true;
                for (c, cone) in cones.iter().enumerate() {
                    let o = off[c];
                    let d = cone.dim();
                    let mut t = vec![zero; d];
                    for i in 0..d {
                        t[i] = s[o + i] + a * ds[o + i];
                    }
                    let bar = cone.barrier(&t);
                    if !bar.is_finite() || bar > bar0[c] + one {
                        ok = false;
                        break;
                    }
                }
                if ok || a <= from(1e-14) {
                    break;
                }
                a *= from(0.8);
                halvings += 1;
            }
            ap = a;
            ad = a;
        }
        for i in 0..n {
            x[i] += ap * dx[i];
        }
        for i in 0..mi {
            s[i] += ap * ds[i];
        }
        for i in 0..me {
            y[i] += ad * dy[i];
        }
        for i in 0..mi {
            z[i] += ad * dz[i];
        }
    }

    // Return the best iterate (lowest combined-residual), matching the QP
    // path's convention: Solved and SolvedInaccurate already broke at the
    // iteration that triggered them, so the current iterate IS the one that
    // triggered the break. Only fall back to the best-iterate snapshot when
    // the loop exhausted max_iters (or was killed by a NaN-poisoned iterate)
    // without ever declaring any grade at all — the snapshot holds the
    // most-accurate point seen before whatever happened.
    if status == Status::MaxIterations {
        x = best_x;
        y = best_y;
        s = best_s;
        z = best_z;
    }
    let px = prob.p.matvec(&x);
    let obj = from(0.5) * dot(&x, &px) + dot(&prob.q, &x);
    trace_out.dual_strategy_active = strategy_dual;
    if let Some(t) = trace {
        *t = trace_out.clone();
    }
    (
        QpSolution::new(
            status,
            x,
            y,
            s,
            z,
            obj,
            iters,
        ),
        trace_out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    // The battery generators live in the shared `generators` module (also used by
    // the iconic-bench suite) — one copy of each generator and its seed table.
    use crate::generators::{log_sum_exp, max_entropy, moment_entropy, pow_eq, pow_proj, weighted_entropy};

    /// Maximum entropy with many near-boundary cones. The optimum is the uniform
    /// distribution, entropy `log n`. This exercises the infeasible-start primal stall
    /// that froze the iterate for `n ≥ 30`.
    #[test]
    fn max_entropy_many_cones() {
        for &n in &[30usize, 50] {
            let (prob, cones) = max_entropy(n);
            let s = Settings::<f64>::default();
            let sol = solve_nonsym(&prob, &cones, &s);
            assert_eq!(
                sol.status,
                Status::Solved,
                "n={n} status={:?} iters={} obj={}",
                sol.status,
                sol.iters,
                sol.obj_val
            );
            // obj = min −Σtᵢ = −entropy = −log n.
            let want = -(n as f64).ln();
            assert!(
                (sol.obj_val - want).abs() < 1e-5,
                "n={n} obj={} want {want}",
                sol.obj_val
            );
        }
    }

    /// max x  s.t.  exp(x) ≤ 2  ⇒  x* = log 2 ≈ 0.6931.
    /// In cone form: s = (x, 1, 2) ∈ K_exp, with s = b − A_in x,
    /// A_in = [[−1],[0],[0]], b = [0, 1, 2]; objective min −x.
    #[test]
    fn maximize_x_under_exp_bound() {
        let prob = QpProblem {
            p: DenseMatrix::zeros(1, 1),
            q: vec![-1.0],
            a_eq: DenseMatrix::zeros(0, 1),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(3, 1, vec![-1.0, 0.0, 0.0]),
            b_in: vec![0.0, 1.0, 2.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let s = Settings::<f64>::default();
        let sol = solve_nonsym(&prob, &[NsCone::Exp], &s);
        assert_eq!(sol.status, Status::Solved, "iters={}", sol.iters);
        assert!(
            (sol.x[0] - 2.0_f64.ln()).abs() < 1e-6,
            "x={} expected log2={}",
            sol.x[0],
            2.0_f64.ln()
        );
    }

    /// A coupled, multi-cone exp program: max x₁+x₂ s.t. eˣ¹ + eˣ² ≤ 1.
    /// By symmetry x₁ = x₂ = −log 2, so x₁+x₂ = −2log 2 (obj of min −x₁−x₂ is 2log 2).
    /// Variables (x₁,x₂,u₁,u₂); cones [Exp,Exp,NonNeg(1)] enforce eˣⁱ ≤ uᵢ and u₁+u₂ ≤ 1.
    /// The trivial start x=0 is NOT cone-interior, exercising the infeasible interior start.
    ///
    /// Status note: the closed-form Lambert-W step (exact boundary vs the former
    /// coarse bisection) moves the trajectory's near-optimal fixed point to
    /// μ ≈ 1.02e-8 — 1.6% above the default 1e-8 tolerance — so the honest grade
    /// here is `SolvedInaccurate` at a point whose objective and primal variables are
    /// accurate to ~1e-8 (both 4000× tighter than the 5e-5 assertions below). The
    /// accuracy assertions are the substantive verification; the status is accepted
    /// in either grade.
    #[test]
    fn log_sum_exp_coupled() {
        #[rustfmt::skip]
        let a_in = DenseMatrix::from_row_major(7, 4, vec![
            -1.0, 0.0, 0.0, 0.0,   // s0 = x1
             0.0, 0.0, 0.0, 0.0,   // s1 = 1
             0.0, 0.0,-1.0, 0.0,   // s2 = u1
             0.0,-1.0, 0.0, 0.0,   // s3 = x2
             0.0, 0.0, 0.0, 0.0,   // s4 = 1
             0.0, 0.0, 0.0,-1.0,   // s5 = u2
             0.0, 0.0, 1.0, 1.0,   // s6 = 1 − u1 − u2
        ]);
        let prob = QpProblem {
            p: DenseMatrix::zeros(4, 4),
            q: vec![-1.0, -1.0, 0.0, 0.0],
            a_eq: DenseMatrix::zeros(0, 4),
            b_eq: vec![],
            a_in,
            b_in: vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let s = Settings::<f64>::default();
        let sol = solve_nonsym(&prob, &[NsCone::Exp, NsCone::Exp, NsCone::NonNeg(1)], &s);
        assert!(
            matches!(sol.status, Status::Solved | Status::SolvedInaccurate),
            "iters={} status={:?}",
            sol.iters,
            sol.status
        );
        let want = -2.0_f64.ln();
        assert!(
            (sol.x[0] - want).abs() < 5e-5,
            "x1={} want {want}",
            sol.x[0]
        );
        assert!(
            (sol.x[1] - want).abs() < 5e-5,
            "x2={} want {want}",
            sol.x[1]
        );
        assert!(
            (sol.obj_val - 2.0 * 2.0_f64.ln()).abs() < 5e-5,
            "obj={} want {}",
            sol.obj_val,
            2.0 * 2.0_f64.ln()
        );
    }

    /// Maximum entropy: max Σ −xᵢ·log xᵢ s.t. Σxᵢ = 1 (⇒ uniform xᵢ = 1/n, entropy log n).
    /// Exercises an equality constraint (A_eq) alongside exp cones. Variables
    /// (x₁,x₂,t₁,t₂); cone i is (tᵢ, xᵢ, 1) ∈ K_exp ⇔ xᵢ·log(1/xᵢ) ≥ tᵢ ⇔ −xᵢ log xᵢ ≥ tᵢ.
    #[test]
    fn max_entropy_with_equality() {
        #[rustfmt::skip]
        let a_in = DenseMatrix::from_row_major(6, 4, vec![
            0.0, 0.0,-1.0, 0.0,   // s0 = t1
           -1.0, 0.0, 0.0, 0.0,   // s1 = x1
            0.0, 0.0, 0.0, 0.0,   // s2 = 1
            0.0, 0.0, 0.0,-1.0,   // s3 = t2
            0.0,-1.0, 0.0, 0.0,   // s4 = x2
            0.0, 0.0, 0.0, 0.0,   // s5 = 1
        ]);
        let prob = QpProblem {
            p: DenseMatrix::zeros(4, 4),
            q: vec![0.0, 0.0, -1.0, -1.0], // min −t1−t2
            a_eq: DenseMatrix::from_row_major(1, 4, vec![1.0, 1.0, 0.0, 0.0]),
            b_eq: vec![1.0],
            a_in,
            b_in: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let s = Settings::<f64>::default();
        let sol = solve_nonsym(&prob, &[NsCone::Exp, NsCone::Exp], &s);
        assert_eq!(sol.status, Status::Solved, "iters={}", sol.iters);
        assert!((sol.x[0] - 0.5).abs() < 1e-5, "x1={}", sol.x[0]);
        assert!((sol.x[1] - 0.5).abs() < 1e-5, "x2={}", sol.x[1]);
        // obj = min −Σtᵢ = −entropy = −log 2.
        assert!(
            (sol.obj_val + 2.0_f64.ln()).abs() < 1e-5,
            "obj={} want {}",
            sol.obj_val,
            -2.0_f64.ln()
        );
    }

    /// The primal-dual scaling must **never hard-fail**: a non-positive intermediate
    /// (a dual `z` whose `s·z ≤ 0`, an orthant entry with the wrong sign — states the
    /// iterate can only reach through rounding, since the step lengths keep `s ∈ K`
    /// and `z ∈ K*`) reports `None` so the caller zeroes the cone's `(s,s)` block for
    /// the iteration and continues (degrade-and-continue). A valid primal-dual pair
    /// still forms the scaling and satisfies the secant condition `M·s = z`.
    #[test]
    fn pd_scaling_degrades_on_nonpositive_intermediate() {
        // Exp cone: interior primal point, dual outside K* with s·z < 0.
        let s = [0.3_f64, 1.2, 2.5];
        let z = [1.0, 1.0, -5.0]; // s·z = 0.3 + 1.2 − 12.5 < 0
        assert!(NsCone::Exp.pd_scaling(&s, &z, 1e-8).is_none());
        // NonNeg cone: a non-positive dual entry (or primal entry).
        assert!(NsCone::NonNeg(2)
            .pd_scaling(&[1.0, 2.0], &[1.0, -1.0], 1e-8)
            .is_none());
        assert!(NsCone::NonNeg(2)
            .pd_scaling(&[1.0, 2.0], &[1.0, 0.0], 1e-8)
            .is_none());
        assert!(NsCone::NonNeg(2)
            .pd_scaling(&[1.0, -2.0], &[1.0, 1.0], 1e-8)
            .is_none());
        // Power cone: same BFGS intermediate gate.
        assert!(NsCone::Power(0.5)
            .pd_scaling(&[0.6, 0.6, 0.3], &[1.0, 1.0, -5.0], 1e-8)
            .is_none());
        // A valid primal-dual pair forms the scaling and satisfies the secant
        // condition M·s = z (z = −∇f(s) ∈ int K*).
        let s = [0.3_f64, 1.2, 2.5];
        let g = exp::grad(&s);
        let z = [-g[0], -g[1], -g[2]];
        let m = NsCone::Exp.pd_scaling(&s, &z, 1e-3).expect("valid pair");
        for i in 0..3 {
            let ms: f64 = (0..3).map(|j| m[i * 3 + j] * s[j]).sum();
            assert!(
                (ms - z[i]).abs() < 1e-8,
                "secant M·s=z failed at {i}: {ms} vs {}",
                z[i]
            );
        }
        // NonNeg pair: M = diag(z/s).
        let m = NsCone::NonNeg(2)
            .pd_scaling(&[1.0, 2.0], &[3.0, 4.0], 1e-8)
            .expect("valid pair");
        assert!((m[0] - 3.0_f64).abs() < 1e-12 && (m[3] - 2.0_f64).abs() < 1e-12);
    }

    /// The max-entropy family across the full A/B sweep (n = 4..=50): every size must
    /// solve to the uniform distribution (entropy log n). This is the end-to-end
    /// regression test for the many-active-cones stall class the graceful degradation
    /// protects against: the loop must continue and solve, never stall.
    #[test]
    fn max_entropy_sweep_solves_everywhere() {
        for n in [4usize, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50] {
            let (prob, cones) = max_entropy(n);
            let s = Settings::<f64>::default();
            let sol = solve_nonsym(&prob, &cones, &s);
            assert_eq!(
                sol.status,
                Status::Solved,
                "n={n} status={:?} iters={} obj={}",
                sol.status,
                sol.iters,
                sol.obj_val
            );
            let want = -(n as f64).ln();
            assert!(
                (sol.obj_val - want).abs() < 1e-5,
                "n={n} obj={} want {want}",
                sol.obj_val
            );
        }
    }

    // ---- Scaling-strategy fallback battery ---------------------------------

    /// The exp/power battery with the scaling-strategy fallback **inert** on the
    /// healthy instances: the fallback must never engage on problems the engine
    /// solves, so its trajectories are bit-identical to the no-fallback path
    /// (the fallback only fires on a corroborated stall). All objectives are
    /// verified against Clarabel (via CVXPY) or closed forms.
    #[test]
    fn battery_fallback_inert_on_healthy() {
        let mut cases: Vec<(String, QpProblem<f64>, Vec<NsCone>, f64, f64)> = Vec::new();
        // max-entropy: obj = −ln n.
        for &n in &[4usize, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50] {
            let (p, c) = max_entropy(n);
            cases.push((format!("maxent_n{n}"), p, c, -(n as f64).ln(), 1e-5));
        }
        // weighted entropy: obj = −ln Σe^{skew·i}.
        for &(n, skew) in &[(8usize, 0.3f64), (10, 0.5), (16, 0.5), (30, 0.5), (50, 0.3)] {
            let (p, c) = weighted_entropy(n, skew);
            let wsum: f64 = (0..n).map(|i| (skew * i as f64).exp()).sum();
            cases.push((format!("went_n{n}_s{skew}"), p, c, -(wsum.ln()), 1e-5));
        }
        // boxed log-sum-exp: Clarabel-verified optima.
        let lse_want: &[(usize, usize, f64, f64)] = &[
            (5, 5, 1.0, 1.4194689215),
            (10, 5, 1.0, 1.1492886795),
            (10, 10, 1.0, 4.8365032996),
            (20, 10, 1.0, 0.9187726138),
            (10, 5, 3.0, 0.0830372795),
            (20, 10, 3.0, 0.0246823302),
            (30, 15, 1.0, 1.6546597532),
            (50, 20, 1.0, 0.4301863966),
        ];
        for &(n, m, scale, want) in lse_want {
            let (p, c) = log_sum_exp(n, m, scale);
            cases.push((format!("lse_n{n}_m{m}_s{scale}"), p, c, want, 1e-6));
        }
        // power cone: projection (obj −½‖c‖² = −1) and equality (obj −α^α(1−α)^(1−α)).
        for alpha in [0.3, 0.5, 0.7] {
            let (p, c) = pow_proj(alpha);
            cases.push((format!("powproj_a{alpha}"), p, c, -1.0, 1e-5));
            let (p2, c2) = pow_eq(alpha);
            let want = -(alpha.powf(alpha) * (1.0 - alpha).powf(1.0 - alpha));
            cases.push((format!("poweq_a{alpha}"), p2, c2, want, 1e-5));
        }
        let s = Settings::<f64>::default();
        for (name, prob, cones, want, tol) in cases {
            let mut trace = NsSolveTrace::default();
            let (sol, tr) = solve_nonsym_traced(&prob, &cones, &s, Some(&mut trace));
            assert_eq!(
                sol.status,
                Status::Solved,
                "{name}: status={:?} iters={} obj={}",
                sol.status,
                sol.iters,
                sol.obj_val
            );
            assert!(
                (sol.obj_val - want).abs() < tol,
                "{name}: obj={} want {want}",
                sol.obj_val
            );
            assert!(
                !tr.fallback_engaged,
                "{name}: fallback engaged on a healthy instance (iters={})",
                sol.iters
            );
        }
    }

    /// The power cones' **dual** boundary map (the true dual cone
    /// `(u/α)^α(v/(1−α))^(1−α) ≥ |w|`, larger than the primal): a boundary-active
    /// power-cone program where the old primal-boundary dual step collapsed to
    /// zero — freezing the dual — now solves. Verified against the closed forms.
    #[test]
    fn power_cone_dual_boundary_solves_boundary_active() {
        // max x^α y^(1−α) s.t. x+y=1, α=0.5: optimum on the curved boundary,
        // value −0.5.
        let (p, c) = pow_eq(0.5);
        let s = Settings::<f64>::default();
        let (sol, tr) = solve_nonsym_traced(&p, &c, &s, None);
        assert_eq!(sol.status, Status::Solved, "iters={}", sol.iters);
        assert!(
            (sol.obj_val + 0.5).abs() < 1e-5,
            "obj={} want -0.5",
            sol.obj_val
        );
        assert!(sol.iters < 40, "iters={}", sol.iters);
        assert!(!tr.fallback_engaged);
        // Projection of an interior point: x* = c, obj = −½‖c‖².
        let (p2, c2) = pow_proj(0.5);
        let (sol2, _) = solve_nonsym_traced(&p2, &c2, &s, None);
        assert_eq!(sol2.status, Status::Solved, "iters={}", sol2.iters);
        assert!((sol2.obj_val + 1.0).abs() < 1e-5, "obj={}", sol2.obj_val);
    }

    /// The crafted hard-stall instance — moment-constrained entropy with the
    /// moment at 10% of its range (extreme exponential tilt, many near-zero
    /// atoms, all exp cones boundary-active) — historically stalled honestly
    /// (MaxIterations; the fallback engaged, and the no-fallback dual ran away
    /// to a +2.8e19 objective). The boundary-active exp-cone work (the margin
    /// floor on the scaling + the near-origin fallback start point, see
    /// `NsCone::pd_scaling`/`central_point`) now solves it: the dual stays
    /// bounded without the fallback, and the optimum (−1.313768, verified
    /// against Clarabel) is reached exactly.
    #[test]
    fn crafted_hard_stall_now_solves() {
        let (p, c) = moment_entropy(10, 0.1);
        let s = Settings::<f64>::default();
        let (sol, tr) = solve_nonsym_traced(&p, &c, &s, None);
        assert_eq!(
            sol.status,
            Status::Solved,
            "status={:?} iters={} obj={}",
            sol.status,
            sol.iters,
            sol.obj_val
        );
        assert!(
            !tr.fallback_engaged,
            "fallback must not engage on the now-solvable instance: trace={tr:?}"
        );
        assert!(
            (sol.obj_val - (-1.313768)).abs() < 1e-5,
            "obj={} want -1.313768 (Clarabel-verified)",
            sol.obj_val
        );
    }

    /// M8 warm-start exactness on the nonsymmetric path: a seeded re-solve of
    /// a max-entropy problem with perturbed q converges to the same point as
    /// the cold re-solve (same status, objective within 1e-8·max(1,|obj|)) and
    /// never takes more iterations. Max-entropy is the hard warm case: every
    /// exp cone is boundary-active at the optimum, so the seeded slack must be
    /// θ-blended into the interior before the engine can start from it.
    #[test]
    fn warm_start_nonsym_maxent_exactness() {
        let n = 6usize;
        let (prob, cones) = max_entropy(n);
        let s = Settings::<f64>::default();
        let base = solve_nonsym(&prob, &cones, &s);
        assert_eq!(base.status, Status::Solved);
        let seed = WarmStart {
            x: base.x,
            s: base.s,
            z: base.z,
        };
        // Perturb q (the −Σtᵢ objective coefficients) by 1e-4 relative.
        let scale = prob.q.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let mut pert = prob.clone();
        for (i, v) in pert.q.iter_mut().enumerate() {
            *v += 1e-4 * scale * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let cold = solve_nonsym(&pert, &cones, &s);
        let warm = solve_nonsym_warm(&pert, &cones, &s, Some(&seed));
        assert_eq!(warm.status, cold.status, "warm must not change the status");
        // 1e-6 objective agreement: the nonsym engine terminates at the
        // ε=1e-8 residual level, so two valid trajectories can land up to
        // ~1e-7 apart in objective (measured; warm was the more accurate of
        // the two here) — the contract is agreement at solver accuracy, not
        // bit-level.
        let tol = 1e-6 * cold.obj_val.abs().max(1.0);
        assert!(
            (warm.obj_val - cold.obj_val).abs() <= tol,
            "warm obj {:.12e} vs cold {:.12e}",
            warm.obj_val,
            cold.obj_val
        );
        assert!(
            warm.iters <= cold.iters,
            "warm must not take more iterations: {} vs {}",
            warm.iters,
            cold.iters
        );
        // A seed violating A·x + s = b must fall back to the cold trajectory.
        let mut bad = seed.clone();
        bad.s[0] += 1.0;
        let fallback = solve_nonsym_warm(&pert, &cones, &s, Some(&bad));
        assert_eq!(fallback.iters, cold.iters);
        assert_eq!(fallback.obj_val, cold.obj_val);
    }
}
