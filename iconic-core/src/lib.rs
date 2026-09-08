//! `iconic-core` — problem types, cones, settings, status, and solution containers
//! shared by the ICONIC solver engines.
//!
//! This crate holds no I/O, no Python, and no concrete linear algebra — only the
//! standard-form data model and the trait surface the engines program against.

use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::{Debug, Display};

pub mod rng;

/// Scalar field the solver is generic over.
///
/// `f64` is the default and the only fully-validated type; `f32` is supported for
/// embedded / memory-constrained use (with looser default tolerances).
pub trait Scalar: Float + NumAssign + FromPrimitive + Debug + Display + 'static {}
impl<T> Scalar for T where T: Float + NumAssign + FromPrimitive + Debug + Display + 'static {}

/// Terminal status of a solve, reported in original (unscaled) units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Optimality certificate met at the requested tolerance.
    Solved,
    /// Optimality met only at reduced (10×) tolerance after a limit was hit.
    SolvedInaccurate,
    /// A Farkas primal-infeasibility certificate was found.
    PrimalInfeasible,
    /// An unbounded (dual-infeasibility) direction was found.
    DualInfeasible,
    /// Iteration limit reached before convergence.
    MaxIterations,
    /// Wall-clock limit reached before convergence.
    TimeLimit,
    /// Factorization / refinement could not produce a usable step.
    NumericalError,
    /// No solve has been run yet.
    Unsolved,
}

impl Status {
    /// Whether this status carries a usable primal point (`Solved` or
    /// `SolvedInaccurate`) — the predicate every caller's accept/retry
    /// logic tests.
    pub fn has_solution(self) -> bool {
        matches!(self, Status::Solved | Status::SolvedInaccurate)
    }
}

/// A single cone in the canonical Cartesian product `K`.
///
/// The product is ordered canonically (matching CVXPY's cone convention): `Zero`,
/// `NonNegative`, `SecondOrder`, `PsdTriangle`, `Exponential`, `Power`, `GenPower`.
#[derive(Clone, Debug, PartialEq)]
pub enum Cone {
    /// Equality block `s = 0` of the given dimension.
    Zero(usize),
    /// Nonnegative orthant of the given dimension.
    NonNegative(usize),
    /// Second-order (Lorentz) cone of the given total dimension.
    SecondOrder(usize),
    /// PSD cone over the upper-triangular vectorization of an `n × n` matrix.
    PsdTriangle(usize),
    /// 3-dimensional exponential cone.
    Exponential,
    /// 3-dimensional power cone with exponent `α ∈ (0, 1)`.
    Power(f64),
    /// Generalized power cone with exponents `α` and a tail dimension.
    GenPower(Vec<f64>, usize),
}

impl Cone {
    /// Number of scalar entries this cone occupies in `s`/`z`.
    pub fn dim(&self) -> usize {
        match self {
            Cone::Zero(n) | Cone::NonNegative(n) | Cone::SecondOrder(n) | Cone::PsdTriangle(n) => {
                *n
            }
            Cone::Exponential | Cone::Power(_) => 3,
            Cone::GenPower(alpha, tail) => alpha.len() + tail,
        }
    }
}

/// Solver settings controlling tolerances, iteration limits, presolve, and
/// algorithm selection.  All tolerances apply in the original (unscaled) problem
/// space so tightening `eps_abs` predictably improves solution accuracy
/// regardless of whether presolve equilibration is enabled.
///
/// # Tolerances (absolute + relative)
///
/// ICONIC declares convergence when **all three** criteria are satisfied:
///
/// 1. **Primal** `||Ax + s - b||_inf <= eps_abs + eps_rel*max(||Ax||,||s||,||b||)`
/// 2. **Dual** `||Px + q + A^T z||_inf <= eps_abs + eps_rel*max(||Px||,||q||,||A^T z||)`
/// 3. **Gap** `|x^T P x + q^T x + b^T z| <= eps_gap*(1 + |primal| + |dual|)`
///
/// # Presolve
///
/// When `presolve` is enabled (default), ICONIC runs Ruiz equilibration + cost
/// scaling, then a sequence of reduction passes: fixed-variable elimination,
/// doubleton-equality substitution, empty-column removal, null/dominated/duplicate
/// row removal, redundant-inequality detection, and linearly-dependent equality
/// removal.  Each reduction is recorded for postsolve so the solution is always
/// returned in the original variable space.
#[derive(Clone, Debug)]
pub struct Settings<T: Scalar> {
    /// Absolute tolerance on primal/dual residuals and duality gap.
    /// Lower = more accurate; higher = faster termination.
    /// **Default:** `1e-8`.  **Range:** `[1e-12, 1e-2]`.
    pub eps_abs: T,

    /// Relative tolerance on primal/dual residuals.
    /// Multiplied by the magnitude of the largest term in each residual.
    /// **Default:** `1e-8`.  **Range:** `[1e-12, 1e-2]`.
    pub eps_rel: T,

    /// Duality-gap tolerance (absolute+relative combined).
    /// For pure LPs the gap is `|c^T x + b^T z|`; for QPs it includes `x^T P x`.
    /// **Default:** `1e-8`.  **Range:** `[1e-12, 1e-2]`.
    pub eps_gap: T,

    /// Maximum number of interior-point iterations.
    /// The solver terminates with `MaxIterations` (and an "almost solved" status
    /// at relaxed tolerances) when this limit is reached.
    /// **Default:** `200`.  **Range:** `[1, 10000]`.
    pub max_iters: usize,

    /// Run presolve (Ruiz equilibration + structural reductions) before the
    /// interior-point loop.  Presolve is almost always beneficial -- it shrinks
    /// the problem, improves conditioning, and de-degenerates the KKT system.
    /// Disable only when debugging or when the problem is known to be well-scaled
    /// and small.
    /// **Default:** `true`.
    pub presolve: bool,

    /// Maximum number of Ruiz equilibration sweeps.  Each sweep computes inf-norms
    /// of the constraint matrix rows/columns and updates the scaling factors.
    /// **Default:** `5`.  **Range:** `[0, 20]`.
    pub equilibration_iters: usize,

    /// Convergence tolerance for Ruiz equilibration.  Sweeps stop early when every
    /// scale factor is within this distance of 1.0 (i.e. no scaling is needed).
    /// **Default:** `1e-2`.
    pub equilibration_tol: T,

    /// Prefer the sparse augmented KKT factorization (fill-reducing LDL^T) over the
    /// dense condensed one.  Auto-selected by the dispatch layer for sparse problems;
    /// set this only to force the sparse path for debugging.
    /// **Default:** `false` (auto).
    pub sparse_kkt: bool,

    /// Aggregation fill budget: the doubleton-substitution pass
    /// skips an elimination whose projected fill (new inequality-row entries
    /// plus the Hessian row it densifies) would push the total nonzeros past
    /// `fill_budget` × the current total. 1.0 = no fill allowed; larger = more
    /// aggressive substitution. **Default:** `10.0` (permissive — the suite's
    /// doubleton-heavy problems need the eliminations; the gate exists for
    /// pathological dense-coupling shapes).
    pub fill_budget: f64,

    /// Dualize wide LPs: when the LP has no equality rows and
    /// the inequality count exceeds `dualize_ratio` × n, solve the dual
    /// (n equality rows instead of m inequality rows) and recover the primal
    /// solution from the dual's multipliers. Dimensionality-neutral for ICONIC's
    /// engines (the condensed system is n×n either way; the simplex tableau is
    /// symmetric under transposition) — the value is the dual's conditioning
    /// on numerically asymmetric data. Default: `false` (capability, opt-in).
    pub dualize: bool,

    /// Wide-LP dualization threshold (inequality rows / variables).
    /// **Default:** `4.0`.
    pub dualize_ratio: f64,

    /// Platform-BLAS worker-thread cap — a Threads-style parameter
    /// (`None` = automatic — the measured optimum, 4; the large LAPACK
    /// `dsytrf` factors scale with it, and 16+ threads measurably hurt the
    /// pivoted factorizations). **Default:** `None` (automatic).
    pub blas_threads: Option<usize>,

    /// Presolve reduction rounds (solve-check-solve orchestration): re-run
    /// the structural reduction chain while a round still removes ≥ 0.5% of
    /// rows+cols. **Default:** `1` — the re-entry machinery is opt-in: a
    /// second round changes search trajectories on quadratic MIPs (measured:
    /// portf_n25_k8's solve moved to a better feasible point whose proof
    /// interacts with the round-1 path's established proof — needs a
    /// dedicated soundness investigation before it becomes the default).
    pub presolve_rounds: usize,

    /// Use the Homogeneous Self-Dual embedding.  The HSDE simultaneously detects
    /// optimality, infeasibility, and unboundedness in a single solve without
    /// separate certificate computation.  Adds one extra variable (`tau`) and one
    /// extra constraint (`kappa`).
    /// **Default:** `true`.
    pub hsd: bool,

    /// Run the cone-aware presolve passes on the conic standard form before
    /// engine dispatch: empty-cone dropping (all-zero constraint rows whose
    /// fixed slack `s = b_block` is strictly inside the cone), free-variable
    /// elimination with the exact Schur fold, and redundant conic row removal
    /// (identical cone rows; singleton-coordinate SOC balls implied by a
    /// sibling). Every reduction is an exact-equivalence rewrite with a
    /// postsolve record, so the returned solution is always in the original
    /// problem space. When `false`, the conic and nonsymmetric paths run
    /// exactly as before.
    /// **Default:** `true`.
    pub cone_presolve: bool,
    /// Accept a warm-start iterate (a previous near-solution) on the
    /// `iconic-api::solve_warm` entry. When `false`, a seed passed to
    /// `solve_warm` is ignored and the cold start is used, so the gate
    /// makes the warm-start behavior fully opt-in.
    /// **Default:** `false`.
    pub warm_start: bool,
}

impl<T: Scalar> Default for Settings<T> {
    fn default() -> Self {
        let from = |x: f64| T::from_f64(x).expect("scalar must represent tolerance literal");
        Self {
            eps_abs: from(1e-8),
            eps_rel: from(1e-8),
            eps_gap: from(1e-8),
            max_iters: 200,
            presolve: true,
            equilibration_iters: 5,
            equilibration_tol: from(1e-2),
            sparse_kkt: false,
            fill_budget: 10.0,
            dualize: false,
            dualize_ratio: 4.0,
            presolve_rounds: 1,
            blas_threads: None,
            hsd: true,
            cone_presolve: true,
            warm_start: false,
        }
    }
}

/// A warm-start iterate: `x`, `s` and `z` from a previous near-solution,
/// laid out exactly as the returned `Solution` of the same problem (the
/// equality block's `s` is zero and `z` is the equality multiplier).
///
/// Seeded into an engine solve, the iterates must satisfy `A x + s = b`
/// (a previous solution does, to solver tolerance) and `s`/`z` must be
/// strictly inside their cones — the seeding routine blends boundary-active
/// components toward each cone's center and silently falls back to the cold
/// start when the seed fails validation, so a warm start can only change
/// convergence speed, never the converged point.
#[derive(Clone, Debug)]
pub struct WarmStart<T: Scalar> {
    /// Primal variable `x`.
    pub x: Vec<T>,
    /// Slack `s` (with `A x + s = b`).
    pub s: Vec<T>,
    /// Dual variable `z` for the conic constraint.
    pub z: Vec<T>,
}

/// The solution returned to the caller, in original (unscaled) units.
#[derive(Clone, Debug)]
pub struct Solution<T: Scalar> {
    /// Terminal status.
    pub status: Status,
    /// Primal variable `x`.
    pub x: Vec<T>,
    /// Slack `s` (with `Ax + s = b`).
    pub s: Vec<T>,
    /// Dual variable `z` for the conic constraint.
    pub z: Vec<T>,
    /// Objective value `½xᵀPx + qᵀx`.
    pub obj_val: T,
    /// Iterations taken.
    pub iters: usize,
    /// Equality multipliers (for simplex crossover).
    pub y: Vec<T>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cone_dims() {
        assert_eq!(Cone::Zero(4).dim(), 4);
        assert_eq!(Cone::SecondOrder(3).dim(), 3);
        assert_eq!(Cone::Exponential.dim(), 3);
        assert_eq!(Cone::GenPower(vec![0.3, 0.7], 5).dim(), 7);
    }

    #[test]
    fn default_settings_f64() {
        let s = Settings::<f64>::default();
        assert_eq!(s.eps_abs, 1e-8);
        assert_eq!(s.max_iters, 200);
    }
}
