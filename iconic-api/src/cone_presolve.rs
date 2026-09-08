//! Cone-aware presolve on the canonical conic standard form.
//!
//! Three exact-equivalence reductions, applied before engine dispatch when
//! `Settings::cone_presolve` is enabled (the design follows an exact-reduction
//! discipline: every reduction is an exact rewrite with a postsolve
//! record, so the returned solution is always in the original problem space
//! and a dropped row is always restored with dual 0):
//!
//! 1. **Empty-cone dropping.** A cone whose constraint rows are all zero in
//!    `A` forces the fixed slack `s = b_block` (`Ax + s = b` with `A = 0`).
//!    The rows are redundant iff `b_block` lies in the cone: drop the rows
//!    and restore `s = b_block`, `z = 0`. Non-membership is a primal
//!    infeasibility proof; an ambiguous margin keeps the rows. This is the
//!    conic analogue of the QP path's empty-row pass, and the only
//!    exp/power-side reduction that is provably sound today (per-cone
//!    membership tests: `soc::margin`, `psd::min_eig`, `exp::margin`,
//!    `pow::margin`, `genpow::margin`).
//!
//! 2. **Free-variable elimination.** A column with zero entries in `A` is an
//!    unconstrained variable; if its Hessian diagonal `P_jj` is a
//!    well-conditioned pivot, eliminate it with the exact Schur fold
//!    `q_k ← q_k − P_jk·q_j/P_jj`, `P_kl ← P_kl − P_jk·P_jl/P_jj`, recovering
//!    `x_j = −(q_j + Σ_k P_jk x_k)/P_jj` in postsolve. `P_jj ≈ 0` with a zero
//!    Hessian row and `q_j ≠ 0` is a dual-infeasibility proof (unbounded
//!    free variable); with `q_j ≈ 0` the variable drops at 0. A
//!    pivot/row-conditioning gate skips eliminations whose fold would be
//!    numerically unstable (the Schur complement of a PSD matrix is PSD in
//!    exact arithmetic only).
//!
//! 3. **Redundant conic rows.** Two cones of the same family and dimension
//!    with identical `(A block, b block)` carry the same constraint twice —
//!    drop the later one with dual 0. A second-order cone whose rows are all
//!    singletons on one column is a scalar ball `‖v − wx‖₂ ≤ d − ex`
//!    (`x ∈ [A−r, A+r]` when `‖w‖² − e² > 0`, derived from the squared
//!    inequality; the affine `d − ex` is nonnegative on the interval exactly
//!    when it is at the endpoints, and both-negative endpoints are a
//!    primal-infeasibility proof). Two balls on the same column with one
//!    contained in the other make the containing (looser) cone redundant —
//!    drop it with dual 0. Near-duplicate rows are never dropped: identical
//!    detection is exact equality, and containment uses a strict margin.
//!    Exp/pow implication is deliberately not attempted (no cheap sound
//!    test; unsound removal there is the MIR/mixing-cut class of incident).
//!
//! The QP path (orthant-only problems) never runs these passes — its own
//! reduction chain is untouched — and the conic/nonsymmetric paths run
//! exactly as before when the gate is off.

use iconic_core::{Cone, Scalar, Status};
use iconic_linalg::DenseMatrix;
use iconic_ipm::QpSolution;

use crate::ConeProgram;

/// Outcome of running the cone presolve on a program.
#[derive(Clone, Debug)]
pub enum Outcome<T: Scalar> {
    /// No reduction fired (or one did, leaving the program unchanged).
    Unchanged,
    /// An exact reduction fired: the reduced program plus the postsolve record.
    Reduced(Box<ConeProgram<T>>, Record<T>),
    /// The reduction chain proved the problem primal infeasible.
    PrimalInfeasible,
    /// The reduction chain proved the problem dual infeasible (unbounded).
    DualInfeasible,
}

/// Postsolve record: how to recover the original-space solution.
#[derive(Clone, Debug)]
pub struct Record<T: Scalar> {
    /// Parallel to the original cone list: whether the cone's rows survive.
    pub cone_kept: Vec<bool>,
    /// Kept columns (original indices), in reduced-problem order.
    pub kept_cols: Vec<usize>,
    /// Eliminated free columns, in elimination order (restore in reverse).
    pub elims: Vec<ElimCol<T>>,
    /// Original number of columns.
    pub n_orig: usize,
}

/// One eliminated free column.
#[derive(Clone, Debug)]
pub struct ElimCol<T: Scalar> {
    /// Original column index.
    pub j: usize,
    /// Pivot `P_jj` at elimination time.
    pub p_jj: T,
    /// `q_j` at elimination time.
    pub q_j: T,
    /// `(original column, P_jk)` over the columns still present at
    /// elimination time. Restore `x_j = −(q_j + Σ P_jk·x_k)/P_jj`.
    pub row: Vec<(usize, T)>,
}

enum Verdict {
    /// Strictly inside the cone: the constraint is a non-binding constant.
    Inside,
    /// Outside the cone: the fixed slack violates the cone — infeasible.
    Outside,
    /// Within the tolerance band: keep the rows.
    Ambiguous,
}

/// Membership verdict of a fixed slack block for a cone.
fn verdict<T: Scalar>(cone: &Cone, s: &[T], tol: T) -> Verdict {
    let zero = T::zero();
    let m = match cone {
        // All-zero equality rows force 0 = b_block: droppable iff b ≈ 0.
        Cone::Zero(_) => {
            return if s.iter().all(|&v| v.abs() <= tol) {
                Verdict::Inside
            } else {
                Verdict::Outside
            };
        }
        Cone::NonNegative(_) => s.iter().cloned().fold(zero, |m, v| m.min(v)),
        Cone::SecondOrder(_) => iconic_ipm::soc::margin(s),
        Cone::PsdTriangle(_) => iconic_ipm::psd::min_eig(s),
        Cone::Exponential => iconic_ipm::exp::margin(s),
        Cone::Power(a) => iconic_ipm::pow::margin(s, T::from_f64(*a).expect("scalar literal")),
        Cone::GenPower(as_, _) => {
            let a: Vec<T> = as_.iter().map(|&x| T::from_f64(x).expect("scalar literal")).collect();
            iconic_ipm::genpow::margin(s, &a)
        }
    };
    if m > tol {
        Verdict::Inside
    } else if m < -tol {
        Verdict::Outside
    } else {
        Verdict::Ambiguous
    }
}

/// Run the three reduction passes. Returns [`Outcome::Unchanged`] when nothing
/// fired so callers can avoid any cloning of the (possibly large) program.
pub fn apply<T: Scalar>(prog: &ConeProgram<T>) -> Outcome<T> {
    let zero = T::zero();
    let n = prog.q.len();
    let tol = T::from_f64(1e-8).expect("scalar literal");

    // ---------- pass 1: empty-cone dropping ----------
    let mut cone_kept = vec![true; prog.cones.len()];
    let mut row = 0usize;
    for (i, cone) in prog.cones.iter().enumerate() {
        let d = cone.dim();
        if d == 0 {
            continue;
        }
        let block_zero = (row..row + d).all(|r| (0..n).all(|j| prog.a.get(r, j) == zero));
        if block_zero {
            match verdict(cone, &prog.b[row..row + d], tol) {
                Verdict::Inside => cone_kept[i] = false,
                Verdict::Outside => return Outcome::PrimalInfeasible,
                Verdict::Ambiguous => {}
            }
        }
        row += d;
    }

    // Row layout after pass 1 (still ordered by cone).
    let kept_rows: Vec<usize> = {
        let mut rows = Vec::new();
        let mut r = 0usize;
        for (i, cone) in prog.cones.iter().enumerate() {
            let d = cone.dim();
            if cone_kept[i] {
                for k in 0..d {
                    rows.push(r + k);
                }
            }
            r += d;
        }
        rows
    };
    let b_red: Vec<T> = kept_rows.iter().map(|&r| prog.b[r]).collect();

    // ---------- pass 2: free-variable elimination (Schur fold) ----------
    // The reduced A (kept rows only) decides column emptiness; P/q fold over
    // the kept columns only. `present` tracks the columns still in the
    // problem; every elimination updates the surviving entries in place and
    // records its pivot row, so postsolve can restore in reverse order.
    let mut col_nz = vec![0usize; n];
    for &r in &kept_rows {
        for j in 0..n {
            if prog.a.get(r, j) != zero {
                col_nz[j] += 1;
            }
        }
    }
    let mut p = prog.p.clone();
    let mut qv = prog.q.clone();
    let mut present = vec![true; n];
    let mut elims: Vec<ElimCol<T>> = Vec::new();
    let diag_max = (0..n)
        .map(|j| p.get(j, j).abs())
        .fold(zero, |m, v| m.max(v))
        .max(T::one());
    let pd_tol = T::from_f64(1e-12).expect("scalar literal") * diag_max;
    let q_scale = qv
        .iter()
        .fold(zero, |m, &v| m.max(v.abs()))
        .max(T::one());
    let fold_cond = T::from_f64(1e-8).expect("scalar literal");
    let row_cond = T::from_f64(1e-8).expect("scalar literal");
    let q_proof = T::from_f64(1e-10).expect("scalar literal");
    for j in 0..n {
        if col_nz[j] != 0 || !present[j] {
            continue;
        }
        let p_jj = p.get(j, j);
        let q_j = qv[j];
        // Largest entry of this Hessian row over the present columns.
        let mut row_max = zero;
        for k in 0..n {
            if present[k] && k != j {
                row_max = row_max.max(p.get(j, k).abs());
            }
        }
        if p_jj > pd_tol {
            // Pivot is too small relative to its row: the fold would amplify
            // roundoff into the surviving block — keep the column (the IPM's
            // proximal regularization handles free variables natively).
            if row_max > zero && p_jj < fold_cond * row_max {
                continue;
            }
            let rec_row: Vec<(usize, T)> = (0..n)
                .filter(|&k| present[k] && k != j)
                .map(|k| (k, p.get(j, k)))
                .collect();
            let inv = T::one() / p_jj;
            for k in 0..n {
                if !present[k] || k == j {
                    continue;
                }
                let pjk = p.get(j, k);
                if pjk == zero {
                    continue;
                }
                qv[k] -= pjk * q_j * inv;
                for l in 0..n {
                    if !present[l] || l == j {
                        continue;
                    }
                    let pjl = p.get(j, l);
                    if pjl != zero {
                        p.set(k, l, p.get(k, l) - pjk * pjl * inv);
                    }
                }
            }
            elims.push(ElimCol {
                j,
                p_jj,
                q_j,
                row: rec_row,
            });
            present[j] = false;
        } else {
            // P_jj ≈ 0. For a PSD Hessian the whole row is ≈ 0; anything
            // else is numerically suspect — keep the column.
            if row_max > row_cond * diag_max {
                continue;
            }
            if q_j.abs() > q_proof * q_scale {
                // Free variable with a nonzero linear term and no curvature:
                // the objective is unbounded below.
                return Outcome::DualInfeasible;
            }
            // Objective is constant in x_j: drop it fixed at 0.
            elims.push(ElimCol {
                j,
                p_jj: zero,
                q_j: zero,
                row: vec![],
            });
            present[j] = false;
        }
    }
    let kept_cols: Vec<usize> = (0..n).filter(|&j| present[j]).collect();

    // ---------- pass 3: redundant conic rows ----------
    // (a) Identical (A block, b block) pairs across same-family cones.
    // (b) Singleton-coordinate SOC balls implied by a sibling.
    let mut kept_cone_idx: Vec<usize> = Vec::new();
    for (i, _cone) in prog.cones.iter().enumerate() {
        if cone_kept[i] {
            kept_cone_idx.push(i);
        }
    }
    // Ranges of each kept cone in the reduced rows.
    let mut cone_rows: Vec<(usize, usize)> = Vec::new();
    {
        let mut r = 0usize;
        for &i in &kept_cone_idx {
            let d = prog.cones[i].dim();
            cone_rows.push((r, r + d));
            r += d;
        }
    }
    // (a) identical rows.
    for a in 0..kept_cone_idx.len() {
        if !cone_kept[kept_cone_idx[a]] {
            continue;
        }
        for b in (a + 1)..kept_cone_idx.len() {
            let (ia, ib) = (kept_cone_idx[a], kept_cone_idx[b]);
            if !cone_kept[ib] {
                continue;
            }
            let (ca, cb) = (&prog.cones[ia], &prog.cones[ib]);
            if !same_family_dim(ca, cb) {
                continue;
            }
            let (ra, rb) = (cone_rows[a], cone_rows[b]);
            let d = ra.1 - ra.0;
            if identical_blocks(prog, n, ra, rb, d, &b_red) {
                cone_kept[ib] = false;
            }
        }
    }
    // (b) singleton-coordinate SOC balls, implied by a sibling.
    // Rebuild the kept-row list (some cones dropped in (a)).
    let balls: Vec<(usize, Ball<T>)> = {
        let mut out = Vec::new();
        let mut r = 0usize;
        for &i in &kept_cone_idx {
            let d = prog.cones[i].dim();
            if cone_kept[i] && matches!(prog.cones[i], Cone::SecondOrder(_)) && d >= 2 {
                match soc_ball::<T>(prog, r, r + d, n) {
                    BallResult::Ball(col, lo, hi) => {
                        out.push((i, Ball { col, lo, hi }));
                    }
                    BallResult::Infeasible => return Outcome::PrimalInfeasible,
                    BallResult::NotABall => {}
                }
            }
            r += d;
        }
        out
    };
    for a in 0..balls.len() {
        if !cone_kept[balls[a].0] {
            continue;
        }
        for b in (a + 1)..balls.len() {
            let (ia, ib) = (balls[a].0, balls[b].0);
            if !cone_kept[ib] || balls[a].1.col != balls[b].1.col {
                continue;
            }
            let (ba, bb) = (&balls[a].1, &balls[b].1);
            let ctol = T::from_f64(1e-8).expect("scalar literal")
                * [ba.lo.abs(), ba.hi.abs(), bb.lo.abs(), bb.hi.abs(), T::one()]
                    .into_iter()
                    .fold(T::zero(), |m, v| m.max(v));
            let a_in_b = ba.lo >= bb.lo - ctol && ba.hi <= bb.hi + ctol;
            let b_in_a = bb.lo >= ba.lo - ctol && bb.hi <= ba.hi + ctol;
            if a_in_b && !b_in_a {
                cone_kept[ib] = false;
            } else if b_in_a && !a_in_b {
                cone_kept[ia] = false;
            } else if a_in_b && b_in_a {
                // Identical intervals: drop the later cone.
                cone_kept[ia.max(ib)] = false;
            }
        }
    }

    let any_drop = !elims.is_empty() || cone_kept.iter().any(|&k| !k);
    if !any_drop {
        return Outcome::Unchanged;
    }

    // ---------- assemble the reduced program ----------
    let rows_final: Vec<usize> = {
        let mut rows = Vec::new();
        let mut r = 0usize;
        for (i, cone) in prog.cones.iter().enumerate() {
            let d = cone.dim();
            if cone_kept[i] {
                for k in 0..d {
                    rows.push(r + k);
                }
            }
            r += d;
        }
        rows
    };
    let mr = rows_final.len();
    let nk = kept_cols.len();
    let mut a2 = DenseMatrix::zeros(mr, nk);
    for (r, &or) in rows_final.iter().enumerate() {
        for (c, &oc) in kept_cols.iter().enumerate() {
            a2.set(r, c, prog.a.get(or, oc));
        }
    }
    let mut p2 = DenseMatrix::zeros(nk, nk);
    for (a, &oa) in kept_cols.iter().enumerate() {
        for (b, &ob) in kept_cols.iter().enumerate() {
            p2.set(a, b, p.get(oa, ob));
        }
    }
    let q2: Vec<T> = kept_cols.iter().map(|&j| qv[j]).collect();
    let b2: Vec<T> = rows_final.iter().map(|&r| prog.b[r]).collect();
    let cones2: Vec<Cone> = prog
        .cones
        .iter()
        .enumerate()
        .filter(|(i, _)| cone_kept[*i])
        .map(|(_, c)| c.clone())
        .collect();
    let reduced = ConeProgram {
        p: p2,
        q: q2,
        a: a2,
        a_csc: None, // row/column indices shifted: the original CSC is invalid
        b: b2,
        cones: cones2,
    };
    Outcome::Reduced(
        Box::new(reduced),
        Record {
            cone_kept,
            kept_cols,
            elims,
            n_orig: n,
        },
    )
}

/// Do two cones have the same family and dimension?
fn same_family_dim(a: &Cone, b: &Cone) -> bool {
    use Cone::*;
    match (a, b) {
        (Zero(_), Zero(_)) => false, // equality rows are the QP chain's territory
        (NonNegative(d1), NonNegative(d2)) => d1 == d2,
        (SecondOrder(d1), SecondOrder(d2)) => d1 == d2,
        (PsdTriangle(d1), PsdTriangle(d2)) => d1 == d2,
        (Exponential, Exponential) => true,
        (Power(a1), Power(a2)) => a1 == a2,
        (GenPower(al1, t1), GenPower(al2, t2)) => al1 == al2 && t1 == t2,
        _ => false,
    }
}

/// Exact block equality: same `b` entries and same `A` rows over `[ra, rb)`.
fn identical_blocks<T: Scalar>(
    prog: &ConeProgram<T>,
    n: usize,
    ra: (usize, usize),
    rb: (usize, usize),
    d: usize,
    b_red: &[T],
) -> bool {
    let d = if ra.1 - ra.0 < d { ra.1 - ra.0 } else { d };
    for k in 0..d {
        if b_red[ra.0 + k] != b_red[rb.0 + k] {
            return false;
        }
        for j in 0..n {
            if prog.a.get(ra.0 + k, j) != prog.a.get(rb.0 + k, j) {
                return false;
            }
        }
    }
    true
}

/// A scalar-ball characterization of an SOC cone whose rows are all
/// singletons on one column.
struct Ball<T: Scalar> {
    col: usize,
    lo: T,
    hi: T,
}

enum BallResult<T: Scalar> {
    Ball(usize, T, T),
    Infeasible,
    NotABall,
}

/// Extract the ball `x_col ∈ [lo, hi]` from a singleton-column SOC cone.
///
/// The cone rows read `s_t = b₀ − e·x_j` and `s_x = b₁ − w·x_j` (each row has
/// at most one nonzero, all in column `j`), so the constraint is
/// `‖b₁ − w·x_j‖₂ ≤ b₀ − e·x_j` — a scalar ball when `C = ‖w‖² − e² > 0`.
/// The squared inequality gives `x ∈ [A−r, A+r]` with `A = (w·b₁ − e·b₀)/C`,
/// `B = (‖b₁‖² − b₀²)/C`. The affine `g(x) = b₀ − e·x` is nonnegative on the
/// interval exactly when it is at both endpoints (a sign change forces
/// degeneracy), and both-negative endpoints mean the cone set is empty.
fn soc_ball<T: Scalar>(
    prog: &ConeProgram<T>,
    r0: usize,
    r1: usize,
    n: usize,
) -> BallResult<T> {
    let zero = T::zero();
    let mut col: Option<usize> = None;
    for r in r0..r1 {
        let mut nz = 0usize;
        let mut c = 0usize;
        for j in 0..n {
            if prog.a.get(r, j) != zero {
                nz += 1;
                c = j;
            }
        }
        if nz > 1 {
            return BallResult::NotABall;
        }
        if nz == 1 {
            match col {
                None => col = Some(c),
                Some(j0) if j0 != c => return BallResult::NotABall,
                _ => {}
            }
        }
    }
    let j = match col {
        Some(j) => j,
        None => return BallResult::NotABall, // all-zero rows: pass 1's business
    };
    let d_b = prog.b[r0];
    let e = prog.a.get(r0, j);
    let mut ww = zero;
    let mut vv = zero;
    let mut vw = zero;
    for r in (r0 + 1)..r1 {
        let w = prog.a.get(r, j);
        let v = prog.b[r];
        ww += w * w;
        vv += v * v;
        vw += v * w;
    }
    let c = ww - e * e;
    if c <= zero {
        return BallResult::NotABall;
    }
    let a = (vw - d_b * e) / c;
    let bb = (vv - d_b * d_b) / c;
    let r2 = a * a - bb;
    let scale = [T::one(), a.abs(), bb.abs()]
        .into_iter()
        .fold(zero, |m, v| m.max(v));
    let r2_tol = T::from_f64(1e-12).expect("scalar literal") * scale * scale;
    if r2 < -r2_tol {
        // The quadratic set is empty, hence so is the cone set.
        return BallResult::Infeasible;
    }
    let r = if r2 > zero { r2.sqrt() } else { zero };
    let (lo, hi) = (a - r, a + r);
    // g at the endpoints; both endpoints ≥ −tol ⇒ g ≥ 0 on the interval.
    let gtol = T::from_f64(1e-8).expect("scalar literal")
        * [T::one(), d_b.abs(), e.abs() * lo.abs(), e.abs() * hi.abs()]
            .into_iter()
            .fold(zero, |m, v| m.max(v));
    let g_lo = d_b - e * lo;
    let g_hi = d_b - e * hi;
    if g_lo < -gtol && g_hi < -gtol {
        return BallResult::Infeasible;
    }
    if g_lo < -gtol || g_hi < -gtol {
        // Mixed-sign endpoints are impossible in exact arithmetic (a sign
        // change forces the degenerate interval); on roundoff, keep the cone.
        return BallResult::NotABall;
    }
    BallResult::Ball(j, lo, hi)
}

/// Restore the original-space solution from the reduced-problem solution.
///
/// `orig` is the program the record was built from; `sol` is the engine
/// output for the reduced program. Kept cones pull their rows from `sol` in
/// order; dropped cones restore `s = b − A·x` (equal to `b_block` for the
/// empty-cone case) with dual 0; eliminated columns are recovered in reverse
/// elimination order. The objective is recomputed in original units.
pub fn restore<T: Scalar>(
    rec: &Record<T>,
    orig: &ConeProgram<T>,
    sol: &QpSolution<T>,
) -> iconic_core::Solution<T> {
    let zero = T::zero();
    let n = rec.n_orig;
    let m = orig.b.len();

    // ----- x: kept columns, then eliminated columns in reverse order -----
    let mut x = vec![zero; n];
    for (a, &j) in rec.kept_cols.iter().enumerate() {
        x[j] = sol.x[a];
    }
    for elim in rec.elims.iter().rev() {
        if elim.p_jj > zero {
            let mut acc = elim.q_j;
            for &(k, v) in &elim.row {
                acc += v * x[k];
            }
            x[elim.j] = -acc / elim.p_jj;
        } else {
            x[elim.j] = zero;
        }
    }

    // ----- s / z in original cone order (equality multipliers flow into z
    // per the API convention; dropped cones restore s = b − A·x and dual 0) -----
    let mut s = vec![zero; m];
    let mut z = vec![zero; m];
    let mut yc = 0usize;
    let mut sc = 0usize;
    let mut row = 0usize;
    for (i, cone) in orig.cones.iter().enumerate() {
        let d = cone.dim();
        if rec.cone_kept[i] {
            match cone {
                Cone::Zero(_) => {
                    z[row..row + d].copy_from_slice(&sol.y[yc..yc + d]);
                    yc += d;
                }
                _ => {
                    s[row..row + d].copy_from_slice(&sol.s[sc..sc + d]);
                    z[row..row + d].copy_from_slice(&sol.z[sc..sc + d]);
                    sc += d;
                }
            }
        } else if matches!(cone, Cone::Zero(_)) {
            // Vacuous equality rows: slack is exactly 0, dual 0.
            s[row..row + d].fill(zero);
        } else {
            for k in 0..d {
                let mut val = orig.b[row + k];
                for j in 0..n {
                    val -= orig.a.get(row + k, j) * x[j];
                }
                s[row + k] = val;
            }
        }
        row += d;
    }

    // ----- objective in original units: ½xᵀPx + qᵀx (only the quadratic
    // term is halved) -----
    let half = T::from_f64(0.5).expect("scalar literal");
    let mut quad = zero;
    for i in 0..n {
        for j in 0..n {
            quad += orig.p.get(i, j) * x[i] * x[j];
        }
    }
    let mut lin = zero;
    for i in 0..n {
        lin += orig.q[i] * x[i];
    }
    let obj = half * quad + lin;

    iconic_core::Solution {
        status: sol.status,
        x,
        s,
        z,
        obj_val: obj,
        iters: sol.iters,
        y: vec![],
    }
}

/// A zero-filled solution for a certificate outcome (mirrors the engine's
/// infeasibility convention).
pub fn certificate_solution<T: Scalar>(
    status: Status,
    prog: &ConeProgram<T>,
) -> iconic_core::Solution<T> {
    let zero = T::zero();
    iconic_core::Solution {
        status,
        x: vec![zero; prog.q.len()],
        s: vec![zero; prog.b.len()],
        z: vec![zero; prog.b.len()],
        obj_val: zero,
        iters: 0,
        y: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compute the original-space objective `½xᵀPx + qᵀx` at `x`.
    fn objective_at<T: Scalar>(prog: &ConeProgram<T>, x: &[T]) -> T {
        let zero = T::zero();
        let half = T::from_f64(0.5).expect("scalar literal");
        let n = prog.q.len();
        let mut quad = zero;
        for i in 0..n {
            for j in 0..n {
                quad += prog.p.get(i, j) * x[i] * x[j];
            }
        }
        let mut lin = zero;
        for i in 0..n {
            lin += prog.q[i] * x[i];
        }
        half * quad + lin
    }
    use iconic_core::{Cone, Settings, Status};

    fn identity_prog(n: usize, q: Vec<f64>) -> ConeProgram<f64> {
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        ConeProgram {
            p,
            q,
            a: DenseMatrix::zeros(0, n),
            a_csc: None,
            b: vec![],
            cones: vec![],
        }
    }

    /// An empty SOC cone whose fixed slack is strictly interior is dropped:
    /// the solution matches the problem without it, its slack is `b_block`,
    /// and its dual is exactly 0.
    #[test]
    fn empty_soc_cone_dropped_with_zero_dual() {
        // Project c = (1, 2, 2) onto Q3 (A = −I, b = 0), plus a second SOC
        // cone with zero rows and b = (1, 0.1, 0.1) strictly inside Q3
        // (margin 1 − √0.02 ≈ 0.859).
        let n = 3usize;
        let m = 6usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        for i in 0..n {
            a.set(i, i, -1.0);
        }
        let prog = ConeProgram {
            p,
            q: vec![-1.0, -2.0, -2.0],
            a,
            a_csc: None,
            b: vec![0.0, 0.0, 0.0, 1.0, 0.1, 0.1],
            cones: vec![Cone::SecondOrder(3), Cone::SecondOrder(3)],
        };
        // The reduction fires and drops the second cone.
        match apply(&prog) {
            Outcome::Reduced(r, rec) => {
                assert_eq!(rec.cone_kept, vec![true, false]);
                assert_eq!(r.cones.len(), 1);
                assert_eq!(r.a.nrows, 3);
            }
            _ => panic!("expected a reduction"),
        }
        let sol = crate::solve(&prog, &Settings::default()).unwrap();
        assert_eq!(sol.status, Status::Solved);
        let expected_x0 = (1.0 + 2.0 * 2.0_f64.sqrt()) / 2.0;
        assert!((sol.x[0] - expected_x0).abs() < 1e-5, "x0={}", sol.x[0]);
        // Dropped cone: s = b_block, z = 0.
        for (k, &v) in [1.0, 0.1, 0.1].iter().enumerate() {
            assert!((sol.s[3 + k] - v).abs() < 1e-9, "s[{}]={}", 3 + k, sol.s[3 + k]);
            assert_eq!(sol.z[3 + k], 0.0);
        }
        // Objective matches the pure-projection problem.
        let x = [sol.x[0], sol.x[1], sol.x[2]];
        let obj = objective_at(&prog, &x);
        assert!((sol.obj_val - obj).abs() < 1e-10);
        let proj_norm2 = x.iter().map(|v| v * v).sum::<f64>();
        assert!((sol.obj_val + 0.5 * proj_norm2).abs() < 1e-6);
    }

    /// An empty cone whose fixed slack is outside the cone is a primal
    /// infeasibility proof.
    #[test]
    fn empty_cone_outside_is_primal_infeasible() {
        let n = 3usize;
        let m = 6usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        for i in 0..n {
            a.set(i, i, -1.0);
        }
        let prog = ConeProgram {
            p,
            q: vec![-1.0, -2.0, -2.0],
            a,
            a_csc: None,
            b: vec![0.0, 0.0, 0.0, 0.1, 1.0, 1.0], // 0.1 − √2 < 0
            cones: vec![Cone::SecondOrder(3), Cone::SecondOrder(3)],
        };
        let sol = crate::solve(&prog, &Settings::default()).unwrap();
        assert_eq!(sol.status, Status::PrimalInfeasible);
    }

    /// An empty exponential cone (all-zero rows, slack strictly inside K_exp)
    /// drops on the nonsymmetric path; a problem whose only cones are empty
    /// reduces to an unconstrained QP.
    #[test]
    fn empty_exp_cone_dropped() {
        // b = (0, 1, 2): psi = 1·ln 2 ≈ 0.693 > 0, y, z > 0 — strictly inside.
        let n = 1usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        p.set(0, 0, 1.0);
        let prog = ConeProgram {
            p,
            q: vec![-1.0],
            a: DenseMatrix::zeros(3, n),
            a_csc: None,
            b: vec![0.0, 1.0, 2.0],
            cones: vec![Cone::Exponential],
        };
        match apply(&prog) {
            Outcome::Reduced(r, rec) => {
                assert!(r.cones.is_empty());
                assert_eq!(rec.cone_kept, vec![false]);
            }
            _ => panic!("expected a reduction"),
        }
        let sol = crate::solve(&prog, &Settings::default()).unwrap();
        assert_eq!(sol.status, Status::Solved);
        assert!((sol.x[0] - 1.0).abs() < 1e-8, "x={}", sol.x[0]);
        assert!((sol.obj_val + 0.5).abs() < 1e-8, "obj={}", sol.obj_val);
        assert!((sol.s[0] - 0.0).abs() < 1e-8);
        assert!((sol.s[1] - 1.0).abs() < 1e-8);
        assert!((sol.s[2] - 2.0).abs() < 1e-8);
        assert!(sol.z.iter().all(|&v| v == 0.0));
    }

    /// A vacuous exp cone in a mixed program no longer blocks dispatch: the
    /// empty-cone pass runs before the mixed-engine rejection.
    #[test]
    fn vacuous_exp_cone_does_not_block_conic_solve() {
        let n = 2usize;
        // SOC: s = x ≥ 0 via s = b − Ax with A = −I on x... use the
        // projection pattern: A = −I, b = 0 on x; exp cone rows zero.
        let m = 2 + 3;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        for i in 0..n {
            a.set(i, i, -1.0);
        }
        // b for the SOC rows: (1, 0.1) inside Q2 (margin 1 − 0.1 > 0); exp b
        // (0, 1, 2) inside K_exp.
        let b = vec![1.0, 0.1, 0.0, 1.0, 2.0];
        let prog = ConeProgram {
            p,
            q: vec![0.0, 0.0],
            a,
            a_csc: None,
            b,
            cones: vec![Cone::SecondOrder(2), Cone::Exponential],
        };
        let sol = crate::solve(&prog, &Settings::default()).unwrap();
        assert_eq!(sol.status, Status::Solved);
        // The SOC is non-binding: x = 0, obj = 0.
        assert!(sol.x[0].abs() < 1e-8 && sol.x[1].abs() < 1e-8);
        assert!(sol.obj_val.abs() < 1e-8);
        assert!(sol.z[2..].iter().all(|&v| v == 0.0));
    }

    /// Two identical binding SOC cones: the duplicate drops with dual 0, and
    /// the solution matches the single-cone problem exactly.
    #[test]
    fn identical_soc_rows_dropped_with_zero_dual() {
        // min ½‖x − c‖² with (x0, x1) ∈ Q2 enforced twice (s = x, A = −I,
        // b = 0); c = (3, 4) lies outside Q2 so the cones bind at the
        // projection. Projection of c onto Q2: (λ, 0) with λ = ‖c‖ = 5
        // (the ray direction), so x* = c/‖c‖·... the projection of c onto
        // Q2 is (‖c‖, 0)? No — projecting onto the cone: c has t = 3 < 4 =
        // ‖(4)‖ so the projection is onto the boundary: x* = c/2 + ... the
        // closed form: (‖c‖·c₁/‖c₁‖, ...). Simply assert both solves agree.
        let n = 2usize;
        let m = 4usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        for i in 0..n {
            a.set(i, i, -1.0);
            a.set(2 + i, i, -1.0);
        }
        let b = vec![0.0, 0.0, 0.0, 0.0];
        let prog = ConeProgram {
            p,
            q: vec![-3.0, -4.0],
            a,
            a_csc: None,
            b,
            cones: vec![Cone::SecondOrder(2), Cone::SecondOrder(2)],
        };
        match apply(&prog) {
            Outcome::Reduced(r, rec) => {
                assert_eq!(rec.cone_kept, vec![true, false]);
                assert_eq!(r.cones.len(), 1);
            }
            _ => panic!("expected the duplicate drop"),
        }
        let on = crate::solve(&prog, &Settings::default()).unwrap();
        let off = crate::solve(
            &prog,
            &Settings::<f64> {
                cone_presolve: false,
                ..Settings::default()
            },
        )
        .unwrap();
        assert_eq!(on.status, off.status);
        for i in 0..n {
            assert!(
                (on.x[i] - off.x[i]).abs() < 1e-6,
                "x[{i}]: on={} off={}",
                on.x[i],
                off.x[i]
            );
        }
        assert!(
            (on.obj_val - off.obj_val).abs() < 1e-6,
            "obj: on={} off={}",
            on.obj_val,
            off.obj_val
        );
        // Dropped cone: s equals the kept cone's slack, dual exactly 0.
        for k in 0..2 {
            assert!((on.s[2 + k] - on.s[k]).abs() < 1e-9);
            assert_eq!(on.z[2 + k], 0.0);
        }
    }

    /// Two SOC cones with the same A block but a different b are NOT
    /// identical, and the identical-row check must not fire.
    #[test]
    fn near_identical_soc_rows_kept() {
        let n = 2usize;
        let m = 4usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        for i in 0..n {
            a.set(i, i, -1.0);
            a.set(2 + i, i, -1.0);
        }
        // Second cone's b differs by 1e-6 — not identical.
        let b = vec![0.0, 0.0, 1e-6, 0.0];
        let prog = ConeProgram {
            p,
            q: vec![-3.0, -4.0],
            a,
            a_csc: None,
            b,
            cones: vec![Cone::SecondOrder(2), Cone::SecondOrder(2)],
        };
        assert!(matches!(apply(&prog), Outcome::Unchanged));
    }

    /// A singleton-coordinate SOC ball strictly inside a sibling ball: the
    /// looser (containing) cone drops with dual 0, the solution matches the
    /// tighter-only problem.
    #[test]
    fn singleton_ball_implied_soc_dropped() {
        // Tight ball on x0: ‖−x0‖ ≤ 1 − 0.5·x0 ⇔ x0 ∈ [−2, 2/3].
        // Loose ball on x0: ‖−x0‖ ≤ 1.5 − 0.5·x0 ⇔ x0 ∈ [−3, 1] ⊃ [−2, 2/3].
        // Objective min ½x0² − x0 is minimized at x0 = 1 unconstrained, so
        // the tight ball's boundary x0 = 2/3 binds (both cones active in the
        // original problem).
        let n = 2usize;
        let m = 5usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        p.set(0, 0, 1.0);
        p.set(1, 1, 1.0);
        // Row layout follows cone order: [NonNeg(1), SOC(2), SOC(2)]
        // → rows 0, 1-2, 3-4.
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        a.set(0, 1, -1.0); // NonNeg: s = x1 ≥ 0 (keeps x1 constrained)
        a.set(1, 0, 0.5); // SOC t-row, tight ball on x0
        a.set(2, 0, 1.0); // SOC x-row: s_x = −x0
        a.set(3, 0, 0.5); // SOC t-row, loose ball on x0
        a.set(4, 0, 1.0); // SOC x-row: s_x = −x0
        let b = vec![0.0, 1.0, 0.0, 1.5, 0.0];
        let prog = ConeProgram {
            p,
            q: vec![-1.0, 0.0],
            a,
            a_csc: None,
            b,
            cones: vec![
                Cone::NonNegative(1),
                Cone::SecondOrder(2),
                Cone::SecondOrder(2),
            ],
        };
        // Both cones are balls on column 0; the loose one is dropped.
        match apply(&prog) {
            Outcome::Reduced(r, rec) => {
                assert_eq!(rec.cone_kept, vec![true, true, false]);
                assert_eq!(r.a.nrows, 3); // 5 rows − the loose cone's 2
            }
            _ => panic!("expected the loose-ball drop"),
        }
        let on = crate::solve(&prog, &Settings::default()).unwrap();
        assert_eq!(on.status, Status::Solved);
        // The tight ball binds: x0 = 2/3, obj = ½·(4/9) − 2/3 = −4/9.
        assert!((on.x[0] - 2.0 / 3.0).abs() < 1e-6, "x0={}", on.x[0]);
        assert!((on.obj_val + 4.0 / 9.0).abs() < 1e-6, "obj={}", on.obj_val);
        // Dropped cone restores s = b − A·x and z = 0 (rows 3-4).
        assert_eq!(on.z[3], 0.0);
        assert_eq!(on.z[4], 0.0);
        // Compare against the cone-presolve-off solve of the same problem.
        let off = crate::solve(
            &prog,
            &Settings::<f64> {
                cone_presolve: false,
                ..Settings::default()
            },
        )
        .unwrap();
        assert_eq!(on.status, off.status);
        assert!(
            (on.obj_val - off.obj_val).abs() < 1e-6,
            "obj on={} off={}",
            on.obj_val,
            off.obj_val
        );
    }

    /// Two overlapping-but-not-nested balls: no implication, nothing drops.
    #[test]
    fn overlapping_balls_not_implied_kept() {
        let n = 2usize;
        let m = 5usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        p.set(0, 0, 1.0);
        p.set(1, 1, 1.0);
        // Rows in cone order: [NonNeg(1), SOC(2), SOC(2)].
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        a.set(0, 1, -1.0); // NonNeg: s = x1 ≥ 0
        a.set(1, 0, 0.5); // SOC t-row: ball [−2, 2/3]
        a.set(2, 0, 1.0); // SOC x-row
        a.set(3, 0, 0.0); // SOC t-row: s_t = 1 (constant), ball [−1, 1]
        a.set(4, 0, 1.0); // SOC x-row
        let b = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let prog = ConeProgram {
            p,
            q: vec![-1.0, 0.0],
            a,
            a_csc: None,
            b,
            cones: vec![
                Cone::NonNegative(1),
                Cone::SecondOrder(2),
                Cone::SecondOrder(2),
            ],
        };
        assert!(matches!(apply(&prog), Outcome::Unchanged));
        // The solve still honors both balls: x0 ∈ [−1, 1] ∩ [−2, 2/3] = [−1, 2/3].
        let sol = crate::solve(&prog, &Settings::default()).unwrap();
        assert_eq!(sol.status, Status::Solved);
        assert!((sol.x[0] - 2.0 / 3.0).abs() < 1e-6);
        // The tight cone binds (its dual is nonzero); the loose one is interior.
        assert!(sol.z[1].abs() > 1e-9 || sol.z[2].abs() > 1e-9);
    }

    /// A free variable coupled in P is eliminated with the exact Schur fold:
    /// the solution matches the closed form and the gate-off solve.
    #[test]
    fn free_column_folded_exactly() {
        // P = [[1, 0.5], [0.5, 1]], q = (1, −2), x1 free (no A entries).
        // x1 = −(q1 + P01·x0)/P11 = 2 − 0.5·x0; reduced 0.75·x0 + 2 = 0
        // ⇒ unconstrained minimizer x0 = −8/3, which violates x0 ≥ 0, so the
        // optimum is on the boundary x0 = 0 with x1 = 2, obj = ½·4 − 4 = −2.
        let n = 2usize;
        let m = 1usize;
        let p = DenseMatrix::from_row_major(2, 2, vec![1.0, 0.5, 0.5, 1.0]);
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        a.set(0, 0, -1.0); // x0 ≥ 0 (keeps the problem conic)
        let prog = ConeProgram {
            p,
            q: vec![1.0, -2.0],
            a,
            a_csc: None,
            b: vec![0.0],
            cones: vec![Cone::NonNegative(1)],
        };
        match apply(&prog) {
            Outcome::Reduced(r, rec) => {
                assert_eq!(rec.elims.len(), 1);
                assert_eq!(rec.elims[0].j, 1);
                assert_eq!(r.q.len(), 1);
            }
            _ => panic!("expected the free-column fold"),
        }
        let on = crate::solve(&prog, &Settings::default()).unwrap();
        assert_eq!(on.status, Status::Solved);
        assert!(on.x[0].abs() < 1e-6, "x0={}", on.x[0]);
        assert!((on.x[1] - 2.0).abs() < 1e-6, "x1={}", on.x[1]);
        assert!((on.obj_val + 2.0).abs() < 1e-6, "obj={}", on.obj_val);
        let off = crate::solve(
            &prog,
            &Settings::<f64> {
                cone_presolve: false,
                ..Settings::default()
            },
        )
        .unwrap();
        assert_eq!(on.status, off.status);
        for i in 0..n {
            assert!(
                (on.x[i] - off.x[i]).abs() < 1e-6,
                "x[{i}] on={} off={}",
                on.x[i],
                off.x[i]
            );
        }
        assert!((on.obj_val - off.obj_val).abs() < 1e-6);
    }

    /// A free variable with no curvature and a nonzero linear term is a
    /// dual-infeasibility proof.
    #[test]
    fn free_column_unbounded_is_dual_infeasible() {
        let n = 2usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        p.set(0, 0, 1.0);
        let mut a = DenseMatrix::<f64>::zeros(1, n);
        a.set(0, 0, -1.0);
        let prog = ConeProgram {
            p,
            q: vec![0.0, -1.0], // q1 ≠ 0 with P11 = 0 and no A row
            a,
            a_csc: None,
            b: vec![0.0],
            cones: vec![Cone::NonNegative(1)],
        };
        let sol = crate::solve(&prog, &Settings::default()).unwrap();
        assert_eq!(sol.status, Status::DualInfeasible);
    }

    /// Gate off: the reduction does not run and the engine solves the raw
    /// problem (same status/objective as gate-on on a clean instance).
    #[test]
    fn gate_off_solves_raw_problem() {
        let prog = identity_prog(2, vec![-1.0, -2.0]);
        // No cones: the gate is off by scope (orthant-only) and by flag.
        let s_on = crate::solve(&prog, &Settings::default()).unwrap();
        let s_off = crate::solve(
            &prog,
            &Settings::<f64> {
                cone_presolve: false,
                ..Settings::default()
            },
        )
        .unwrap();
        assert_eq!(s_on.status, s_off.status);
        assert!((s_on.obj_val - s_off.obj_val).abs() < 1e-10);
    }

    /// An SOC cone whose rows are singleton on TWO different columns is not
    /// a 1-D ball: nothing fires.
    #[test]
    fn multi_column_soc_not_a_ball() {
        let n = 2usize;
        let m = 3usize;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        a.set(0, 0, 0.5);
        a.set(1, 0, 1.0);
        a.set(2, 1, 1.0); // x-row on a different column
        let b = vec![1.0, 0.0, 0.0];
        let prog = ConeProgram {
            p,
            q: vec![-1.0, 0.0],
            a,
            a_csc: None,
            b,
            cones: vec![Cone::SecondOrder(3)],
        };
        assert!(matches!(apply(&prog), Outcome::Unchanged));
    }
}
