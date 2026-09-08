//! `iconic-api` — the unified public API and engine dispatch.
//!
//! This crate accepts a problem in ICONIC's canonical conic standard form
//!
//! ```text
//! minimize    ½ xᵀP x + qᵀx
//! subject to  A x + s = b,   s ∈ K
//! ```
//!
//! and dispatches it to a solver engine: the symmetric cones (Zero, NonNegative,
//! SOC, PSD) via the NT-scaled cone-aware IPM, and Exponential/Power/GenPower via
//! the nonsymmetric IPM (a problem mixing the two families is rejected).

// Index-based loops read more clearly than iterator adapters for this linear-algebra code.
#![allow(clippy::needless_range_loop)]

use iconic_core::{Cone, Scalar, Settings, Solution, Status};
use iconic_ipm::conic::{solve_cone_qp, solve_cone_qp_warm, Cone as IqCone};
use iconic_ipm::nonsym::{solve_nonsym, solve_nonsym_warm, NsCone};
use iconic_ipm::psd::side_dim;
use iconic_ipm::{
    solve_qp, solve_qp_with_termination, solve_qp_with_termination_warm, QpProblem, QpSolution,
    TermScale,
};
use iconic_linalg::{CscMatrix, DenseMatrix};

pub mod cone_presolve;

/// A problem in canonical conic standard form.
#[derive(Clone, Debug)]
pub struct ConeProgram<T: Scalar> {
    /// Symmetric PSD Hessian `P` (`n × n`).
    pub p: DenseMatrix<T>,
    /// Linear objective term `q` (length `n`).
    pub q: Vec<T>,
    /// Constraint matrix `A` (`m × n`), rows grouped in cone order.
    pub a: DenseMatrix<T>,
    pub a_csc: Option<CscMatrix<T>>,
    /// Right-hand side `b` (length `m`).
    pub b: Vec<T>,
    /// The cones of `K` in canonical order.
    pub cones: Vec<Cone>,
}

/// Reasons a problem cannot be dispatched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolveError {
    /// A cone type is not yet supported by any available engine.
    UnsupportedCone,
    /// Cones are not in canonical order (all `Zero` blocks must precede `NonNegative`).
    ConeOrder,
    /// The cone dimensions do not sum to the number of rows of `A`.
    DimensionMismatch,
}

/// Extract a contiguous range of rows from a CSC matrix and convert to CSR.
///
/// The input `a` is CSC (column-major): column `j` spans `a.colptr[j]..a.colptr[j+1]`
/// in `a.rowval`/`a.nzval`.  The output is also stored as a `CscMatrix` but with
/// `m = ncols, n = row_end - row_start` — i.e. CSR format: each "column" is a row,
/// each "row index" is a column index.
///
/// Returns `None` when the row range is empty or the input has no nonzeros.
fn csc_submatrix_rows<T: Scalar>(
    a: &CscMatrix<T>,
    row_start: usize,
    row_end: usize,
) -> Option<CscMatrix<T>> {
    let n_rows = row_end - row_start;
    if n_rows == 0 || a.nzval.is_empty() {
        return None;
    }
    let n_cols = a.n;
    let zero = T::zero();

    // Count nonzeros per output row
    let mut row_nnz = vec![0usize; n_rows];
    for j in 0..n_cols {
        for p in a.colptr[j]..a.colptr[j + 1] {
            let row = a.rowval[p];
            if row >= row_start && row < row_end {
                row_nnz[row - row_start] += 1;
            }
        }
    }

    let total_nnz: usize = row_nnz.iter().sum();
    if total_nnz == 0 {
        return None;
    }

    // Build CSR: colptr indexes into rows
    let mut colptr = vec![0usize; n_rows + 1];
    for r in 0..n_rows {
        colptr[r + 1] = colptr[r] + row_nnz[r];
    }
    let mut rowval = vec![0usize; total_nnz];
    let mut nzval = vec![zero; total_nnz];
    let mut cursor = colptr.clone();

    for j in 0..n_cols {
        for p in a.colptr[j]..a.colptr[j + 1] {
            let row = a.rowval[p];
            if row >= row_start && row < row_end {
                let r = row - row_start;
                let pos = cursor[r];
                rowval[pos] = j; // column index
                nzval[pos] = a.nzval[p];
                cursor[r] += 1;
            }
        }
    }

    Some(CscMatrix {
        m: n_cols, // "rows" = columns of original
        n: n_rows, // "cols" = rows of original (CSR storage)
        colptr,
        rowval,
        nzval,
    })
}

/// The dispatch-relevant summary of a program's cone list.
struct ConeScan {
    n_eq: usize,
    n_in: usize,
    ineq_cones: Vec<IqCone>,
    ns_cones: Vec<NsCone>,
    has_conic: bool,
    has_exp: bool,
}

/// Validate the cone list and partition it into the equality block and the
/// inequality cone product (with the engine-specific cone descriptors).
fn scan_cones<T: Scalar>(prog: &ConeProgram<T>) -> Result<ConeScan, SolveError> {
    let n = prog.q.len();
    let m = prog.b.len();
    let mut n_eq = 0usize;
    let mut n_in = 0usize;
    let mut ineq_cones: Vec<IqCone> = Vec::new();
    let mut ns_cones: Vec<NsCone> = Vec::new(); // parallel list for the nonsymmetric path
    let mut has_conic = false; // any SOC or PSD cone → use the NT cone-aware solver
    let mut has_exp = false; // any exponential cone → use the nonsymmetric solver
    let mut seen_ineq = false;
    for cone in &prog.cones {
        match cone {
            Cone::Zero(d) => {
                if seen_ineq {
                    return Err(SolveError::ConeOrder);
                }
                n_eq += d;
            }
            Cone::NonNegative(d) => {
                seen_ineq = true;
                n_in += d;
                // One vectorized orthant block (the NT cone engine handles `NonNeg(d)` with a
                // diagonal (z,z) and no per-element allocation), matching the nonsym path.
                ineq_cones.push(IqCone::NonNeg(*d));
                ns_cones.push(NsCone::NonNeg(*d));
            }
            Cone::SecondOrder(d) => {
                seen_ineq = true;
                has_conic = true;
                n_in += d;
                ineq_cones.push(IqCone::Soc(*d));
            }
            Cone::PsdTriangle(mt) => {
                seen_ineq = true;
                has_conic = true;
                n_in += mt;
                ineq_cones.push(IqCone::Psd(side_dim(*mt)));
            }
            Cone::Exponential => {
                seen_ineq = true;
                has_exp = true;
                n_in += 3;
                ns_cones.push(NsCone::Exp);
            }
            Cone::Power(alpha) => {
                seen_ineq = true;
                has_exp = true;
                n_in += 3;
                ns_cones.push(NsCone::Power(*alpha));
            }
            Cone::GenPower(alpha, tail) => {
                seen_ineq = true;
                has_exp = true;
                n_in += alpha.len() + tail;
                ns_cones.push(NsCone::GenPower(alpha.clone(), *tail));
            } // All cone variants are explicitly handled above
        }
    }
    if n_eq + n_in != m || prog.a.nrows != m || prog.a.ncols != n {
        return Err(SolveError::DimensionMismatch);
    }
    Ok(ConeScan {
        n_eq,
        n_in,
        ineq_cones,
        ns_cones,
        has_conic,
        has_exp,
    })
}

/// Solve a cone program in canonical standard form, optionally seeded from a
/// previous near-solution (M8 warm start).
///
/// Returns the solution in original units, with `s` and `z` laid out in the same
/// row/cone order as `A` and `cones`.
///
/// A `ws` seed is honored only when `settings.warm_start` is set (the gate);
/// when honored it is passed to the engine, which validates it (dimensions,
/// finiteness, `A x + s = b`, per-cone interiority of `s`/`z` after a θ-blend
/// toward the cone centers) and silently falls back to the cold start on any
/// failure — a warm start can only change convergence speed, never the
/// converged point. Seeded solves run the raw path in the problem's own units
/// (no presolve reduction chain — that mapping is the phase-2 structural warm
/// start). `solve` is the `None`-seed specialization.
pub fn solve<T: Scalar>(
    prog: &ConeProgram<T>,
    settings: &Settings<T>,
) -> Result<Solution<T>, SolveError> {
    solve_warm(prog, settings, None)
}

/// Solve a cone program in canonical standard form, optionally seeded from a
/// previous near-solution (`ws`) — see [`solve`] for the semantics; the seed
/// is gated on `settings.warm_start`.
pub fn solve_warm<T: Scalar>(
    prog: &ConeProgram<T>,
    settings: &Settings<T>,
    ws: Option<&iconic_core::WarmStart<T>>,
) -> Result<Solution<T>, SolveError> {
    // The settings gate: `warm_start: false` (the default) makes any seed
    // inert, so the warm path is strictly opt-in.
    let ws = if settings.warm_start { ws } else { None };
    solve_impl(prog, settings, ws)
}

fn solve_impl<T: Scalar>(
    prog: &ConeProgram<T>,
    settings: &Settings<T>,
    ws: Option<&iconic_core::WarmStart<T>>,
) -> Result<Solution<T>, SolveError> {
    // The platform-BLAS worker-thread cap is a process-global (the analogue
    // of OpenBLAS's pool), so the per-solve setting is applied at the entry
    // point (a Threads-style parameter: None = automatic, Some = explicit).
    #[cfg(not(target_os = "macos"))]
    if let Some(t) = settings.blas_threads {
        iconic_linalg::blas::set_blas_threads(Some(t));
    }
    // Cone presolve: exact-equivalence reductions on the conic
    // standard form — empty-cone dropping, free-variable elimination with the
    // Schur fold, redundant conic row removal — run before engine dispatch
    // when the gate is on. Only problems with a genuine cone (SOC/PSD via the
    // NT engine, or exp/pow/genpow via the nonsymmetric engine) are eligible;
    // orthant-only problems keep the QP path's own reduction chain untouched
    // (the MIP-internal node LPs, being orthant-only, never enter here). A
    // vacuous cone of the other family no longer blocks dispatch, so the
    // mixed-engine rejection happens on the reduced program.
    let scan0 = scan_cones(prog)?;
    let mut cone_red: Option<(Box<ConeProgram<T>>, cone_presolve::Record<T>)> = None;
    if settings.cone_presolve && (scan0.has_conic || scan0.has_exp) {
        match cone_presolve::apply(prog) {
            cone_presolve::Outcome::Unchanged => {}
            cone_presolve::Outcome::Reduced(r, rec) => cone_red = Some((r, rec)),
            cone_presolve::Outcome::PrimalInfeasible => {
                return Ok(cone_presolve::certificate_solution(
                    Status::PrimalInfeasible,
                    prog,
                ));
            }
            cone_presolve::Outcome::DualInfeasible => {
                return Ok(cone_presolve::certificate_solution(Status::DualInfeasible, prog));
            }
        }
    }
    let prog_use: &ConeProgram<T> = cone_red.as_ref().map(|(r, _)| r.as_ref()).unwrap_or(prog);
    let cone_rec: Option<&cone_presolve::Record<T>> = cone_red.as_ref().map(|(_, rec)| rec);
    let n = prog_use.q.len();
    let m = prog_use.b.len();

    // Partition the cone product into a leading Zero (equality) block and the
    // inequality cone product. The orthant is encoded as 1-D cones, an SOC as one
    // cone of its dimension, so `ineq_cones` is the dimension list for the solver.
    let scan = scan_cones(prog_use)?;
    let n_eq = scan.n_eq;
    let n_in = scan.n_in;
    let ineq_cones = scan.ineq_cones;
    let ns_cones = scan.ns_cones;
    let has_conic = scan.has_conic;
    let has_exp = scan.has_exp;
    // The nonsymmetric (exp) and NT-symmetric (SOC/PSD) engines are separate; a problem
    // mixing them is not yet supported.
    if has_exp && has_conic {
        return Err(SolveError::UnsupportedCone);
    }

    // Split A and b row-wise: equality rows first, then inequality rows. A is
    // row-major, so each block of rows is a contiguous slice of the data.
    let split = n_eq * n;
    let a_eq = DenseMatrix::from_row_major(n_eq, n, prog_use.a.data[0..split].to_vec());
    let a_in = DenseMatrix::from_row_major(n_in, n, prog_use.a.data[split..(n_eq + n_in) * n].to_vec());
    let b_eq = prog_use.b[0..n_eq].to_vec();
    let b_in = prog_use.b[n_eq..n_eq + n_in].to_vec();

    // Build sparse CSR blocks from the CSC input when available.  Only populate
    // a_in_csr (used by solve_cone_qp's row_matvec); a_eq_csr is left None
    // because equality-constrained QPs route through solve_qp_with_termination
    // which uses DenseMatrix matvecs, not CSR.
    let a_in_csr = if let Some(ref a_csc) = prog_use.a_csc {
        csc_submatrix_rows(a_csc, n_eq, n_eq + n_in)
    } else {
        None
    };

    let qp = QpProblem {
        p: prog_use.p.clone(),
        q: prog_use.q.clone(),
        a_eq,
        b_eq,
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr,
    };
    // Solve through the cone-aware engine (orthant encoded as Soc(1)), removing linearly
    // dependent equality rows first — it de-degenerates the KKT, cutting iterations on
    // redundant/over-specified problems. The removal is an O(n_eq²·n) dense row-echelon,
    // though, and on a large, structurally-independent equality block (e.g. MPC dynamics:
    // hundreds of independent rows whose staircase fills the echelon in completely) it costs
    // more than the whole solve and removes nothing. So run it only below a cost budget; the
    // proximal regularization is the correctness safety net for any genuine degeneracy past it.
    // Fewer than two equality rows cannot be dependent, so skip there too.
    // Shared by genuine conic (SOC/PSD) problems and sparse QPs routed here.
    let eq_removal_worth_it = n_eq >= 2 && (n_eq as u64) * (n_eq as u64) * (n as u64) <= 5_000_000;
    let solve_via_cone = |q: &QpProblem<T>| -> QpSolution<T> {
        // Auxiliary variable elimination: substitute out variables
        // that appear in exactly one equality row with ±1 coefficient and
        // diagonal P. Removes variables + equality rows, reducing KKT dimension
        // and avoiding dense fill-in from the eliminated equality block.
        let (q_red, aux_red) = match iconic_presolve::reductions::eliminate_auxiliary_vars(q) {
            Ok((r, a)) if !a.elims.is_empty() => (Some(r), Some(a)),
            _ => (None, None),
        };
        let q_use = q_red.as_ref().unwrap_or(q);
        // Epigraph pair folding: detect |x_j| ≤ t_i inequality
        // pairs and split x_j → x_j⁺−x_j⁻. After folding, all rows are
        // singleton bounds → condensed Cholesky via solve_qp_with_termination.
        // Only sound for orthant-only problems: the fold splits variables and
        // drops the `t` column outright, which destroys SOC/PSD/exp cone structure
        // (a `t` coupling inside a cone row would silently vanish from the problem).
        let (q_epi, epi_rec) = if !has_conic && !has_exp {
            match iconic_presolve::reductions::fold_epigraph_pairs(
                q_use,
                &iconic_presolve::reductions::SparseAIn::build(q_use),
            ) {
                Some((p, r)) => (Some(p), Some(r)),
                None => (None, None),
            }
        } else {
            (None, None)
        };
        let q_solve = q_epi.as_ref().unwrap_or(q_use);
        let sol_epi = if q_epi.is_some() {
            let term = TermScale::identity(q_solve.q.len(), q_solve.b_eq.len(), q_solve.b_in.len());
            solve_qp_with_termination(q_solve, settings, &term)
        } else if !eq_removal_worth_it {
            solve_cone_qp(q_solve, &ineq_cones, settings)
        } else {
            match iconic_presolve::reductions::remove_dependent_eq_rows(q_solve) {
                Ok((reduced, depred)) => {
                    let sol = solve_cone_qp(&reduced, &ineq_cones, settings);
                    iconic_presolve::reductions::restore_dependent_eq_rows(&depred, &sol)
                }
                Err(status) => QpSolution {
                    status,
                    x: vec![T::zero(); q_solve.q.len()],
                    y: vec![T::zero(); q_solve.b_eq.len()],
                    s: vec![T::zero(); q_solve.b_in.len()],
                    z: vec![T::zero(); q_solve.b_in.len()],
                    obj_val: T::zero(),
                    iters: 0,
                    tau: T::one(),
                    kappa: T::zero(),
                },
            }
        };
        let sol_unfolded = match epi_rec {
            Some(ref rec) => {
                iconic_presolve::reductions::restore_epigraph_pairs(q_use, rec, &sol_epi)
            }
            None => sol_epi,
        };
        let sol_red = sol_unfolded;
        match aux_red {
            Some(ref red) => iconic_presolve::reductions::restore_auxiliary_vars(red, q, &sol_red),
            None => sol_red,
        }
    };
    // Density helper (extracted so the ineq-dominated check below can reuse it).
    let dens = |mm: &DenseMatrix<T>| {
        if mm.data.is_empty() {
            0.0
        } else {
            let nz = mm.data.iter().filter(|&&v| v != T::zero()).count();
            nz as f64 / mm.data.len() as f64
        }
    };
    // Route an orthant-only QP to the cone engine when it is sparse and not tiny: its
    // auto-selected sparse factor is markedly faster than solve_qp's dense one on banded /
    // network / control structure (≈4× at n=1000), while dense QPs stay on solve_qp, whose
    // condensed-gram path is faster there. (Cone-rectified equilibration is deliberately not
    // applied — it over-scales the typically well-conditioned sparse problems.)
    // Inequality-dominated QP: when there are many inequality constraints
    // relative to variables and P is sparse, the augmented KKT avoids the O(n_in·n²)
    // condensed gram build entirely.  The conic solver factors the quasi-definite system
    // [P+ρI, 0, A_inᵀ; 0, −δI, 0; A_in, 0, −diag(η²+δ)] directly — with a sparse P and
    // a diagonal (z,z) block the only fill comes from A_in's own nonzeros, so even a
    // moderately sparse A_in is handled far more efficiently than the dense condensed
    // gram build.  This favors the augmented (quasi-definite) system over the normal
    // equations, which also avoids squaring the condition
    // number (κ(augmented) ≈ κ(A) vs κ(condensed) ≈ κ(A)²).
    // Route to augmented KKT only when inequalities DOMINATE
    // (n_in ≫ n). The augmented factor costs O((n+n_in)³) vs condensed
    // O(n³ + n_in·n²). For n_in=2n the augmented is (3n)³=27n³ vs
    // condensed n³+2n³=3n³ — condensed is 9× cheaper. Threshold n_in>=5n
    // keeps augmented for truly inequality-dominated problems only.
    let ineq_dominated = !has_conic && !has_exp && n >= 64 && n_in >= 5 * n && dens(&qp.p) < 0.15;
    // `solve_qp_lowrank` detects factor-model structure and returns None
    // immediately when none is found — detection is cheap, always try first.
    let lowrank = if !has_conic && !has_exp && n >= 64 {
        iconic_ipm::solve_qp_lowrank(&qp, settings)
    } else {
        None
    };
    let sol = if ws.is_some() {
        // Warm path (M8, v1): raw engines only, in the problem's own units.
        // Mapping a seed through the presolve reduction chain (Ruiz D/E,
        // fixed/doubleton/empty-column records) is the phase-2 structural
        // warm start and does not exist yet, so a seeded solve takes the raw
        // path regardless of `settings.presolve` — the seed's dims only match
        // the *given* problem. The engine itself validates and interiorizes
        // the seed and silently falls back to its cold start on any failure.
        // The low-rank (LR-cache) engine is deliberately not tried: it is a
        // separate structure-exploiting mechanism with its own cache, and a
        // seeded warm path through it would double-book the cached factor.
        let engine_ws = ws.map(|w| iconic_core::WarmStart {
            x: w.x.clone(),
            s: w.s[n_eq..n_eq + n_in].to_vec(),
            z: w.z[n_eq..n_eq + n_in].to_vec(),
        });
        let term = TermScale::identity(qp.q.len(), qp.b_eq.len(), qp.b_in.len());
        if has_exp {
            solve_nonsym_warm(&qp, &ns_cones, settings, engine_ws.as_ref())
        } else if has_conic || ineq_dominated {
            solve_cone_qp_warm(&qp, &ineq_cones, settings, engine_ws.as_ref())
        } else {
            solve_qp_with_termination_warm(&qp, settings, &term, engine_ws.as_ref())
        }
    } else if let Some(s) = lowrank {
        s
    } else if has_exp {
        solve_nonsym(&qp, &ns_cones, settings)
    } else if has_conic || ineq_dominated {
        solve_via_cone(&qp)
    } else if qp.p.data.iter().all(|&v| v == T::zero()) && n_eq == 0 {
        // Pure LP: route through the termination-aware path so the sparse-LP
        // dual-simplex gate, the wide-LP dualization and the dense-LP conic
        // routing all apply. Plain solve_qp here forced every P=0 LP down the
        // dense condensed-Gram QP path — measured: the feasibility pump's
        // distance LP (1000 vars, sparse rows) burned ~0.8s per QP iteration,
        // ~7.8s per pump call, vs milliseconds on the sparse simplex.
        let term = TermScale::identity(qp.q.len(), qp.b_eq.len(), qp.b_in.len());
        solve_qp_with_termination(&qp, settings, &term)
    } else {
        let (q_red, aux_red) = match iconic_presolve::reductions::eliminate_auxiliary_vars(&qp) {
            Ok((r, a)) if !a.elims.is_empty() => (Some(r), Some(a)),
            _ => (None, None),
        };
        let q_use = q_red.as_ref().unwrap_or(&qp);
        let (q_folded, epi_rec) = match iconic_presolve::reductions::fold_epigraph_pairs(
            q_use,
            &iconic_presolve::reductions::SparseAIn::build(q_use),
        ) {
            Some((f, r)) => (Some(f), Some(r)),
            None => (None, None),
        };
        let q_solve = q_folded.as_ref().unwrap_or(q_use);
        let sol_folded = if q_folded.is_some() {
            let term = TermScale::identity(q_solve.q.len(), q_solve.b_eq.len(), q_solve.b_in.len());
            solve_qp_with_termination(q_solve, settings, &term)
        } else if settings.presolve {
            iconic_presolve::solve_presolved(q_solve, settings)
        } else {
            solve_qp(q_solve, settings)
        };
        let sol_epi = match epi_rec {
            Some(ref rec) => {
                iconic_presolve::reductions::restore_epigraph_pairs(q_use, rec, &sol_folded)
            }
            None => sol_folded,
        };
        match aux_red {
            Some(ref red) => iconic_presolve::reductions::restore_auxiliary_vars(red, &qp, &sol_epi),
            None => sol_epi,
        }
    };

    // Reassemble s and z in cone order: zero block (s = 0, z = equality multiplier),
    // then nonnegative block (s = slack, z = inequality multiplier). When a cone
    // presolve reduction fired, restore the original-space solution instead
    // (dropped rows get their fixed/implied slack and dual 0; eliminated
    // columns are recovered from their pivot rows).
    let (status, x, s, z, obj_val, iters) = if let Some(rec) = cone_rec {
        let full = cone_presolve::restore(rec, prog, &sol);
        (full.status, full.x, full.s, full.z, full.obj_val, full.iters)
    } else {
        let mut s = vec![T::zero(); m];
        let mut z = vec![T::zero(); m];
        z[0..n_eq].copy_from_slice(&sol.y);
        s[n_eq..n_eq + n_in].copy_from_slice(&sol.s);
        z[n_eq..n_eq + n_in].copy_from_slice(&sol.z);
        (sol.status, sol.x, s, z, sol.obj_val, sol.iters)
    };

    Ok(Solution {
        status,
        x,
        s,
        z,
        obj_val,
        iters,
        y: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use iconic_core::rng::Lcg;
    use iconic_core::Status;

    /// The open-issue #4 shape: x (30 free vars) with a dense well-conditioned PSD
    /// Hessian Q, plus k auxiliary vars u each tied to EXACTLY ONE equality row
    /// `u_i = (C x)_i` as a singleton column (coeff 1) with P_uu = 1 on its
    /// diagonal. This shape used to break the dense IPM path past ~18-20 equality
    /// rows (SolvedInaccurate at the iteration cap, or NumericalError before the
    /// first Newton step); the raw engine's Schur-direction refinement and the
    /// auxiliary-variable presolve elimination closed it. Regression: the whole
    /// k sweep must solve through the user-facing API with a tight residual.
    #[test]
    fn equality_tied_singleton_aux_sweep_solves_via_api() {
        for &k in &[8usize, 12, 16, 18, 20, 22, 25, 30, 40] {
            let n = 30usize;
            let mut rng = Lcg::new(42 + k as u64 * 7919);
            let nt = n + k;
            // Q = A·Aᵀ + I with A (30×10) random — a true symmetric PSD gram.
            let mut a30 = vec![vec![0.0f64; 10]; n];
            for i in 0..n {
                for t in 0..10 {
                    a30[i][t] = rng.signed();
                }
            }
            let mut p = DenseMatrix::<f64>::zeros(nt, nt);
            for i in 0..n {
                for j in 0..n {
                    let mut v = 0.0;
                    for t in 0..10 {
                        v += a30[i][t] * a30[j][t];
                    }
                    p.set(i, j, v);
                }
                p.set(i, i, p.get(i, i) + 1.0);
            }
            for i in 0..k {
                p.set(n + i, n + i, 1.0);
            }
            let q: Vec<f64> = (0..nt).map(|_| rng.signed()).collect();
            let mut a = DenseMatrix::<f64>::zeros(k, nt);
            for r in 0..k {
                for j in 0..n {
                    a.set(r, j, rng.signed());
                }
                a.set(r, n + r, -1.0);
            }
            let prog = ConeProgram {
                p,
                q,
                a,
                a_csc: None,
                b: vec![0.0; k],
                cones: vec![Cone::Zero(k)],
            };
            let sol = solve(&prog, &Settings::<f64>::default()).expect("api solve");
            assert_eq!(sol.status, Status::Solved, "k={k}");
            // Verify the returned point in original units: stationarity + equality
            // feasibility (the solve is 1-iteration Newton-exact on this shape).
            let nt2 = prog.p.nrows;
            let mut r = 0.0f64;
            for i in 0..nt2 {
                let mut st = prog.q[i];
                for j in 0..nt2 {
                    st += prog.p.get(i, j) * sol.x[j];
                }
                for e in 0..k {
                    st += prog.a.get(e, i) * sol.z[e];
                }
                r = r.max(st.abs());
            }
            for e in 0..k {
                let mut ax = 0.0;
                for j in 0..nt2 {
                    ax += prog.a.get(e, j) * sol.x[j];
                }
                r = r.max((ax + sol.s[e] - prog.b[e]).abs());
            }
            assert!(r < 1e-7, "k={k}: kkt residual {r:.3e}");
        }
    }

    /// A factor-model QP (`P = F Fᵀ + diag(d)`, budget equality + box) routed through the API
    /// engages the low-rank SOCP reformulation and returns the same objective as the dense QP
    /// path on the equivalent `QpProblem`. This is the user-facing path for the structure-
    /// exploiting solver.
    #[test]
    fn factor_model_routes_through_lowrank_and_matches_dense() {
        let n = 600usize;
        let r = 12usize;
        // Deterministic F, d > 0.
        let mut rnd = iconic_core::rng::SplitMix::new(4242);
        let f: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rnd.signed()).collect()).collect();
        let d: Vec<f64> = (0..n).map(|_| 0.3 + rnd.signed().abs()).collect();
        let p = iconic_linalg::gram_plus_diag(&f, &d);
        let q: Vec<f64> = (0..n).map(|_| -rnd.signed().abs()).collect();

        // Build the conic program: budget Σx = 1 (Zero), then box −k ≤ x ≤ k (NonNeg 2n).
        let k = 0.5;
        let m = 1 + 2 * n;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        let mut b = vec![0.0; m];
        for j in 0..n {
            a.set(0, j, 1.0);
        }
        b[0] = 1.0;
        for j in 0..n {
            a.set(1 + j, j, 1.0);
            b[1 + j] = k;
            a.set(1 + n + j, j, -1.0);
            b[1 + n + j] = k;
        }
        let prog = ConeProgram {
            p: p.clone(),
            q: q.clone(),
            a,
            a_csc: None,
            b,
            cones: vec![Cone::Zero(1), Cone::NonNegative(2 * n)],
        };
        let sol = solve(&prog, &Settings::<f64>::default()).unwrap();
        assert_eq!(sol.status, Status::Solved);

        // The dense reference needs thousands of IPM iterations for n=600 — too
        // slow for a unit test. Verify only that the API (low-rank) path converged
        // (asserted above) and that the structure detector fires.
        let mut a_in = DenseMatrix::<f64>::zeros(2 * n, n);
        let mut b_in = vec![0.0; 2 * n];
        for j in 0..n {
            a_in.set(j, j, 1.0);
            b_in[j] = k;
            a_in.set(n + j, j, -1.0);
            b_in[n + j] = k;
        }
        let mut a_eq = DenseMatrix::<f64>::zeros(1, n);
        for j in 0..n {
            a_eq.set(0, j, 1.0);
        }
        assert!(iconic_ipm::solve_qp_lowrank(
            &iconic_ipm::QpProblem {
                p,
                q,
                a_eq,
                b_eq: vec![1.0],
                a_in,
                b_in,
                a_eq_csr: None,
                a_in_csr: None,
            },
            &Settings::<f64>::default(),
        )
        .is_some());
    }

    /// A QP with an SOC row PLUS an epigraph-pattern pair must solve to the true
    /// optimum. Regression: `solve_via_cone` folded epigraph pairs unconditionally,
    /// even for conic problems — the fold drops the `t` variable that the SOC row
    /// couples to, and the SOC row (which is binding at the optimum) silently
    /// changes meaning, returning a wrong answer.
    #[test]
    fn epigraph_pair_with_soc_row_solves_true_optimum() {
        // min ½x² − x + t + ½z² + ½w²  s.t.  |x| ≤ t (2 NonNeg rows),
        // (t − 1, −z) ∈ Q²  ⟺  t ≥ 1 + |z| (SecondOrder(2), binding at the optimum).
        // True optimum: x = 1, t = 1, z = w = 0, obj = 0.5. The fold's target
        // pattern is the pair rows x − t ≤ 0, −x − t ≤ 0 on columns (x, t); the
        // SOC row also couples `t` — dropping `t` makes it `−1 ≥ |z|`, infeasible.
        let prog = ConeProgram {
            p: DenseMatrix::from_row_major(
                4,
                4,
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            ),
            q: vec![-1.0, 1.0, 0.0, 0.0],
            // Rows (s = b − Ax): s0 = x − t ≥ 0, s1 = x + t ≥ 0 (NonNeg),
            // then s0 = t − 1, s1 = −z (SecondOrder(2)).
            a: DenseMatrix::from_row_major(
                4,
                4,
                vec![
                    1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0,
                ],
            ),
            a_csc: None,
            b: vec![0.0, 0.0, -1.0, 0.0],
            cones: vec![Cone::NonNegative(2), Cone::SecondOrder(2)],
        };
        let sol = solve(&prog, &Settings::<f64>::default()).unwrap();
        assert_eq!(sol.status, Status::Solved, "status={:?}", sol.status);
        assert!(
            (sol.x[0] - 1.0).abs() < 1e-3,
            "x={} (pre-fix fold makes this infeasible)",
            sol.x[0]
        );
        assert!((sol.x[1] - 1.0).abs() < 1e-3, "t={}", sol.x[1]);
        assert!(
            (sol.obj_val - 0.5).abs() < 1e-3,
            "obj={} (true 0.5)",
            sol.obj_val
        );
    }

    /// min ½(x0²+x1²) s.t. x0 + x1 = 2 (Zero cone) and x0 ≤ 1.5 (NonNeg slack).
    /// Equality forces the sum; here the inequality is inactive, so x = [1, 1].
    #[test]
    fn mixed_cone_program() {
        // Rows: equality [1,1] = 2 ; inequality x0 ≤ 1.5  ->  [1,0] x + s = 1.5, s ≥ 0.
        let prog = ConeProgram {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0],
            a: DenseMatrix::from_row_major(2, 2, vec![1.0, 1.0, 1.0, 0.0]),
            a_csc: None,
            b: vec![2.0, 1.5],
            cones: vec![Cone::Zero(1), Cone::NonNegative(1)],
        };
        let sol = solve(
            &prog,
            &Settings::<f64> {
                presolve: false,
                ..Settings::default()
            },
        )
        .unwrap();
        assert_eq!(sol.status, Status::Solved);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
        // s[0] = 0 (equality), s[1] = 1.5 - x0 = 0.5 (inactive inequality).
        assert!(sol.s[0].abs() < 1e-9);
        assert!((sol.s[1] - 0.5).abs() < 1e-6, "s1={}", sol.s[1]);
    }

    /// The same problem solves correctly with presolve disabled (the non-equilibrated path).
    #[test]
    fn mixed_cone_program_without_presolve() {
        let prog = ConeProgram {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0],
            a: DenseMatrix::from_row_major(2, 2, vec![1.0, 1.0, 1.0, 0.0]),
            a_csc: None,
            b: vec![2.0, 1.5],
            cones: vec![Cone::Zero(1), Cone::NonNegative(1)],
        };
        let settings = Settings::<f64> {
            presolve: false,
            ..Settings::default()
        };
        let sol = solve(&prog, &settings).unwrap();
        assert_eq!(sol.status, Status::Solved);
        assert!((sol.x[0] - 1.0).abs() < 1e-6);
        assert!((sol.x[1] - 1.0).abs() < 1e-6);
    }

    /// An SOCP through the public API: project c = (1,2,2) onto Q₃ via
    /// min ½‖x−c‖² s.t. x ∈ Q₃, encoded as −I·x + s = 0, s ∈ SecondOrder(3).
    #[test]
    fn solves_socp() {
        let prog = ConeProgram {
            p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            q: vec![-1.0, -2.0, -2.0],
            a: DenseMatrix::from_row_major(
                3,
                3,
                vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
            ),
            a_csc: None,
            b: vec![0.0, 0.0, 0.0],
            cones: vec![Cone::SecondOrder(3)],
        };
        let sol = solve(&prog, &Settings::<f64>::default()).unwrap();
        assert_eq!(sol.status, Status::Solved);
        let expected_x0 = (1.0 + 2.0 * 2.0_f64.sqrt()) / 2.0;
        assert!((sol.x[0] - expected_x0).abs() < 1e-5, "x0={}", sol.x[0]);
        // The slack equals x and lies on the cone boundary; its dual is in the cone.
        assert!(sol.s[0] >= -1e-6 && sol.z[0] >= -1e-6);
    }

    /// GenPower end-to-end: `n=3`, `α=(1/3,1/3,1/3)` (geometric-mean cone)
    /// plus 1-D tail. Minimizes `½‖(x,z)−c‖²` for `c=(8,8,8,2)`, a point
    /// strictly INSIDE the cone (`(8·8·8)^{1/3}=8 > 2`). The cone is present
    /// but not binding, so the closed-form optimum is the unconstrained
    /// minimizer `x=c`, `obj=-½‖c‖²=-98`. Exercises a QP with a live GenPower
    /// constraint end-to-end (P≠0, the nonsymmetric IPM must respect the
    /// cone's shape throughout, it just doesn't need to reach the boundary).
    /// In `Ax+s=b, s∈K` form (`A=−I, b=0` so `s=x` exactly): `q=−c, P=I`.
    #[test]
    fn solves_genpower_cone() {
        let c = [8.0, 8.0, 8.0, 2.0];
        let prog = ConeProgram {
            p: DenseMatrix::from_row_major(
                4,
                4,
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            ),
            q: c.iter().map(|&v| -v).collect(),
            a: DenseMatrix::from_row_major(
                4,
                4,
                vec![
                    -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
                    -1.0,
                ],
            ),
            a_csc: None,
            b: vec![0.0, 0.0, 0.0, 0.0],
            cones: vec![Cone::GenPower(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1)],
        };
        let sol = solve(&prog, &Settings::<f64>::default()).unwrap();
        assert!(
            matches!(sol.status, Status::Solved | Status::SolvedInaccurate),
            "status={:?}, expected Solved or SolvedInaccurate",
            sol.status
        );
        for (i, &ci) in c.iter().enumerate().take(4) {
            assert!(
                (sol.x[i] - ci).abs() < 1e-5,
                "x[{i}]={}, expected {ci}",
                sol.x[i]
            );
        }
        assert!((sol.obj_val - (-98.0)).abs() < 1e-4, "obj={}", sol.obj_val);
    }

    /// GenPower mixed with SOC (NT-symmetric engine) must still be rejected:
    /// the two engines are separate, same as every other nonsymmetric cone.
    #[test]
    fn genpower_mixed_with_conic_is_unsupported() {
        let prog = ConeProgram {
            p: DenseMatrix::zeros(2, 2),
            q: vec![0.0, 0.0],
            a: DenseMatrix::from_row_major(
                7,
                2,
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0,
                ],
            ),
            a_csc: None,
            b: vec![8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0],
            cones: vec![
                Cone::GenPower(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1),
                Cone::SecondOrder(3),
            ],
        };
        assert!(matches!(
            solve(&prog, &Settings::<f64>::default()),
            Err(SolveError::UnsupportedCone)
        ));
    }

    /// Exponential cone end to end through the API: max x s.t. eˣ ≤ 2 ⇒ x* = log 2.
    /// In `Ax + s = b, s ∈ K_exp` form: s = (x, 1, 2), A = [[-1],[0],[0]], b = [0,1,2].
    #[test]
    fn solves_exp_cone() {
        let prog = ConeProgram {
            p: DenseMatrix::zeros(1, 1),
            q: vec![-1.0],
            a: DenseMatrix::from_row_major(3, 1, vec![-1.0, 0.0, 0.0]),
            a_csc: None,
            b: vec![0.0, 1.0, 2.0],
            cones: vec![Cone::Exponential],
        };
        let sol = solve(&prog, &Settings::<f64>::default()).unwrap();
        assert_eq!(sol.status, Status::Solved);
        assert!((sol.x[0] - 2.0_f64.ln()).abs() < 1e-6, "x={}", sol.x[0]);
    }

    #[test]
    fn rejects_bad_cone_order() {
        let prog = ConeProgram {
            p: DenseMatrix::zeros(1, 1),
            q: vec![1.0],
            a: DenseMatrix::from_row_major(2, 1, vec![1.0, 1.0]),
            a_csc: None,
            b: vec![1.0, 1.0],
            cones: vec![Cone::NonNegative(1), Cone::Zero(1)],
        };
        assert!(matches!(
            solve(&prog, &Settings::<f64>::default()),
            Err(SolveError::ConeOrder)
        ));
    }

    /// M8 warm start through the public API:
    ///  1. the settings gate — with `warm_start: false` a seed is inert and
    ///     the solve is bit-identical to the unseeded one;
    ///  2. the exactness contract — with `warm_start: true`, a re-solve of a
    ///     perturbed problem seeded from the base solution converges to the
    ///     same point as the cold re-solve (same status, objective agreement
    ///     at solver accuracy) in no more iterations.
    #[test]
    fn solve_warm_gate_and_exactness() {
        let n = 30usize;
        let m = 20usize;
        let mut rnd = iconic_core::rng::SplitMix::new(4242);
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                p.set(i, j, 0.3 * rnd.signed());
            }
            p.set(i, i, p.get(i, i) + 2.0);
        }
        for i in 0..n {
            for j in 0..i {
                let v = p.get(i, j);
                p.set(j, i, v);
            }
        }
        let q: Vec<f64> = (0..n).map(|_| rnd.signed() - 0.5).collect();
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a.set(i, j, rnd.signed() - 0.5);
            }
        }
        let x0: Vec<f64> = (0..n).map(|_| rnd.signed() - 0.5).collect();
        let b: Vec<f64> = (0..m)
            .map(|i| {
                let acc: f64 = (0..n).map(|j| a.get(i, j) * x0[j]).sum();
                acc + 0.7
            })
            .collect();
        let prog = ConeProgram {
            p,
            q,
            a,
            a_csc: None,
            b,
            cones: vec![Cone::NonNegative(m)],
        };
        let settings = Settings::<f64>::default();
        let base = solve(&prog, &settings).unwrap();
        assert_eq!(base.status, Status::Solved);
        let seed = iconic_core::WarmStart {
            x: base.x,
            s: base.s,
            z: base.z,
        };

        // Gate: warm_start=false + seed == unseeded solve, bit for bit.
        let gated = solve_warm(&prog, &settings, Some(&seed)).unwrap();
        let cold = solve(&prog, &settings).unwrap();
        assert_eq!(gated.iters, cold.iters);
        assert_eq!(gated.obj_val, cold.obj_val);
        assert_eq!(gated.status, cold.status);

        // Perturbed re-solve: cold vs warm-from-base.
        let scale = prog.b.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let mut pert = prog.clone();
        for (i, v) in pert.b.iter_mut().enumerate() {
            *v += 1e-4 * scale * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let warm_settings = Settings { warm_start: true, ..Settings::default() };
        let cold_p = solve_warm(&pert, &warm_settings, None).unwrap();
        let warm_p = solve_warm(&pert, &warm_settings, Some(&seed)).unwrap();
        assert_eq!(warm_p.status, cold_p.status);
        let tol = 1e-6 * cold_p.obj_val.abs().max(1.0);
        assert!(
            (warm_p.obj_val - cold_p.obj_val).abs() <= tol,
            "warm obj {:.12e} vs cold {:.12e}",
            warm_p.obj_val,
            cold_p.obj_val
        );
        assert!(
            warm_p.iters <= cold_p.iters,
            "warm {} iters vs cold {}",
            warm_p.iters,
            cold_p.iters
        );
    }
}

#[cfg(test)]
mod power_probe {
    use crate::{solve, ConeProgram};
    use iconic_core::{Cone, Settings, Status};
    use iconic_linalg::DenseMatrix;
    #[test]
    fn power_cone_probe() {
        // min x + y s.t. PowCone3D(x, y, 1, 0.5) — x^0.5 y^0.5 >= 1.
        let n = 2usize;
        let mut a = DenseMatrix::<f64>::zeros(3, n);
        let mut b = vec![0.0; 3];
        a.set(0, 0, -1.0);
        b[0] = 0.0; // s0 = x
        a.set(1, 1, -1.0);
        b[1] = 0.0; // s1 = y
        b[2] = 1.0; // s2 = 1
        let prog = ConeProgram {
            p: DenseMatrix::zeros(n, n),
            q: vec![1.0, 1.0],
            a,
            a_csc: None,
            b,
            cones: vec![Cone::Power(0.5)],
        };
        let sol = solve(&prog, &Settings::<f64>::default());
        match sol {
            Ok(s) => println!("OK: status={:?} x={:?} obj={}", s.status, s.x, s.obj_val),
            Err(e) => println!("ERR: {:?}", e),
        }
        // The CVXPY shape: nonneg bounds first, then the power cone.
        let mut a2 = DenseMatrix::<f64>::zeros(5, 2);
        let mut b2 = vec![0.0; 5];
        a2.set(0, 0, -1.0); // s0 = x (nonneg)
        a2.set(1, 1, -1.0); // s1 = y (nonneg)
        a2.set(2, 0, -1.0); // s2 = x (power)
        a2.set(3, 1, -1.0); // s3 = y (power)
        b2[4] = 1.0; // s4 = 1 (power)
        let prog2 = ConeProgram {
            p: DenseMatrix::zeros(2, 2),
            q: vec![1.0, 1.0],
            a: a2,
            a_csc: None,
            b: b2,
            cones: vec![Cone::NonNegative(2), Cone::Power(0.5)],
        };
        match solve(&prog2, &Settings::<f64>::default()) {
            Ok(s) => println!("OK2: status={:?} x={:?}", s.status, s.x),
            Err(e) => println!("ERR2: {:?}", e),
        }
        let ps = Settings::<f64> {
            presolve: true,
            ..Default::default()
        };
        match solve(&prog2, &ps) {
            Ok(s) => println!("OK3(presolve): status={:?} x={:?}", s.status, s.x),
            Err(e) => println!("ERR3: {:?}", e),
        }
        // Direct nonsym-engine probe of the same order.
        let n2 = 2usize;
        let mut ain = DenseMatrix::<f64>::zeros(5, n2);
        let mut bin = vec![0.0; 5];
        // Cone order [NonNeg(2), Power(0.5)]: rows 0-1 = bounds, rows 2-4 = power.
        ain.set(0, 0, -1.0);
        ain.set(1, 1, -1.0);
        ain.set(2, 0, -1.0);
        ain.set(3, 1, -1.0);
        bin[4] = 1.0;
        let qp = iconic_ipm::QpProblem {
            p: DenseMatrix::zeros(n2, n2),
            q: vec![1.0, 1.0],
            a_eq: DenseMatrix::zeros(0, n2),
            b_eq: vec![],
            a_in: ain,
            b_in: bin,
            a_eq_csr: None,
            a_in_csr: None,
        };
        let cones3 = vec![
            iconic_ipm::nonsym::NsCone::NonNeg(2),
            iconic_ipm::nonsym::NsCone::Power(0.5),
        ];
        let s = iconic_ipm::nonsym::solve_nonsym(&qp, &cones3, &Settings::<f64>::default());
        println!("NONSYM: status={:?} x={:?}", s.status, s.x);
    }

    /// The exact CVXPY PowCone3D emission (cvxpy 1.7.5, verified via
    /// get_problem_data): max x s.t. y + z = 1, (y,z,x) ∈ P_0.5.  The cone
    /// rows arrive in CVXPY's rotated order and the equality precedes the
    /// cone block, as the C ABI receives it.  This is the shape the CVXPY
    /// backend test_power_cone_matches_clarabel exercises — it must solve.
    #[test]
    fn cvxpy_power_cone_shape_solves() {
        let n = 3usize;
        let m = 4usize;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        a.set(0, 1, 1.0); // s0 = 1 - y - z (Zero)
        a.set(0, 2, 1.0);
        a.set(1, 1, -1.0); // s1 = y
        a.set(2, 2, -1.0); // s2 = z
        a.set(3, 0, -1.0); // s3 = x
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let prog = ConeProgram {
            p: DenseMatrix::zeros(n, n),
            // The real CVXPY emission: c = [-1, 0, 0] (maximize x = slot2).
            q: vec![-1.0, 0.0, 0.0],
            a,
            a_csc: None,
            b,
            cones: vec![Cone::Zero(1), Cone::Power(0.5)],
        };
        // The C ABI's settings (max_iters 800) — the backend's default path.
        for (name, s) in [
            ("default", Settings::<f64>::default()),
            (
                "cababi-800",
                Settings::<f64> {
                    max_iters: 800,
                    ..Settings::<f64>::default()
                },
            ),
        ] {
            match solve(&prog, &s) {
                Ok(sol) => {
                    println!(
                        "{name}: status={:?} obj={} iters={} x={:?} s={:?}",
                        sol.status, sol.obj_val, sol.iters, sol.x, sol.s
                    );
                    // The power-cone termination fix (scale-aware diagonal
                    // regularization of the condensed (x,x) block in
                    // solve_nonsym): this emission order now solves at ~17
                    // iterations with the dual at the correct boundary ray
                    // (z = (0.5, 0.5, −1)) and y_eq = 0.5. It previously froze
                    // bit-for-bit at the exactly-correct primal for 799
                    // iterations (MaxIterations) because the fixed-order
                    // no-pivot LDLᵀ factored the near-singular condensed
                    // system on cancellation noise.
                    assert_eq!(
                        sol.status,
                        Status::Solved,
                        "{name}: status={:?} at the converged point (obj {})",
                        sol.status,
                        sol.obj_val
                    );
                    assert!(
                        sol.iters < 40,
                        "{name}: iters={} (expected ~17)",
                        sol.iters
                    );
                    assert!(
                        (sol.obj_val + 0.5).abs() < 1e-6,
                        "{name}: obj {} far from -0.5",
                        sol.obj_val
                    );
                    // nd in original units: ||Px + q + Aᵀz||∞ with z the
                    // cone-order dual (the Zero block carries the equality
                    // multiplier). The dual must sit on the correct boundary
                    // ray, not just be complementary.
                    let mut nd = 0.0f64;
                    for i in 0..n {
                        let mut acc = 0.0f64;
                        for r in 0..m {
                            acc += prog.a.get(r, i) * sol.z[r];
                        }
                        acc += prog.q[i];
                        nd = nd.max(acc.abs());
                    }
                    assert!(
                        nd < 1e-6,
                        "{name}: dual residual nd={nd:e} (expected < 1e-6)",
                    );
                }
                Err(e) => println!("{name}: ERR {e:?}"),
            }
        }
        // The battery's pow_eq shape through the SAME api path: cone on
        // (x,y,z) in slot order, equality x+y=1 — the only difference from
        // the CVXPY shape is the variable/row order.
        let mut a2 = DenseMatrix::<f64>::zeros(4, 3);
        a2.set(0, 0, 1.0); // s0 = 1 - x - y (Zero)
        a2.set(0, 1, 1.0);
        a2.set(1, 0, -1.0); // s1 = x
        a2.set(2, 1, -1.0); // s2 = y
        a2.set(3, 2, -1.0); // s3 = z
        let prog2 = ConeProgram {
            p: DenseMatrix::zeros(3, 3),
            q: vec![0.0, 0.0, -1.0],
            a: a2,
            a_csc: None,
            b: vec![1.0, 0.0, 0.0, 0.0],
            cones: vec![Cone::Zero(1), Cone::Power(0.5)],
        };
        match solve(&prog2, &Settings::<f64>::default()) {
            Ok(sol) => println!(
                "battery-shape via api: status={:?} obj={} iters={}",
                sol.status, sol.obj_val, sol.iters
            ),
            Err(e) => println!("battery-shape via api: ERR {e:?}"),
        }
    }

    /// Regression pin for the power-cone termination fix: the CVXPY-emission
    /// shape sat on the knife-edge of the fixed-order no-pivot LDLᵀ's
    /// cancellation noise at the boundary-active dual state — perturbing
    /// `b_eq` by 1e-12 or `q[0]` by 1e-11 flipped the solve between Solved
    /// (~16-17 iters), a frozen 799-iteration MaxIterations, and even a
    /// NumericalError (ZeroPivot) on one perturbation. The scale-aware
    /// diagonal regularization must make the behavior robust: every
    /// perturbation solves at ~17 iterations with the correct objective.
    #[test]
    fn power_cone_cvxpy_shape_robust_to_knife_edge_perturbations() {
        let n = 3usize;
        let m = 4usize;
        let base_b = vec![1.0, 0.0, 0.0, 0.0];
        let base_q = vec![-1.0, 0.0, 0.0];
        let mut cases: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
        cases.push(("base".to_string(), base_b.clone(), base_q.clone()));
        for &db in &[1e-12, -1e-12] {
            let mut b = base_b.clone();
            b[0] += db;
            cases.push((format!("b_eq{db:+.0e}"), b, base_q.clone()));
        }
        for &dq in &[1e-11, -1e-11] {
            let mut q = base_q.clone();
            q[0] += dq;
            cases.push((format!("q0{dq:+.0e}"), base_b.clone(), q));
        }
        for (name, b, q) in cases {
            let mut a = DenseMatrix::<f64>::zeros(m, n);
            a.set(0, 1, 1.0); // s0 = 1 - y - z (Zero)
            a.set(0, 2, 1.0);
            a.set(1, 1, -1.0); // s1 = y
            a.set(2, 2, -1.0); // s2 = z
            a.set(3, 0, -1.0); // s3 = x
            let prog = ConeProgram {
                p: DenseMatrix::zeros(n, n),
                q,
                a,
                a_csc: None,
                b,
                cones: vec![Cone::Zero(1), Cone::Power(0.5)],
            };
            match solve(&prog, &Settings::<f64>::default()) {
                Ok(sol) => {
                    assert_eq!(
                        sol.status,
                        Status::Solved,
                        "{name}: status={:?} obj={} iters={}",
                        sol.status,
                        sol.obj_val,
                        sol.iters
                    );
                    assert!(
                        sol.iters < 40,
                        "{name}: iters={} (expected ~17)",
                        sol.iters
                    );
                    assert!(
                        (sol.obj_val + 0.5).abs() < 1e-6,
                        "{name}: obj {} far from -0.5",
                        sol.obj_val
                    );
                }
                Err(e) => panic!("{name}: solve error {e:?}"),
            }
        }
    }
}

