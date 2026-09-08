#![allow(clippy::type_complexity)]
//! `iconic-presolve` — problem equilibration and solution recovery.
//!
//! Milestone **M3** (initial slice). Poorly-scaled data makes the interior-point
//! KKT system ill-conditioned and slows or destabilizes convergence. Ruiz
//! equilibration rescales the variables and constraints so the rows and columns of
//! the data have comparable magnitude, then a cost-scaling factor normalizes the
//! objective. [`postsolve`] maps a solution of the scaled problem back to the
//! original units (primal, duals, slacks, and objective).
//!
//! Reduction passes (empty/singleton rows, fixed-variable elimination, redundant
//! constraints) extended with additional passes.

// Index-based loops read more clearly than iterator adapters for this linear-algebra code.
#![allow(clippy::needless_range_loop)]

pub mod reductions;

use iconic_core::{Scalar, Settings, Status};
use iconic_ipm::{solve_qp, solve_qp_with_termination, QpProblem, QpSolution, TermScale};
use iconic_linalg::DenseMatrix;

/// Diagonal scalings recovered from equilibration: `x = D x̂`, constraint scalings
/// `E`, and a scalar cost scaling `c` applied to the objective.
#[derive(Clone, Debug)]
pub(crate) struct Scaling<T: Scalar> {
    /// Variable scaling `D` (length `n`), with `x = D x̂`.
    pub d: Vec<T>,
    /// Equality-row scaling `E_eq` (length `m_eq`).
    pub e_eq: Vec<T>,
    /// Inequality-row scaling `E_in` (length `m_in`).
    pub e_in: Vec<T>,
    /// Scalar cost scaling applied to `P` and `q`.
    pub c: T,
}

fn scale_factor<T: Scalar>(norm: T) -> T {
    if norm > T::zero() {
        T::one() / norm.sqrt()
    } else {
        T::one()
    }
}

/// Sparse-view variant of the cone-aware equilibration: the constraint
/// row/column ∞-norm scans go through the sparse views when built (the
/// transport LPs' 99.94%-sparse A_in costs ~0.5–1s of dense scans across the
/// Ruiz sweeps alone); the dense fallback is bit-identical. The row scaling
/// is **rectified per cone** — a single shared factor across each
/// multi-dimensional cone block listed in `ineq_dims` (the cone dimensions,
/// summing to `m_in`), because elementwise row scaling would break SOC/PSD
/// membership. (For the orthant every block has dimension 1, recovering plain
/// Ruiz.)
pub(crate) fn equilibrate_coned_sp<T: Scalar>(
    prob: &QpProblem<T>,
    iters: usize,
    ineq_dims: &[usize],
    tol: T,
    sp: &reductions::SparseAIn<T>,
) -> (QpProblem<T>, Scaling<T>) {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let one = T::one();

    let trace = std::env::var_os("ICONIC_TRACE_PRESOLVE").is_some();
    let t_trace = std::time::Instant::now();
    let mut p = prob.p.clone();
    let mut a_eq = prob.a_eq.clone();
    let mut a_in = prob.a_in.clone();
    let mut q = prob.q.clone();
    let mut b_eq = prob.b_eq.clone();
    let mut b_in = prob.b_in.clone();
    if trace {
        eprintln!(
            "[ruiz clones] {:>9.1} ms",
            t_trace.elapsed().as_secs_f64() * 1e3
        );
    }

    let mut d = vec![one; n];
    let mut e_eq = vec![one; me];
    let mut e_in = vec![one; mi];

    // Pre-allocated scaling buffers reused each Ruiz sweep.
    let mut dd = vec![T::zero(); n];
    let mut de = vec![T::zero(); me];
    let mut di = vec![T::zero(); mi];
    // A diagonal P (the common SDP/transport shape — dense-stored, n²
    // entries for n nonzeros) is detected once so the per-sweep P scans and
    // applies are O(n) instead of O(n²): 5 sweeps × 2 × n² reads/writes on
    // the transport's 4800×4800 diagonal P is ~1s of pure memory traffic.
    // The detection early-exits on the first off-diagonal nonzero, so
    // genuinely dense P problems pay one element check. P is symmetric
    // (the QP convention), so a single triangle suffices — halving the
    // 184MB diagonal-P read (identical result on symmetric data; an
    // asymmetric lower-triangle-only nonzero would need the full scan, and
    // the suite gate covers every shipped problem).
    let t_pdiag = std::time::Instant::now();
    let p_diag = (0..n).all(|i| (i + 1..n).all(|j| p.get(i, j) == zero));
    if trace {
        eprintln!(
            "[ruiz p_diag detect] {:>9.1} ms",
            t_pdiag.elapsed().as_secs_f64() * 1e3
        );
    }

    // Buffer mirrors for the sparse path. When the views are active (and for
    // a diagonal P), the running scaled values live in `cur_rows` / `p_cur`
    // instead of the dense buffers, so the sweeps never touch the 190MB
    // dense A_in / 184MB dense P — whose copy-on-write deep copies on first
    // write cost ~200ms each on the 4940×4800 transport (measured). The
    // dense buffers are materialized fresh (lazy zero pages + the nonzeros)
    // once, after the last sweep. Bit-identical to the dense path: the
    // row-norm scans read the immutable views exactly as before (the views
    // are never re-scaled), the apply's products use the same operands and
    // order (the P side reads the mirror's current value, the A_in side the
    // view's original value — replicating the shipped operand sources), and
    // the materialized zeros are +0.0 in both paths (a diagonal P has only
    // +0.0 off-diagonals; the sparse extraction skips exact zeros).
    let mut cur_rows: Vec<Vec<(usize, T)>> = if sp.is_sparse() {
        (0..mi).map(|r| sp.row(r).expect("r < mi").to_vec()).collect()
    } else {
        Vec::new()
    };
    let mut p_cur: Vec<T> = if p_diag {
        (0..n).map(|i| p.get(i, i)).collect()
    } else {
        Vec::new()
    };

    let t_sweeps = std::time::Instant::now();
    for _ in 0..iters {
        // Column (variable) ∞-norms of P: scan row-by-row (cache-friendly for
        // row-major P) and take the column max — P is symmetric so this is equivalent.
        let mut col = vec![zero; n];
        if p_diag {
            for i in 0..n {
                col[i] = p_cur[i].abs();
            }
        } else {
            for i in 0..n {
                let mut mx = zero;
                for j in 0..n {
                    mx = mx.max(p.get(i, j).abs());
                }
                col[i] = mx;
            }
        }
        // Constraint row/col ∞-norms in a single row-major pass per matrix:
        // for each row r, update the row max and scatter into col[j] (the
        // scattered writes are dwarfed by the row-major reads).
        let mut row_eq = vec![zero; me];
        for r in 0..me {
            let mut mx = zero;
            for j in 0..n {
                let v = a_eq.get(r, j).abs();
                mx = mx.max(v);
                col[j] = col[j].max(v);
            }
            row_eq[r] = mx;
        }
        let mut row_in = vec![zero; mi];
        for r in 0..mi {
            let mut mx = zero;
            match sp.row(r) {
                Some(nz) => {
                    for &(j, v) in nz {
                        let av = v.abs();
                        mx = mx.max(av);
                        col[j] = col[j].max(av);
                    }
                }
                None => {
                    for j in 0..n {
                        let v = a_in.get(r, j).abs();
                        mx = mx.max(v);
                        col[j] = col[j].max(v);
                    }
                }
            }
            row_in[r] = mx;
        }

        for (out, &x) in dd.iter_mut().zip(col.iter()) {
            *out = scale_factor(x);
        }
        for (out, &x) in de.iter_mut().zip(row_eq.iter()) {
            *out = scale_factor(x);
        }
        for (out, &x) in di.iter_mut().zip(row_in.iter()) {
            *out = scale_factor(x);
        }

        // Rectify the inequality-row scaling per cone: a shared geometric-mean factor
        // across each multi-dim cone block, so SOC/PSD membership is preserved.
        let mut off = 0usize;
        for &d in ineq_dims {
            if d > 1 {
                let mut logsum = zero;
                for r in off..off + d {
                    logsum += di[r].ln();
                }
                let gm = (logsum / T::from_usize(d).expect("scalar literal")).exp();
                for r in off..off + d {
                    di[r] = gm;
                }
            }
            off += d;
        }

        // Apply: P ← DPD, A ← EAD, q ← Dq, b ← Eb. The p_diag P apply and
        // the sparse A_in apply go through the mirrors; the dense paths are
        // unchanged (dense P is genuinely dense, and the dense A_in fallback
        // scales every entry).
        if p_diag {
            for i in 0..n {
                p_cur[i] = p_cur[i] * dd[i] * dd[i];
            }
        } else {
            for i in 0..n {
                // `dd[i]` is invariant across j; bind once per row.
                let di = dd[i];
                for j in 0..n {
                    p.set(i, j, p.get(i, j) * di * dd[j]);
                }
            }
        }
        for i in 0..n {
            q[i] *= dd[i];
        }
        for r in 0..me {
            let dr = de[r];
            for j in 0..n {
                a_eq.set(r, j, a_eq.get(r, j) * dr * dd[j]);
            }
            b_eq[r] *= de[r];
        }
        if sp.is_sparse() {
            // Sparse apply: zeros stay zero under the scaling, so only the
            // nonzeros need touching (13k writes vs 23.7M on the transport).
            for r in 0..mi {
                let dr = di[r];
                let row = &mut cur_rows[r];
                for (t, &(j, v)) in sp.row(r).expect("r < mi").iter().enumerate() {
                    row[t].1 = v * dr * dd[j];
                }
            }
        } else {
            for r in 0..mi {
                let dr = di[r];
                for j in 0..n {
                    a_in.set(r, j, a_in.get(r, j) * dr * dd[j]);
                }
            }
        }
        for r in 0..mi {
            b_in[r] *= di[r];
        }

        for i in 0..n {
            d[i] *= dd[i];
        }
        for r in 0..me {
            e_eq[r] *= de[r];
        }
        for r in 0..mi {
            e_in[r] *= di[r];
        }

        // Convergence-aware stop: once a sweep's scale factors are
        // all ≈ 1 the data is equilibrated, so the remaining fixed sweeps are wasted work.
        let conv = dd
            .iter()
            .chain(de.iter())
            .chain(di.iter())
            .fold(zero, |m, &x| m.max((x - one).abs()));
        if conv < tol {
            break;
        }
    }
    if trace {
        eprintln!(
            "[ruiz sweeps] {:>9.1} ms",
            t_sweeps.elapsed().as_secs_f64() * 1e3
        );
    }
    // Cost scaling c = 1 / max(mean column ∞-norm of P, ‖q‖∞).
    let t_pmean = std::time::Instant::now();
    let mut psum = zero;
    if p_diag {
        for j in 0..n {
            psum += p_cur[j].abs();
        }
    } else {
        for j in 0..n {
            let mut mx = zero;
            for i in 0..n {
                mx = mx.max(p.get(i, j).abs());
            }
            psum += mx;
        }
    }
    let pmean = if n > 0 {
        psum / T::from_usize(n).expect("scalar literal")
    } else {
        one
    };
    let qn = q.iter().fold(zero, |a, &b| a.max(b.abs()));
    let denom = pmean.max(qn);
    let c = if denom > zero { one / denom } else { one };
    if p_diag {
        for j in 0..n {
            p_cur[j] *= c;
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                p.set(i, j, p.get(i, j) * c);
            }
        }
    }
    for i in 0..n {
        q[i] *= c;
    }

    // Materialize the dense buffers fresh from the mirrors (the sparse path
    // never wrote the cloned buffers, so they still share the input's pages —
    // a fresh zeroed allocation plus the nonzeros costs a few ms where the
    // copy-on-write deep copy cost ~200ms each).
    if sp.is_sparse() {
        let mut a_in_new = DenseMatrix::zeros(mi, n);
        for r in 0..mi {
            for &(j, v) in &cur_rows[r] {
                a_in_new.set(r, j, v);
            }
        }
        a_in = a_in_new;
    }
    if p_diag {
        let mut p_new = DenseMatrix::zeros(n, n);
        for i in 0..n {
            p_new.set(i, i, p_cur[i]);
        }
        p = p_new;
    }

    // Keep the CSR alive through equilibration. Ruiz's apply only touches
    // nonzeros (the sparse apply), so the structure is unchanged — only the
    // values are scaled. Build the scaled CSR from the mirrors in O(nnz);
    // the solver then never pays the O(mi·n) `csr_of_dense` re-derivation
    // (measured ~50ms on the 4940×4800 transport LP).
    let a_in_csr = if sp.is_sparse() {
        let zero = T::zero();
        let mut colptr = vec![0usize; mi + 1];
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        for r in 0..mi {
            for &(j, v) in &cur_rows[r] {
                if v != zero {
                    rowval.push(j);
                    nzval.push(v);
                }
            }
            colptr[r + 1] = rowval.len();
        }
        Some(iconic_linalg::CscMatrix {
            m: n,
            n: mi,
            colptr,
            rowval,
            nzval,
        })
    } else {
        None
    };
    if trace {
        eprintln!(
            "[ruiz pmean+csr] {:>9.1} ms",
            t_pmean.elapsed().as_secs_f64() * 1e3
        );
    }
    (
        QpProblem {
            p,
            q,
            a_eq,
            b_eq,
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr,
        },
        Scaling { d, e_eq, e_in, c },
    )
}

/// Map a solution of the equilibrated problem back to original units.
///
/// With `P̂ = c·DPD`, `q̂ = c·Dq`, `Â = EAD`, `b̂ = Eb`, the recovery is
/// `x = D x̂`, `y = E_eq ŷ / c`, `z = E_in ẑ / c`, `s = ŝ / E_in`, and the
/// objective is `obĵ / c`.
pub(crate) fn postsolve<T: Scalar>(scaling: &Scaling<T>, scaled: &QpSolution<T>) -> QpSolution<T> {
    let n = scaled.x.len();
    let me = scaled.y.len();
    let mi = scaled.s.len();
    let zero = T::zero();

    let mut x = vec![zero; n];
    for i in 0..n {
        x[i] = scaling.d[i] * scaled.x[i];
    }
    let mut y = vec![zero; me];
    for r in 0..me {
        y[r] = scaling.e_eq[r] * scaled.y[r] / scaling.c;
    }
    let mut s = vec![zero; mi];
    let mut z = vec![zero; mi];
    for r in 0..mi {
        s[r] = scaled.s[r] / scaling.e_in[r];
        z[r] = scaling.e_in[r] * scaled.z[r] / scaling.c;
    }

    QpSolution::new(
        scaled.status,
        x,
        y,
        s,
        z,
        scaled.obj_val / scaling.c,
        scaled.iters,
    )
}

/// Equilibrate, solve, and postsolve in one call. The solver iterates in scaled
/// space (faster, better-conditioned) but termination is judged on the recovered
/// original-unit residuals, so the returned solution meets the requested tolerance
/// in the user's units. Returns the solution already mapped back to original units.
pub(crate) fn solve_equilibrated<T: Scalar>(
    prob: &QpProblem<T>,
    settings: &Settings<T>,
) -> QpSolution<T> {
    let mi = prob.b_in.len();
    solve_equilibrated_sp(
        prob,
        settings,
        &reductions::SparseAIn::build(prob),
        &vec![1usize; mi],
    )
}

/// Sparse-view variant of [`solve_equilibrated`]: the caller's round has
/// already extracted the views, so the equilibration reuses them instead of
/// paying a second dense scan (the transport LPs' 4940×4800 extraction is
/// ~50ms). `ineq_dims` are the cone dimensions for cone-rectified scaling.
pub(crate) fn solve_equilibrated_sp<T: Scalar>(
    prob: &QpProblem<T>,
    settings: &Settings<T>,
    sp: &reductions::SparseAIn<T>,
    ineq_dims: &[usize],
) -> QpSolution<T> {
    let (scaled, sc) = equilibrate_coned_sp(
        prob,
        settings.equilibration_iters,
        ineq_dims,
        settings.equilibration_tol,
        sp,
    );
    let one = T::one();
    let term = TermScale {
        dual: sc.d.iter().map(|&d| one / (sc.c * d)).collect(),
        prim_eq: sc.e_eq.iter().map(|&e| one / e).collect(),
        prim_in: sc.e_in.iter().map(|&e| one / e).collect(),
        comp: one / sc.c,
    };
    let scaled_sol = solve_qp_with_termination(&scaled, settings, &term);
    postsolve(&sc, &scaled_sol)
}

fn infeasible_solution<T: Scalar>(prob: &QpProblem<T>, status: Status) -> QpSolution<T> {
    let zero = T::zero();
    QpSolution::new(
        status,
        vec![zero; prob.q.len()],
        vec![zero; prob.b_eq.len()],
        vec![zero; prob.b_in.len()],
        vec![zero; prob.b_in.len()],
        zero,
        0,
    )
}

/// Full presolve pipeline: eliminate fixed vars → eliminate doubletons → eliminate
/// empty columns → reduce rows (null/dominated/duplicate) → remove redundant ineqs →
/// remove dependent eq rows → equilibrate → solve with original-unit termination →
/// postsolve → restore in reverse order. A contradictory row short-circuits to
/// `PrimalInfeasible`; an unbounded empty column to `DualInfeasible`.
///
/// When the problem has no equalities and every inequality row is a singleton
/// (a bound on a single variable), all six reduction passes are guaranteed no-ops
/// — we skip them and go straight to equilibration.
pub fn solve_presolved<T: Scalar>(prob: &QpProblem<T>, settings: &Settings<T>) -> QpSolution<T> {
    // Fast no-op detection: bound-only QPs (no equalities, every inequality row has
    // exactly one nonzero) cannot benefit from any reduction pass — no fixed vars
    // (no eq rows), no doubletons (no eq rows), no empty cols (every variable appears
    // in a bound), no null/dominated/duplicate rows (every row is a unique singleton
    // bound), no redundant ineqs (the bounds ARE the variable box), no dependent eq
    // rows (no eq rows). Skip the 6 × O(n+m+nnz) scans.
    if prob.b_eq.is_empty() {
        let mut all_singleton = true;
        for r in 0..prob.b_in.len() {
            let mut nnz = 0usize;
            for j in 0..prob.q.len() {
                if prob.a_in.get(r, j) != T::zero() {
                    nnz += 1;
                    if nnz > 1 {
                        all_singleton = false;
                        break;
                    }
                }
            }
            if nnz == 0 {
                all_singleton = false;
                break;
            }
            if !all_singleton {
                break;
            }
        }
        if all_singleton {
            // Diagonal P + all bounds = independent 1D QPs → closed-form solution.
            // Detected in presolve and solved analytically in O(n), bypassing the IPM.
            // Each variable i: min ½ P_ii x_i² + q_i x_i  s.t. l_i ≤ x_i ≤ u_i
            // Solution: x_i = clip(−q_i / P_ii, l_i, u_i)
            let p_diag = (0..prob.q.len())
                .all(|i| (0..prob.q.len()).all(|j| i == j || prob.p.get(i, j) == T::zero()));
            if p_diag {
                let n = prob.q.len();
                let zero = T::zero();
                let mut x = vec![zero; n];
                let mut z = vec![zero; prob.b_in.len()];
                // Extract bounds from unit rows: each row is a_i * x_j <= b_i
                let huge = T::from_f64(1e20).expect("scalar literal");
                let mut lb = vec![-huge; n];
                let mut ub = vec![huge; n];
                for r in 0..prob.b_in.len() {
                    let mut col = 0usize;
                    let mut val = zero;
                    let mut found = false;
                    for j in 0..n {
                        if prob.a_in.get(r, j) != zero {
                            if found {
                                col = usize::MAX;
                                break;
                            }
                            col = j;
                            val = prob.a_in.get(r, j);
                            found = true;
                        }
                    }
                    if col == usize::MAX {
                        continue;
                    } // shouldn't happen (all_singleton check)
                    let rhs = prob.b_in[r] / val;
                    if val > zero {
                        ub[col] = ub[col].min(rhs);
                    } else {
                        lb[col] = lb[col].max(rhs);
                    }
                }
                // Solve each variable independently
                let mut obj = zero;
                for i in 0..n {
                    let p_ii = prob.p.get(i, i);
                    if p_ii <= zero {
                        // Non-positive curvature → LP-like, use IPM fallback
                        return solve_equilibrated(prob, settings);
                    }
                    let x_unb = -prob.q[i] / p_ii;
                    x[i] = x_unb.max(lb[i]).min(ub[i]);
                    obj += T::from_f64(0.5).expect("scalar literal") * p_ii * x[i] * x[i] + prob.q[i] * x[i];
                }
                // Dual multipliers from stationarity: only the row whose own implied
                // bound x is actually sitting at is active -- a variable can carry
                // both an upper and a lower singleton row, and complementary
                // slackness requires the non-binding one's dual to be zero. (The
                // previous version derived every row's z from the variable's
                // stationarity regardless of whether *that specific row* was
                // binding, which put a nonzero dual on slack rows and, for a plain
                // box constraint, on the wrong row entirely.) The slack s is the
                // row's own residual against x, correct for active and inactive
                // rows alike (was hardcoded to zero, which is only right when
                // active).
                let mut s = vec![zero; prob.b_in.len()];
                for r in 0..prob.b_in.len() {
                    let mut col = 0usize;
                    let mut val = zero;
                    for j in 0..n {
                        if prob.a_in.get(r, j) != zero {
                            col = j;
                            val = prob.a_in.get(r, j);
                            break;
                        }
                    }
                    s[r] = prob.b_in[r] - val * x[col];
                    if s[r].abs() <= settings.eps_abs {
                        let stat = prob.p.get(col, col) * x[col] + prob.q[col];
                        z[r] = (-stat / val).max(zero);
                    }
                }
                return QpSolution::new(
                    Status::Solved,
                    x,
                    vec![],
                    s,
                    z,
                    obj,
                    0,
                );
            }
            return solve_equilibrated(prob, settings);
        }
    }

    /// One presolve reduction round: the full structural chain
    /// (negated-pair folding → equality chain → column elimination → row
    /// reductions → redundant/dependent-row removal). Returns the reduced
    /// problem and the records needed to restore the solution to this round's
    /// input space.
    struct RoundRecords<T: Scalar> {
        pair_reduced: QpProblem<T>,
        pairred: reductions::NegatedPairReduction<T>,
        fix_reduced: QpProblem<T>,
        fixred: reductions::FixedVarReduction<T>,
        dbl_reduced: QpProblem<T>,
        dblred: reductions::DoubletonReduction<T>,
        free_reduced: QpProblem<T>,
        freered: reductions::FreeVarReduction<T>,
        mergedred: reductions::MergeReduction<T>,
        colred: reductions::ColReduction<T>,
        rowred: reductions::RowReduction<T>,
        redunred: reductions::RedundancyReduction<T>,
        depred: reductions::EqDepReduction,
    }

    #[allow(clippy::type_complexity)]
    fn presolve_round<T: Scalar>(
        prob: &QpProblem<T>,
        settings: &Settings<T>,
    ) -> Result<(QpProblem<T>, RoundRecords<T>), Status> {
        // Sparse row/column views of A_in, built from a single dense pass. The
        // passes' dense scans of a 99.94%-sparse 4940×4800 A_in cost ~0.5–0.9s
        // each (measured on the transport LPs); the views are ~0.05s. Gated on
        // size+density inside; small/dense problems fall back to the dense
        // scans, bit-identical. The views are valid only for the problem they
        // were built from, so the chain rebuilds them lazily — only when a pass
        // actually changed the problem (the no-op chain, e.g. the transport LPs,
        // pays exactly one build).
        let sp = reductions::SparseAIn::build(prob);
        let (pair_reduced, pairred) = reductions::fold_negated_pairs(prob, &sp)?;

        let no_eq = pair_reduced.b_eq.is_empty();
        let eq_chain_noop: Option<QpProblem<T>> = if no_eq {
            Some(pair_reduced.clone())
        } else {
            None
        };
        let (fix_reduced, fixred) = match &eq_chain_noop {
            Some(shared) => (
                shared.clone(),
                reductions::FixedVarReduction::no_op(shared.q.len()),
            ),
            None => reductions::eliminate_fixed_vars(&pair_reduced)?,
        };
        let dbl_reduced_owned;
        let (dbl_reduced, dblred): (&QpProblem<T>, _) = match &eq_chain_noop {
            Some(shared) => (
                shared,
                reductions::DoubletonReduction::no_op(shared.q.len()),
            ),
            None => {
                let (d, r) =
                    reductions::eliminate_doubleton_eqs(&fix_reduced, settings.fill_budget)?;
                dbl_reduced_owned = d;
                (&dbl_reduced_owned, r)
            }
        };
        let free_reduced_owned;
        let (free_reduced, freered): (&QpProblem<T>, _) = match &eq_chain_noop {
            Some(shared) => (
                shared,
                reductions::FreeVarReduction {
                    kept_cols: (0..shared.q.len()).collect(),
                    kept_eq: vec![],
                    elims: vec![],
                    n_orig: shared.q.len(),
                    n_eq_orig: 0,
                },
            ),
            None if dbl_reduced.q.len() >= 4 => {
                let (f, r) = reductions::eliminate_free_vars(dbl_reduced, 12, 48)?;
                free_reduced_owned = f;
                (&free_reduced_owned, r)
            }
            None => {
                free_reduced_owned = dbl_reduced.clone();
                let r = reductions::FreeVarReduction {
                    kept_cols: (0..dbl_reduced.q.len()).collect(),
                    kept_eq: (0..dbl_reduced.b_eq.len()).collect(),
                    elims: vec![],
                    n_orig: dbl_reduced.q.len(),
                    n_eq_orig: dbl_reduced.b_eq.len(),
                };
                (&free_reduced_owned, r)
            }
        };
        let merged_reduced_owned;
        let (merged_reduced, mergedred): (&QpProblem<T>, _) = match &eq_chain_noop {
            Some(shared) => (
                shared,
                reductions::MergeReduction {
                    kept_cols: (0..shared.q.len()).collect(),
                    kept_eq_rows: vec![],
                    merges: vec![],
                    n_orig: shared.q.len(),
                    n_eq_orig: 0,
                },
            ),
            None if free_reduced.q.len() >= 5 => {
                let (m, r) = reductions::merge_equality_rows(free_reduced, 48)?;
                merged_reduced_owned = m;
                (&merged_reduced_owned, r)
            }
            None => {
                merged_reduced_owned = free_reduced.clone();
                let r = reductions::MergeReduction {
                    kept_cols: (0..free_reduced.q.len()).collect(),
                    kept_eq_rows: (0..free_reduced.b_eq.len()).collect(),
                    merges: vec![],
                    n_orig: free_reduced.q.len(),
                    n_eq_orig: free_reduced.b_eq.len(),
                };
                (&merged_reduced_owned, r)
            }
        };
        // Lazy sparse-view rebuilds: each pass's views must match its input
        // problem. The fold changed the problem only if it folded pairs; the
        // equality chain runs only with equalities (its views would be stale
        // only then, so rebuild in that case too).
        let sp = if pairred.changed() {
            reductions::SparseAIn::build(&pair_reduced)
        } else {
            sp
        };
        let sp = if no_eq {
            sp
        } else {
            reductions::SparseAIn::build(merged_reduced)
        };
        let (col_reduced, colred) = reductions::eliminate_empty_cols(merged_reduced, &sp)?;
        let sp = if colred.changed() {
            reductions::SparseAIn::build(&col_reduced)
        } else {
            sp
        };
        let (reduced, rowred) = reductions::reduce_rows(&col_reduced, &sp)?;
        let sp = if rowred.changed() {
            reductions::SparseAIn::build(&reduced)
        } else {
            sp
        };
        let (redun_reduced, redunred) = reductions::remove_redundant_ineqs(&reduced, &sp)?;
        // The dependent-eq pass needs equality rows to fire; with none it still
        // rebuilds (clones) the full problem — on the big-sparse transport LP
        // that is a 0.4s no-op. The no_eq fast path already covers the other
        // equality passes; skip this one the same way. The O(me²·n) dense
        // row-echelon also gets a cost budget (mirroring the shipped iconic-api
        // gate): on a large structurally-independent equality block (e.g. MPC
        // dynamics) it costs more than the whole solve and removes nothing —
        // the proximal regularization is the correctness net past the budget.
        let me_red = redun_reduced.b_eq.len();
        let dep_worth_it = me_red >= 2
            && (me_red as u64) * (me_red as u64) * (redun_reduced.q.len() as u64) <= 5_000_000;
        let (dep_reduced, depred) = if !dep_worth_it {
            (redun_reduced, reductions::EqDepReduction::no_op())
        } else {
            reductions::remove_dependent_eq_rows(&redun_reduced)?
        };
        Ok((
            dep_reduced,
            RoundRecords {
                pair_reduced,
                pairred,
                fix_reduced,
                fixred,
                dbl_reduced: dbl_reduced.clone(),
                dblred,
                free_reduced: free_reduced.clone(),
                freered,
                mergedred,
                colred,
                rowred,
                redunred,
                depred,
            },
        ))
    }

    /// Restore a round's solution back to its input space (LIFO over the
    /// round's records, mirroring the original hand-written restore chain).
    fn restore_round<T: Scalar>(
        rec: RoundRecords<T>,
        reduced_sol: &QpSolution<T>,
    ) -> QpSolution<T> {
        let dep_restored = reductions::restore_dependent_eq_rows(&rec.depred, reduced_sol);
        let redun_restored = reductions::restore_redundant_ineqs(&rec.redunred, &dep_restored);
        let row_restored = reductions::restore_rows(&rec.rowred, &redun_restored);
        let col_restored = reductions::restore_cols(&rec.colred, &row_restored);
        let merged_restored =
            reductions::restore_merged_rows(&rec.free_reduced, &rec.mergedred, &col_restored);
        let free_restored =
            reductions::restore_free_vars(&rec.dbl_reduced, &rec.freered, &merged_restored);
        let dbl_restored =
            reductions::restore_doubleton_eqs(&rec.fix_reduced, &rec.dblred, &free_restored);
        let fixed_restored =
            reductions::restore_fixed_vars(&rec.pair_reduced, &rec.fixred, &dbl_restored);
        reductions::restore_negated_pairs(&rec.pairred, &fixed_restored)
    }

    // Fold negated-scaled duplicate inequality pairs (aᵀx ≤ b and −aᵀx ≤ −b with
    // equal normalized bounds) into single equality rows — net −2 rows per pair,
    // and the created equalities feed the fixed/doubleton/free/merge chain below
    // (a singleton pair becomes a pinned variable; a doubleton pair an eliminated
    // column). Exact reformulation; the pair's free multiplier splits back on
    // restore. Runs after the all-singleton fast path (which guarantees no pairs)
    // and before the equality chain. (Achterberg et al. 2020 §5.2.)
    // ── Solve-check-solve presolve orchestration ────────────────────
    // The re-entrant presolve pattern: reduction rounds run and re-enter while they still
    // improve the model; the 0.5% gate stops the loop when a round removes
    // less than half a percent of rows+cols (the second round mostly
    // catches reductions only visible after the first round's eliminations —
    // e.g. a fold-created singleton becoming a pinned variable). Records
    // from all rounds stack; the restore unwinds the newest round first.
    let model_size = |p: &QpProblem<T>| p.q.len() + p.b_eq.len() + p.b_in.len();
    let trace = std::env::var_os("ICONIC_TRACE_PRESOLVE").is_some();
    let t0 = std::time::Instant::now();
    let (round1_reduced, rec1) = match presolve_round(prob, settings) {
        Err(status) => return infeasible_solution(prob, status),
        Ok(v) => v,
    };
    if trace {
        eprintln!(
            "[presolve_round] {:>9.1} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let size0 = model_size(prob);
    let size1 = model_size(&round1_reduced);
    let improved = (size0 - size1) as f64 / size0.max(1) as f64;
    let (solve_problem, rec_stack) = if improved >= 0.005 && settings.presolve_rounds > 1 {
        let t1 = std::time::Instant::now();
        match presolve_round(&round1_reduced, settings) {
            Ok((r2, rec2)) => {
                if trace {
                    eprintln!(
                        "[presolve_round2] {:>9.1} ms",
                        t1.elapsed().as_secs_f64() * 1e3
                    );
                }
                (r2, vec![rec2, rec1])
            }
            Err(status) => return infeasible_solution(prob, status),
        }
    } else {
        (round1_reduced, vec![rec1])
    };

    // Thread a sparse view of the reduced problem into the equilibration so
    // the Ruiz sweeps reuse it (with the CSR kept alive through presolve the
    // build is O(nnz); on the dense-input bench path it is one O(mi·n) scan —
    // exactly what the equilibration would otherwise pay internally).
    let t2 = std::time::Instant::now();
    let sp_final = reductions::SparseAIn::build(&solve_problem);
    if trace {
        eprintln!(
            "[sp_final build] {:>9.1} ms",
            t2.elapsed().as_secs_f64() * 1e3
        );
    }
    let t3 = std::time::Instant::now();
    let reduced_sol = solve_equilibrated_sp(
        &solve_problem,
        settings,
        &sp_final,
        &vec![1usize; solve_problem.b_in.len()],
    );
    if trace {
        eprintln!(
            "[solve_equilibrated_sp] {:>9.1} ms",
            t3.elapsed().as_secs_f64() * 1e3
        );
    }
    let mut sol = reduced_sol;
    for rec in rec_stack {
        sol = restore_round(rec, &sol);
    }

    // Column elimination dropped objective terms; recompute at original dimensions.
    let px = prob.p.matvec(&sol.x);
    let half = T::from_f64(0.5).expect("scalar literal");
    let mut obj = T::zero();
    for i in 0..sol.x.len() {
        obj += half * sol.x[i] * px[i] + prob.q[i] * sol.x[i];
    }
    sol.obj_val = obj;

    // Presolve-failure fallback: the equilibration itself can make the scaled
    // problem *harder* for the IPM than the original (measured on the
    // condition-sweep family at κ ≥ 1e5: Ruiz's row/RHS scaling drives the
    // scaled-space dual iterate into a runaway that returns a point ~3 orders of
    // magnitude off in the objective, graded SolvedInaccurate with an honest
    // kkt_res ~50-84, while the un-equilibrated problem converges in ~15
    // iterations at machine accuracy). When the presolved solve came back
    // honestly-bad — not Solved, and the restored point's original-unit KKT
    // residual is large (so the failure is a wrong point, not a good point
    // graded loosely) — re-solve the ORIGINAL problem directly and keep the
    // better of the two, verified in original units. Bounded by problem size so
    // a genuinely hard large instance (e.g. transport, where both paths stall)
    // does not pay a double solve, and by a capped iteration budget on the
    // retry. The whole loop is exact: both candidates are graded in original
    // units, so the fallback can never return a worse point than the status
    // claims.
    if sol.status != Status::Solved {
        let n_tot = prob.q.len() + prob.b_eq.len() + prob.b_in.len();
        let kkt = orig_kkt_residual(prob, &sol);
        // The retry only fires when the presolved solve exited EARLY with a bad
        // point. An early exit (well inside the iteration budget) with a large
        // KKT residual is a trajectory failure — the scaled-space iterate
        // diverged (the condsweep pattern) — where the un-equilibrated problem
        // is worth trying. Grinding through most of the budget to a bad point
        // is a genuine stall (the transport pattern) that both paths exhibit
        // and the retry would only double the cost.
        if sol.iters < 60 && n_tot <= 1500 && kkt > T::from_f64(1e-3).expect("scalar literal") {
            let mut retry_settings = settings.clone();
            // Cap the retry's budget: the raw path converges in ≤ ~30 iterations
            // on the problems this fires for; anything slower is a genuine stall
            // and the presolved point is kept.
            retry_settings.max_iters = retry_settings.max_iters.min(60);
            let raw = solve_qp(prob, &retry_settings);
            if raw.status == Status::Solved {
                sol = raw;
            } else {
                let kkt_raw = orig_kkt_residual(prob, &raw);
                if kkt_raw < kkt {
                    sol = raw;
                }
            }
            // Conic-engine tier: the condensed-Gram path (A_inᵀ(Z/S)A_in) squares
            // the conditioning of the primal-dual ratio, so on ill-conditioned
            // QPs (κ ≥ ~1e5 on tall-thin shapes) the QP path can return a point
            // whose KKT residual is O(1)-large while the conic engine's
            // quasidefinite augmented system — conditioning tracking κ(A), not
            // κ(A)² — solves the identical problem at machine accuracy (measured:
            // 15×10 condsweep shape at κ=1e8 — QP path kkt 7.81 / obj 0.0 vs
            // conic kkt 1.16e-10 / exact). The QP form's inequality rows are the
            // nonneg-cone rows by construction, so Zero(n_eq) + NonNeg(mi) is the
            // exact cone mapping; same early-exit/size/budget gates as the raw
            // retry.
            if sol.status != Status::Solved && p_is_convex(&prob.p) {
                // The conic engine takes the equality block from prob.a_eq
                // directly; `cones` describes only the inequality rows, which
                // in the QP form are the nonneg-cone rows by construction.
                // Convex-P gate: the tier's soundness argument (the conic
                // engine is an exact reformulation of the same QP) only holds
                // for convex P. For a nonconvex P the continuous "relaxation"
                // is itself a local-point problem, and swapping engines merely
                // perturbs which local point the MIP search sees — measured on
                // the qkp family: the tier fired in node LPs and moved the
                // incumbent 3→58 iters at a worse objective. MIP node LPs with
                // nonconvex P keep the QP path's behavior.
                let cones = vec![iconic_ipm::conic::Cone::NonNeg(prob.b_in.len())];
                let conic = iconic_ipm::conic::solve_cone_qp(prob, &cones, &retry_settings);
                if conic.status == Status::Solved {
                    sol = conic;
                } else {
                    let kkt_conic = orig_kkt_residual(prob, &conic);
                    let kkt_sol = orig_kkt_residual(prob, &sol);
                    if kkt_conic < kkt_sol {
                        sol = conic;
                    }
                }
            }
        }
    }
    sol
}

/// Original-unit KKT residual of a (restored) solution against the ORIGINAL
/// problem: max over stationarity, equality/inequality feasibility, and
/// complementarity. Used to verify that a returned point is a solution of the
/// problem the user gave — presolve reductions and scaling are undone by
/// `postsolve`, so a point can only fail this check if the solve itself was bad.
/// Cheap convexity test for the fallback gates: P is a diagonal-dominant
/// PSD matrix iff every 2×2 principal minor is nonnegative (necessary for
/// PSD; with zero diagonals it degenerates to the off-diagonal check, which
/// is exactly the QUBO-linearization shape −Q with Q ≥ 0 that the conic tier
/// must not touch). Matches iconic-mip's `lp_bound_is_valid` criterion.
fn p_is_convex<T: Scalar>(p: &DenseMatrix<T>) -> bool {
    let n = p.nrows;
    let zero = T::zero();
    let eps = T::from_f64(1e-9).expect("scalar literal");
    for i in 0..n {
        if p.get(i, i) < -eps {
            return false;
        }
    }
    for i in 0..n {
        let pii = p.get(i, i);
        for j in (i + 1)..n {
            let pij = p.get(i, j);
            if pij == zero {
                continue;
            }
            if pii * p.get(j, j) < pij * pij - eps {
                return false;
            }
        }
    }
    true
}

fn orig_kkt_residual<T: Scalar>(prob: &QpProblem<T>, sol: &QpSolution<T>) -> T {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let mut worst = zero;
    let px = prob.p.matvec(&sol.x);
    let aty = prob.a_eq.matvec_t(&sol.y);
    let atz = prob.a_in.matvec_t(&sol.z);
    for i in 0..n {
        worst = worst.max((px[i] + prob.q[i] + aty[i] + atz[i]).abs());
    }
    let aeqx = prob.a_eq.matvec(&sol.x);
    for i in 0..me {
        worst = worst.max((aeqx[i] - prob.b_eq[i]).abs());
    }
    let ainx = prob.a_in.matvec(&sol.x);
    for i in 0..mi {
        worst = worst.max((ainx[i] + sol.s[i] - prob.b_in[i]).abs());
        worst = worst.max((sol.s[i] * sol.z[i]).abs());
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use iconic_ipm::solve_qp;
    use iconic_linalg::DenseMatrix;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
    }

    /// Equilibrate → solve → postsolve must reproduce the direct solve, including
    /// on a badly-scaled problem (entries spanning ~4 orders of magnitude).
    #[test]
    fn postsolve_recovers_direct_solution() {
        // min ½(1e4 x0² + x1²) − 1e4 x0 − x1  s.t.  x0 + x1 ≤ 5.
        // Badly scaled (4 orders of magnitude) but well-determined: unconstrained
        // optimum x = [1, 1] with both curvatures well above the regularization.
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1e4, 0.0, 0.0, 1.0]),
            q: vec![-1e4, -1.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(1, 2, vec![1.0, 1.0]),
            b_in: vec![5.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let settings = Settings::<f64>::default();

        let direct = solve_qp(&prob, &settings);

        let mi = prob.b_in.len();
        let (scaled_prob, scaling) = equilibrate_coned_sp(
            &prob,
            10,
            &vec![1usize; mi],
            1e-2,
            &reductions::SparseAIn::build(&prob),
        );
        let scaled_sol = solve_qp(&scaled_prob, &settings);
        let recovered = postsolve(&scaling, &scaled_sol);

        assert_eq!(recovered.status, Status::Solved);
        assert!(
            max_abs_diff(&recovered.x, &direct.x) < 1e-5,
            "x mismatch: {:?} vs {:?}",
            recovered.x,
            direct.x
        );
        assert!((recovered.x[0] - 1.0).abs() < 1e-5);
        assert!((recovered.x[1] - 1.0).abs() < 1e-5);
    }

    /// On a constrained problem with an active inequality, the recovered duals and
    /// slacks must match the direct solve too.
    #[test]
    fn postsolve_recovers_duals() {
        // min ½(x0²+x1²) s.t. −x0−x1 ≤ −2 (i.e. x0+x1 ≥ 2). Optimum x=[1,1], z>0.
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![3.0, 0.0, 0.0, 0.5]),
            q: vec![0.0, 0.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(1, 2, vec![-1.0, -1.0]),
            b_in: vec![-2.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let settings = Settings::<f64>::default();

        let direct = solve_qp(&prob, &settings);
        let mi = prob.b_in.len();
        let (scaled, sc) = equilibrate_coned_sp(
            &prob,
            10,
            &vec![1usize; mi],
            1e-2,
            &reductions::SparseAIn::build(&prob),
        );
        let recovered = postsolve(&sc, &solve_qp(&scaled, &settings));

        assert_eq!(recovered.status, Status::Solved);
        assert!(max_abs_diff(&recovered.x, &direct.x) < 1e-6);
        assert!(
            max_abs_diff(&recovered.z, &direct.z) < 1e-6,
            "dual mismatch"
        );
        assert!(
            max_abs_diff(&recovered.s, &direct.s) < 1e-6,
            "slack mismatch"
        );
    }

    /// `solve_equilibrated` judges termination in original units, so the recovered
    /// solution is accurate there even though the solve happens in scaled space.
    #[test]
    fn equilibrated_solve_accurate_in_original_units() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1e4, 0.0, 0.0, 1.0]),
            q: vec![-1e4, -1.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(1, 2, vec![1.0, 1.0]),
            b_in: vec![5.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_equilibrated(&prob, &Settings::<f64>::default());
        assert_eq!(sol.status, Status::Solved);
        assert!((sol.x[0] - 1.0).abs() < 1e-7, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-7, "x1={}", sol.x[1]);
    }

    /// The full pipeline drops a redundant empty row, solves, and restores the
    /// solution at the original dimensions.
    #[test]
    fn solve_presolved_handles_redundant_row() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(2, 2, vec![-1.0, -1.0, 0.0, 0.0]),
            b_in: vec![-2.0, 5.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_presolved(&prob, &Settings::<f64>::default());
        assert_eq!(sol.status, Status::Solved);
        assert!((sol.x[0] - 1.0).abs() < 1e-6);
        assert!((sol.x[1] - 1.0).abs() < 1e-6);
        // Solution is restored at the original dimension (2 inequality rows).
        assert_eq!(sol.s.len(), 2);
        assert!((sol.s[1] - 5.0).abs() < 1e-9);
    }

    /// A contradictory empty row short-circuits to PrimalInfeasible.
    #[test]
    fn solve_presolved_detects_infeasibility() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(1, 1, vec![1.0]),
            q: vec![0.0],
            a_eq: DenseMatrix::zeros(0, 1),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(1, 1, vec![0.0]),
            b_in: vec![-1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_presolved(&prob, &Settings::<f64>::default());
        assert_eq!(sol.status, Status::PrimalInfeasible);
    }

    /// A redundant equality row (row₃ = row₁ + row₂) is removed, and the degenerate QP
    /// still solves to the correct min-norm point.
    #[test]
    fn removes_redundant_equality_row() {
        // min ½‖x‖² s.t. x0+x1=1, x1+x2=1, x0+2x1+x2=2 (third = first+second, redundant).
        // Optimum: x = [1/3, 2/3, 1/3].
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            q: vec![0.0; 3],
            a_eq: DenseMatrix::from_row_major(3, 3, vec![1., 1., 0., 0., 1., 1., 1., 2., 1.]),
            b_eq: vec![1.0, 1.0, 2.0],
            a_in: DenseMatrix::zeros(0, 3),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        // The reduction drops one of the three rows.
        let (reduced, _red) = reductions::remove_dependent_eq_rows(&prob).unwrap();
        assert_eq!(reduced.b_eq.len(), 2, "should keep 2 independent rows");

        let sol = solve_presolved(&prob, &Settings::<f64>::default());
        assert_eq!(sol.status, Status::Solved);
        let expect = [1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0];
        assert!(max_abs_diff(&sol.x, &expect) < 1e-6, "x = {:?}", sol.x);
        // Equality multipliers are returned at full (3) dimension; the dropped row's dual is 0.
        assert_eq!(sol.y.len(), 3);
    }

    /// Dependent rows of A with an inconsistent right-hand side are detected as infeasible.
    #[test]
    fn inconsistent_dependent_row_is_infeasible() {
        // x0+x1=1, x1+x2=1, x0+2x1+x2=3 — the third's rhs contradicts the first two (2≠3).
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            q: vec![0.0; 3],
            a_eq: DenseMatrix::from_row_major(3, 3, vec![1., 1., 0., 0., 1., 1., 1., 2., 1.]),
            b_eq: vec![1.0, 1.0, 3.0],
            a_in: DenseMatrix::zeros(0, 3),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        assert_eq!(
            reductions::remove_dependent_eq_rows(&prob).err(),
            Some(Status::PrimalInfeasible)
        );
    }

    /// Regression: `eliminate_by_substitution` has no postsolve counterpart
    /// (`restore_by_substitution` does not exist anywhere in the crate), so it must
    /// not be wired into `solve_presolved`'s forward pipeline. Previously its
    /// output was fed into `merge_equality_rows` and then into
    /// `restore_merged_rows`, while `restore_free_vars` downstream still assumed
    /// the restored solution had `free_reduced.q.len()` entries — whenever the
    /// substitution pass actually eliminated ≥1 variable, that solution vector was
    /// one element too short and `restore_free_vars` panicked with an
    /// index-out-of-bounds on `reduced.x[a]`.
    ///
    /// n=6 variables, diagonal `P=I`, a single equality row summing all six
    /// variables to 6, and one independent loose box inequality per variable (so
    /// every column is inequality-touched: `eliminate_free_vars`'s `col_in_ineq`
    /// guard blocks it from firing, leaving `eliminate_by_substitution` as the
    /// sole pass that could act on the equality row — exactly the scenario that
    /// used to panic at the `restore_free_vars` index).
    #[test]
    fn solve_presolved_sum_equality_with_box_ineqs_does_not_panic_and_matches_direct() {
        let n = 6;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let mut a_in = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            a_in.set(i, i, 1.0);
        }
        let prob = QpProblem {
            p,
            q: vec![0.0; n],
            a_eq: DenseMatrix::from_row_major(1, n, vec![1.0; n]),
            b_eq: vec![6.0],
            a_in,
            b_in: vec![10.0; n],
            a_eq_csr: None,
            a_in_csr: None,
        };

        // Sanity: confirm this instance exercises `eliminate_by_substitution`
        // actually eliminating a variable (the case that used to corrupt the
        // pipeline), with `eliminate_free_vars` fully blocked by the
        // inequality-membership guard so substitution is the sole active pass on
        // the equality row.
        let (dbl_reduced, _) = reductions::eliminate_doubleton_eqs(
            &reductions::eliminate_fixed_vars(&prob).unwrap().0,
            10.0,
        )
        .unwrap();
        let (free_reduced, freered) =
            reductions::eliminate_free_vars(&dbl_reduced, 12, 48).unwrap();
        assert!(
            freered.elims.is_empty(),
            "expected eliminate_free_vars to be blocked by col_in_ineq"
        );
        assert_eq!(
            free_reduced.b_eq.len(),
            1,
            "equality row must survive (substitution fn was removed)"
        );

        // Direct solve of the unreduced problem: minimize sum(x_i^2)/2 s.t.
        // sum(x_i)=6, 0<=x_i<=10 (upper bound non-binding) -> uniform x_i=1 by
        // symmetry, obj = 3.0.
        let direct = solve_qp(&prob, &Settings::<f64>::default());
        assert_eq!(direct.status, Status::Solved);

        // Must not panic (the original bug) and must match the direct solve.
        let presolved = solve_presolved(&prob, &Settings::<f64>::default());
        assert_eq!(presolved.status, Status::Solved);
        assert_eq!(presolved.x.len(), n, "restored x must have original n={n}");

        for j in 0..n {
            assert!(
                (presolved.x[j] - 1.0).abs() < 1e-5,
                "x{j}={} should be 1.0 by symmetry (sum(x)=6, min sum(x^2))",
                presolved.x[j]
            );
            assert!(
                (presolved.x[j] - direct.x[j]).abs() < 1e-5,
                "x{j}: presolved={} direct={}",
                presolved.x[j],
                direct.x[j]
            );
        }
        assert!(
            (presolved.obj_val - 3.0).abs() < 1e-5,
            "obj_val={} should be 3.0 (=0.5*6*1^2), not corrupted by the \
             dimension-desync bug",
            presolved.obj_val
        );
        assert!(
            (presolved.obj_val - direct.obj_val).abs() < 1e-6,
            "obj: presolved={} direct={}",
            presolved.obj_val,
            direct.obj_val
        );
    }

    /// Regression for the `eliminate_by_substitution` `None`-fallback bug: when
    /// `eliminate_free_vars` fires and fully consumes every equality row
    /// (`free_reduced.b_eq.len() == 0`), `eliminate_by_substitution`'s `me == 0`
    /// guard trivially returns `None`. The buggy fallback rebased the forward
    /// pipeline (`merge_equality_rows` onward) on `dbl_reduced` — the
    /// *pre*-free-var-elimination problem — instead of `free_reduced`, silently
    /// discarding `eliminate_free_vars`' reduction while `restore_free_vars` was
    /// still invoked downstream with the `freered` record built for the smaller
    /// `free_reduced` dimension. This desynced the forward chain's problem size
    /// from what `restore_free_vars` expected, producing a wrong, index-shifted
    /// solution reported with a confident `Status::Solved` and no diagnostic.
    ///
    /// Two equality rows (3 nonzeros each, so `eliminate_doubleton_eqs` is a
    /// no-op) each fully determine one variable in terms of the other two via
    /// `eliminate_free_vars`, dropping n 6->4 and me 2->0 — exactly the
    /// `me == 0` gate that gives `eliminate_by_substitution` no equality rows to
    /// act on.
    #[test]
    fn solve_presolved_free_var_full_row_consumption_matches_direct_solve() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(
                6,
                6,
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                ],
            ),
            q: vec![-1.0, -2.0, 0.5, -0.5, 0.1, 0.2],
            a_eq: DenseMatrix::from_row_major(
                2,
                6,
                vec![
                    1.0, 0.0, 1.0, 1.0, 0.0, 0.0, //  x0 + x2 + x3 = 4
                    0.0, 1.0, 0.0, 0.0, 1.0, 1.0, //  x1 + x4 + x5 = 5
                ],
            ),
            b_eq: vec![4.0, 5.0],
            a_in: DenseMatrix::zeros(0, 6),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };

        // Sanity: confirm this instance actually exercises the target code path
        // (free-var elimination fully consumes both equality rows, so the
        // substitution pass's `me == 0` guard returns `None`).
        let (dbl_reduced, _) = reductions::eliminate_doubleton_eqs(
            &reductions::eliminate_fixed_vars(&prob).unwrap().0,
            10.0,
        )
        .unwrap();
        let (free_reduced, freered) =
            reductions::eliminate_free_vars(&dbl_reduced, 12, 48).unwrap();
        assert!(!freered.elims.is_empty(), "expected free-var elim to fire");
        assert_eq!(free_reduced.q.len(), 4, "expected n to shrink 6->4");
        assert_eq!(free_reduced.b_eq.len(), 0, "expected both eq rows consumed");
        // `eliminate_by_substitution` was removed from `reductions`; the
        // presolve pipeline below exercises the me==0 path without it.

        let direct = solve_qp(&prob, &Settings::<f64>::default());
        assert_eq!(direct.status, Status::Solved);

        let presolved = solve_presolved(&prob, &Settings::<f64>::default());
        assert_eq!(presolved.status, Status::Solved);
        assert_eq!(presolved.x.len(), 6, "restored x must have original n=6");

        // Specific numeric assertions: with the bug, the pipeline mis-indexed
        // the solution during restore, producing x ≈ [-1.782, 3.682, 2.682,
        // 3.100, 0.591, 0.727] and obj ≈ 24.825 (~14x too high) instead of the
        // true optimum below — while still satisfying the two equality
        // constraints and being reported `Status::Solved`, so only an exact
        // numeric check (not just status or feasibility) catches it.
        let expect_x = [2.6818181818, 3.1, 0.5909090909, 0.7272727273, 1.0, 0.9];
        for j in 0..6 {
            assert!(
                (presolved.x[j] - expect_x[j]).abs() < 1e-6,
                "x{j}: presolved={} expected~{}",
                presolved.x[j],
                expect_x[j]
            );
            assert!(
                (presolved.x[j] - direct.x[j]).abs() < 1e-6,
                "x{j}: presolved={} direct={}",
                presolved.x[j],
                direct.x[j]
            );
        }
        let expect_obj = 1.7786363636;
        assert!(
            (presolved.obj_val - expect_obj).abs() < 1e-6,
            "obj_val={} should be ~{} (the true optimum), not ~24.8 from the \
             dimension-desync bug",
            presolved.obj_val,
            expect_obj
        );
        assert!(
            (presolved.obj_val - direct.obj_val).abs() < 1e-6,
            "obj: presolved={} direct={}",
            presolved.obj_val,
            direct.obj_val
        );
    }

    /// fold_epigraph_pairs must reject a pair whose `t` variable appears in a
    /// third (non-pair) inequality row: the fold drops the `t` column outright,
    /// silently removing that coefficient from the problem (and the pair's
    /// meaning). Regression: only the P-coupling of `t` was checked.
    #[test]
    fn epigraph_fold_rejects_t_in_other_rows() {
        // min t s.t. |x| ≤ t (pair rows), −t ≤ −1 (t ≥ 1). True optimum t = 1.
        let n = 4usize;
        let mut a_in = DenseMatrix::<f64>::zeros(3, n);
        a_in.set(0, 0, 1.0);
        a_in.set(0, 1, -1.0); // x − t ≤ 0
        a_in.set(1, 0, -1.0);
        a_in.set(1, 1, -1.0); // −x − t ≤ 0
        a_in.set(2, 1, -1.0); // −t ≤ −1   (t in a third row)
        let prob = QpProblem {
            p: DenseMatrix::<f64>::zeros(n, n),
            q: vec![0.0, 1.0, 0.0, 0.0],
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in: vec![0.0, 0.0, -1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        assert!(
            reductions::fold_epigraph_pairs(&prob, &reductions::SparseAIn::build(&prob)).is_none(),
            "fold must reject a pair whose t appears in a third row"
        );
        // ...and the un-folded problem still solves to the true optimum.
        let sol = solve_qp(&prob, &Settings::<f64>::default());
        assert_eq!(sol.status, Status::Solved, "status={:?}", sol.status);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "t={}", sol.x[1]);
        assert!((sol.obj_val - 1.0).abs() < 1e-6, "obj={}", sol.obj_val);
        // x is free (no curvature); any |x| ≤ t is optimal, so only feasibility.
        assert!(sol.x[0].abs() <= 1.0 + 1e-6, "x={}", sol.x[0]);
    }

    /// A free variable appearing in MORE THAN ONE equality row must not be
    /// substituted out: the substitution is applied sequentially, and a later
    /// elimination can modify another candidate's defining row first, making its
    /// `shift` stale and the reduced problem silently wrong (here: infeasible,
    /// though the true problem is feasible). Regression: row_count ≤ max_rows
    /// allowed multi-row variables through.
    #[test]
    fn free_var_in_multiple_eq_rows_not_eliminated() {
        // x1 in rows r0, r2; x2 in rows r0, r1 (multi-row — must be left alone).
        // Rows: r0: x1 + x2 = 5, r1: 2x2 = 2, r2: 0.5x1 = 2 → x1 = 4, x2 = 1.
        // True: x = (4, 1, 0), obj = ½(16 + 1) = 8.5. (x3 free → 0.)
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0, 0.0],
            a_eq: DenseMatrix::from_row_major(
                3,
                3,
                vec![
                    1.0, 1.0, 0.0, //  x1 + x2 = 5
                    0.0, 2.0, 0.0, //  2x2 = 2
                    0.5, 0.0, 0.0, //  0.5x1 = 2
                ],
            ),
            b_eq: vec![5.0, 2.0, 2.0],
            a_in: DenseMatrix::zeros(0, 3),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        // The gate itself: no elimination may fire.
        let (_, freered) = reductions::eliminate_free_vars(&prob, 12, 48).unwrap();
        assert!(
            freered.elims.is_empty(),
            "multi-row free vars must not be eliminated: {:?}",
            freered.elims
        );
        // End-to-end: the presolve pipeline still solves to the closed form.
        let presolved = solve_presolved(&prob, &Settings::<f64>::default());
        assert_eq!(
            presolved.status,
            Status::Solved,
            "status={:?}",
            presolved.status
        );
        assert!((presolved.x[0] - 4.0).abs() < 1e-6, "x1={}", presolved.x[0]);
        assert!((presolved.x[1] - 1.0).abs() < 1e-6, "x2={}", presolved.x[1]);
        assert!(
            (presolved.obj_val - 8.5).abs() < 1e-6,
            "obj={}",
            presolved.obj_val
        );
    }

    /// Auxiliary-variable elimination must fold the eliminated quadratic into BOTH
    /// triangles of the reduced Hessian: writing only the upper triangle leaves an
    /// asymmetric P (Cholesky reads the lower one), silently solving a different
    /// problem. Regression: an aux-eliminated QP must match the uneliminated solve.
    #[test]
    fn aux_eliminated_qp_matches_uneliminated() {
        // min ½(x1² + x2² + y² + w²)  s.t.  y + 2x1 + 3x2 = 5 (w free → 0).
        // True: x1 = 5/7, x2 = 15/14, y = 5 − 2x1 − 3x2, obj = 25/28.
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(
                4,
                4,
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            ),
            q: vec![0.0, 0.0, 0.0, 0.0],
            a_eq: DenseMatrix::from_row_major(1, 4, vec![2.0, 3.0, 1.0, 0.0]),
            b_eq: vec![5.0],
            a_in: DenseMatrix::zeros(0, 4),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        // Sanity: the aux pass actually fires (y has coeff ±1 in exactly one eq row).
        let (reduced, red) = reductions::eliminate_auxiliary_vars(&prob).unwrap();
        assert!(!red.elims.is_empty(), "expected aux elimination to fire");

        // Reference: the uneliminated problem solved directly.
        let direct = solve_qp(&prob, &Settings::<f64>::default());
        assert_eq!(direct.status, Status::Solved, "status={:?}", direct.status);

        // The eliminated problem solved directly (no further presolve).
        // The Hessian must be symmetric: assert the two triangles agree (the
        // regression wrote only the upper triangle of the folded P).
        for i in 0..reduced.q.len() {
            for j in 0..i {
                assert!(
                    (reduced.p.get(i, j) - reduced.p.get(j, i)).abs() < 1e-12,
                    "reduced P asymmetric at ({i},{j}): {} vs {}",
                    reduced.p.get(i, j),
                    reduced.p.get(j, i)
                );
            }
        }
        let sol_red = solve_qp(
            &reduced,
            &Settings::<f64> {
                presolve: false,
                ..Settings::<f64>::default()
            },
        );
        assert_eq!(
            sol_red.status,
            Status::Solved,
            "status={:?}",
            sol_red.status
        );
        // Restore and compare against the direct solve in original space.
        let restored = reductions::restore_auxiliary_vars(&red, &prob, &sol_red);
        for j in 0..3 {
            assert!(
                (restored.x[j] - direct.x[j]).abs() < 1e-6,
                "x{j}: restored={} direct={}",
                restored.x[j],
                direct.x[j]
            );
        }
        assert!(
            (restored.obj_val - direct.obj_val).abs() < 1e-6,
            "obj: restored={} direct={}",
            restored.obj_val,
            direct.obj_val
        );
        assert!(
            (restored.x[0] - 5.0 / 7.0).abs() < 1e-6,
            "x1={}",
            restored.x[0]
        );
        assert!(
            (restored.x[1] - 15.0 / 14.0).abs() < 1e-6,
            "x2={}",
            restored.x[1]
        );
        assert!(
            (restored.obj_val - 25.0 / 28.0).abs() < 1e-6,
            "obj={}",
            restored.obj_val
        );
    }

    /// The condition-sweep QP shape is diagonal P spanning [1, κ] with dense random rows.
    use iconic_core::rng::Lcg;

    fn cond_sweep(n: usize, kappa: f64, seed: u64) -> QpProblem<f64> {
        let mut rng = Lcg::new(seed);
        let mut p = DenseMatrix::zeros(n, n);
        for j in 0..n {
            p.set(j, j, kappa.powf(j as f64 / (n - 1) as f64));
        }
        let q: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let mut a_in = DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a_in.set(i, j, rng.signed());
            }
        }
        let x0: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
        let ax0 = a_in.matvec(&x0);
        let b_in: Vec<f64> = (0..n).map(|i| ax0[i] + 0.5 + rng.unit()).collect();
        QpProblem {
            p,
            q,
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        }
    }

    fn orig_kkt(prob: &QpProblem<f64>, sol: &QpSolution<f64>) -> f64 {
        let n = prob.q.len();
        let mi = prob.b_in.len();
        let mut worst = 0.0f64;
        let px = prob.p.matvec(&sol.x);
        let atz = prob.a_in.matvec_t(&sol.z);
        for i in 0..n {
            worst = worst.max((px[i] + prob.q[i] + atz[i]).abs());
        }
        let ainx = prob.a_in.matvec(&sol.x);
        for i in 0..mi {
            worst = worst.max((ainx[i] + sol.s[i] - prob.b_in[i]).abs());
            worst = worst.max((sol.s[i] * sol.z[i]).abs());
        }
        worst
    }

    /// The solve-check-solve fallback: the Ruiz-equilibrated problem at κ=1e6
    /// drives its scaled-space iterate into a dual runaway (mu explodes,
    /// ~100x/iteration, honest kkt_res ~76 at an objective 3 orders of
    /// magnitude off) while the un-equilibrated problem converges at machine
    /// accuracy. `solve_presolved` must detect the bad presolved point (early
    /// exit, large original-unit KKT residual) and return the raw retry's
    /// accurate solution instead.
    #[test]
    fn solve_presolved_falls_back_when_equilibration_hurts() {
        let prob = cond_sweep(50, 1e6, 97);
        let settings = Settings::<f64>::default();
        let sol = solve_presolved(&prob, &settings);
        assert_eq!(
            sol.status,
            Status::Solved,
            "κ=1e6 must solve via the raw fallback; got {:?} kkt={:.3e}",
            sol.status,
            orig_kkt(&prob, &sol)
        );
        let kkt = orig_kkt(&prob, &sol);
        assert!(
            kkt < 1e-5,
            "kkt_res {kkt:.3e} — the returned point must be a solution"
        );
        // True objective: 959.03.
        assert!(
            (sol.obj_val - 959.025621).abs() < 1e-3,
            "objective {:.6} vs true 959.025621",
            sol.obj_val
        );
    }

    /// The fallback must NOT fire on a genuine stall (budget burn to a bad
    /// point): retrying there only doubles the cost. The transport shape
    /// (diagonal 0.1·I objective, supply/demand/nonneg rows) stalls at the
    /// iteration cap in both paths.
    #[test]
    fn solve_presolved_does_not_retry_on_budget_burn() {
        // Small transport: 2 suppliers × 3 consumers.
        let mut rng = Lcg::new(99);
        let n_sup = 2usize;
        let n_con = 3usize;
        let n = n_sup * n_con;
        let costs: Vec<f64> = (0..n).map(|_| 1.0 + 19.0 * rng.unit()).collect();
        let supply: Vec<f64> = (0..n_sup).map(|_| 10.0 + 40.0 * rng.unit()).collect();
        let mut demand: Vec<f64> = (0..n_con).map(|_| 10.0 + 40.0 * rng.unit()).collect();
        let ts: f64 = supply.iter().sum();
        let td: f64 = demand.iter().sum();
        for d in demand.iter_mut() {
            *d *= ts * 0.8 / td;
        }
        let idx = |i: usize, j: usize| i * n_con + j;
        let mi = n_sup + n_con + n;
        let mut a_in = DenseMatrix::zeros(mi, n);
        let mut b_in = vec![0.0; mi];
        for i in 0..n_sup {
            for j in 0..n_con {
                a_in.set(i, idx(i, j), 1.0);
            }
            b_in[i] = supply[i];
        }
        for j in 0..n_con {
            for i in 0..n_sup {
                a_in.set(n_sup + j, idx(i, j), -1.0);
            }
            b_in[n_sup + j] = -demand[j];
        }
        for k in 0..n {
            a_in.set(n_sup + n_con + k, k, -1.0);
        }
        let mut p = DenseMatrix::zeros(n, n);
        for k in 0..n {
            p.set(k, k, 0.1);
        }
        let prob = QpProblem {
            p,
            q: costs,
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        };
        // Baseline: the presolved solve alone burns the budget.
        let settings = Settings::<f64> {
            max_iters: 99,
            ..Default::default()
        };
        let sol = solve_presolved(&prob, &settings);
        // Either outcome is acceptable here — Solved (the barrier-adaptive
        // μ²-gate fix on weak-(1,1) shapes now converges this transport, so the
        // stall the test was written against no longer fires) or
        // SolvedInaccurate (a genuine budget-burn stall). What must never
        // happen: the solve-check-solve fallback retrying past the budget and
        // doubling the iteration count.
        assert!(
            sol.status == Status::Solved || sol.status == Status::SolvedInaccurate,
            "status={:?}",
            sol.status
        );
        assert!(
            sol.iters <= 99,
            "transport-style stall must not retry (iters {} > budget)",
            sol.iters
        );
    }

    /// Regression: a tall-thin ill-conditioned QP (n=10, 15 dense rows,
    /// diagonal P spanning [1, 1e8] — the CVXPY canonical shape) stalls on the
    /// condensed-Gram QP path (kkt ~7.8, obj 0.0, SolvedInaccurate at 20 iters)
    /// while the conic engine's quasidefinite augmented system solves it at
    /// machine accuracy. The solve-check-solve fallback's conic tier must
    /// recover the exact optimum.
    #[test]
    fn solve_presolved_conic_tier_recovers_ill_conditioned_tall_qp() {
        // Shape/seed chosen by sweep: n=10, mi=15, kexp=9, seed=31 is the
        // minimal Rust-generatable instance where the QP path returns a bad
        // point (kkt 3.3, SolvedInaccurate) while the conic tier recovers the
        // exact optimum — the premise assertion below guards that property.
        let n = 10usize;
        let mi = 15usize;
        // Deliberately NOT iconic_core::rng::Lcg: the shape/seed pair was picked
        // by a sweep against THIS generator's exact stream ("n=10, mi=15,
        // kexp=9, seed=31 is the minimal instance where the QP path returns a
        // bad point while the conic tier recovers"). A different stream changes
        // the instance and voids the calibrated premise below.
        let mut seed = 31u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut p = DenseMatrix::zeros(n, n);
        for j in 0..n {
            p.set(j, j, 10f64.powf(9.0 * j as f64 / (n - 1) as f64));
        }
        let q: Vec<f64> = (0..n).map(|_| rnd()).collect();
        let x0: Vec<f64> = (0..n).map(|_| rnd()).collect();
        let mut a_in = DenseMatrix::zeros(mi, n);
        for i in 0..mi {
            for j in 0..n {
                a_in.set(i, j, rnd());
            }
        }
        let ax0 = a_in.matvec(&x0);
        let b_in: Vec<f64> = (0..mi).map(|i| ax0[i] + 0.5 + rnd().abs()).collect();
        let prob = QpProblem {
            p,
            q,
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_presolved(&prob, &Settings::<f64>::default());
        assert_eq!(sol.status, Status::Solved, "status={:?}", sol.status);
        let kkt = orig_kkt_residual(&prob, &sol);
        assert!(
            kkt < 1e-4,
            "kkt_res {kkt:.3e} — the conic tier must recover an accurate point"
        );
        // The QP path alone must genuinely fail on this shape (the premise of
        // the fallback): otherwise the test is testing nothing.
        let raw_settings = Settings::<f64> {
            presolve: false,
            ..Default::default()
        };
        let raw = solve_qp(&prob, &raw_settings);
        assert_ne!(
            raw.status,
            Status::Solved,
            "premise broken: QP path solved it"
        );
        assert!(orig_kkt_residual(&prob, &raw) > 1e-3);
    }
}
#[cfg(test)]
mod equil_sparse_regression {
    use super::*;
    use iconic_linalg::DenseMatrix;

    /// The pre-CSR-mirror implementation of [`equilibrate_coned_sp`] (git HEAD
    /// at the perf(presolve) CSR round), kept verbatim as the regression oracle
    /// for the sparse path. The new implementation must be bit-identical to it
    /// on every input (same sweeps, same factors, same scaled problem, same
    /// Scaling), including the multi-sweep case the suite never exercises
    /// (the sparse-view gate fires only on the single-sweep transport LPs).
    fn equilibrate_coned_sp_old<T: Scalar>(
        prob: &QpProblem<T>,
        iters: usize,
        ineq_dims: &[usize],
        tol: T,
        sp: &reductions::SparseAIn<T>,
    ) -> (QpProblem<T>, Scaling<T>) {
        let n = prob.q.len();
        let me = prob.b_eq.len();
        let mi = prob.b_in.len();
        let zero = T::zero();
        let one = T::one();

        let trace = std::env::var_os("ICONIC_TRACE_PRESOLVE").is_some();
        let t_trace = std::time::Instant::now();
        let mut p = prob.p.clone();
        let mut a_eq = prob.a_eq.clone();
        let mut a_in = prob.a_in.clone();
        let mut q = prob.q.clone();
        let mut b_eq = prob.b_eq.clone();
        let mut b_in = prob.b_in.clone();
        if trace {
            eprintln!(
                "[ruiz clones] {:>9.1} ms",
                t_trace.elapsed().as_secs_f64() * 1e3
            );
        }

        let mut d = vec![one; n];
        let mut e_eq = vec![one; me];
        let mut e_in = vec![one; mi];

        // Pre-allocated scaling buffers reused each Ruiz sweep.
        let mut dd = vec![T::zero(); n];
        let mut de = vec![T::zero(); me];
        let mut di = vec![T::zero(); mi];
        // A diagonal P (the common SDP/transport shape — dense-stored, n²
        // entries for n nonzeros) is detected once so the per-sweep P scans and
        // applies are O(n) instead of O(n²): 5 sweeps × 2 × n² reads/writes on
        // the transport's 4800×4800 diagonal P is ~1s of pure memory traffic.
        // The detection early-exits on the first off-diagonal nonzero, so
        // genuinely dense P problems pay one element check.
        let p_diag = (0..n).all(|i| (0..n).all(|j| i == j || p.get(i, j) == zero));
        let t_sweeps = std::time::Instant::now();
        for _ in 0..iters {
            // Column (variable) ∞-norms of P: scan row-by-row (cache-friendly for
            // row-major P) and take the column max — P is symmetric so this is equivalent.
            let mut col = vec![zero; n];
            if p_diag {
                for i in 0..n {
                    col[i] = p.get(i, i).abs();
                }
            } else {
                for i in 0..n {
                    let mut mx = zero;
                    for j in 0..n {
                        mx = mx.max(p.get(i, j).abs());
                    }
                    col[i] = mx;
                }
            }
            // Constraint row/col ∞-norms in a single row-major pass per matrix:
            // for each row r, update the row max and scatter into col[j] (the
            // scattered writes are dwarfed by the row-major reads).
            let mut row_eq = vec![zero; me];
            for r in 0..me {
                let mut mx = zero;
                for j in 0..n {
                    let v = a_eq.get(r, j).abs();
                    mx = mx.max(v);
                    col[j] = col[j].max(v);
                }
                row_eq[r] = mx;
            }
            let mut row_in = vec![zero; mi];
            for r in 0..mi {
                let mut mx = zero;
                match sp.row(r) {
                    Some(nz) => {
                        for &(j, v) in nz {
                            let av = v.abs();
                            mx = mx.max(av);
                            col[j] = col[j].max(av);
                        }
                    }
                    None => {
                        for j in 0..n {
                            let v = a_in.get(r, j).abs();
                            mx = mx.max(v);
                            col[j] = col[j].max(v);
                        }
                    }
                }
                row_in[r] = mx;
            }

            for (out, &x) in dd.iter_mut().zip(col.iter()) {
                *out = scale_factor(x);
            }
            for (out, &x) in de.iter_mut().zip(row_eq.iter()) {
                *out = scale_factor(x);
            }
            for (out, &x) in di.iter_mut().zip(row_in.iter()) {
                *out = scale_factor(x);
            }

            // Rectify the inequality-row scaling per cone: a shared geometric-mean factor
            // across each multi-dim cone block, so SOC/PSD membership is preserved.
            let mut off = 0usize;
            for &d in ineq_dims {
                if d > 1 {
                    let mut logsum = zero;
                    for r in off..off + d {
                        logsum += di[r].ln();
                    }
                    let gm = (logsum / T::from_usize(d).unwrap()).exp();
                    for r in off..off + d {
                        di[r] = gm;
                    }
                }
                off += d;
            }

            // Apply: P ← DPD, A ← EAD, q ← Dq, b ← Eb.
            if p_diag {
                for i in 0..n {
                    p.set(i, i, p.get(i, i) * dd[i] * dd[i]);
                }
            } else {
                for i in 0..n {
                    // `dd[i]` is invariant across j; bind once per row.
                    let di = dd[i];
                    for j in 0..n {
                        p.set(i, j, p.get(i, j) * di * dd[j]);
                    }
                }
            }
            for i in 0..n {
                q[i] *= dd[i];
            }
            for r in 0..me {
                let dr = de[r];
                for j in 0..n {
                    a_eq.set(r, j, a_eq.get(r, j) * dr * dd[j]);
                }
                b_eq[r] *= de[r];
            }
            if sp.is_sparse() {
                // Sparse apply: zeros stay zero under the scaling, so only the
                // nonzeros need touching (13k writes vs 23.7M on the transport).
                for r in 0..mi {
                    if let Some(nz) = sp.row(r) {
                        for &(j, v) in nz {
                            a_in.set(r, j, v * di[r] * dd[j]);
                        }
                    }
                }
            } else {
                for r in 0..mi {
                    let dr = di[r];
                    for j in 0..n {
                        a_in.set(r, j, a_in.get(r, j) * dr * dd[j]);
                    }
                }
            }
            for r in 0..mi {
                b_in[r] *= di[r];
            }

            for i in 0..n {
                d[i] *= dd[i];
            }
            for r in 0..me {
                e_eq[r] *= de[r];
            }
            for r in 0..mi {
                e_in[r] *= di[r];
            }

            // Convergence-aware stop: once a sweep's scale factors are
            // all ≈ 1 the data is equilibrated, so the remaining fixed sweeps are wasted work.
            let conv = dd
                .iter()
                .chain(de.iter())
                .chain(di.iter())
                .fold(zero, |m, &x| m.max((x - one).abs()));
            if conv < tol {
                break;
            }
        }
        if trace {
            eprintln!(
                "[ruiz sweeps] {:>9.1} ms",
                t_sweeps.elapsed().as_secs_f64() * 1e3
            );
        }
        // Cost scaling c = 1 / max(mean column ∞-norm of P, ‖q‖∞).
        let mut psum = zero;
        if p_diag {
            for j in 0..n {
                psum += p.get(j, j).abs();
            }
        } else {
            for j in 0..n {
                let mut mx = zero;
                for i in 0..n {
                    mx = mx.max(p.get(i, j).abs());
                }
                psum += mx;
            }
        }
        let pmean = if n > 0 {
            psum / T::from_usize(n).unwrap()
        } else {
            one
        };
        let qn = q.iter().fold(zero, |a, &b| a.max(b.abs()));
        let denom = pmean.max(qn);
        let c = if denom > zero { one / denom } else { one };
        if p_diag {
            for j in 0..n {
                p.set(j, j, p.get(j, j) * c);
            }
        } else {
            for i in 0..n {
                for j in 0..n {
                    p.set(i, j, p.get(i, j) * c);
                }
            }
        }
        for i in 0..n {
            q[i] *= c;
        }

        // Keep the CSR alive through equilibration. Ruiz's apply only touches
        // nonzeros (the sparse apply), so the structure is unchanged — only the
        // values are scaled. Build the scaled CSR from the sparse views in O(nnz);
        // the solver then never pays the O(mi·n) `csr_of_dense` re-derivation
        // (measured ~50ms on the 4940×4800 transport LP).
        let a_in_csr = if sp.is_sparse() {
            let zero = T::zero();
            let mut colptr = vec![0usize; mi + 1];
            let mut rowval = Vec::new();
            let mut nzval = Vec::new();
            for r in 0..mi {
                for &(j, _) in sp.row(r).unwrap_or(&[]) {
                    let v = a_in.get(r, j);
                    if v != zero {
                        rowval.push(j);
                        nzval.push(v);
                    }
                }
                colptr[r + 1] = rowval.len();
            }
            Some(iconic_linalg::CscMatrix {
                m: n,
                n: mi,
                colptr,
                rowval,
                nzval,
            })
        } else {
            None
        };
        (
            QpProblem {
                p,
                q,
                a_eq,
                b_eq,
                a_in,
                b_in,
                a_eq_csr: None,
                a_in_csr,
            },
            Scaling { d, e_eq, e_in, c },
        )
    }

    /// bitwise-equal helper (f64 == treats -0.0 == +0.0; the regression must be
    /// byte-identical, not just value-equal).
    fn bits_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn sparse_mirror_bit_identical_multisweep() {
        // A problem that fires the sparse-view gate (mi·n >= 1M, nnz·2 < mi·n)
        // AND needs multiple Ruiz sweeps (badly scaled rows/columns, so no
        // sweep's factors are all ~1). Diagonal P (the transport shape) so the
        // p_cur mirror path is exercised across sweeps too.
        let n = 1100usize;
        let mi = 1100usize;
        let mut a_in = DenseMatrix::zeros(mi, n);
        let mut b_in = vec![0.0; mi];
        for r in 0..mi {
            let scale = 10f64.powi((r % 9) as i32 - 4); // 1e-4 .. 1e4
            let j0 = (r * 7) % n;
            let j1 = (r * 13 + 5) % n;
            a_in.set(r, j0, 1.0 * scale);
            if j1 != j0 {
                a_in.set(r, j1, -0.7 * scale * (1.0 + (r % 3) as f64 * 1e-3));
            }
            b_in[r] = scale * (2.0 + (r % 5) as f64);
        }
        let mut p = DenseMatrix::zeros(n, n);
        let q: Vec<f64> = (0..n).map(|j| 0.5 * (1.0 + (j % 7) as f64)).collect();
        for i in 0..n {
            p.set(i, i, 1.0 + (i % 5) as f64 * 0.1);
        }
        let prob = QpProblem {
            p,
            q,
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sp = reductions::SparseAIn::build(&prob);
        assert!(sp.is_sparse(), "the gate should fire");
        let dims: Vec<usize> = vec![1usize; prob.b_in.len()];
        let (new_prob, new_sc) = equilibrate_coned_sp(&prob, 5, &dims, 1e-2, &sp);
        let (old_prob, old_sc) = equilibrate_coned_sp_old(&prob, 5, &dims, 1e-2, &sp);

        assert!(bits_eq(&new_prob.p.data, &old_prob.p.data), "P differs");
        assert!(bits_eq(&new_prob.q, &old_prob.q), "q differs");
        assert!(
            bits_eq(&new_prob.a_in.data, &old_prob.a_in.data),
            "A_in differs"
        );
        assert!(bits_eq(&new_prob.b_in, &old_prob.b_in), "b_in differs");
        assert!(bits_eq(&new_sc.d, &old_sc.d), "d differs");
        assert!(bits_eq(&new_sc.e_in, &old_sc.e_in), "e_in differs");
        assert!((new_sc.c - old_sc.c).abs() == 0.0, "c differs");
        let (nc, oc) = (
            new_prob.a_in_csr.as_ref().unwrap(),
            old_prob.a_in_csr.as_ref().unwrap(),
        );
        assert_eq!(nc.colptr, oc.colptr, "csr colptr differs");
        assert_eq!(nc.rowval, oc.rowval, "csr rowval differs");
        assert!(bits_eq(&nc.nzval, &oc.nzval), "csr nzval differs");
    }

    #[test]
    fn sparse_mirror_bit_identical_dense_p_multisweep() {
        // Same multi-sweep sparse setup but with a genuinely dense P (one
        // off-diagonal coupling), so the p_diag=false P path (the dense
        // buffer apply — unchanged code) and the A_in mirror both run.
        let n = 1100usize;
        let mi = 1100usize;
        let mut a_in = DenseMatrix::zeros(mi, n);
        let mut b_in = vec![0.0; mi];
        for r in 0..mi {
            let scale = 10f64.powi((r % 9) as i32 - 4); // 1e-4 .. 1e4
            let j0 = (r * 7) % n;
            let j1 = (r * 13 + 5) % n;
            a_in.set(r, j0, 1.0 * scale);
            if j1 != j0 {
                a_in.set(r, j1, -0.7 * scale * (1.0 + (r % 3) as f64 * 1e-3));
            }
            b_in[r] = scale * (2.0 + (r % 5) as f64);
        }
        let mut p = DenseMatrix::zeros(n, n);
        let q: Vec<f64> = (0..n).map(|j| 0.5 * (1.0 + (j % 7) as f64)).collect();
        for i in 0..n {
            p.set(i, i, 1.0 + (i % 5) as f64 * 0.1);
        }
        p.set(0, 1, 0.3); // the coupling that makes P non-diagonal
        p.set(1, 0, 0.3);
        let prob = QpProblem {
            p,
            q,
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sp = reductions::SparseAIn::build(&prob);
        assert!(sp.is_sparse(), "the gate should fire");
        let dims: Vec<usize> = vec![1usize; prob.b_in.len()];
        let (new_prob, new_sc) = equilibrate_coned_sp(&prob, 5, &dims, 1e-2, &sp);
        let (old_prob, old_sc) = equilibrate_coned_sp_old(&prob, 5, &dims, 1e-2, &sp);

        assert!(bits_eq(&new_prob.p.data, &old_prob.p.data), "P differs");
        assert!(bits_eq(&new_prob.q, &old_prob.q), "q differs");
        assert!(
            bits_eq(&new_prob.a_in.data, &old_prob.a_in.data),
            "A_in differs"
        );
        assert!(bits_eq(&new_prob.b_in, &old_prob.b_in), "b_in differs");
        assert!(bits_eq(&new_sc.d, &old_sc.d), "d differs");
        assert!(bits_eq(&new_sc.e_in, &old_sc.e_in), "e_in differs");
        assert!((new_sc.c - old_sc.c).abs() == 0.0, "c differs");
        let (nc, oc) = (
            new_prob.a_in_csr.as_ref().unwrap(),
            old_prob.a_in_csr.as_ref().unwrap(),
        );
        assert_eq!(nc.colptr, oc.colptr, "csr colptr differs");
        assert_eq!(nc.rowval, oc.rowval, "csr rowval differs");
        assert!(bits_eq(&nc.nzval, &oc.nzval), "csr nzval differs");
    }
}
