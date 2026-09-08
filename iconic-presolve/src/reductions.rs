//! Constraint-row reduction passes and their postsolve.
//!
//! [`reduce_rows`] drops, from the constraint system:
//!  - **null rows** (structurally zero) — redundant, or contradictory → infeasible;
//!  - **dominated / parallel inequality rows** — among rows with the same direction
//!    only the tightest bound matters; the looser ones are redundant;
//!  - **duplicate inequality rows** — identical rows collapse to one.
//!
//! It also detects **anti-parallel contradictions** (`aᵀx ≤ u` with `aᵀx ≥ l`,
//! `l > u`) as primal infeasibility.
//!
//! [`restore_rows`] reconstructs the original-dimension solution: a dropped row gets
//! dual `0` (it is inactive at the kept solution) and its slack recomputed from the
//! recovered primal (`s = b − aᵀx`, or `s = b` for a null row).

use iconic_core::{Scalar, Status};
use iconic_ipm::{QpProblem, QpSolution};
use iconic_linalg::DenseMatrix;

/// Record of the surviving / dropped rows, enough to rebuild the original-dimension
/// solution in [`restore_rows`].
#[derive(Clone, Debug)]
pub struct RowReduction<T: Scalar> {
    kept_eq: Vec<usize>,
    kept_in: Vec<usize>,
    /// Null inequality rows `(old_row, b)`; slack reconstructs as `b`.
    dropped_null: Vec<(usize, T)>,
    /// Dominated/duplicate inequality rows; slack reconstructs as `b − aᵀx`.
    dropped_resid: Vec<usize>,
    a_in_orig: DenseMatrix<T>,
    b_in_orig: Vec<T>,
    n_eq_orig: usize,
    n_in_orig: usize,
}

impl<T: Scalar> RowReduction<T> {
    /// True when any row was dropped.
    pub(crate) fn changed(&self) -> bool {
        !self.dropped_null.is_empty() || !self.dropped_resid.is_empty()
    }
}

/// Sparse row/column views of `A_in` for the reduction passes, built ONCE per
/// presolve round from a single dense pass. The passes iterate these instead
/// of the dense matrix: a 99.94%-sparse 4940×4800 `A_in` costs ~0.5–0.9s of
/// dense scans per pass (measured on the transport LPs) vs ~0.05s for the
/// sparse views. Built only when the matrix is large and genuinely sparse
/// (the gate); small or dense problems keep the dense scans, bit-identical.
pub struct SparseAIn<T: Scalar> {
    /// Row-major nonzeros: `rows[r]` = A_in's `(col, value)` pairs in row r.
    rows: Vec<Vec<(usize, T)>>,
    /// Column-major nonzeros: `cols[c]` = A_in's `(row, value)` pairs in column c.
    cols: Vec<Vec<(usize, T)>>,
    /// True when the views are populated; false = iterate the dense matrix.
    sparse: bool,
}

impl<T: Scalar> SparseAIn<T> {
    /// Build the views from a single dense pass over `A_in`, gated on size
    /// and density. The extraction is one O(mi·n) scan — amortized over the
    /// several passes that would otherwise each pay it. The nnz count that
    /// gates the dense fallback is folded into the same scan (the extraction
    /// aborts, discarding the partial views, the moment it has counted
    /// `ceil(total/2)` nonzeros — the exact `nnz·2 >= total` decision, so the
    /// dense fallback engages bit-identically without a separate count pass).
    pub fn build(prob: &QpProblem<T>) -> Self {
        let n = prob.q.len();
        let mi = prob.b_in.len();
        let total = mi * n;
        if total < 1_000_000 {
            return Self {
                rows: Vec::new(),
                cols: Vec::new(),
                sparse: false,
            };
        }
        if let Some(csr) = &prob.a_in_csr {
            if csr.rowval.len() * 2 >= total {
                return Self {
                    rows: Vec::new(),
                    cols: Vec::new(),
                    sparse: false,
                };
            }
            return Self::from_csr(csr, mi, n);
        }
        // One dense pass: extract rows/cols, aborting at the density gate.
        let mut rows = vec![Vec::new(); mi];
        let mut cols = vec![Vec::new(); n];
        let mut nnz = 0usize;
        let dense_gate = total.div_ceil(2); // nnz·2 >= total ⇔ nnz >= ceil(total/2)
        for r in 0..mi {
            for j in 0..n {
                let v = prob.a_in.get(r, j);
                if v != T::zero() {
                    rows[r].push((j, v));
                    cols[j].push((r, v));
                    nnz += 1;
                    if nnz >= dense_gate {
                        return Self {
                            rows: Vec::new(),
                            cols: Vec::new(),
                            sparse: false,
                        };
                    }
                }
            }
        }
        Self {
            rows,
            cols,
            sparse: true,
        }
    }

    /// Build the views from an existing `a_in_csr` in O(nnz): the CSR is
    /// already row-major with ascending column indices per row (both
    /// `csr_of_dense` and the CVXPY-side `csc_submatrix_rows` build it that
    /// way), so the rows are direct copies of the `colptr` ranges and only
    /// the column index needs one scatter pass. Preserves the dense build's
    /// exact-zero skip (a CVXPY-side CSC can carry explicit zeros).
    fn from_csr(csr: &iconic_linalg::CscMatrix<T>, mi: usize, n: usize) -> Self {
        let zero = T::zero();
        let mut rows = vec![Vec::new(); mi];
        let mut cols = vec![Vec::new(); n];
        for r in 0..mi {
            let row = &mut rows[r];
            for p in csr.colptr[r]..csr.colptr[r + 1] {
                let v = csr.nzval[p];
                if v != zero {
                    let j = csr.rowval[p];
                    row.push((j, v));
                    cols[j].push((r, v));
                }
            }
        }
        Self {
            rows,
            cols,
            sparse: true,
        }
    }

    /// The forced-dense fallback (used by the equivalence test; the passes
    /// then iterate the dense matrix exactly as before the sparse views).
    pub fn dense() -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            sparse: false,
        }
    }

    /// True when the sparse views are populated (the Ruiz's in-place apply
    /// then touches only the nonzeros).
    pub fn is_sparse(&self) -> bool {
        self.sparse
    }

    /// Nonzeros of row `r`, or `None` when the dense fallback is active.
    #[inline]
    pub fn row(&self, r: usize) -> Option<&[(usize, T)]> {
        if self.sparse {
            Some(&self.rows[r])
        } else {
            None
        }
    }

    /// Nonzeros of column `c`, or `None` when the dense fallback is active.
    #[inline]
    pub fn col(&self, c: usize) -> Option<&[(usize, T)]> {
        if self.sparse {
            Some(&self.cols[c])
        } else {
            None
        }
    }
}

/// Build the CSR of a rebuilt A_in (`new row i = old row kept_rows[i]`, `new
/// column = col_map[old column]`, `usize::MAX` dropping the column) from the
/// sparse views in O(nnz). The reduction passes rebuild A_in densely anyway;
/// emitting the CSR in the same pass keeps it alive through presolve so the
/// solver never re-derives it with an O(mi·n) `csr_of_dense` scan. Returns
/// `None` when the views are dense — the caller then leaves `a_in_csr: None`
/// exactly as before. Preserves the zero-skip of the dense builders.
pub(crate) fn csr_from_rows_cols<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
    kept_rows: &[usize],
    col_map: &[usize],
    mi_new: usize,
    n_new: usize,
) -> Option<iconic_linalg::CscMatrix<T>> {
    if !sp.is_sparse() {
        return None;
    }
    let zero = T::zero();
    let mut colptr = vec![0usize; mi_new + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for (ni, &or) in kept_rows.iter().enumerate() {
        for &(j, _) in sp.row(or).unwrap_or(&[]) {
            let nj = col_map[j];
            if nj == usize::MAX {
                continue;
            }
            let v = prob.a_in.get(or, j);
            if v != zero {
                rowval.push(nj);
                nzval.push(v);
            }
        }
        colptr[ni + 1] = rowval.len();
    }
    Some(iconic_linalg::CscMatrix {
        m: n_new,
        n: mi_new,
        colptr,
        rowval,
        nzval,
    })
}

/// Identity-column variant of [`csr_from_rows_cols`] for passes that keep all
/// columns (fold, row reductions).
pub(crate) fn csr_from_rows<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
    kept_rows: &[usize],
    mi_new: usize,
    n: usize,
) -> Option<iconic_linalg::CscMatrix<T>> {
    let col_map: Vec<usize> = (0..n).collect();
    csr_from_rows_cols(prob, sp, kept_rows, &col_map, mi_new, n)
}

fn tol_coef<T: Scalar>() -> T {
    T::from_f64(1e-10).expect("scalar literal")
}
fn tol_rhs<T: Scalar>() -> T {
    T::from_f64(1e-9).expect("scalar literal")
}
fn tol_parallel<T: Scalar>() -> T {
    T::from_f64(1e-9).expect("scalar literal")
}

/// A row is null if every entry is below a relative threshold (so a genuine row,
/// whose largest entry equals its scale, is never falsely dropped).
fn row_is_null<T: Scalar>(a: &DenseMatrix<T>, r: usize) -> bool {
    let mut scale = T::one();
    for j in 0..a.ncols {
        scale = scale.max(a.get(r, j).abs());
    }
    let thr = tol_coef::<T>() * scale;
    (0..a.ncols).all(|j| a.get(r, j).abs() <= thr)
}

fn copy_row<T: Scalar>(dst: &mut DenseMatrix<T>, dst_r: usize, src: &DenseMatrix<T>, src_r: usize) {
    let n = src.ncols;
    let d = dst.data_mut();
    d[dst_r * n..(dst_r + 1) * n].copy_from_slice(&src.data[src_r * n..(src_r + 1) * n]);
}

/// Canonicalize inequality row `r` (`aᵀx ≤ b`) by its signed largest-magnitude entry
/// `c = a[p]`: returns the unit direction `û = a/c` (so `û[p] = 1`), the pivot index
/// `p`, the direction `dir = sign(c)` (+1 ⇒ upper bound, −1 ⇒ lower bound after the
/// divide), and the normalized bound `b̂ = b/c`.
/// The canonicalization core: the row's largest-magnitude entry (the pivot
/// column `p` and its signed value `c`). Every variant below divides through
/// by `c`, so the shared scan keeps them bit-identical.
fn pivot_of<T: Scalar>(pv: impl Fn(usize) -> T, n: usize, zero: T) -> (usize, T) {
    let mut p = 0usize;
    let mut mx = zero;
    let mut c = zero;
    for j in 0..n {
        let v = pv(j);
        if v.abs() > mx {
            mx = v.abs();
            p = j;
            c = v;
        }
    }
    (p, c)
}

fn canon<T: Scalar>(a: &DenseMatrix<T>, r: usize, b: T, n: usize) -> (Vec<T>, usize, T, T) {
    let zero = T::zero();
    let (p, c) = pivot_of(|j| a.get(r, j), n, zero);
    let inv_c = c.recip();
    let uhat: Vec<T> = (0..n).map(|j| a.get(r, j) * inv_c).collect();
    let dir = if c > zero { T::one() } else { -T::one() };
    (uhat, p, dir, b / c)
}

/// A canonical-direction group of candidate rows sharing one pivot. The unit
/// direction `uhat` is materialized **lazily**: a single-member group (every
/// row on the big-sparse transport LPs has its own pivot, so the groups never
/// compare against anything) never pays the O(n) direction vector — the eager
/// build stored `mi × n` floats (190MB on the 4940×4800 transport) that were
/// only ever compared against nothing. On a second member's arrival the stored
/// source row is re-canonicalized — bit-identical to the eager build.
struct PivotGroup<T: Scalar> {
    /// Canonical direction of the first member, materialized on first compare.
    uhat: Option<Vec<T>>,
    /// The member row that produced `uhat` (the lazy re-canonicalization source).
    src_row: usize,
    /// `(row, dir, b̂)` members.
    members: Vec<(usize, T, T)>,
}

impl<T: Scalar> PivotGroup<T> {
    /// Materialize (once) and return the group's canonical direction — the
    /// same `canon_row`/`canon` the eager build called at creation time on the
    /// same source row, so the comparison values are bit-identical.
    fn ensure_uhat(&mut self, prob: &QpProblem<T>, sp: &SparseAIn<T>, n: usize) -> &[T] {
        if self.uhat.is_none() {
            let r = self.src_row;
            self.uhat = Some(row_uhat(prob, sp, r, n));
        }
        self.uhat.as_ref().expect("upper-triangular form requested at build")
    }
}

/// Sparse-row variant of [`canon`]: same unit direction `û = a/c`, pivot,
/// direction, and normalized bound, from the row's nonzero `(col, value)`
/// list. The dense version divides every entry by `c` (zeros stay zero);
/// this skips the zero entries, which the tolerance-based `vec_close` group
/// tests treat identically.
fn canon_row<T: Scalar>(
    nz: &[(usize, T)],
    b: T,
    n: usize,
    zero: T,
    one: T,
) -> (Vec<T>, usize, T, T) {
    let (p, c) = pivot_of(|k| nz[k].1, nz.len(), zero);
    let mut uhat: Vec<T> = vec![zero; n];
    for &(j, v) in nz {
        uhat[j] = v / c;
    }
    let dir = if c > zero { one } else { -one };
    (uhat, p, dir, b / c)
}

/// Pivot, direction, and normalized bound only — the [`canon_row`] direction
/// vector is built separately and only when a same-pivot group exists to
/// compare against (the lazy-grouping passes).
fn canon_row_dir<T: Scalar>(nz: &[(usize, T)], b: T, zero: T, one: T) -> (usize, T, T) {
    let (p, c) = pivot_of(|k| nz[k].1, nz.len(), zero);
    let dir = if c > zero { one } else { -one };
    (p, dir, b / c)
}

/// Dense variant of [`canon_row_dir`]: pivot/direction/normalized bound from
/// a dense row without building the direction vector.
fn canon_dir<T: Scalar>(a: &DenseMatrix<T>, r: usize, b: T) -> (usize, T, T) {
    let zero = T::zero();
    let one = T::one();
    let (p, c) = pivot_of(|j| a.get(r, j), a.ncols, zero);
    let dir = if c > zero { one } else { -one };
    (p, dir, b / c)
}

/// The direction vector of row `r` — the lazy grouping materialization path:
/// same `canon_row`/`canon` the eager build called on the same source row.
fn row_uhat<T: Scalar>(prob: &QpProblem<T>, sp: &SparseAIn<T>, r: usize, n: usize) -> Vec<T> {
    let zero = T::zero();
    let one = T::one();
    match sp.row(r) {
        Some(nz) => canon_row(nz, prob.b_in[r], n, zero, one).0,
        None => canon(&prob.a_in, r, prob.b_in[r], n).0,
    }
}

fn vec_close<T: Scalar>(a: &[T], b: &[T], tol: T) -> bool {
    a.iter().zip(b).all(|(&x, &y)| (x - y).abs() <= tol)
}

/// Run the row-reduction passes. Returns the reduced problem and a [`RowReduction`],
/// or `Err(Status::PrimalInfeasible)` on a contradictory (null or anti-parallel) row.
pub fn reduce_rows<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
) -> Result<(QpProblem<T>, RowReduction<T>), Status> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let one = T::one();
    let trhs = tol_rhs::<T>();

    // Equalities: drop null rows, detect 0 = b ≠ 0.
    let mut kept_eq = Vec::new();
    for r in 0..me {
        if row_is_null(&prob.a_eq, r) {
            if prob.b_eq[r].abs() > trhs {
                return Err(Status::PrimalInfeasible);
            }
        } else {
            kept_eq.push(r);
        }
    }

    // Inequalities: classify null vs candidate (non-null) rows. Row scans go
    // through the sparse views when built (see `fold_negated_pairs`); the
    // dense fallback is bit-identical.
    let mut dropped_null = Vec::new();
    let mut candidates = Vec::new();
    for r in 0..mi {
        // Null test identical to `row_is_null` (see `fold_negated_pairs`):
        // null iff every entry is <= 1e-10, which for the sparse row means
        // the row's max <= 1e-10 (the extraction keeps tiny-but-nonzero
        // entries, and `row_is_null`'s scale = max(1, row max) makes that
        // the exact condition).
        let is_null = match sp.row(r) {
            Some(nz) => {
                let mx = nz.iter().fold(zero, |m, &(_, v)| m.max(v.abs()));
                mx <= tol_coef::<T>()
            }
            None => row_is_null(&prob.a_in, r),
        };
        if is_null {
            if prob.b_in[r] < -trhs {
                return Err(Status::PrimalInfeasible);
            }
            dropped_null.push((r, prob.b_in[r]));
        } else {
            candidates.push(r);
        }
    }

    // Group candidates by canonical direction; within a group keep the tightest
    // bound per side and detect anti-parallel contradictions. Same pivot→group
    // index as `fold_negated_pairs`: O(1) lookups instead of an O(mi) scan per
    // row (24M pivot compares on the transport LP), and lazily-materialized
    // direction vectors (the all-unique-pivot shape stores no uhat at all).
    let ptol = tol_parallel::<T>();
    let mut groups: Vec<PivotGroup<T>> = Vec::new();
    let mut group_by_pivot: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for &r in &candidates {
        let (p, dir, bhat) = match sp.row(r) {
            Some(nz) => canon_row_dir(nz, prob.b_in[r], zero, one),
            None => canon_dir(&prob.a_in, r, prob.b_in[r]),
        };
        let mut gi = None;
        if let Some(ids) = group_by_pivot.get(&p) {
            let uhat = row_uhat(prob, sp, r, n);
            for &k in ids {
                if vec_close(groups[k].ensure_uhat(prob, sp, n), &uhat, ptol) {
                    gi = Some(k);
                    break;
                }
            }
        }
        match gi {
            Some(k) => groups[k].members.push((r, dir, bhat)),
            None => {
                groups.push(PivotGroup {
                    uhat: None,
                    src_row: r,
                    members: vec![(r, dir, bhat)],
                });
                group_by_pivot.entry(p).or_default().push(groups.len() - 1);
            }
        }
    }

    let mut kept_in = Vec::new();
    let mut dropped_resid = Vec::new();
    for g in &groups {
        let mut upper: Option<(T, usize)> = None; // tightest (min b̂) of the +dir side
        let mut lower: Option<(T, usize)> = None; // tightest (max b̂) of the −dir side
        for &(r, dir, bhat) in &g.members {
            if dir > zero {
                if upper.is_none_or(|(ub, _)| bhat < ub) {
                    upper = Some((bhat, r));
                }
            } else if lower.is_none_or(|(lb, _)| bhat > lb) {
                lower = Some((bhat, r));
            }
        }
        if let (Some((ub, _)), Some((lb, _))) = (upper, lower) {
            let scale = one.max(ub.abs()).max(lb.abs());
            if lb - ub > trhs * scale {
                return Err(Status::PrimalInfeasible);
            }
        }
        let rep_u = upper.map(|(_, r)| r);
        let rep_l = lower.map(|(_, r)| r);
        for &(r, _, _) in &g.members {
            if Some(r) == rep_u || Some(r) == rep_l {
                kept_in.push(r);
            } else {
                dropped_resid.push(r);
            }
        }
    }
    kept_in.sort_unstable();

    // Nothing dropped: return the input unchanged — avoids the O(n²) P clone and
    // the record's A/B clones (the common well-formed case).
    if kept_eq.len() == me && kept_in.len() == mi {
        return Ok((
            prob.clone(),
            RowReduction {
                kept_eq,
                kept_in,
                dropped_null: Vec::new(),
                dropped_resid: Vec::new(),
                a_in_orig: DenseMatrix::zeros(0, 0),
                b_in_orig: Vec::new(),
                n_eq_orig: me,
                n_in_orig: mi,
            },
        ));
    }

    let mut a_eq = DenseMatrix::zeros(kept_eq.len(), n);
    let mut b_eq = vec![zero; kept_eq.len()];
    for (new_r, &old_r) in kept_eq.iter().enumerate() {
        copy_row(&mut a_eq, new_r, &prob.a_eq, old_r);
        b_eq[new_r] = prob.b_eq[old_r];
    }
    let mut a_in = DenseMatrix::zeros(kept_in.len(), n);
    let mut b_in = vec![zero; kept_in.len()];
    for (new_r, &old_r) in kept_in.iter().enumerate() {
        copy_row(&mut a_in, new_r, &prob.a_in, old_r);
        b_in[new_r] = prob.b_in[old_r];
    }
    let a_in_csr = csr_from_rows(prob, sp, &kept_in, kept_in.len(), n);

    let reduced = QpProblem {
        p: prob.p.clone(),
        q: prob.q.clone(),
        a_eq,
        b_eq,
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr,
    };
    let reduction = RowReduction {
        kept_eq,
        kept_in,
        dropped_null,
        dropped_resid,
        a_in_orig: prob.a_in.clone(),
        b_in_orig: prob.b_in.clone(),
        n_eq_orig: me,
        n_in_orig: mi,
    };
    Ok((reduced, reduction))
}

/// Record of inequality pairs `aᵀx ≤ b` and `−aᵀx ≤ −b` with equal normalized
/// bounds, folded into a single equality row — net −2 rows. The fold is an exact
/// reformulation (equality of feasible sets), so the primal optimum is unchanged;
/// the pair's free multiplier splits back on restore as
/// `z_upper = max(y, 0)`, `z_lower = max(−y, 0)` (the two rows bind together at
/// any optimum, so complementarity holds for any split). Members of the group
/// looser than the two representatives are strictly dominated and drop with
/// dual 0. (Published two-row family: Achterberg, Bixby, Guénoche & Savelsbergh,
/// INFORMS JoC 32(2), 2020, §5.2; the Achterberg 2007 thesis, Alg. 10.5.)
#[derive(Clone, Debug)]
pub struct NegatedPairReduction<T: Scalar> {
    /// Inequality rows kept verbatim (in order).
    kept_in: Vec<usize>,
    /// Folded pairs `(upper_row, lower_row)`: pair k becomes equality row
    /// `n_eq_orig + k` of the reduced problem.
    pairs: Vec<(usize, usize)>,
    /// Group members looser than both representatives: strictly dominated by the
    /// pair, dropped with dual 0 and slack `b − aᵀx`.
    dropped_rest: Vec<usize>,
    a_in_orig: DenseMatrix<T>,
    b_in_orig: Vec<T>,
    n_in_orig: usize,
    n_eq_orig: usize,
}

impl<T: Scalar> NegatedPairReduction<T> {
    /// True when the pass changed the problem (the chain rebuilds its sparse
    /// views only then — the no-op case keeps the input's views valid).
    pub(crate) fn changed(&self) -> bool {
        !self.pairs.is_empty() || !self.dropped_rest.is_empty()
    }
}

impl<T: Scalar> NegatedPairReduction<T> {
    /// A no-op record (nothing folded): the pass ran and found no tight pairs.
    pub(crate) fn no_op(prob: &QpProblem<T>) -> Self {
        NegatedPairReduction {
            kept_in: (0..prob.b_in.len()).collect(),
            pairs: vec![],
            dropped_rest: vec![],
            a_in_orig: prob.a_in.clone(),
            b_in_orig: prob.b_in.clone(),
            n_in_orig: prob.b_in.len(),
            n_eq_orig: prob.b_eq.len(),
        }
    }
}

/// Fold negated-scaled duplicate inequality pairs into single equality rows.
///
/// Rows are canonicalized by their signed largest-magnitude entry (as in
/// [`reduce_rows`]); a group containing both directions is a ranged row
/// `b̂_l ≤ ûᵀx ≤ b̂_u`. When the two bounds coincide up to the RHS tolerance the
/// range collapses to the equality `ûᵀx = b̂` — the pair is replaced by one
/// equality row (net −2 rows). A contradictory range (empty interior) is
/// `PrimalInfeasible`; a non-tight range is left for [`reduce_rows`] to keep as
/// a two-sided pair. Runs before the equality chain so the created equalities
/// feed fixed-variable / doubleton / free-variable / merge elimination.
pub fn fold_negated_pairs<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
) -> Result<(QpProblem<T>, NegatedPairReduction<T>), Status> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let one = T::one();
    let trhs = tol_rhs::<T>();
    let ptol = tol_parallel::<T>();

    // Group candidate rows by canonical direction (identical to reduce_rows).
    // Row scans go through the sparse views when built (the transport LPs'
    // 99.94%-sparse 4940×4800 A_in costs ~0.9s of dense scans here alone);
    // the dense fallback is bit-identical (the null test is "no nonzeros" —
    // a row with any entry has max|v| = scale > 1e-10·scale, so the relative
    // `row_is_null` only fires on the all-zero row, same as `nz.is_empty()`).
    // Groups indexed by pivot position: the linear scan over every group per
    // row is O(mi) pivot compares (24M on the 4940-row transport LP, where
    // every row has its own pivot and no group ever matches); the map turns
    // it into O(1) lookups over only the same-pivot candidates. The group
    // direction vectors are materialized lazily (see [`PivotGroup`]) so the
    // all-unique-pivot shape stores no uhat at all.
    let mut groups: Vec<PivotGroup<T>> = Vec::new();
    let mut group_by_pivot: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for r in 0..mi {
        let mut cols: Vec<(usize, T)> = Vec::new();
        match sp.row(r) {
            Some(nz) => cols.extend_from_slice(nz),
            None => {
                for j in 0..n {
                    let a = prob.a_in.get(r, j);
                    if a != zero {
                        cols.push((j, a));
                    }
                }
            }
        }
        // Null test replicating `row_is_null` (scale = max(1, row max), so a
        // row is null iff every entry is <= 1e-10 — including a row whose
        // entries are nonzero-but-tiny; `is_empty` alone would treat it as a
        // candidate and could fold it into a bogus equality).
        let mx = cols.iter().fold(zero, |m, &(_, v)| m.max(v.abs()));
        if mx <= tol_coef::<T>() {
            continue; // null rows are handled by reduce_rows later
        }
        // The direction vector is only needed when a same-pivot group exists to
        // compare against — on the all-unique-pivot shapes it is never built.
        let (p, dir, bhat) = canon_row_dir(&cols, prob.b_in[r], zero, one);
        let mut gi = None;
        if let Some(ids) = group_by_pivot.get(&p) {
            let uhat = row_uhat(prob, sp, r, n);
            for &k in ids {
                if vec_close(groups[k].ensure_uhat(prob, sp, n), &uhat, ptol) {
                    gi = Some(k);
                    break;
                }
            }
        }
        match gi {
            Some(k) => groups[k].members.push((r, dir, bhat)),
            None => {
                groups.push(PivotGroup {
                    uhat: None,
                    src_row: r,
                    members: vec![(r, dir, bhat)],
                });
                group_by_pivot.entry(p).or_default().push(groups.len() - 1);
            }
        }
    }

    let mut pairs = Vec::new();
    let mut dropped_rest = Vec::new();
    for g in &groups {
        let mut upper: Option<(T, usize)> = None; // tightest (min b̂) of the +dir side
        let mut lower: Option<(T, usize)> = None; // tightest (max b̂) of the −dir side
        for &(r, dir, bhat) in &g.members {
            if dir > zero {
                if upper.is_none_or(|(ub, _)| bhat < ub) {
                    upper = Some((bhat, r));
                }
            } else if lower.is_none_or(|(lb, _)| bhat > lb) {
                lower = Some((bhat, r));
            }
        }
        if let (Some((ub, ru)), Some((lb, rl))) = (upper, lower) {
            let scale = one.max(ub.abs()).max(lb.abs());
            if lb - ub > trhs * scale {
                return Err(Status::PrimalInfeasible);
            }
            if (ub - lb).abs() <= trhs * scale {
                // Tight pair → one equality row (the upper representative's data).
                pairs.push((ru, rl));
                for &(r, _, _) in &g.members {
                    if r != ru && r != rl {
                        dropped_rest.push(r);
                    }
                }
                continue;
            }
        }
    }
    // Keep every row that was not folded or dropped — including null rows and
    // rows that never entered a group (reduce_rows handles those later).
    let kept_in: Vec<usize> = (0..mi)
        .filter(|&r| !dropped_rest.contains(&r) && !pairs.iter().any(|&(u, l)| r == u || r == l))
        .collect();

    // Nothing folded: return the input unchanged (avoids rebuilding the problem
    // when no tight pairs exist — the common case).
    if pairs.is_empty() && dropped_rest.is_empty() {
        return Ok((prob.clone(), NegatedPairReduction::no_op(prob)));
    }

    let mut a_eq = DenseMatrix::zeros(me + pairs.len(), n);
    let mut b_eq = vec![zero; me + pairs.len()];
    for r in 0..me {
        copy_row(&mut a_eq, r, &prob.a_eq, r);
        b_eq[r] = prob.b_eq[r];
    }
    for (k, &(ru, _)) in pairs.iter().enumerate() {
        copy_row(&mut a_eq, me + k, &prob.a_in, ru);
        b_eq[me + k] = prob.b_in[ru];
    }
    let mut a_in = DenseMatrix::zeros(kept_in.len(), n);
    let mut b_in = vec![zero; kept_in.len()];
    for (nr, &or) in kept_in.iter().enumerate() {
        copy_row(&mut a_in, nr, &prob.a_in, or);
        b_in[nr] = prob.b_in[or];
    }
    let reduction = NegatedPairReduction {
        kept_in,
        pairs,
        dropped_rest,
        a_in_orig: prob.a_in.clone(),
        b_in_orig: prob.b_in.clone(),
        n_in_orig: mi,
        n_eq_orig: me,
    };
    Ok((
        QpProblem {
            p: prob.p.clone(),
            q: prob.q.clone(),
            a_eq,
            b_eq,
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: csr_from_rows(prob, sp, &reduction.kept_in, reduction.kept_in.len(), n),
        },
        reduction,
    ))
}

/// Reconstruct an original-dimension solution: folded pair multipliers split
/// back into the two row duals; kept rows pass through; dominated siblings get
/// dual 0 and slack `b − aᵀx`.
pub fn restore_negated_pairs<T: Scalar>(
    red: &NegatedPairReduction<T>,
    reduced: &QpSolution<T>,
) -> QpSolution<T> {
    // No-op shortcut: nothing was folded, so the solution passes through
    // unchanged. Skips the full a_in_orig matvec, which the all-no-op chain
    // (e.g. the transport LPs) would otherwise pay per restore (~47M dense
    // ops each; the pair slacks below are the only consumer).
    if red.pairs.is_empty() && red.dropped_rest.is_empty() {
        return reduced.cloned();
    }
    let zero = T::zero();
    let mut s = vec![zero; red.n_in_orig];
    let mut z = vec![zero; red.n_in_orig];
    let ax = red.a_in_orig.matvec(&reduced.x);
    for (nr, &or) in red.kept_in.iter().enumerate() {
        s[or] = reduced.s[nr];
        z[or] = reduced.z[nr];
    }
    for (k, &(ru, rl)) in red.pairs.iter().enumerate() {
        let y = reduced.y[red.n_eq_orig + k];
        z[ru] = y.max(zero);
        z[rl] = (-y).max(zero);
        s[ru] = (red.b_in_orig[ru] - ax[ru]).max(zero);
        s[rl] = (red.b_in_orig[rl] - ax[rl]).max(zero);
    }
    for &or in &red.dropped_rest {
        s[or] = (red.b_in_orig[or] - ax[or]).max(zero);
        z[or] = zero;
    }
    // The folded rows were appended after the original equality rows; the head
    // of the reduced multipliers is exactly the original equalities' duals.
    reduced.with_duals(
        reduced.y[..red.n_eq_orig].to_vec(),
        s,
        z,
    )
}

/// Reconstruct an original-dimension solution from a solve of the reduced problem.
pub fn restore_rows<T: Scalar>(
    reduction: &RowReduction<T>,
    reduced: &QpSolution<T>,
) -> QpSolution<T> {
    let zero = T::zero();

    let mut y = vec![zero; reduction.n_eq_orig];
    for (new_r, &old_r) in reduction.kept_eq.iter().enumerate() {
        y[old_r] = reduced.y[new_r];
    }

    let mut s = vec![zero; reduction.n_in_orig];
    let mut z = vec![zero; reduction.n_in_orig];
    for (new_r, &old_r) in reduction.kept_in.iter().enumerate() {
        s[old_r] = reduced.s[new_r];
        z[old_r] = reduced.z[new_r];
    }
    for &(old_r, b) in &reduction.dropped_null {
        s[old_r] = b.max(zero);
        z[old_r] = zero;
    }
    if !reduction.dropped_resid.is_empty() {
        let ax = reduction.a_in_orig.matvec(&reduced.x);
        for &old_r in &reduction.dropped_resid {
            s[old_r] = (reduction.b_in_orig[old_r] - ax[old_r]).max(zero);
            z[old_r] = zero;
        }
    }

    reduced.with_duals(
        y,
        s,
        z,
    )
}

/// Record of inequality rows dropped as redundant (implied by the variable bounds).
#[derive(Clone, Debug)]
pub struct RedundancyReduction<T: Scalar> {
    kept_in: Vec<usize>,
    /// Redundant rows dropped; slack reconstructs as `b − aᵀx`, dual is 0.
    dropped: Vec<usize>,
    a_in_orig: DenseMatrix<T>,
    b_in_orig: Vec<T>,
    n_in_orig: usize,
}

/// Structural nonzeros of inequality row `r` (relative threshold, like [`row_is_null`]).
/// Pairs variant of [`ineq_row_nonzeros`] (the value is needed by the
/// redundant-row evaluator).
fn ineq_row_nonzeros_pairs<T: Scalar>(a: &DenseMatrix<T>, r: usize, n: usize) -> Vec<(usize, T)> {
    let mut scale = T::one();
    for j in 0..n {
        scale = scale.max(a.get(r, j).abs());
    }
    let thr = tol_coef::<T>() * scale;
    (0..n)
        .filter(|&j| a.get(r, j).abs() > thr)
        .map(|j| (j, a.get(r, j)))
        .collect()
}

/// Drop inequality rows that are **implied by the variable bounds** — i.e. unnecessary
/// constraints — and detect bound-activity infeasibility. Variable bounds `[lbⱼ, ubⱼ]` are
/// read off the singleton inequality rows (`a·xⱼ ≤ b` gives an upper bound if `a > 0`, a lower
/// bound if `a < 0`). For each multi-variable row `aᵀx ≤ b` the activity range over the box is
/// `[inf, sup]`. If the minimum activity `inf > b` the constraint cannot be satisfied →
/// `PrimalInfeasible`. If the maximum activity `sup ≤ b` the constraint can never be violated
/// → removed (postsolve reinserts it with dual 0 and slack `b − aᵀx ≥ 0`). Singleton
/// (bound-defining) rows are always kept. `O(m_in · n)`.
pub fn remove_redundant_ineqs<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
) -> Result<(QpProblem<T>, RedundancyReduction<T>), Status> {
    remove_redundant_ineqs_with_bounds(prob, sp, None)
}

/// Like [`remove_redundant_ineqs`] but seeded with pre-tightened variable bounds
/// (e.g. from FBBT). External bounds replace the singleton-row extraction for the
/// initial lb/ub; singleton rows in the problem still override individual entries.
pub fn remove_redundant_ineqs_with_bounds<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
    ext_bounds: Option<(&[T], &[T])>,
) -> Result<(QpProblem<T>, RedundancyReduction<T>), Status> {
    let n = prob.q.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let one = T::one();
    let inf = T::infinity();

    // Variable bounds: seed from FBBT-tightened bounds if provided, refine with
    // singleton rows from the problem (the tightest bound always wins).
    let mut lb = if let Some((elb, _)) = ext_bounds {
        elb.to_vec()
    } else {
        vec![-inf; n]
    };
    let mut ub = if let Some((_, eub)) = ext_bounds {
        eub.to_vec()
    } else {
        vec![inf; n]
    };
    for r in 0..mi {
        // The nonzero test is RELATIVE (entries above 1e-10 of the row's max);
        // the sparse rows replicate it from their own max.
        let nz = match sp.row(r) {
            Some(row_nz) => {
                let mx = row_nz.iter().fold(zero, |m, &(_, v)| m.max(v.abs()));
                let thr = tol_coef::<T>() * mx.max(one);
                row_nz
                    .iter()
                    .filter(|&&(_, v)| v.abs() > thr)
                    .copied()
                    .collect::<Vec<_>>()
            }
            None => ineq_row_nonzeros_pairs(&prob.a_in, r, n),
        };
        if nz.len() == 1 {
            let (j, a) = nz[0];
            let bnd = prob.b_in[r] / a;
            if a > zero {
                ub[j] = ub[j].min(bnd);
            } else {
                lb[j] = lb[j].max(bnd);
            }
        }
    }

    // Evaluate each multi-variable row's activity range over the (derivation-tightened)
    // box: drop it if the maximum activity stays within the rhs (redundant), or flag
    // infeasibility if the minimum exceeds it. Each contributing bound is the
    // maximizing/minimizing one for that variable's term.
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for r in 0..mi {
        let nz = match sp.row(r) {
            Some(row_nz) => {
                let mx = row_nz.iter().fold(zero, |m, &(_, v)| m.max(v.abs()));
                let thr = tol_coef::<T>() * mx.max(one);
                row_nz
                    .iter()
                    .filter(|&&(_, v)| v.abs() > thr)
                    .copied()
                    .collect::<Vec<_>>()
            }
            None => ineq_row_nonzeros_pairs(&prob.a_in, r, n),
        };
        if nz.len() <= 1 {
            kept.push(r);
            continue;
        }
        let (mut sup, mut inf_act) = (zero, zero);
        let (mut sup_finite, mut inf_finite) = (true, true);
        for &(j, a) in &nz {
            let (sup_bnd, inf_bnd) = if a > zero {
                (ub[j], lb[j])
            } else {
                (lb[j], ub[j])
            };
            if sup_bnd.is_finite() {
                sup += a * sup_bnd;
            } else {
                sup_finite = false;
            }
            if inf_bnd.is_finite() {
                inf_act += a * inf_bnd;
            } else {
                inf_finite = false;
            }
        }
        let scale = T::one().max(prob.b_in[r].abs());
        let tol = tol_rhs::<T>() * scale;
        if inf_finite && inf_act > prob.b_in[r] + tol {
            return Err(Status::PrimalInfeasible);
        }
        if sup_finite && sup <= prob.b_in[r] + tol {
            dropped.push(r);
        } else {
            kept.push(r);
        }
    }

    if dropped.is_empty() {
        // No-op: return the input unchanged; the record's A/B originals are never
        // read (the restore matvecs only when dropped is non-empty). The sparse
        // path also keeps the CSR alive through the no-op chain: the input's own
        // CSR (when present) is preserved by the clone, and when the problem came
        // from a dense source (a_in_csr: None) the chain's last pass emits one
        // from the views in O(nnz) — the downstream sparse-view build and the
        // solver then never pay the O(mi·n) dense extraction/csr_of_dense.
        let mut noop = prob.clone();
        if sp.is_sparse() && noop.a_in_csr.is_none() {
            noop.a_in_csr = csr_from_rows(prob, sp, &kept, kept.len(), n);
        }
        return Ok((
            noop,
            RedundancyReduction {
                kept_in: kept,
                dropped: Vec::new(),
                a_in_orig: DenseMatrix::zeros(0, 0),
                b_in_orig: Vec::new(),
                n_in_orig: mi,
            },
        ));
    }
    let reduction = RedundancyReduction {
        kept_in: kept.clone(),
        dropped: dropped.clone(),
        a_in_orig: prob.a_in.clone(),
        b_in_orig: prob.b_in.clone(),
        n_in_orig: mi,
    };

    let mut a_in = DenseMatrix::zeros(kept.len(), n);
    let mut b_in = vec![zero; kept.len()];
    for (nr, &or) in kept.iter().enumerate() {
        copy_row(&mut a_in, nr, &prob.a_in, or);
        b_in[nr] = prob.b_in[or];
    }
    Ok((
        QpProblem {
            p: prob.p.clone(),
            q: prob.q.clone(),
            a_eq: prob.a_eq.clone(),
            b_eq: prob.b_eq.clone(),
            a_in,
            b_in,
            a_eq_csr: None,
            // Same CSR-emission discipline as the sibling passes: rebuild the
            // reduced problem's CSR from the views in the same pass (the
            // dense fallback keeps None exactly as before).
            a_in_csr: csr_from_rows(prob, sp, &kept, kept.len(), n),
        },
        reduction,
    ))
}

/// Reinsert redundant inequality rows: each gets dual 0 and slack `b − aᵀx`.
pub fn restore_redundant_ineqs<T: Scalar>(
    red: &RedundancyReduction<T>,
    reduced: &QpSolution<T>,
) -> QpSolution<T> {
    let zero = T::zero();
    let mut s = vec![zero; red.n_in_orig];
    let mut z = vec![zero; red.n_in_orig];
    for (nr, &or) in red.kept_in.iter().enumerate() {
        s[or] = reduced.s[nr];
        z[or] = reduced.z[nr];
    }
    if !red.dropped.is_empty() {
        let ax = red.a_in_orig.matvec(&reduced.x);
        for &or in &red.dropped {
            s[or] = (red.b_in_orig[or] - ax[or]).max(zero);
            z[or] = zero;
        }
    }
    reduced.with_duals(
        reduced.y.clone(),
        s,
        z,
    )
}

/// Record of the independent equality rows kept after removing linearly dependent ones.
#[derive(Clone, Debug)]
pub struct EqDepReduction {
    /// Indices (into the input problem's equality rows) of the kept independent rows.
    kept_eq: Vec<usize>,
    /// Input equality-row count, so the dual can be re-expanded.
    n_eq_in: usize,
}

impl EqDepReduction {
    /// The no-op record for a problem with no equality rows (identical to what
    /// [`remove_dependent_eq_rows`]'s own empty-eq branch builds).
    pub(crate) fn no_op() -> Self {
        Self {
            kept_eq: Vec::new(),
            n_eq_in: 0,
        }
    }
}

/// Remove **linearly dependent equality rows** — the classic presolve reduction for
/// degenerate problems (redundant or rank-deficient equality blocks). Equality rows are
/// processed in order through a running row-echelon basis: a row that reduces to zero
/// against the basis is dependent — redundant if its reduced right-hand side is also zero
/// (dropped), else the system is inconsistent (`PrimalInfeasible`). Independent rows are
/// kept verbatim (only the dependence test uses the reduced form), so no fill is
/// introduced. Removing them de-degenerates the KKT, which the proximal IPM then solves
/// in fewer, better-conditioned iterations.
///
/// `O(m_eq · rank · n)`; gated to run only when there are at least two equality rows.
pub fn remove_dependent_eq_rows<T: Scalar>(
    prob: &QpProblem<T>,
) -> Result<(QpProblem<T>, EqDepReduction), Status> {
    let me = prob.b_eq.len();
    let n = prob.q.len();
    let zero = T::zero();
    let tol = tol_parallel::<T>();

    // (reduced row in echelon form with a unit pivot, reduced rhs, pivot column).
    let mut basis: Vec<(Vec<T>, T, usize)> = Vec::new();
    let mut kept_eq: Vec<usize> = Vec::new();

    for r in 0..me {
        let mut row: Vec<T> = (0..n).map(|j| prob.a_eq.get(r, j)).collect();
        let mut b = prob.b_eq[r];
        // Scale for the relative dependence test: the row's own magnitude.
        let scale = row
            .iter()
            .fold(b.abs(), |m, &v| m.max(v.abs()))
            .max(T::one());

        // Eliminate against the existing echelon basis.
        for (prow, pb, pcol) in &basis {
            let f = row[*pcol];
            if f != zero {
                for j in 0..n {
                    row[j] -= f * prow[j];
                }
                b -= f * *pb;
            }
        }

        // Largest remaining entry → candidate pivot.
        let (mut p, mut mx) = (0usize, zero);
        for j in 0..n {
            let v = row[j].abs();
            if v > mx {
                mx = v;
                p = j;
            }
        }

        if mx > tol * scale {
            // Independent: normalize to a unit pivot and extend the basis.
            let c = row[p];
            for j in 0..n {
                row[j] /= c;
            }
            b /= c;
            basis.push((row, b, p));
            kept_eq.push(r);
        } else if b.abs() > tol_rhs::<T>() * scale {
            // Dependent rows of A but an inconsistent rhs ⇒ no feasible point.
            return Err(Status::PrimalInfeasible);
        }
        // Otherwise the row is redundant (0 = 0) — drop it.
    }

    if kept_eq.len() == me {
        // Full rank: nothing to do (avoid a needless clone-rebuild).
        return Ok((
            prob.clone(),
            EqDepReduction {
                kept_eq,
                n_eq_in: me,
            },
        ));
    }

    let mut a_eq = DenseMatrix::zeros(kept_eq.len(), n);
    let mut b_eq = vec![zero; kept_eq.len()];
    for (new_r, &old_r) in kept_eq.iter().enumerate() {
        copy_row(&mut a_eq, new_r, &prob.a_eq, old_r);
        b_eq[new_r] = prob.b_eq[old_r];
    }
    let reduced = QpProblem {
        p: prob.p.clone(),
        q: prob.q.clone(),
        a_eq,
        b_eq,
        a_in: prob.a_in.clone(),
        b_in: prob.b_in.clone(),
        a_eq_csr: None,
        a_in_csr: None,
    };
    Ok((
        reduced,
        EqDepReduction {
            kept_eq,
            n_eq_in: me,
        },
    ))
}

/// Re-expand the equality multipliers to the input dimension: each dropped (redundant)
/// row gets dual `0`, a valid choice since it is a linear combination of the kept rows.
pub fn restore_dependent_eq_rows<T: Scalar>(
    red: &EqDepReduction,
    reduced: &QpSolution<T>,
) -> QpSolution<T> {
    if red.kept_eq.len() == red.n_eq_in {
        return reduced.clone();
    }
    let mut y = vec![T::zero(); red.n_eq_in];
    for (i, &old_r) in red.kept_eq.iter().enumerate() {
        y[old_r] = reduced.y[i];
    }
    reduced.with_duals(
        y,
        reduced.s.clone(),
        reduced.z.clone(),
    )
}

/// Record of variables eliminated as empty (unconstrained) columns.
#[derive(Clone, Debug)]
pub struct ColReduction<T: Scalar> {
    kept: Vec<usize>,
    fixed: Vec<(usize, T)>,
    n_orig: usize,
}

impl<T: Scalar> ColReduction<T> {
    /// True when at least one empty column was eliminated.
    pub(crate) fn changed(&self) -> bool {
        !self.fixed.is_empty()
    }
}

/// A column `j` is empty if the variable appears in no constraint and couples to no
/// other variable through `P` (its own diagonal aside). `p_diag` is a precomputed
/// diagonal-P detection so the per-column off-diagonal `P` probe (an O(n²) scan,
/// 23M reads on the transport LP) is skipped entirely for the common diagonal-P
/// shape — only `P_jj` can couple the column then.
fn col_is_empty<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
    j: usize,
    me: usize,
    n: usize,
    p_diag: bool,
) -> bool {
    let mut colmax = T::zero();
    for r in 0..me {
        colmax = colmax.max(prob.a_eq.get(r, j).abs());
    }
    match sp.col(j) {
        Some(col_nz) => {
            for &(_, v) in col_nz {
                colmax = colmax.max(v.abs());
            }
        }
        None => {
            for r in 0..prob.b_in.len() {
                colmax = colmax.max(prob.a_in.get(r, j).abs());
            }
        }
    }
    if !p_diag {
        for i in 0..n {
            if i != j {
                colmax = colmax.max(prob.p.get(i, j).abs());
            }
        }
    }
    // colmax is the largest entry; thr = tol·max(colmax,1), so colmax ≤ thr ⇔ all ≤ thr.
    colmax <= tol_coef::<T>() * colmax.max(T::one())
}

/// Eliminate empty (unconstrained) columns, fixing each variable at its closed-form
/// minimizer `x_j = −q_j/P_jj` (or `0` if flat). An unbounded empty column
/// (`P_jj ≈ 0`, `q_j ≠ 0`) yields `Err(Status::DualInfeasible)`.
pub fn eliminate_empty_cols<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
) -> Result<(QpProblem<T>, ColReduction<T>), Status> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let pd_tol = T::from_f64(1e-12).expect("scalar literal");
    let qn = prob.q.iter().fold(zero, |m, &v| m.max(v.abs()));

    // Diagonal-P detection (early-exits on the first off-diagonal nonzero, same
    // as the Ruiz path): with a diagonal P only the (j,j) entry can couple the
    // column, so the per-column off-diagonal probe becomes a no-op.
    // P is symmetric (the QP convention), so a single triangle suffices —
    // halving the diagonal-P detection's 184MB read on the transport (the
    // same half-triangle scan the Ruiz path uses).
    let p_diag = (0..n).all(|i| (i + 1..n).all(|j| prob.p.get(i, j) == zero));
    let mut kept = Vec::new();
    let mut fixed = Vec::new();
    for j in 0..n {
        if col_is_empty(prob, sp, j, me, n, p_diag) {
            let pjj = prob.p.get(j, j);
            let qj = prob.q[j];
            let value = if pjj > pd_tol {
                -qj / pjj
            } else if qj.abs() <= tol_coef::<T>() * qn.max(T::one()) {
                zero
            } else {
                return Err(Status::DualInfeasible);
            };
            fixed.push((j, value));
        } else {
            kept.push(j);
        }
    }
    if fixed.is_empty() {
        return Ok((
            prob.clone(),
            ColReduction {
                kept,
                fixed,
                n_orig: n,
            },
        ));
    }

    let nk = kept.len();
    let mut p = DenseMatrix::zeros(nk, nk);
    for (a, &ja) in kept.iter().enumerate() {
        for (b, &jb) in kept.iter().enumerate() {
            p.set(a, b, prob.p.get(ja, jb));
        }
    }
    let q: Vec<T> = kept.iter().map(|&j| prob.q[j]).collect();
    let mut a_eq = DenseMatrix::zeros(me, nk);
    for r in 0..me {
        for (a, &j) in kept.iter().enumerate() {
            a_eq.set(r, a, prob.a_eq.get(r, j));
        }
    }
    let mut a_in = DenseMatrix::zeros(mi, nk);
    for r in 0..mi {
        for (a, &j) in kept.iter().enumerate() {
            a_in.set(r, a, prob.a_in.get(r, j));
        }
    }
    // CSR with the same column remap (rows are untouched — every row survives).
    let mut new_col = vec![usize::MAX; n];
    for (a, &j) in kept.iter().enumerate() {
        new_col[j] = a;
    }
    let all_rows: Vec<usize> = (0..mi).collect();
    let a_in_csr = csr_from_rows_cols(prob, sp, &all_rows, &new_col, mi, nk);
    let reduced = QpProblem {
        p,
        q,
        a_eq,
        b_eq: prob.b_eq.clone(),
        a_in,
        b_in: prob.b_in.clone(),
        a_eq_csr: None,
        a_in_csr,
    };
    Ok((
        reduced,
        ColReduction {
            kept,
            fixed,
            n_orig: n,
        },
    ))
}

/// Reinsert eliminated columns into the primal vector (duals/slacks are unchanged,
/// since empty columns touch no constraint).
pub fn restore_cols<T: Scalar>(colred: &ColReduction<T>, reduced: &QpSolution<T>) -> QpSolution<T> {
    let mut x = vec![T::zero(); colred.n_orig];
    for (a, &j) in colred.kept.iter().enumerate() {
        x[j] = reduced.x[a];
    }
    for &(j, v) in &colred.fixed {
        x[j] = v;
    }
    QpSolution::new(
        reduced.status,
        x,
        reduced.y.clone(),
        reduced.s.clone(),
        reduced.z.clone(),
        reduced.obj_val,
        reduced.iters,
    )
}

/// Record of variables fixed by equality singletons (`a·x_j = b → x_j = b/a`).
#[derive(Clone, Debug)]
pub struct FixedVarReduction<T: Scalar> {
    kept_cols: Vec<usize>,
    fixed: Vec<(usize, T)>,
    kept_eq: Vec<usize>,
    /// `(pin_eq_row, col, a)` for each fixer used as the representative.
    pins: Vec<(usize, usize, T)>,
    n_orig: usize,
    n_eq_orig: usize,
}

impl<T: Scalar> FixedVarReduction<T> {
    /// The no-op record for a problem with no equality rows to fix from (identical
    /// to what [`eliminate_fixed_vars`]'s own `fixed.is_empty()` branch builds).
    pub(crate) fn no_op(n: usize) -> Self {
        Self {
            kept_cols: (0..n).collect(),
            fixed: Vec::new(),
            kept_eq: Vec::new(),
            pins: Vec::new(),
            n_orig: n,
            n_eq_orig: 0,
        }
    }
}

/// Eliminate variables pinned by an equality singleton. Substitutes `x_j = b/a` out
/// of `P`, `q`, `A_eq`, `A_in` (folding the quadratic coupling into `q` and the linear
/// coupling into the right-hand sides) and drops the pinning rows. Inconsistent
/// fixers (`x_j = v₀` and `x_j = v₁ ≠ v₀`) yield `Err(Status::PrimalInfeasible)`.
pub fn eliminate_fixed_vars<T: Scalar>(
    prob: &QpProblem<T>,
) -> Result<(QpProblem<T>, FixedVarReduction<T>), Status> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let one = T::one();
    let tcoef = tol_coef::<T>();
    let trhs = tol_rhs::<T>();

    let mut fixed_val: Vec<Option<(T, usize, T)>> = vec![None; n];
    let mut kept_eq = Vec::new();
    for r in 0..me {
        let mut scale = one;
        for j in 0..n {
            scale = scale.max(prob.a_eq.get(r, j).abs());
        }
        let thr = tcoef * scale;
        let nz: Vec<usize> = (0..n)
            .filter(|&j| prob.a_eq.get(r, j).abs() > thr)
            .collect();
        if nz.len() == 1 {
            let j = nz[0];
            let a = prob.a_eq.get(r, j);
            let v = prob.b_eq[r] / a;
            match fixed_val[j] {
                None => fixed_val[j] = Some((v, r, a)),
                Some((v0, _, _)) => {
                    if (v - v0).abs() > trhs * (one + v0.abs()) {
                        return Err(Status::PrimalInfeasible);
                    }
                }
            }
        } else {
            kept_eq.push(r);
        }
    }

    let fixed: Vec<(usize, T)> = (0..n)
        .filter_map(|j| fixed_val[j].map(|(v, _, _)| (j, v)))
        .collect();
    if fixed.is_empty() {
        return Ok((
            prob.clone(),
            FixedVarReduction {
                kept_cols: (0..n).collect(),
                fixed,
                kept_eq,
                pins: vec![],
                n_orig: n,
                n_eq_orig: me,
            },
        ));
    }
    let kept_cols: Vec<usize> = (0..n).filter(|&j| fixed_val[j].is_none()).collect();
    let pins: Vec<(usize, usize, T)> = (0..n)
        .filter_map(|j| fixed_val[j].map(|(_, r, a)| (r, j, a)))
        .collect();

    let nk = kept_cols.len();
    let mut p = DenseMatrix::zeros(nk, nk);
    for (a, &ja) in kept_cols.iter().enumerate() {
        for (b, &jb) in kept_cols.iter().enumerate() {
            p.set(a, b, prob.p.get(ja, jb));
        }
    }
    let mut q = vec![zero; nk];
    for (k, &jk) in kept_cols.iter().enumerate() {
        let mut qq = prob.q[jk];
        for &(jf, vf) in &fixed {
            qq += prob.p.get(jk, jf) * vf;
        }
        q[k] = qq;
    }
    let mut a_eq = DenseMatrix::zeros(kept_eq.len(), nk);
    let mut b_eq = vec![zero; kept_eq.len()];
    for (nr, &or) in kept_eq.iter().enumerate() {
        for (k, &jk) in kept_cols.iter().enumerate() {
            a_eq.set(nr, k, prob.a_eq.get(or, jk));
        }
        let mut bb = prob.b_eq[or];
        for &(jf, vf) in &fixed {
            bb -= prob.a_eq.get(or, jf) * vf;
        }
        b_eq[nr] = bb;
    }
    let mut a_in = DenseMatrix::zeros(mi, nk);
    let mut b_in = vec![zero; mi];
    for r in 0..mi {
        for (k, &jk) in kept_cols.iter().enumerate() {
            a_in.set(r, k, prob.a_in.get(r, jk));
        }
        let mut bb = prob.b_in[r];
        for &(jf, vf) in &fixed {
            bb -= prob.a_in.get(r, jf) * vf;
        }
        b_in[r] = bb;
    }

    Ok((
        QpProblem {
            p,
            q,
            a_eq,
            b_eq,
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        },
        FixedVarReduction {
            kept_cols,
            fixed,
            kept_eq,
            pins,
            n_orig: n,
            n_eq_orig: me,
        },
    ))
}

/// Reinsert fixed variables and recover their pinning-equality multipliers from the
/// original stationarity at the fixed columns.
pub fn restore_fixed_vars<T: Scalar>(
    orig: &QpProblem<T>,
    red: &FixedVarReduction<T>,
    reduced: &QpSolution<T>,
) -> QpSolution<T> {
    // No-op shortcut (nothing fixed/pinned): pass through unchanged, skipping
    // the full P/A_eq/A_in matvecs for the pin multipliers and the objective
    // recomputation (~47M dense ops each on the big-sparse shapes).
    if red.fixed.is_empty() && red.pins.is_empty() {
        return reduced.cloned();
    }
    let zero = T::zero();
    let mut x = vec![zero; red.n_orig];
    for (k, &j) in red.kept_cols.iter().enumerate() {
        x[j] = reduced.x[k];
    }
    for &(j, v) in &red.fixed {
        x[j] = v;
    }
    let mut y = vec![zero; red.n_eq_orig];
    for (nr, &or) in red.kept_eq.iter().enumerate() {
        y[or] = reduced.y[nr];
    }

    // Pin multipliers from original stationarity: at fixed column j the only equality
    // contribution from the pin row is a·y_pin, so solve for it (other y already set).
    let px = orig.p.matvec(&x);
    let aty = orig.a_eq.matvec_t(&y); // pin rows still 0 here
    let atz = orig.a_in.matvec_t(&reduced.z);
    for &(pin_row, j, a) in &red.pins {
        y[pin_row] = -(px[j] + orig.q[j] + aty[j] + atz[j]) / a;
    }

    // The reduced problem's objective omits the fixed variables' contribution
    // (the P/q folding only carries the linear *coupling* into survivors); recompute
    // at original dimensions so obj_val is correct standalone, not just via a
    // caller's own final recompute.
    let half = T::from_f64(0.5).expect("scalar literal");
    let mut obj_val = T::zero();
    for i in 0..x.len() {
        obj_val += half * x[i] * px[i] + orig.q[i] * x[i];
    }

    QpSolution::new(
        reduced.status,
        x,
        y,
        reduced.s.clone(),
        reduced.z.clone(),
        obj_val,
        reduced.iters,
    )
}

/// Record of variables eliminated by an equality **doubleton** (`aᵢxᵢ + aⱼxⱼ = b`).
#[derive(Clone, Debug)]
pub struct DoubletonReduction<T: Scalar> {
    /// Columns kept (the eliminated variables removed), in order.
    kept_cols: Vec<usize>,
    /// Equality rows kept (the pivot rows removed), in order.
    kept_eq: Vec<usize>,
    /// One per elimination: `(pivot_row, elim_col i, surv_col j, beta, gamma, a_i)` with
    /// `xᵢ = beta + gamma·xⱼ`.
    elims: Vec<(usize, usize, usize, T, T, T)>,
    n_orig: usize,
    n_eq_orig: usize,
}

impl<T: Scalar> DoubletonReduction<T> {
    /// The no-op record for a problem with no equality rows to pair up (identical to
    /// what [`eliminate_doubleton_eqs`]'s own `elims.is_empty()` branch builds).
    pub(crate) fn no_op(n: usize) -> Self {
        Self {
            kept_cols: (0..n).collect(),
            kept_eq: Vec::new(),
            elims: Vec::new(),
            n_orig: n,
            n_eq_orig: 0,
        }
    }
}

/// Eliminate variables linked by an equality **doubleton** — a row with exactly two
/// nonzeros `aᵢxᵢ + aⱼxⱼ = b`. The larger-magnitude coefficient's variable `xᵢ` is
/// substituted out as `xᵢ = β + γxⱼ` (`β = b/aᵢ`, `γ = −aⱼ/aᵢ`), folding the quadratic
/// coupling into `P`'s `j`-row/column, the linear coupling into `q`, and the constraint
/// couplings into the surviving column / right-hand sides; the pivot row is dropped. Only a
/// **non-overlapping** set is taken in one pass (each eliminated/surviving variable and each
/// pivot row used at most once), which keeps every substitution — and the dual recovery —
/// independent. Re-run the pass to collapse chains.
pub fn eliminate_doubleton_eqs<T: Scalar>(
    prob: &QpProblem<T>,
    fill_budget: f64,
) -> Result<(QpProblem<T>, DoubletonReduction<T>), Status> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let two = T::from_f64(2.0).expect("scalar literal");
    let tcoef = tol_coef::<T>();

    // Select non-overlapping doubleton rows.
    let mut used_var = vec![false; n];
    let mut elim_row = vec![false; me];
    let mut elims: Vec<(usize, usize, usize, T, T, T)> = Vec::new();
    for r in 0..me {
        let mut scale = T::one();
        for k in 0..n {
            scale = scale.max(prob.a_eq.get(r, k).abs());
        }
        let thr = tcoef * scale;
        let nz: Vec<usize> = (0..n)
            .filter(|&k| prob.a_eq.get(r, k).abs() > thr)
            .collect();
        if nz.len() != 2 {
            continue;
        }
        let (p, q2) = (nz[0], nz[1]);
        let (i, j) = if prob.a_eq.get(r, p).abs() >= prob.a_eq.get(r, q2).abs() {
            (p, q2)
        } else {
            (q2, p)
        };
        if used_var[i] || used_var[j] {
            continue;
        }
        // Fill budget: rows containing i but not j gain one entry each,
        // and the Hessian fold densifies P's row-i columns. Skip the
        // elimination when the projected fill would push the total nonzeros
        // past `fill_budget` × the current total (fill_budget <= 0.0 disables the
        // gate entirely, preserving the unconditional behaviour).
        if fill_budget > 0.0 {
            let mut total_nnz = 0usize;
            for r2 in 0..me {
                for k in 0..n {
                    if prob.a_eq.get(r2, k) != zero {
                        total_nnz += 1;
                    }
                }
            }
            for r2 in 0..mi {
                for k in 0..n {
                    if prob.a_in.get(r2, k) != zero {
                        total_nnz += 1;
                    }
                }
            }
            let mut fill = 0usize;
            for r2 in 0..mi {
                let mut has_i = false;
                let mut has_j = false;
                for k in 0..n {
                    let v = prob.a_in.get(r2, k);
                    if v == zero {
                        continue;
                    }
                    if k == i {
                        has_i = true;
                    } else if k == j {
                        has_j = true;
                    }
                }
                if has_i && !has_j {
                    fill += 1;
                }
            }
            for k in 0..n {
                if prob.p.get(i, k) != zero {
                    fill += 1;
                }
            }
            if total_nnz == 0 || (total_nnz + fill) as f64 > fill_budget * total_nnz as f64 {
                continue;
            }
        }
        used_var[i] = true;
        used_var[j] = true;
        elim_row[r] = true;
        let a_i = prob.a_eq.get(r, i);
        let a_j = prob.a_eq.get(r, j);
        let gamma = -a_j / a_i;
        let beta = prob.b_eq[r] / a_i;
        elims.push((r, i, j, beta, gamma, a_i));
    }

    if elims.is_empty() {
        return Ok((
            prob.clone(),
            DoubletonReduction {
                kept_cols: (0..n).collect(),
                kept_eq: (0..me).collect(),
                elims,
                n_orig: n,
                n_eq_orig: me,
            },
        ));
    }

    // Apply each substitution `xᵢ = β + γxⱼ` to a working copy. The eliminations are
    // independent (disjoint vars/rows), so applying them in sequence on the shared copy is
    // equivalent to applying them simultaneously.
    let mut p = prob.p.clone();
    let mut q = prob.q.clone();
    let mut a_eq = prob.a_eq.clone();
    let mut b_eq = prob.b_eq.clone();
    let mut a_in = prob.a_in.clone();
    let mut b_in = prob.b_in.clone();

    for &(_r, i, j, beta, gamma, _a_i) in &elims {
        // Quadratic: P' = TᵀPT, localized to the surviving column j.
        for k in 0..n {
            if k == i || k == j {
                continue;
            }
            let add = gamma * p.get(i, k);
            p.set(j, k, p.get(j, k) + add);
            p.set(k, j, p.get(k, j) + add);
            // Linear from the constant offset β: q_k += β·P_ik.
            q[k] += beta * p.get(i, k);
        }
        let pij = p.get(i, j);
        let pii = p.get(i, i);
        p.set(j, j, p.get(j, j) + two * gamma * pij + gamma * gamma * pii);
        let qj_add = beta * pij + gamma * (beta * pii + q[i]);
        q[j] += qj_add;

        // Constraints: fold xᵢ's column into xⱼ and the rhs, over the surviving rows.
        for r2 in 0..me {
            if elim_row[r2] {
                continue;
            }
            let aii = a_eq.get(r2, i);
            if aii != zero {
                a_eq.set(r2, j, a_eq.get(r2, j) + gamma * aii);
                b_eq[r2] -= beta * aii;
            }
        }
        for r2 in 0..mi {
            let aii = a_in.get(r2, i);
            if aii != zero {
                a_in.set(r2, j, a_in.get(r2, j) + gamma * aii);
                b_in[r2] -= beta * aii;
            }
        }
    }

    // Compact out the eliminated columns and pivot rows.
    let kept_cols: Vec<usize> = (0..n).filter(|&k| !used_var_is_elim(&elims, k)).collect();
    let kept_eq: Vec<usize> = (0..me).filter(|&r| !elim_row[r]).collect();
    let nk = kept_cols.len();

    let mut rp = DenseMatrix::zeros(nk, nk);
    for (a, &ca) in kept_cols.iter().enumerate() {
        for (b, &cb) in kept_cols.iter().enumerate() {
            rp.set(a, b, p.get(ca, cb));
        }
    }
    let rq: Vec<T> = kept_cols.iter().map(|&k| q[k]).collect();
    let mut raeq = DenseMatrix::zeros(kept_eq.len(), nk);
    let mut rbeq = vec![zero; kept_eq.len()];
    for (nr, &or) in kept_eq.iter().enumerate() {
        for (a, &c) in kept_cols.iter().enumerate() {
            raeq.set(nr, a, a_eq.get(or, c));
        }
        rbeq[nr] = b_eq[or];
    }
    let mut rain = DenseMatrix::zeros(mi, nk);
    for r in 0..mi {
        for (a, &c) in kept_cols.iter().enumerate() {
            rain.set(r, a, a_in.get(r, c));
        }
    }

    Ok((
        QpProblem {
            p: rp,
            q: rq,
            a_eq: raeq,
            b_eq: rbeq,
            a_in: rain,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        },
        DoubletonReduction {
            kept_cols,
            kept_eq,
            elims,
            n_orig: n,
            n_eq_orig: me,
        },
    ))
}

fn used_var_is_elim<T: Scalar>(elims: &[(usize, usize, usize, T, T, T)], col: usize) -> bool {
    elims.iter().any(|&(_, i, _, _, _, _)| i == col)
}

/// Reinsert doubleton-eliminated variables and recover the pivot-row multipliers.
pub fn restore_doubleton_eqs<T: Scalar>(
    orig: &QpProblem<T>,
    red: &DoubletonReduction<T>,
    reduced: &QpSolution<T>,
) -> QpSolution<T> {
    // No-op shortcut (no doubleton eliminated): pass through unchanged,
    // skipping the full P/A_eq/A_in matvecs and the objective recomputation.
    if red.elims.is_empty() {
        return reduced.cloned();
    }
    let zero = T::zero();
    let n = red.n_orig;

    let mut x = vec![zero; n];
    for (a, &c) in red.kept_cols.iter().enumerate() {
        x[c] = reduced.x[a];
    }
    // xᵢ = β + γ·xⱼ (the survivor xⱼ is a kept column, already filled).
    for &(_r, i, j, beta, gamma, _a_i) in &red.elims {
        x[i] = beta + gamma * x[j];
    }

    let mut y = vec![zero; red.n_eq_orig];
    for (nr, &or) in red.kept_eq.iter().enumerate() {
        y[or] = reduced.y[nr];
    }

    // Pivot-row multipliers from original stationarity at the eliminated column i: the only
    // unknown equality contribution there is aᵢ·y_pivot (column i appears in no other
    // eliminated row), so y_pivot = −[(Px+q)_i + (A_eqᵀy)_i + (A_inᵀz)_i]/aᵢ.
    let px = orig.p.matvec(&x);
    let aty = orig.a_eq.matvec_t(&y); // pivot rows still 0 here
    let atz = orig.a_in.matvec_t(&reduced.z);
    for &(r, i, _j, _beta, _gamma, a_i) in &red.elims {
        y[r] = -(px[i] + orig.q[i] + aty[i] + atz[i]) / a_i;
    }

    // The reduced problem's objective omits the eliminated variables' contribution
    // (the P/q folding only carries the coupling into the survivor); recompute at
    // original dimensions so obj_val is correct standalone.
    let half = T::from_f64(0.5).expect("scalar literal");
    let mut obj_val = T::zero();
    for k in 0..n {
        obj_val += half * x[k] * px[k] + orig.q[k] * x[k];
    }

    QpSolution::new(
        reduced.status,
        x,
        y,
        reduced.s.clone(),
        reduced.z.clone(),
        obj_val,
        reduced.iters,
    )
}

// ─────────────────────────────────────────────────────────────
//  Implied-free variable substitution
// ─────────────────────────────────────────────────────────────

/// Record of variables substituted via implied-free elimination.
#[derive(Clone, Debug)]
pub struct FreeVarReduction<T: Scalar> {
    pub kept_cols: Vec<usize>,
    pub kept_eq: Vec<usize>,
    /// `(pivot_row, elim_col, coeff, shift)` — the defining row.
    pub elims: Vec<(usize, usize, T, T)>,
    pub n_orig: usize,
    pub n_eq_orig: usize,
}

fn col_is_diag_p<T: Scalar>(p: &DenseMatrix<T>, j: usize, n: usize) -> bool {
    let zero = T::zero();
    for i in 0..n {
        if i != j && (p.get(i, j) != zero || p.get(j, i) != zero) {
            return false;
        }
    }
    true
}

fn col_in_ineq<T: Scalar>(a_in: &DenseMatrix<T>, j: usize, mi: usize) -> bool {
    let zero = T::zero();
    (0..mi).any(|r| a_in.get(r, j) != zero)
}

/// Substitute **implied-free variables**: for each variable `xⱼ` that has no
/// P-coupling (diagonal-only), no inequality involvement, and appears in exactly
/// one equality row, substitute `xⱼ` out of the objective via
/// `xⱼ = (b_r − Σ_{k≠j} a_rk x_k) / a_rj`.
///
/// Gated: only variables whose defining row has ≤ `max_fill` nonzeros.  A
/// variable appearing in more than one equality row is NOT eliminated: the
/// substitution is applied sequentially and a later elimination can modify
/// another candidate's defining row first, making its `shift` stale and the
/// reduced problem silently wrong.  This generalises [`eliminate_fixed_vars`]
/// (defining row has 1 nonzero).
pub fn eliminate_free_vars<T: Scalar>(
    prob: &QpProblem<T>,
    max_fill: usize,
    _max_rows: usize,
) -> Result<(QpProblem<T>, FreeVarReduction<T>), Status> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let tcoef = tol_coef::<T>();

    // No equality rows ⇒ every column's `row_count` is 0 ⇒ `elims` is always empty
    // (identical to the `elims.is_empty()` path below). Skip the O(n·(n+mi)) column
    // scan (`col_is_diag_p`/`col_in_ineq` per column) that can only ever find nothing.
    if me == 0 {
        return Ok((
            prob.clone(),
            FreeVarReduction {
                kept_cols: (0..n).collect(),
                kept_eq: Vec::new(),
                elims: Vec::new(),
                n_orig: n,
                n_eq_orig: me,
            },
        ));
    }

    let mut best: Vec<Option<(usize, T, T)>> = vec![None; n];
    for j in 0..n {
        if !col_is_diag_p(&prob.p, j, n) || col_in_ineq(&prob.a_in, j, mi) {
            continue;
        }
        let mut row_count = 0usize;
        let mut best_row = None;
        let mut best_abs = zero;
        for r in 0..me {
            let a = prob.a_eq.get(r, j);
            if a.abs() <= tcoef * a.abs().max(T::one()) {
                continue;
            }
            row_count += 1;
            if a.abs() > best_abs {
                best_abs = a.abs();
                best_row = Some((r, a));
            }
        }
        // Only variables in EXACTLY ONE equality row are sound to substitute.
        // The substitution `xⱼ = (b_r − Σ_{k≠j} a_rk x_k)/a_rj` is derived from the
        // defining row's coefficients (`shift` from the original `b_eq`), and
        // eliminations are applied sequentially: if another elimination touches
        // this row first, `shift` is stale and the reduced problem (and its
        // postsolve recovery) is silently wrong — the variable's occurrences in
        // other equality rows are inconsistent with the substitution. With
        // `row_count == 1` no other eliminated variable can share the defining
        // row, so every substitution is exact. (`max_rows` is kept for the
        // signature; the only sound case is a single row.)
        if row_count != 1 {
            continue;
        }
        let (r, coeff) = match best_row {
            Some(v) => v,
            None => continue,
        };
        let mut nnz = 0usize;
        for k in 0..n {
            if k != j {
                let scale = prob.a_eq.get(r, k).abs().max(T::one());
                if prob.a_eq.get(r, k).abs() > tcoef * scale {
                    nnz += 1;
                }
            }
        }
        if nnz > max_fill {
            continue;
        }
        best[j] = Some((r, coeff, prob.b_eq[r] / coeff));
    }

    let mut elims: Vec<(usize, usize, T, T)> = Vec::new();
    let mut used_row = vec![false; me];
    for j in 0..n {
        if let Some((r, coeff, shift)) = best[j] {
            if used_row[r] {
                continue;
            }
            used_row[r] = true;
            elims.push((r, j, coeff, shift));
        }
    }

    if elims.is_empty() {
        return Ok((
            prob.clone(),
            FreeVarReduction {
                kept_cols: (0..n).collect(),
                kept_eq: (0..me).collect(),
                elims,
                n_orig: n,
                n_eq_orig: me,
            },
        ));
    }

    let mut p_k = prob.p.clone();
    let mut q_k = prob.q.clone();
    let (mut a_eq_k, mut b_eq_k) = (prob.a_eq.clone(), prob.b_eq.clone());
    let (mut a_in_k, mut b_in_k) = (prob.a_in.clone(), prob.b_in.clone());
    let elim_cols: std::collections::HashSet<usize> = elims.iter().map(|&(_, j, _, _)| j).collect();

    for &(r, j, coeff, shift) in &elims {
        let mut gamma = vec![zero; n];
        let inv_coeff = coeff.recip();
        for k in 0..n {
            if k != j {
                let a = a_eq_k.get(r, k);
                if a != zero {
                    gamma[k] = a * inv_coeff;
                }
            }
        }
        let pjj = p_k.get(j, j);
        let qj = q_k[j];
        for a in 0..n {
            if a == j || gamma[a] == zero {
                continue;
            }
            for b in 0..n {
                if b == j || gamma[b] == zero {
                    continue;
                }
                p_k.set(a, b, p_k.get(a, b) + pjj * gamma[a] * gamma[b]);
            }
            q_k[a] -= pjj * shift * gamma[a] + qj * gamma[a];
        }
        for r2 in 0..prob.b_eq.len() {
            if r2 == r {
                continue;
            }
            let a_r2j = a_eq_k.get(r2, j);
            if a_r2j != zero {
                for k in 0..n {
                    if k != j && gamma[k] != zero {
                        a_eq_k.set(r2, k, a_eq_k.get(r2, k) - a_r2j * gamma[k]);
                    }
                }
                b_eq_k[r2] -= a_r2j * shift;
                a_eq_k.set(r2, j, zero);
            }
        }
        for r2 in 0..prob.b_in.len() {
            let a_r2j = a_in_k.get(r2, j);
            if a_r2j != zero {
                for k in 0..n {
                    if k != j && gamma[k] != zero {
                        a_in_k.set(r2, k, a_in_k.get(r2, k) - a_r2j * gamma[k]);
                    }
                }
                b_in_k[r2] -= a_r2j * shift;
                a_in_k.set(r2, j, zero);
            }
        }
    }

    let kept_cols: Vec<usize> = (0..n).filter(|k| !elim_cols.contains(k)).collect();
    let elim_row_set: std::collections::HashSet<usize> =
        elims.iter().map(|&(r, _, _, _)| r).collect();
    let kept_eq: Vec<usize> = (0..me).filter(|&r| !elim_row_set.contains(&r)).collect();
    let nk = kept_cols.len();

    let mut rp = DenseMatrix::zeros(nk, nk);
    for (a, &ca) in kept_cols.iter().enumerate() {
        for (b, &cb) in kept_cols.iter().enumerate() {
            rp.set(a, b, p_k.get(ca, cb));
        }
    }
    let rq: Vec<T> = kept_cols.iter().map(|&k| q_k[k]).collect();
    let mut ra_eq = DenseMatrix::zeros(kept_eq.len(), nk);
    let mut rb_eq = vec![zero; kept_eq.len()];
    for (nr, &or) in kept_eq.iter().enumerate() {
        for (a, &c) in kept_cols.iter().enumerate() {
            ra_eq.set(nr, a, a_eq_k.get(or, c));
        }
        rb_eq[nr] = b_eq_k[or];
    }
    let mut ra_in = DenseMatrix::zeros(mi, nk);
    let mut rb_in = vec![zero; mi];
    for r in 0..mi {
        for (a, &c) in kept_cols.iter().enumerate() {
            ra_in.set(r, a, a_in_k.get(r, c));
        }
        rb_in[r] = b_in_k[r];
    }

    Ok((
        QpProblem {
            p: rp,
            q: rq,
            a_eq: ra_eq,
            b_eq: rb_eq,
            a_in: ra_in,
            b_in: rb_in,
            a_eq_csr: None,
            a_in_csr: None,
        },
        FreeVarReduction {
            kept_cols,
            kept_eq,
            elims,
            n_orig: n,
            n_eq_orig: me,
        },
    ))
}

/// Reconstruct the original solution after free-variable substitution.
pub fn restore_free_vars<T: Scalar>(
    orig: &QpProblem<T>,
    red: &FreeVarReduction<T>,
    reduced: &QpSolution<T>,
) -> QpSolution<T> {
    // No-op shortcut (no free variable eliminated): pass through unchanged,
    // skipping the full P/A_eq matvecs.
    if red.elims.is_empty() {
        return reduced.cloned();
    }
    let zero = T::zero();
    let n = red.n_orig;
    let mut x = vec![zero; n];
    for (a, &c) in red.kept_cols.iter().enumerate() {
        x[c] = reduced.x[a];
    }
    for &(r, j, coeff, shift) in &red.elims {
        let inv_coeff = coeff.recip();
        let mut acc = shift;
        for k in 0..n {
            if k == j {
                continue;
            }
            let a_rk = orig.a_eq.get(r, k);
            if a_rk != zero {
                acc -= (a_rk * inv_coeff) * x[k];
            }
        }
        x[j] = acc;
    }
    let mut y = vec![zero; red.n_eq_orig];
    for (nr, &or) in red.kept_eq.iter().enumerate() {
        y[or] = reduced.y[nr];
    }
    let px = orig.p.matvec(&x);
    let aty = orig.a_eq.matvec_t(&y);
    for &(r, j, coeff, _shift) in &red.elims {
        y[r] = -(px[j] + orig.q[j] + aty[j]) / coeff;
    }

    // The reduced problem's objective omits the substituted-out variables'
    // contribution; recompute at original dimensions so obj_val is correct
    // standalone.
    let half = T::from_f64(0.5).expect("scalar literal");
    let mut obj_val = T::zero();
    for k in 0..n {
        obj_val += half * x[k] * px[k] + orig.q[k] * x[k];
    }

    QpSolution::new(
        reduced.status,
        x,
        y,
        reduced.s.clone(),
        reduced.z.clone(),
        obj_val,
        reduced.iters,
    )
}

// ─────────────────────────────────────────────────────────────
//  Equality–equality row merging (nonzero cancellation via cross-multiply)
// ─────────────────────────────────────────────────────────────

/// Record of row-merging eliminations for postsolve.
#[derive(Clone, Debug)]
pub struct MergeReduction<T: Scalar> {
    pub kept_cols: Vec<usize>,
    pub kept_eq_rows: Vec<usize>,
    /// `(elim_col, row_a, row_b, coeff_a, coeff_b)`: row_b dropped,
    /// row_a replaced by `a_b·row_a − a_a·row_b`.
    pub merges: Vec<(usize, usize, usize, T, T)>,
    pub n_orig: usize,
    pub n_eq_orig: usize,
}

/// Merge equality rows that share a variable appearing in exactly two
/// equality rows and nowhere else (no P-coupling, no inequalities).
/// Cross-multiplying cancels the shared variable, replacing both rows with
/// one aggregate row.  Generalises doubleton elimination to dense rows.
pub fn merge_equality_rows<T: Scalar>(
    prob: &QpProblem<T>,
    max_merged_nnz: usize,
) -> Result<(QpProblem<T>, MergeReduction<T>), Status> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let tcoef = tol_coef::<T>();

    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for r in 0..me {
        for j in 0..n {
            let scale = prob.a_eq.get(r, j).abs().max(T::one());
            if prob.a_eq.get(r, j).abs() > tcoef * scale {
                col_rows[j].push(r);
            }
        }
    }

    let mut merges: Vec<(usize, usize, usize, T, T)> = Vec::new();
    let mut used_row = vec![false; me];
    for j in 0..n {
        if col_rows[j].len() != 2 {
            continue;
        }
        // `col_is_diag_p` only guarantees x_j has no OFF-diagonal P coupling; a
        // nonzero diagonal P[j,j] on the merged-out variable is a genuine
        // quadratic term that this two-row cross-multiply elimination has no
        // way to fold into the reduced problem (unlike the single-row
        // substitution used by `eliminate_free_vars`/doubleton elimination),
        // so such columns must be excluded from candidacy entirely.
        let pjj = prob.p.get(j, j);
        if !col_is_diag_p(&prob.p, j, n)
            || col_in_ineq(&prob.a_in, j, mi)
            || pjj.abs() > tcoef * pjj.abs().max(T::one())
        {
            continue;
        }
        let (r1, r2) = (col_rows[j][0], col_rows[j][1]);
        if used_row[r1] || used_row[r2] {
            continue;
        }
        let a1 = prob.a_eq.get(r1, j);
        let a2 = prob.a_eq.get(r2, j);
        let mut nnz = 0usize;
        for k in 0..n {
            if k == j {
                continue;
            }
            let v = a2 * prob.a_eq.get(r1, k) - a1 * prob.a_eq.get(r2, k);
            if v.abs() > tcoef * v.abs().max(T::one()) {
                nnz += 1;
            }
        }
        if nnz > max_merged_nnz {
            continue;
        }
        used_row[r1] = true;
        used_row[r2] = true;
        merges.push((j, r1, r2, a1, a2));
    }

    if merges.is_empty() {
        return Ok((
            prob.clone(),
            MergeReduction {
                kept_cols: (0..n).collect(),
                kept_eq_rows: (0..me).collect(),
                merges,
                n_orig: n,
                n_eq_orig: me,
            },
        ));
    }

    let idrop: std::collections::HashSet<usize> = merges.iter().map(|&(j, _, _, _, _)| j).collect();
    let kept_cols: Vec<usize> = (0..n).filter(|j| !idrop.contains(j)).collect();
    let nk = kept_cols.len();
    let row_set: std::collections::HashSet<usize> = merges
        .iter()
        .flat_map(|&(_, r1, r2, _, _)| vec![r1, r2])
        .collect();
    let kept_eq: Vec<usize> = (0..me).filter(|r| !row_set.contains(r)).collect();
    let nr_new = kept_eq.len() + merges.len();

    let mut ra_eq = DenseMatrix::zeros(nr_new, nk);
    let mut rb_eq = vec![zero; nr_new];
    for (nr, &or) in kept_eq.iter().enumerate() {
        for (a, &c) in kept_cols.iter().enumerate() {
            ra_eq.set(nr, a, prob.a_eq.get(or, c));
        }
        rb_eq[nr] = prob.b_eq[or];
    }
    for (mi_, &(_, r1, r2, a1, a2)) in merges.iter().enumerate() {
        let nr = kept_eq.len() + mi_;
        for (a, &c) in kept_cols.iter().enumerate() {
            ra_eq.set(nr, a, a2 * prob.a_eq.get(r1, c) - a1 * prob.a_eq.get(r2, c));
        }
        rb_eq[nr] = a2 * prob.b_eq[r1] - a1 * prob.b_eq[r2];
        let has_nz =
            (0..nk).any(|a| ra_eq.get(nr, a).abs() > tcoef * ra_eq.get(nr, a).abs().max(T::one()));
        if !has_nz && rb_eq[nr].abs() > tol_rhs::<T>() {
            return Err(Status::PrimalInfeasible);
        }
    }

    let mut ra_in = DenseMatrix::zeros(mi, nk);
    let mut rb_in = vec![zero; mi];
    for r in 0..mi {
        for (a, &c) in kept_cols.iter().enumerate() {
            ra_in.set(r, a, prob.a_in.get(r, c));
        }
        rb_in[r] = prob.b_in[r];
    }

    // Fold each eliminated variable's own objective term (`P_jj`, `q_j`) into the
    // surviving columns via the same substitution `restore_merged_rows` uses to
    // recover x_j: `x_j = shift - sum_{k!=j} gamma_k*x_k`, `gamma_k = a_eq(r1,k)/a1`,
    // `shift = b_eq[r1]/a1`. `col_is_diag_p` only guarantees x_j has no P
    // cross-coupling to *other* variables -- it does not guarantee `P_jj == 0` or
    // `q_j == 0`. Silently dropping those terms (the old behaviour) discarded x_j's
    // diagonal Hessian/linear cost, producing a feasible-but-suboptimal reduced
    // problem. Follows the same pattern `eliminate_free_vars` already applies for the
    // `col_is_diag_p` precondition.
    let mut p_folded = prob.p.clone();
    let mut q_folded = prob.q.clone();
    for &(j, r1, _r2, a1, _a2) in &merges {
        let pjj = prob.p.get(j, j);
        let qj = prob.q[j];
        if pjj == zero && qj == zero {
            continue;
        }
        let shift = prob.b_eq[r1] / a1;
        let mut gamma = vec![zero; n];
        for k in 0..n {
            if k != j {
                let a = prob.a_eq.get(r1, k);
                if a != zero {
                    gamma[k] = a / a1;
                }
            }
        }
        for a in 0..n {
            if a == j || gamma[a] == zero {
                continue;
            }
            for b in 0..n {
                if b == j || gamma[b] == zero {
                    continue;
                }
                p_folded.set(a, b, p_folded.get(a, b) + pjj * gamma[a] * gamma[b]);
            }
            q_folded[a] -= pjj * shift * gamma[a] + qj * gamma[a];
        }
    }

    let mut rp = DenseMatrix::zeros(nk, nk);
    for (a, &ca) in kept_cols.iter().enumerate() {
        for (b, &cb) in kept_cols.iter().enumerate() {
            rp.set(a, b, p_folded.get(ca, cb));
        }
    }
    let rq: Vec<T> = kept_cols.iter().map(|&k| q_folded[k]).collect();

    Ok((
        QpProblem {
            p: rp,
            q: rq,
            a_eq: ra_eq,
            b_eq: rb_eq,
            a_in: ra_in,
            b_in: rb_in,
            a_eq_csr: None,
            a_in_csr: None,
        },
        MergeReduction {
            kept_cols,
            kept_eq_rows: kept_eq.clone(),
            merges,
            n_orig: n,
            n_eq_orig: me,
        },
    ))
}

/// Reconstruct the original solution after equality-row merging.
pub fn restore_merged_rows<T: Scalar>(
    orig: &QpProblem<T>,
    red: &MergeReduction<T>,
    reduced: &QpSolution<T>,
) -> QpSolution<T> {
    // No-op shortcut (no equality rows merged): pass through unchanged,
    // skipping the full P matvec.
    if red.merges.is_empty() {
        return reduced.cloned();
    }
    let zero = T::zero();
    let n = red.n_orig;
    let mut x = vec![zero; n];
    for (a, &c) in red.kept_cols.iter().enumerate() {
        x[c] = reduced.x[a];
    }
    for &(j, r1, _r2, a1, _a2) in &red.merges {
        let mut acc = orig.b_eq[r1];
        for k in 0..n {
            if k == j {
                continue;
            }
            let a = orig.a_eq.get(r1, k);
            if a != zero {
                acc -= a * x[k];
            }
        }
        x[j] = acc / a1;
    }
    let mut y = vec![zero; red.n_eq_orig];
    for (nr, &or) in red.kept_eq_rows.iter().enumerate() {
        y[or] = reduced.y[nr];
    }
    let agg_start = red.kept_eq_rows.len();
    for (mi_, &(_, r1, r2, a1, a2)) in red.merges.iter().enumerate() {
        let y_agg = reduced.y[agg_start + mi_];
        y[r1] = a2 * y_agg;
        y[r2] = -a1 * y_agg;
    }

    // The reduced problem's objective omits the merged-out variables' contribution;
    // recompute at original dimensions so obj_val is correct standalone.
    let px = orig.p.matvec(&x);
    let half = T::from_f64(0.5).expect("scalar literal");
    let mut obj_val = T::zero();
    for k in 0..n {
        obj_val += half * x[k] * px[k] + orig.q[k] * x[k];
    }

    QpSolution::new(
        reduced.status,
        x,
        y,
        reduced.s.clone(),
        reduced.z.clone(),
        obj_val,
        reduced.iters,
    )
}

// ─────────────────────────────────────────────────────────────
//  Dual presolve (dualize when the dual QP is smaller)
// ─────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────
//  Auxiliary variable elimination
// ─────────────────────────────────────────────────────────────

/// Record of auxiliary variables eliminated from an equality block.
#[derive(Clone, Debug)]
pub struct AuxVarReduction<T: Scalar> {
    /// Columns kept (eliminated aux vars removed), in order.
    pub kept_cols: Vec<usize>,
    /// Equality rows kept (pivot rows removed), in order.
    pub kept_eq: Vec<usize>,
    /// One per elimination: `(pivot_row, aux_col, coeff, p_aux)` where
    /// the equality row is `coeff·x_aux + Σ_{j≠aux} aⱼxⱼ = b`
    /// with `coeff ∈ {−1, +1}`, and `P[aux,aux]` = `p_aux`.
    pub elims: Vec<(usize, usize, T, T)>,
    pub n_orig: usize,
    pub n_eq_orig: usize,
    /// Constant objective offset accumulated from folding ½·p_aux·b² terms.
    obj_offset: T,
}

impl<T: Scalar> AuxVarReduction<T> {
    pub(crate) fn no_op(n: usize) -> Self {
        Self {
            kept_cols: (0..n).collect(),
            kept_eq: Vec::new(),
            elims: Vec::new(),
            n_orig: n,
            n_eq_orig: 0,
            obj_offset: T::zero(),
        }
    }
}

/// Record of an epigraph pair folding for solution recovery.
pub struct EpigraphFoldReduction {
    /// New column index for each original column (MAX if dropped).
    pub new_col: Vec<usize>,
    /// For each folded pair: (x_orig_col, t_col, coeff).
    pub pairs: Vec<(usize, usize, f64)>,
    /// Original row indices of the kept inequality rows (in folded order).
    /// Row idx in folded s/z → original row index.
    pub kept_rows: Vec<usize>,
    /// Original dimensions.
    pub n_orig: usize,
    pub mi_orig: usize,
}

/// Map a folded solution back to original variables.
pub fn restore_epigraph_pairs<T: Scalar>(
    orig: &QpProblem<T>,
    recovery: &EpigraphFoldReduction,
    sol_folded: &QpSolution<T>,
) -> QpSolution<T> {
    let zero = T::zero();
    let n = recovery.n_orig;
    let mi = recovery.mi_orig;
    let mut x = vec![zero; n];
    let mut s = vec![zero; mi];
    let mut z = vec![zero; mi];

    // Recover x: for each original column, map from folded pos/neg columns
    for j in 0..n {
        let nj = recovery.new_col[j];
        if nj == usize::MAX {
            continue;
        } // dropped (t variables)
          // Check if this is a folded x variable (has a neg column at nj+1 with
          // nj+1 < sol_folded.x.len() and nj+1 not in new_col for any orig col)
        let is_folded = j < n && recovery.pairs.iter().any(|&(x_col, _, _)| x_col == j);
        if is_folded && nj + 1 < sol_folded.x.len() {
            // x_orig = x_pos - x_neg
            x[j] = sol_folded.x[nj] - sol_folded.x[nj + 1];
        } else {
            // Non-folded variable: direct copy
            x[j] = sol_folded.x[nj];
        }
    }

    // Recover t variables from the epigraph condition: t = |c|·(x_pos + x_neg)
    for &(x_col, t_col, coeff) in &recovery.pairs {
        let nj = recovery.new_col[x_col];
        if nj + 1 < sol_folded.x.len() {
            let c_abs = T::from_f64(coeff.abs()).expect("scalar literal");
            x[t_col] = c_abs * (sol_folded.x[nj] + sol_folded.x[nj + 1]);
        }
    }

    // For s and z: the folded problem has kept rows at positions 0..kept_rows.len()
    // followed by 2k bound rows for split-variable nonnegativity.  Only the kept rows
    // map back to original constraints; bound-row duals are internal auxiliary values.
    // Dropped rows (epigraph constraints removed by the fold) have dual zero.
    let n_kept = recovery.kept_rows.len();
    for ni in 0..n_kept {
        let old_r = recovery.kept_rows[ni];
        s[old_r] = sol_folded.s[ni];
        z[old_r] = sol_folded.z[ni];
    }

    let _orig = orig; // reserved for future dimension validation

    QpSolution::new(
        sol_folded.status,
        x,
        sol_folded.y.clone(),
        s,
        z,
        sol_folded.obj_val,
        sol_folded.iters,
    )
}

/// Fold epigraph inequality pairs into split-variable bounds.
///
/// CVXPY's |x| and huber reformulations introduce auxiliary `t` variables
/// with inequality pairs `±c·x_j − t_i ≤ 0` (encoding `|c·x_j| ≤ t_i`).
/// At optimality `t_i = |c·x_j|` (the objective penalizes `t_i` linearly),
/// so this pass splits `x_j = x_j⁺ − x_j⁻` with `x_j⁺, x_j⁻ ≥ 0`, folds
/// the `λ·t_i` penalty into `λ·|c|·(x_j⁺ + x_j⁻)`, and drops both the `t_i`
/// variable and its two epigraph rows.
///
/// The result is a NonNeg-only QP where ALL remaining inequality rows are
/// singleton bounds — the condensed Cholesky path then factors the n×n
/// dense Hessian instead of the (n+mi)×(n+mi) augmented KKT.
///
/// Returns `None` when no epigraph structure is detected, or `Some(prob, recovery)`.
pub fn fold_epigraph_pairs<T: Scalar>(
    prob: &QpProblem<T>,
    sp: &SparseAIn<T>,
) -> Option<(QpProblem<T>, EpigraphFoldReduction)> {
    let n = prob.q.len();
    let mi = prob.b_in.len();
    let me = prob.b_eq.len();
    let zero = T::zero();
    let one = T::one();

    if me > 0 || mi < 2 || n < 4 {
        return None;
    }

    // Scan inequality rows for epigraph pairs: for each variable, look for
    // two rows `+c·x_j − t_i ≤ 0` and `−c·x_j − t_i ≤ 0`.
    // Structure: each epigraph pair involves the same `t_i` (coeff −1 in both
    // rows) and opposite coefficients on `x_j` (±c). The `t_i` variable does
    // NOT appear in any other row.
    #[derive(Clone)]
    struct EpiPair<T: Scalar> {
        t_col: usize, // column of the t variable
        x_col: usize, // column of the x variable
        coeff: T,     // coefficient |c| on x (positive)
        t_penalty: T, // linear penalty on t in objective (λ)
    }

    use std::collections::HashMap;
    let mut colpair_rows: HashMap<(usize, usize), Vec<(usize, T, T)>> = HashMap::new();
    // key=(min_col, max_col) → vec of (row, coeff_min, coeff_max)
    for r in 0..mi {
        // Row scans through the sparse views (sp is a parameter — only the
        // check_col below used it before): the dense per-row scan is a full
        // O(mi·n) pass on the big-sparse me=0 LPs (transport/LASSO/huber
        // shapes) that the views already carry at O(nnz).
        let cols: Vec<(usize, T)> = match sp.row(r) {
            Some(nz) => nz.to_vec(),
            None => {
                let mut cols = Vec::new();
                for j in 0..n {
                    let a = prob.a_in.get(r, j);
                    if a != zero {
                        cols.push((j, a));
                    }
                }
                cols
            }
        };
        if cols.len() != 2 {
            continue;
        }
        let (c0, a0) = cols[0];
        let (c1, a1) = cols[1];
        let key = if c0 < c1 { (c0, c1) } else { (c1, c0) };
        let (ca, cb) = if c0 < c1 { (a0, a1) } else { (a1, a0) };
        colpair_rows.entry(key).or_default().push((r, ca, cb));
    }

    if colpair_rows.is_empty() {
        return None;
    }

    let mut epi_pairs: Vec<EpiPair<T>> = Vec::new();
    let mut drop_rows: Vec<bool> = vec![false; mi];
    let mut paired_t: Vec<bool> = vec![false; n];
    let mut paired_x: Vec<bool> = vec![false; n];

    for ((c0, c1), rows) in &colpair_rows {
        if rows.len() != 2 {
            continue;
        }
        let (r0, a00, a01) = rows[0];
        let (r1, a10, a11) = rows[1];
        // Identify which column is the x variable: its coefficient changes sign
        // between the two rows (e.g. +1 in one, −1 in the other). The t coefficient
        // stays the same sign (−1 in both in CVXPY's convention).
        // x = changing, t = constant.
        let (t_col, x_col, x_coeff): (usize, usize, T);
        if a00 == a10 && a01 == one && a11 == -one {
            // c0 constant (t), c1 changes: +1→−1 (x)
            t_col = *c0;
            x_col = *c1;
            x_coeff = a01.abs();
        } else if a00 == a10 && a01 == -one && a11 == one {
            // c0 constant (t), c1 changes: −1→+1 (x)
            t_col = *c0;
            x_col = *c1;
            x_coeff = a01.abs();
        } else if a01 == a11 && a00 == one && a10 == -one {
            // c1 constant (t), c0 changes: +1→−1 (x)
            t_col = *c1;
            x_col = *c0;
            x_coeff = a00.abs();
        } else if a01 == a11 && a00 == -one && a10 == one {
            // c1 constant (t), c0 changes: −1→+1 (x)
            t_col = *c1;
            x_col = *c0;
            x_coeff = a00.abs();
        } else {
            continue;
        }
        let c_abs = x_coeff.abs();
        if c_abs == zero {
            continue;
        }

        // t must have no off-diagonal P entries
        let mut p_ok = true;
        for k in 0..n {
            if k != t_col && (prob.p.get(t_col, k) != zero || prob.p.get(k, t_col) != zero) {
                p_ok = false;
                break;
            }
        }
        if !p_ok {
            continue;
        }
        // Soundness: the fold drops BOTH pair rows and removes the `t` column
        // outright, and splits `x_j = x⁺ − x⁻` with only `x⁺` mapped into the
        // surviving rows. So `t` must appear in no other inequality row (its
        // coefficient there would silently vanish from the problem), and `x_j`
        // must appear in no other inequality row either (the `−x⁻` term would
        // be dropped). Equality rows are excluded by the `me > 0` guard at the
        // top; `t`'s P couplings are checked above.
        let check_col = |c: usize, r0: usize, r1: usize| -> bool {
            // True when column c has no nonzero outside rows r0/r1.
            match sp.col(c) {
                Some(nz) => nz.iter().all(|&(r, _)| r == r0 || r == r1),
                None => {
                    for r in 0..mi {
                        if r == r0 || r == r1 {
                            continue;
                        }
                        if prob.a_in.get(r, c) != zero {
                            return false;
                        }
                    }
                    true
                }
            }
        };
        if !check_col(t_col, r0, r1) || !check_col(x_col, r0, r1) {
            continue;
        }
        // t must not appear in any other row pair
        if paired_t[t_col] || paired_x[x_col] {
            continue;
        }

        let t_penalty = prob.q[t_col];
        if t_penalty < zero {
            continue;
        }

        epi_pairs.push(EpiPair {
            t_col,
            x_col,
            coeff: c_abs,
            t_penalty,
        });
        paired_t[t_col] = true;
        paired_x[x_col] = true;
        drop_rows[r0] = true;
        drop_rows[r1] = true;
    }

    if epi_pairs.is_empty() {
        return None;
    }
    let k = epi_pairs.len();

    // Build transformed problem: each folded variable x_j becomes x_j⁺, x_j⁻.
    // New dimensions: n_new = n_orig + k (one extra column per folded pair).
    // Keep: all columns EXCEPT the t columns, plus new x_j⁻ columns.
    // Mapping: old_col → new_col (usize::MAX if dropped)
    let mut new_col: Vec<usize> = vec![usize::MAX; n];
    let mut split_info: Vec<(usize, usize)> = Vec::new(); // (new_col_pos, new_col_neg) for each folded x
    let mut next_col = 0usize;
    for j in 0..n {
        if paired_t[j] {
            continue;
        } // drop t variables
        new_col[j] = next_col;
        next_col += 1;
        if paired_x[j] {
            // x_j gets an extra neg-part column
            split_info.push((new_col[j], next_col));
            next_col += 1; // allocate column for x_j⁻
        }
    }
    let n_new = next_col;

    // Keep all inequality rows except the dropped epigraph rows
    let kept_rows: Vec<usize> = (0..mi).filter(|&r| !drop_rows[r]).collect();
    let mi_new = kept_rows.len() + 2 * k; // kept rows + 2 bound rows per folded pair

    // Check if the t penalty is nonzero (has diagonal P)
    let has_t_diag = epi_pairs
        .iter()
        .any(|ep| prob.p.get(ep.t_col, ep.t_col) != zero);
    if has_t_diag {
        return None;
    }

    // Build new P: n_new × n_new.  Compute directly from the original P to avoid
    // order-dependent cross-term interference between folded variables.
    let mut p_new = DenseMatrix::<T>::zeros(n_new, n_new);
    let mut q_new = vec![zero; n_new];

    // Copy non-folded entries
    for j in 0..n {
        let nj = new_col[j];
        if nj == usize::MAX {
            continue;
        }
        q_new[nj] = prob.q[j];
        for l in 0..n {
            let nl = new_col[l];
            if nl == usize::MAX {
                continue;
            }
            let v = prob.p.get(j, l);
            if v != zero {
                p_new.set(nj, nl, v);
            }
        }
    }

    // For each folded variable x_j → x_j⁺ − x_j⁻, expand P:
    //   P_new[pos,pos] = P_orig[j,j]
    //   P_new[pos,neg] = −P_orig[j,j]
    //   P_new[neg,neg] = +P_orig[j,j]
    //   For k ≠ j:  P_new[pos, k'] = P_orig[j, k]
    //               P_new[neg, k'] = −P_orig[j, k]   (k' = new_col[k])
    for ep in &epi_pairs {
        let pos_col = new_col[ep.x_col];
        // Find neg_col: the column right after pos_col
        let neg_col = pos_col + 1; // allocated right after pos_col
        let x_orig = ep.x_col;
        let t_penalty = ep.t_penalty * ep.coeff;
        q_new[pos_col] += t_penalty;
        // x_orig = pos − neg → qᵀx_orig = qᵀpos + (−q)ᵀneg
        // neg starts at 0; add -q[x_orig] + t_penalty
        q_new[neg_col] += -prob.q[x_orig] + t_penalty;

        // Diagonal block expansion from original P[x_orig, x_orig]
        let p_xx = prob.p.get(x_orig, x_orig);
        p_new.set(neg_col, neg_col, p_new.get(neg_col, neg_col) + p_xx);
        p_new.set(pos_col, neg_col, p_new.get(pos_col, neg_col) - p_xx);
        p_new.set(neg_col, pos_col, p_new.get(neg_col, pos_col) - p_xx);

        // Cross-terms: for every other variable l (original), map to new column k'
        for l in 0..n {
            if l == x_orig {
                continue;
            }
            let kp = new_col[l];
            if kp == usize::MAX || kp == pos_col || kp == neg_col {
                continue;
            }
            let p_xl = prob.p.get(x_orig, l);
            if p_xl != zero {
                // P_new[pos, k'] already has P_orig[x_orig, l] from copy.
                // P_new[neg, k'] should be −P_orig[x_orig, l].
                p_new.set(neg_col, kp, -p_xl);
                p_new.set(kp, neg_col, -p_xl);
            }
        }
        // Also handle cross-terms with pos/neg of OTHER folded variables:
        // these are the neg_col of other split vars, not covered by the original-P loop.
        for other_ep in &epi_pairs {
            if other_ep.x_col <= x_orig {
                continue;
            } // process each unordered pair once
            let other_pos = new_col[other_ep.x_col];
            let other_neg = other_pos + 1;
            let p_xo = prob.p.get(x_orig, other_ep.x_col);
            if p_xo != zero {
                // P_new[pos, other_pos] = P_orig[x_orig, other_x] (already from copy)
                // P_new[pos, other_neg] = -P_orig[x_orig, other_x]
                p_new.set(pos_col, other_neg, -p_xo);
                p_new.set(other_neg, pos_col, -p_xo);
                // P_new[neg, other_pos] = -P_orig[x_orig, other_x]
                p_new.set(neg_col, other_pos, -p_xo);
                p_new.set(other_pos, neg_col, -p_xo);
                // P_new[neg, other_neg] = +P_orig[x_orig, other_x] (product of two neg signs)
                p_new.set(neg_col, other_neg, p_new.get(neg_col, other_neg) + p_xo);
                p_new.set(other_neg, neg_col, p_new.get(other_neg, neg_col) + p_xo);
            }
        }
    }

    // Build new A_in
    let mut a_in_new = DenseMatrix::<T>::zeros(mi_new, n_new);
    let mut b_in_new = vec![zero; mi_new];

    // Copy kept rows (sparse rows when available — same nonzero set, same
    // ascending column order, so the rebuilt A_in is bit-identical)
    for (ni, &old_r) in kept_rows.iter().enumerate() {
        match sp.row(old_r) {
            Some(nz) => {
                for &(j, v) in nz {
                    let nj = new_col[j];
                    if nj != usize::MAX {
                        a_in_new.set(ni, nj, v);
                    }
                }
            }
            None => {
                for j in 0..n {
                    let nj = new_col[j];
                    if nj == usize::MAX {
                        continue;
                    }
                    let v = prob.a_in.get(old_r, j);
                    if v != zero {
                        a_in_new.set(ni, nj, v);
                    }
                }
            }
        }
        b_in_new[ni] = prob.b_in[old_r];
    }

    // Add bound rows for split variables: -x_j⁺ ≤ 0 and -x_j⁻ ≤ 0
    let mut bound_row = kept_rows.len();
    for &(pos_col, neg_col) in &split_info {
        a_in_new.set(bound_row, pos_col, -one);
        b_in_new[bound_row] = zero;
        bound_row += 1;
        a_in_new.set(bound_row, neg_col, -one);
        b_in_new[bound_row] = zero;
        bound_row += 1;
    }

    let pairs_f64: Vec<(usize, usize, f64)> = epi_pairs
        .iter()
        .map(|ep| (ep.x_col, ep.t_col, ep.coeff.to_f64().expect("finite scalar")))
        .collect();
    // Build CSR for the singleton-bound A_in so matvecs are O(nnz)=O(mi)
    // instead of O(n*mi) dense BLAS — critical for folded LASSO/huber.
    let a_in_csr = if mi_new > 0 && n_new >= 64 {
        Some(iconic_ipm::conic::csr_of_dense(&a_in_new, mi_new, n_new))
    } else {
        None
    };
    // No low-rank precache here: LASSO's fold rank exceeds the Woodbury gate
    // (r <= n/3), and dense Cholesky on the folded problem is already fast.

    Some((
        QpProblem {
            p: p_new,
            q: q_new,
            a_eq: DenseMatrix::<T>::zeros(0, n_new),
            b_eq: vec![],
            a_in: a_in_new,
            b_in: b_in_new,
            a_eq_csr: None,
            a_in_csr,
        },
        EpigraphFoldReduction {
            new_col,
            pairs: pairs_f64,
            kept_rows,
            n_orig: n,
            mi_orig: mi,
        },
    ))
}

/// Eliminate auxiliary variables: each variable that appears in exactly one
/// equality row with coefficient ±1, has a diagonal P entry ≥ 0, and is
/// otherwise isolated (no other constraint rows, no off-diagonal P couplings)
/// is substituted out.  Its quadratic cost `½ p_aux · y²` is folded into the
/// surviving variables' P/q via `y = coeff·(b − Σ aⱼxⱼ)`.
///
/// This is the key presolve for CVXPY-decomposed QPs: CVXPY's
/// epigraph / sum_squares reformulations introduce auxiliary variables with
/// diagonal P that are tied to exactly one equality row.  Substituting them
/// out reduces n, removes equality rows, and avoids the dense-Schur-elimination
/// path in the QP solver.
///
/// Only variables whose coefficient magnitude is exactly 1 are eliminated
/// (preserving exact arithmetic: no division by the coefficient is needed,
/// only the sign flip).  The equality row is scaled by `coeff` so the
/// substitution reads `y = coeff·(b − Σ aⱼxⱼ)` with coeff² = 1.
pub fn eliminate_auxiliary_vars<T: Scalar>(
    prob: &QpProblem<T>,
) -> Result<(QpProblem<T>, AuxVarReduction<T>), Status> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let one = T::one();
    let half = T::from_f64(0.5).expect("scalar literal");

    if me == 0 || n < 4 {
        return Ok((prob.clone(), AuxVarReduction::no_op(n)));
    }

    // --- find candidates: variables with coeff ±1 in exactly one equality row ---
    let mut aux_row: Vec<Option<usize>> = vec![None; n]; // which eq row (if any) this var is aux in
    let mut aux_coeff: Vec<T> = vec![zero; n]; // coefficient in that row

    for r in 0..me {
        for j in 0..n {
            let a = prob.a_eq.get(r, j);
            if a == zero {
                continue;
            }
            let abs_a = a.abs();
            if abs_a == one {
                // Candidate: variable j appears with ±1 in row r
                if aux_row[j].is_none() {
                    aux_row[j] = Some(r);
                    aux_coeff[j] = a;
                } else {
                    // Variable appears with ±1 in multiple equality rows → not isolated
                    aux_row[j] = None; // poison
                }
            }
        }
    }

    // --- filter: candidate must have diagonal P, no off-diagonal P, and appear
    //     in no other equality/inequality row besides its pivot row ---
    let mut elims: Vec<(usize, usize, T, T)> = Vec::new(); // (pivot_row, aux_col, coeff, p_aux)
    let mut eliminated_rows: Vec<bool> = vec![false; me];
    let mut eliminated_cols: Vec<bool> = vec![false; n];

    for j in 0..n {
        if eliminated_cols[j] {
            continue;
        }
        let row_opt = aux_row[j];
        if row_opt.is_none() {
            continue;
        }
        let r = row_opt.expect("checked is_none above");
        if eliminated_rows[r] {
            continue;
        }
        let coeff = aux_coeff[j];
        let p_aux = prob.p.get(j, j);

        // P must be non-negative (so ½ p_aux · y² is convex) — p_aux == 0 is fine
        // (linear auxiliary, fold into q only).
        if p_aux < zero {
            continue;
        }

        // Check: no off-diagonal P entries for this variable
        let mut p_offdiag = false;
        for k in 0..n {
            if k != j && (prob.p.get(j, k) != zero || prob.p.get(k, j) != zero) {
                p_offdiag = true;
                break;
            }
        }
        if p_offdiag {
            continue;
        }

        // Check: variable appears in no other equality row
        let mut other_eq = false;
        for r2 in 0..me {
            if r2 != r && prob.a_eq.get(r2, j) != zero {
                other_eq = true;
                break;
            }
        }
        if other_eq {
            continue;
        }

        // Check: variable appears in no inequality row
        let mut in_ineq = false;
        for r2 in 0..mi {
            if prob.a_in.get(r2, j) != zero {
                in_ineq = true;
                break;
            }
        }
        if in_ineq {
            continue;
        }

        // Accept this candidate
        elims.push((r, j, coeff, p_aux));
        eliminated_rows[r] = true;
        eliminated_cols[j] = true;
    }

    if elims.is_empty() {
        return Ok((prob.clone(), AuxVarReduction::no_op(n)));
    }

    // --- build reduced problem ---
    let kept_cols: Vec<usize> = (0..n).filter(|&j| !eliminated_cols[j]).collect();
    let kept_eq: Vec<usize> = (0..me).filter(|&r| !eliminated_rows[r]).collect();
    let n_new = kept_cols.len();
    let me_new = kept_eq.len();

    let mut p_new = DenseMatrix::<T>::zeros(n_new, n_new);
    let mut q_new = vec![zero; n_new];

    // Copy surviving P entries
    for (ni, &oi) in kept_cols.iter().enumerate() {
        for (nj, &oj) in kept_cols.iter().enumerate() {
            let v = prob.p.get(oi, oj);
            if v != zero {
                p_new.set(ni, nj, v);
            }
        }
        q_new[ni] = prob.q[oi];
    }

    // Fold eliminated variables into P and q
    // For each eliminated aux: y = coeff*(b − Σ aⱼxⱼ)
    // Contribution: ½ p_aux · (b − Σ aⱼxⱼ)² + q_aux · coeff · (b − Σ aⱼxⱼ)
    //   = ½ p_aux · [b² − 2b·Σ aⱼxⱼ + (Σ aⱼxⱼ)²] + q_aux·coeff·b − q_aux·coeff·Σ aⱼxⱼ
    // P contribution: p_aux · aⱼ · aₖ  (for the cross terms)
    // q contribution: −p_aux · b · aⱼ − q_aux · coeff · aⱼ
    //   (The eliminated variable's linear term q_aux·y was previously dropped
    //   entirely, despite the selection comment promising "linear auxiliary,
    //   fold into q only" — reproduced on linked_qp(2,47): every eliminated aux
    //   carried a nonzero q, the reduced problem solved a q-mutilated objective,
    //   and the returned point was ~0.64 suboptimal — true optimum
    //   -0.8990479320 vs iconic -0.25943, identical with presolve on or off
    //   because this pass runs inside the cone-path solve.)
    // Constant offset: ½ p_aux · b² + q_aux · coeff · b
    let mut obj_offset = zero;
    for &(r, j, coeff, p_aux) in &elims {
        let b = prob.b_eq[r];
        let q_aux = prob.q[j];
        // Collect surviving coefficients from this row
        for (nk, &ok) in kept_cols.iter().enumerate() {
            let ak = prob.a_eq.get(r, ok);
            if ak == zero {
                continue;
            }
            // q contribution
            q_new[nk] -= p_aux * b * ak; // y² = (b − Σ aⱼxⱼ)² since coeff²=1
            q_new[nk] -= q_aux * coeff * ak; // linear term q_aux·y
                                             // P contribution: cross terms. Write BOTH triangles: the fold adds
                                             // p_aux·aₖ·aₗ to P[k,l] for k ≤ l only (the "upper" triangle of the
                                             // row-major DenseMatrix), leaving P[l,k] = 0 — an asymmetric Hessian
                                             // that silently corrupts the reduced solve (mirror
                                             // eliminate_doubleton_eqs, which folds both).
            for (nl, &ol) in kept_cols.iter().enumerate() {
                if nl < nk {
                    continue;
                } // sweep the upper triangle once
                let al = prob.a_eq.get(r, ol);
                if al == zero {
                    continue;
                }
                let add = p_aux * ak * al; // coeff² = 1
                if nl == nk {
                    p_new.set(nk, nk, p_new.get(nk, nk) + add);
                } else {
                    p_new.set(nk, nl, p_new.get(nk, nl) + add);
                    p_new.set(nl, nk, p_new.get(nl, nk) + add);
                }
            }
        }
        // Constant offset
        obj_offset += half * p_aux * b * b;
        obj_offset += q_aux * coeff * b; // linear term q_aux·y
    }

    // Build reduced equality block
    let mut a_eq_new = DenseMatrix::<T>::zeros(me_new, n_new);
    let mut b_eq_new = vec![zero; me_new];
    for (nr, &old_r) in kept_eq.iter().enumerate() {
        for (nj, &oj) in kept_cols.iter().enumerate() {
            let v = prob.a_eq.get(old_r, oj);
            if v != zero {
                a_eq_new.set(nr, nj, v);
            }
        }
        b_eq_new[nr] = prob.b_eq[old_r];
    }

    // Inequality block: unchanged except columns
    let mut a_in_new = DenseMatrix::<T>::zeros(mi, n_new);
    let mut b_in_new = vec![zero; mi];
    for r in 0..mi {
        for (nj, &oj) in kept_cols.iter().enumerate() {
            let v = prob.a_in.get(r, oj);
            if v != zero {
                a_in_new.set(r, nj, v);
            }
        }
        b_in_new[r] = prob.b_in[r];
    }

    let reduced = QpProblem {
        p: p_new,
        q: q_new,
        a_eq: a_eq_new,
        b_eq: b_eq_new,
        a_in: a_in_new,
        b_in: b_in_new,
        a_eq_csr: None,
        a_in_csr: None,
    };

    Ok((
        reduced,
        AuxVarReduction {
            kept_cols,
            kept_eq,
            elims,
            n_orig: n,
            n_eq_orig: me,
            obj_offset,
        },
    ))
}

/// Postsolve: recover the original solution from the reduced problem solution.
/// Reconstructs eliminated auxiliary variables and their equality dual multipliers.
pub fn restore_auxiliary_vars<T: Scalar>(
    red: &AuxVarReduction<T>,
    prob_orig: &QpProblem<T>,
    sol_red: &QpSolution<T>,
) -> QpSolution<T> {
    let zero = T::zero();
    let n = red.n_orig;
    let me = red.n_eq_orig;
    let mi = prob_orig.b_in.len();

    // Reconstruct x: insert eliminated variables
    let mut x = vec![zero; n];
    for (ni, &oi) in red.kept_cols.iter().enumerate() {
        x[oi] = sol_red.x[ni];
    }
    // Compute eliminated aux variables: y = coeff*(b − Σ aⱼxⱼ)
    for &(r, j, coeff, _p_aux) in &red.elims {
        let b = prob_orig.b_eq[r];
        let mut sum_ax = zero;
        for k in 0..n {
            if k == j {
                continue;
            }
            let ak = prob_orig.a_eq.get(r, k);
            if ak != zero {
                sum_ax += ak * x[k];
            }
        }
        x[j] = coeff * (b - sum_ax);
    }

    // Reconstruct y (equality duals)
    let mut y = vec![zero; me];
    for (ni, &oi) in red.kept_eq.iter().enumerate() {
        y[oi] = sol_red.y[ni];
    }
    // Recover duals for eliminated rows from the stationarity condition:
    // P_aux * x_aux + q_aux + coeff * z_eq + Σ(other dual contributions) = 0
    // Since x_aux appears only in this one equality row (with coeff ±1) and
    // nowhere else: z_eq = −coeff * (P_aux * x_aux + q_aux)
    for &(r, j, coeff, p_aux) in &red.elims {
        let q_aux = prob_orig.q[j];
        y[r] = -coeff * (p_aux * x[j] + q_aux);
    }

    // s and z (inequality duals) are unchanged dimension
    let mut s = sol_red.s.clone();
    let mut z_copy = sol_red.z.clone();
    s.resize(mi, zero);
    z_copy.resize(mi, zero);

    QpSolution::new(
        sol_red.status,
        x,
        y,
        s,
        z_copy,
        sol_red.obj_val + red.obj_offset,
        sol_red.iters,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use iconic_core::Settings;
    use iconic_ipm::solve_qp;

    /// A redundant null inequality row is dropped; restored slack = b, dual = 0.
    #[test]
    fn drops_null_inequality_row() {
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
        let (reduced, red) = reduce_rows(&prob, &SparseAIn::build(&prob)).unwrap();
        assert_eq!(reduced.b_in.len(), 1);
        let restored = restore_rows(&red, &solve_qp(&reduced, &Settings::<f64>::default()));
        assert_eq!(restored.status, Status::Solved);
        assert!((restored.x[0] - 1.0).abs() < 1e-6);
        assert!((restored.x[1] - 1.0).abs() < 1e-6);
        assert!((restored.s[1] - 5.0).abs() < 1e-12);
        assert!(restored.z[1].abs() < 1e-12);
    }

    /// An inequality implied by the variable bounds (max activity ≤ rhs) is dropped, and
    /// the recovered solution — primal, dual, slack — matches a direct solve.
    #[test]
    fn drops_redundant_inequality_via_bounds() {
        // min ½‖x − 2·1‖²  s.t.  0 ≤ x_j ≤ 1 and x0+x1+x2 ≤ 10 (redundant: sup activity = 3).
        // The optimum clamps each x_j to its upper bound 1; the aggregate row is implied.
        #[rustfmt::skip]
        let a_in = DenseMatrix::from_row_major(7, 3, vec![
            1.0, 0.0, 0.0,   //  x0 ≤ 1
            0.0, 1.0, 0.0,   //  x1 ≤ 1
            0.0, 0.0, 1.0,   //  x2 ≤ 1
           -1.0, 0.0, 0.0,   // -x0 ≤ 0
            0.0,-1.0, 0.0,   // -x1 ≤ 0
            0.0, 0.0,-1.0,   // -x2 ≤ 0
            1.0, 1.0, 1.0,   //  x0+x1+x2 ≤ 10  (redundant)
        ]);
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            q: vec![-2.0, -2.0, -2.0],
            a_eq: DenseMatrix::zeros(0, 3),
            b_eq: vec![],
            a_in,
            b_in: vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 10.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let direct = solve_qp(&prob, &Settings::<f64>::default());
        let (reduced, red) = remove_redundant_ineqs(&prob, &SparseAIn::build(&prob)).unwrap();
        assert_eq!(reduced.b_in.len(), 6, "the aggregate row should be dropped");
        let restored =
            restore_redundant_ineqs(&red, &solve_qp(&reduced, &Settings::<f64>::default()));
        assert_eq!(restored.status, Status::Solved);
        for j in 0..3 {
            assert!((restored.x[j] - 1.0).abs() < 1e-6, "x{j}={}", restored.x[j]);
            assert!((restored.x[j] - direct.x[j]).abs() < 1e-6, "x{j} vs direct");
        }
        // The dropped row is restored with positive slack (10 − 3 = 7) and zero dual.
        assert!(
            (restored.s[6] - 7.0).abs() < 1e-6,
            "slack={}",
            restored.s[6]
        );
        assert!(restored.z[6].abs() < 1e-9);
    }

    /// A genuinely active aggregate inequality (max activity > rhs) is NOT dropped.
    #[test]
    fn keeps_non_redundant_inequality() {
        // Bounds 0 ≤ x_j ≤ 1 and x0+x1+x2 ≤ 2 (sup activity 3 > 2): binding, must be kept.
        #[rustfmt::skip]
        let a_in = DenseMatrix::from_row_major(7, 3, vec![
            1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,
           -1.0, 0.0, 0.0,  0.0,-1.0, 0.0,  0.0, 0.0,-1.0,
            1.0, 1.0, 1.0,
        ]);
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            q: vec![-2.0, -2.0, -2.0],
            a_eq: DenseMatrix::zeros(0, 3),
            b_eq: vec![],
            a_in,
            b_in: vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, _) = remove_redundant_ineqs(&prob, &SparseAIn::build(&prob)).unwrap();
        assert_eq!(
            reduced.b_in.len(),
            7,
            "the binding aggregate row must be kept"
        );
    }

    /// Bound-activity infeasibility: 0 ≤ x_j ≤ 1 but x0+x1+x2 ≤ −1 (min activity 0 > −1).
    #[test]
    fn detects_bound_activity_infeasibility() {
        #[rustfmt::skip]
        let a_in = DenseMatrix::from_row_major(7, 3, vec![
            1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,
           -1.0, 0.0, 0.0,  0.0,-1.0, 0.0,  0.0, 0.0,-1.0,
            1.0, 1.0, 1.0,
        ]);
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            q: vec![0.0; 3],
            a_eq: DenseMatrix::zeros(0, 3),
            b_eq: vec![],
            a_in,
            b_in: vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        assert_eq!(
            remove_redundant_ineqs(&prob, &SparseAIn::build(&prob)).err(),
            Some(Status::PrimalInfeasible)
        );
    }

    /// A dominated parallel inequality (looser bound, same direction) is dropped and
    /// the recovered solution matches the original.
    #[test]
    fn drops_dominated_parallel_row() {
        // x0+x1 ≥ 2  (−x0−x1 ≤ −2) and the looser −2x0−2x1 ≤ −1 (i.e. x0+x1 ≥ 0.5).
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(2, 2, vec![-1.0, -1.0, -2.0, -2.0]),
            b_in: vec![-2.0, -1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, red) = reduce_rows(&prob, &SparseAIn::build(&prob)).unwrap();
        assert_eq!(reduced.b_in.len(), 1, "one parallel row should remain");
        let restored = restore_rows(&red, &solve_qp(&reduced, &Settings::<f64>::default()));
        assert_eq!(restored.status, Status::Solved);
        assert!((restored.x[0] - 1.0).abs() < 1e-6, "x0={}", restored.x[0]);
        assert!((restored.x[1] - 1.0).abs() < 1e-6, "x1={}", restored.x[1]);
        // Both original rows restored: the kept one active, the dominated one slack>0.
        assert_eq!(restored.s.len(), 2);
        assert!(restored.s.iter().all(|&si| si > -1e-7));
        assert!(restored.z.iter().all(|&zi| zi > -1e-7));
    }

    /// Exact duplicate inequality rows collapse to one.
    #[test]
    fn drops_duplicate_row() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(2, 2, vec![-1.0, -1.0, -1.0, -1.0]),
            b_in: vec![-2.0, -2.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, _) = reduce_rows(&prob, &SparseAIn::build(&prob)).unwrap();
        assert_eq!(reduced.b_in.len(), 1);
    }

    /// Anti-parallel rows x0 ≤ −1 and −x0 ≤ −1 (x0 ≥ 1) are contradictory.
    #[test]
    fn detects_anti_parallel_infeasibility() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(1, 1, vec![1.0]),
            q: vec![0.0],
            a_eq: DenseMatrix::zeros(0, 1),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(2, 1, vec![1.0, -1.0]),
            b_in: vec![-1.0, -1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        assert!(matches!(
            reduce_rows(&prob, &SparseAIn::build(&prob)),
            Err(Status::PrimalInfeasible)
        ));
    }

    /// A feasible two-sided range (−1 ≤ x0 ≤ 3) keeps both anti-parallel rows.
    #[test]
    fn keeps_feasible_two_sided_range() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(1, 1, vec![1.0]),
            q: vec![0.0],
            a_eq: DenseMatrix::zeros(0, 1),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(2, 1, vec![1.0, -1.0]),
            b_in: vec![3.0, 1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, _) = reduce_rows(&prob, &SparseAIn::build(&prob)).unwrap();
        assert_eq!(reduced.b_in.len(), 2);
    }

    #[test]
    fn detects_contradictory_equality() {
        let infeasible = QpProblem {
            p: DenseMatrix::from_row_major(1, 1, vec![1.0]),
            q: vec![0.0],
            a_eq: DenseMatrix::from_row_major(1, 1, vec![0.0]),
            b_eq: vec![1.0],
            a_in: DenseMatrix::zeros(0, 1),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        assert!(matches!(
            reduce_rows(&infeasible, &SparseAIn::build(&infeasible)),
            Err(Status::PrimalInfeasible)
        ));
        let redundant = QpProblem {
            b_eq: vec![0.0],
            a_eq_csr: None,
            a_in_csr: None,
            ..infeasible
        };
        let (reduced, _) = reduce_rows(&redundant, &SparseAIn::build(&redundant)).unwrap();
        assert_eq!(reduced.b_eq.len(), 0);
    }

    /// An empty (unconstrained) column is fixed at its closed-form minimizer and
    /// reinserted by postsolve.
    #[test]
    fn eliminates_empty_column() {
        // min ½(x0² + 2x1²) − 4x1  s.t.  x0 ≤ 1.  x1 is empty → x1 = 4/2 = 2.
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 2.0]),
            q: vec![0.0, -4.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(1, 2, vec![1.0, 0.0]),
            b_in: vec![1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, col) = eliminate_empty_cols(&prob, &SparseAIn::build(&prob)).unwrap();
        assert_eq!(reduced.q.len(), 1, "x1 eliminated");
        let restored = restore_cols(&col, &solve_qp(&reduced, &Settings::<f64>::default()));
        assert_eq!(restored.status, Status::Solved);
        assert!((restored.x[1] - 2.0).abs() < 1e-9, "x1={}", restored.x[1]);
    }

    /// An unbounded empty column (P_jj ≈ 0, q_j ≠ 0) is dual-infeasible.
    #[test]
    fn empty_column_unbounded_is_dual_infeasible() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 0.0]),
            q: vec![0.0, -3.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(1, 2, vec![1.0, 0.0]),
            b_in: vec![1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        assert!(matches!(
            eliminate_empty_cols(&prob, &SparseAIn::build(&prob)),
            Err(Status::DualInfeasible)
        ));
    }

    /// An equality singleton fixes a variable; postsolve recovers primal AND the
    /// pin multiplier, matching a direct solve.
    #[test]
    fn eliminates_fixed_variable_with_dual_recovery() {
        // min ½(x0²+x1²) s.t. x0 = 3, x0 + x1 = 5  →  x=[3,2], y=[-1,-2], obj=6.5.
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0],
            a_eq: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 1.0, 1.0]),
            b_eq: vec![3.0, 5.0],
            a_in: DenseMatrix::zeros(0, 2),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, fix) = eliminate_fixed_vars(&prob).unwrap();
        assert_eq!(reduced.q.len(), 1, "x0 eliminated");
        let restored = restore_fixed_vars(
            &prob,
            &fix,
            &solve_qp(&reduced, &Settings::<f64>::default()),
        );
        assert_eq!(restored.status, Status::Solved);
        assert!((restored.x[0] - 3.0).abs() < 1e-7, "x0={}", restored.x[0]);
        assert!((restored.x[1] - 2.0).abs() < 1e-7, "x1={}", restored.x[1]);
        assert!((restored.y[0] + 1.0).abs() < 1e-6, "y0={}", restored.y[0]);
        assert!((restored.y[1] + 2.0).abs() < 1e-6, "y1={}", restored.y[1]);
    }

    /// Inconsistent fixers x0 = 2 and x0 = 3 are primal-infeasible.
    #[test]
    fn inconsistent_fixers_infeasible() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(1, 1, vec![1.0]),
            q: vec![0.0],
            a_eq: DenseMatrix::from_row_major(2, 1, vec![1.0, 1.0]),
            b_eq: vec![2.0, 3.0],
            a_in: DenseMatrix::zeros(0, 1),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        assert!(matches!(
            eliminate_fixed_vars(&prob),
            Err(Status::PrimalInfeasible)
        ));
    }

    /// An equality doubleton substitutes a variable out; the recovered primal, dual, and
    /// objective match a direct solve. (min ½‖x‖² s.t. x0 + 2x1 = 6 → x=[1.2,2.4], y=−1.2.)
    #[test]
    fn eliminates_doubleton_with_dual_recovery() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0],
            a_eq: DenseMatrix::from_row_major(1, 2, vec![1.0, 2.0]),
            b_eq: vec![6.0],
            a_in: DenseMatrix::zeros(0, 2),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let direct = solve_qp(&prob, &Settings::<f64>::default());
        let (reduced, red) = eliminate_doubleton_eqs(&prob, 10.0).unwrap();
        assert_eq!(reduced.q.len(), 1, "one variable eliminated");
        assert_eq!(reduced.b_eq.len(), 0, "pivot row removed");
        let restored = restore_doubleton_eqs(
            &prob,
            &red,
            &solve_qp(&reduced, &Settings::<f64>::default()),
        );
        assert_eq!(restored.status, Status::Solved);
        assert!((restored.x[0] - 1.2).abs() < 1e-6, "x0={}", restored.x[0]);
        assert!((restored.x[1] - 2.4).abs() < 1e-6, "x1={}", restored.x[1]);
        assert!((restored.y[0] + 1.2).abs() < 1e-6, "y={}", restored.y[0]);
        assert!((restored.x[0] - direct.x[0]).abs() < 1e-6);
    }

    /// A doubleton with off-diagonal `P` coupling and an inequality: the substitution must
    /// fold the quadratic coupling into `P`'s surviving column. Full primal/dual/slack/obj
    /// recovery matches a direct solve.
    #[test]
    fn doubleton_with_coupling_and_inequality() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![2.0, 0.5, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 2.0]),
            q: vec![-1.0, -2.0, -3.0],
            a_eq: DenseMatrix::from_row_major(1, 3, vec![1.0, 1.0, 0.0]),
            b_eq: vec![1.0],
            a_in: DenseMatrix::from_row_major(1, 3, vec![0.0, 0.0, 1.0]),
            b_in: vec![0.5],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let direct = solve_qp(&prob, &Settings::<f64>::default());
        let (reduced, red) = eliminate_doubleton_eqs(&prob, 10.0).unwrap();
        assert_eq!(reduced.q.len(), 2);
        let restored = restore_doubleton_eqs(
            &prob,
            &red,
            &solve_qp(&reduced, &Settings::<f64>::default()),
        );
        assert_eq!(restored.status, Status::Solved);
        for k in 0..3 {
            assert!(
                (restored.x[k] - direct.x[k]).abs() < 1e-6,
                "x{k}: {} vs {}",
                restored.x[k],
                direct.x[k]
            );
        }
        assert!((restored.y[0] - direct.y[0]).abs() < 1e-6, "y mismatch");
        assert!(
            (restored.obj_val - direct.obj_val).abs() < 1e-6,
            "obj mismatch"
        );
    }

    /// Non-doubleton equality rows (one or three nonzeros) are left untouched.
    #[test]
    fn leaves_non_doubleton_rows() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            q: vec![0.0; 3],
            a_eq: DenseMatrix::from_row_major(1, 3, vec![1.0, 1.0, 1.0]), // 3 nonzeros
            b_eq: vec![1.0],
            a_in: DenseMatrix::zeros(0, 3),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, red) = eliminate_doubleton_eqs(&prob, 10.0).unwrap();
        assert_eq!(reduced.q.len(), 3, "nothing eliminated");
        assert!(red.elims.is_empty());
    }

    /// Fill-budget gate: an elimination whose projected fill (the survivor
    /// coupling in inequality rows and the Hessian row) would push the total
    /// nonzeros past the budget is skipped; a generous budget keeps it.
    #[test]
    fn doubleton_fill_budget_gate_skips_dense_coupling() {
        // x0 = x1 (doubleton), with x1 coupled to a dense inequality row and
        // a dense P row — the substitution would densify them.
        let n = 6usize;
        let mut a_eq = DenseMatrix::zeros(1, n);
        a_eq.set(0, 0, 1.0);
        a_eq.set(0, 1, -1.0);
        let mut a_in = DenseMatrix::zeros(2, n);
        a_in.set(0, 1, 1.0);
        a_in.set(1, 1, 1.0);
        for j in 0..n {
            a_in.set(0, j, 1.0); // dense row coupling x1 to everything
        }
        let mut p = DenseMatrix::zeros(n, n);
        for j in 0..n {
            p.set(0, j, 1.0); // dense P row on the substituted variable
            p.set(j, 0, 1.0);
        }
        let prob = QpProblem {
            p,
            q: vec![1.0; n],
            a_eq,
            b_eq: vec![0.0],
            a_in,
            b_in: vec![10.0, 10.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        // Tight budget: the fill exceeds 1.0 x the current nnz -> skipped.
        let (_, red_tight) = eliminate_doubleton_eqs(&prob, 1.0).unwrap();
        assert_eq!(
            red_tight.elims.len(),
            0,
            "dense-coupling elimination must be gated"
        );
        // Generous budget: the elimination goes through (load-bearing case).
        let (_, red_loose) = eliminate_doubleton_eqs(&prob, 100.0).unwrap();
        assert_eq!(
            red_loose.elims.len(),
            1,
            "generous budget keeps the elimination"
        );
    }

    /// A column coupled through P off-diagonals is NOT empty (correctness guard).
    #[test]
    fn coupled_column_is_not_eliminated() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(2, 2, vec![1.0, 1.0, 1.0, 2.0]),
            q: vec![0.0, 0.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in: DenseMatrix::from_row_major(1, 2, vec![1.0, 0.0]),
            b_in: vec![1.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, col) = eliminate_empty_cols(&prob, &SparseAIn::build(&prob)).unwrap();
        assert_eq!(reduced.q.len(), 2, "neither column is empty");
        assert!(col.fixed.is_empty());
    }

    /// Regression for a bug where `merge_equality_rows` eliminated a variable
    /// with a nonzero *diagonal* P entry: the candidacy gate (`col_is_diag_p`)
    /// only rules out OFF-diagonal P coupling, so a genuine quadratic term
    /// `P[j,j] != 0` on the merged-out variable passed through unfolded, and
    /// its contribution to the KKT system was silently dropped, producing a
    /// wrong optimum. The fix must exclude such columns from candidacy.
    ///
    /// P = diag(1,1,1); rows `x0+x1=3` and `2x1+x2=8` share x1 exclusively.
    /// The true optimum (verified by direct KKT solve) is
    /// x = (-1/6, 19/6, 5/3) = (-0.16666..., 3.16666..., 1.66666...).
    #[test]
    fn merge_equality_rows_skips_nonzero_diagonal_p_column() {
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            q: vec![0.0, 0.0, 0.0],
            a_eq: DenseMatrix::from_row_major(
                2,
                3,
                vec![
                    1.0, 1.0, 0.0, //  x0 + x1      = 3
                    0.0, 2.0, 1.0, //       2x1 + x2 = 8
                ],
            ),
            b_eq: vec![3.0, 8.0],
            a_in: DenseMatrix::zeros(0, 3),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };

        // The gate must reject column 1 (P[1,1] = 1 != 0): no merge fires.
        let (reduced, red) = merge_equality_rows(&prob, 8).expect("merge should not error");
        assert!(
            red.merges.is_empty(),
            "column with nonzero diagonal P must not be merged out"
        );
        assert_eq!(reduced.q.len(), 3, "problem must be returned unreduced");

        // Solving through the (no-op) merge pipeline must still recover the
        // true optimum, not the wrong answer the bug used to produce
        // (x = (-0.8, 3.8, 0.4), which fails stationarity at index 1).
        let reduced_sol = solve_qp(&reduced, &Settings::<f64>::default());
        let restored = restore_merged_rows(&prob, &red, &reduced_sol);
        assert_eq!(restored.status, Status::Solved);
        assert!(
            (restored.x[0] - (-1.0 / 6.0)).abs() < 1e-6,
            "x0 = {}",
            restored.x[0]
        );
        assert!(
            (restored.x[1] - (19.0 / 6.0)).abs() < 1e-6,
            "x1 = {}",
            restored.x[1]
        );
        assert!(
            (restored.x[2] - (5.0 / 3.0)).abs() < 1e-6,
            "x2 = {}",
            restored.x[2]
        );

        // Original-space stationarity: (Px + q + A_eq^T y)[1] must be ~0. The
        // pre-fix bug produced 3.8 here (a 3.8-magnitude violation).
        let px = prob.p.matvec(&restored.x);
        let aty = prob.a_eq.matvec_t(&restored.y);
        let stat1 = px[1] + prob.q[1] + aty[1];
        assert!(
            stat1.abs() < 1e-6,
            "stationarity residual at x1 = {stat1} (should be ~0)"
        );
    }

    /// The aux-var fold keeps the eliminated variable's LINEAR objective term.
    /// Regression: linked_qp(2,47) — min ½(x0²+x1²+x2²+x3²) + q·x s.t. x0=x1,
    /// x2=x3, with every q_j ≠ 0 — eliminated x0/x2 (first var of each pair)
    /// and dropped q0/q2 entirely, solving a q-mutilated objective: returned
    /// -0.25943 against the true optimum -0.8990479320, identical
    /// with presolve on or off (the pass runs inside the cone-path solve).
    #[test]
    fn aux_fold_keeps_linear_term_of_eliminated_variable() {
        let q = vec![
            -0.8809576843939488,
            -0.5024877448229195,
            0.4108945748657331,
            0.8861291228657118,
        ];
        let mut p = DenseMatrix::zeros(4, 4);
        for j in 0..4 {
            p.set(j, j, 1.0);
        }
        let mut a_eq = DenseMatrix::zeros(2, 4);
        a_eq.set(0, 0, 1.0);
        a_eq.set(0, 1, -1.0);
        a_eq.set(1, 2, 1.0);
        a_eq.set(1, 3, -1.0);
        let prob = QpProblem {
            p,
            q: q.clone(),
            a_eq,
            b_eq: vec![0.0, 0.0],
            a_in: DenseMatrix::zeros(0, 4),
            b_in: vec![],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let (reduced, red) = eliminate_auxiliary_vars(&prob).expect("elim");
        assert_eq!(red.elims.len(), 2, "both pair members eliminated");
        let sol = solve_qp(&reduced, &Settings::<f64>::default());
        let restored = restore_auxiliary_vars(&red, &prob, &sol);
        let obj: f64 = {
            let x = &restored.x;
            let mut o = 0.0;
            for i in 0..4 {
                o += 0.5 * x[i] * x[i] + q[i] * x[i];
            }
            o
        };
        assert!(
            (obj - (-0.8990479320)).abs() < 1e-8,
            "recovered objective {obj} != true optimum -0.8990479320"
        );
    }
}

#[cfg(test)]
mod negated_pair_tests {
    use super::*;
    use iconic_core::Settings;
    use iconic_ipm::solve_qp;

    fn qp(a_in: DenseMatrix<f64>, b_in: Vec<f64>, q: Vec<f64>) -> QpProblem<f64> {
        QpProblem {
            p: DenseMatrix::zeros(q.len(), q.len()),
            q: q.clone(),
            a_eq: DenseMatrix::zeros(0, q.len()),
            b_eq: vec![],
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        }
    }

    /// A tight negated pair (x1+x2 ≤ 3 and x1+x2 ≥ 3) folds into one equality
    /// row — net −2 rows — and the restored solution matches a direct solve.
    #[test]
    fn tight_pair_folds_to_equality() {
        // min x0 + x1 s.t. x0 + x1 ≤ 3, −x0 − x1 ≤ −3, x0 ≥ 0, x1 ≥ 0.
        // Optimum: x0 + x1 = 3 exactly; the two pair rows bind together.
        let prob = qp(
            DenseMatrix::from_row_major(
                4,
                2,
                vec![
                    1.0, 1.0, // x0 + x1 ≤ 3
                    -1.0, -1.0, // −x0 − x1 ≤ −3
                    -1.0, 0.0, // x0 ≥ 0
                    0.0, -1.0, // x1 ≥ 0
                ],
            ),
            vec![3.0, -3.0, 0.0, 0.0],
            vec![1.0, 1.0],
        );
        let (reduced, red) = fold_negated_pairs(&prob, &SparseAIn::build(&prob)).expect("fold");
        // Pair folded: one new equality row, the two pair rows gone.
        assert_eq!(reduced.b_eq.len(), 1, "folded equality expected");
        assert_eq!(reduced.b_in.len(), 2, "only the two bounds remain");
        assert_eq!(red.pairs.len(), 1);
        assert_eq!(red.pairs[0], (0usize, 1usize));

        let sol = solve_qp(&reduced, &Settings::<f64>::default());
        assert_eq!(sol.status, Status::Solved);
        let restored = restore_negated_pairs(&red, &sol);
        // Original-space solution: x0 + x1 = 3, x ≥ 0.
        assert!((restored.x[0] + restored.x[1] - 3.0).abs() < 1e-6);
        // Duals: the equality multiplier splits into the two nonneg row duals.
        let y = sol.y[0];
        assert!((restored.z[0] - y.max(0.0)).abs() < 1e-9);
        assert!((restored.z[1] - (-y).max(0.0)).abs() < 1e-9);
        assert!(restored.z[0] >= 0.0 && restored.z[1] >= 0.0);
        // Slacks of the folded rows ~ 0 (they bind together).
        assert!(restored.s[0].abs() < 1e-6 && restored.s[1].abs() < 1e-6);
        // Compare against a direct solve of the original: the optimum is a face
        // (x0 + x1 = 3), so the objective is the meaningful invariant.
        let direct = solve_qp(&prob, &Settings::<f64>::default());
        assert!((direct.obj_val - restored.obj_val).abs() < 1e-9);
        assert!((direct.obj_val - 3.0).abs() < 1e-6);
    }

    /// The folded equality feeds fixed-variable elimination: a singleton pair
    /// (x ≤ 1, x ≥ 1) becomes the pinned variable x = 1.
    #[test]
    fn singleton_pair_feeds_fixed_vars() {
        let prob = qp(
            DenseMatrix::from_row_major(2, 1, vec![1.0, -1.0]),
            vec![1.0, -1.0],
            vec![-2.0], // min −2x
        );
        let (reduced, red) = fold_negated_pairs(&prob, &SparseAIn::build(&prob)).expect("fold");
        assert_eq!(reduced.b_eq.len(), 1);
        assert_eq!(reduced.b_in.len(), 0);
        let sol = solve_qp(&reduced, &Settings::<f64>::default());
        let restored = restore_negated_pairs(&red, &sol);
        assert!((restored.x[0] - 1.0).abs() < 1e-9, "x pinned to 1");
        // min −2x at x=1 → obj −2.
        assert!((restored.obj_val + 2.0).abs() < 1e-9);
        assert!(restored.z[0] >= 0.0 && restored.z[1] >= 0.0);
    }

    /// A looser sibling of a folded pair is dropped with dual 0 and its slack
    /// restored from the recovered primal.
    #[test]
    fn looser_sibling_drops_with_zero_dual() {
        // x ≤ 1, x ≥ 1 (pair → x = 1) and x ≤ 2 (looser upper sibling).
        let prob = qp(
            DenseMatrix::from_row_major(3, 1, vec![1.0, -1.0, 1.0]),
            vec![1.0, -1.0, 2.0],
            vec![0.0],
        );
        let (reduced, red) = fold_negated_pairs(&prob, &SparseAIn::build(&prob)).expect("fold");
        assert_eq!(red.pairs.len(), 1);
        assert_eq!(red.dropped_rest, vec![2usize]);
        assert_eq!(reduced.b_in.len(), 0, "pair + sibling all gone");
        let sol = solve_qp(&reduced, &Settings::<f64>::default());
        let restored = restore_negated_pairs(&red, &sol);
        assert!((restored.x[0] - 1.0).abs() < 1e-9);
        assert_eq!(restored.z[2], 0.0, "sibling dual is 0");
        assert!((restored.s[2] - 1.0).abs() < 1e-9, "sibling slack = 2 − x");
    }

    /// A contradictory pair is PrimalInfeasible.
    #[test]
    fn contradictory_pair_is_infeasible() {
        let prob = qp(
            DenseMatrix::from_row_major(2, 1, vec![1.0, -1.0]),
            vec![1.0, -2.0], // x ≤ 1 and x ≥ 2
            vec![0.0],
        );
        assert!(matches!(
            fold_negated_pairs(&prob, &SparseAIn::build(&prob)),
            Err(Status::PrimalInfeasible)
        ));
    }

    /// A non-tight ranged pair (x ≤ 3, x ≥ 1) is NOT folded — reduce_rows keeps
    /// it as a two-sided range.
    #[test]
    fn loose_range_is_not_folded() {
        let prob = qp(
            DenseMatrix::from_row_major(2, 1, vec![1.0, -1.0]),
            vec![3.0, -1.0],
            vec![0.0],
        );
        let (reduced, red) = fold_negated_pairs(&prob, &SparseAIn::build(&prob)).expect("fold");
        assert!(red.pairs.is_empty());
        assert_eq!(reduced.b_eq.len(), 0);
        assert_eq!(reduced.b_in.len(), 2);
    }

    /// The sparse views must reproduce the dense path bit-identically: the
    /// gate fires only on large sparse problems, so the suite's small tests
    /// never exercise the sparse iteration. This test forces both paths on a
    /// problem the gate DOES fire for (>= 1M entries, < 50% density) and
    /// compares every pass's output problem and reduction record.
    #[test]
    fn sparse_views_match_dense_path() {
        let n = 1100usize;
        let mi = 1100usize;
        let _zero = 0.0;
        let mut a_in = DenseMatrix::zeros(mi, n);
        let mut b_in = vec![0.0; mi];
        // Two nonzeros per row (distinct columns), random-ish values, feasible
        // box: 0 <= x <= 1 with rows x_a - x_b <= 1.
        let mut next = iconic_core::rng::Lcg::new(12345);
        for r in 0..mi {
            let (j0, j1) = ((r * 7) % n, (r * 13 + 5) % n);
            if j0 != j1 {
                let v0 = 1.0 + next.unit();
                let v1 = -(0.5 + next.unit());
                a_in.set(r, j0, v0);
                a_in.set(r, j1, v1);
                b_in[r] = 1.0;
            }
        }
        // Near-zero rows straddling the null threshold (scale = max(1, row
        // max), null iff every entry <= 1e-10): the sparse path must treat
        // them exactly as the dense path does.
        for (k, v) in [(1e-12, 1e-13), (5e-10, -1e-10), (1e-8, -2e-9)] {
            let r = mi - 1 - k as usize % 10;
            a_in.set(r, (r * 3) % n, v);
            a_in.set(r, (r * 3 + 1) % n, v * 0.5);
            b_in[r] = 0.0;
        }
        let prob = QpProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![0.0; n],
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sp = SparseAIn::build(&prob);
        assert!(sp.sparse, "the gate should fire at >= 1M entries");
        let dense = SparseAIn::dense();

        let (r_sp, red_sp) = fold_negated_pairs(&prob, &sp).unwrap();
        let (r_den, red_den) = fold_negated_pairs(&prob, &dense).unwrap();
        assert_eq!(r_sp.a_in.data, r_den.a_in.data);
        assert_eq!(red_sp.pairs, red_den.pairs);
        assert_eq!(red_sp.dropped_rest, red_den.dropped_rest);
        assert_eq!(red_sp.kept_in, red_den.kept_in);

        let (c_sp, colred_sp) = eliminate_empty_cols(&r_sp, &sp).unwrap();
        let (c_den, colred_den) = eliminate_empty_cols(&r_den, &dense).unwrap();
        assert_eq!(c_sp.a_in.data, c_den.a_in.data);
        assert_eq!(colred_sp.kept, colred_den.kept);
        assert_eq!(colred_sp.fixed, colred_den.fixed);

        let (rr_sp, rowred_sp) = reduce_rows(&c_sp, &sp).unwrap();
        let (rr_den, rowred_den) = reduce_rows(&c_den, &dense).unwrap();
        assert_eq!(rr_sp.a_in.data, rr_den.a_in.data);
        assert_eq!(rowred_sp.kept_in, rowred_den.kept_in);
        assert_eq!(rowred_sp.dropped_null, rowred_den.dropped_null);
        assert_eq!(rowred_sp.dropped_resid, rowred_den.dropped_resid);

        let (rd_sp, redun_sp) = remove_redundant_ineqs(&rr_sp, &sp).unwrap();
        let (rd_den, redun_den) = remove_redundant_ineqs(&rr_den, &dense).unwrap();
        assert_eq!(rd_sp.a_in.data, rd_den.a_in.data);
        assert_eq!(redun_sp.dropped, redun_den.dropped);

        // The CSR-emission passes must produce CSRs consistent with their
        // rebuilt dense A_in (the solver consumes a_in_csr when present —
        // a mismatch would silently change the matvecs). A no-op pass keeps
        // the input's CSR (None here — the test problem carries none), so the
        // consistency check is gated on the CSR being present. The redundant
        // pass is included: its no-op path now also emits the CSR from the
        // views (the chain's keep-alive), and its changed path rebuilds it.
        for (p, label) in [
            (&r_sp, "fold"),
            (&c_sp, "empty-cols"),
            (&rr_sp, "reduce_rows"),
            (&rd_sp, "redundant"),
        ] {
            if let Some(csr) = &p.a_in_csr {
                assert_eq!(csr.n, p.a_in.nrows, "{label}: csr row count");
                assert_eq!(csr.m, p.a_in.ncols, "{label}: csr col count");
                for r in 0..p.a_in.nrows {
                    for p_idx in csr.colptr[r]..csr.colptr[r + 1] {
                        let j = csr.rowval[p_idx];
                        assert_eq!(
                            csr.nzval[p_idx],
                            p.a_in.get(r, j),
                            "{label}: csr value mismatch at ({r},{j})"
                        );
                    }
                }
                // The CSR must carry every nonzero the dense matrix has.
                let nnz_dense = p.a_in.data.iter().filter(|&&v| v != 0.0).count();
                assert_eq!(csr.rowval.len(), nnz_dense, "{label}: csr nnz count");
            }
        }
    }

    /// A pass that actually rebuilds A_in must emit a CSR for the reduced
    /// problem on the sparse path (the no-op path keeps the input's).
    #[test]
    fn changing_pass_emits_a_in_csr() {
        // A dominated row pair (same direction, looser sibling) forces
        // reduce_rows to rebuild the problem; the sparse path must emit the
        // reduced problem's CSR in the same pass. Same 2-nonzero-per-row
        // shape and gate-crossing size as the equivalence test above.
        let n = 1100usize;
        let mi = 1100usize;
        let mut a_in = DenseMatrix::zeros(mi, n);
        let mut b_in = vec![0.0; mi];
        for r in 0..mi - 1 {
            let (j0, j1) = ((r * 7) % n, (r * 13 + 5) % n);
            if j0 != j1 {
                a_in.set(r, j0, 1.0 + (r as f64) * 1e-6);
                a_in.set(r, j1, -(0.5 + (r as f64) * 1e-6));
                b_in[r] = 1.0;
            }
        }
        // The dominated pair: row mi-1 carries row 0's exact nonzeros
        // (pivot 0, same unit direction) with a looser bound (10 > 1) — the
        // looser sibling is dropped and the problem rebuilt.
        a_in.set(mi - 1, 0, a_in.get(0, 0));
        a_in.set(mi - 1, 5, a_in.get(0, 5));
        b_in[mi - 1] = 10.0;
        let prob = QpProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![0.0; n],
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in,
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sp = SparseAIn::build(&prob);
        assert!(sp.sparse, "the gate should fire at >= 1M cells");
        let (reduced, rowred) = reduce_rows(&prob, &sp).unwrap();
        assert!(!rowred.dropped_resid.is_empty(), "looser sibling dropped");
        let csr = reduced.a_in_csr.as_ref().expect("rebuilt pass emits CSR");
        assert_eq!(csr.n, reduced.a_in.nrows);
        assert_eq!(csr.m, reduced.a_in.ncols);
        for r in 0..reduced.a_in.nrows {
            for p_idx in csr.colptr[r]..csr.colptr[r + 1] {
                let j = csr.rowval[p_idx];
                assert_eq!(csr.nzval[p_idx], reduced.a_in.get(r, j));
            }
        }
    }
}
