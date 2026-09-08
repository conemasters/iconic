//! Sparse LDLᵀ factorization for symmetric matrices in CSC form.
//!
//! This is an up-looking sparse factorization (the classic two-phase approach: a
//! symbolic phase builds the elimination tree and column counts, a numeric phase
//! fills `L` and `D`). The input stores the **upper triangle** of a symmetric
//! matrix in CSC: column `j` holds entries `(i, A_ij)` with `i ≤ j`.
//!
//! Like the dense version, no pivoting is performed — ICONIC's regularized KKT
//! systems are quasidefinite, so `D` simply carries positive and negative pivots.
//! The sparse LDL^T factorization the interior-point engine will move onto; fill-reducing
//! (AMD) ordering layers on top by permuting the matrix before factorization.
//!
//! A sub-tolerance pivot aborts with [`LdlError::ZeroPivot`] and recovery belongs
//! to the caller's escalation ladder (re-regularize globally and refactor). Local
//! pivot boosting was tried here and rejected: measured on the ill-conditioned QP
//! suite it regressed convergence regardless of floor — the escalation path yields
//! a uniformly stable factor where a clamp leaves the factor ill-conditioned.
//!
//! **NaN-diagonal repair:** a NaN on the diagonal poisons the factor — NaN
//! comparisons are always false, so no pivot-tolerance check can catch it and
//! the poisoned pivot would silently propagate through every `L` column and the
//! solve. The factorization replaces it with the shared small nonzero repair
//! value (see [`crate::ldl`]) and continues, counting the repair on the factor
//! (`nan_repairs`).

use crate::ldl::LdlError;
use crate::CscMatrix;
use num_traits::Float;

/// A sparse `L·D·Lᵀ` factorization. `L` is unit lower-triangular, stored by column
/// (strictly-lower entries only); `D` is the diagonal.
#[derive(Clone, Debug)]
pub struct SparseLdl<T> {
    pub(crate) n: usize,
    pub(crate) lp: Vec<usize>,
    pub(crate) li: Vec<usize>,
    pub(crate) lx: Vec<T>,
    #[allow(dead_code)] // diagonal storage; read by the boosted-pivot test
    pub(crate) d: Vec<T>,
    /// Reciprocals of the pivots, built during the numeric factorization (each
    /// pivot is finalized before any column uses it as a divisor, so `d_inv[i]`
    /// is valid whenever `d[i]` would be). Turns the per-L-entry division in
    /// the ancestor walk into a multiply.
    pub(crate) d_inv: Vec<T>,
    /// Number of NaN diagonal entries repaired during the most recent
    /// factorization (a NaN on the diagonal would otherwise poison the factor).
    pub nan_repairs: usize,
}

/// Symbolic analysis: elimination tree `parent` and the column pointers `lp` of `L`.
fn symbolic<T>(a: &CscMatrix<T>) -> (Vec<isize>, Vec<usize>) {
    let n = a.n;
    let mut parent = vec![-1isize; n];
    let mut flag = vec![usize::MAX; n];
    let mut lnz = vec![0usize; n];

    for k in 0..n {
        flag[k] = k;
        for p in a.colptr[k]..a.colptr[k + 1] {
            let i = a.rowval[p];
            if i < k {
                let mut ii = i;
                while flag[ii] != k {
                    if parent[ii] == -1 {
                        parent[ii] = k as isize;
                    }
                    lnz[ii] += 1;
                    flag[ii] = k;
                    ii = parent[ii] as usize;
                }
            }
        }
    }

    let mut lp = vec![0usize; n + 1];
    for k in 0..n {
        lp[k + 1] = lp[k] + lnz[k];
    }
    (parent, lp)
}

/// Reusable symbolic analysis (elimination tree + column pointers of `L`). Built
/// once per sparsity pattern and reused across numeric factorizations whose matrices
/// share that pattern — exactly the interior-point case, where only values change.
#[derive(Clone, Debug)]
pub struct Symbolic {
    pub(crate) parent: Vec<isize>,
    pub(crate) lp: Vec<usize>,
    pub n: usize,
    pub sno: Vec<usize>,
}

/// Analyze the sparsity pattern of `a` (upper triangle in CSC).
pub fn analyze<T>(a: &CscMatrix<T>) -> Symbolic {
    assert_eq!(a.m, a.n, "LDL requires a square matrix");
    let (parent, lp) = symbolic(a);
    let n = a.n;
    // Detect fundamental supernodes.  Columns `f..l` are a supernode if for each
    // `i` in `f..l-1`: parent[i] == i+1 AND the number of nonzeros in column i of
    // L is exactly one more than in column i+1 (the extra nz is the subdiagonal).
    let lnz: Vec<usize> = (0..n).map(|c| lp[c + 1] - lp[c]).collect();
    let mut sno = Vec::new();
    sno.push(0);
    for i in 0..n - 1 {
        if parent[i] != (i + 1) as isize || lnz[i] != lnz[i + 1] + 1 {
            sno.push(i + 1);
        }
    }
    sno.push(n);
    Symbolic { parent, lp, n, sno }
}

/// Reusable workspace for the numeric factorization. Allocate once per solve
/// (the KKT dimension is fixed across IPM iterations) and pass to `factor_with_ws`
/// to avoid four size-`n` allocations per call.
pub struct LdlWorkspace<T> {
    pub(crate) y: Vec<T>,
    pub(crate) pattern: Vec<usize>,
    pub(crate) flag: Vec<usize>,
    pub(crate) count: Vec<usize>,
    // Supernodal panel buffers (used only by `factor_supernodal_with_ws`): the
    // row-position map is fixed-size; the panels grow on demand and are refilled
    // (with zero-fill where the factorization relies on untouched entries) every
    // panel, so repeated factor calls allocate once instead of per panel.
    pub(crate) row_pos: Vec<usize>,
    pub(crate) panel_rows: Vec<usize>,
    pub(crate) panel_data: Vec<T>,
    pub(crate) diag_block: Vec<T>,
    pub(crate) l_sn: Vec<T>,
    pub(crate) d_sn: Vec<T>,
}

impl<T: Float> LdlWorkspace<T> {
    /// Create a workspace for a KKT of dimension `n`.
    pub fn new(n: usize) -> Self {
        LdlWorkspace {
            y: vec![T::zero(); n],
            pattern: vec![0usize; n],
            flag: vec![usize::MAX; n],
            count: vec![0usize; n],
            row_pos: vec![usize::MAX; n],
            panel_rows: Vec::new(),
            panel_data: Vec::new(),
            diag_block: Vec::new(),
            l_sn: Vec::new(),
            d_sn: Vec::new(),
        }
    }

    /// Reset the flag and count arrays for a new factorization (the other arrays are
    /// overwritten element-by-element during the factorization and don't need
    /// a separate clear).
    pub fn clear(&mut self) {
        self.flag.fill(usize::MAX);
        self.count.fill(0);
    }
}

/// Numeric factorization reusing a [`Symbolic`] analysis and a pre-allocated
/// [`LdlWorkspace`]. `a` must have the same sparsity pattern and dimension the
/// analysis and workspace were built for.
///
/// A sub-tolerance pivot returns [`LdlError::ZeroPivot`] (recovery belongs to the
/// caller's escalation ladder). A NaN diagonal is repaired unconditionally
/// (replaced with zero) before that check — the factor proceeds, and the repair is
/// counted on the returned factor (`nan_repairs`).
pub fn factor_with_ws<T: Float>(
    a: &CscMatrix<T>,
    sym: &Symbolic,
    pivot_tol: T,
    ws: &mut LdlWorkspace<T>,
) -> Result<SparseLdl<T>, LdlError> {
    let mut out = SparseLdl {
        n: 0,
        lp: Vec::new(),
        li: Vec::new(),
        lx: Vec::new(),
        d: Vec::new(),
        d_inv: Vec::new(),
        nan_repairs: 0,
    };
    factor_with_ws_into(a, sym, pivot_tol, ws, &mut out)?;
    Ok(out)
}

/// Numeric factorization into a caller-owned [`SparseLdl`], reusing its
/// buffers (the caller keeps the factor across iterations and refactors into
/// it — the first call allocates, subsequent calls reuse capacity). The
/// caller must reuse the factor with the same symbolic structure; every L/D
/// entry is rewritten by each factorization, so stale data cannot leak, and
/// `lp` is only re-copied when its length changes.
pub fn factor_with_ws_into<T: Float>(
    a: &CscMatrix<T>,
    sym: &Symbolic,
    pivot_tol: T,
    ws: &mut LdlWorkspace<T>,
    out: &mut SparseLdl<T>,
) -> Result<(), LdlError> {
    let n = sym.n;
    let parent = &sym.parent;
    let lp = &sym.lp;
    if out.lp.len() != lp.len() {
        out.lp = lp.clone();
    }
    if out.li.len() < lp[n] {
        out.li.resize(lp[n], 0);
    }
    if out.lx.len() < lp[n] {
        out.lx.resize(lp[n], T::zero());
    }
    if out.d.len() < n {
        out.d.resize(n, T::zero());
    }
    if out.d_inv.len() < n {
        out.d_inv.resize(n, T::zero());
    }
    out.n = n;
    out.nan_repairs = 0;
    let li = &mut out.li;
    let lx = &mut out.lx;
    let d = &mut out.d;
    let d_inv = &mut out.d_inv;
    let mut nan_repairs = 0usize;

    let y = &mut ws.y;
    let pattern = &mut ws.pattern;
    let flag = &mut ws.flag;
    let count = &mut ws.count;
    // flag is cleared by the caller via ws.clear() between factorizations,
    // since the numeric pass reads stale markers left by the previous call.
    // Individual entries are set to k and don't need resetting — the
    // algorithm relies on comparing flag[ii] != k.
    for k in 0..n {
        // Scatter column k of A (upper part) into Y and build the column pattern.
        let mut top = n;
        flag[k] = k;
        y[k] = T::zero();
        for p in a.colptr[k]..a.colptr[k + 1] {
            let i = a.rowval[p];
            if i > k {
                continue;
            }
            y[i] = y[i] + a.nzval[p];
            let mut len = 0usize;
            let mut ii = i;
            while flag[ii] != k {
                pattern[len] = ii;
                len += 1;
                flag[ii] = k;
                ii = parent[ii] as usize;
            }
            while len > 0 {
                len -= 1;
                top -= 1;
                pattern[top] = pattern[len];
            }
        }

        // D[k] starts at the (scattered) diagonal; clear Y[k].
        d[k] = y[k];
        y[k] = T::zero();

        // Walk the column pattern in topological order, applying each ancestor's
        // column of L and emitting L[k, i].
        for s in top..n {
            let i = pattern[s];
            let yi = y[i];
            y[i] = T::zero();
            for p in lp[i]..lp[i] + count[i] {
                let row = li[p];
                y[row] = y[row] - lx[p] * yi;
            }
            let lki = yi * d_inv[i]; // d[i] is final: ancestors precede k
            d[k] = d[k] - lki * yi;
            let p = lp[i] + count[i];
            li[p] = k;
            lx[p] = lki;
            count[i] += 1;
        }

        // Pivot handling: NaN-diagonal repair, then the tolerance check.
        if d[k].is_nan() {
            // A NaN on the diagonal poisons the factor: NaN comparisons are
            // always false, so the tolerance check cannot catch it and the
            // poisoned pivot would silently propagate through every L column
            // and the solve. Repair to the shared nonzero value (keeps every
            // subsequent division finite) — used as-is (not re-checked against
            // the tolerance), counted on the factor (`nan_repairs`).
            d[k] = crate::ldl::nan_repair_value::<T>();
            nan_repairs += 1;
        } else if d[k].abs() <= pivot_tol {
            // Sub-tolerance pivot: abort and let the caller's escalation ladder
            // respond (re-regularize globally and refactor — measured to beat
            // any local clamp on ICONIC's ill-conditioned dynamics).
            return Err(LdlError::ZeroPivot(k));
        }
        // Pivot finalized (post repair/boost) — cache its reciprocal.
        d_inv[k] = d[k].recip();
    }

    out.nan_repairs = nan_repairs;
    Ok(())
}

/// Numeric factorization reusing a [`Symbolic`] analysis. Creates a temporary
/// workspace internally; prefer [`factor_with_ws`] when factoring repeatedly
/// (e.g. the IPM loop) to avoid per-iteration allocations.
pub fn factor_with<T: Float>(
    a: &CscMatrix<T>,
    sym: &Symbolic,
    pivot_tol: T,
) -> Result<SparseLdl<T>, LdlError> {
    let mut ws = LdlWorkspace::new(sym.n);
    factor_with_ws(a, sym, pivot_tol, &mut ws)
}

/// One-shot symbolic + numeric factorization of `a` (upper triangle in CSC).
///
/// `pivot_tol` guards against a (near-)zero pivot, yielding [`LdlError::ZeroPivot`].
pub fn sparse_ldl_factor<T: Float>(
    a: &CscMatrix<T>,
    pivot_tol: T,
) -> Result<SparseLdl<T>, LdlError> {
    let sym = analyze(a);
    factor_with(a, &sym, pivot_tol)
}

impl<T: Float> SparseLdl<T> {
    /// An empty factor — the starting point for the buffer-reuse path
    /// ([`factor_with_ws_into`] / [`factor_supernodal_with_ws_into`] resize on
    /// demand, so the first factorization allocates).
    pub fn empty() -> Self {
        SparseLdl {
            n: 0,
            lp: Vec::new(),
            li: Vec::new(),
            lx: Vec::new(),
            d: Vec::new(),
            d_inv: Vec::new(),
            nan_repairs: 0,
        }
    }

    /// Number of stored nonzeros in `L` (strictly lower triangle).
    pub const fn l_nnz(&self) -> usize {
        self.lx.len()
    }

    /// Solve `K u = rhs` for `u`, where `K = L·D·Lᵀ`.
    pub fn solve(&self, rhs: &[T]) -> Vec<T> {
        debug_assert_eq!(rhs.len(), self.n);
        let mut x = rhs.to_vec();
        self.solve_into(rhs, &mut x);
        x
    }

    /// Solve `LDLᵀ x = rhs` into a pre-allocated output buffer (the caller reuses
    /// `x` across solves, avoiding the per-solve allocation).
    pub fn solve_into(&self, rhs: &[T], x: &mut [T]) {
        debug_assert_eq!(rhs.len(), self.n);
        debug_assert_eq!(x.len(), self.n);
        let n = self.n;
        x.copy_from_slice(rhs);

        // Forward: L w = rhs (L unit lower, stored by column).
        for i in 0..n {
            for p in self.lp[i]..self.lp[i + 1] {
                let row = self.li[p];
                x[row] = x[row] - self.lx[p] * x[i];
            }
        }
        // Diagonal: D v = w.
        for i in 0..n {
            x[i] = x[i] * self.d_inv[i];
        }
        // Backward: Lᵀ u = v.
        for i in (0..n).rev() {
            for p in self.lp[i]..self.lp[i + 1] {
                x[i] = x[i] - self.lx[p] * x[self.li[p]];
            }
        }
    }
}

/// Upper-triangle CSC from a dense row-major symmetric matrix (test helper,
/// shared by the sparse/supernodal/ordering test modules).
#[cfg(test)]
pub(crate) fn upper_csc(n: usize, dense: &[f64]) -> CscMatrix<f64> {
    let mut colptr = vec![0usize; n + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for j in 0..n {
        for i in 0..=j {
            let v = dense[i * n + j];
            if v != 0.0 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr[j + 1] = rowval.len();
    }
    CscMatrix {
        m: n,
        n,
        colptr,
        rowval,
        nzval,
    }
}

/// Dense row-major symmetric matvec + max-|Ax−b| residual (test helper,
/// shared by the sparse/supernodal test modules).
#[cfg(test)]
pub(crate) fn max_resid(n: usize, dense: &[f64], x: &[f64], b: &[f64]) -> f64 {
    (0..n)
        .map(|i| {
            let kx: f64 = (0..n).map(|j| dense[i * n + j] * x[j]).sum();
            (kx - b[i]).abs()
        })
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_with_reuses_symbolic_for_same_pattern() {
        // Same sparsity, different values -> one analysis, two numeric factorizations.
        let d1 = [5.0, 0.0, 1.0, 0.0, 6.0, 2.0, 1.0, 2.0, 9.0];
        let d2 = [8.0, 0.0, 3.0, 0.0, 4.0, 1.0, 3.0, 1.0, 7.0];
        let a1 = upper_csc(3, &d1);
        let a2 = upper_csc(3, &d2);
        let sym = analyze(&a1);

        let reuse = factor_with(&a2, &sym, 1e-14).unwrap();
        let fresh = sparse_ldl_factor(&a2, 1e-14).unwrap();
        let b = [1.0, 2.0, 3.0];
        let xr = reuse.solve(&b);
        let xf = fresh.solve(&b);
        for i in 0..3 {
            assert!((xr[i] - xf[i]).abs() < 1e-12);
        }
        assert!(max_resid(3, &d2, &xr, &b) < 1e-10);
    }

    #[test]
    fn solves_spd_2x2() {
        let dense = [4.0, 2.0, 2.0, 3.0];
        let a = upper_csc(2, &dense);
        let f = sparse_ldl_factor(&a, 1e-14).unwrap();
        let b = [2.0, 1.0];
        let x = f.solve(&b);
        assert!(max_resid(2, &dense, &x, &b) < 1e-12);
    }

    #[test]
    fn solves_quasidefinite() {
        let dense = [2.0, 1.0, 1.0, -3.0];
        let a = upper_csc(2, &dense);
        let f = sparse_ldl_factor(&a, 1e-14).unwrap();
        let b = [1.0, -1.0];
        let x = f.solve(&b);
        assert!(max_resid(2, &dense, &x, &b) < 1e-12);
    }

    #[test]
    fn solves_sparse_arrowhead() {
        // 4x4 symmetric "arrowhead": dense last row/col, diagonal elsewhere.
        // Quasidefinite-friendly: large positive diagonal.
        let dense = [
            5.0, 0.0, 0.0, 1.0, //
            0.0, 6.0, 0.0, 2.0, //
            0.0, 0.0, 7.0, 3.0, //
            1.0, 2.0, 3.0, 9.0,
        ];
        let a = upper_csc(4, &dense);
        let f = sparse_ldl_factor(&a, 1e-14).unwrap();
        // L should be sparse: only column-3 fill below the diagonal entries.
        let b = [1.0, 2.0, 3.0, 4.0];
        let x = f.solve(&b);
        assert!(max_resid(4, &dense, &x, &b) < 1e-10, "residual too large");
    }

    #[test]
    fn matches_dense_on_random_spd() {
        // Build a dense SPD matrix M = R + R^T + nI from a small LCG, factor both
        // sparse and dense, and confirm both solve the same system accurately.
        let n = 12;
        let mut state = 0x1234_5678_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut dense = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in i..n {
                let v = next();
                dense[i * n + j] = v;
                dense[j * n + i] = v;
            }
            dense[i * n + i] += n as f64; // strong diagonal -> SPD
        }
        let a = upper_csc(n, &dense);
        let f = sparse_ldl_factor(&a, 1e-14).unwrap();

        let b: Vec<f64> = (0..n).map(|i| (i as f64) - 5.0).collect();
        let x = f.solve(&b);
        assert!(max_resid(n, &dense, &x, &b) < 1e-9, "residual too large");
        // Sanity: L should have at least the strictly-lower fill of an SPD matrix.
        assert!(f.l_nnz() >= n - 1);
    }

    #[test]
    fn detects_zero_pivot() {
        let dense = [0.0, 0.0, 0.0, 1.0];
        let a = upper_csc(2, &dense);
        assert!(matches!(
            sparse_ldl_factor(&a, 1e-14),
            Err(LdlError::ZeroPivot(0))
        ));
    }

    #[test]
    fn near_singular_pivot_is_an_error_for_the_escalation_ladder() {
        // A 3x3 matrix whose first diagonal is 1e-12 (below the tolerance):
        // the factorization must abort with ZeroPivot rather than clamp —
        // recovery belongs to the caller's escalation ladder (re-regularize
        // globally and refactor), which measured better than any local boost
        // on ICONIC's ill-conditioned dynamics.
        let dense = [1e-12, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 3.0];
        let a = upper_csc(3, &dense);
        let sym = analyze(&a);
        assert!(matches!(
            factor_with_ws(&a, &sym, 1e-10, &mut LdlWorkspace::new(3)),
            Err(LdlError::ZeroPivot(0))
        ));
    }

    #[test]
    fn repairs_nan_diagonal_and_keeps_valid_block_exact() {
        // Block-diagonal [[5, 2], [2, 6]] with a NaN diagonal singleton third
        // block. A NaN would otherwise poison the factor (NaN comparisons are
        // always false, so the tolerance check cannot catch it, and every
        // subsequent division would carry the NaN through); the repair
        // replaces the pivot with the small nonzero value and the factor
        // proceeds, leaving the decoupled valid block exact.
        let dense = [5.0, 2.0, 0.0, 2.0, 6.0, 0.0, 0.0, 0.0, f64::NAN];
        let a = upper_csc(3, &dense);
        let f = sparse_ldl_factor(&a, 1e-14).unwrap();
        assert_eq!(f.nan_repairs, 1);
        let b = [1.0, 2.0, 3.0];
        let x = f.solve(&b);
        // Valid 2x2 block: [[5,2],[2,6]]^{-1} [1,2] = [1/13, 4/13].
        assert!((x[0] - 1.0 / 13.0).abs() < 1e-12);
        assert!((x[1] - 4.0 / 13.0).abs() < 1e-12);
        // The repaired nonzero pivot keeps the NaN-block entry finite.
        assert!(x[2].is_finite());
    }

    #[test]
    fn nan_repair_is_unconditional_on_the_workspace_path() {
        // The NaN repair fires on the workspace path too (not just the plain
        // constructor) — a repaired NaN counts once and the valid block stays
        // exact.
        let dense = [5.0, 2.0, 0.0, 2.0, 6.0, 0.0, 0.0, 0.0, f64::NAN];
        let a = upper_csc(3, &dense);
        let sym = analyze(&a);
        let mut ws = LdlWorkspace::new(3);
        let f = factor_with_ws(&a, &sym, 1e-14, &mut ws).unwrap();
        assert_eq!(f.nan_repairs, 1);
    }
}
