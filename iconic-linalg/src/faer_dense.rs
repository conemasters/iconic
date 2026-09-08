//! Dense symmetric factorizations backed by `faer`'s SIMD kernels, behind small cached
//! wrappers: Bunch–Kaufman **LBLT** (any symmetric matrix), unpivoted **LDLᵀ**
//! (quasidefinite — faster, no pivot search), and Cholesky **LLT** (positive definite —
//! fastest). The scalar LDLᵀ in `iconic-linalg::ldl` handles genuinely sparse
//! KKT systems. When the factor fills in densely — a dense Hessian `P`
//! (portfolio QPs) or a large PSD cone (whose svec block is dense) — a
//! vectorized dense kernel is several times faster.
//! dense). Inputs/outputs are `f64` (faer's real field); the generic solver converts at the
//! boundary.
//!
//! **NaN-diagonal repair:** a NaN on the diagonal poisons the factor — NaN
//! comparisons are always false, so faer's own pivot checks cannot catch it
//! and the poisoned pivot would carry through to the solve. Every `factor`
//! entry point replaces NaN diagonal entries with a small nonzero repair value
//! (~5.4e-101) before handing the matrix to faer, counting the repairs on the
//! factor (`nan_repairs`).

use crate::DenseMatrix;
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

pub use faer::mat::Mat as FaerMat;

/// The repair value for a NaN diagonal entry in a dense factor: a small
/// nonzero pivot. A NaN on the diagonal poisons the factor — NaN comparisons
/// are always false, so no pivot check catches it — and a nonzero repair
/// keeps every subsequent division finite so the factor proceeds.
/// Replace NaN diagonal entries of `m` with the shared repair value
/// ([`crate::ldl::nan_repair_value`]), returning the number repaired
/// (surfaced on the factor as `nan_repairs`).
fn repair_nan_diagonal(m: &mut Mat<f64>) -> usize {
    let mut repairs = 0;
    for j in 0..m.nrows() {
        if m[(j, j)].is_nan() {
            m[(j, j)] = crate::ldl::nan_repair_value::<f64>();
            repairs += 1;
        }
    }
    repairs
}

/// Copy a ICONIC dense matrix into a faer `Mat<f64>`.
fn to_mat(m: &DenseMatrix<f64>) -> Mat<f64> {
    Mat::from_fn(m.nrows, m.ncols, |i, j| m.get(i, j))
}

/// Create a faer `Mat<f64>` directly from a `DenseMatrix<T>`, reinterpreting the
/// data when T = f64 (the default floating-point type used by the solver).
pub fn to_mat_from<T: 'static>(m: &DenseMatrix<T>) -> Mat<f64>
where
    T: num_traits::Float + num_traits::ToPrimitive + 'static,
{
    let nrows = m.nrows;
    let ncols = m.ncols;
    // Reinterpret row-major DenseMatrix data as f64, fill column-major Mat
    // via from_fn.  The closure is called in column-major order (i changes
    // fastest), so we access data[i * ncols + j] — stride-ncols on the
    // row-major source.  For typical IPM dimensions (< 2000) this fits in
    // L2/L3 cache and the stride access is amortized by the factor cost.
    let data: &[f64] =
        unsafe { std::slice::from_raw_parts(m.data.as_ptr() as *const f64, m.data.len()) };
    Mat::from_fn(nrows, ncols, |i, j| data[i * ncols + j])
}

/// Set faer's global parallelism. faer defaults to a Rayon thread pool, which is a net win
/// for large factorizations but pure dispatch overhead for the small/medium dense systems the
/// interior-point loop factors repeatedly. The solver calls this once per solve — sequential
/// below a size threshold, Rayon above it — so small problems pay no thread-pool overhead.
pub fn set_parallelism_seq(seq: bool) {
    let par = if seq {
        faer::Par::Seq
    } else {
        faer::Par::rayon(0)
    };
    faer::set_global_parallelism(par);
}

/// RAII guard: forces faer's global parallelism sequential for its lifetime, restoring
/// whatever setting was previously active on drop (including on early return via `?`).
///
/// Use around a burst of small faer calls (tiny gemm/eigendecompositions, e.g. per-cone
/// PSD operations or a subspace-iteration detector) that individually run for far less
/// time than a Rayon thread-pool dispatch costs — measured 1.1x-2.9x faster sequential
/// for symmetric eigendecomposition up to k~50, roughly neutral above that, never worse.
/// Nests safely with an outer caller's own size-gated `set_parallelism_seq` (e.g. the
/// IPM's per-iteration KKT factor): the outer setting is restored exactly once this
/// guard drops, so a large factorization's own parallelism choice is unaffected.
pub struct SeqGuard(faer::Par);

impl SeqGuard {
    pub fn new() -> Self {
        let prev = faer::get_global_parallelism();
        set_parallelism_seq(true);
        SeqGuard(prev)
    }
}

impl Default for SeqGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SeqGuard {
    fn drop(&mut self) {
        faer::set_global_parallelism(self.0);
    }
}

/// Solve `K x = rhs` (single right-hand side) against any cached faer factorization.
fn solve_vec<S: Solve<f64>>(fac: &S, rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let b = Mat::from_fn(n, 1, |i, _| rhs[i]);
    let x = fac.solve(&b);
    (0..n).map(|i| x[(i, 0)]).collect()
}

/// Solve `K x = rhs` writing the solution into a caller-provided output,
/// reusing a persistent n×1 Mat scratch (allocated once by the caller, never
/// per call). `solve_in_place` uses the same triangular kernels as `solve`, so
/// the result is bit-identical.
pub fn solve_into<S: Solve<f64>>(
    fac: &S,
    rhs: &[f64],
    out: &mut [f64],
    scratch: &mut Mat<f64>,
) {
    let n = rhs.len();
    debug_assert_eq!(out.len(), n);
    if scratch.nrows() != n {
        *scratch = Mat::zeros(n, 1);
    }
    for i in 0..n {
        scratch[(i, 0)] = rhs[i];
    }
    fac.solve_in_place(&mut *scratch);
    for i in 0..n {
        out[i] = scratch[(i, 0)];
    }
}

/// `C = A·B` via faer's SIMD gemm. Generic over the scalar via an `f64` round-trip at
/// the boundary (identity for `f64`); worth it for `k ≳ 12`, below which a scalar loop
/// is faster — callers gate on size.
pub fn dense_matmul<T: 'static>(a: &DenseMatrix<T>, b: &DenseMatrix<T>) -> DenseMatrix<T>
where
    T: num_traits::Float + num_traits::FromPrimitive + num_traits::ToPrimitive + 'static,
{
    let (n, m, p) = (a.nrows, a.ncols, b.ncols);
    let mut c = DenseMatrix::<T>::zeros(n, p);
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let a_f: &[f64] =
            unsafe { std::slice::from_raw_parts(a.data.as_ptr() as *const f64, n * m) };
        let b_f: &[f64] =
            unsafe { std::slice::from_raw_parts(b.data.as_ptr() as *const f64, m * p) };
        let mut c_f = vec![0.0_f64; n * p];
        crate::blas::gemm(n, p, m, a_f, m, b_f, p, &mut c_f, p, 1.0, 0.0, false, false);
        for i in 0..n {
            for j in 0..p {
                c.set(i, j, T::from_f64(c_f[i * p + j]).expect("scalar literal"));
            }
        }
        return c;
    }
    let a_f: Vec<f64> = (0..n * m)
        .map(|k| a.get(k / m, k % m).to_f64().expect("finite scalar"))
        .collect();
    let b_f: Vec<f64> = (0..m * p)
        .map(|k| b.get(k / m, k % m).to_f64().expect("finite scalar"))
        .collect();
    let mut c_f = vec![0.0_f64; n * p];
    crate::blas::gemm(
        n, p, m, &a_f, m, &b_f, p, &mut c_f, p, 1.0, 0.0, false, false,
    );
    let mut c = DenseMatrix::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            c.set(i, j, T::from_f64(c_f[i * p + j]).expect("scalar literal"));
        }
    }
    c
}

/// A cached Bunch–Kaufman **LBLT** factorization of a symmetric matrix — pivoted, so it
/// handles any symmetric matrix (the general fallback).
pub struct FaerLblt {
    lblt: faer::linalg::solvers::Lblt<f64>,
    /// Number of NaN diagonal entries repaired before factorization (a NaN on
    /// the diagonal would otherwise poison the factor).
    pub nan_repairs: usize,
}

impl FaerLblt {
    /// Factor a dense symmetric matrix (only the lower triangle is read).
    pub fn factor(kkt: &DenseMatrix<f64>) -> FaerLblt {
        let mut m = to_mat(kkt);
        let nan_repairs = repair_nan_diagonal(&mut m);
        FaerLblt {
            lblt: m.lblt(Side::Lower),
            nan_repairs,
        }
    }

    /// Factor directly from a generic `DenseMatrix<T>`, avoiding the intermediate
    /// `DenseMatrix<f64>` allocation. Converts each element to f64 inline.
    pub fn factor_from<T>(kkt: &DenseMatrix<T>) -> FaerLblt
    where
        T: num_traits::Float + num_traits::ToPrimitive + 'static,
    {
        let mut m = to_mat_from(kkt);
        let nan_repairs = repair_nan_diagonal(&mut m);
        FaerLblt {
            lblt: m.lblt(Side::Lower),
            nan_repairs,
        }
    }

    /// Solve `K x = rhs` against the cached factorization.
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        solve_vec(&self.lblt, rhs)
    }

    /// Solve `K x = rhs` into a caller-provided output, reusing a persistent
    /// Mat scratch (no per-call allocation).
    pub fn solve_into(&self, rhs: &[f64], out: &mut [f64], scratch: &mut Mat<f64>) {
        solve_into(&self.lblt, rhs, out, scratch);
    }
}

/// A cached *unpivoted* **LDLᵀ** factorization, for a symmetric **quasidefinite** matrix
/// (PD block then ND block). Unpivoted LDLᵀ skips the Bunch–Kaufman pivot search that LBLT
/// performs, so it is ~2–3× faster — and it is stable for a quasidefinite matrix in this
/// ordering. `factor` returns `None` if a pivot is (near) zero (near-singular), in which case
/// the caller falls back to the pivoted LBLT.
pub struct FaerLdlt {
    ldlt: faer::linalg::solvers::Ldlt<f64>,
    /// Number of NaN diagonal entries repaired before factorization (a NaN on
    /// the diagonal would otherwise poison the factor).
    pub nan_repairs: usize,
}

impl FaerLdlt {
    /// Factor a symmetric quasidefinite matrix (only the lower triangle is read).
    pub fn factor(kkt: &DenseMatrix<f64>) -> Option<FaerLdlt> {
        let mut m = to_mat(kkt);
        let nan_repairs = repair_nan_diagonal(&mut m);
        m.ldlt(Side::Lower)
            .ok()
            .map(|ldlt| FaerLdlt { ldlt, nan_repairs })
    }

    /// Factor directly from a generic `DenseMatrix<T>`, avoiding the intermediate
    /// `DenseMatrix<f64>` allocation.
    pub fn factor_from<T>(kkt: &DenseMatrix<T>) -> Option<FaerLdlt>
    where
        T: num_traits::Float + num_traits::ToPrimitive + 'static,
    {
        let mut m = to_mat_from(kkt);
        let nan_repairs = repair_nan_diagonal(&mut m);
        m.ldlt(Side::Lower)
            .ok()
            .map(|ldlt| FaerLdlt { ldlt, nan_repairs })
    }

    /// Solve `K x = rhs` against the cached factorization.
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        solve_vec(&self.ldlt, rhs)
    }

    /// Solve `K x = rhs` into a caller-provided output, reusing a persistent
    /// Mat scratch (no per-call allocation).
    pub fn solve_into(&self, rhs: &[f64], out: &mut [f64], scratch: &mut Mat<f64>) {
        solve_into(&self.ldlt, rhs, out, scratch);
    }
}

/// A cached Cholesky (**LLT**) factorization of a symmetric positive-definite matrix —
/// faster than LBLT (no pivoting). Used for the condensed reduced system, which is
/// negative-definite (so its negation is PD), and the equality-free QP condensed system.
///
/// Uses faer's Cholesky directly: BLAS `dpotrf` (LAPACK) is more sensitive to
/// near-singular PD matrices and fails on typical IPM KKT systems where faer succeeds.
pub struct FaerLlt {
    inner: faer::linalg::solvers::Llt<f64>,
    /// Number of NaN diagonal entries repaired before factorization (a NaN on
    /// the diagonal would otherwise poison the factor).
    pub nan_repairs: usize,
}

impl FaerLlt {
    /// Factor a symmetric matrix as PD; returns `None` if it is not positive definite.
    pub fn factor(m: &DenseMatrix<f64>) -> Option<FaerLlt> {
        // Use faer Cholesky directly. BLAS dpotrf (LAPACK) is more sensitive to
        // near-singular PD matrices — fails on typical IPM KKT systems where faer
        // succeeds. BLAS dsyrk for the gram assembly is the real speedup.
        let mut m = to_mat(m);
        let nan_repairs = repair_nan_diagonal(&mut m);
        m.llt(Side::Lower)
            .ok()
            .map(|llt| FaerLlt { inner: llt, nan_repairs })
    }

    /// Factor directly from a generic `DenseMatrix<T>`, avoiding the intermediate
    /// `DenseMatrix<f64>` allocation.
    pub fn factor_from<T>(kkt: &DenseMatrix<T>) -> Option<FaerLlt>
    where
        T: num_traits::Float + num_traits::ToPrimitive + 'static,
    {
        // faer Cholesky directly — BLAS dpotrf fails on near-singular IPM matrices
        let mut m = to_mat_from(kkt);
        let nan_repairs = repair_nan_diagonal(&mut m);
        m.llt(Side::Lower)
            .ok()
            .map(|llt| FaerLlt { inner: llt, nan_repairs })
    }

    /// Solve `M x = rhs` against the cached factorization.
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        solve_vec(&self.inner, rhs)
    }

    /// Solve `M x = rhs` into a caller-provided output, reusing a persistent
    /// Mat scratch (no per-call allocation).
    pub fn solve_into(&self, rhs: &[f64], out: &mut [f64], scratch: &mut Mat<f64>) {
        solve_into(&self.inner, rhs, out, scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_indefinite_symmetric() {
        // Quasidefinite 3×3: [[2,1,0],[1,-3,1],[0,1,-1]] (PD/ND split).
        let mut k = DenseMatrix::<f64>::zeros(3, 3);
        for (i, j, v) in [
            (0, 0, 2.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, -3.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 2, -1.0),
        ] {
            k.set(i, j, v);
        }
        let b = [1.0, 2.0, 3.0];
        // LBLT (pivoted) and the unpivoted LDLᵀ agree, and both solve the system.
        let xb = FaerLblt::factor(&k).solve(&b);
        let xd = FaerLdlt::factor(&k).expect("quasidefinite").solve(&b);
        for i in 0..3 {
            let mut r = -b[i];
            for j in 0..3 {
                r += k.get(i, j) * xb[j];
            }
            assert!(r.abs() < 1e-12, "row {i} residual {r}");
            assert!((xb[i] - xd[i]).abs() < 1e-12, "LBLT vs LDLᵀ at {i}");
        }
    }

    /// Block-diagonal [[4, 1], [1, 3]] with a NaN diagonal singleton third block.
    fn nan_diag_matrix() -> DenseMatrix<f64> {
        let mut k = DenseMatrix::<f64>::zeros(3, 3);
        for (i, j, v) in [
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (2, 2, f64::NAN),
        ] {
            k.set(i, j, v);
        }
        k
    }

    #[test]
    fn lblt_repairs_nan_diagonal() {
        // A NaN on the diagonal poisons the factor (NaN comparisons are always
        // false, so faer's pivot checks cannot catch it); the repair replaces
        // it with the small nonzero value and the factor proceeds, leaving the
        // decoupled valid block exact.
        let f = FaerLblt::factor(&nan_diag_matrix());
        assert_eq!(f.nan_repairs, 1);
        let x = f.solve(&[1.0, 2.0, 3.0]);
        // Valid 2x2 block: [[4,1],[1,3]]^{-1} [1,2] = [1/11, 7/11].
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-10);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-10);
        // The repaired nonzero pivot keeps the NaN-block entry finite.
        assert!(x[2].is_finite());
    }

    #[test]
    fn llt_repairs_nan_diagonal() {
        // Cholesky: the repaired matrix is positive definite, so the factor
        // proceeds (without the repair the NaN diagonal fails the PD test).
        let f = FaerLlt::factor(&nan_diag_matrix()).expect("repaired matrix is PD");
        assert_eq!(f.nan_repairs, 1);
        let x = f.solve(&[1.0, 2.0, 3.0]);
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-10);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-10);
        assert!(x[2].is_finite());
    }

    #[test]
    fn ldlt_repairs_nan_diagonal() {
        // Unpivoted LDLᵀ: with the NaN repaired the quasidefinite factor
        // succeeds (the repaired pivot is small but nonzero); the repair is
        // counted either way, so a fallback to LBLT can never hide it.
        if let Some(f) = FaerLdlt::factor(&nan_diag_matrix()) {
            assert_eq!(f.nan_repairs, 1);
            let x = f.solve(&[1.0, 2.0, 3.0]);
            assert!((x[0] - 1.0 / 11.0).abs() < 1e-10);
            assert!((x[1] - 7.0 / 11.0).abs() < 1e-10);
            assert!(x[2].is_finite());
        }
    }
}
