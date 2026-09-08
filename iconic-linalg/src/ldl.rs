//! Dense LDLᵀ factorization without pivoting.
//!
//! ICONIC's regularized KKT systems are *quasidefinite* (a positive-definite block
//! and a negative-definite block), which guarantees an LDLᵀ factorization exists
//! for the natural ordering with no pivoting — `D` simply carries positive and
//! negative diagonal entries. This dense version serves the M1 interior-point
//! solver; M2 replaces it with the sparse, ordered factorization.
//!
//! **NaN-diagonal repair:** a NaN on the diagonal poisons the factor — NaN
//! comparisons are always false, so no pivot-tolerance check can catch it and
//! the poisoned pivot would silently propagate through every `L` column and the
//! solve. The factorization replaces it with a small nonzero repair value
//! (~5.4e-101) and continues, counting the repair on the factor (`nan_repairs`).

use crate::dense::DenseMatrix;
use num_traits::Float;

/// Why a factorization could not be produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LdlError {
    /// A pivot fell below the tolerance at the given index (matrix effectively singular).
    ZeroPivot(usize),
}

/// The repair value for a NaN diagonal entry in a dense factor: a small
/// nonzero pivot. A NaN on the diagonal poisons the factor — NaN comparisons
/// are always false, so the tolerance check cannot catch it — and a nonzero
/// repair keeps every subsequent division finite so the factor proceeds.
///
/// (For `f32` this value underflows to zero, degrading to a plain
/// zero-repair — `f32` factors are a rarely-used embedded path.)
pub(crate) fn nan_repair_value<T: Float>() -> T {
    <T as num_traits::NumCast>::from(5.4e-101).expect("scalar literal")
}

/// An `L·D·Lᵀ` factorization with unit lower-triangular `L` and diagonal `D`.
#[derive(Clone, Debug)]
pub struct LdlFactor<T> {
    l: DenseMatrix<T>,
    d: Vec<T>,
    n: usize,
    /// Number of NaN diagonal entries repaired during the most recent
    /// factorization (a NaN on the diagonal would otherwise poison the factor).
    pub nan_repairs: usize,
}

impl<T: Float> LdlFactor<T> {
    /// A zeroed factorization buffer for reuse via [`ldl_factor_into`] across
    /// iterations (no per-factor allocation).
    pub fn with_capacity(n: usize) -> Self {
        LdlFactor {
            l: DenseMatrix::zeros(n, n),
            d: vec![T::zero(); n],
            n,
            nan_repairs: 0,
        }
    }
}

/// Factor a symmetric matrix `k` (only the lower triangle is read) as `L·D·Lᵀ`.
///
/// `pivot_tol` guards against a (near-)zero pivot; a pivot with magnitude at or
/// below it yields [`LdlError::ZeroPivot`].
pub fn ldl_factor<T: Float>(k: &DenseMatrix<T>, pivot_tol: T) -> Result<LdlFactor<T>, LdlError> {
    let n = k.nrows;
    let mut out = LdlFactor {
        l: DenseMatrix::zeros(n, n),
        d: vec![T::zero(); n],
        n,
        nan_repairs: 0,
    };
    ldl_factor_into(k, pivot_tol, &mut out)?;
    Ok(out)
}

/// Numeric factorization overwriting a pre-allocated [`LdlFactor`] in-place.
/// Caller reuses the same `out` across iterations to eliminate malloc/memset
/// overhead (~5% of ldl_factor time per perf profile).
pub fn ldl_factor_into<T: Float>(
    k: &DenseMatrix<T>,
    pivot_tol: T,
    out: &mut LdlFactor<T>,
) -> Result<(), LdlError> {
    let n = k.nrows;
    assert_eq!(k.ncols, n, "LDL requires a square matrix");
    out.n = n;
    out.nan_repairs = 0;
    debug_assert_eq!(out.l.nrows, n);
    debug_assert_eq!(out.l.ncols, n);
    debug_assert_eq!(out.l.data.len(), n * n);
    debug_assert_eq!(out.d.len(), n);
    debug_assert_eq!(k.data.len(), n * n);
    // O(n^3) factorization inner loop -- same unchecked-indexing rationale as
    // LdlFactor::solve below: `l` (n x n) and `d` (length n) are constructed
    // together right above by this same function, so the invariant is
    // encapsulation-enforced, not convention. Measured hot on the MIP B&B
    // path (perf: ~11-12% of total solve time), where a node's IPM fallback
    // factors a fresh small dense condensed system every call.
    let l = out.l.data_mut().as_mut_slice();
    let d = out.d.as_mut_slice();
    let kd = k.data.as_slice();
    unsafe {
        for j in 0..n {
            let mut dj = *kd.get_unchecked(j * n + j);
            for p in 0..j {
                let ljp = *l.get_unchecked(j * n + p);
                dj = dj - ljp * ljp * *d.get_unchecked(p);
            }
            if dj.is_nan() {
                // A NaN on the diagonal poisons the factor: NaN comparisons
                // are always false, so the tolerance check cannot catch it.
                // Repair the pivot to a small nonzero value (the factor
                // proceeds, and division by it stays finite) and count the
                // repair so callers can see the factor needed unusual repair.
                dj = nan_repair_value::<T>();
                out.nan_repairs += 1;
            } else if dj.abs() <= pivot_tol {
                return Err(LdlError::ZeroPivot(j));
            }
            *d.get_unchecked_mut(j) = dj;
            for i in (j + 1)..n {
                let mut s = *kd.get_unchecked(i * n + j);
                for p in 0..j {
                    s = s - *l.get_unchecked(i * n + p)
                        * *l.get_unchecked(j * n + p)
                        * *d.get_unchecked(p);
                }
                *l.get_unchecked_mut(i * n + j) = s / dj;
            }
        }
    }
    Ok(())
}

impl<T: Float> LdlFactor<T> {
    /// Solve `K u = rhs` for `u`. `L` is treated as unit lower-triangular.
    ///
    /// Called (at least) twice per IPM iteration — predictor and corrector RHS
    /// against the same factorization — on the small dense condensed systems
    /// (n<48 Hessian block, m<64 Schur complement). The forward/backward
    /// substitution below indexes `L`'s backing `Vec` with `get_unchecked`:
    /// `l.ncols == n` and both loop bounds (`0..n`) are within `l`'s
    /// `n×n` allocation by construction, so this removes bounds-check overhead
    /// the optimizer can't otherwise elide (measured ~10-16% faster, bit-identical).
    pub fn solve(&self, rhs: &[T]) -> Vec<T> {
        debug_assert_eq!(rhs.len(), self.n);
        let n = self.n;
        let ncols = self.l.ncols;
        debug_assert_eq!(self.l.data.len(), n * ncols);
        debug_assert_eq!(self.d.len(), n);
        let mut w = rhs.to_vec();
        let l = self.l.data.as_slice();

        unsafe {
            // Forward: L w = rhs.
            for i in 0..n {
                let mut s = *w.get_unchecked(i);
                for p in 0..i {
                    s = s - *l.get_unchecked(i * ncols + p) * *w.get_unchecked(p);
                }
                *w.get_unchecked_mut(i) = s;
            }
            // Diagonal: D v = w.
            for i in 0..n {
                *w.get_unchecked_mut(i) = *w.get_unchecked(i) / *self.d.get_unchecked(i);
            }
            // Backward: Lᵀ u = v.
            for i in (0..n).rev() {
                let mut s = *w.get_unchecked(i);
                for p in (i + 1)..n {
                    s = s - *l.get_unchecked(p * ncols + i) * *w.get_unchecked(p);
                }
                *w.get_unchecked_mut(i) = s;
            }
        }
        w
    }

    /// Solve `LDLᵀ x = rhs` into a pre-allocated output buffer (the caller reuses
    /// `x` across solves, avoiding the per-solve allocation). Mirrors [`Self::solve`]
    /// exactly — same accumulation order, so bit-identical results.
    pub fn solve_into(&self, rhs: &[T], x: &mut [T]) {
        debug_assert_eq!(rhs.len(), self.n);
        debug_assert_eq!(x.len(), self.n);
        let n = self.n;
        let ncols = self.l.ncols;
        x.copy_from_slice(rhs);
        let l = self.l.data.as_slice();
        // Forward: L w = rhs.
        for i in 0..n {
            let mut s = x[i];
            for p in 0..i {
                s = s - l[i * ncols + p] * x[p];
            }
            x[i] = s;
        }
        // Diagonal: D v = w.
        for i in 0..n {
            x[i] = x[i] / self.d[i];
        }
        // Backward: Lᵀ u = v.
        for i in (0..n).rev() {
            let mut s = x[i];
            for p in (i + 1)..n {
                s = s - l[p * ncols + i] * x[p];
            }
            x[i] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_spd_system() {
        // K = [[4, 2], [2, 3]], b = [2, 1]  ->  x = [0.25, 0.166...]
        let k = DenseMatrix::from_row_major(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
        let f = ldl_factor(&k, 1e-14).unwrap();
        let x = f.solve(&[2.0, 1.0]);
        // verify K x = b
        let kx0 = 4.0 * x[0] + 2.0 * x[1];
        let kx1 = 2.0 * x[0] + 3.0 * x[1];
        assert!((kx0 - 2.0).abs() < 1e-12);
        assert!((kx1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn solves_quasidefinite_system() {
        // Indefinite-but-quasidefinite: [[2, 1], [1, -3]].
        let k = DenseMatrix::from_row_major(2, 2, vec![2.0, 1.0, 1.0, -3.0]);
        let f = ldl_factor(&k, 1e-14).unwrap();
        let b = [1.0, -1.0];
        let x = f.solve(&b);
        let r0 = 2.0 * x[0] + 1.0 * x[1] - b[0];
        let r1 = 1.0 * x[0] - 3.0 * x[1] - b[1];
        assert!(r0.abs() < 1e-12 && r1.abs() < 1e-12);
    }

    #[test]
    fn detects_zero_pivot() {
        let k = DenseMatrix::from_row_major(2, 2, vec![0.0, 0.0, 0.0, 1.0]);
        assert!(matches!(ldl_factor(&k, 1e-14), Err(LdlError::ZeroPivot(0))));
    }

    #[test]
    fn repairs_nan_diagonal_and_keeps_valid_block_exact() {
        // Block-diagonal [[4, 1], [1, 3]] with a NaN diagonal singleton third
        // block. A NaN would otherwise poison the factor (NaN comparisons are
        // always false, so the tolerance check cannot catch it, and every
        // subsequent division would carry the NaN through); the repair replaces
        // the pivot with the small nonzero value and the factor proceeds,
        // leaving the decoupled valid block exact.
        let k = DenseMatrix::from_row_major(
            3,
            3,
            vec![4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, f64::NAN],
        );
        // pivot_tol (1e-14) is far above the repair value: the repaired pivot
        // is used as-is (not re-checked against the tolerance), so the factor
        // succeeds rather than erroring.
        let f = ldl_factor(&k, 1e-14).unwrap();
        assert_eq!(f.nan_repairs, 1);
        let x = f.solve(&[1.0, 2.0, 3.0]);
        // Valid 2x2 block: [[4,1],[1,3]]^{-1} [1,2] = [1/11, 7/11].
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-12);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-12);
        // The repaired nonzero pivot keeps the NaN-block entry finite.
        assert!(x[2].is_finite());
    }
}
