//! Dense row-major matrix and the basic operations dense solvers need.
//!
//! Sparse storage (`CscMatrix`) is the long-term representation; the first
//! interior-point milestone works densely for simplicity and is swapped onto the
//! sparse KKT solver.

use num_traits::Float;
use std::sync::Arc;

/// A dense matrix in row-major order: element `(i, j)` is `data[i * ncols + j]`.
///
/// The backing buffer is `Arc`-shared so `clone` is O(1); writes go through
/// `Arc::make_mut` (copy-on-write), so a pass that genuinely mutates a cloned
/// matrix pays exactly one deep copy — the presolve chain's no-op `prob.clone()`
/// traffic (~8 × 374MB on the 4800×4940 transport LPs) vanishes without
/// changing any mutation semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix<T> {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row-major entries (length `nrows * ncols`).
    pub data: Arc<Vec<T>>,
}

impl<T: Float> DenseMatrix<T> {
    /// An `nrows × ncols` matrix of zeros.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            data: Arc::new(vec![T::zero(); nrows * ncols]),
        }
    }

    /// Build from a row-major value vector. Panics if the length is wrong.
    pub fn from_row_major(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), nrows * ncols, "data length must be nrows*ncols");
        Self {
            nrows,
            ncols,
            data: Arc::new(data),
        }
    }

    /// The backing buffer if this is the sole owner, else a fresh deep copy
    /// (copy-on-write). All mutation goes through this or [`Self::set`].
    #[inline]
    pub fn data_mut(&mut self) -> &mut Vec<T> {
        Arc::make_mut(&mut self.data)
    }

    /// Consume and return the backing buffer, without copying when this is the
    /// sole owner (the common case for freshly built matrices).
    pub fn into_vec(self) -> Vec<T> {
        Arc::try_unwrap(self.data).unwrap_or_else(|a| (*a).clone())
    }

    /// Element `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i * self.ncols + j]
    }

    /// Set element `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: T) {
        let idx = i * self.ncols + j;
        self.data_mut()[idx] = v;
    }

    /// Overwrite the top-left `rows × cols` block with the same block of
    /// `src` (the "extend a matrix with extra rows" pattern). Panics if
    /// `self` is smaller than the block or `src` has fewer than `cols`
    /// columns.
    pub fn copy_block_from(&mut self, src: &Self, rows: usize, cols: usize) {
        assert!(rows <= self.nrows && cols <= self.ncols);
        assert!(cols <= src.ncols && rows <= src.nrows);
        let stride = self.ncols;
        let dst = self.data_mut();
        for (i, src_row) in src.data.chunks_exact(src.ncols).take(rows).enumerate() {
            let lo = i * stride;
            dst[lo..lo + cols].copy_from_slice(&src_row[..cols]);
        }
    }

    /// `y = M x`, length `nrows`.
    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        debug_assert_eq!(x.len(), self.ncols);
        // Row-slice iteration instead of self.get(i,j)/x[j] indexing: contiguous
        // slice iterators carry their own length invariant, so LLVM elides the
        // per-element bounds check that indexed access can't always prove away.
        // Measured consistently faster across two independent A/B runs (n=50..400),
        // by 1.1-2.1x depending on machine load -- no `unsafe` needed, unlike the
        // ldl_factor_into/LdlFactor::solve fix, since DenseMatrix's fields are
        // public and not constructor-enforced (same reasoning that ruled out
        // unchecked indexing for CscMatrix).
        let ncols = self.ncols;
        // chunks_exact panics if the chunk size is 0 -- a zero-column matrix (e.g.
        // a MIP node whose remaining free variables were all pinned by presolve)
        // is a legitimate, well-defined all-zero matvec, not an error.
        if ncols == 0 {
            return vec![T::zero(); self.nrows];
        }
        let mut y = Vec::with_capacity(self.nrows);
        for row in self.data.chunks_exact(ncols) {
            let mut s = T::zero();
            for (&a, &xj) in row.iter().zip(x.iter()) {
                s = s + a * xj;
            }
            y.push(s);
        }
        y
    }

    /// `y = Mᵀ x`, length `ncols`.
    /// `y = M x` into a pre-allocated output (caller reuses `y` across calls).
    pub fn matvec_into(&self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.ncols);
        debug_assert_eq!(y.len(), self.nrows);
        let ncols = self.ncols;
        if ncols == 0 {
            y.fill(T::zero());
            return;
        }
        for (row, yv) in self.data.chunks_exact(ncols).zip(y.iter_mut()) {
            let mut s = T::zero();
            for (&a, &xj) in row.iter().zip(x.iter()) {
                s = s + a * xj;
            }
            *yv = s;
        }
    }

    /// `y = Mᵀ x`, length `ncols`.
    pub fn matvec_t(&self, x: &[T]) -> Vec<T> {
        debug_assert_eq!(x.len(), self.nrows);
        let ncols = self.ncols;
        let mut y = vec![T::zero(); ncols];
        // See matvec: chunks_exact(0) panics, but a zero-column matrix has a
        // well-defined (empty) transpose-matvec result -- just return it.
        if ncols == 0 {
            return y;
        }
        for (row, &xi) in self.data.chunks_exact(ncols).zip(x.iter()) {
            for (yj, &a) in y.iter_mut().zip(row.iter()) {
                *yj = *yj + a * xi;
            }
        }
        y
    }

    /// `y = Mᵀ x`, length `ncols`, into a pre-allocated output (the caller reuses
    /// `y` across calls).
    pub fn matvec_t_into(&self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.nrows);
        debug_assert_eq!(y.len(), self.ncols);
        let ncols = self.ncols;
        y.fill(T::zero());
        if ncols == 0 {
            return;
        }
        for (row, &xi) in self.data.chunks_exact(ncols).zip(x.iter()) {
            for (yj, &a) in y.iter_mut().zip(row.iter()) {
                *yj = *yj + a * xi;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_and_transpose() {
        // M = [[1, 2, 3], [4, 5, 6]]
        let m = DenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.matvec(&[1.0, 1.0, 1.0]), vec![6.0, 15.0]);
        assert_eq!(m.matvec_t(&[1.0, 1.0]), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn empty_rows_transpose_is_zero() {
        let m = DenseMatrix::<f64>::zeros(0, 3);
        assert_eq!(m.matvec_t(&[]), vec![0.0, 0.0, 0.0]);
        assert_eq!(m.matvec(&[1.0, 2.0, 3.0]), Vec::<f64>::new());
    }

    #[test]
    fn zero_columns_matvec_is_zero_not_a_panic() {
        // A zero-column matrix arises legitimately (e.g. a MIP B&B node whose
        // every remaining free variable was pinned by presolve/branching) --
        // chunks_exact(0) panics, so matvec/matvec_t must special-case it.
        let m = DenseMatrix::<f64>::zeros(3, 0);
        assert_eq!(m.matvec(&[]), vec![0.0, 0.0, 0.0]);
        assert_eq!(m.matvec_t(&[1.0, 2.0, 3.0]), Vec::<f64>::new());
    }
}
