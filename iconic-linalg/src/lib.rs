#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::multiple_bound_locations,
    clippy::incompatible_msrv,
    clippy::empty_line_after_doc_comments
)]

//! `iconic-linalg` — sparse CSC matrix storage and numeric kernels.
//!
//! This is the only crate that knows about sparsity. It owns the `CscMatrix`
//! type, the dense/sparse kernels, and (from M2) the quasidefinite LDLᵀ KKT
//! solver behind the `KktSolver` trait.

pub mod blas;
// Index-based loops read more clearly than iterator adapters for this linear-algebra code.

pub mod dense;
pub mod eig;
pub mod faer_dense;
pub mod ldl;
pub mod lowrank;
pub mod ordering;
pub mod sparse_ldl;

pub use dense::DenseMatrix;
pub use ldl::{ldl_factor, ldl_factor_into, LdlError, LdlFactor};
pub use lowrank::{gram_plus_diag, low_rank_plus_diag, LowRankDiag};
pub use ordering::{min_degree, min_degree_with_threshold, permute_upper};
pub use sparse_ldl::{analyze, factor_with, factor_with_ws, sparse_ldl_factor, LdlWorkspace, SparseLdl, Symbolic};

use num_traits::Float;

/// Compressed-sparse-column matrix.
///
/// `colptr` has length `n + 1`; column `j` occupies `nzval[colptr[j]..colptr[j+1]]`
/// with corresponding row indices in `rowval`. Row indices within a column are
/// kept sorted ascending.
#[derive(Clone, Debug, PartialEq)]
pub struct CscMatrix<T> {
    /// Row count.
    pub m: usize,
    /// Column count.
    pub n: usize,
    /// Column pointers (length `n + 1`).
    pub colptr: Vec<usize>,
    /// Row indices of stored entries (length `nnz`).
    pub rowval: Vec<usize>,
    /// Stored values (length `nnz`).
    pub nzval: Vec<T>,
}

impl<T> CscMatrix<T> {
    /// Number of stored nonzeros.
    pub const fn nnz(&self) -> usize {
        self.nzval.len()
    }

    /// An `m × n` matrix with no stored entries.
    pub fn zeros(m: usize, n: usize) -> Self {
        Self {
            m,
            n,
            colptr: vec![0; n + 1],
            rowval: Vec::new(),
            nzval: Vec::new(),
        }
    }
}

impl<T: Float> CscMatrix<T> {
    /// Sparse matrix-vector product y = A*x (O(nnz)).
    pub fn matvec(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.m];
        for j in 0..self.n {
            let xj = x[j];
            if xj != T::zero() {
                for p in self.colptr[j]..self.colptr[j + 1] {
                    y[self.rowval[p]] = y[self.rowval[p]] + self.nzval[p] * xj;
                }
            }
        }
        y
    }

    /// y = A*x into a pre-allocated output (caller reuses `y` across calls).
    /// Same accumulation order as `matvec` (zero-fills `y` first).
    pub fn matvec_into(&self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(y.len(), self.m);
        y.fill(T::zero());
        for j in 0..self.n {
            let xj = x[j];
            if xj != T::zero() {
                for p in self.colptr[j]..self.colptr[j + 1] {
                    y[self.rowval[p]] = y[self.rowval[p]] + self.nzval[p] * xj;
                }
            }
        }
    }

    /// Sparse transpose matrix-vector product y = A^T*x (O(nnz)).
    pub fn matvec_t(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        for j in 0..self.n {
            let mut acc = T::zero();
            for p in self.colptr[j]..self.colptr[j + 1] {
                acc = acc + self.nzval[p] * x[self.rowval[p]];
            }
            y[j] = acc;
        }
        y
    }

    /// y = A^T*x into a pre-allocated output (caller reuses `y` across calls).
    /// Same accumulation order as `matvec_t` (overwrites every entry of `y`).
    pub fn matvec_t_into(&self, x: &[T], y: &mut [T]) {
        debug_assert_eq!(x.len(), self.m);
        debug_assert_eq!(y.len(), self.n);
        for j in 0..self.n {
            let mut acc = T::zero();
            for p in self.colptr[j]..self.colptr[j + 1] {
                acc = acc + self.nzval[p] * x[self.rowval[p]];
            }
            y[j] = acc;
        }
    }
}

/// Dense dot product `xᵀy`. Panics in debug if lengths differ.
pub fn dot<T: Float>(x: &[T], y: &[T]) -> T {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = T::zero();
    for (&a, &b) in x.iter().zip(y) {
        acc = acc + a * b;
    }
    acc
}

/// Infinity norm `‖x‖∞ = maxᵢ |xᵢ|`.
pub fn inf_norm<T: Float>(x: &[T]) -> T {
    let mut m = T::zero();
    for &v in x {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_no_nnz() {
        let a = CscMatrix::<f64>::zeros(3, 4);
        assert_eq!(a.nnz(), 0);
        assert_eq!(a.colptr.len(), 5);
    }

    #[test]
    fn dot_product() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        assert_eq!(dot(&x, &y), 32.0);
    }
}
pub mod supernodal_ldl;
