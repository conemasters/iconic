//! Symmetric eigendecomposition by the cyclic Jacobi method.
//!
//! For a dense symmetric matrix `A`, returns eigenvalues and the orthonormal
//! eigenvectors (as columns) with `A = V Λ Vᵀ`. Jacobi is simple, accurate for
//! symmetric matrices, and needs no external LAPACK — it is the substrate the PSD
//! cone needs (matrix square roots, projections, and the NT scaling all reduce to an
//! eigendecomposition).

use crate::dense::DenseMatrix;
use num_traits::Float;

/// Copy `a` into a fresh column-major faer matrix. The f64 case reinterprets the
/// row-major buffer directly inside `from_fn` (no per-element conversion); other
/// scalars convert through `to_f64`.
fn mat_from_dense<T: 'static>(a: &DenseMatrix<T>, k: usize) -> faer::Mat<f64>
where
    T: Float + num_traits::ToPrimitive,
{
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let data: &[f64] =
            unsafe { std::slice::from_raw_parts(a.data.as_ptr() as *const f64, k * k) };
        faer::Mat::from_fn(k, k, |i, j| data[i * k + j])
    } else {
        faer::Mat::from_fn(k, k, |i, j| a.get(i, j).to_f64().expect("finite scalar"))
    }
}

/// Refill the persistent scratch copy of `a` in place (same conversion rules as
/// [`mat_from_dense`]).
fn fill_scratch<T: 'static>(a: &DenseMatrix<T>, mat_scratch: &mut faer::Mat<f64>)
where
    T: Float + num_traits::ToPrimitive,
{
    let k = a.nrows;
    if mat_scratch.nrows() != k {
        *mat_scratch = faer::Mat::zeros(k, k);
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let data: &[f64] =
            unsafe { std::slice::from_raw_parts(a.data.as_ptr() as *const f64, k * k) };
        for i in 0..k {
            for j in 0..k {
                mat_scratch[(i, j)] = data[i * k + j];
            }
        }
    } else {
        for i in 0..k {
            for j in 0..k {
                mat_scratch[(i, j)] = a.get(i, j).to_f64().expect("finite scalar");
            }
        }
    }
}

/// Scatter faer's eigendecomposition results into the caller's buffers.
fn store_eigen<T>(s: &[f64], u: &faer::MatRef<'_, f64>, evals_out: &mut [T], v_out: &mut DenseMatrix<T>)
where
    T: num_traits::FromPrimitive + Float,
{
    let k = evals_out.len();
    for (out, &v) in evals_out.iter_mut().zip(s.iter()) {
        *out = T::from_f64(v).expect("scalar literal");
    }
    for i in 0..k {
        for j in 0..k {
            v_out.set(i, j, T::from_f64(u[(i, j)]).expect("scalar literal"));
        }
    }
}

/// Smallest eigenvalue from faer's eigenvalues-only solve.
fn min_of<T: num_traits::FromPrimitive + Float>(evals: &[f64]) -> T {
    let mn = evals.iter().cloned().fold(f64::INFINITY, f64::min);
    T::from_f64(mn).expect("scalar literal")
}

/// Symmetric eigendecomposition, auto-selecting the backend by size: scalar Jacobi for
/// small matrices (lower overhead, wins below ~8) and faer's SIMD self-adjoint solver
/// for larger ones (≈2× at k=10 up to ≈6× at k=30). Eigenvectors are columns; the
/// eigenvalue ordering is unspecified (callers reconstruct order-independently).
pub fn symmetric_eigh<T: 'static>(a: &DenseMatrix<T>) -> (Vec<T>, DenseMatrix<T>)
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    let k = a.nrows;
    let mut evals = vec![T::zero(); k];
    let mut v = DenseMatrix::zeros(k, k);
    symmetric_eigh_into(a, &mut evals, &mut v);
    (evals, v)
}

/// [`symmetric_eigh`] into caller buffers: eigenvalues into `evals_out` (length `k`),
/// eigenvectors (columns) into `v_out` (`k×k`, fully overwritten). Same backend
/// dispatch and copy order as [`symmetric_eigh`], so a cached pair can be refilled
/// in place with no allocation.
pub fn symmetric_eigh_into<T: 'static>(
    a: &DenseMatrix<T>,
    evals_out: &mut [T],
    v_out: &mut DenseMatrix<T>,
) where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    let k = a.nrows;
    if k < 8 {
        jacobi_eigh_into(a, evals_out, v_out);
        return;
    }
    let eigen = mat_from_dense(a, k)
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("self-adjoint eigendecomposition failed");
    let sd = eigen.S();
    let s: Vec<f64> = (0..evals_out.len()).map(|i| sd[i]).collect();
    store_eigen(&s, &eigen.U(), evals_out, v_out);
}

/// Symmetric eigendecomposition writing into caller-provided outputs (`evals`
/// length k, `v_out` k×k, eigenvectors as columns), reusing a persistent faer
/// Mat scratch for the input copy (the eigen call itself still allocates
/// internally). Same backend gate and results as [`symmetric_eigh`]; the
/// eigenvalue ordering is unspecified.
pub fn symmetric_eigh_into_scratch<T: 'static>(
    a: &DenseMatrix<T>,
    evals: &mut [T],
    v_out: &mut DenseMatrix<T>,
    mat_scratch: &mut faer::Mat<f64>,
) where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    let k = a.nrows;
    if k < 8 {
        let (e, v) = jacobi_eigh(a);
        evals[..k].copy_from_slice(&e);
        for i in 0..k {
            for j in 0..k {
                v_out.set(i, j, v.get(i, j));
            }
        }
        return;
    }
    fill_scratch(a, mat_scratch);
    let eigen = mat_scratch
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("self-adjoint eigendecomposition failed");
    let sd = eigen.S();
    let s: Vec<f64> = (0..evals.len()).map(|i| sd[i]).collect();
    store_eigen(&s, &eigen.U(), evals, v_out);
}

/// Smallest eigenvalue of a symmetric matrix, **without** computing eigenvectors —
/// roughly twice as fast as the full decomposition. Used by the cone step length, which
/// needs only `λ_min`. Small matrices still go through Jacobi (vector cost is negligible).
pub fn min_eigenvalue<T: 'static>(a: &DenseMatrix<T>) -> T
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    let k = a.nrows;
    if k < 8 {
        let (evals, _) = jacobi_eigh(a);
        return evals
            .iter()
            .cloned()
            .fold(T::infinity(), |m, e| if e < m { e } else { m });
    }
    let evals = mat_from_dense(a, k)
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .expect("self-adjoint eigenvalues failed");
    min_of(&evals)
}

/// Smallest eigenvalue into a caller-provided output, reusing a persistent
/// faer Mat scratch for the input copy (the eigen call itself still allocates
/// internally — hoisting the `Mat::from_fn` copy is the achievable part).
pub fn min_eigenvalue_into<T: 'static>(
    a: &DenseMatrix<T>,
    out: &mut T,
    mat_scratch: &mut faer::Mat<f64>,
) where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive,
{
    let k = a.nrows;
    if k < 8 {
        let (evals, _) = jacobi_eigh(a);
        *out = evals
            .iter()
            .cloned()
            .fold(T::infinity(), |m, e| if e < m { e } else { m });
        return;
    }
    fill_scratch(a, mat_scratch);
    let evals = mat_scratch
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .expect("self-adjoint eigenvalues failed");
    *out = min_of(&evals);
}

/// Eigenvalues and eigenvectors (columns of the returned matrix) of a symmetric
/// matrix, via cyclic Jacobi rotations. `A = V·diag(eigvals)·Vᵀ`.
fn jacobi_eigh<T: Float>(a: &DenseMatrix<T>) -> (Vec<T>, DenseMatrix<T>) {
    let n = a.nrows;
    let mut evals = vec![T::zero(); n];
    let mut v = DenseMatrix::zeros(n, n);
    jacobi_eigh_into(a, &mut evals, &mut v);
    (evals, v)
}

/// [`jacobi_eigh`] into caller buffers. `v_out` is fully overwritten (explicitly
/// reset to the identity first — it may be a reused buffer) and `evals_out` is
/// filled with the diagonal in the same order as [`jacobi_eigh`]'s return.
fn jacobi_eigh_into<T: Float>(a: &DenseMatrix<T>, evals_out: &mut [T], v_out: &mut DenseMatrix<T>) {
    let n = a.nrows;
    debug_assert_eq!(a.ncols, n);
    let mut m = a.clone();
    for i in 0..n {
        for j in 0..n {
            v_out.set(i, j, T::zero());
        }
    }
    for i in 0..n {
        v_out.set(i, i, T::one());
    }
    if n <= 1 {
        if n == 1 {
            evals_out[0] = m.get(0, 0);
        }
        return;
    }

    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius norm.
        let mut off = T::zero();
        for p in 0..n {
            for q in (p + 1)..n {
                off = off + m.get(p, q) * m.get(p, q);
            }
        }
        if off <= T::epsilon() * T::epsilon() {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m.get(p, q);
                if apq.abs() <= T::epsilon() {
                    continue;
                }
                let app = m.get(p, p);
                let aqq = m.get(q, q);
                // Rotation angle: t = tan(θ).
                let theta = (aqq - app) / (apq + apq);
                let sign = if theta >= T::zero() {
                    T::one()
                } else {
                    -T::one()
                };
                let t = sign / (theta.abs() + (theta * theta + T::one()).sqrt());
                let c = T::one() / (t * t + T::one()).sqrt();
                let s = t * c;

                // Apply the Givens rotation J(p,q,θ) to both sides: M ← JᵀMJ.
                for k in 0..n {
                    let mkp = m.get(k, p);
                    let mkq = m.get(k, q);
                    m.set(k, p, c * mkp - s * mkq);
                    m.set(k, q, s * mkp + c * mkq);
                }
                for k in 0..n {
                    let mpk = m.get(p, k);
                    let mqk = m.get(q, k);
                    m.set(p, k, c * mpk - s * mqk);
                    m.set(q, k, s * mpk + c * mqk);
                }
                // Accumulate eigenvectors: V ← V·J.
                for k in 0..n {
                    let vkp = v_out.get(k, p);
                    let vkq = v_out.get(k, q);
                    v_out.set(k, p, c * vkp - s * vkq);
                    v_out.set(k, q, s * vkp + c * vkq);
                }
            }
        }
    }

    for i in 0..n {
        evals_out[i] = m.get(i, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(eigvals: &[f64], v: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        // V diag(λ) Vᵀ.
        let n = v.nrows;
        let mut a = DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += v.get(i, k) * eigvals[k] * v.get(j, k);
                }
                a.set(i, j, acc);
            }
        }
        a
    }

    fn max_diff(a: &DenseMatrix<f64>, b: &DenseMatrix<f64>) -> f64 {
        a.data
            .iter()
            .zip(b.data.iter())
            .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
    }

    #[test]
    fn diagonal_matrix() {
        let a = DenseMatrix::from_row_major(2, 2, vec![3.0, 0.0, 0.0, 5.0]);
        let (mut ev, _) = jacobi_eigh(&a);
        ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((ev[0] - 3.0).abs() < 1e-12 && (ev[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn known_2x2() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        let a = DenseMatrix::from_row_major(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (mut ev, v) = jacobi_eigh(&a);
        let recon = reconstruct(&ev, &v);
        assert!(max_diff(&a, &recon) < 1e-12, "reconstruction off");
        ev.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((ev[0] - 1.0).abs() < 1e-12 && (ev[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn reconstructs_random_symmetric() {
        // Deterministic pseudo-random symmetric 5x5.
        let n = 5;
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut a = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let val = next();
                a.set(i, j, val);
                a.set(j, i, val);
            }
        }
        let (ev, v) = jacobi_eigh(&a);
        assert!(max_diff(&a, &reconstruct(&ev, &v)) < 1e-10);
        // Eigenvectors orthonormal: VᵀV = I.
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += v.get(k, i) * v.get(k, j);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-10, "VᵀV not identity");
            }
        }
    }
}
