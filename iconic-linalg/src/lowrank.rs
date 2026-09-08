//! Low-rank-plus-diagonal decomposition of a symmetric positive-semidefinite Hessian.
//!
//! Many quadratic objectives have a Hessian that is **low rank plus diagonal**: a
//! factor-model covariance `P = F Fᵀ + diag(d)` with `F` an `n×r` factor loading
//! (`r ≪ n`) is the canonical example. When a modeling layer multiplies that out into a
//! dense `n×n` `P`, the structure is lost and a dense solver factors the full matrix
//! every interior-point iteration at `O(n³)`. This module recovers `L` (n×r) and the
//! diagonal `d ≥ 0` directly from the dense `P`, so the quadratic can be re-expressed as
//!
//! ```text
//! ½ xᵀP x  =  ½‖Lᵀx‖²  +  ½ Σⱼ dⱼ xⱼ²,
//! ```
//!
//! and the `‖Lᵀx‖²` term pushed into an `r`-dimensional second-order cone — turning the
//! per-iteration factor cost from `O(n³)` into something that scales with `r`.
//!
//! ## Why not a plain pivoted Cholesky of `P`
//! `F Fᵀ + diag(d)` with `d > 0` is **full rank**: a Cholesky of `P` entangles the
//! diagonal `d` into every column, so its residual only becomes diagonal at full rank.
//! The diagonal must be separated from the low-rank part. The off-diagonal of `P` *is*
//! the off-diagonal of `F Fᵀ` exactly, so this is the classic factor-analysis problem:
//! find `d` such that `P − diag(d)` is PSD of rank `r`. We solve it by the standard
//! fixed point — given a diagonal estimate, take the top-`r` eigenpairs of `P − diag(d)`
//! as `L` and reset `d = diag(P) − diag(L Lᵀ)` — which converges geometrically for a
//! genuine factor model. The rank is read from the eigenvalue gap of `P`, and the final
//! off-diagonal residual is the structure test (large ⇒ not low-rank-plus-diagonal).

use crate::dense::DenseMatrix;
use crate::eig::symmetric_eigh;
use faer::Mat;
use num_traits::{Float, FromPrimitive, ToPrimitive};

/// A low-rank-plus-diagonal decomposition `P ≈ L Lᵀ + diag(d)`.
#[derive(Clone, Debug)]
pub struct LowRankDiag<T> {
    /// Low-rank factor `L` (`n × rank`, row-major). `½‖Lᵀx‖²` is the low-rank quadratic.
    pub l: DenseMatrix<T>,
    /// Non-negative diagonal `d` (length `n`).
    pub d: Vec<T>,
    /// The revealed rank (number of factor columns) — the cone dimension of the reformulation.
    pub rank: usize,
    /// Largest off-diagonal magnitude of the residual `P − L Lᵀ − diag(d)`, relative to
    /// `‖diag(P)‖∞`. Near zero iff `P` is genuinely low-rank-plus-diagonal.
    pub offdiag_rel: T,
}

/// Sort indices `0..n` by descending eigenvalue.
fn descending<T: Float>(evals: &[T]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..evals.len()).collect();
    idx.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Orthonormalize the columns of the faer matrix `v` (`n × r`) in place by modified
/// Gram–Schmidt.
fn orthonormalize_mat(v: &mut Mat<f64>) {
    let (n, r) = (v.nrows(), v.ncols());
    for j in 0..r {
        for k in 0..j {
            let mut proj = 0.0;
            for i in 0..n {
                proj += v[(i, j)] * v[(i, k)];
            }
            for i in 0..n {
                v[(i, j)] -= proj * v[(i, k)];
            }
        }
        let mut nrm = 0.0;
        for i in 0..n {
            nrm += v[(i, j)] * v[(i, j)];
        }
        nrm = nrm.sqrt();
        if nrm > 0.0 {
            let inv_nrm = 1.0 / nrm;
            for i in 0..n {
                v[(i, j)] *= inv_nrm;
            }
        }
    }
}

/// `M·V` for the symmetric `M = P − diag(d)`: faer's SIMD gemm `P·V` minus the column-wise
/// diagonal correction `d∘V`. `pf` is the prebuilt faer `P` (converted once); `V` is `n×r`.
fn apply_m(pf: &Mat<f64>, d: &[f64], v: &Mat<f64>) -> Mat<f64> {
    let (n, r) = (v.nrows(), v.ncols());
    let mut y = pf * v; // P·V (n×r)
    for i in 0..n {
        let di = d[i];
        for j in 0..r {
            y[(i, j)] -= di * v[(i, j)];
        }
    }
    y
}

/// Top-`r` eigenpairs of the symmetric `M = P − diag(d)` by **subspace iteration**, warm-
/// started from `v` (`n × cols`, refined in place). The `P·V` gemm is the only `O(n²·cols)`
/// cost — far below a full `O(n³)` eigendecomposition for `cols ≪ n`. Returns `(θ, L)` with
/// `θ` the `cols` Ritz values (descending) and `L[:,t] = sqrt(max(θ_t,0))·ritzvec_t` (the
/// columns of the `n × cols` factor in descending Ritz order). `power_steps` inner iterations
/// sharpen the subspace per call.
fn subspace_top_r(
    pf: &Mat<f64>,
    d: &[f64],
    v: &mut Mat<f64>,
    power_steps: usize,
) -> (Vec<f64>, Mat<f64>) {
    let r = v.ncols();

    // Subspace (orthogonal) iteration: V ← orth(M·V), repeated.
    for _ in 0..power_steps {
        *v = apply_m(pf, d, v);
        orthonormalize_mat(v);
    }

    // Rayleigh–Ritz: B = Vᵀ M V (r×r), eig(B) → Ritz values/vectors; eigenvectors of M are V·W.
    let mvv = apply_m(pf, d, v);
    let vt = v.transpose();
    let bf = vt * &mvv; // r×r
    let mut b = DenseMatrix::<f64>::zeros(r, r);
    for a in 0..r {
        for c in 0..r {
            // Symmetrize (guard rounding).
            b.set(a, c, 0.5 * (bf[(a, c)] + bf[(c, a)]));
        }
    }
    let (theta, w) = symmetric_eigh(&b);
    let order = descending(&theta);
    // Ritz eigenvectors V·W (n×r) in descending order, scaled by sqrt(θ) to form L.
    let mut wsorted = Mat::<f64>::zeros(r, r);
    let mut th = vec![0.0; r];
    for (t, &k) in order.iter().enumerate() {
        th[t] = theta[k];
        let s = theta[k].max(0.0).sqrt();
        for a in 0..r {
            wsorted[(a, t)] = s * w.get(a, k);
        }
    }
    let l = v as &Mat<f64> * &wsorted; // n×r
    (th, l)
}

/// Largest off-diagonal magnitude of `P − L Lᵀ`, with `P` the faer matrix and `L` the faer
/// `n × rank` factor. Computes `L Lᵀ` by faer gemm (the only `O(n²·rank)` cost) then scans.
fn offdiag_residual(pf: &Mat<f64>, l: &Mat<f64>, rank: usize) -> f64 {
    let n = pf.nrows();
    let lr = l.subcols(0, rank);
    let llt = lr * lr.transpose(); // n×n
    let mut mo = 0.0f64;
    for i in 0..n {
        for j in 0..i {
            let r = (pf[(i, j)] - llt[(i, j)]).abs();
            if r > mo {
                mo = r;
            }
        }
    }
    mo
}

/// Recover a low-rank-plus-diagonal decomposition `P ≈ L Lᵀ + diag(d)` of the symmetric PSD
/// matrix `p`.
///
/// The rank is read from the largest relative gap in the eigenvalue spectrum of `P` (capped
/// at `max_rank`), then `(L, d)` are refined by the factor-analysis fixed point. `rel_tol`
/// (scaled by `‖diag(P)‖∞`) sets the eigenvalue floor and the convergence threshold;
/// iteration stops once the off-diagonal residual falls below it or after a fixed number of
/// sweeps. Returns the decomposition; the caller inspects `rank` and `offdiag_rel` to decide
/// whether to use it. Returns `None` for an empty or rank-0 (already diagonal) matrix.
pub fn low_rank_plus_diag<T>(
    p: &DenseMatrix<T>,
    rel_tol: T,
    max_rank: usize,
) -> Option<LowRankDiag<T>>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    let n = p.nrows;
    debug_assert_eq!(p.ncols, n, "decomposition requires a square matrix");
    if n == 0 || max_rank == 0 {
        return None;
    }

    // Force sequential faer dispatch: subspace iteration issues ~2 tiny gemms
    // (O(n²·cols), cols ≤ 64) per sweep over up to 34 sweeps. Each is far too
    // small to amortize Rayon thread-pool dispatch overhead — unlike the IPM's
    // per-iteration factor (which already gates on size via `set_parallelism_seq`
    // in `solve_qp_with_termination`), this routine runs standalone and would
    // otherwise inherit whatever global parallelism faer defaults to.
    let _seq = crate::faer_dense::SeqGuard::new();

    // The decomposition is a numeric routine on f64 (subspace iteration with faer's SIMD
    // gemm); convert `P` once and operate in f64, converting `L`/`d` back to `T` at the end.
    let pf = Mat::<f64>::from_fn(n, n, |i, j| p.get(i, j).to_f64().expect("finite scalar"));
    let mut diag_scale = 0.0f64;
    for i in 0..n {
        let v = pf[(i, i)].abs();
        if v > diag_scale {
            diag_scale = v;
        }
    }
    let scale = diag_scale.max(1.0);
    let tol = rel_tol.to_f64().expect("finite scalar") * scale;

    // Track a `cap`-column subspace and refine it with subspace iteration throughout — both
    // the rank detection and the factor-analysis fixed point share it, avoiding any full
    // `O(n³)` eigendecomposition. `cap` bounds the detectable rank; it is kept modest (the
    // routine only pays off when r ≪ n, and a larger detection subspace makes every
    // `O(n²·cap)` gemm dearer), so a factor model whose rank exceeds it simply isn't reported
    // as low-rank and the caller falls back to the dense path.
    let cap = max_rank.min(n.saturating_sub(1)).max(1);
    // Random orthonormal start (deterministic LCG, so the decomposition is reproducible).
    let mut seed: u64 = 0x1234_5678_9abc_def0;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut v = Mat::<f64>::from_fn(n, cap, |_, _| rnd());
    orthonormalize_mat(&mut v);

    // --- rank from the Ritz-value gap on P (warm the subspace with extra power steps) ---
    let zeros_d = vec![0.0f64; n];
    let (theta0, _) = subspace_top_r(&pf, &zeros_d, &mut v, 4);
    let mut rank = 0usize;
    let mut best_gap = 1.0f64;
    for k in 0..cap.saturating_sub(1) {
        let hi = theta0[k].max(0.0);
        let lo = theta0[k + 1].max(0.0);
        if hi <= tol {
            break;
        }
        let gap = if lo > tol { hi / lo } else { hi / tol };
        if gap > best_gap {
            best_gap = gap;
            rank = k + 1;
        }
    }
    if rank == 0 {
        return None;
    }

    // Shrink the working subspace to `rank` (plus a small oversampling for robust subspace
    // convergence) — the fixed-point sweeps only need the top-`rank` directions, and the
    // per-sweep cost (`O(n²·cols)` gemm) drops with the column count. Reuse the leading
    // columns of the warmed subspace.
    let cols = (rank + 4).min(cap);
    if cols < cap {
        let mut vr = Mat::<f64>::from_fn(n, cols, |i, j| v[(i, j)]);
        orthonormalize_mat(&mut vr);
        v = vr;
    }

    // --- factor-analysis fixed point ---
    // d ← diag(P) − diag(L Lᵀ), L = top-r eigenpairs of (P − diag(d)). Start from d = 0; each
    // sweep refines the shared subspace (one power step) then a Rayleigh–Ritz extraction.
    // Convergence is tracked by the cheap `O(n)` change in `d` (the fixed point is contractive
    // for a genuine factor model); the `O(n²·rank)` off-diagonal residual is computed only
    // *once* at the end, as the reported structure measure — keeping it off the per-sweep path.
    let mut d = vec![0.0f64; n];
    let mut lf = Mat::<f64>::zeros(n, rank);
    let max_sweeps = 30usize;
    for _ in 0..max_sweeps {
        let (_theta, lfull) = subspace_top_r(&pf, &d, &mut v, 1);
        // Keep the top-`rank` Ritz columns (lfull's columns are already descending).
        for t in 0..rank {
            for i in 0..n {
                lf[(i, t)] = lfull[(i, t)];
            }
        }
        // d ← diag(P) − diag(L Lᵀ), clamped non-negative; track the max change.
        let mut max_change = 0.0f64;
        for i in 0..n {
            let mut llt = 0.0;
            for t in 0..rank {
                let val = lf[(i, t)];
                llt += val * val;
            }
            let di = (pf[(i, i)] - llt).max(0.0);
            let ch = (di - d[i]).abs();
            if ch > max_change {
                max_change = ch;
            }
            d[i] = di;
        }
        if max_change <= tol {
            break;
        }
    }
    // Final residual (the reported structure measure).
    let offdiag = offdiag_residual(&pf, &lf, rank);

    // Convert back to T.
    let mut l = DenseMatrix::<T>::zeros(n, rank);
    for i in 0..n {
        for t in 0..rank {
            l.set(i, t, T::from_f64(lf[(i, t)]).expect("scalar literal"));
        }
    }
    let d_t: Vec<T> = d.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect();
    let offdiag_rel = T::from_f64(offdiag / scale).expect("scalar literal");
    Some(LowRankDiag {
        l,
        d: d_t,
        rank,
        offdiag_rel,
    })
}

/// Build `P = F·Fᵀ + diag(d)` densely from factor loadings `F` (given as `n` rows
/// of `r` loadings) and idiosyncratic variances `d` — the shared fixture for the
/// low-rank tests and downstream crates' structure-exploiting test problems.
pub fn gram_plus_diag(f: &[Vec<f64>], d: &[f64]) -> DenseMatrix<f64> {
    let n = f.len();
    let r = f[0].len();
    let mut p = DenseMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for t in 0..r {
                acc += f[i][t] * f[j][t];
            }
            p.set(i, j, acc);
        }
        p.set(i, i, p.get(i, i) + d[i]);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random `F` (n×r) and positive `d`, plus the resulting `P`.
    fn random_gram_plus_diag(
        n: usize,
        r: usize,
        varying: bool,
        seed: u64,
    ) -> (DenseMatrix<f64>, Vec<f64>) {
        let mut s = seed;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let f: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rnd()).collect()).collect();
        let d: Vec<f64> = (0..n)
            .map(|_| {
                if varying {
                    0.2 + (rnd() * 0.8).abs()
                } else {
                    0.3
                }
            })
            .collect();
        (gram_plus_diag(&f, &d), d)
    }

    /// `P = L Lᵀ + diag(d)` reconstructs the input to within the returned max error.
    fn reconstruct_err(p: &DenseMatrix<f64>, lr: &LowRankDiag<f64>) -> f64 {
        let n = p.nrows;
        let mut e = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut llt = 0.0;
                for k in 0..lr.rank {
                    llt += lr.l.get(i, k) * lr.l.get(j, k);
                }
                if i == j {
                    llt += lr.d[i];
                }
                e = e.max((llt - p.get(i, j)).abs());
            }
        }
        e
    }

    #[test]
    fn recovers_constant_diagonal_gram_plus_diag() {
        let (p, _) = random_gram_plus_diag(20, 4, false, 12345);
        let lr = low_rank_plus_diag(&p, 1e-9, 20).unwrap();
        assert_eq!(lr.rank, 4, "rank (offdiag_rel={:e})", lr.offdiag_rel);
        assert!(lr.offdiag_rel < 1e-7, "offdiag_rel {}", lr.offdiag_rel);
        assert!(reconstruct_err(&p, &lr) < 1e-7);
    }

    #[test]
    fn recovers_varying_diagonal_gram_plus_diag() {
        for &(n, r, seed) in &[(40, 5, 7u64), (100, 6, 3), (60, 3, 99)] {
            let (p, _) = random_gram_plus_diag(n, r, true, seed);
            let lr = low_rank_plus_diag(&p, 1e-9, n).unwrap();
            assert_eq!(lr.rank, r, "n={n} rank (offdiag_rel={:e})", lr.offdiag_rel);
            assert!(
                lr.offdiag_rel < 1e-7,
                "n={n} offdiag_rel {}",
                lr.offdiag_rel
            );
            assert!(reconstruct_err(&p, &lr) < 1e-6, "n={n} reconstruct");
        }
    }

    /// A genuinely dense full-rank `P` cannot be captured at low rank: with the rank capped
    /// well below `n`, the off-diagonal residual stays large — the structure check flags it.
    #[test]
    fn dense_matrix_flagged() {
        let n = 30;
        let mut s = 555u64;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut m = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m.set(i, j, rnd());
            }
        }
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += m.get(i, t) * m.get(j, t);
                }
                p.set(i, j, acc);
            }
            p.set(i, i, p.get(i, i) + n as f64);
        }
        let lr = low_rank_plus_diag(&p, 1e-9, 5).unwrap();
        assert!(lr.rank <= 5);
        assert!(
            lr.offdiag_rel > 1e-3,
            "dense P should leave off-diagonal residual, got {}",
            lr.offdiag_rel
        );
    }
}
