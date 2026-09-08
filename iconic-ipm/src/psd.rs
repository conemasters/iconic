//! Positive-semidefinite cone primitives (the matrix analogue of [`crate::soc`]).
//!
//! Symmetric `k×k` matrices are vectorized by **svec** — the upper triangle stacked
//! column by column with off-diagonals scaled by `√2`, so that
//! `⟨svec(X), svec(Y)⟩ = tr(XY)`. The cone is `{X : X ⪰ 0}`. Membership, the
//! Jordan product `X∘Y = ½(XY+YX)`, and the exact step-to-the-PSD-boundary all
//! reduce to the symmetric eigendecomposition in `iconic-linalg::eig`.

use iconic_core::Scalar;
use iconic_linalg::eig::{symmetric_eigh, symmetric_eigh_into};
use iconic_linalg::DenseMatrix;

/// Side length `k` of a symmetric matrix whose svec has length `m = k(k+1)/2`.
pub fn side_dim(m: usize) -> usize {
    // k(k+1)/2 = m  ->  k = (−1 + √(1+8m))/2.
    let mut k = 0;
    while k * (k + 1) / 2 < m {
        k += 1;
    }
    k
}

/// Vectorize a symmetric `k×k` matrix (upper triangle, column-major, `√2` on the
/// off-diagonals).
pub fn svec<T: Scalar>(x: &DenseMatrix<T>) -> Vec<T> {
    let k = x.nrows;
    let r2 = T::from_f64(2.0).expect("scalar literal").sqrt();
    let mut v = Vec::with_capacity(k * (k + 1) / 2);
    for j in 0..k {
        for i in 0..=j {
            v.push(if i == j {
                x.get(i, j)
            } else {
                r2 * x.get(i, j)
            });
        }
    }
    v
}

/// Inverse of [`svec`]: rebuild the symmetric matrix.
pub fn smat<T: Scalar>(v: &[T]) -> DenseMatrix<T> {
    let k = side_dim(v.len());
    let inv_r2 = T::one() / T::from_f64(2.0).expect("scalar literal").sqrt();
    let mut x = DenseMatrix::zeros(k, k);
    let mut idx = 0;
    for j in 0..k {
        for i in 0..=j {
            let val = v[idx];
            idx += 1;
            if i == j {
                x.set(i, j, val);
            } else {
                let off = val * inv_r2;
                x.set(i, j, off);
                x.set(j, i, off);
            }
        }
    }
    x
}

/// Smallest eigenvalue of `smat(v)` — the PSD-cone margin (≥ 0 in the cone). Needs no
/// eigenvectors, so it uses the eigenvalues-only path.
pub fn min_eig<T: Scalar>(v: &[T]) -> T {
    iconic_linalg::eig::min_eigenvalue(&smat(v))
}

/// Cone identity `svec(I)`.
pub fn identity<T: Scalar>(m: usize) -> Vec<T> {
    let k = side_dim(m);
    let mut x = DenseMatrix::zeros(k, k);
    for i in 0..k {
        x.set(i, i, T::one());
    }
    svec(&x)
}

/// Jordan product `X ∘ Y = ½(XY + YX)`, in svec coordinates.
pub fn jordan<T: Scalar>(u: &[T], v: &[T]) -> Vec<T> {
    let x = smat(u);
    let y = smat(v);
    let k = x.nrows;
    let half = T::from_f64(0.5).expect("scalar literal");
    // ½(XY + YX).
    let xy = mat_mul(&x, &y);
    let yx = mat_mul(&y, &x);
    let mut w = DenseMatrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            w.set(i, j, half * (xy.get(i, j) + yx.get(i, j)));
        }
    }
    svec(&w)
}

/// Largest `α ≥ 0` with `smat(v) + α·smat(dv) ⪰ 0`, or `+∞` if unbounded.
///
/// Works in the coordinates of `X⁻¹ᐟ²(dX)X⁻¹ᐟ²` (requires `X ≻ 0`): the step is
/// `−1/μ_min` over the negative eigenvalues `μ` of that matrix.
pub fn max_step<T: Scalar>(v: &[T], dv: &[T]) -> T {
    let cache = invsqrt(v);
    max_step_cached(&cache, v, dv)
}

/// `X⁻¹ᐟ²` (from `smat(v)`) plus whether `X` is well-conditioned — the part of the
/// step computation that depends only on the cone point `v`, so it can be computed
/// once per iteration and reused across the several directions checked against it.
pub struct InvSqrt<T> {
    pub(crate) xih: DenseMatrix<T>,
    pub(crate) well_conditioned: bool,
}

/// Compute [`InvSqrt`] for cone point `v` (one eigendecomposition).
pub fn invsqrt<T: Scalar>(v: &[T]) -> InvSqrt<T> {
    let x = smat(v);
    let k = x.nrows;
    let (evals, vmat) = symmetric_eigh(&x);
    let xmax = evals
        .iter()
        .fold(T::zero(), |m, &e| if e > m { e } else { m });
    let xmin = evals
        .iter()
        .fold(T::infinity(), |m, &e| if e < m { e } else { m });
    let mut xih = DenseMatrix::zeros(k, k);
    // `1/√λ` depends only on `l`; hoisted out of the (i,j) loops of the O(k³)
    // accumulation (one sqrt+recip per λ instead of one per element).
    let inv_sqrt_lam: Vec<T> = (0..k)
        .map(|l| {
            if evals[l] > T::zero() {
                evals[l].sqrt().recip()
            } else {
                T::zero()
            }
        })
        .collect();
    for i in 0..k {
        for j in 0..k {
            let mut acc = T::zero();
            for l in 0..k {
                acc += vmat.get(i, l) * inv_sqrt_lam[l] * vmat.get(j, l);
            }
            xih.set(i, j, acc);
        }
    }
    let well_conditioned = xmin > T::from_f64(1e-9).expect("scalar literal") * xmax;
    InvSqrt {
        xih,
        well_conditioned,
    }
}

/// [`invsqrt`] into a caller-provided [`InvSqrt`]: the smat/xih/eigenvector
/// allocations are hoisted into the persistent scratch (faer's internal eigen
/// workspace still allocates). `out.xih` must be k×k (every entry is
/// overwritten). Same results as [`invsqrt`].
pub fn invsqrt_into<T: Scalar>(v: &[T], out: &mut InvSqrt<T>, scr: &mut PsdScratch<T>) {
    smat_into(v, &mut scr.x);
    let k = scr.x.nrows;
    iconic_linalg::eig::symmetric_eigh_into_scratch(&scr.x, &mut scr.evals, &mut scr.vmat, &mut scr.eig_mat);
    let xmax = scr
        .evals
        .iter()
        .fold(T::zero(), |m, &e| if e > m { e } else { m });
    let xmin = scr
        .evals
        .iter()
        .fold(T::infinity(), |m, &e| if e < m { e } else { m });
    let inv_sqrt_lam = &mut scr.inv_sqrt_lam;
    for l in 0..k {
        inv_sqrt_lam[l] = if scr.evals[l] > T::zero() {
            scr.evals[l].sqrt().recip()
        } else {
            T::zero()
        };
    }
    let xih = &mut out.xih;
    for i in 0..k {
        for j in 0..k {
            let mut acc = T::zero();
            for l in 0..k {
                acc += scr.vmat.get(i, l) * inv_sqrt_lam[l] * scr.vmat.get(j, l);
            }
            xih.set(i, j, acc);
        }
    }
    out.well_conditioned = xmin > T::from_f64(1e-9).expect("scalar literal") * xmax;
}

/// Largest step using a precomputed [`InvSqrt`] of the cone point `v` — avoids
/// recomputing `eig(X)` when several directions are checked against the same `v`.
pub fn max_step_cached<T: Scalar>(cache: &InvSqrt<T>, v: &[T], dv: &[T]) -> T {
    let dx = smat(dv);
    // M = X^{-1/2} dX X^{-1/2}. The step to the boundary is −1/λ_min(M) (when λ_min < 0),
    // so only the smallest eigenvalue is needed — no eigenvectors.
    let m = mat_mul(&mat_mul(&cache.xih, &dx), &cache.xih);
    let mn = iconic_linalg::eig::min_eigenvalue(&m);
    let alpha = if mn < T::zero() {
        -T::one() / mn
    } else {
        T::infinity()
    };
    if !alpha.is_finite() {
        return alpha; // dX never pushes out of the cone
    }
    // Fixed boundary margin: a step computed exactly to the cone boundary can
    // round past it, leaving a non-interior iterate; backing off a fixed 1e-13
    // keeps the point strictly interior with negligible step cost (the caller's
    // fraction-to-boundary eta does the bulk backoff).
    let margin = T::from_f64(1e-13).expect("scalar literal");
    // For a well-conditioned X the estimate is the exact boundary step; only a
    // near-singular X (where X^{-1/2} drops directions) needs verify-and-back-off.
    if cache.well_conditioned {
        return (alpha - margin).max(T::zero());
    }
    let tol = T::from_f64(-1e-9).expect("scalar literal");
    let backoff = T::from_f64(0.9).expect("scalar literal");
    let mut a = alpha;
    for _ in 0..50 {
        let test: Vec<T> = (0..v.len()).map(|i| v[i] + a * dv[i]).collect();
        if min_eig(&test) >= tol {
            return (a - margin).max(T::zero());
        }
        a *= backoff;
    }
    (a - margin).max(T::zero())
}

/// [`max_step_cached`] writing the step into `out`, reusing a persistent
/// scratch (no per-call k² allocations except faer's internal ones). Same
/// results as [`max_step_cached`].
pub fn max_step_cached_into<T: Scalar>(
    cache: &InvSqrt<T>,
    v: &[T],
    dv: &[T],
    out: &mut T,
    scr: &mut PsdScratch<T>,
) {
    smat_into(dv, &mut scr.x);
    // M = X^{-1/2} dX X^{-1/2}. The step to the boundary is −1/λ_min(M) (when λ_min < 0),
    // so only the smallest eigenvalue is needed — no eigenvectors.
    mat_mul_into(&cache.xih, &scr.x, &mut scr.z);
    mat_mul_into(&scr.z, &cache.xih, &mut scr.w);
    let mut mn = T::zero();
    iconic_linalg::eig::min_eigenvalue_into(&scr.w, &mut mn, &mut scr.eig_mat);
    let alpha = if mn < T::zero() {
        -T::one() / mn
    } else {
        T::infinity()
    };
    if !alpha.is_finite() {
        *out = alpha; // dX never pushes out of the cone
        return;
    }
    // Fixed boundary margin (see max_step_cached).
    let margin = T::from_f64(1e-13).expect("scalar literal");
    if cache.well_conditioned {
        *out = (alpha - margin).max(T::zero());
        return;
    }
    let tol = T::from_f64(-1e-9).expect("scalar literal");
    let backoff = T::from_f64(0.9).expect("scalar literal");
    let mut a = alpha;
    for _ in 0..50 {
        for i in 0..v.len() {
            scr.test_buf[i] = v[i] + a * dv[i];
        }
        smat_into(&scr.test_buf, &mut scr.x);
        let mut m = T::zero();
        iconic_linalg::eig::min_eigenvalue_into(&scr.x, &mut m, &mut scr.eig_mat);
        if m >= tol {
            *out = (a - margin).max(T::zero());
            return;
        }
        a *= backoff;
    }
    *out = (a - margin).max(T::zero());
}

fn mat_mul<T: Scalar>(a: &DenseMatrix<T>, b: &DenseMatrix<T>) -> DenseMatrix<T> {
    let (n, m, p) = (a.nrows, a.ncols, b.ncols);
    // faer's SIMD gemm wins from k ≈ 9 (measured: 0.78x at k=9, 0.63x at k=10,
    // 0.44x at k=12 vs the scalar loop); below that the call overhead dominates.
    if n >= 9 && p >= 9 {
        return iconic_linalg::faer_dense::dense_matmul(a, b);
    }
    let mut c = DenseMatrix::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            let mut acc = T::zero();
            for l in 0..m {
                acc += a.get(i, l) * b.get(l, j);
            }
            c.set(i, j, acc);
        }
    }
    c
}

/// `A^p` for symmetric PD `A`, via the eigendecomposition (`p` is a real power).
pub fn matrix_power<T: Scalar>(a: &DenseMatrix<T>, p: f64) -> DenseMatrix<T> {
    let (ev, v) = symmetric_eigh(a);
    let k = a.nrows;
    let pe = T::from_f64(p).expect("scalar literal");
    // `ev^pe` depends only on `l`; hoisted out of the (i,j) loops of the O(k³)
    // accumulation. For p = 0.5 (the only production caller, `nt_scaling`) the
    // runtime-parameter libm `pow` (~50-100 cycles) becomes `sqrt` (~4).
    let half = T::from_f64(0.5).expect("scalar literal");
    let ev_pow: Vec<T> = (0..k)
        .map(|l| {
            if ev[l] > T::zero() {
                if pe == half {
                    ev[l].sqrt()
                } else {
                    ev[l].powf(pe)
                }
            } else {
                T::zero()
            }
        })
        .collect();
    let mut out = DenseMatrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            let mut acc = T::zero();
            for l in 0..k {
                acc += v.get(i, l) * ev_pow[l] * v.get(j, l);
            }
            out.set(i, j, acc);
        }
    }
    out
}

/// Transpose of a dense matrix.
fn transpose<T: Scalar>(a: &DenseMatrix<T>) -> DenseMatrix<T> {
    let (r, c) = (a.nrows, a.ncols);
    let mut t = DenseMatrix::zeros(c, r);
    for i in 0..r {
        for j in 0..c {
            t.set(j, i, a.get(i, j));
        }
    }
    t
}

/// Lower-triangular Cholesky factor `L` of a symmetric PD matrix (`A = L Lᵀ`). The
/// diagonal is guarded against a tiny/negative rounding at the cone boundary.
fn cholesky_lower<T: Scalar>(a: &DenseMatrix<T>) -> DenseMatrix<T> {
    let k = a.nrows;
    let floor = T::from_f64(1e-300).expect("scalar literal");
    let mut l = DenseMatrix::zeros(k, k);
    for j in 0..k {
        let mut d = a.get(j, j);
        for p in 0..j {
            d -= l.get(j, p) * l.get(j, p);
        }
        let ljj = d.max(floor).sqrt();
        l.set(j, j, ljj);
        // Reciprocal-multiply: one division per column instead of one per entry.
        let inv_ljj = ljj.recip();
        for i in (j + 1)..k {
            let mut s = a.get(i, j);
            for p in 0..j {
                s -= l.get(i, p) * l.get(j, p);
            }
            l.set(i, j, s * inv_ljj);
        }
    }
    l
}

/// Inverse of a lower-triangular matrix (by forward substitution, column by column).
fn tri_inv_lower<T: Scalar>(l: &DenseMatrix<T>) -> DenseMatrix<T> {
    let k = l.nrows;
    let mut inv = DenseMatrix::zeros(k, k);
    for c in 0..k {
        for i in c..k {
            let mut s = if i == c { T::one() } else { T::zero() };
            for p in c..i {
                s -= l.get(i, p) * inv.get(p, c);
            }
            inv.set(i, c, s / l.get(i, i));
        }
    }
    inv
}

/// Nesterov–Todd scaling matrix `W` for the PSD cone (the geometric mean of `S` and
/// `Z⁻¹`), satisfying `W Z W = S` (i.e. `H z = s`); congruence by `W` is the cone's
/// `(z,z)` block. Computed in the **Cholesky** form `W = L⁻ᵀ (Lᵀ S L)^{1/2} L⁻¹` with
/// `Z = L Lᵀ` — one Cholesky plus one eigendecomposition instead of two.
/// Inputs are svec vectors.
pub fn nt_scaling<T: Scalar>(s: &[T], z: &[T]) -> DenseMatrix<T> {
    let sm = smat(s);
    let zm = smat(z);
    let l = cholesky_lower(&zm); // Z = L Lᵀ
    let lt = transpose(&l);
    let m = mat_mul(&mat_mul(&lt, &sm), &l); // M = Lᵀ S L
    let mhalf = matrix_power(&m, 0.5); // the single eigendecomposition
    let linv = tri_inv_lower(&l);
    let linvt = transpose(&linv);
    mat_mul(&mat_mul(&linvt, &mhalf), &linv) // W = L⁻ᵀ M^{1/2} L⁻¹
}

/// Build `W^{1/2}` and `W^{-1/2}` from an eigendecomposition `W = Q diag(ev) Qᵀ`, as
/// `Q diag(ev^{±1/2}) Qᵀ` (summing only the positive eigenvalues).
fn halves_from_eig<T: Scalar>(ev: &[T], q: &DenseMatrix<T>) -> (DenseMatrix<T>, DenseMatrix<T>) {
    let k = q.nrows;
    // `√λ` and `1/√λ` depend only on `l`; hoisted out of the (i,j) loops.
    let sq: Vec<T> = ev
        .iter()
        .map(|&e| if e > T::zero() { e.sqrt() } else { T::zero() })
        .collect();
    let isq: Vec<T> = sq
        .iter()
        .map(|&r| if r > T::zero() { r.recip() } else { T::zero() })
        .collect();
    let mut wh = DenseMatrix::zeros(k, k);
    let mut wih = DenseMatrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            let mut a = T::zero();
            let mut b = T::zero();
            for l in 0..k {
                a += q.get(i, l) * sq[l] * q.get(j, l);
                b += q.get(i, l) * isq[l] * q.get(j, l);
            }
            wh.set(i, j, a);
            wih.set(i, j, b);
        }
    }
    (wh, wih)
}

/// `W^{1/2}` and `W^{-1/2}` for symmetric PD `W`, from a *single* eigendecomposition
/// (the two are otherwise computed by two separate `matrix_power` calls, each its own
/// eig). Returns `(W^{1/2}, W^{-1/2})`.
pub fn scaling_halves<T: Scalar>(w: &DenseMatrix<T>) -> (DenseMatrix<T>, DenseMatrix<T>) {
    let (ev, q) = symmetric_eigh(w);
    halves_from_eig(&ev, &q)
}

/// Like [`scaling_halves`], but also returns the [`KronEig`] of `W` from the *same*
/// eigendecomposition — for the Kronecker SDP path, which needs both `W^{±1/2}` (for the
/// half-scaling in the corrector) and `eig(W)` (for `kron_solve`), avoiding a second eig.
pub fn scaling_halves_and_eig<T: Scalar>(
    w: &DenseMatrix<T>,
) -> (DenseMatrix<T>, DenseMatrix<T>, KronEig<T>) {
    let (ev, q) = symmetric_eigh(w);
    let (wh, wih) = halves_from_eig(&ev, &q);
    let vt = transpose(&q);
    (wh, wih, KronEig { lam: ev, v: q, vt })
}

/// Apply the PSD scaling block (congruence by `W`): `svec(W · smat(v) · W)`.
pub fn apply_nt<T: Scalar>(w: &DenseMatrix<T>, v: &[T]) -> Vec<T> {
    let vm = smat(v);
    svec(&mat_mul(&mat_mul(w, &vm), w))
}

/// Eigendecomposition `W = V Λ Vᵀ` of the (PD) NT scaling matrix, cached so the
/// Kronecker solve reuses it across the several right-hand sides of one iteration.
pub struct KronEig<T: Scalar> {
    lam: Vec<T>,
    v: DenseMatrix<T>,
    vt: DenseMatrix<T>,
}

impl<T: Scalar> KronEig<T> {
    #[cfg(test)]
    pub fn new(w: &DenseMatrix<T>) -> KronEig<T> {
        let (lam, v) = symmetric_eigh(w);
        let k = w.nrows;
        let mut vt = DenseMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                vt.set(i, j, v.get(j, i));
            }
        }
        KronEig { lam, v, vt }
    }
}

/// Solve `(W ⊗ₛ W + c·I) x = b` for `x` (svec coordinates), exploiting the symmetric
/// Kronecker structure: since `(W⊗ₛW)·svec(M) = svec(W·M·W)`, the system is
/// `W·smat(x)·W + c·smat(x) = smat(b)`, which in `W`'s eigenbasis decouples to
/// `(λᵢλⱼ + c)·M'ᵢⱼ = (Vᵀ·smat(b)·V)ᵢⱼ`. `O(k³)` (two `k×k` matmuls), versus `O(k⁶)`
/// to factor the dense `k²×k²` block — and it needs no assembled block at all.
pub fn kron_solve<T: Scalar>(eig: &KronEig<T>, c: T, b: &[T]) -> Vec<T> {
    let bmat = smat(b);
    let bp = mat_mul(&mat_mul(&eig.vt, &bmat), &eig.v); // Vᵀ B V
    let k = bmat.nrows;
    let mut mp = DenseMatrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            mp.set(i, j, bp.get(i, j) / (eig.lam[i] * eig.lam[j] + c));
        }
    }
    let m = mat_mul(&mat_mul(&eig.v, &mp), &eig.vt); // V M' Vᵀ
    svec(&m)
}

/// Pre-allocated scratch for the `_into` cone-op variants (four k² matrices,
/// reused across calls — sized once per cone dimension).
pub struct PsdScratch<T: Scalar> {
    x: DenseMatrix<T>,
    y: DenseMatrix<T>,
    z: DenseMatrix<T>,
    w: DenseMatrix<T>,
    /// `1/(λᵢλⱼ + c)` for the Kronecker solve — invariant across the RHS within
    /// an iteration, refilled per `kron_solve_into` call.
    inv_den: DenseMatrix<T>,
    /// Persistent outputs for the step-length eigendecompositions
    /// (`invsqrt_into` / `max_step_cached_into`): the eigenvalue vector, the
    /// eigenvector matrix, the hoisted `1/√λ` vector, the persistent faer
    /// input Mat (the eigen call itself still allocates internally), and the
    /// verify-and-back-off trial-point buffer.
    evals: Vec<T>,
    vmat: DenseMatrix<T>,
    inv_sqrt_lam: Vec<T>,
    eig_mat: iconic_linalg::faer_dense::FaerMat<f64>,
    test_buf: Vec<T>,
}

impl<T: Scalar> PsdScratch<T> {
    pub fn new(k: usize) -> Self {
        PsdScratch {
            x: DenseMatrix::zeros(k, k),
            y: DenseMatrix::zeros(k, k),
            z: DenseMatrix::zeros(k, k),
            w: DenseMatrix::zeros(k, k),
            inv_den: DenseMatrix::zeros(k, k),
            evals: vec![T::zero(); k],
            vmat: DenseMatrix::zeros(k, k),
            inv_sqrt_lam: vec![T::zero(); k],
            eig_mat: iconic_linalg::faer_dense::FaerMat::zeros(0, 0),
            // Trial points live in svec space — k(k+1)/2 entries, not k.
            test_buf: vec![T::zero(); k * (k + 1) / 2],
        }
    }

    /// Resize in place to side `k` when the current size differs (a no-op
    /// otherwise). Lets a persistent scratch held across per-iteration solve
    /// calls be (re)sized lazily instead of reallocated on every call.
    pub(crate) fn ensure_k(&mut self, k: usize) {
        if self.x.nrows == k {
            return;
        }
        *self = PsdScratch::new(k);
    }
}

/// Per-cone cached eigenpair of the arrow-inverse scaling matrix `smat(lam)`:
/// storage allocated once per solve, refilled in place every iteration. The
/// combined-step loop computes the pair once and the Gondzio corrector rounds
/// (which run strictly after) reuse it, so each PSD cone does one
/// eigendecomposition per iteration instead of one per corrector call.
pub struct ArrowEig<T: Scalar> {
    /// Eigenvalues of `smat(lam)`.
    pub d: Vec<T>,
    /// Eigenvectors (columns) of `smat(lam)`.
    pub q: DenseMatrix<T>,
}

impl<T: Scalar> ArrowEig<T> {
    pub fn new(k: usize) -> Self {
        ArrowEig {
            d: vec![T::zero(); k],
            q: DenseMatrix::zeros(k, k),
        }
    }

    /// Refill the cached `smat(lam)` eigenpair in place.
    pub fn compute(&mut self, lam: &[T], scr: &mut PsdScratch<T>) {
        smat_into(lam, &mut scr.x);
        let k = self.q.nrows;
        symmetric_eigh_into(&scr.x, &mut self.d[..k], &mut self.q);
    }
}

/// `svec` into a caller buffer (same accumulation order as [`svec`]).
pub fn svec_into<T: Scalar>(x: &DenseMatrix<T>, out: &mut [T]) {
    let k = x.nrows;
    let r2 = T::from_f64(2.0).expect("scalar literal").sqrt();
    let mut idx = 0usize;
    for j in 0..k {
        for i in 0..=j {
            out[idx] = if i == j {
                x.get(i, j)
            } else {
                r2 * x.get(i, j)
            };
            idx += 1;
        }
    }
}

/// `smat` into a caller matrix (every entry written — no zero-fill needed).
pub fn smat_into<T: Scalar>(v: &[T], x: &mut DenseMatrix<T>) {
    let k = x.nrows;
    let inv_r2 = T::one() / T::from_f64(2.0).expect("scalar literal").sqrt();
    let mut idx = 0;
    for j in 0..k {
        for i in 0..=j {
            let val = v[idx];
            idx += 1;
            if i == j {
                x.set(i, j, val);
            } else {
                let off = val * inv_r2;
                x.set(i, j, off);
                x.set(j, i, off);
            }
        }
    }
}

/// `mat_mul` into a caller matrix, mirroring [`mat_mul`]'s gemm gate exactly.
fn mat_mul_into<T: Scalar>(a: &DenseMatrix<T>, b: &DenseMatrix<T>, out: &mut DenseMatrix<T>) {
    let (n, m, p) = (a.nrows, a.ncols, b.ncols);
    if n >= 9 && p >= 9 {
        let c = iconic_linalg::faer_dense::dense_matmul(a, b);
        out.data_mut().copy_from_slice(&c.data);
    } else {
        for i in 0..n {
            for j in 0..p {
                let mut acc = T::zero();
                for k in 0..m {
                    acc += a.get(i, k) * b.get(k, j);
                }
                out.set(i, j, acc);
            }
        }
    }
}

/// [`apply_nt`] into a caller buffer.
pub fn apply_nt_into<T: Scalar>(
    w: &DenseMatrix<T>,
    v: &[T],
    out: &mut [T],
    scr: &mut PsdScratch<T>,
) {
    smat_into(v, &mut scr.x);
    mat_mul_into(w, &scr.x, &mut scr.y);
    mat_mul_into(&scr.y, w, &mut scr.z);
    svec_into(&scr.z, out);
}

/// [`jordan`] into a caller buffer.
pub fn jordan_into<T: Scalar>(u: &[T], v: &[T], out: &mut [T], scr: &mut PsdScratch<T>) {
    smat_into(u, &mut scr.x);
    smat_into(v, &mut scr.y);
    let k = scr.x.nrows;
    let half = T::from_f64(0.5).expect("scalar literal");
    mat_mul_into(&scr.x, &scr.y, &mut scr.z); // XY
    mat_mul_into(&scr.y, &scr.x, &mut scr.w); // YX
    for i in 0..k {
        for j in 0..k {
            scr.z.set(i, j, half * (scr.z.get(i, j) + scr.w.get(i, j)));
        }
    }
    svec_into(&scr.z, out);
}

/// [`arrow_inverse_apply`] into a caller buffer (the eigendecomposition is
/// inherent; the surrounding k² matrices come from the scratch).
pub fn arrow_inverse_apply_into<T: Scalar>(
    lam: &[T],
    b: &[T],
    out: &mut [T],
    scr: &mut PsdScratch<T>,
) {
    smat_into(lam, &mut scr.x); // lm
    let (d, q) = symmetric_eigh(&scr.x);
    arrow_inverse_apply_eig_into(&d, &q, b, out, scr);
}

/// [`arrow_inverse_apply`] into a caller buffer, with the eigenpair `(d, q)` of
/// `smat(lam)` supplied by the caller (e.g. from an [`ArrowEig`] cache) instead of
/// recomputed here. `d`/`q` must live outside `scr` (no aliasing). This reorders
/// the `smat_into(b, scr.y)` relative to the eigh — harmless, the two reads are
/// independent and the arithmetic is identical.
pub fn arrow_inverse_apply_eig_into<T: Scalar>(
    d: &[T],
    q: &DenseMatrix<T>,
    b: &[T],
    out: &mut [T],
    scr: &mut PsdScratch<T>,
) {
    smat_into(b, &mut scr.y); // bm
    let k = q.nrows;
    let two = T::from_f64(2.0).expect("scalar literal");

    // B̃ = Qᵀ B Q: qt in z, (qt·bm) in w, then ·q back into z... need qt for the final
    // step, so keep it in w's slot and chain the matmuls.
    // qt = Qᵀ in z.
    for i in 0..k {
        for j in 0..k {
            scr.z.set(i, j, q.get(j, i));
        }
    }
    // bt = (qt·bm)·q: stage 1 in w, stage 2 back into y (bm is dead after this).
    mat_mul_into(&scr.z, &scr.y, &mut scr.w);
    mat_mul_into(&scr.w, q, &mut scr.y);
    // w̃_ij = 2 b̃_ij / (d_i + d_j) — in place over the bt copy in y.
    for i in 0..k {
        for j in 0..k {
            scr.y.set(i, j, two * scr.y.get(i, j) / (d[i] + d[j]));
        }
    }
    // w = Q w̃ Qᵀ: (q·wt) in x (lm is dead), then ·qt into w.
    mat_mul_into(q, &scr.y, &mut scr.x);
    mat_mul_into(&scr.x, &scr.z, &mut scr.w);
    svec_into(&scr.w, out);
}

/// [`band_project`] into a caller buffer.
pub fn band_project_into<T: Scalar>(v: &[T], lo: T, hi: T, out: &mut [T], scr: &mut PsdScratch<T>) {
    smat_into(v, &mut scr.x);
    let (evals, q) = symmetric_eigh(&scr.x);
    let k = scr.x.nrows;
    for i in 0..k {
        for j in 0..k {
            let mut acc = T::zero();
            for l in 0..k {
                let e = evals[l];
                let clamped = if e < lo {
                    lo
                } else if e > hi {
                    hi
                } else {
                    e
                };
                acc += q.get(i, l) * clamped * q.get(j, l);
            }
            scr.y.set(i, j, acc);
        }
    }
    svec_into(&scr.y, out);
}

/// [`kron_solve`] into a caller buffer.
pub fn kron_solve_into<T: Scalar>(
    eig: &KronEig<T>,
    c: T,
    b: &[T],
    out: &mut [T],
    scr: &mut PsdScratch<T>,
) {
    smat_into(b, &mut scr.x); // bmat
    let k = scr.x.nrows;
    mat_mul_into(&eig.vt, &scr.x, &mut scr.y); // Vᵀ B
    mat_mul_into(&scr.y, &eig.v, &mut scr.z); // bp
    // The denominator `λᵢλⱼ + c` is invariant across RHS within an iteration
    // (kron_solve runs once per RHS: affine + corrector + ≤2 Gondzio) — one
    // reciprocal pass per solve instead of k² divisions each.
    for i in 0..k {
        for j in 0..k {
            scr.inv_den
                .set(i, j, (eig.lam[i] * eig.lam[j] + c).recip());
        }
    }
    for i in 0..k {
        for j in 0..k {
            scr.z.set(i, j, scr.z.get(i, j) * scr.inv_den.get(i, j));
        }
    }
    mat_mul_into(&eig.v, &scr.z, &mut scr.y); // V M'
    mat_mul_into(&scr.y, &eig.vt, &mut scr.x); // M
    svec_into(&scr.x, out);
}

/// The dense `(z,z)` block `H` (svec-dim × svec-dim) of congruence by `W`, with
/// `H · z = s`. Column `a` of `H` is `svec(W · smat(eₐ) · W)`; since `smat(eₐ)` is
/// `Eᵢᵢ` (diagonal) or `(Eᵢⱼ+Eⱼᵢ)/√2` (off-diagonal), the congruence collapses to an
/// outer product of columns of `W` — `O(k²)` per column, so the block is `O(k⁴)`
/// instead of the `O(k⁵)` of applying a full congruence per basis vector.
pub fn nt_block<T: Scalar>(w: &DenseMatrix<T>) -> DenseMatrix<T> {
    let k = w.nrows;
    let m = k * (k + 1) / 2;
    let mut h = DenseMatrix::zeros(m, m);
    let r2 = T::from_f64(2.0).expect("scalar literal").sqrt();
    // Pre-compute svec indices: idx[p][q] = svec position of (p,q) for p <= q.
    let mut svec_idx = vec![vec![0usize; k]; k];
    let mut pos = 0usize;
    for q in 0..k {
        for p in 0..=q {
            svec_idx[p][q] = pos;
            pos += 1;
        }
    }
    // Build each column a (corresponding to svec basis (i,j)) directly
    // via the svec formula, avoiding the intermediate k×k matrix and svec call.
    let mut a = 0usize;
    for j in 0..k {
        for i in 0..=j {
            if i == j {
                // svec(W[:,i] * W[:,i]ᵀ)
                for q in 0..k {
                    let wq = w.get(q, i);
                    for p in 0..=q {
                        let wp = w.get(p, i);
                        let v = if p == q { wp * wq } else { r2 * wp * wq };
                        h.set(svec_idx[p][q], a, v);
                    }
                }
            } else {
                // svec((W[:,i]*W[:,j]ᵀ + W[:,j]*W[:,i]ᵀ) / √2)
                for q in 0..k {
                    let wqi = w.get(q, i);
                    let wqj = w.get(q, j);
                    for p in 0..=q {
                        let wpi = w.get(p, i);
                        let wpj = w.get(p, j);
                        let v = if p == q {
                            r2 * wpi * wpj // √2 * W[p,i]*W[p,j]
                        } else {
                            wpi * wqj + wpj * wqi // W[p,i]*W[q,j] + W[p,j]*W[q,i]
                        };
                        h.set(svec_idx[p][q], a, v);
                    }
                }
            }
            a += 1;
        }
    }
    h
}

/// Jordan inverse-operator apply: the `w` (svec) with `λ ∘ w = b`, i.e. solve the
/// Lyapunov equation `½(Λw + wΛ) = B`. Diagonalizing `Λ = QDQᵀ` makes it elementwise:
/// `w̃_ij = 2 b̃_ij / (d_i + d_j)` in the eigenbasis. Requires `λ ≻ 0`.
pub fn arrow_inverse_apply<T: Scalar>(lam: &[T], b: &[T]) -> Vec<T> {
    let lm = smat(lam);
    let bm = smat(b);
    let (d, q) = symmetric_eigh(&lm);
    let k = lm.nrows;
    let two = T::from_f64(2.0).expect("scalar literal");

    // B̃ = Qᵀ B Q.
    let mut qt = DenseMatrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            qt.set(i, j, q.get(j, i));
        }
    }
    let bt = mat_mul(&mat_mul(&qt, &bm), &q);
    // w̃_ij = 2 b̃_ij / (d_i + d_j).
    let mut wt = DenseMatrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            wt.set(i, j, two * bt.get(i, j) / (d[i] + d[j]));
        }
    }
    // w = Q w̃ Qᵀ.
    svec(&mat_mul(&mat_mul(&q, &wt), &qt))
}

/// Project the eigenvalues of `smat(v)` into `[lo, hi]`, used by the Gondzio centrality
/// corrector: `Q·clamp(Λ)·Qᵀ` reconstructed from the eigendecomposition.
pub fn band_project<T: Scalar>(v: &[T], lo: T, hi: T) -> Vec<T> {
    let x = smat(v);
    let (evals, q) = symmetric_eigh(&x);
    let k = x.nrows;
    let clamped: Vec<T> = evals
        .iter()
        .map(|&e| {
            if e < lo {
                lo
            } else if e > hi {
                hi
            } else {
                e
            }
        })
        .collect();
    let mut out = DenseMatrix::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            let mut acc = T::zero();
            for l in 0..k {
                acc += q.get(i, l) * clamped[l] * q.get(j, l);
            }
            out.set(i, j, acc);
        }
    }
    svec(&out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svec_roundtrip_and_inner_product() {
        let x = DenseMatrix::from_row_major(2, 2, vec![2.0_f64, 1.0, 1.0, 3.0]);
        let y = DenseMatrix::from_row_major(2, 2, vec![1.0_f64, -1.0, -1.0, 4.0]);
        // Round-trip.
        let back = smat(&svec(&x));
        for i in 0..4 {
            assert!((back.data[i] - x.data[i]).abs() < 1e-12);
        }
        // ⟨svec X, svec Y⟩ = tr(XY).
        let sv = svec(&x);
        let sw = svec(&y);
        let ip: f64 = sv.iter().zip(&sw).map(|(&a, &b)| a * b).sum();
        let mut tr = 0.0;
        for i in 0..2 {
            for l in 0..2 {
                tr += x.get(i, l) * y.get(l, i);
            }
        }
        assert!((ip - tr).abs() < 1e-12, "ip={ip} tr={tr}");
    }

    #[test]
    fn membership_and_identity() {
        // [[2,1],[1,2]] ⪰ 0 (eigvalues 1,3); [[0,1],[1,0]] is indefinite.
        let pd = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![2.0_f64, 1.0, 1.0, 2.0],
        ));
        let indef = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![0.0_f64, 1.0, 1.0, 0.0],
        ));
        assert!(min_eig(&pd) > 0.5);
        assert!(min_eig(&indef) < -0.5);
        // X ∘ I = X.
        let e = identity::<f64>(3);
        let prod = jordan(&pd, &e);
        for i in 0..3 {
            assert!((prod[i] - pd[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn step_to_psd_boundary() {
        // From I along diag(0,-1): I + α diag(0,-1) ⪰ 0 until α = 1.
        let x = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![1.0_f64, 0.0, 0.0, 1.0],
        ));
        let dx = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![0.0_f64, 0.0, 0.0, -1.0],
        ));
        assert!((max_step(&x, &dx) - 1.0).abs() < 1e-9);
        // Moving deeper into the cone never exits.
        let dpos = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![1.0_f64, 0.0, 0.0, 2.0],
        ));
        assert!(max_step(&x, &dpos).is_infinite());
    }

    #[test]
    fn nt_scaling_maps_z_to_s() {
        // For S, Z ≻ 0, the PSD scaling block satisfies H z = s.
        let s = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![3.0_f64, 1.0, 1.0, 2.0],
        ));
        let z = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![2.0_f64, -0.5, -0.5, 4.0],
        ));
        let w = nt_scaling(&s, &z);
        // apply_nt(W, z) = svec(W Z W) = s.
        let hz = apply_nt(&w, &z);
        for i in 0..hz.len() {
            assert!(
                (hz[i] - s[i]).abs() < 1e-9,
                "H z ≠ s at {i}: {} vs {}",
                hz[i],
                s[i]
            );
        }
        // The assembled dense block agrees: H · z = s.
        let h = nt_block(&w);
        let hz2 = h.matvec(&z);
        for i in 0..hz2.len() {
            assert!((hz2[i] - s[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn kron_solve_matches_explicit_block() {
        // (W ⊗ₛ W + c·I)⁻¹ via the Kronecker structure equals the explicit dense solve.
        let s = svec(&DenseMatrix::from_row_major(
            3,
            3,
            vec![4.0_f64, 1.0, 0.5, 1.0, 3.0, -0.7, 0.5, -0.7, 2.5],
        ));
        let z = svec(&DenseMatrix::from_row_major(
            3,
            3,
            vec![2.0_f64, -0.3, 0.2, -0.3, 5.0, 0.4, 0.2, 0.4, 3.0],
        ));
        let w = nt_scaling(&s, &z);
        let c = 0.37_f64;
        let m = w.nrows * (w.nrows + 1) / 2;
        let b: Vec<f64> = (0..m).map(|i| 1.0 + 0.3 * i as f64).collect();
        let eig = KronEig::new(&w);
        let x = kron_solve(&eig, c, &b);
        // (nt_block(W) + cI) x should equal b.
        let h = nt_block(&w);
        let hx = h.matvec(&x);
        for i in 0..m {
            let r = hx[i] + c * x[i] - b[i];
            assert!(r.abs() < 1e-9, "residual {r} at {i}");
        }
    }

    #[test]
    fn arrow_inverse_undoes_jordan_product() {
        // λ ∘ (Arw(λ)⁻¹ b) = b for λ ≻ 0.
        let lam = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![3.0_f64, 1.0, 1.0, 2.0],
        ));
        let b = svec(&DenseMatrix::from_row_major(
            2,
            2,
            vec![1.0_f64, -0.5, -0.5, 2.0],
        ));
        let w = arrow_inverse_apply(&lam, &b);
        let back = jordan(&lam, &w);
        for i in 0..back.len() {
            assert!((back[i] - b[i]).abs() < 1e-9, "λ∘(Arw⁻¹b) ≠ b at {i}");
        }
    }
}
