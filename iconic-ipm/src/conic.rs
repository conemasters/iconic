//! Cone-aware interior-point solver over a product of symmetric cones.
//!
//! One Nesterov–Todd-scaled path handles LP, QP, SOCP, and SDP uniformly: the
//! nonnegative orthant is a product of 1-D second-order cones, and the per-cone
//! operations (scaling, Jordan product, arrow-inverse, step) dispatch on [`Cone`]
//! between the second-order cone ([`crate::soc`]) and the PSD cone ([`crate::psd`]).
//! The inequality cone product is given as a list of [`Cone`]s.
//!
//! Standard form: `min ½xᵀPx + qᵀx  s.t.  A_eq x = b_eq,  A_in x + s = b_in, s ∈ K`.

use crate::{grade_status, psd, soc};
use crate::{QpProblem, QpSolution};
use iconic_core::{Scalar, Settings, Status, WarmStart};
use iconic_linalg::{dot, inf_norm, min_degree, permute_upper, CscMatrix, DenseMatrix};

/// A cone in the inequality product. `NonNeg(d)` is the nonnegative orthant `ℝ₊^d`
/// (the batched form of `d` separate `Soc(1)` components — same math, but one vectorized
/// block with a diagonal `(z,z)` and no per-element allocation); `Soc(d)` is a second-order
/// cone of dimension `d`; `Psd(k)` is the cone of `k×k` symmetric positive-semidefinite
/// matrices, occupying `k(k+1)/2` slack entries (svec).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cone {
    NonNeg(usize),
    Soc(usize),
    Psd(usize),
}

impl Cone {
    /// Number of slack/dual entries this cone occupies.
    fn dim(&self) -> usize {
        match self {
            Cone::NonNeg(d) | Cone::Soc(d) => *d,
            Cone::Psd(k) => k * (k + 1) / 2,
        }
    }
    /// Barrier degree (`d` for the orthant `ℝ₊^d`, 1 per SOC, `k` per `k×k` PSD cone).
    fn degree(&self) -> usize {
        match self {
            Cone::NonNeg(d) => *d,
            Cone::Soc(_) => 1,
            Cone::Psd(k) => *k,
        }
    }
    /// Jordan-algebra identity of the cone (svec coordinates).
    fn identity<T: Scalar>(&self) -> Vec<T> {
        match self {
            Cone::NonNeg(d) => vec![T::one(); *d],
            Cone::Soc(d) => soc::identity(*d),
            Cone::Psd(_) => psd::identity(self.dim()),
        }
    }
    /// [`Self::jordan`] into a caller buffer.
    fn jordan_into<T: Scalar>(
        &self,
        u: &[T],
        v: &[T],
        out: &mut [T],
        scr: &mut psd::PsdScratch<T>,
    ) {
        match self {
            Cone::NonNeg(_) => {
                for i in 0..u.len() {
                    out[i] = u[i] * v[i];
                }
            }
            Cone::Soc(_) => soc::jordan_into(u, v, out),
            Cone::Psd(_) => psd::jordan_into(u, v, out, scr),
        }
    }
    /// [`Self::arrow_inv`] into a caller buffer.
    fn arrow_inv_into<T: Scalar>(
        &self,
        lam: &[T],
        b: &[T],
        out: &mut [T],
        scr: &mut psd::PsdScratch<T>,
    ) {
        match self {
            Cone::NonNeg(_) => {
                for i in 0..lam.len() {
                    out[i] = b[i] / lam[i];
                }
            }
            Cone::Soc(_) => soc::arrow_inverse_apply_into(lam, b, out),
            Cone::Psd(_) => psd::arrow_inverse_apply_into(lam, b, out, scr),
        }
    }
    /// [`Self::centrality_target`] into a caller buffer.
    fn centrality_target_into<T: Scalar>(
        &self,
        v: &[T],
        lo: T,
        hi: T,
        out: &mut [T],
        scr: &mut psd::PsdScratch<T>,
    ) {
        if let Cone::NonNeg(_) = self {
            for i in 0..v.len() {
                out[i] = v[i].max(lo).min(hi) - v[i];
            }
            return;
        }
        match self {
            Cone::Soc(_) => {
                soc::band_project_into(v, lo, hi, out);
                for i in 0..v.len() {
                    out[i] -= v[i];
                }
            }
            Cone::Psd(_) => {
                psd::band_project_into(v, lo, hi, out, scr);
                for i in 0..v.len() {
                    out[i] -= v[i];
                }
            }
            Cone::NonNeg(_) => unreachable!(),
        }
    }
}

/// Per-cone Nesterov–Todd scaling state.
enum ConeScaling<T: Scalar> {
    /// Nonnegative orthant: the per-component scaling `wᵢ = √(sᵢ/zᵢ)` (so `W = diag(w)`,
    /// `W² = diag(sᵢ/zᵢ)` is the `(z,z)` block — the batched form of each `Soc(1)`'s `η`).
    NonNeg { w: Vec<T> },
    /// SOC: `η` and the (det-1) scaling point `w̄`.
    Soc { eta: T, wbar: Vec<T> },
    /// PSD: the half-scaling `W^{1/2}` and its inverse `W^{-1/2}`.
    Psd {
        wh: DenseMatrix<T>,
        wih: DenseMatrix<T>,
    },
}

impl<T: Scalar> ConeScaling<T> {
    /// [`Self::apply_w`] into a caller buffer.
    fn apply_w_into(&self, v: &[T], out: &mut [T], scr: &mut psd::PsdScratch<T>) {
        match self {
            ConeScaling::NonNeg { w } => {
                // OVERWRITE, not accumulate: `out` is a reused per-iteration
                // scratch (zeroed once at setup, never re-zeroed), and the
                // SOC/PSD arms overwrite — an accumulate arm would compound
                // stale values across iterations and, on the first iteration,
                // multiply into the setup-time zeros (lam = 0·w = 0 breaks
                // the later Arw(λ)⁻¹ division).
                for i in 0..v.len() {
                    out[i] = w[i] * v[i];
                }
            }
            ConeScaling::Soc { eta, wbar } => {
                let r = soc::apply_w(*eta, wbar, v);
                out.copy_from_slice(&r);
            }
            ConeScaling::Psd { wh, .. } => psd::apply_nt_into(wh, v, out, scr),
        }
    }
    /// [`Self::apply_w_inv`] into a caller buffer.
    fn apply_w_inv_into(&self, v: &[T], out: &mut [T], scr: &mut psd::PsdScratch<T>) {
        match self {
            ConeScaling::NonNeg { w } => {
                // Overwrite, same contract as apply_w_into (see above).
                for i in 0..v.len() {
                    out[i] = v[i] / w[i];
                }
            }
            ConeScaling::Soc { eta, wbar } => {
                let r = soc::apply_w_inv(*eta, wbar, v);
                out.copy_from_slice(&r);
            }
            ConeScaling::Psd { wih, .. } => psd::apply_nt_into(wih, v, out, scr),
        }
    }
}

/// Prefix offsets of the cones in the slack vector.
fn cone_offsets(cones: &[Cone]) -> Vec<usize> {
    let mut off = Vec::with_capacity(cones.len());
    let mut acc = 0;
    for c in cones {
        off.push(acc);
        acc += c.dim();
    }
    off
}

/// Dense NT block `H = η²(2 w̄w̄ᵀ − J)` (row-major), the cone's `(z,z)` block.
fn nt_dense_block<T: Scalar>(eta_sq: T, wbar: &[T]) -> Vec<T> {
    let m = wbar.len();
    let two = T::from_f64(2.0).expect("scalar literal");
    let mut h = vec![T::zero(); m * m];
    for a in 0..m {
        for b in 0..m {
            let j = if a == b {
                if a == 0 {
                    T::one()
                } else {
                    -T::one()
                }
            } else {
                T::zero()
            };
            h[a * m + b] = eta_sq * (two * wbar[a] * wbar[b] - j);
        }
    }
    h
}

/// Largest joint step `α` keeping every cone's `v + α·dv` in its cone.
fn cone_step<T: Scalar>(
    cones: &[Cone],
    offsets: &[usize],
    v: &[T],
    dv: &[T],
    cache: &[Option<psd::InvSqrt<T>>],
) -> T {
    let mut a = T::infinity();
    for (c, cone) in cones.iter().enumerate() {
        let o = offsets[c];
        let d = cone.dim();
        let (vs, dvs) = (&v[o..o + d], &dv[o..o + d]);
        let step = match cone {
            // Orthant ratio test: max α with vᵢ + α·dvᵢ ≥ 0 for every i.
            Cone::NonNeg(_) => {
                let mut a = T::infinity();
                for (&vi, &dvi) in vs.iter().zip(dvs) {
                    if dvi < T::zero() {
                        a = a.min(-vi / dvi);
                    }
                }
                // Fixed boundary margin (as in soc::max_step / psd::max_step_cached):
                // keep the step strictly interior against rounding.
                let margin = T::from_f64(1e-13).expect("scalar literal");
                (a - margin).max(T::zero())
            }
            Cone::Soc(_) => soc::max_step(vs, dvs),
            // Reuse the precomputed X^{-1/2} of `v` when available (it's the same across
            // the several directions checked against the current iterate each step).
            Cone::Psd(_) => match &cache[c] {
                Some(inv) => psd::max_step_cached(inv, vs, dvs),
                None => psd::max_step(vs, dvs),
            },
        };
        if step < a {
            a = step;
        }
    }
    a
}

/// Precompute the PSD `X^{-1/2}` (and conditioning) for each cone at point `v`, so the
/// step-length checks within an iteration share one eigendecomposition per cone.
fn invsqrt_cache<T: Scalar>(
    cones: &[Cone],
    offsets: &[usize],
    v: &[T],
) -> Vec<Option<psd::InvSqrt<T>>> {
    cones
        .iter()
        .enumerate()
        .map(|(c, cone)| match cone {
            Cone::Psd(_) => {
                let o = offsets[c];
                Some(psd::invsqrt(&v[o..o + cone.dim()]))
            }
            Cone::NonNeg(_) | Cone::Soc(_) => None,
        })
        .collect()
}

/// Row-sparse (CSR) view of a dense `rows×cols` matrix, stored as a CSC of its transpose:
/// "column" r holds the nonzeros of row r as `(col-index, value)` in ascending column order,
/// so `colptr[r]..colptr[r+1]` indexes row r. Built once so the conic loop touches A in
/// O(nnz) — both the KKT assembly and the residual matvecs — instead of the O(rows·cols)
/// dense scans it otherwise repeats every iteration. (CVXPY hands us A sparse; this is where
/// ICONIC stops throwing that away.)
pub fn csr_of_dense<T: Scalar>(a: &DenseMatrix<T>, rows: usize, cols: usize) -> CscMatrix<T> {
    let zero = T::zero();
    let mut colptr = vec![0usize; rows + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = a.get(r, c);
            if v != zero {
                rowval.push(c);
                nzval.push(v);
            }
        }
        colptr[r + 1] = rowval.len();
    }
    CscMatrix {
        m: cols,
        n: rows,
        colptr,
        rowval,
        nzval,
    }
}

/// Push row `r` of A (its x-couplings) into the KKT column being built: from the prebuilt
/// row-CSR when available, else a dense scan. Both emit ascending column order, so the KKT
/// sparsity pattern is identical either way (symbolic reuse stays valid).
fn push_a_row<T: Scalar>(
    csr: Option<&CscMatrix<T>>,
    dense: &DenseMatrix<T>,
    r: usize,
    n: usize,
    rowval: &mut Vec<usize>,
    nzval: &mut Vec<T>,
) {
    match csr {
        Some(c) => {
            for p in c.colptr[r]..c.colptr[r + 1] {
                rowval.push(c.rowval[p]);
                nzval.push(c.nzval[p]);
            }
        }
        None => {
            for i in 0..n {
                let v = dense.get(r, i);
                if v != T::zero() {
                    rowval.push(i);
                    nzval.push(v);
                }
            }
        }
    }
}

/// Push column `c` of `P + ρI` (upper triangle) into the KKT being built.
/// `p_diag` skips the strictly-upper scan when P is known diagonal (O(n²)→O(n)).
pub(crate) fn push_p_col<T: Scalar>(
    p: &DenseMatrix<T>,
    c: usize,
    rho: T,
    p_diag: bool,
    rowval: &mut Vec<usize>,
    nzval: &mut Vec<T>,
    colptr: &mut [usize],
) {
    let zero = T::zero();
    if !p_diag {
        for i in 0..c {
            let v = p.get(i, c);
            if v != zero {
                rowval.push(i);
                nzval.push(v);
            }
        }
    }
    rowval.push(c);
    nzval.push(p.get(c, c) + rho);
    colptr[c + 1] = rowval.len();
}

/// Push the dual column for constraint row `r` (A-row couplings + diagonal
/// `diag_val`) at KKT index `col`.
pub(crate) fn push_dual_col<T: Scalar>(
    csr: Option<&CscMatrix<T>>,
    dense: &DenseMatrix<T>,
    r: usize,
    n: usize,
    col: usize,
    diag_val: T,
    rowval: &mut Vec<usize>,
    nzval: &mut Vec<T>,
    colptr: &mut [usize],
) {
    push_a_row(csr, dense, r, n, rowval, nzval);
    rowval.push(col);
    nzval.push(diag_val);
    colptr[col + 1] = rowval.len();
}

/// Cached sparse KKT pattern + static nzval template.  Built once before the IPM
/// loop; each iteration clones the template and overwrites only the dynamic entries
/// (ρ/δ on the diagonal, cone NT-scaling blocks).  Eliminates the per-iteration
/// O(nnz) scan of P/A_eq/A_in and the colptr/rowval allocation.
struct SparseKktCache<T: Scalar> {
    colptr: Vec<usize>,
    rowval: Vec<usize>,
    /// Static nzval template: all entries that don't change across iterations.
    nzval_static: Vec<T>,
    /// (position in nzval, column index) for x-block diagonal entries.
    x_diag: Vec<(usize, usize)>,
    /// (position in nzval, row index) for y-block diagonal entries.
    y_diag: Vec<(usize, usize)>,
    /// Per-cone dynamic entry positions: (position in nzval, block linear index).
    /// For NonNeg: block index is l.  For SOC/PSD: block index is lp * dimc + l.
    cone_dyn: Vec<Vec<(usize, usize)>>,
    dim: usize,
}

impl<T: Scalar> SparseKktCache<T> {
    /// Build the cache from a full KKT assembly.  The assembly writes every entry;
    /// we record the positions of dynamic entries as we go and save the static
    /// entries as a template.
    /// `kkt` is the PERMUTED KKT (permute_upper(kkt0, perm)); `perm[k]` is the
    /// ORIGINAL index sitting at permuted position k. All recorded positions are
    /// permuted nzval positions, but every recorded *index* (x-diag column, cone
    /// block linear index) must be an ORIGINAL index — `update_nzval` looks
    /// values up in the unpermuted `prob.p` and the per-cone blocks. Walking the
    /// permuted columns with the original index (the pre-fix version indexed
    /// `prob.p.get(c, c)` with the permuted column, and searched original
    /// z-columns `n+me+r` in the permuted matrix — both silently wrong for any
    /// non-trivial fill-reducing ordering; measured on the huber epigraph shape
    /// where the cached KKT differed from a fresh assembly by ~O(1) from the
    /// second iteration, exploding the Newton step and pinning the solve at
    /// SolvedInaccurate with a wrong objective).
    fn from_assembly(
        kkt: &CscMatrix<T>,
        perm: &[usize],
        cones: &[Cone],
        offsets: &[usize],
        n: usize,
        me: usize,
    ) -> Self {
        let dim = kkt.n;
        let mut nzval_static = kkt.nzval.clone();
        let mut x_diag = Vec::with_capacity(n);
        let mut y_diag = Vec::with_capacity(me);
        let mut cone_dyn: Vec<Vec<(usize, usize)>> = Vec::with_capacity(cones.len());

        // Walk every permuted column, dispatching on the ORIGINAL index
        // `perm[c]` — the fill-reducing ordering mixes the x/y/z blocks, so a
        // permuted column in [0, n) is not necessarily an x-column. The diagonal
        // of permuted column c is the permuted image of the original diagonal
        // (permutations preserve diagonal positions), so the position search
        // `rowval[p] == c` is unaffected; only the recorded *index* must be the
        // original one (`update_nzval` looks values up in the unpermuted
        // `prob.p` and the per-cone blocks).
        let mut cd_by_cone: Vec<Vec<(usize, usize)>> = vec![Vec::new(); cones.len()];
        for c in 0..dim {
            let o = perm[c];
            if o < n {
                // x-block diagonal.
                for p in kkt.colptr[c]..kkt.colptr[c + 1] {
                    if kkt.rowval[p] == c {
                        x_diag.push((p, o));
                        break;
                    }
                }
            } else if o < n + me {
                // y-block diagonal.
                for p in kkt.colptr[c]..kkt.colptr[c + 1] {
                    if kkt.rowval[p] == c {
                        y_diag.push((p, o - n));
                        break;
                    }
                }
            } else {
                // z-block / cone column. Which cone does original z-row
                // (o - n - me) belong to?
                let r = o - (n + me);
                let cidx = match offsets.binary_search(&r) {
                    Ok(i) => i,
                    Err(i) => i.saturating_sub(1),
                };
                let cone = &cones[cidx];
                let off = offsets[cidx];
                let dimc = cone.dim();
                let l = r - off;
                // Intra-cone entries: those whose ORIGINAL row is within this
                // cone's block. The A_in^T coupling entries have original row < n.
                for p in kkt.colptr[c]..kkt.colptr[c + 1] {
                    let orow = perm[kkt.rowval[p]];
                    if orow >= n + me + off {
                        if let Cone::NonNeg(_) = cone {
                            cd_by_cone[cidx].push((p, l));
                        } else {
                            let lp = orow - (n + me + off);
                            cd_by_cone[cidx].push((p, lp * dimc + l));
                        }
                        // The template's static part is stale at folded positions.
                        nzval_static[p] = T::zero();
                    }
                }
            }
        }
        for cd in cd_by_cone {
            cone_dyn.push(cd);
        }
        // Also clear x-diag and y-diag positions in the static template.
        for &(p, _) in &x_diag {
            nzval_static[p] = T::zero();
        }
        for &(p, _) in &y_diag {
            nzval_static[p] = T::zero();
        }

        SparseKktCache {
            colptr: kkt.colptr.clone(),
            rowval: kkt.rowval.clone(),
            nzval_static,
            x_diag,
            y_diag,
            cone_dyn,
            dim,
        }
    }

    /// Build a fresh nzval from the cached template and current dynamic values.
    fn update_nzval(
        &self,
        prob: &QpProblem<T>,
        cones: &[Cone],
        blocks: &[Vec<T>],
        rho: T,
        delta: T,
    ) -> Vec<T> {
        let mut nzval = self.nzval_static.clone();
        // x-block diagonals: P[c,c] + rho
        for &(p, c) in &self.x_diag {
            nzval[p] = prob.p.get(c, c) + rho;
        }
        // y-block diagonals: -delta
        for &(p, _) in &self.y_diag {
            nzval[p] = -delta;
        }
        // Cone blocks
        for (cidx, cd) in self.cone_dyn.iter().enumerate() {
            let cone = &cones[cidx];
            let blk = &blocks[cidx];
            if let Cone::NonNeg(_) = cone {
                for &(p, l) in cd {
                    nzval[p] = -(blk[l] + delta);
                }
            } else {
                let dimc = cone.dim();
                for &(p, idx) in cd {
                    let lp = idx / dimc;
                    let l = idx % dimc;
                    let hval = blk[lp * dimc + l];
                    nzval[p] = if lp == l { -(hval + delta) } else { -hval };
                }
            }
        }
        nzval
    }
}

/// Assemble the upper-triangular conic augmented KKT as CSC, ordered `[x, y, z]`,
/// with per-cone dense `(z,z)` blocks `−(H_c + δI)`.
#[allow(clippy::too_many_arguments)]
fn assemble_conic_kkt<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[Cone],
    offsets: &[usize],
    blocks: &[Vec<T>],
    rho: T,
    delta: T,
    aeq_csr: Option<&CscMatrix<T>>,
    ain_csr: Option<&CscMatrix<T>>,
    p_diag: bool,
) -> CscMatrix<T> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let dim = n + me + mi;

    let mut colptr = vec![0usize; dim + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    for c in 0..n {
        push_p_col(
            &prob.p,
            c,
            rho,
            p_diag,
            &mut rowval,
            &mut nzval,
            &mut colptr,
        );
    }
    for r in 0..me {
        push_dual_col(
            aeq_csr,
            &prob.a_eq,
            r,
            n,
            n + r,
            -delta,
            &mut rowval,
            &mut nzval,
            &mut colptr,
        );
    }
    for (cidx, cone) in cones.iter().enumerate() {
        let off = offsets[cidx];
        let dimc = cone.dim();
        for l in 0..dimc {
            let r = off + l;
            // A_inᵀ couplings to x.
            push_a_row(ain_csr, &prob.a_in, r, n, &mut rowval, &mut nzval);
            // Intra-cone block (upper triangle), −(H_c + δI). The orthant's block is
            // diagonal (η²ᵢ stored compactly); SOC/PSD blocks are kept structurally dense
            // so the pattern is fixed across iterations (symbolic reuse).
            if let Cone::NonNeg(_) = cone {
                rowval.push(n + me + r);
                nzval.push(-(blocks[cidx][l] + delta));
            } else {
                for lp in 0..=l {
                    let hval = blocks[cidx][lp * dimc + l];
                    let entry = if lp == l { -(hval + delta) } else { -hval };
                    rowval.push(n + me + off + lp);
                    nzval.push(entry);
                }
            }
            colptr[n + me + r + 1] = rowval.len();
        }
    }

    CscMatrix {
        m: dim,
        n: dim,
        colptr,
        rowval,
        nzval,
    }
}

/// Folded KKT assembly: eliminate foldable NonNeg rows, reducing the z-block.
#[allow(clippy::too_many_arguments)]
fn assemble_conic_kkt_folded<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[Cone],
    offsets: &[usize],
    blocks: &[Vec<T>],
    rho: T,
    delta: T,
    aeq_csr: Option<&CscMatrix<T>>,
    ain_csr: Option<&CscMatrix<T>>,
    p_diag: bool,
    fold_rows: &[(usize, usize)],
    fold_hinv: &[T],
) -> CscMatrix<T> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let mut is_folded = vec![false; mi];
    let mut fold_d = vec![zero; n];
    for i in 0..fold_rows.len() {
        let (r, c) = fold_rows[i];
        is_folded[r] = true;
        fold_d[c] += prob.a_in.get(r, c) * prob.a_in.get(r, c) * fold_hinv[i];
    }
    let mut z_map = vec![0usize; mi];
    let mut zc = 0usize;
    for r in 0..mi {
        if !is_folded[r] {
            z_map[r] = zc;
            zc += 1;
        }
    }
    let dim = n + me + zc;
    let mut colptr = vec![0usize; dim + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for c in 0..n {
        if !p_diag {
            for i in 0..c {
                let v = prob.p.get(i, c);
                if v != zero {
                    rowval.push(i);
                    nzval.push(v);
                }
            }
        }
        rowval.push(c);
        nzval.push(prob.p.get(c, c) + rho + fold_d[c]);
        colptr[c + 1] = rowval.len();
    }
    for r in 0..me {
        push_a_row(aeq_csr, &prob.a_eq, r, n, &mut rowval, &mut nzval);
        rowval.push(n + r);
        nzval.push(-delta);
        colptr[n + r + 1] = rowval.len();
    }
    for (cidx, cone) in cones.iter().enumerate() {
        let off = offsets[cidx];
        let dimc = cone.dim();
        for l in 0..dimc {
            let r = off + l;
            if is_folded[r] {
                continue;
            }
            let zr = z_map[r];
            push_a_row(ain_csr, &prob.a_in, r, n, &mut rowval, &mut nzval);
            if let Cone::NonNeg(_) = cone {
                rowval.push(n + me + zr);
                nzval.push(-(blocks[cidx][l] + delta));
            } else {
                for lp in 0..=l {
                    let hval = blocks[cidx][lp * dimc + l];
                    rowval.push(n + me + z_map[off + lp]);
                    nzval.push(if lp == l { -(hval + delta) } else { -hval });
                }
            }
            colptr[n + me + zr + 1] = rowval.len();
        }
    }
    CscMatrix {
        m: dim,
        n: dim,
        colptr,
        rowval,
        nzval,
    }
}

/// Assemble the **sparse** conic KKT with each second-order cone's `(z,z)` block in
/// *arrow form* — a diagonal plus two rank-1 auxiliary columns — instead of the dense
/// `−(H_c+δ)`. For a SOC `(η, w̄)`:
///   `−W² − δ = diag(−δ, −(η²+δ),…) + η²·e₀e₀ᵀ − 2η²·w̄w̄ᵀ`,
/// so two aux variables per cone carry the rank-1s: a `+1`-signed one coupling `√2η·w̄`
/// (spacelike), a `−1`-signed one coupling `η·e₀` (timelike). The KKT then has only a
/// diagonal `(z,z)` plus the (low-degree, min-degree-orders-them-last) aux columns, so it
/// stays sparse when `A_in` is sparse — the production case (a diagonal risk SOC). Returns
/// the upper-triangular CSC and the number of aux columns appended. Orthant (`Soc(1)`) cones
/// stay a plain `−(η²+δ)` diagonal entry (no aux).
#[allow(clippy::too_many_arguments)]
fn assemble_arrow_kkt<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[Cone],
    offsets: &[usize],
    sc: &[ConeScaling<T>],
    rho: T,
    delta: T,
    aeq_csr: Option<&CscMatrix<T>>,
    ain_csr: Option<&CscMatrix<T>>,
    p_diag: bool,
) -> (CscMatrix<T>, usize) {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();
    let one = T::one();
    let sqrt2 = T::from_f64(2.0).expect("scalar literal").sqrt();
    let zoff = n + me;
    let n_aux = cones
        .iter()
        .filter(|c| matches!(c, Cone::Soc(d) if *d > 1))
        .count()
        * 2;
    let dim = n + me + mi + n_aux;

    let mut colptr = vec![0usize; dim + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    // x columns: P + ρ (upper triangle).
    for c in 0..n {
        push_p_col(
            &prob.p,
            c,
            rho,
            p_diag,
            &mut rowval,
            &mut nzval,
            &mut colptr,
        );
    }
    // y columns: A_eqᵀ then −δ.
    for r in 0..me {
        push_dual_col(
            aeq_csr,
            &prob.a_eq,
            r,
            n,
            n + r,
            -delta,
            &mut rowval,
            &mut nzval,
            &mut colptr,
        );
    }
    // z columns: A_inᵀ then the arrow diagonal D_z (no intra-cone off-diagonal).
    for (cidx, cone) in cones.iter().enumerate() {
        let off = offsets[cidx];
        let d = cone.dim();
        for l in 0..d {
            let r = off + l;
            push_a_row(ain_csr, &prob.a_in, r, n, &mut rowval, &mut nzval);
            // Orthant: plain diagonal −(η²ᵢ+δ). SOC: arrow diagonal (apex row carries only
            // −δ; the η² and the rank-1s live on the two aux columns below).
            let diag = match &sc[cidx] {
                ConeScaling::NonNeg { w } => -(w[l] * w[l] + delta),
                ConeScaling::Soc { eta, .. } => {
                    if d > 1 && l == 0 {
                        -delta
                    } else {
                        -(*eta * *eta + delta)
                    }
                }
                ConeScaling::Psd { .. } => unreachable!("arrow path is SOC/orthant only"),
            };
            rowval.push(zoff + r);
            nzval.push(diag);
            colptr[zoff + r + 1] = rowval.len();
        }
    }
    // aux columns: per SOC with d>1, the two rank-1 carriers (z rows are above → upper part).
    let mut acol = n + me + mi;
    for (cidx, cone) in cones.iter().enumerate() {
        let off = offsets[cidx];
        let d = cone.dim();
        // Only multi-dimensional SOCs carry aux columns; the orthant and Soc(1) are diagonal.
        let (eta, wbar) = match &sc[cidx] {
            ConeScaling::Soc { eta, wbar } if d > 1 => (*eta, wbar),
            _ => continue,
        };
        // spacelike (+1): √2η·w̄ to all of z[off..off+d].
        for l in 0..d {
            rowval.push(zoff + off + l);
            nzval.push(sqrt2 * eta * wbar[l]);
        }
        rowval.push(acol);
        nzval.push(one);
        colptr[acol + 1] = rowval.len();
        acol += 1;
        // timelike (−1): η to z[off] (the apex).
        rowval.push(zoff + off);
        nzval.push(eta);
        rowval.push(acol);
        nzval.push(-one);
        colptr[acol + 1] = rowval.len();
        acol += 1;
    }

    (
        CscMatrix {
            m: dim,
            n: dim,
            colptr,
            rowval,
            nzval,
        },
        n_aux,
    )
}

/// Factorization of the condensed reduced system, which is negative definite. For
/// small reduced dimensions (≲175) ICONIC's scalar LDLᵀ is fastest (no faer thread-pool
/// overhead); for larger ones faer's SIMD Cholesky on the (PD) negation is preferred,
/// with LBLT as a fallback when the negation is not numerically positive definite.
enum ReducedFac {
    Scalar(iconic_linalg::ldl::LdlFactor<f64>),
    Chol(iconic_linalg::faer_dense::FaerLlt),
    /// Platform-BLAS Cholesky on the negated (PD) reduced system: OpenBLAS's
    /// multithreaded dpotrf measured 2.4x over faer's par_llt at 1280 dims.
    BlasChol {
        a: Vec<f64>,
        dim: usize,
    },
    Indef(iconic_linalg::faer_dense::FaerLblt),
    /// Platform-BLAS Bunch-Kaufman (dsytrf) on the indefinite reduced system.
    BlasLdlt {
        a: Vec<f64>,
        ipiv: Vec<i32>,
        dim: usize,
    },
}

/// A factorization of the conic KKT. Three backends behind one `solve(rhs) -> sol`
/// interface (full `[Δx; Δy; Δz]` in and out):
/// - `Sparse`: ICONIC's fill-reducing LDLᵀ, best for genuinely sparse systems.
/// - `Dense`: faer's SIMD LBLT on the full augmented KKT, for dense/general cones.
/// - `Condensed`: when `P+ρ` is diagonal, eliminate `Δx` by a Schur complement and
///   faer-factor only the smaller reduced `(y,z)` system — far cheaper when the cone
///   block is large (a big PSD cone), which is where the augmented factor is biggest.
enum Fac<T: Scalar> {
    Sparse(iconic_linalg::sparse_ldl::SparseLdl<T>),
    /// Sparse LDLᵀ on the arrow-form KKT (each SOC's `(z,z)` is a diagonal + 2 aux columns
    /// rather than a dense block), keeping the factor sparse for sparse-`A` SOCPs. The solve
    /// pads the RHS with `n_aux` zeros and drops the aux from the solution.
    ArrowSparse {
        factor: iconic_linalg::sparse_ldl::SparseLdl<T>,
        n_aux: usize,
    },
    Dense(iconic_linalg::faer_dense::FaerLblt),
    /// Unpivoted LDLᵀ on the (quasidefinite) augmented KKT — ~2–3× the LBLT, used whenever
    /// the unpivoted factor succeeds (it does unless a pivot is near-singular).
    DenseLdlt(iconic_linalg::faer_dense::FaerLdlt),
    /// Platform-BLAS (LAPACK `dsytrf`) Bunch–Kaufman on the augmented KKT:
    /// OpenBLAS multithreads it, which faer 0.24's LDLᵀ/LBLT cannot — the
    /// dense-LP routing's 1280-dim factor is ~165ms sequential today. The
    /// matrix is owned (cloned from the per-iteration work matrix, which the
    /// in-place factor cannot consume). `dim` = the system dimension.
    DenseBlasLdlt {
        a: Vec<f64>,
        ipiv: Vec<i32>,
        dim: usize,
    },
    /// ICONIC's scalar LDLᵀ for *tiny* dense KKTs: faer's per-call Mat allocation and thread
    /// dispatch cost more than the factor+solves themselves below ~dim 48, so a plain scalar
    /// factor (no allocation, no dispatch) wins there — the small-problem hot path.
    DenseScalar(iconic_linalg::ldl::LdlFactor<f64>),
    Condensed {
        reduced: ReducedFac,
        dinv: Vec<T>,
        /// Folded bound rows `(global_ineq_row, col, 1/(η²+δ))` — not in `reduced`.
        fold: Vec<(usize, usize, T)>,
        /// Global ineq rows kept in `reduced` (reduced row = `me + index`).
        kept: Vec<usize>,
        n: usize,
        me: usize,
        mi: usize,
    },
    /// O(k³) Kronecker solve for `X⪰0` SDPs transformed to `A_in = −I`: the reduced
    /// `(z,z)` block is `−(W⊗ₛW + cI)`, solved via `eig(W)`; the small `(y)` equality
    /// block is eliminated by a Schur complement (`schur`, with `gzy = (z,z)⁻¹(z,y)`).
    /// No `k²×k²` factor and no assembled NT block.
    Kronecker {
        eig: psd::KronEig<T>,
        c: T,
        schur: Option<iconic_linalg::ldl::LdlFactor<T>>,
        gzy: Vec<Vec<T>>,
        inv_rho: T,
        dinv: Vec<T>,
        n: usize,
        me: usize,
        mi: usize,
    },
}

/// Pre-allocated scratch for [`Fac::solve_into`] — the per-variant intermediate
/// buffers, sized on demand and reused across solves so a factor round allocates
/// nothing (the returned direction slices belong to the caller).
struct SolveScratch<T: Scalar> {
    rhs_p: Vec<T>,
    sol_p: Vec<T>,
    rhs_full: Vec<T>,
    sol_full: Vec<T>,
    rxp: Vec<T>,
    dx0: Vec<T>,
    aeq: Vec<T>,
    ain: Vec<T>,
    rr: Vec<f64>,
    syz: Vec<f64>,
    neg: Vec<f64>,
    aty: Vec<T>,
    atz: Vec<T>,
    rr_y: Vec<T>,
    rr_z: Vec<T>,
    gzz_rz: Vec<T>,
    rhs_y: Vec<T>,
    /// PSD cone-op scratch (k² matrices) for the Kronecker solve.
    psd: psd::PsdScratch<T>,
}

impl<T: Scalar> SolveScratch<T> {
    fn new() -> Self {
        SolveScratch {
            rhs_p: Vec::new(),
            sol_p: Vec::new(),
            rhs_full: Vec::new(),
            sol_full: Vec::new(),
            rxp: Vec::new(),
            dx0: Vec::new(),
            aeq: Vec::new(),
            ain: Vec::new(),
            rr: Vec::new(),
            syz: Vec::new(),
            neg: Vec::new(),
            aty: Vec::new(),
            atz: Vec::new(),
            rr_y: Vec::new(),
            rr_z: Vec::new(),
            gzz_rz: Vec::new(),
            rhs_y: Vec::new(),
            psd: psd::PsdScratch::new(1),
        }
    }
}

impl<T: Scalar> Fac<T> {
    fn solve_into(
        &self,
        prob: &QpProblem<T>,
        perm: &[usize],
        rhs: &[T],
        dx: &mut [T],
        dy: &mut [T],
        dz: &mut [T],
        scr: &mut SolveScratch<T>,
    ) {
        match self {
            Fac::Sparse(factor) => {
                let dim = rhs.len();
                scr.rhs_p.resize(dim, T::zero());
                scr.sol_p.resize(dim, T::zero());
                for (k, &p) in perm.iter().enumerate() {
                    scr.rhs_p[k] = rhs[p];
                }
                factor.solve_into(&scr.rhs_p, &mut scr.sol_p);
                let n = dx.len();
                let me = dy.len();
                for (k, &p) in perm.iter().enumerate() {
                    let v = scr.sol_p[k];
                    if p < n {
                        dx[p] = v;
                    } else if p < n + me {
                        dy[p - n] = v;
                    } else {
                        dz[p - n - me] = v;
                    }
                }
            }
            Fac::ArrowSparse { factor, n_aux } => {
                let base = rhs.len();
                let full = base + n_aux;
                scr.rhs_full.resize(full, T::zero());
                scr.rhs_full[..base].copy_from_slice(rhs);
                scr.rhs_p.resize(full, T::zero());
                scr.sol_p.resize(full, T::zero());
                for (k, &p) in perm.iter().enumerate() {
                    scr.rhs_p[k] = scr.rhs_full[p];
                }
                factor.solve_into(&scr.rhs_p, &mut scr.sol_p);
                scr.sol_full.resize(full, T::zero());
                for (k, &p) in perm.iter().enumerate() {
                    scr.sol_full[p] = scr.sol_p[k];
                }
                // First `base` entries are the true solution (the aux rows dropped).
                let n = dx.len();
                let me = dy.len();
                dx.copy_from_slice(&scr.sol_full[..n]);
                dy.copy_from_slice(&scr.sol_full[n..n + me]);
                dz.copy_from_slice(&scr.sol_full[n + me..base]);
            }
            // faer's solve allocates its result (inherent to its API); the copy into
            // the destination slices replaces the old split's extra Vecs.
            Fac::Dense(factor) => {
                let sol = crate::faer_solve_t(rhs, |r| factor.solve(r));
                let n = dx.len();
                let me = dy.len();
                dx.copy_from_slice(&sol[..n]);
                dy.copy_from_slice(&sol[n..n + me]);
                dz.copy_from_slice(&sol[n + me..]);
            }
            Fac::DenseLdlt(factor) => {
                let sol = crate::faer_solve_t(rhs, |r| factor.solve(r));
                let n = dx.len();
                let me = dy.len();
                dx.copy_from_slice(&sol[..n]);
                dy.copy_from_slice(&sol[n..n + me]);
                dz.copy_from_slice(&sol[n + me..]);
            }
            Fac::DenseBlasLdlt { a, ipiv, dim } => {
                let mut b: Vec<f64> = rhs.iter().map(|v| v.to_f64().expect("finite scalar")).collect();
                iconic_linalg::blas::dsytrs(*dim, a, ipiv, &mut b);
                let n = dx.len();
                let me = dy.len();
                for i in 0..n {
                    dx[i] = T::from_f64(b[i]).expect("scalar literal");
                }
                for i in 0..me {
                    dy[i] = T::from_f64(b[n + i]).expect("scalar literal");
                }
                for i in 0..dz.len() {
                    dz[i] = T::from_f64(b[n + me + i]).expect("scalar literal");
                }
            }
            Fac::DenseScalar(factor) => {
                let sol = crate::faer_solve_t(rhs, |r| factor.solve(r));
                let n = dx.len();
                let me = dy.len();
                dx.copy_from_slice(&sol[..n]);
                dy.copy_from_slice(&sol[n..n + me]);
                dz.copy_from_slice(&sol[n + me..]);
            }
            Fac::Condensed {
                reduced,
                dinv,
                fold,
                kept,
                n,
                me,
                mi,
            } => {
                let (n, me, mi) = (*n, *me, *mi);
                let zero = T::zero();
                let rx = &rhs[..n];
                let ry = &rhs[n..n + me];
                let rz = &rhs[n + me..n + me + mi];
                // Fold the bound rows into rx' = rx + Σ_fold a·(1/(η²+δ))·rz[row] at its column.
                scr.rxp.resize(n, zero);
                scr.rxp.copy_from_slice(rx);
                for &(row, col, hinv) in fold {
                    let a = prob.a_in.get(row, col);
                    scr.rxp[col] += a * hinv * rz[row];
                }
                // Reduced RHS [ry − A_eq D⁻¹ rx' ; rz_kept − A_kept D⁻¹ rx'], D the folded x-block.
                scr.dx0.resize(n, zero);
                for i in 0..n {
                    scr.dx0[i] = dinv[i] * scr.rxp[i];
                }
                scr.aeq.resize(me, zero);
                prob.a_eq.matvec_into(&scr.dx0, &mut scr.aeq);
                scr.ain.resize(mi, zero);
                prob.a_in.matvec_into(&scr.dx0, &mut scr.ain);
                let nk = kept.len();
                scr.rr.resize(me + nk, 0.0);
                for i in 0..me {
                    scr.rr[i] = (ry[i] - scr.aeq[i]).to_f64().expect("T → f64");
                }
                for (k, &row) in kept.iter().enumerate() {
                    scr.rr[me + k] = (rz[row] - scr.ain[row]).to_f64().expect("T → f64");
                }
                // Cholesky factored −reduced (PD), so solve it against −rr; LBLT directly.
                scr.syz.resize(me + nk, 0.0);
                match reduced {
                    ReducedFac::Scalar(ldl) => ldl.solve_into(&scr.rr, &mut scr.syz),
                    ReducedFac::Indef(lblt) => {
                        let syz = lblt.solve(&scr.rr);
                        scr.syz.copy_from_slice(&syz);
                    }
                    ReducedFac::Chol(llt) => {
                        scr.neg.resize(me + nk, 0.0);
                        for (i, &v) in scr.rr.iter().enumerate() {
                            scr.neg[i] = -v;
                        }
                        let syz = llt.solve(&scr.neg);
                        scr.syz.copy_from_slice(&syz);
                    }
                    ReducedFac::BlasChol { a, dim } => {
                        scr.neg.resize(me + nk, 0.0);
                        for (i, &v) in scr.rr.iter().enumerate() {
                            scr.neg[i] = -v;
                        }
                        iconic_linalg::blas::dpotrs(*dim, a, &mut scr.neg, 1);
                        scr.syz.copy_from_slice(&scr.neg);
                    }
                    ReducedFac::BlasLdlt { a, ipiv, dim } => {
                        scr.syz.resize(me + nk, 0.0);
                        scr.syz.copy_from_slice(&scr.rr);
                        iconic_linalg::blas::dsytrs(*dim, a, ipiv, &mut scr.syz);
                    }
                };
                // Scatter Δz_kept into the full Δz (folded rows still 0 here).
                dz.fill(zero);
                for (k, &row) in kept.iter().enumerate() {
                    dz[row] = T::from_f64(scr.syz[me + k]).expect("f64 → T");
                }
                for i in 0..me {
                    dy[i] = T::from_f64(scr.syz[i]).expect("f64 → T");
                }
                // Δx = D⁻¹(rx' − A_eqᵀΔy − A_inᵀΔz) (A_inᵀΔz touches only kept rows here).
                scr.aty.resize(n, zero);
                prob.a_eq.matvec_t_into(dy, &mut scr.aty);
                scr.atz.resize(n, zero);
                prob.a_in.matvec_t_into(dz, &mut scr.atz);
                for i in 0..n {
                    dx[i] = dinv[i] * (scr.rxp[i] - scr.aty[i] - scr.atz[i]);
                }
                // Recover the folded bound duals Δz[row] = (1/(η²+δ))·(a·Δx[col] − rz[row]).
                for &(row, col, hinv) in fold {
                    let a = prob.a_in.get(row, col);
                    dz[row] = hinv * (a * dx[col] - rz[row]);
                }
            }
            Fac::Kronecker {
                eig,
                c,
                schur,
                gzy,
                inv_rho,
                dinv,
                n,
                me,
                mi,
            } => {
                let (n, me, mi) = (*n, *me, *mi);
                let zero = T::zero();
                let rx = &rhs[..n];
                let ry = &rhs[n..n + me];
                let rz = &rhs[n + me..n + me + mi];
                // Reduced RHS: rr_y = ry − A_eq·D⁻¹·rx ; rr_z = rz − A_in·D⁻¹·rx, and with
                // A_in = −I that is rz + D⁻¹·rx.
                scr.dx0.resize(n, zero);
                for i in 0..n {
                    scr.dx0[i] = dinv[i] * rx[i];
                }
                scr.aeq.resize(me, zero);
                prob.a_eq.matvec_into(&scr.dx0, &mut scr.aeq);
                scr.rr_y.resize(me, zero);
                scr.rr_z.resize(mi, zero);
                for i in 0..me {
                    scr.rr_y[i] = ry[i] - scr.aeq[i];
                }
                for i in 0..mi {
                    scr.rr_z[i] = rz[i] + scr.dx0[i];
                }
                // Δy = S_y⁻¹ [rr_y − (y,z)·(z,z)⁻¹·rr_z], (y,z) = inv_rho·A_eq, with
                // (z,z)⁻¹ rr_z = −kron_solve (one solve).
                scr.gzz_rz.resize(mi, zero);
                let k = psd::side_dim(mi);
                // Lazy (re)size: the scratch persists across solve calls; only
                // reallocate when the cone side changes (never in the IPM loop).
                scr.psd.ensure_k(k);
                psd::kron_solve_into(eig, *c, &scr.rr_z, &mut scr.gzz_rz, &mut scr.psd);
                for i in 0..mi {
                    scr.gzz_rz[i] = -scr.gzz_rz[i];
                }
                if me > 0 {
                    prob.a_eq.matvec_into(&scr.gzz_rz, &mut scr.aeq);
                    scr.rhs_y.resize(me, zero);
                    for i in 0..me {
                        scr.rhs_y[i] = scr.rr_y[i] - *inv_rho * scr.aeq[i];
                    }
                    match schur {
                        Some(s) => s.solve_into(&scr.rhs_y, dy),
                        None => dy.fill(zero),
                    }
                } else {
                    dy.fill(zero);
                }
                // Δz = (z,z)⁻¹[rr_z − (z,y)Δy] = gzz_rz − Σⱼ gzy[j]·Δy[j] (gzy = (z,z)⁻¹(z,y)
                // is precomputed), so no second Kronecker solve is needed.
                dz.copy_from_slice(&scr.gzz_rz);
                for j in 0..me {
                    let dyj = dy[j];
                    for r in 0..mi {
                        dz[r] -= gzy[j][r] * dyj;
                    }
                }
                // Recover Δx = D⁻¹(rx − A_eqᵀΔy − A_inᵀΔz) = D⁻¹(rx − A_eqᵀΔy + Δz).
                scr.aty.resize(n, zero);
                prob.a_eq.matvec_t_into(dy, &mut scr.aty);
                for i in 0..n {
                    dx[i] = dinv[i] * (rx[i] - scr.aty[i] + dz[i]);
                }
            }
        }
    }
}

/// Build the reduced `(y,z)` KKT (as dense `f64`) for the condensed path, eliminating
/// the diagonal `x`-block `D = P+ρ`. Returns the reduced matrix and `D⁻¹`.
///
/// Reduced system:
/// `[ -δI − A_eq D⁻¹ A_eqᵀ      −A_eq D⁻¹ A_inᵀ          ] [Δy]`
/// `[ −A_in D⁻¹ A_eqᵀ        −(H+δI) − A_in D⁻¹ A_inᵀ    ] [Δz]`
/// Build the condensed reduced KKT, **folding bound cones into the x-block**.
///
/// A single-nonzero row of a NonNeg cone is a bound; its `(z,z)` is the scalar `η²`, so it
/// folds into the diagonal x-block `D = P+ρ` exactly as `solve_qp` folds bound rows — keeping
/// it OUT of the reduced. Without this, a box-constrained SOCP's reduced inflates to `me+mi`
/// (2n+SOC) and the box↔cone cross-terms force a dense factor; folding shrinks it to `me+SOC`.
/// Returns `(reduced, dinv, fold, kept)` where `fold[i] = (global_ineq_row, col, 1/(η²+δ))`
/// and `kept` are the global ineq rows left in the reduced (their reduced row is `me + index`).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
/// Pre-allocated buffers for the condensed-KKT build — sized once per solve from
/// the fixed problem structure, refilled every iteration by
/// [`build_condensed_into`]. The structural classifications (`fold_rc`/`kept`/
/// `redpos`/`general`/`units`) depend only on sparsity and cone type, never on
/// the per-iteration cone blocks, so they are computed once and reused. The big
/// `r` (rdim²) matrix is reused with a zero-fill at entry — the factorization
/// below is non-destructive (scalar LDLᵀ owns its storage), unlike the dense
/// augmented path.
struct CondensedScratch<T: Scalar> {
    rdim: usize,
    r: DenseMatrix<f64>,
    diag: Vec<T>,
    dinv: Vec<T>,
    redpos: Vec<usize>,
    /// Structural fold classification: (row, col) of each folded bound row.
    fold_rc: Vec<(usize, usize)>,
    /// Per-iteration `1/(η²+δ)` values for the folded rows.
    fold_hinv: Vec<T>,
    kept: Vec<usize>,
    general: Vec<usize>,
    units: Vec<(usize, usize)>,
    /// Dense-Schur scratch (rdim×n gram, g×n / n×g views, sparse-col scratch).
    gs: DenseMatrix<f64>,
    gg: DenseMatrix<f64>,
    ggt: DenseMatrix<f64>,
    col: Vec<(usize, f64)>,
}

impl<T: Scalar> CondensedScratch<T> {
    fn new(
        n: usize,
        me: usize,
        mi: usize,
        fold_rc: Vec<(usize, usize)>,
        kept: Vec<usize>,
        general: Vec<usize>,
        units: Vec<(usize, usize)>,
    ) -> Self {
        let rdim = me + kept.len();
        let g = general.len();
        let n_fold = fold_rc.len();
        let col_cap = me + kept.len();
        CondensedScratch {
            rdim,
            r: DenseMatrix::<f64>::zeros(rdim, rdim),
            diag: vec![T::zero(); n],
            dinv: vec![T::zero(); n],
            redpos: vec![usize::MAX; mi],
            fold_rc,
            fold_hinv: vec![T::zero(); n_fold],
            kept,
            general,
            units,
            gs: DenseMatrix::<f64>::zeros(rdim, n),
            gg: DenseMatrix::<f64>::zeros(g, n),
            ggt: DenseMatrix::<f64>::zeros(n, g),
            col: Vec::with_capacity(col_cap),
        }
    }
}

/// Build the condensed reduced KKT matrix into the caller's pre-allocated
/// [`CondensedScratch`], returning the `rdim×rdim` matrix (borrowed from the
/// scratch). The structural classifications were computed once; only the values
/// (cone blocks, ρ/δ, D⁻¹) are refilled here.
fn build_condensed_into<'a, T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[Cone],
    offsets: &[usize],
    blocks: &[Vec<T>],
    rho: T,
    delta: T,
    dense_schur: bool,
    sc: &'a mut CondensedScratch<T>,
) -> &'a DenseMatrix<f64> {
    let n = prob.q.len();
    let me = prob.b_eq.len();
    let f = |x: T| x.to_f64().expect("scalar to f64");
    let delta_f = f(delta);
    let rdim = sc.rdim;

    // D = P+ρ + Σ_fold a²/(η²+δ) (the bound gram, diagonal); dinv = 1/D.
    for i in 0..n {
        sc.diag[i] = prob.p.get(i, i) + rho;
    }
    for (k, &(row, col)) in sc.fold_rc.iter().enumerate() {
        sc.fold_hinv[k] = T::one() / (blocks[0][row] + delta);
        let a = prob.a_in.get(row, col);
        sc.diag[col] += a * a * sc.fold_hinv[k];
    }
    for i in 0..n {
        sc.dinv[i] = T::one() / sc.diag[i];
    }

    // Reduced rows: y at 0..me, kept ineq row `kept[k]` at me+k. `redpos` maps a
    // global ineq row to its reduced row (usize::MAX if folded).
    for (k, &row) in sc.kept.iter().enumerate() {
        sc.redpos[row] = me + k;
    }
    let r = &mut sc.r;
    // Zero-fill at entry: the assembly below relies on zeros for entries never
    // written (sparse-Schur mode), and the buffer is reused across iterations.
    let rd = r.data_mut();
    rd.fill(0.0);
    let rat = |a: usize, b: usize| a * rdim + b;
    // (z,z) base −(H_c+δI) for kept rows.
    for (cidx, cone) in cones.iter().enumerate() {
        let off = offsets[cidx];
        let d = cone.dim();
        if let Cone::NonNeg(_) = cone {
            for l in 0..d {
                let p = sc.redpos[off + l];
                if p != usize::MAX {
                    rd[rat(p, p)] = -(f(blocks[cidx][l]) + delta_f);
                }
            }
        } else {
            for l in 0..d {
                let pl = sc.redpos[off + l];
                for lp in 0..d {
                    let plp = sc.redpos[off + lp];
                    let hval = f(blocks[cidx][lp * d + l]);
                    let extra = if lp == l { delta_f } else { 0.0 };
                    rd[rat(plp, pl)] = -(hval + extra);
                }
            }
        }
    }
    for i in 0..me {
        rd[rat(i, i)] = -delta_f;
    }
    // Subtract the Schur [A_eq; A_kept] D⁻¹ [...]ᵀ over the reduced rows (gemm for
    // general rows, rank-1 for any kept single-nonzero rows). The general/units
    // classification is structural (fixed sparsity) — computed once at setup.
    let general = &sc.general;
    let units = &sc.units;
    if dense_schur {
        let gs = &mut sc.gs;
        let gsd = gs.data_mut();
        for k in 0..n {
            let sk = f(sc.dinv[k]).sqrt();
            for i in 0..me {
                gsd[i * n + k] = sk * f(prob.a_eq.get(i, k));
            }
            for (kk, &row) in sc.kept.iter().enumerate() {
                gsd[(me + kk) * n + k] = sk * f(prob.a_in.get(row, k));
            }
        }
        let g = general.len();
        if g > 0 {
            let gg = &mut sc.gg;
            let ggd = gg.data_mut();
            for (bi, &ri) in general.iter().enumerate() {
                for k in 0..n {
                    ggd[bi * n + k] = gsd[ri * n + k];
                }
            }
            let ggt = &mut sc.ggt;
            let ggt_d = ggt.data_mut();
            for bi in 0..g {
                for k in 0..n {
                    ggt_d[k * g + bi] = ggd[bi * n + k];
                }
            }
            let schur = iconic_linalg::faer_dense::dense_matmul(gg, ggt);
            for (bi, &ri) in general.iter().enumerate() {
                for (bj, &rj) in general.iter().enumerate() {
                    rd[rat(ri, rj)] -= schur.get(bi, bj);
                }
            }
        }
        for &(ru, c) in units {
            let gu = gsd[ru * n + c];
            for j in 0..rdim {
                rd[rat(ru, j)] -= gu * gsd[j * n + c];
            }
        }
        for &(ru, c) in units {
            let gu = gsd[ru * n + c];
            for &gj in general {
                rd[rat(gj, ru)] -= gsd[gj * n + c] * gu;
            }
        }
    } else {
        for k in 0..n {
            let dk = f(sc.dinv[k]);
            sc.col.clear();
            for i in 0..me {
                let v = f(prob.a_eq.get(i, k));
                if v != 0.0 {
                    sc.col.push((i, v));
                }
            }
            for (kk, &row) in sc.kept.iter().enumerate() {
                let v = f(prob.a_in.get(row, k));
                if v != 0.0 {
                    sc.col.push((me + kk, v));
                }
            }
            for &(ri, vi) in &sc.col {
                for &(ci, vci) in &sc.col {
                    rd[rat(ri, ci)] -= dk * vi * vci;
                }
            }
        }
    }
    r
}

/// If `a_in` is a generalized permutation (square; exactly one nonzero per row and per
/// column), return `(σ, g)` with `σ(i)` the nonzero column of row `i` and `g[i] =
/// −a_in[i][σ(i)]` (so `G = −a_in` has `G[i][σ(i)] = g[i]`). Otherwise `None`.
fn detect_gen_perm<T: Scalar>(a_in: &DenseMatrix<T>) -> Option<(Vec<usize>, Vec<T>)> {
    let (m, nn) = (a_in.nrows, a_in.ncols);
    if m != nn || m == 0 {
        return None;
    }
    let zero = T::zero();
    let mut sigma = vec![0usize; m];
    let mut g = vec![zero; m];
    let mut col_used = vec![false; nn];
    for i in 0..m {
        let mut nz: Option<(usize, T)> = None;
        for j in 0..nn {
            let v = a_in.get(i, j);
            if v != zero {
                if nz.is_some() {
                    return None;
                }
                nz = Some((j, v));
            }
        }
        let (j, v) = nz?;
        if col_used[j] {
            return None;
        }
        col_used[j] = true;
        sigma[i] = j;
        g[i] = -v;
    }
    Some((sigma, g))
}

/// Detect the structure that admits the O(k³) Kronecker SDP solve — a single PSD cone,
/// zero Hessian, and an `A_in` that is a generalized permutation — and return the
/// problem changed to variables `x̃ = −A_in·x` (so `A_in' = −I`), together with `(σ, g)`
/// to undo it. The objective is preserved (`q̃ = G⁻ᵀ q`), as are `b`, `s`, and `z`.
fn kron_transform<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[Cone],
) -> Option<(QpProblem<T>, Vec<usize>, Vec<T>)> {
    if cones.len() != 1 || !matches!(cones[0], Cone::Psd(_)) {
        return None;
    }
    let n = prob.q.len();
    let mi = prob.b_in.len();
    let me = prob.b_eq.len();
    let zero = T::zero();
    if mi != n {
        return None;
    }
    for i in 0..n {
        for j in 0..n {
            if prob.p.get(i, j) != zero {
                return None;
            }
        }
    }
    let (sigma, g) = detect_gen_perm(&prob.a_in)?;
    // q̃[a] = q[σ(a)]/g[a] ; A_eq̃[r][a] = A_eq[r][σ(a)]/g[a] ; A_in' = −I.
    let qt: Vec<T> = (0..n).map(|a| prob.q[sigma[a]] / g[a]).collect();
    let mut aeqt = DenseMatrix::zeros(me, n);
    for r in 0..me {
        for a in 0..n {
            aeqt.set(r, a, prob.a_eq.get(r, sigma[a]) / g[a]);
        }
    }
    let mut aint = DenseMatrix::zeros(mi, n);
    for i in 0..mi {
        aint.set(i, i, -T::one());
    }
    let prob_t = QpProblem {
        p: DenseMatrix::zeros(n, n),
        q: qt,
        a_eq: aeqt,
        b_eq: prob.b_eq.clone(),
        a_in: aint,
        b_in: prob.b_in.clone(),
        a_eq_csr: None,
        a_in_csr: None,
    };
    Some((prob_t, sigma, g))
}

/// Warm-start seed validation + interiorization for the conic engine (M8).
///
/// Returns the seeded `(x, s, z)` — `s` derived as `b − A·x` (so the
/// `A x + s = b` invariant holds exactly) — or `None` when the seed fails any
/// check and the caller must use the cold start (bit-identical to the unseeded
/// path; a warm start can only change convergence speed, never the converged
/// point):
///   1. dimensions match, all entries finite;
///   2. `A·x + s = b` to 1e-6 relative (a previous solution satisfies it to
///      solver tolerance — this guards against a stale or mis-dimensioned
///      seed, e.g. from a cached solve of a differently-shaped problem);
///   3. per cone, the seeded `s` and `z` are strictly interior after blending
///      each block toward the cone identity with θ = 0.05 when its margin sits
///      below 1e-3·scale — optima are boundary-active, so a raw seed sits on
///      the cone boundary, which the NT step-to-boundary cannot start from. A
///      seed grossly outside a cone stays outside after the blend and fails.
///
/// The dual margin test is the same functional as the primal one: NonNeg is
/// self-dual, SOC is self-dual, and the PSD cone is its own dual.
/// Shared warm-seed precheck for both engines: dimensions + finiteness, the
/// `A·x + s = b` invariant on the inequality rows (1e-6 relative), and the
/// exact recomputation `s = b − A·x`. Returns the seed's `(x, s, z)` ready for
/// per-cone interiorization, or `None` when the seed is stale/mis-dimensioned
/// (the caller falls back to the cold start).
pub(crate) fn warm_seed_primal<T: Scalar>(
    prob: &QpProblem<T>,
    init: Option<&WarmStart<T>>,
) -> Option<(Vec<T>, Vec<T>, Vec<T>)> {
    let ws = init?;
    let n = prob.q.len();
    let mi = prob.b_in.len();
    let zero = T::zero();
    let one = T::one();
    let from = |v: f64| T::from_f64(v).expect("scalar literal");

    // Dimensions + finiteness.
    let mut ok = ws.x.len() == n && ws.s.len() == mi && ws.z.len() == mi;
    if ok {
        for v in ws.x.iter().chain(ws.s.iter()).chain(ws.z.iter()) {
            if !v.is_finite() {
                ok = false;
                break;
            }
        }
    }
    if !ok {
        return None;
    }
    // A·x + s = b (guards against a stale or mis-dimensioned seed).
    if mi > 0 {
        let mut s_scale = one;
        for i in 0..mi {
            s_scale = s_scale.max(prob.b_in[i].abs()).max(ws.s[i].abs());
        }
        let tol = from(1e-6) * s_scale;
        for r in 0..mi {
            let mut acc = zero;
            for j in 0..n {
                acc += prob.a_in.get(r, j) * ws.x[j];
            }
            if (acc + ws.s[r] - prob.b_in[r]).abs() > tol {
                return None;
            }
        }
    }
    let x = ws.x.clone();
    let mut s = vec![zero; mi];
    let z = ws.z.clone();
    // s = b − A·x (exact inequality feasibility).
    for r in 0..mi {
        let mut acc = zero;
        for j in 0..n {
            acc += prob.a_in.get(r, j) * x[j];
        }
        s[r] = prob.b_in[r] - acc;
    }
    Some((x, s, z))
}

fn conic_warm_seed<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[Cone],
    offsets: &[usize],
    init: Option<&WarmStart<T>>,
) -> Option<(Vec<T>, Vec<T>, Vec<T>)> {
    let (x, mut s, mut z) = warm_seed_primal(prob, init)?;
    let one = T::one();
    let from = |v: f64| T::from_f64(v).expect("scalar literal");
    let theta = from(0.05);
    let eps_margin = from(1e-3);

    // Per-cone interiorization of s/z toward the cone identity.
    for (c, cone) in cones.iter().enumerate() {
        let o = offsets[c];
        let d = cone.dim();
        let e = cone.identity::<T>();
        // Relative scale of a block (floor 1) and its interior margin. The dual
        // cone margin is the same functional (all three cones are self-dual).
        let scale = |blk: &[T]| blk.iter().fold(one, |a, &v| a.max(v.abs()));
        let margin = |blk: &[T]| match cone {
            Cone::NonNeg(_) => blk.iter().cloned().fold(T::infinity(), |a, v| a.min(v)),
            Cone::Soc(_) => soc::margin(blk),
            Cone::Psd(_) => psd::min_eig(blk),
        };
        let s_block = &s[o..o + d];
        if margin(s_block) < eps_margin * scale(s_block) {
            for i in 0..d {
                s[o + i] = (one - theta) * s[o + i] + theta * e[i];
            }
        }
        let z_block = &z[o..o + d];
        if margin(z_block) < eps_margin * scale(z_block) {
            for i in 0..d {
                z[o + i] = (one - theta) * z[o + i] + theta * e[i];
            }
        }
        // Strict interiority after the blend, or cold start.
        let tol_i = from(1e-9) * scale(&s[o..o + d]).max(scale(&z[o..o + d]));
        if margin(&s[o..o + d]) <= tol_i || margin(&z[o..o + d]) <= tol_i {
            return None;
        }
    }
    Some((x, s, z))
}

/// Solve a QP/SOCP over the product cone `cones` (cone dimensions summing to
/// `m_in`; dimension 1 = a nonnegative component) with a Nesterov–Todd-scaled,
/// sparse, fill-reduced interior-point method, from the cold start. Thin
/// wrapper over [`solve_cone_qp_warm`] with no seed — all existing callers are
/// unchanged.
pub fn solve_cone_qp<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[Cone],
    settings: &Settings<T>,
) -> QpSolution<T> {
    solve_cone_qp_warm(prob, cones, settings, None)
}

/// Solve the conic QP, optionally seeded from a previous near-solution.
///
/// The seed is validated (dimensions, finiteness, the `A x + s = b` invariant,
/// and per-cone strict interiority of `s`/`z` after a θ-blend toward the cone
/// identity); on any failure the solve falls back to the cold start, which is
/// bit-identical to calling [`solve_cone_qp`] — a warm start can only change
/// convergence speed, never the converged point. The Kronecker SDP path changes
/// variables (`x̃ = g∘x∘σ`), so a seed does not map there; it is ignored
/// (cold start) until the phase-2 structural warm start.
pub fn solve_cone_qp_warm<T: Scalar>(
    prob: &QpProblem<T>,
    cones: &[Cone],
    settings: &Settings<T>,
    init: Option<&WarmStart<T>>,
) -> QpSolution<T> {
    // Detect the X⪰0 SDP structure and, if present, change variables to A_in = −I so the
    // reduced (z,z) becomes Kronecker-structured (solved in O(k³)); `kron` carries (σ, g)
    // to undo the transform on the recovered primal `x`.
    let kron = kron_transform(prob, cones);
    let use_kron = kron.is_some();
    let prob: &QpProblem<T> = kron.as_ref().map(|(pt, _, _)| pt).unwrap_or(prob);

    let n = prob.q.len();
    let me = prob.b_eq.len();
    let mi = prob.b_in.len();

    // Dual simplex for SPARSE pure LP: O(m²)/pivot beats IPM O(n³)/factor only
    // when the constraint matrix is sparse enough for cheap pivots.  Dense A_in
    // falls through to the IPM which uses BLAS-accelerated dense factorisation.
    let all_nonneg = cones.iter().all(|c| matches!(c, Cone::NonNeg(_)));
    let p_is_zero = prob.p.data.iter().all(|&v| v == T::zero());
    let a_in_is_sparse = {
        let total = (mi * n) as f64;
        if total == 0.0 {
            true
        } else {
            let nz = prob.a_in.data.iter().filter(|&&v| v != T::zero()).count() as f64;
            nz < 0.15 * total
        }
    };
    // The dual simplex is only worth its O(m²)/pivot cost on genuinely large
    // sparse LPs: on small sparse LPs the IPM solves in ~10 iterations while the
    // simplex grinds out 150-900 pivots (measured on the L1-fit family: 156
    // pivots at n=72 vs 11 IPM iters, 425 vs 12 at n=140). Route those through
    // the IPM instead.
    if all_nonneg
        && p_is_zero
        && me == 0
        && mi > 0
        && mi <= 400
        && n <= 2000
        && a_in_is_sparse
        && n >= 400
    {
        if let Some(sol) = crate::try_sparse_lp_dual_simplex(prob, n, mi) {
            return sol;
        }
    }

    let setup_t0 = std::time::Instant::now();
    let offsets = cone_offsets(cones);
    // Run faer sequentially for small/medium KKTs, where the Rayon thread-pool dispatch costs
    // more than the parallelism saves; let it parallelize the large factors.
    iconic_linalg::faer_dense::set_parallelism_seq(n + me + mi < 256);

    let from = |v: f64| T::from_f64(v).expect("scalar literal");
    let zero = T::zero();
    let one = T::one();
    // P is diagonal for a linear / sum-of-squares objective (the MPC, most routed QPs); detect
    // it once so the per-iteration residual matvec and the KKT assembly both skip P's (empty)
    // off-diagonal, turning their O(n²) P reads into O(n).
    let p_diagonal = use_kron || (0..n).all(|i| (0..n).all(|j| i == j || prob.p.get(i, j) == zero));
    // SOC cones have denser NT scaling that benefits from stronger proximal
    // regularization — without it the IPM can stall on Woodbury-reformulated
    // portfolio problems with large second-order cones.
    let has_soc = cones.iter().any(|c| matches!(c, Cone::Soc(_)));
    // Condition-aware regularization baseline, mirroring the QP path's
    // recently-tightened logic (commit 2f51a89): cones with dense Hessian
    // blocks (SDP, exp, power) provide their own (z,z)-block curvature, so
    // ρ starts at machine-epsilon scale and only escalates on actual
    // degradation. SOC cones keep the existing stronger baseline (1e-6)
    // because their NT scaling can be near-singular at the boundary.
    let has_dense_hessian = cones.iter().any(|c| matches!(c, Cone::Psd(_)));
    let has_only_nonneg = cones.iter().all(|c| matches!(c, Cone::NonNeg(_)));
    let rho0: T = if has_soc {
        from(1e-6)
    } else if has_dense_hessian {
        // PSD: the dense NT block can be ill-conditioned when W has extreme
        // eigenvalues near the boundary.  Use the same floor as SOC (1e-6) —
        // ICONIC's augmented-KKT architecture needs a meaningful baseline:
        // the LDLᵀ operates on a quasidefinite system with separate (1,1)
        // and (2,2) blocks rather than a single PD reduced matrix, so a
        // near-zero primal regularization (viable only on a normal-equations
        // reduced matrix) does not apply here.  Empirically tuned on
        // ICONIC's benchmark suite (QP/SOCP/SDP).
        from(1e-6)
    } else if has_only_nonneg {
        from(1e-6) // LP: ρ IS the only (1,1) curvature
    } else {
        from(1e-7) // mixed/small cones
    };
    // SOC and PSD both need more dual regularization: the dense NT block
    // is ill-conditioned near the boundary and needs a larger δ floor.
    let delta0: T = if has_soc || has_dense_hessian {
        from(1e-6)
    } else {
        from(1e-7)
    };
    // Regularization ladder: discrete levels with aggressive
    // de-escalation (10× residual improvement → one level down). Escalation
    // is reserved for genuine numerical blow-up (mu runaway).
    const REG_LADDER: [f64; 6] = [0.0, 1e-13, 1e-10, 1e-8, 1e-6, 1e-4];
    let mut reg_level = (0..REG_LADDER.len())
        .rev()
        .find(|&i| REG_LADDER[i] <= rho0.to_f64().expect("finite scalar"))
        .unwrap_or(0);
    let mut rho = T::from_f64(REG_LADDER[reg_level]).expect("finite scalar").max(rho0);
    let mut delta = T::from_f64(REG_LADDER[reg_level]).expect("finite scalar").max(delta0);
    // Fraction-to-boundary factor: 0.999, slightly more conservative than the 0.9999 typical
    // of LP-only IPMs, to stay safer near SOC/PSD boundaries. The Gondzio quality metric
    // catches bad-corrected steps.
    let eta_ftb = from(0.999);
    let pivot_tol = from(1e-14);
    let eps = settings.eps_abs;
    let big = from(1e12);
    let degree = from(cones.iter().map(|c| c.degree()).sum::<usize>().max(1) as f64);

    // Warm start (M8): seed x/s/z from a previous near-solution. This engine's
    // regularization is factorization-side (ρI/δI, no stored proximal-reference
    // vectors), so seeding the iterates at the previous solution IS the warm
    // start. The seed is validated and interiorized per cone (see
    // `conic_warm_seed`); on any failure `warm` is None and the cold
    // construction below — bit-identical to the unseeded path — runs instead.
    // The Kronecker SDP path changes variables, so the seed is ignored there.
    let warm = conic_warm_seed(prob, cones, &offsets, if use_kron { None } else { init });

    // Initial iterate: x = 0, y = 0, z = per-cone identity. The slack `s` starts at the
    // *natural* slack `b_in − A_in·0 = b_in` shifted strictly inside each second-order cone,
    // which makes the initial primal residual `b_in − A_in·0 − s` zero — avoiding the early μ
    // blow-up the arbitrary identity start causes on infeasible-start SOCPs. PSD cones keep the
    // identity (the Kronecker SDP path already converges fast).
    // With a validated warm start the iterates are the interiorized seed instead:
    // `s = b − A·x` (exact primal feasibility), per-cone-blended toward the
    // identity where the seed sits on a cone boundary, and the seeded dual `z`.
    let mut x = warm
        .as_ref()
        .map(|w| w.0.clone())
        .unwrap_or_else(|| vec![zero; n]);
    let mut y = vec![zero; me];
    let mut s = vec![zero; mi];
    let mut z = vec![zero; mi];
    if let Some((_, ws_s, ws_z)) = &warm {
        s.copy_from_slice(ws_s);
        z.copy_from_slice(ws_z);
    } else {
        for (c, cone) in cones.iter().enumerate() {
            let o = offsets[c];
            let d = cone.dim();
            let e = cone.identity::<T>();
            z[o..o + d].copy_from_slice(&e);
            match cone {
                Cone::NonNeg(_) => {
                    // Natural slack, each component shifted to ≥ 1 (the Soc(1) apex shift).
                    for i in 0..d {
                        let si = prob.b_in[o + i];
                        s[o + i] = if si < one { one } else { si };
                    }
                }
                Cone::Soc(_) => {
                    let mut sc: Vec<T> = prob.b_in[o..o + d].to_vec();
                    let nrm = sc[1..].iter().fold(zero, |a, &v| a + v * v).sqrt();
                    let margin = sc[0] - nrm;
                    if margin < one {
                        sc[0] += one - margin; // shift the apex so the SOC margin is ≥ 1
                    }
                    s[o..o + d].copy_from_slice(&sc);
                }
                Cone::Psd(_) => s[o..o + d].copy_from_slice(&e),
            }
        }
    }

    // Per-cone NT scaling and the dense (z,z) blocks. Recomputed each iteration. For the
    // Kronecker path, also returns eig(W) of the (single) PSD cone, reused by the factor.
    let scale = |s: &[T], z: &[T]| {
        let mut sc: Vec<ConeScaling<T>> = Vec::with_capacity(cones.len());
        let mut blocks = Vec::with_capacity(cones.len());
        let mut kron_eig: Option<psd::KronEig<T>> = None;
        for (c, cone) in cones.iter().enumerate() {
            let o = offsets[c];
            let d = cone.dim();
            match cone {
                Cone::NonNeg(_) => {
                    // Each component is a Soc(1) with η²ᵢ = sᵢ/zᵢ; the (z,z) block is the
                    // diagonal of those η²ᵢ (stored compactly, d entries not d²).
                    let (mut w, mut diag) = (vec![zero; d], vec![zero; d]);
                    for i in 0..d {
                        let eta_sq = s[o + i] / z[o + i];
                        diag[i] = eta_sq;
                        w[i] = eta_sq.sqrt();
                    }
                    blocks.push(diag);
                    sc.push(ConeScaling::NonNeg { w });
                }
                Cone::Soc(_) => {
                    let (eta_sq, wbar) = soc::nt_scaling(&s[o..o + d], &z[o..o + d]);
                    blocks.push(nt_dense_block(eta_sq, &wbar));
                    sc.push(ConeScaling::Soc {
                        eta: eta_sq.sqrt(),
                        wbar,
                    });
                }
                Cone::Psd(_) => {
                    let w = psd::nt_scaling(&s[o..o + d], &z[o..o + d]);
                    if use_kron {
                        // Skip the O(k⁴) NT block (unused); get W^{±1/2} and eig(W) from
                        // one eigendecomposition.
                        blocks.push(Vec::new());
                        let (wh, wih, eig) = psd::scaling_halves_and_eig(&w);
                        sc.push(ConeScaling::Psd { wh, wih });
                        kron_eig = Some(eig);
                    } else {
                        blocks.push(psd::nt_block(&w).into_vec());
                        let (wh, wih) = psd::scaling_halves(&w);
                        sc.push(ConeScaling::Psd { wh, wih });
                    }
                }
            }
        }
        (sc, blocks, kron_eig)
    };

    let dim = n + me + mi;
    // Pick the factorization backend from a *cheap* density proxy (KKT nonzeros over
    // the upper-triangle area) — crucially without first computing the fill-reducing
    // ordering, which is itself very expensive on a dense pattern. A dense Hessian or
    // a large PSD cone fills the KKT, so its density is high → faer's SIMD dense LBLT
    // (no ordering needed, it pivots internally); genuinely sparse systems stay on the
    // fill-reducing sparse LDLᵀ. The Kronecker path assembles no KKT, so skip the proxy.
    // Build A's row-CSRs once (whenever a KKT is assembled) and reuse them everywhere — the
    // density proxy below, the symbolic analyze, and the per-iteration assembly/matvecs — so A
    // is scanned O(m·n) at most once instead of separately by the proxy, the analyze, and the
    // first assembly. CVXPY hands A sparse; this is where ICONIC stops rescanning it dense.
    let (ain_csr, aeq_csr): (Option<CscMatrix<T>>, Option<CscMatrix<T>>) = if use_kron {
        (None, None)
    } else {
        (
            Some(
                prob.a_in_csr
                    .clone()
                    .unwrap_or_else(|| csr_of_dense(&prob.a_in, mi, n)),
            ),
            Some(
                prob.a_eq_csr
                    .clone()
                    .unwrap_or_else(|| csr_of_dense(&prob.a_eq, me, n)),
            ),
        )
    };
    // Compute NT scaling once for the density proxy, arrow/folded symbolic analyses.
    // Without this cache, scale() (which does O(k³) eigendecompositions and
    // O(Σ dimc²) block constructions) would run up to 3× before the main loop,
    // dominating setup time for SDP and large-SOC problems.
    let pre_scale: Option<(Vec<ConeScaling<T>>, Vec<Vec<T>>, Option<psd::KronEig<T>>)> =
        if use_kron { None } else { Some(scale(&s, &z)) };
    let (kkt0, density) = if use_kron {
        (None, 0.0)
    } else {
        let (_, ref blocks0, _) = pre_scale.as_ref().expect("pre_scale exists when !use_kron");
        let k = assemble_conic_kkt(
            prob,
            cones,
            &offsets,
            blocks0,
            rho,
            delta,
            aeq_csr.as_ref(),
            ain_csr.as_ref(),
            p_diagonal,
        );
        let d = k.nnz() as f64 / (((dim as f64) * (dim as f64) / 2.0).max(1.0));
        (Some(k), d)
    };
    // A single large cone (a big SOC from a norm, or a PSD cone) has a dense (z,z)
    // block — even an otherwise-sparse KKT fills in there, and min-degree ordering is
    // slow on a dense block. Route such problems to faer too, not just globally-dense
    // ones. The orthant's block is diagonal, so it never forces the dense path regardless
    // of its size — exclude it from the max.
    let max_cone_dim = cones
        .iter()
        .filter(|c| !matches!(c, Cone::NonNeg(_)))
        .map(|c| c.dim())
        .max()
        .unwrap_or(0);
    // Arrow-form sparse path: a large SOC's dense `(z,z)` block is what forces the dense
    // factor, but with arrow-form expansion (diagonal + 2 aux per cone) the KKT stays sparse
    // *if A itself is sparse* — the production case (factor-model risk SOC with diagonal A).
    // Decide on the actual A sparsity (the dense `(z,z)` inflates the `density` proxy, so it
    // can't be used here). Only second-order/orthant cones (no PSD), with a genuinely large
    // cone (otherwise the dense path's small factor already wins).
    let a_density = {
        let mut nnz = 0usize;
        for r in 0..mi {
            for i in 0..n {
                if prob.a_in.get(r, i) != zero {
                    nnz += 1;
                }
            }
        }
        nnz as f64 / ((mi as f64 * n as f64).max(1.0))
    };
    let use_arrow = !use_kron
        && max_cone_dim >= 8
        && a_density < 0.1
        && cones
            .iter()
            .all(|c| matches!(c, Cone::Soc(_) | Cone::NonNeg(_)));
    // The Kronecker path needs no fill-reducing ordering (it never assembles a KKT).
    let use_dense = !use_arrow && (density > 0.15 || max_cone_dim >= 8 || use_kron);
    // If P+ρ is diagonal (linear/diagonal-Hessian objective — the common SDP case) and
    // we're on the dense path, eliminate the x-block by a Schur complement and factor
    // only the reduced (y,z) system. With a large cone block this is much smaller than
    // the augmented KKT, so the factor is far cheaper.
    // `p_diagonal` (computed in the preamble) lets the residual matvec collapse `Px` to O(n).
    let p_matvec = |x: &[T]| -> Vec<T> {
        if p_diagonal {
            (0..n).map(|i| prob.p.get(i, i) * x[i]).collect()
        } else {
            prob.p.matvec(x)
        }
    };
    let nonneg_only = cones.iter().all(|c| matches!(c, Cone::NonNeg(_)));
    let nn_prefold: Option<(Vec<usize>, Vec<usize>)> =
        if p_diagonal && nonneg_only && mi > 0 && (n + mi) >= 200 {
            let (mut fr, mut fc) = (Vec::new(), Vec::new());
            for r in 0..mi {
                let (mut nz, mut c) = (0, 0);
                for j in 0..n {
                    if prob.a_in.get(r, j) != T::zero() {
                        nz += 1;
                        c = j;
                        if nz > 1 {
                            break;
                        }
                    }
                }
                if nz == 1 {
                    fr.push(r);
                    fc.push(c);
                }
            }
            if fr.is_empty() {
                None
            } else {
                Some((fr, fc))
            }
        } else {
            None
        };
    // Condense when the diagonal x-block can be cheaply eliminated (P is diagonal)
    // AND either a large cone block OR many bound rows would inflate the augmented KKT.
    // For pure NonNeg (LP/QP) with foldable bounds, n_in >= n means the augmented
    // (n+n_in)×(n+n_in) is at least 2× the condensed n×n — a clear win.
    // Structural row→fold-index map (usize::MAX for non-folded rows), computed once —
    // kills the per-call `fr.iter().position()` scans in the folded solve paths.
    let fold_idx: Option<Vec<usize>> = nn_prefold.as_ref().map(|(fr, _)| {
        let mut m = vec![usize::MAX; mi];
        for (k, &r) in fr.iter().enumerate() {
            m[r] = k;
        }
        m
    });
    let use_condensed = use_dense && p_diagonal && (max_cone_dim >= 8 || (nonneg_only && mi >= n));
    // Build the condensed Schur term by faer gemm when the stacked A is dense (SOCP),
    // by sparse outer products when it is sparse (SDP's A_in = −I). Decided once on the
    // fixed sparsity pattern: >25% dense ⇒ gemm.
    let dense_schur = use_condensed && (me + mi) >= 48 && {
        let mut nnz = 0usize;
        for i in 0..me {
            for k in 0..n {
                if prob.a_eq.get(i, k) != zero {
                    nnz += 1;
                }
            }
        }
        for i in 0..mi {
            for k in 0..n {
                if prob.a_in.get(i, k) != zero {
                    nnz += 1;
                }
            }
        }
        nnz * 4 > n * (me + mi)
    };
    // Condensed-path structural classification (computed once — sparsity and cone
    // type are fixed): single-nonzero NonNeg rows fold, the rest are kept; the
    // reduced rows are split into general vs unit (single-nonzero) for the Schur
    // term. The values (fold hinv, D⁻¹, cone blocks) still change per iteration.
    let mut cond_scratch: Option<CondensedScratch<T>> = None;
    if use_condensed {
        let mut fold_rc = Vec::new();
        let mut kept = Vec::new();
        for (cidx, cone) in cones.iter().enumerate() {
            let off = offsets[cidx];
            let d = cone.dim();
            let is_nn = matches!(cone, Cone::NonNeg(_));
            for l in 0..d {
                let row = off + l;
                let (mut nz, mut col) = (0usize, 0usize);
                for j in 0..n {
                    if prob.a_in.get(row, j) != zero {
                        nz += 1;
                        col = j;
                        if nz > 1 {
                            break;
                        }
                    }
                }
                if is_nn && nz == 1 {
                    fold_rc.push((row, col));
                } else {
                    kept.push(row);
                }
            }
        }
        let rdim = me + kept.len();
        let mut general = Vec::new();
        let mut units = Vec::new();
        for ri in 0..rdim {
            let (mut nz, mut col) = (0usize, 0usize);
            for k in 0..n {
                let v = if ri < me {
                    prob.a_eq.get(ri, k)
                } else {
                    prob.a_in.get(kept[ri - me], k)
                };
                if v != zero {
                    nz += 1;
                    col = k;
                    if nz > 1 {
                        break;
                    }
                }
            }
            match nz {
                0 => {}
                1 => units.push((ri, col)),
                _ => general.push(ri),
            }
        }
        cond_scratch = Some(CondensedScratch::new(
            n, me, mi, fold_rc, kept, general, units,
        ));
    }

    // The sparse paths need the ordering + symbolic analysis (computed once on the fixed
    // pattern); the dense paths need neither. The arrow path analyzes its own (larger,
    // aux-augmented) pattern.
    let (perm, sym) = if use_arrow {
        let (ref sc0, _, _) = pre_scale.as_ref().expect("pre_scale exists when !use_kron");
        let (akkt, _) = assemble_arrow_kkt(
            prob,
            cones,
            &offsets,
            sc0,
            rho,
            delta,
            aeq_csr.as_ref(),
            ain_csr.as_ref(),
            p_diagonal,
        );
        let perm = min_degree(akkt.n, &akkt.colptr, &akkt.rowval);
        let sym = iconic_linalg::analyze(&permute_upper(&akkt, &perm));
        (perm, Some(sym))
    } else if use_dense {
        (Vec::new(), None)
    } else {
        let kkt0 = kkt0.as_ref().expect("sparse path has an assembled KKT");
        let perm = min_degree(kkt0.n, &kkt0.colptr, &kkt0.rowval);
        let sym = iconic_linalg::analyze(&permute_upper(kkt0, &perm));
        (perm, Some(sym))
    };
    // Folded KKT: when NonNeg rows are pre-classified as foldable, compute a
    // separate symbolic factorization for the reduced-dimension KKT.
    let sp_fold_active = nn_prefold.as_ref().is_some_and(|(fr, _)| !fr.is_empty());
    let (fold_perm, fold_sym) = if sp_fold_active {
        let (fr, fc) = nn_prefold.as_ref().expect("prefold built for this branch");
        let fold_pairs: Vec<(usize, usize)> = fr.iter().zip(fc).map(|(&r, &c)| (r, c)).collect();
        let (_, ref blocks0, _) = pre_scale.as_ref().expect("pre_scale exists");
        // Build a structural KKT for symbolic analysis (hinv = 1.0 placeholder)
        let kkt0f = assemble_conic_kkt_folded(
            prob,
            cones,
            &offsets,
            blocks0,
            rho,
            delta,
            aeq_csr.as_ref(),
            ain_csr.as_ref(),
            p_diagonal,
            &fold_pairs,
            &vec![T::one(); fr.len()],
        );
        let perm = min_degree(kkt0f.n, &kkt0f.colptr, &kkt0f.rowval);
        let sym = iconic_linalg::analyze(&permute_upper(&kkt0f, &perm));
        (perm, Some(sym))
    } else {
        (Vec::new(), None)
    };

    // Best (lowest-μ, finite) iterate seen, for graceful return if the iteration
    // degrades near the cone boundary (ill-conditioning → non-finite directions).
    let mut best = (x.clone(), y.clone(), s.clone(), z.clone());
    let mut best_err = T::infinity(); // best max(residual, μ) seen
    let mut diverge = 0; // consecutive iterations far worse than the best
    let grade = |e: T| grade_status(e, eps);

    // A_in matvecs. On the Kronecker path A_in = −I, so they are pure negations (O(m))
    // rather than a dense O(m²) product — a meaningful share of the per-iteration cost
    // once the factor is O(k³).
    // The row-CSRs (`ain_csr`/`aeq_csr`) were built once above. The genuinely sparse factor
    // path drives its matvecs off them; the dense path keeps the dense matvec (bit-identical
    // summation order, so the dense-cone results are unchanged).
    // `A·v`: row r of the result dots the row-CSR's column r (= A's row r) with v.
    let row_matvec = |csr: &Option<CscMatrix<T>>, rows: usize, v: &[T]| -> Vec<T> {
        let a = csr.as_ref().expect("sparse form present in sparse branch");
        let mut out = vec![zero; rows];
        for r in 0..rows {
            let mut s = zero;
            for p in a.colptr[r]..a.colptr[r + 1] {
                s += a.nzval[p] * v[a.rowval[p]];
            }
            out[r] = s;
        }
        out
    };
    // `Aᵀ·v`: scatter each row r's entries weighted by v[r] into the column accumulator.
    let row_matvec_t = |csr: &Option<CscMatrix<T>>, rows: usize, v: &[T]| -> Vec<T> {
        let a = csr.as_ref().expect("sparse form present in sparse branch");
        let mut out = vec![zero; n];
        for r in 0..rows {
            let vr = v[r];
            if vr != zero {
                for p in a.colptr[r]..a.colptr[r + 1] {
                    out[a.rowval[p]] += a.nzval[p] * vr;
                }
            }
        }
        out
    };
    let ain_matvec = |v: &[T]| -> Vec<T> {
        if use_kron {
            v.iter().map(|&x| -x).collect()
        } else if !use_dense {
            row_matvec(&ain_csr, mi, v)
        } else {
            prob.a_in.matvec(v)
        }
    };
    let ain_matvec_t = |v: &[T]| -> Vec<T> {
        if use_kron {
            v.iter().map(|&x| -x).collect()
        } else if !use_dense {
            row_matvec_t(&ain_csr, mi, v)
        } else {
            prob.a_in.matvec_t(v)
        }
    };
    // `_into` variants writing into a caller buffer — identical accumulation order,
    // used by the buffer-reuse paths (recover_ds, solve_dir).
    let ain_matvec_into = |v: &[T], out: &mut [T]| {
        if use_kron {
            for i in 0..mi {
                out[i] = -v[i];
            }
        } else if !use_dense {
            let a = ain_csr.as_ref().expect("sparse A_in built for this path");
            for r in 0..mi {
                let mut s = zero;
                for p in a.colptr[r]..a.colptr[r + 1] {
                    s += a.nzval[p] * v[a.rowval[p]];
                }
                out[r] = s;
            }
        } else {
            prob.a_in.matvec_into(v, out);
        }
    };
    let aeq_matvec = |v: &[T]| -> Vec<T> {
        if !use_dense && !use_kron {
            row_matvec(&aeq_csr, me, v)
        } else {
            prob.a_eq.matvec(v)
        }
    };
    let aeq_matvec_t = |v: &[T]| -> Vec<T> {
        if !use_dense && !use_kron {
            row_matvec_t(&aeq_csr, me, v)
        } else {
            prob.a_eq.matvec_t(v)
        }
    };

    let mut status = Status::MaxIterations;
    let mut iters = 0;
    // Consecutive iterations whose step came out an order of magnitude shorter than
    // its own affine predictor — the stall signature that gates the multiple
    // centrality correctors off (each corrector is an extra solve against the
    // factor, wasted while steps stall). Reset whenever a step clears the ratio.
    let mut short_step_count = 0usize;

    // Pre-allocated scratch buffer for per-iteration mi-sized RHS vectors (affine,
    // combined, corrector) — overwritten each use, never reallocated.
    let mut rhs_scratch = vec![zero; mi];
    // Pre-allocated full-size RHS buffer for solve_dir / solve_cor — avoids an
    // allocation per solve call (~4 per iteration, ~40 across a solve).
    let mut rhs_buf = vec![zero; dim];
    // Pre-allocated per-folded-row reciprocal buffer (1/(η²+δ)); refilled per
    // iteration, read by the solve closures.
    let mut fold_hinv_buf = vec![zero; mi];
    // Pre-allocated residual buffers — reused every iteration.
    let mut r_d_buf = vec![zero; n];
    let mut r_b_buf = vec![zero; me];
    let mut r_h_buf = vec![zero; mi];
    // Per-cone Jordan identities are constant — compute once, reuse every iteration.
    let identities: Vec<Vec<T>> = cones.iter().map(|c| c.identity::<T>()).collect();
    let max_psd_k = cones
        .iter()
        .filter_map(|c| match c {
            Cone::Psd(_) => Some(psd::side_dim(c.dim())),
            _ => None,
        })
        .max()
        .unwrap_or(1);
    // Shared direction buffers: affine and combined solves reuse the same storage
    // (the affine outputs are dead once the combined RHS is built), the Gondzio
    // corrector writes trial buffers swapped in only on acceptance. Saves ~6
    // allocations per solve call and the split/recover_ds temporaries.
    let mut dx_buf = vec![zero; n];
    let mut dy_buf = vec![zero; me];
    let mut dz_buf = vec![zero; mi];
    let mut ds_buf = vec![zero; mi];
    let mut aindx_buf = vec![zero; mi];
    let mut cor_dx = vec![zero; n];
    let mut cor_dy = vec![zero; me];
    let mut cor_dz = vec![zero; mi];
    let mut ndx_buf = vec![zero; n];
    let mut ndy_buf = vec![zero; me];
    let mut nds_buf = vec![zero; mi];
    let mut ndz_buf = vec![zero; mi];
    // The folded-branch buffers and the solve scratch are shared by the solve_dir
    // and solve_cor closures — RefCell'd (interior mutability) so both closures
    // can capture them, the same discipline as rhs_buf on the QP path.
    let solve_scratch = std::cell::RefCell::new(SolveScratch::<T>::new());
    let rhs_fold_buf: std::cell::RefCell<Vec<T>> = std::cell::RefCell::new(Vec::new());
    let dz_kept_buf: std::cell::RefCell<Vec<T>> = std::cell::RefCell::new(Vec::new());

    // Per-cone scratch buffer sized to the largest cone dimension.
    let max_cdim = cones.iter().map(|c| c.dim()).max().unwrap_or(0);
    let mut cbuf = vec![zero; max_cdim];
    // Per-cone temporaries for the combined/Gondzio loops (max_cdim-sized, the
    // cbuf pattern) and the PSD cone-op scratch.
    let mut lam_buf = vec![zero; max_cdim];
    let mut winv_buf = vec![zero; max_cdim];
    let mut wdz_buf = vec![zero; max_cdim];
    let mut corr_buf = vec![zero; max_cdim];
    let mut ainv_buf = vec![zero; max_cdim];
    let mut wainv_buf = vec![zero; max_cdim];
    let mut st_buf = vec![zero; max_cdim];
    let mut zt_buf = vec![zero; max_cdim];
    let mut v_buf = vec![zero; max_cdim];
    let mut psi_buf = vec![zero; max_cdim];
    let mut negpsi_buf = vec![zero; max_cdim];
    let mut psd_scratch = psd::PsdScratch::<T>::new(max_psd_k);
    // Per-cone cached PSD arrow-inverse eigenpairs (storage once per solve; the
    // combined loop refills every PSD slot each iteration and the Gondzio loop,
    // which runs strictly after, reuses it — one eigendecomposition per PSD cone
    // per iteration instead of one per corrector round).
    let mut lam_eigs: Vec<Option<psd::ArrowEig<T>>> = cones
        .iter()
        .map(|c| match c {
            Cone::Psd(k) => Some(psd::ArrowEig::new(*k)),
            _ => None,
        })
        .collect();
    // Dense-KKT static block caching. The static part
    // (P, A_eq, A_in couplings) is built once; each iteration only updates the
    // dynamic entries (ρ on x-diag, δ on y-diag, cone blocks). Saves the O(dim²)
    // read + write of P/A_eq/A_in per iteration on the dense conic path.
    let mut dense_kkt_static: Option<DenseMatrix<f64>> = None;
    let mut dense_kkt_work: Option<DenseMatrix<f64>> = None;
    // Sparse KKT pattern cache: built from the first iteration's permuted KKT.
    // Subsequent iterations reuse colptr/rowval and only update dynamic nzval entries.
    let mut sparse_kkt_cache: Option<SparseKktCache<T>> = None;
    // Adaptive-regularization state (multi-level escalation/de-escalation).
    let mut prev_rel = T::zero();
    let mut prev_nb = T::zero();
    let mut prev_mu = T::zero();

    // Per-iteration phase profiling (env ICONIC_CONIC_PROF=1): the
    // scale / assemble+factor / solves / step-length / rest split,
    // printed after the solve.
    let prof_on = std::env::var_os("ICONIC_CONIC_PROF").is_some();
    if prof_on {
        eprintln!(
            "[conic-prof] setup (pre-loop): {:.3}ms",
            setup_t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let prof: [std::cell::Cell<f64>; 5] = std::array::from_fn(|_| std::cell::Cell::new(0.0)); // rest, scale, fac, solve, step
    let mut prof_t = std::time::Instant::now();
    let mut p_phase = 0u8; // 0=rest 1=scale 2=fac 3=solve 4=step
                           // Each marker ENDS the phase that was running and starts the named one:
                           // the elapsed since the previous marker is attributed to the phase that
                           // just ended (the marker's accumulator receives the PREVIOUS phase's
                           // time, so the call sites pass the phase that is ending).
    #[allow(unused_mut)]
    let mut acc = |ended_idx: usize, new_phase: u8, t: &mut std::time::Instant, cur: &mut u8| {
        if prof_on {
            if *cur != 0 {
                prof[ended_idx].set(prof[ended_idx].get() + t.elapsed().as_secs_f64());
            }
            *cur = new_phase;
            *t = std::time::Instant::now();
        }
    };
    for it in 0..settings.max_iters {
        iters = it;
        if prof_on {
            prof_t = std::time::Instant::now();
            p_phase = 0;
        }

        // Residuals.
        let px = p_matvec(&x);
        let aty = aeq_matvec_t(&y);
        let atz = ain_matvec_t(&z);
        let r_d = &mut r_d_buf;
        for i in 0..n {
            r_d[i] = px[i] + prob.q[i] + aty[i] + atz[i];
        }
        let aeqx = aeq_matvec(&x);
        let r_b = &mut r_b_buf;
        for i in 0..me {
            r_b[i] = aeqx[i] - prob.b_eq[i];
        }
        let ainx = ain_matvec(&x);
        let r_h = &mut r_h_buf;
        for i in 0..mi {
            r_h[i] = ainx[i] + s[i] - prob.b_in[i];
        }
        let mut mu = zero;
        for i in 0..mi {
            mu += s[i] * z[i];
        }
        mu /= degree;
        // Min-complementarity tracking: min(sᵢzᵢ) detects stalled variables
        // before the average mu catches up. When min/mu < 1e-3, guard convergence.
        let min_sz = if mi > 0 {
            let mut ms = s[0] * z[0];
            for i in 1..mi {
                let sz = s[i] * z[i];
                if sz < ms {
                    ms = sz;
                }
            }
            ms
        } else {
            zero
        };
        let min_mu_ratio = if mu > zero { min_sz / mu } else { T::one() };

        // Relative stopping test (absolute + relative to each residual's term
        // magnitudes), the standard conic-IPM criterion. An absolute 1e-8 on the dual
        // residual is too strict when the dual is large (active constraints), which is
        // exactly where a fixed absolute tolerance reports a spurious inaccuracy.
        let one_t = T::one();
        let sd = inf_norm(&px)
            .max(inf_norm(&prob.q))
            .max(inf_norm(&aty))
            .max(inf_norm(&atz));
        let sb = inf_norm(&aeqx).max(inf_norm(&prob.b_eq));
        let sh = inf_norm(&ainx).max(inf_norm(&s)).max(inf_norm(&prob.b_in));
        let nd = inf_norm(r_d);
        let nb = inf_norm(r_b);
        let nh = inf_norm(r_h);
        let rel = (nd / (one_t + sd))
            .max(nb / (one_t + sb))
            .max(nh / (one_t + sh));
        let err = rel.max(mu);

        // Non-finite iterate: ill-conditioning at the boundary produced a bad step.
        // Restore the best finite iterate and stop with an honest status.
        //
        // The finiteness test is on the ITERATE, not on `err`: `err = rel.max(mu)`
        // follows IEEE-754 maxNum semantics, so a NaN operand is silently ignored
        // when the other is finite — a NaN-poisoned iterate whose surviving entries
        // happen to be small (the boundary-dir blow-up kills most of s/z while the
        // residual entries stay ~0) computes a *finite* err and slips past a bare
        // `!err.is_finite()`, then crawls to the iteration cap as NaN (measured on a
        // small random SOCP: everything NaN from iteration 11, rel=0, 199 iters, then
        // a bogus NumericalError). Same trap as the QP loop's `iterate_finite`.
        let iterate_finite = x.iter().all(|v| v.is_finite())
            && y.iter().all(|v| v.is_finite())
            && s.iter().all(|v| v.is_finite())
            && z.iter().all(|v| v.is_finite());
        if !iterate_finite {
            x.clone_from(&best.0);
            y.clone_from(&best.1);
            s.clone_from(&best.2);
            z.clone_from(&best.3);
            status = grade(best_err);
            break;
        }
        if err < best_err {
            best_err = err;
            best.0.clone_from(&x);
            best.1.clone_from(&y);
            best.2.clone_from(&s);
            best.3.clone_from(&z);
        }
        // Min-complementarity guard: when min(sᵢzᵢ)/mu < 1e-3, require the
        // stalled variable to be converged too, not just the average.
        let comp_ok = mu <= eps && (min_mu_ratio >= from(1e-3) || min_sz <= eps);
        if rel <= eps && comp_ok {
            status = Status::Solved;
            break;
        }

        // Sustained cone exit (μ < 0): near a low-rank cone solution the NT scaling
        // becomes ill-conditioned and a step overshoots out of the cone, from which
        // the iterate does not recover. Fraction-to-boundary keeps a healthy iterate
        // strictly interior (μ > 0), so a persistently negative μ is terminal — stop
        // and return the best iterate rather than wandering to the cap. (A brief
        // numerical dip is tolerated by the consecutive count.)
        if mu < zero {
            diverge += 1;
            if diverge >= 3 {
                x.clone_from(&best.0);
                y.clone_from(&best.1);
                s.clone_from(&best.2);
                z.clone_from(&best.3);
                status = grade(best_err);
                break;
            }
        } else {
            diverge = 0;
        }
        if crate::is_unbounded(prob, &x) {
            status = Status::DualInfeasible;
            break;
        }
        // A blown-up iterate only *signals* possible infeasibility (see the matching
        // comment in lib.rs's QP loop) -- verify a Farkas certificate from the
        // diverging iterate before trusting it, rather than reporting a status the
        // heuristic merely guessed. Unverified, fall through (not break): the next
        // iteration's own Newton step gets a chance to self-correct, and the
        // non-finite-iterate guard above already catches a genuine numerical
        // breakdown next time through, grading via best-iterate rather than
        // asserting a wrong status here.
        if crate::check_diverge(&x, &z, big).is_some() {
            let ns = inf_norm(&x)
                .max(if me > 0 { inf_norm(&y) } else { zero })
                .max(if mi > 0 { inf_norm(&z) } else { zero })
                .max(one);
            if ns > zero {
                let xh: Vec<T> = x.iter().map(|&v| v / ns).collect();
                let yh: Vec<T> = y.iter().map(|&v| v / ns).collect();
                let zh: Vec<T> = z.iter().map(|&v| v / ns).collect();
                let bty = dot(&prob.b_eq, &yh) + dot(&prob.b_in, &zh);
                let mut atyz = aeq_matvec_t(&yh);
                let atz_h = ain_matvec_t(&zh);
                for i in 0..n {
                    atyz[i] += atz_h[i];
                }
                if bty > from(1e-10) && inf_norm(&atyz) < from(1e-6) {
                    x.clone_from(&best.0);
                    y.clone_from(&best.1);
                    s.clone_from(&best.2);
                    z.clone_from(&best.3);
                    status = Status::PrimalInfeasible;
                    break;
                }
                let qtx = dot(&prob.q, &xh);
                let aeqxh = aeq_matvec(&xh);
                let ainxh = ain_matvec(&xh);
                let axm = inf_norm(&aeqxh).max(inf_norm(&ainxh));
                if qtx < -from(1e-10) && axm < from(1e-6) && inf_norm(&p_matvec(&xh)) < from(1e-6) {
                    x.clone_from(&best.0);
                    y.clone_from(&best.1);
                    s.clone_from(&best.2);
                    z.clone_from(&best.3);
                    status = Status::DualInfeasible;
                    break;
                }
            }
        }

        // ----- adaptive proximal penalty update (regularization ladder) -----
        // Discrete regularization ladder with aggressive de-escalation.
        // Escalation is reserved for mu runaway (genuine numerical blow-up);
        // de-escalation drops one level immediately on 10× residual improvement.
        if it > 0 {
            // mu runaway: genuine numerical blow-up — jump levels up.
            if mu > prev_mu * from(5.0) && mu > from(1e6) {
                let boost = (mu / prev_mu).min(from(1e6));
                let levels_up = ((boost.to_f64().expect("finite scalar").log10() / 2.0).ceil() as usize).min(3);
                reg_level = (reg_level + levels_up).min(REG_LADDER.len() - 1);
            }
            // Aggressive de-escalation: 10× improvement → one level down.
            if rel < from(0.1) * prev_rel && nb < from(0.1) * prev_nb {
                reg_level = reg_level.saturating_sub(1);
            }
            let new_rho = T::from_f64(REG_LADDER[reg_level]).expect("scalar literal").max(rho0);
            let new_delta = T::from_f64(REG_LADDER[reg_level]).expect("scalar literal").max(delta0);
            rho = new_rho;
            delta = new_delta;
        }
        prev_rel = rel;
        prev_nb = nb;
        prev_mu = mu;

        // Scale, assemble, factor (condensed / dense faer / sparse LDLᵀ per the choice).
        acc(0, 1, &mut prof_t, &mut p_phase);
        let (sc, blocks, kron_eig) = scale(&s, &z);
        acc(1, 2, &mut prof_t, &mut p_phase);
        let fac: Fac<T> = if use_kron {
            // Kronecker SDP solve (A_in = −I, single PSD cone, P = 0). eig(W) comes from
            // `scale`; pre-eliminate the small equality block by a Schur complement using
            // gzy = (z,z)⁻¹(z,y).
            let eig = kron_eig.expect("use_kron implies a PSD-cone eig from scale");
            let dinv: Vec<T> = (0..n).map(|i| one / (prob.p.get(i, i) + rho)).collect();
            let inv_rho = dinv[0];
            let c = delta + inv_rho;
            // gzy_j = (z,z)⁻¹(z,y)_j with (z,y)_j = inv_rho·A_eq row j; reused both for the
            // Schur S_y[i][j] = (y,y)[i][j] − (y,z)_i·gzy_j and to recover Δz cheaply.
            let mut gzy: Vec<Vec<T>> = Vec::with_capacity(me);
            for j in 0..me {
                let zyj: Vec<T> = (0..mi).map(|r| inv_rho * prob.a_eq.get(j, r)).collect();
                let g = psd::kron_solve(&eig, c, &zyj);
                gzy.push(g.iter().map(|&v| -v).collect());
            }
            let schur = if me > 0 {
                let mut sy = DenseMatrix::<T>::zeros(me, me);
                for i in 0..me {
                    for j in 0..me {
                        let mut aij = zero;
                        let mut aig = zero;
                        for r in 0..mi {
                            let ar = prob.a_eq.get(i, r);
                            aij += ar * prob.a_eq.get(j, r);
                            aig += ar * gzy[j][r];
                        }
                        let yy = -(if i == j { delta } else { zero }) - inv_rho * aij;
                        sy.set(i, j, yy - inv_rho * aig);
                    }
                }
                match iconic_linalg::ldl_factor(&sy, pivot_tol) {
                    Ok(f) => Some(f),
                    Err(_) => {
                        status = Status::NumericalError;
                        break;
                    }
                }
            } else {
                None
            };
            Fac::Kronecker {
                eig,
                c,
                schur,
                gzy,
                inv_rho,
                dinv,
                n,
                me,
                mi,
            }
        } else if use_condensed {
            let sc = cond_scratch
                .as_mut()
                .expect("condensed scratch allocated when use_condensed");
            let red =
                build_condensed_into(prob, cones, &offsets, &blocks, rho, delta, dense_schur, sc);
            // Small reduced systems without a PSD cone: scalar LDLᵀ factors the ND matrix
            // directly, beating faer's (erratic) thread-pool overhead at small dims given
            // the ~6 solves/iter the condensed path does. PSD problems keep faer — the
            // dense NT block is ill-conditioned and the no-pivot scalar LDLᵀ does worse on
            // it. Larger systems: faer Cholesky on the (PD) negation, LBLT fallback.
            let has_psd = cones.iter().any(|c| matches!(c, Cone::Psd(_)));
            let reduced = if red.nrows < 48 && !has_psd {
                match iconic_linalg::ldl_factor(red, 1e-14) {
                    Ok(f) => ReducedFac::Scalar(f),
                    Err(_) => {
                        let mut neg = red.clone();
                        for v in neg.data_mut().iter_mut() {
                            *v = -*v;
                        }
                        // Platform-BLAS Cholesky on the (PD) negation:
                        // OpenBLAS's multithreaded dpotrf measured 2.4x over
                        // faer's par_llt at 1280 dims.
                        if red.nrows >= 48 {
                            match iconic_linalg::blas::dpotrf(red.nrows, neg.data_mut()) {
                                true => ReducedFac::BlasChol {
                                    a: neg.into_vec(),
                                    dim: red.nrows,
                                },
                                false => match iconic_linalg::faer_dense::FaerLlt::factor(&neg) {
                                    Some(llt) => ReducedFac::Chol(llt),
                                    None => ReducedFac::Indef(
                                        iconic_linalg::faer_dense::FaerLblt::factor(red),
                                    ),
                                },
                            }
                        } else {
                            match iconic_linalg::faer_dense::FaerLlt::factor(&neg) {
                                Some(llt) => ReducedFac::Chol(llt),
                                None => ReducedFac::Indef(
                                    iconic_linalg::faer_dense::FaerLblt::factor(red),
                                ),
                            }
                        }
                    }
                }
            } else if red.nrows >= 48 {
                // Large reduced system: the platform BLAS's multithreaded
                // Bunch-Kaufman (dsytrf) on the indefinite matrix directly —
                // no clone/negate allocation, and 6.8-25x over faer's
                // sequential LBLT at the large dims.
                let mut a = red.data.clone();
                let a = std::sync::Arc::make_mut(&mut a);
                match iconic_linalg::blas::dsytrf(red.nrows, a) {
                    Some(ipiv) => ReducedFac::BlasLdlt {
                        a: std::mem::take(a),
                        ipiv,
                        dim: red.nrows,
                    },
                    None => ReducedFac::Indef(iconic_linalg::faer_dense::FaerLblt::factor(red)),
                }
            } else {
                // Large reduced system: faer LBLT handles the indefinite matrix
                // directly — no clone/negate allocation per iteration.
                let lblt = iconic_linalg::faer_dense::FaerLblt::factor(red);
                ReducedFac::Indef(lblt)
            };
            // Small owned copies for the factor (n + k entries — negligible against
            // the rdim² matrix that now lives in the scratch). `red`'s borrow of the
            // scratch ends with the factor construction above.
            let dinv: Vec<T> = sc.dinv.clone();
            let fold: Vec<(usize, usize, T)> = sc
                .fold_rc
                .iter()
                .zip(sc.fold_hinv.iter())
                .map(|(&(r, c), &h)| (r, c, h))
                .collect();
            let kept = sc.kept.clone();
            Fac::Condensed {
                reduced,
                dinv,
                fold,
                kept,
                n,
                me,
                mi,
            }
        } else if use_dense {
            // Build the static part once (P, A_eq, A_in couplings — everything except
            // ρ/δ/cone-blocks), then each iteration only updates the dynamic entries.
            // Same pattern as the QP DenseCond path's m_static/m_work.
            if dense_kkt_static.is_none() {
                let mut dks = DenseMatrix::<f64>::zeros(dim, dim);
                let f = |x: T| x.to_f64().expect("scalar to f64");
                // P block.
                for i in 0..n {
                    for j in 0..n {
                        dks.set(i, j, f(prob.p.get(i, j)));
                    }
                }
                // A_eq couplings (off-diagonals only; diag goes in dynamic part).
                for r in 0..me {
                    for i in 0..n {
                        let v = f(prob.a_eq.get(r, i));
                        dks.set(n + r, i, v);
                        dks.set(i, n + r, v);
                    }
                }
                // A_in couplings (off-diagonals only).
                for (cidx, cone) in cones.iter().enumerate() {
                    let off = offsets[cidx];
                    let dimc = cone.dim();
                    for l in 0..dimc {
                        let r = off + l;
                        for i in 0..n {
                            let v = f(prob.a_in.get(r, i));
                            dks.set(n + me + r, i, v);
                            dks.set(i, n + me + r, v);
                        }
                    }
                }
                dense_kkt_static = Some(dks);
                dense_kkt_work = Some(DenseMatrix::<f64>::zeros(dim, dim));
            }
            let dkkt_st = dense_kkt_static.as_ref().expect("dense KKT template built");
            let dkkt = dense_kkt_work.as_mut().expect("dense KKT work allocated");
            // The static part is copied into the working matrix every iteration —
            // deliberately: the scalar LDLᵀ below factors the matrix IN PLACE
            // (L/D overwrite the input), so the work matrix cannot be persisted
            // across iterations without corrupting the off-diagonal KKT values.
            dkkt.data_mut().copy_from_slice(&dkkt_st.data);
            let rho_f = rho.to_f64().expect("finite scalar");
            let delta_f = delta.to_f64().expect("finite scalar");
            let f = |x: T| x.to_f64().expect("finite scalar");
            // Dynamic: x-block diagonal += ρ, y-block diagonal = −δ.
            for i in 0..n {
                dkkt.set(i, i, dkkt.get(i, i) + rho_f);
            }
            for r in 0..me {
                dkkt.set(n + r, n + r, -delta_f);
            }
            // Dynamic: cone (z,z) blocks = −(H_c + δI).
            for (cidx, cone) in cones.iter().enumerate() {
                let off = offsets[cidx];
                let dimc = cone.dim();
                for l in 0..dimc {
                    let r = off + l;
                    if let Cone::NonNeg(_) = cone {
                        dkkt.set(n + me + r, n + me + r, -(f(blocks[cidx][l]) + delta_f));
                    } else {
                        for lp in 0..dimc {
                            let hval = f(blocks[cidx][lp * dimc + l]);
                            let entry = if lp == l { -(hval + delta_f) } else { -hval };
                            dkkt.set(n + me + off + lp, n + me + r, entry);
                        }
                    }
                }
            }
            // Tiny systems: ICONIC's scalar LDLᵀ beats faer (no Mat allocation, no thread
            // dispatch) for the factor and the several solves per iteration. Larger systems:
            // the augmented KKT is quasidefinite (PD x-block before the ND y/z blocks), so an
            // *unpivoted* LDLᵀ is stable and ~2–3× the Bunch–Kaufman LBLT. Either dense factor
            // falls back to the pivoted LBLT on a near-singular pivot.
            if dim < 48 {
                match iconic_linalg::ldl::ldl_factor(dkkt, pivot_tol.to_f64().expect("finite scalar")) {
                    Ok(f) => Fac::DenseScalar(f),
                    Err(_) => match iconic_linalg::faer_dense::FaerLdlt::factor(dkkt) {
                        Some(f) => Fac::DenseLdlt(f),
                        None => Fac::Dense(iconic_linalg::faer_dense::FaerLblt::factor(dkkt)),
                    },
                }
            } else {
                // Large systems: the platform BLAS's multithreaded
                // Bunch–Kaufman (dsytrf) beats faer's sequential LDLᵀ/LBLT
                // (measured: the dense-LP routing's 1280-dim factor is
                // 432ms sequential vs 23ms dsytrf — 18.8x). The matrix is
                // cloned — the per-iteration work matrix must survive (the
                // in-place factor consumes its own copy). The dimension is
                // the FOLDED matrix's actual size (`dkkt.nrows`, not the
                // outer `dim` = n+me+mi, which counts the folded rows too —
                // a mismatch read out of bounds). Fall back to faer when
                // BLAS is unavailable.
                let kkt_dim = dkkt.nrows;
                if kkt_dim >= 48 {
                    let mut a = dkkt.data.clone();
                    let a = std::sync::Arc::make_mut(&mut a);
                    match iconic_linalg::blas::dsytrf(kkt_dim, a) {
                        Some(ipiv) => Fac::DenseBlasLdlt {
                            a: std::mem::take(a),
                            ipiv,
                            dim: kkt_dim,
                        },
                        None => match iconic_linalg::faer_dense::FaerLdlt::factor(dkkt) {
                            Some(f) => Fac::DenseLdlt(f),
                            None => Fac::Dense(iconic_linalg::faer_dense::FaerLblt::factor(dkkt)),
                        },
                    }
                } else {
                    match iconic_linalg::faer_dense::FaerLdlt::factor(dkkt) {
                        Some(f) => Fac::DenseLdlt(f),
                        None => Fac::Dense(iconic_linalg::faer_dense::FaerLblt::factor(dkkt)),
                    }
                }
            }
        } else if use_arrow {
            let (akkt, n_aux) = assemble_arrow_kkt(
                prob,
                cones,
                &offsets,
                &sc,
                rho,
                delta,
                aeq_csr.as_ref(),
                ain_csr.as_ref(),
                p_diagonal,
            );
            let pkkt = permute_upper(&akkt, &perm);
            match iconic_linalg::factor_with(&pkkt, sym.as_ref().expect("symbolic built above"), pivot_tol) {
                Ok(f) => Fac::ArrowSparse { factor: f, n_aux },
                Err(_) => {
                    status = Status::NumericalError;
                    break;
                }
            }
        } else {
            let (pkkt, _use_perm, use_sym) = if sp_fold_active {
                // Folded path: not cached (folding already reduces KKT size).
                let (fr, fc) = nn_prefold.as_ref().expect("prefold built for this branch");
                let fold_pairs: Vec<(usize, usize)> =
                    fr.iter().zip(fc).map(|(&r, &c)| (r, c)).collect();
                let fold_hinv: Vec<T> = fr
                    .iter()
                    .map(|&r| T::one() / (blocks[0][r] + delta))
                    .collect();
                let kkt = assemble_conic_kkt_folded(
                    prob,
                    cones,
                    &offsets,
                    &blocks,
                    rho,
                    delta,
                    aeq_csr.as_ref(),
                    ain_csr.as_ref(),
                    p_diagonal,
                    &fold_pairs,
                    &fold_hinv,
                );
                (
                    permute_upper(&kkt, &fold_perm),
                    &fold_perm,
                    fold_sym.as_ref().expect("fold symbolic built above"),
                )
            } else if let Some(ref cache) = sparse_kkt_cache {
                // Cached: reuse colptr/rowval, rebuild only dynamic nzval entries.
                let nzval = cache.update_nzval(prob, cones, &blocks, rho, delta);
                let pkkt_cached = CscMatrix {
                    m: cache.dim,
                    n: cache.dim,
                    colptr: cache.colptr.clone(),
                    rowval: cache.rowval.clone(),
                    nzval,
                };
                (pkkt_cached, &perm, sym.as_ref().expect("symbolic built above"))
            } else {
                // First iteration: assemble, permute, build cache.
                let kkt = assemble_conic_kkt(
                    prob,
                    cones,
                    &offsets,
                    &blocks,
                    rho,
                    delta,
                    aeq_csr.as_ref(),
                    ain_csr.as_ref(),
                    p_diagonal,
                );
                let pkkt_first = permute_upper(&kkt, &perm);
                sparse_kkt_cache = Some(SparseKktCache::from_assembly(
                    &pkkt_first,
                    &perm,
                    cones,
                    &offsets,
                    n,
                    me,
                ));
                (pkkt_first, &perm, sym.as_ref().expect("symbolic built above"))
            };
            match iconic_linalg::factor_with(&pkkt, use_sym, pivot_tol) {
                Ok(f) => Fac::Sparse(f),
                Err(_) => {
                    status = Status::NumericalError;
                    break;
                }
            }
        };
        // Solve the augmented system (possibly folded). When folding, rz contributions
        // are absorbed into rx, reducing the RHS dimension and eliminating z-rows.
        // Solves write into the shared dx/dy/dz buffers (FnMut: each call's outputs
        // are consumed before the next call overwrites them — the affine outputs die
        // once the combined RHS is built; the Gondzio corrector uses its own trial
        // buffers). `fold_idx` is the precomputed row→fold-index map.
        let fold_idx_opt = fold_idx.as_ref();
        // `1/(η²+δ)` per folded row: invariant across the RHS within an iteration
        // (used at four sites below — fold build in both solves and both dz
        // recoveries). One reciprocal pass per iteration instead of per RHS.
        let fold_hinv_iter: Option<&[T]> = if sp_fold_active {
            let (fr, _) = nn_prefold.as_ref().expect("prefold built for this branch");
            for (k, &r) in fr.iter().enumerate() {
                fold_hinv_buf[k] = T::one() / (blocks[0][r] + delta);
            }
            Some(&fold_hinv_buf[..fr.len()])
        } else {
            None
        };
        let solve_dir = |rhs_z: &[T], buf: &mut [T], dx: &mut [T], dy: &mut [T], dz: &mut [T]| {
            if sp_fold_active {
                let (fr, fc) = nn_prefold.as_ref().expect("prefold built for this branch");
                let folded_dim = n + me + (mi - fr.len());
                let mut rhs_fold = rhs_fold_buf.borrow_mut();
                rhs_fold.resize(folded_dim, zero);
                // Fold rz into rx: rhs[c] += a * hinv * rhs_z[r] for each folded row
                for i in 0..n {
                    rhs_fold[i] = -r_d[i];
                }
                for k in 0..fr.len() {
                    let (r, c) = (fr[k], fc[k]);
                    let a = prob.a_in.get(r, c);
                    let hinv = fold_hinv_iter.expect("precomputed with sp_fold_active")[k];
                    rhs_fold[c] += a * hinv * rhs_z[r];
                }
                for i in 0..me {
                    rhs_fold[n + i] = -r_b[i];
                }
                // Non-folded z rows fill the remaining RHS.
                let mut zi = 0usize;
                for r in 0..mi {
                    if fold_idx_opt.is_none_or(|m| m[r] == usize::MAX) {
                        rhs_fold[n + me + zi] = rhs_z[r];
                        zi += 1;
                    }
                }
                let mut dz_kept = dz_kept_buf.borrow_mut();
                dz_kept.resize(mi - fr.len(), zero);
                fac.solve_into(
                    prob,
                    &fold_perm,
                    &rhs_fold,
                    dx,
                    dy,
                    &mut dz_kept,
                    &mut solve_scratch.borrow_mut(),
                );
                // Recover folded dz: dz[r] = hinv * (a * dx[c] - rhs_z[r]).
                dz.fill(zero);
                let mut zk = 0usize;
                for r in 0..mi {
                    let k = fold_idx_opt.map_or(usize::MAX, |m| m[r]);
                    if k != usize::MAX {
                        let c = fc[k];
                        let a = prob.a_in.get(r, c);
                        let hinv = fold_hinv_iter.expect("precomputed with sp_fold_active")[k];
                        dz[r] = hinv * (a * dx[c] - rhs_z[r]);
                    } else {
                        dz[r] = dz_kept[zk];
                        zk += 1;
                    }
                }
            } else {
                for i in 0..n {
                    buf[i] = -r_d[i];
                }
                for i in 0..me {
                    buf[n + i] = -r_b[i];
                }
                buf[n + me..].copy_from_slice(rhs_z);
                fac.solve_into(
                    prob,
                    &perm,
                    &*buf,
                    dx,
                    dy,
                    dz,
                    &mut solve_scratch.borrow_mut(),
                );
            }
        };
        let recover_ds = |dx: &[T], ds_out: &mut [T], aindx_out: &mut [T]| {
            ain_matvec_into(dx, aindx_out);
            for r in 0..mi {
                ds_out[r] = -r_h[r] - aindx_out[r];
            }
        };
        // Corrector solve: the same factorization with a pure complementarity RHS
        // (zero feasibility residual in the x/y blocks) — used by the centrality
        // correctors, whose Δs is then −A_in·Δx.
        // Corrector solve: writes into the trial buffers (the current directions in
        // dx/dy/dz must survive the corrector, so the caller swaps on acceptance).
        let solve_cor =
            |rhs_z: &[T], buf: &mut [T], cdx: &mut [T], cdy: &mut [T], cdz: &mut [T]| {
                if sp_fold_active {
                    let (fr, fc) = nn_prefold.as_ref().expect("prefold built for this branch");
                    let folded_dim = n + me + (mi - fr.len());
                    let mut rhs_fold = rhs_fold_buf.borrow_mut();
                    rhs_fold.resize(folded_dim, zero);
                    // x/y blocks are zero (no feasibility residual in corrector solve).
                    // Fold rz into rx: rhs[c] += a * hinv * rhs_z[r] for each folded row.
                    for k in 0..fr.len() {
                        let (r, c) = (fr[k], fc[k]);
                        let a = prob.a_in.get(r, c);
                        let hinv = fold_hinv_iter.expect("precomputed with sp_fold_active")[k];
                        rhs_fold[c] += a * hinv * rhs_z[r];
                    }
                    // Non-folded z rows fill the remaining RHS.
                    let mut zi = 0usize;
                    for r in 0..mi {
                        if fold_idx_opt.is_none_or(|m| m[r] == usize::MAX) {
                            rhs_fold[n + me + zi] = rhs_z[r];
                            zi += 1;
                        }
                    }
                    let mut dz_kept = dz_kept_buf.borrow_mut();
                    dz_kept.resize(mi - fr.len(), zero);
                    fac.solve_into(
                        prob,
                        &fold_perm,
                        &rhs_fold,
                        cdx,
                        cdy,
                        &mut dz_kept,
                        &mut solve_scratch.borrow_mut(),
                    );
                    // Recover folded dz: dz[r] = hinv * (a * dx[c] - rhs_z[r]).
                    cdz.fill(zero);
                    let mut zk = 0usize;
                    for r in 0..mi {
                        let k = fold_idx_opt.map_or(usize::MAX, |m| m[r]);
                        if k != usize::MAX {
                            let c = fc[k];
                            let a = prob.a_in.get(r, c);
                            let hinv = fold_hinv_iter.expect("precomputed with sp_fold_active")[k];
                            cdz[r] = hinv * (a * cdx[c] - rhs_z[r]);
                        } else {
                            cdz[r] = dz_kept[zk];
                            zk += 1;
                        }
                    }
                } else {
                    for i in 0..n + me {
                        buf[i] = zero;
                    }
                    buf[n + me..].copy_from_slice(rhs_z);
                    fac.solve_into(
                        prob,
                        &perm,
                        &*buf,
                        cdx,
                        cdy,
                        cdz,
                        &mut solve_scratch.borrow_mut(),
                    );
                }
            };

        acc(2, 3, &mut prof_t, &mut p_phase);
        // Per-cone X^{-1/2} of the current s and z — shared by every step-length check
        // this iteration (affine, combined, and each corrector trial).
        let cache_s = invsqrt_cache(cones, &offsets, &s);
        let cache_z = invsqrt_cache(cones, &offsets, &z);

        acc(3, 4, &mut prof_t, &mut p_phase);
        // Affine (predictor): rhs_z = −r_h + s.
        for r in 0..mi {
            rhs_scratch[r] = -r_h[r] + s[r];
        }
        solve_dir(
            &rhs_scratch[..],
            &mut rhs_buf,
            &mut dx_buf,
            &mut dy_buf,
            &mut dz_buf,
        );
        recover_ds(&dx_buf, &mut ds_buf, &mut aindx_buf);
        let ap_a = (eta_ftb * cone_step(cones, &offsets, &s, &ds_buf, &cache_s))
            .min(one)
            .max(zero);
        let ad_a = (eta_ftb * cone_step(cones, &offsets, &z, &dz_buf, &cache_z))
            .min(one)
            .max(zero);
        let mut mu_aff = zero;
        for i in 0..mi {
            mu_aff += (s[i] + ap_a * ds_buf[i]) * (z[i] + ad_a * dz_buf[i]);
        }
        mu_aff /= degree;
        let sigma = if mu > zero {
            // Clamp alpha to [0,1] before cubing: a near-degenerate affine direction
            // can push mu_aff above mu, and capping only the squared term while
            // multiplying by the uncapped alpha leaves sigma unbounded for alpha > 1
            // (observed reaching absurd magnitudes and corrupting the corrector's
            // centering target on a knife-edge instance).
            let alpha = (mu_aff / mu).max(zero).min(one);
            // Centering floor on the Mehrotra sigma: the raw alpha^3 collapses
            // to ~0 at a near-complementary point, and the combined step then
            // degenerates to the affine step. At an active SOC/PSD boundary the
            // affine direction is dominated by NT-scaling noise (W's smallest
            // eigenvalue → 0, amplified by the condensed solve), so the step
            // becomes a boundary-limited wobble: measured 20-iteration limit
            // cycles on the small random SOCPs (26 iters, or a NaN-poisoned
            // iterate and 199 wasted iterations). A sigma floor keeps the
            // step central in the late phase — the suite's healthy conic
            // solves pay at most +1-2 iters (SGM +1.6%), while the failing
            // shapes go from cycles/NaN to Solved in <= 10 iters.
            let sigma_raw = (alpha * alpha).min(from(0.25)) * alpha;
            // The Kronecker SDP path (A_in = −I, single PSD cone) is exempt from
            // the floor: its Mehrotra step is well-centered with full step lengths
            // (the affine predictor already drives mu at ~1e-3/iter there), so the
            // 0.03 floor would only cap the mu decay at ~30x/iter. Measured:
            // the min-eig SDP family converges 9→7 (k=10), 8→6 (k=12) iterations
            // with a 0.001 floor, bit-identical objectives. The SOC/PSD boundary
            // wobble the floor guards against (short boundary-limited steps) does
            // not occur on this path — full steps throughout (ap/ad = 1).
            let sigma_floor = if use_kron { from(0.001) } else { from(0.03) };
            sigma_raw.max(sigma_floor)
        } else {
            zero
        };
        let sm = sigma * mu;

        // Combined (corrector): per cone, rhs_z = −r_h + s + W(Arw(λ)⁻¹(corr − σμ e)),
        // corr = (W⁻¹ Δs_aff) ∘ (W Δz_aff), λ = W z. Reuse rhs_scratch.
        rhs_scratch.fill(zero);
        let rhs_c = &mut rhs_scratch[..];
        for (c, cone) in cones.iter().enumerate() {
            let o = offsets[c];
            let d = cone.dim();
            let scaling = &sc[c];
            let lam = &mut lam_buf[..d];
            let winv_ds = &mut winv_buf[..d];
            let w_dz = &mut wdz_buf[..d];
            let corr = &mut corr_buf[..d];
            let ainv = &mut ainv_buf[..d];
            let w_ainv = &mut wainv_buf[..d];
            scaling.apply_w_into(&z[o..o + d], lam, &mut psd_scratch);
            scaling.apply_w_inv_into(&ds_buf[o..o + d], winv_ds, &mut psd_scratch);
            scaling.apply_w_into(&dz_buf[o..o + d], w_dz, &mut psd_scratch);
            cone.jordan_into(winv_ds, w_dz, corr, &mut psd_scratch);
            let e = &identities[c];
            let inner = &mut cbuf[..d];
            for i in 0..d {
                inner[i] = corr[i] - sm * e[i];
            }
            match cone {
                Cone::Psd(_) => {
                    let eig = lam_eigs[c].as_mut().expect("ArrowEig per PSD cone");
                    eig.compute(lam, &mut psd_scratch);
                    psd::arrow_inverse_apply_eig_into(&eig.d, &eig.q, inner, ainv, &mut psd_scratch);
                }
                _ => cone.arrow_inv_into(lam, inner, ainv, &mut psd_scratch),
            }
            scaling.apply_w_into(ainv, w_ainv, &mut psd_scratch);
            for i in 0..d {
                rhs_c[o + i] = -r_h[o + i] + s[o + i] + w_ainv[i];
            }
        }
        solve_dir(rhs_c, &mut rhs_buf, &mut dx_buf, &mut dy_buf, &mut dz_buf);
        recover_ds(&dx_buf, &mut ds_buf, &mut aindx_buf);
        let mut ap = (eta_ftb * cone_step(cones, &offsets, &s, &ds_buf, &cache_s))
            .min(one)
            .max(zero);
        let mut ad = (eta_ftb * cone_step(cones, &offsets, &z, &dz_buf, &cache_z))
            .min(one)
            .max(zero);

        // Gondzio multiple centrality correctors. Push the scaled complementarity at
        // an enlarged trial step toward the central band [β_min·μ, β_max·μ] and add the
        // resulting correction to the direction — but only keep it if the step length
        // strictly improves. The acceptance test makes this safe: a correction that
        // does not help is discarded, so it can only speed convergence, never harm it.
        if mu > zero {
            let beta_lo = from(0.1) * mu;
            let beta_hi = from(10.0) * mu;
            let gamma = from(0.1); // trial-step enlargement
            let cor_gain = from(0.01); // minimum step-length gain to accept
                                       // The Kronecker SDP path converges in the same iteration count with or without the
                                       // centrality correctors (the Mehrotra step already centers it well), so they only
                                       // add wasted corrector solves there; the SOC/orthant path genuinely needs them.
                                       // SOC converges faster with centrality correctors than the
                                       // general case — one corrector is enough when no PSD cone is
                                       // present (the Kronecker SDP path skips them entirely).
            let has_psd = cones.iter().any(|c| matches!(c, Cone::Psd(_)));
            // Adaptive corrector disable: a step that comes out an order of magnitude
            // shorter than its own affine predictor is a stall signature — the centering
            // is fighting the boundary (e.g. against an active cone). Each corrector is
            // an extra solve against the factor, wasted while steps stall, so after 3
            // consecutive short-step iterations we stop attempting correctors entirely
            // until a step clears the ratio, then the counter resets.
            if ap.min(ad) < from(0.1) * ap_a.min(ad_a) {
                short_step_count += 1;
            } else {
                short_step_count = 0;
            }
            let gondzio_max = if use_kron || short_step_count >= 3 {
                0
            } else if has_psd {
                2
            } else {
                1
            };
            for _ in 0..gondzio_max {
                let ap_t = (ap + gamma).min(one);
                let ad_t = (ad + gamma).min(one);
                rhs_scratch.fill(zero);
                let rhs_cor = &mut rhs_scratch[..];
                let mut any = false;
                for (c, cone) in cones.iter().enumerate() {
                    let o = offsets[c];
                    let d = cone.dim();
                    let scaling = &sc[c];
                    let s_t = &mut st_buf[..d];
                    let z_t = &mut zt_buf[..d];
                    let winv = &mut v_buf[..d];
                    let wdz = &mut negpsi_buf[..d];
                    let v = &mut corr_buf[..d];
                    let psi = &mut psi_buf[..d];
                    let ainv = &mut ainv_buf[..d];
                    for i in 0..d {
                        s_t[i] = s[o + i] + ap_t * ds_buf[o + i];
                        z_t[i] = z[o + i] + ad_t * dz_buf[o + i];
                    }
                    scaling.apply_w_inv_into(s_t, winv, &mut psd_scratch);
                    scaling.apply_w_into(z_t, wdz, &mut psd_scratch);
                    cone.jordan_into(winv, wdz, v, &mut psd_scratch);
                    // ψ = π_band(v) − v; the target enters the complementarity RHS with
                    // a minus, matching the −σμ·e centering term of the combined step. ψ is
                    // exactly zero for components already inside the band.
                    cone.centrality_target_into(v, beta_lo, beta_hi, psi, &mut psd_scratch);
                    if !any && psi.iter().any(|&p| p != zero) {
                        any = true;
                    }
                    let neg_psi = &mut wainv_buf[..d];
                    for i in 0..d {
                        neg_psi[i] = -psi[i];
                    }
                    match cone {
                        Cone::Psd(_) => {
                            // z/sc are unmodified since the combined loop refilled the
                            // cache (updates land in st_buf/zt_buf), so λ = W·z and the
                            // cached eigenpair is exactly what a per-call eigh computed.
                            let eig = lam_eigs[c].as_ref().expect("ArrowEig per PSD cone");
                            psd::arrow_inverse_apply_eig_into(
                                &eig.d,
                                &eig.q,
                                neg_psi,
                                ainv,
                                &mut psd_scratch,
                            );
                        }
                        _ => {
                            let lam = &mut lam_buf[..d];
                            scaling.apply_w_into(&z[o..o + d], lam, &mut psd_scratch);
                            cone.arrow_inv_into(lam, neg_psi, ainv, &mut psd_scratch);
                        }
                    }
                    let w_ainv = &mut wainv_buf[..d];
                    scaling.apply_w_into(ainv, w_ainv, &mut psd_scratch);
                    rhs_cor[o..o + d].copy_from_slice(&w_ainv[..d]);
                }
                // Every cone already centered → nothing to correct; skip the corrector solve.
                if !any {
                    break;
                }
                solve_cor(rhs_cor, &mut rhs_buf, &mut cor_dx, &mut cor_dy, &mut cor_dz);
                ain_matvec_into(&cor_dx, &mut aindx_buf);
                for i in 0..n {
                    ndx_buf[i] = dx_buf[i] + cor_dx[i];
                }
                for i in 0..mi {
                    nds_buf[i] = ds_buf[i] - aindx_buf[i];
                    ndz_buf[i] = dz_buf[i] + cor_dz[i];
                }
                let nap = (eta_ftb * cone_step(cones, &offsets, &s, &nds_buf, &cache_s))
                    .min(one)
                    .max(zero);
                let nad = (eta_ftb * cone_step(cones, &offsets, &z, &ndz_buf, &cache_z))
                    .min(one)
                    .max(zero);
                // Two-phase Gondzio acceptance (see lib.rs for explanation): first require the
                // corrected step to strictly lengthen the combined step, then require the
                // resulting duality-gap reduction to clear a minimum quality bar — a longer
                // step that barely improves the gap is not worth the extra factorization solve.
                let new_step = nap.min(nad);
                let old_step = ap.min(ad);
                if new_step > old_step + cor_gain {
                    let factor = one - sigma;
                    let gap_new = one - new_step * factor;
                    let gap_old = one - old_step * factor;
                    let quality = if gap_old > zero {
                        one - gap_new / gap_old
                    } else {
                        one
                    };
                    if quality < from(0.001) {
                        break; // step improved but gap reduction too small
                    }
                    for i in 0..me {
                        ndy_buf[i] = dy_buf[i] + cor_dy[i];
                    }
                    std::mem::swap(&mut dx_buf, &mut ndx_buf);
                    std::mem::swap(&mut dy_buf, &mut ndy_buf);
                    std::mem::swap(&mut ds_buf, &mut nds_buf);
                    std::mem::swap(&mut dz_buf, &mut ndz_buf);
                    ap = nap;
                    ad = nad;
                } else {
                    break;
                }
            }
        }

        for i in 0..n {
            x[i] += ap * dx_buf[i];
        }
        for i in 0..mi {
            s[i] += ap * ds_buf[i];
        }
        for i in 0..me {
            y[i] += ad * dy_buf[i];
        }
        for i in 0..mi {
            z[i] += ad * dz_buf[i];
        }
        if prof_on {
            acc(4, 0, &mut prof_t, &mut p_phase);
            prof[0].set(prof[0].get() + prof_t.elapsed().as_secs_f64());
        }
    }

    // Ran out of iterations: return the best iterate with an honest status.
    if status == Status::MaxIterations {
        let (bx, by, bs, bz) = best;
        x = bx;
        y = by;
        s = bs;
        z = bz;
        status = grade(best_err);
    }

    let px = p_matvec(&x);
    let half = from(0.5);
    let obj_val = half * dot(&x, &px) + dot(&prob.q, &x);

    if prof_on {
        let route = if use_kron {
            "kron"
        } else if use_condensed {
            "condensed"
        } else if use_dense {
            "dense"
        } else if use_arrow {
            "arrow"
        } else {
            "sparse"
        };
        eprintln!(
            "[conic-prof] route={} iters={} scale={:.3}ms fac={:.3}ms solve={:.3}ms step={:.3}ms rest={:.3}ms (per-iter: {:.3}/{:.3}/{:.3}/{:.3}/{:.3})",
            route,
            iters,
            prof[1].get() * 1e3,
            prof[2].get() * 1e3,
            prof[3].get() * 1e3,
            prof[4].get() * 1e3,
            prof[0].get() * 1e3,
            prof[1].get() * 1e3 / (iters.max(1) as f64),
            prof[2].get() * 1e3 / (iters.max(1) as f64),
            prof[3].get() * 1e3 / (iters.max(1) as f64),
            prof[4].get() * 1e3 / (iters.max(1) as f64),
            prof[0].get() * 1e3 / (iters.max(1) as f64),
        );
    }

    // Undo the Kronecker change of variables on the primal: x[σ(i)] = x̃[i]/g[i]. (s, z,
    // and obj_val are invariant under it, so they need no adjustment.)
    let x = if let Some((_, sigma, g)) = &kron {
        let mut xo = vec![zero; n];
        for i in 0..n {
            xo[sigma[i]] = x[i] / g[i];
        }
        xo
    } else {
        x
    };

    // Never report a non-finite iterate as solved — a NaN/Inf anywhere (e.g. an
    // ill-conditioned cone scaling that the best-iterate guard didn't catch) is an honest
    // numerical failure, not an optimum.
    let finite = x
        .iter()
        .chain(s.iter())
        .chain(z.iter())
        .all(|v| v.is_finite());
    let status = if finite {
        status
    } else {
        Status::NumericalError
    };

    QpSolution::new(
        status,
        x,
        y,
        s,
        z,
        obj_val,
        iters,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use iconic_linalg::DenseMatrix;

    fn settings() -> Settings<f64> {
        Settings::default()
    }

    /// The Kronecker SDP path (triggered by A_in = −I, single PSD cone, P = 0):
    /// min ⟨C,X⟩ s.t. X ⪰ 0, tr(X) = 1 has optimum λ_min(C). Here C has eigenvalues
    /// {1, 3, 3}, so the objective is 1.
    #[test]
    fn kronecker_sdp_min_eigenvalue() {
        let cmat =
            DenseMatrix::from_row_major(3, 3, vec![2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        let q = psd::svec(&cmat);
        let mut imat = DenseMatrix::<f64>::zeros(3, 3);
        for i in 0..3 {
            imat.set(i, i, 1.0);
        }
        let svec_i = psd::svec(&imat); // trace functional in svec coordinates
        let m = 3 * 4 / 2; // 6
        let mut a_in = DenseMatrix::zeros(m, m);
        for i in 0..m {
            a_in.set(i, i, -1.0);
        }
        let prob = QpProblem {
            p: DenseMatrix::zeros(m, m),
            q,
            a_eq: DenseMatrix::from_row_major(1, m, svec_i),
            b_eq: vec![1.0],
            a_in,
            b_in: vec![0.0; m],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_cone_qp(&prob, &[Cone::Psd(3)], &settings());
        assert_eq!(sol.status, Status::Solved, "iters={}", sol.iters);
        assert!(
            (sol.obj_val - 1.0).abs() < 1e-6,
            "obj={} expected λ_min=1",
            sol.obj_val
        );
    }

    /// The huber epigraph-split shape (equality-tied singleton u/p/n columns with
    /// diagonal curvature on `u`): `min ½‖u‖² + δ·1ᵀ(p+n) s.t. u + p − n = Cx − d,
    /// p,n ≥ 0`. The conic engine routes it through the sparse augmented KKT with
    /// the cached-pattern index-map (`SparseKktCache`); the fill-reducing ordering
    /// is non-trivial here, which used to make the cache silently wrong from the
    /// second iteration onward (the cached x-diags were read under the permuted
    /// index and the cone-block positions were searched in permuted columns under
    /// original indices) — the Newton step then exploded ~1e10 and the solve froze
    /// at SolvedInaccurate with a wrong objective. Regression: the conic engine
    /// must converge to the same objective as the QP engine.
    #[test]
    fn sparse_kkt_cache_permuted_index_maps_solve_huber_singletons() {
        let n_x = 5usize;
        let m = 20usize;
        let delta = 0.5;
        let mut rng = Lcg::new(42);
        let n = n_x + 3 * m;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in n_x..n_x + m {
            p.set(i, i, 1.0);
        }
        let mut q = vec![0.0; n];
        for i in n_x + m..n {
            q[i] = delta;
        }
        let mut a_eq = DenseMatrix::<f64>::zeros(m, n);
        for i in 0..m {
            for j in 0..n_x {
                a_eq.set(i, j, -rng.signed());
            }
            a_eq.set(i, n_x + i, 1.0);
            a_eq.set(i, n_x + m + i, 1.0);
            a_eq.set(i, n_x + 2 * m + i, -1.0);
        }
        let mut a_in = DenseMatrix::<f64>::zeros(2 * m, n);
        for i in 0..m {
            a_in.set(i, n_x + m + i, -1.0);
            a_in.set(m + i, n_x + 2 * m + i, -1.0);
        }
        let prob = QpProblem {
            p,
            q,
            a_eq,
            b_eq: vec![0.0; m],
            a_in,
            b_in: vec![0.0; 2 * m],
            a_eq_csr: None,
            a_in_csr: None,
        };
        // Reference: the dense QP engine.
        let ref_sol = crate::solve_qp(&prob, &Settings::default());
        assert_eq!(ref_sol.status, Status::Solved, "QP ref must solve");
        // Conic engine (sparse KKT path — the bug's path).
        let sol = solve_cone_qp(&prob, &[Cone::NonNeg(2 * m)], &settings());
        assert_eq!(sol.status, Status::Solved, "iters={}", sol.iters);
        assert!(
            (sol.obj_val - ref_sol.obj_val).abs() < 1e-6 * (1.0 + ref_sol.obj_val.abs()),
            "obj conic={} vs QP={}",
            sol.obj_val,
            ref_sol.obj_val
        );
        // The QP engine's solution itself is verified against the closed form
        // elsewhere; here the conic-vs-QP agreement is the canary.
    }

    /// With all cones of dimension 1 (pure nonnegative orthant), the cone solver
    /// reproduces the dedicated nonnegative solver on an inequality QP.
    #[test]
    fn nonneg_cones_match_bounded_lp() {
        // min −x0 − x1 s.t. x0 ≤ 1, x1 ≤ 1, x ≥ 0  → x = [1, 1].
        let a_in =
            DenseMatrix::from_row_major(4, 2, vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]);
        let prob = QpProblem {
            p: DenseMatrix::zeros(2, 2),
            q: vec![-1.0, -1.0],
            a_eq: DenseMatrix::zeros(0, 2),
            b_eq: vec![],
            a_in,
            b_in: vec![1.0, 1.0, 0.0, 0.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_cone_qp(
            &prob,
            &[Cone::Soc(1), Cone::Soc(1), Cone::Soc(1), Cone::Soc(1)],
            &settings(),
        );
        assert_eq!(sol.status, Status::Solved);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
    }

    /// Regression (e074023): the orthant (Cone::NonNeg) `_into` scaling arms must
    /// OVERWRITE their output buffers, not accumulate. The per-cone scratch
    /// (`lam_buf` etc.) is zeroed once at setup and reused across iterations; an
    /// accumulate arm multiplies into setup-time zeros on the first iteration
    /// (`lam = 0·w = 0`, breaking the later Arw(λ)⁻¹ division → ±∞) and then
    /// compounds stale values forever. Every in-tree test used dim-1 SOCs, whose
    /// arms overwrite — only the API's Zero+NonNeg conic route (factor-model
    /// dense-P QP) hit the orthant arm, returning SolvedInaccurate.
    #[test]
    fn nonneg_cone_through_conic_path() {
        // min ½‖x‖² − 2·1ᵀx s.t. x ∈ ℝ³₊ → unconstrained min x=2 (feasible),
        // obj = −6. A_in = −I so s = x in the NonNeg cone.
        let n = 3;
        let mut p = DenseMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let a_in = DenseMatrix::<f64>::from_row_major(
            n,
            n,
            (0..n)
                .flat_map(|i| (0..n).map(move |j| if i == j { -1.0 } else { 0.0 }))
                .collect(),
        );
        let prob = QpProblem {
            p,
            q: vec![-2.0; n],
            a_eq: DenseMatrix::zeros(0, n),
            b_eq: vec![],
            a_in,
            b_in: vec![0.0; n],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_cone_qp(&prob, &[Cone::NonNeg(n)], &settings());
        assert_eq!(sol.status, Status::Solved, "iters={}", sol.iters);
        for i in 0..n {
            assert!((sol.x[i] - 2.0).abs() < 1e-6, "x{}={}", i, sol.x[i]);
        }
        assert!((sol.obj_val - (-6.0)).abs() < 1e-5, "obj={}", sol.obj_val);
    }

    /// A genuine SOCP: minimize tᵀ... project the origin-shift onto a second-order
    /// cone constraint ‖(x1, x2)‖ ≤ x0 with x0 fixed via an equality.
    ///
    /// min ½‖x − c‖²  s.t.  x ∈ Q₃   (c = (1, 2, 2), so ‖c₁‖ = 2√2 > c₀ = 1).
    /// Encoded as A_in x + s = 0 with A_in = −I (s = x ∈ Q₃).
    #[test]
    fn socp_projection_onto_cone() {
        // ½xᵀx − cᵀx, c = (1,2,2). Unconstrained min is c, which is outside Q₃,
        // so the solution lies on the cone boundary x0 = ‖(x1,x2)‖.
        let prob = QpProblem {
            p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            q: vec![-1.0, -2.0, -2.0],
            a_eq: DenseMatrix::zeros(0, 3),
            b_eq: vec![],
            // s = x ∈ Q₃:  −I x + s = 0.
            a_in: DenseMatrix::from_row_major(
                3,
                3,
                vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
            ),
            b_in: vec![0.0, 0.0, 0.0],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_cone_qp(&prob, &[Cone::Soc(3)], &settings());
        assert_eq!(sol.status, Status::Solved, "iters={}", sol.iters);
        // On the boundary: x0 ≈ ‖(x1, x2)‖.
        let nrm = (sol.x[1] * sol.x[1] + sol.x[2] * sol.x[2]).sqrt();
        assert!(
            (sol.x[0] - nrm).abs() < 1e-5,
            "x0={} ‖x1:‖={}",
            sol.x[0],
            nrm
        );
        // Closed-form projection of c=(1,2,2) onto Q₃: scale factor (1 + 1/√2)/2.
        // x0 = (c0 + ‖c1‖)/2 = (1 + 2√2)/2.
        let expected_x0 = (1.0 + 2.0 * 2.0_f64.sqrt()) / 2.0;
        assert!(
            (sol.x[0] - expected_x0).abs() < 1e-5,
            "x0={} exp={}",
            sol.x[0],
            expected_x0
        );
    }

    /// A genuine SDP: project C = [[1,2],[2,1]] (eigenvalues 3, −1) onto the PSD
    /// cone. The minimizer clamps the negative eigenvalue to 0, giving
    /// X* = 1.5·[[1,1],[1,1]].
    #[test]
    fn sdp_psd_projection() {
        let c = DenseMatrix::from_row_major(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
        let cvec = crate::psd::svec(&c);
        let m = cvec.len(); // 3
        let mut p = DenseMatrix::zeros(m, m);
        let mut a_in = DenseMatrix::zeros(m, m);
        for i in 0..m {
            p.set(i, i, 1.0); // min ½‖X − C‖²_F  (svec preserves the inner product)
            a_in.set(i, i, -1.0); // s = X ∈ PSD:  −I·svec(X) + s = 0
        }
        let prob = QpProblem {
            p,
            q: cvec.iter().map(|&v| -v).collect(),
            a_eq: DenseMatrix::zeros(0, m),
            b_eq: vec![],
            a_in,
            b_in: vec![0.0; m],
            a_eq_csr: None,
            a_in_csr: None,
        };
        let sol = solve_cone_qp(&prob, &[Cone::Psd(2)], &settings());
        assert_eq!(sol.status, Status::Solved, "iters={}", sol.iters);
        let xstar = crate::psd::smat(&sol.x);
        for (i, &e) in [1.5, 1.5, 1.5, 1.5].iter().enumerate() {
            assert!(
                (xstar.data[i] - e).abs() < 1e-4,
                "X*[{i}]={}",
                xstar.data[i]
            );
        }
    }

    use iconic_core::rng::Lcg;

    /// The shape-1 random SOCP (n=12, 5 SOC cones of dim 4): a small dense
    /// cone problem whose near-optimal iterates sit on the active SOC boundary.
    /// Regression test for the conic loop's centering floor and iterate-
    /// finiteness guard: without the floor the Mehrotra sigma collapses to
    /// ~alpha^3 ~ 1e-9 at the near-optimal point, the combined step becomes
    /// affine-dominated, and the affine direction's boundary-noise component
    /// (the NT scaling degenerates at the active cone) launches a 20-iteration
    /// limit cycle — seed 19 took 26 iters, seed 9 NaN-poisoned the iterate at
    /// iteration 11 and ran the 199-iteration budget as NaN before grading
    /// NumericalError (the err = rel.max(mu) maxNum trap let it slip past the
    /// finiteness guard). With the sigma floor the trajectories stay central:
    /// both seeds Solve in <= 10 iters.
    #[test]
    fn shape1_socp_centering_floor_converges() {
        for seed in [9u64, 19u64] {
            let mut rng = Lcg::new(seed);
            let (n, n_cones, cone_dim) = (12usize, 5usize, 4usize);
            let mut l = DenseMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    l.set(i, j, rng.signed());
                }
            }
            let mut p = DenseMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0.0;
                    for t in 0..n {
                        acc += l.get(i, t) * l.get(j, t);
                    }
                    p.set(i, j, acc / n as f64);
                }
                p.set(i, i, p.get(i, i) + 1.0);
            }
            let q: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let m = n_cones * cone_dim;
            let mut a_in = DenseMatrix::<f64>::zeros(m, n);
            for r in 0..m {
                for j in 0..n {
                    a_in.set(r, j, rng.signed());
                }
            }
            let x0: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
            let mut b_in = vec![0.0; m];
            for c in 0..n_cones {
                let off = c * cone_dim;
                let mut ax = 0.0;
                for j in 0..n {
                    ax += a_in.get(off, j) * x0[j];
                }
                let mut nrm = 0.0;
                let mut tail = Vec::new();
                for _i in 1..cone_dim {
                    let t = 0.3 * rng.signed();
                    tail.push(t);
                    nrm += t * t;
                }
                b_in[off] = ax + nrm.sqrt() + 1.0;
                for (i, &ti) in tail.iter().enumerate() {
                    let mut axi = 0.0;
                    for j in 0..n {
                        axi += a_in.get(off + 1 + i, j) * x0[j];
                    }
                    b_in[off + 1 + i] = axi + ti;
                }
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
            let cones = vec![Cone::Soc(cone_dim); n_cones];
            let sol = solve_cone_qp(&prob, &cones, &settings());
            assert_eq!(
                sol.status,
                Status::Solved,
                "seed {seed}: status {:?} iters={}",
                sol.status,
                sol.iters
            );
            assert!(
                sol.iters <= 12,
                "seed {seed}: iters={} (limit cycle without the floor)",
                sol.iters
            );
            assert!(
                sol.obj_val.is_finite(),
                "seed {seed}: objective must be finite"
            );
        }
    }
}
#[cfg(test)]
mod warm_start_tests {
    use super::*;
    use iconic_core::WarmStart;


/// Warm-start exactness on the conic path: a seeded re-solve of a perturbed
/// SOCP converges to the same point as the cold re-solve (same status,
/// objective within 1e-8·max(1,|obj|)) and never takes more iterations.
#[test]
fn warm_start_conic_exactness() {
    // Signed [-1, 1) draws; any deterministic stream works (property test).
    let mut lcg = iconic_core::rng::SplitMix::new(2026);

    let n = 20usize;
    let (mut a_in, mut b_in) = (DenseMatrix::<f64>::zeros(12, n), vec![0.0; 12]);
    for r in 0..12 {
        for j in 0..n {
            a_in.set(r, j, lcg.signed());
        }
    }
    // b with a strictly-interior slack for each SOC (4 cones of dim 3).
    for c in 0..4usize {
        let o = c * 3;
        let tail: Vec<f64> = (1..3).map(|_| 0.4 * lcg.signed()).collect();
        let nrm = tail.iter().map(|&v| v * v).sum::<f64>().sqrt();
        b_in[o] = nrm + 2.0;
        for (i, &t) in tail.iter().enumerate() {
            b_in[o + 1 + i] = t;
        }
    }
    let mut p_diag = DenseMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        p_diag.set(i, i, 1.0);
    }
    let prob = QpProblem {
        p: p_diag,
        q: (0..n).map(|_| lcg.signed()).collect(),
        a_eq: DenseMatrix::zeros(0, n),
        b_eq: vec![],
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr: None,
    };
    let cones = vec![Cone::Soc(3); 4];
    let settings = Settings::<f64>::default();
    let base = solve_cone_qp(&prob, &cones, &settings);
    assert_eq!(base.status, Status::Solved);
    let seed = WarmStart {
        x: base.x,
        s: base.s,
        z: base.z,
    };
    let scale = prob.b_in.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let mut pert = prob.clone();
    for (i, v) in pert.b_in.iter_mut().enumerate() {
        *v += 1e-4 * scale * if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let cold = solve_cone_qp(&pert, &cones, &settings);
    let warm = solve_cone_qp_warm(&pert, &cones, &settings, Some(&seed));
    assert_eq!(warm.status, cold.status, "warm must not change the status");
    let tol = 1e-8 * cold.obj_val.abs().max(1.0);
    assert!(
        (warm.obj_val - cold.obj_val).abs() <= tol,
        "warm obj {:.12e} vs cold {:.12e}",
        warm.obj_val,
        cold.obj_val
    );
    assert!(
        warm.iters <= cold.iters,
        "warm must not take more iterations: {} vs {}",
        warm.iters,
        cold.iters
    );
    // A non-interior seed (slack on the SOC boundary) must fall back to the
    // cold trajectory, not crash or misbehave.
    let mut bad = seed.clone();
    let soc_scale = bad.s[1..3].iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    bad.s[0] = soc_scale; // apex == tail norm: on the boundary
    let fallback = solve_cone_qp_warm(&pert, &cones, &settings, Some(&bad));
    assert_eq!(fallback.iters, cold.iters);
    assert_eq!(fallback.obj_val, cold.obj_val);
}
}
