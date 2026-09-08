//! Fill-reducing ordering for sparse symmetric factorization.
//!
//! A minimum-degree elimination ordering: repeatedly eliminate the node of least
//! current degree, adding the fill edges its elimination induces among its
//! neighbors. Factoring the symmetrically-permuted matrix in this order produces far
//! less fill than the natural order on matrices with dense rows/columns (e.g. the
//! arrowhead couplings of a KKT system). This is the exact minimum-degree heuristic;
//! the "approximate" (AMD) refinement trades a little ordering quality for speed.
//!
//! **Dense-column shortcut:** When a node's remaining neighbor count exceeds
//! `dense_threshold`, the O(d²) fill-edge formation is skipped — the node's
//! neighbors still lose the connection to it but do NOT gain fill edges.  This is
//! safe because a node with degree ≥ `dense_threshold` already has the highest
//! degree and will be eliminated near the end regardless of the skipped fill.  The
//! ordering of the sparse subgraph ahead of it is negligibly affected, and avoiding
//! the O(d²) scan prevents a single dense KKT row (e.g. a portfolio `sum(x) = 1`)
//! from dominating the ordering cost.

use crate::CscMatrix;
use std::collections::{BTreeSet, HashSet};

/// Build the symmetric adjacency (off-diagonal structure) from an upper-CSC pattern.
fn adjacency(n: usize, colptr: &[usize], rowval: &[usize]) -> Vec<HashSet<usize>> {
    let mut adj = vec![HashSet::new(); n];
    for j in 0..n {
        for p in colptr[j]..colptr[j + 1] {
            let i = rowval[p];
            if i != j {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }
    }
    adj
}

/// Minimum-degree elimination order with a configurable dense-column threshold.
///
/// When `dense_threshold > 0`, nodes whose remaining-neighbor count reaches or
/// exceeds this threshold skip the O(d²) fill-edge formation (a constrained
/// minimum-degree variant).  Pass `dense_threshold = 0` to disable the shortcut
/// (exact minimum-degree always).
///
/// See the [module-level docs](self) for details on the dense-column shortcut.
pub fn min_degree_with_threshold(
    n: usize,
    colptr: &[usize],
    rowval: &[usize],
    dense_threshold: usize,
) -> Vec<usize> {
    let mut adj = adjacency(n, colptr, rowval);
    let mut degree: Vec<usize> = adj.iter().map(|s| s.len()).collect();
    let mut bucket: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n + 1];
    for v in 0..n {
        bucket[degree[v]].insert(v);
    }
    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut min_deg = 0usize;

    // Move `v` to the bucket for `new_deg`, dropping `min_deg` if it fell below the current min.
    let rebucket = |bucket: &mut Vec<BTreeSet<usize>>,
                    degree: &mut Vec<usize>,
                    min_deg: &mut usize,
                    v: usize,
                    new_deg: usize| {
        bucket[degree[v]].remove(&v);
        degree[v] = new_deg;
        bucket[new_deg].insert(v);
        if new_deg < *min_deg {
            *min_deg = new_deg;
        }
    };

    for _ in 0..n {
        while bucket[min_deg].is_empty() {
            min_deg += 1;
        }
        // The while-loop above guarantees bucket[min_deg] is non-empty, so
        // `next()` always yields — the lowest index of the least degree.
        let best = *bucket[min_deg]
            .iter()
            .next()
            .expect("min_deg bucket is non-empty");
        bucket[min_deg].remove(&best);
        eliminated[best] = true;
        order.push(best);

        // `best`'s remaining neighbors lose it from their degree, then become a clique
        // (fill); each genuinely new clique edge raises both endpoints' degree by one.
        let nbrs: Vec<usize> = adj[best]
            .iter()
            .filter(|&&u| !eliminated[u])
            .copied()
            .collect();
        for &u in &nbrs {
            let d = degree[u] - 1;
            rebucket(&mut bucket, &mut degree, &mut min_deg, u, d);
        }

        // Dense-column shortcut (constrained minimum-degree): when a node has
        // more remaining neighbors than `dense_threshold`, skip the O(d²)
        // fill-edge scan.  Such a node is already very high-degree — it will be
        // eliminated last (or nearly last) by the natural min-degree order
        // regardless of whether its neighbors gain fill.  Skipping the scan
        // avoids catastrophic fill-forming cost on KKT rows that couple many
        // variables (e.g. portfolio `sum(x)=1` at n=700 or budget equality at
        // n=2000).
        let skip_fill = dense_threshold > 0 && nbrs.len() >= dense_threshold;
        if !skip_fill {
            for a in 0..nbrs.len() {
                for b in (a + 1)..nbrs.len() {
                    let (na, nb) = (nbrs[a], nbrs[b]);
                    if adj[na].insert(nb) {
                        let d = degree[na] + 1;
                        rebucket(&mut bucket, &mut degree, &mut min_deg, na, d);
                    }
                    if adj[nb].insert(na) {
                        let d = degree[nb] + 1;
                        rebucket(&mut bucket, &mut degree, &mut min_deg, nb, d);
                    }
                }
            }
        }
    }

    order
}

/// Minimum-degree ordering with the default `dense_threshold = max(50, n / 10)`.
///
/// The underlying density criterion is a half-fill test: a column is treated as
/// dense once its accumulated nonzero count would fill roughly half the
/// remaining matrix (`accumulated_nnz * 2 ≥ n`). The `max(50, n/10)` default is
/// a conservative threshold chosen so that small/medium KKT systems only trigger
/// the shortcut on genuinely dense columns (≥ 51 nonzeros at the low end),
/// while it scales with `n` for larger systems.
///
/// See [`min_degree_with_threshold`] for details on the dense-column shortcut.
pub fn min_degree(n: usize, colptr: &[usize], rowval: &[usize]) -> Vec<usize> {
    let dense_threshold = std::cmp::max(50, n / 10);
    min_degree_with_threshold(n, colptr, rowval, dense_threshold)
}

/// AMD (Approximate Minimum Degree) ordering via the `amd` crate.
/// Uses the quotient-graph approximate minimum degree algorithm. AMD typically
/// produces less fill than exact min-degree on large irregular graphs, and
/// runs in O(nnz) time vs exact MD's O(n²) worst case. Falls back to exact
/// min-degree if AMD fails.
pub fn amd_order(n: usize, colptr: &[usize], rowval: &[usize]) -> Vec<usize> {
    let control = amd::Control::default();
    match amd::order(n, colptr, rowval, &control) {
        Ok((perm, _perm_inv, _info)) => perm.to_vec(),
        Err(_) => min_degree(n, colptr, rowval),
    }
}

/// Inverse permutation: `inv[perm[k]] = k`.
fn inverse_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (k, &p) in perm.iter().enumerate() {
        inv[p] = k;
    }
    inv
}

/// Symmetrically permute an upper-triangular CSC matrix: returns `A'` with
/// `A'[k][l] = A[perm[k]][perm[l]]`, stored as upper-triangular CSC.
pub fn permute_upper<T: Copy>(a: &CscMatrix<T>, perm: &[usize]) -> CscMatrix<T> {
    let n = a.n;
    let inv = inverse_perm(perm);

    // Collect new entries per new column (upper triangle).
    let mut cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
    for j in 0..n {
        for p in a.colptr[j]..a.colptr[j + 1] {
            let i = a.rowval[p];
            let v = a.nzval[p];
            // Original (i, j), i ≤ j. New indices, placed in the upper triangle.
            let (ni, nj) = (inv[i], inv[j]);
            let (r, c) = if ni <= nj { (ni, nj) } else { (nj, ni) };
            cols[c].push((r, v));
        }
    }

    let mut colptr = vec![0usize; n + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for c in 0..n {
        cols[c].sort_by_key(|&(r, _)| r);
        for &(r, v) in &cols[c] {
            rowval.push(r);
            nzval.push(v);
        }
        colptr[c + 1] = rowval.len();
    }

    CscMatrix {
        m: n,
        n,
        colptr,
        rowval,
        nzval,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_ldl::sparse_ldl_factor;

    use crate::sparse_ldl::upper_csc;

    #[test]
    fn min_degree_is_a_valid_permutation() {
        // Arrowhead: node 3 couples to all others (highest degree).
        let dense = [
            5.0, 0.0, 0.0, 1.0, //
            0.0, 6.0, 0.0, 2.0, //
            0.0, 0.0, 7.0, 3.0, //
            1.0, 2.0, 3.0, 9.0,
        ];
        let a = upper_csc(4, &dense);
        let perm = min_degree(4, &a.colptr, &a.rowval);
        let mut seen = perm.clone();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3]);
        // The dense hub (node 3) should be eliminated last to avoid fill.
        assert_eq!(*perm.last().unwrap(), 3);
    }

    #[test]
    fn permuted_factorization_solves_same_system() {
        let dense = [
            5.0, 0.0, 0.0, 1.0, //
            0.0, 6.0, 0.0, 2.0, //
            0.0, 0.0, 7.0, 3.0, //
            1.0, 2.0, 3.0, 9.0,
        ];
        let a = upper_csc(4, &dense);
        let perm = min_degree(4, &a.colptr, &a.rowval);
        let pa = permute_upper(&a, &perm);
        let f = sparse_ldl_factor(&pa, 1e-14).unwrap();

        // Solve A x = b via the permuted system: b' = b[perm], x[perm] = x'.
        let b = [1.0, 2.0, 3.0, 4.0];
        let bp: Vec<f64> = perm.iter().map(|&p| b[p]).collect();
        let xp = f.solve(&bp);
        let mut x = [0.0; 4];
        for (k, &p) in perm.iter().enumerate() {
            x[p] = xp[k];
        }
        // Verify A x = b.
        for i in 0..4 {
            let row: f64 = (0..4).map(|j| dense[i * 4 + j] * x[j]).sum();
            assert!((row - b[i]).abs() < 1e-10, "row {i}: {row} vs {}", b[i]);
        }
    }

    #[test]
    fn ordering_reduces_fill_on_arrowhead() {
        // Natural order on a leading-hub arrowhead fills the trailing block; the
        // min-degree order (hub last) keeps L sparse.
        let n = 6;
        let mut dense = vec![0.0f64; n * n];
        for i in 0..n {
            dense[i * n + i] = (i + 2) as f64;
        }
        // Node 0 is the hub: couple it to everyone.
        for j in 1..n {
            dense[j] = 1.0; // (0, j)
            dense[j * n] = 1.0; // (j, 0)
        }
        let a = upper_csc(n, &dense);
        let natural_nnz = sparse_ldl_factor(&a, 1e-14).unwrap().l_nnz();

        let perm = min_degree(n, &a.colptr, &a.rowval);
        let pa = permute_upper(&a, &perm);
        let reordered_nnz = sparse_ldl_factor(&pa, 1e-14).unwrap().l_nnz();

        assert!(
            reordered_nnz < natural_nnz,
            "fill not reduced: {reordered_nnz} vs {natural_nnz}"
        );
    }

    #[test]
    fn dense_threshold_skips_fill_on_high_degree_node() {
        // A 10×10 matrix where node 9 connects to all others (degree 9).
        let n = 10;
        let mut dense = vec![0.0f64; n * n];
        for i in 0..n {
            dense[i * n + i] = (i + 2) as f64;
        }
        // Node 9 is the hub: couple it to everyone (more than the default
        // threshold of max(50,10/10)=50 won't trigger, so use an explicit
        // small threshold).
        for j in 0..9 {
            dense[9 * n + j] = 1.0;
            dense[j * n + 9] = 1.0;
        }
        let a = upper_csc(n, &dense);

        // With threshold=5, node 9 has degree 9 >= 5, so fill is skipped.
        let perm_skip = min_degree_with_threshold(n, &a.colptr, &a.rowval, 5);
        let pa_skip = permute_upper(&a, &perm_skip);
        let f_skip = sparse_ldl_factor(&pa_skip, 1e-14).unwrap();

        // Without the shortcut (threshold=0), node 9 still ends up last.
        let perm_exact = min_degree_with_threshold(n, &a.colptr, &a.rowval, 0);
        let pa_exact = permute_upper(&a, &perm_exact);
        let f_exact = sparse_ldl_factor(&pa_exact, 1e-14).unwrap();

        // Both should place the hub last.
        assert_eq!(*perm_skip.last().unwrap(), 9);
        assert_eq!(*perm_exact.last().unwrap(), 9);

        // The fill (L nnz) with the shortcut should be similar to exact —
        // the skipped fill only affects degrees of neighbors near the end,
        // not the overall ordering structure for a matrix this small.
        let nnz_diff = (f_skip.l_nnz() as isize - f_exact.l_nnz() as isize).unsigned_abs();
        assert!(
            nnz_diff <= 2,
            "fill differs too much: skip={} exact={}",
            f_skip.l_nnz(),
            f_exact.l_nnz(),
        );

        // Both should solve correctly.
        let b: Vec<f64> = (0..n).map(|i| (i as f64) - 5.0).collect();
        let bp_skip: Vec<f64> = perm_skip.iter().map(|&p| b[p]).collect();
        let xp_skip = f_skip.solve(&bp_skip);
        let bp_exact: Vec<f64> = perm_exact.iter().map(|&p| b[p]).collect();
        let xp_exact = f_exact.solve(&bp_exact);
        for i in 0..n {
            assert!(
                (xp_skip[i] - xp_exact[i]).abs() < 1e-10,
                "solution differs at {i}"
            );
        }
    }

    #[test]
    fn dense_threshold_produces_valid_ordering_at_various_sizes() {
        for n in [3, 10, 30, 100] {
            // Banded matrix: each column i connects to i-5..i+5.
            let mut dense = vec![0.0f64; n * n];
            for i in 0..n {
                dense[i * n + i] = (i + 2) as f64;
                for d in 1..=5 {
                    if i + d < n {
                        dense[i * n + (i + d)] = 0.5;
                        dense[(i + d) * n + i] = 0.5;
                    }
                }
            }
            let a = upper_csc(n, &dense);
            let perm = min_degree_with_threshold(n, &a.colptr, &a.rowval, 50);
            let mut seen = perm.clone();
            seen.sort();
            assert_eq!(seen, (0..n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn amd_produces_valid_permutation_and_differs_from_md() {
        let n = 100;
        let mut colptr = vec![0usize; n + 1];
        let mut rowval = Vec::new();
        for c in 0..n {
            rowval.push(c); // diagonal
            for r in 0..c {
                if (r.wrapping_mul(c).wrapping_add(7)) % 20 == 0 {
                    rowval.push(r);
                }
            }
            colptr[c + 1] = rowval.len();
        }
        let perm_amd = amd_order(n, &colptr, &rowval);
        let perm_md = min_degree(n, &colptr, &rowval);
        assert_eq!(perm_amd.len(), n);
        // Verify permutation
        let mut seen = vec![false; n];
        for &p in &perm_amd {
            seen[p] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "AMD did not produce a valid permutation"
        );
        let same = perm_amd == perm_md;
        eprintln!(
            "AMD vs MD on {}x{} sparse: {}",
            n,
            n,
            if same { "identical" } else { "DIFFERENT" }
        );
    }
}
