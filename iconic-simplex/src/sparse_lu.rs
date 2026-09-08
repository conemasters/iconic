//! Sparse LU factorization with Markowitz pivoting for the dual simplex.
//!
//! The dense path stores the basis factor as an `m x m` array and pays dense
//! `O(m^2)` triangular solves and an `O(m^3)` factorization on every basis,
//! regardless of how empty the basis matrix is. MIP node LPs make that the
//! whole cost: the convexified quadratic relaxations (e.g. qkp) build bases
//! with ~3 nonzeros per row at `m ~ 700`, i.e. ~0.4% density -- measured as
//! one 7.6ms factorization plus ~0.6ms per pivot, nearly all of it arithmetic
//! on structural zeros. This module is the standard remedy every production
//! simplex uses: a Markowitz-ordered sparse LU with threshold pivoting (Suhl
//! & Suhl 1991), and triangular solves that touch only stored entries.
//!
//! The factorization produces `L U = P B Q` for the basis matrix `B`, with
//! `L` unit-lower (stored by elimination column, CSC), `U` upper (stored by
//! elimination row, CSR, diagonal kept separately), and permutations `prow`
//! (step -> original row) / `qcol` (step -> original basis slot). Both solve
//! directions are `O(nnz(L) + nnz(U))` per RHS: the forward solve scatters
//! down L's columns, the backward solve gathers across U's rows, and the
//! transposed solves walk the same arrays mirror-image. A RHS that is mostly
//! zeros (the FTRAN unit-vector case) additionally skips every zero pivot
//! component, so its cost tracks the *touched* fill, not the total.
//!
//! Rank deficiency follows the caller's dense-path contract: [`SparseLu::build`]
//! returns the basis slots that could not be pivoted plus a `perm` array with
//! `perm[slot] = row` for every slot (pivoted slots map to their pivot row,
//! deficient slots to distinct leftover rows), so the caller's repair --
//! substitute the logical (slack) column of `perm[slot]` for basis slot
//! `slot`, refactor -- runs unchanged. Solves on a deficient factor remain
//! total (free slots read 0) but are never used: the caller repairs and
//! refactors before solving.

use iconic_core::Scalar;

/// Relative threshold for accepting a Markowitz candidate: a pivot `a_ij` is
/// acceptable when `|a_ij| >= PIVOT_REL_TAU * max_j |a_ij|` over its row. The
/// Markowitz quotient picks among acceptable entries; the threshold keeps the
/// accepted one from being numerically tiny relative to what it eliminates
/// (the classic stability/fill trade-off -- Suhl & Suhl use 0.1, LUSOL
/// defaults near it).
const PIVOT_REL_TAU: f64 = 0.1;

/// Give up on the sparse factorization when the stored fill reaches this
/// share of `m^2`: past it the packed representation costs more than the
/// dense loop it replaces, so the caller falls back to its dense/BLAS path.
/// Checked both before factoring (structural nonzeros of the basis) and
/// during elimination (running fill).
pub const FILL_DENSITY_BAIL: f64 = 0.20;

/// Absolute backstop for pivot acceptance, scaled by the largest entry of the
/// basis matrix: rejects exact zeros and denormal-scale noise that slipped
/// past the row-relative threshold.
const PIVOT_ABS_FLOOR_REL: f64 = 1e-12;

/// Packed sparse LU of a basis matrix. Row ids are constraint rows; column
/// ids are basis slots (position `k` of the caller's basis array).
#[derive(Clone, Debug)]
pub struct SparseLu<T> {
    m: usize,
    /// Elimination step -> original row of the pivot (length = steps).
    prow: Vec<u32>,
    /// Elimination step -> original basis slot of the pivot (length = steps).
    qcol: Vec<u32>,
    /// Unit-lower L by elimination column (= the step that produced the
    /// multipliers): entries (elimination row, value) below the diagonal.
    l_start: Vec<u32>,
    l_row: Vec<u32>,
    l_val: Vec<T>,
    /// Upper U by elimination row: entries (elimination col, value) strictly
    /// right of the diagonal. The diagonal pivots live in `u_diag`.
    u_start: Vec<u32>,
    u_col: Vec<u32>,
    u_val: Vec<T>,
    u_diag: Vec<T>,
}

impl<T: Scalar> SparseLu<T> {
    /// Stored fill across L and U (the L diagonal is implicit unit, the U
    /// diagonal lives in `u_diag`).
    pub fn offdiag_nnz(&self) -> usize {
        self.l_val.len() + self.u_val.len()
    }

    /// Factor the `m x m` matrix whose column `k` is the sparse triple slice
    /// `(row_idx, val)` over `col_start` -- the basis matrix in the caller's
    /// CSC storage, column `k` = basis slot `k`.
    ///
    /// Returns `None` when the fill guard trips (caller falls back to its
    /// dense path); otherwise `(factor, deficient, perm)` where `deficient`
    /// lists the basis slots that could not be pivoted (rank deficiency,
    /// ascending slot order) and `perm` satisfies `perm[slot] = row` for
    /// every slot: pivoted slots map to their pivot row, deficient slots to
    /// distinct leftover rows -- the dense path's repair contract.
    pub fn build(
        m: usize,
        col_start: &[usize],
        row_idx: &[usize],
        val: &[T],
        basis_cols: &[usize],
    ) -> Option<(SparseLu<T>, Vec<usize>, Vec<usize>)> {
        if m == 0 {
            return Some((
                SparseLu {
                    m: 0,
                    prow: Vec::new(),
                    qcol: Vec::new(),
                    l_start: vec![0],
                    l_row: Vec::new(),
                    l_val: Vec::new(),
                    u_start: vec![0],
                    u_col: Vec::new(),
                    u_val: Vec::new(),
                    u_diag: Vec::new(),
                },
                Vec::new(),
                Vec::new(),
            ));
        }
        // Active rows as sorted (slot, val) vectors. Slack columns arrive
        // through the same CSC (a single nonzero), so nothing is special-cased.
        let mut rows: Vec<Vec<(u32, T)>> = vec![Vec::new(); m];
        let mut ccount = vec![0u32; m];
        let mut amax = T::zero();
        let mut structural_nnz = 0usize;
        for (k, &col) in basis_cols.iter().enumerate().take(m) {
            for idx in col_start[col]..col_start[col + 1] {
                let r = row_idx[idx];
                if r >= m {
                    continue;
                }
                rows[r].push((k as u32, val[idx]));
                ccount[k] += 1;
                structural_nnz += 1;
                amax = amax.max(val[idx].abs());
            }
        }
        for row in &mut rows {
            row.sort_unstable_by_key(|&(c, _)| c);
        }
        // Dense bases gain nothing here: bail before allocating factor work.
        if structural_nnz as f64 > FILL_DENSITY_BAIL * (m as f64) * (m as f64) {
            return None;
        }
        let floor = T::from_f64(PIVOT_ABS_FLOOR_REL).unwrap_or_else(T::zero) * amax
            + T::from_f64(1e-30).unwrap_or_else(T::zero);

        // Column member lists: every row ever containing column c (append-
        // only, possibly stale). Lets elimination touch only rows sharing the
        // pivot column instead of scanning all m rows; staleness is filtered
        // at consumption time.
        let mut col_rows: Vec<Vec<u32>> = vec![Vec::new(); m];
        for (r, row) in rows.iter().enumerate() {
            for &(c, _) in row {
                col_rows[c as usize].push(r as u32);
            }
        }

        let cap = 4 * structural_nnz + 2 * m;
        let mut l_row: Vec<u32> = Vec::with_capacity(cap); // staged: ORIGINAL row ids
        let mut l_val: Vec<T> = Vec::with_capacity(cap);
        let mut l_counts: Vec<u32> = Vec::with_capacity(m);
        let mut u_col: Vec<u32> = Vec::with_capacity(cap); // staged: ORIGINAL slot ids
        let mut u_val: Vec<T> = Vec::with_capacity(cap);
        let mut u_counts: Vec<u32> = Vec::with_capacity(m);
        let mut u_diag: Vec<T> = Vec::with_capacity(m);
        let mut prow: Vec<u32> = Vec::with_capacity(m);
        let mut qcol: Vec<u32> = Vec::with_capacity(m);
        // Per-step multiplier staging (original row ids; remapped at pack).
        let mut step_rows: Vec<u32> = Vec::new();
        let mut step_vals: Vec<T> = Vec::new();

        let mut active_row = vec![true; m];
        let mut active_col = vec![true; m];
        // Per-row ACTIVE entry count, maintained incrementally by the
        // elimination merges; feeds the Markowitz width and the singleton-row
        // fast path without rescanning stored rows.
        let mut act_len: Vec<i32> = rows.iter().map(|r| r.len() as i32).collect();
        // Rows currently at one active entry: pivoting on one costs zero
        // fill and needs no search. Duplicates are allowed; validity is
        // re-checked at consumption.
        let mut singles: Vec<u32> = Vec::with_capacity(m);
        for (r, &l) in act_len.iter().enumerate() {
            if l == 1 {
                singles.push(r as u32);
            }
        }
        let fill_cap = (FILL_DENSITY_BAIL * (m as f64) * (m as f64)) as usize;
        // Counting-sort workspace for the shortest-rows-first search order.
        let mut bucket = vec![0u32; m + 2];
        let mut order: Vec<u32> = Vec::with_capacity(m);

        for _step in 0..m {
            // ── Pivot search: among acceptable entries, minimize the
            // Markowitz quotient (row_len - 1) * max(1, col_cnt - 1). Rows
            // are visited shortest-first (counting sort, O(m)), and the walk
            // stops once the incumbent quotient cannot be beaten: rows further
            // down the order are longer, and a row of length L has minimum
            // quotient L - 1 under the max(1, .) convention.
            //
            // Before paying for the O(m)-per-step order rebuild, drain the
            // singleton-row queue: a row with a single active entry pivots on
            // that entry with zero fill -- the quotient-optimal choice -- and
            // simplex bases (identity slacks against sparse structural
            // columns) produce long runs of singletons. The full search runs
            // only when the queue is dry.
            let mut picked: Option<(usize, usize)> = None;
            while let Some(r) = singles.pop() {
                let ri = r as usize;
                if !active_row[ri] || act_len[ri] != 1 {
                    continue; // stale queue entry
                }
                // Locate the row's one active entry (stored rows may carry
                // dead entries at eliminated columns).
                let mut found: Option<(usize, T)> = None;
                for &(c, v) in rows[ri].iter() {
                    if active_col[c as usize] {
                        found = Some((c as usize, v));
                        break;
                    }
                }
                let Some((ci, v)) = found else {
                    continue; // stale
                };
                if v.abs() >= floor {
                    picked = Some((ri, ci));
                    break;
                }
                // Numerically dead sole entry: discard and keep draining --
                // another queued singleton may still be acceptable.
            }
            let best = match picked {
                Some(p) => Some(p),
                None => {
                    order.clear();
            bucket.fill(0);
            let mut longest = 1usize;
            let mut n_active = 0usize;
            for r in 0..m {
                if !active_row[r] || rows[r].is_empty() {
                    continue;
                }
                let len = rows[r].len();
                bucket[len] += 1;
                longest = longest.max(len);
                n_active += 1;
            }
            let mut sum = 0usize;
            for b in &mut bucket[1..=longest] {
                let c = *b;
                *b = sum as u32;
                sum += c as usize;
            }
            debug_assert_eq!(sum, n_active);
            order.resize(n_active, 0);
            for r in 0..m {
                if !active_row[r] || rows[r].is_empty() {
                    continue;
                }
                let len = rows[r].len();
                order[bucket[len] as usize] = r as u32;
                bucket[len] += 1;
            }
            order.truncate(n_active);

            let mut best_q = u64::MAX;
            let mut best: Option<(usize, usize)> = None; // (row, slot)
            'search: for &r in &order {
                let ri = r as usize;
                if act_len[ri] == 0 {
                    continue; // fully eliminated row still pending deactivation
                }
                let row = &rows[ri];
                // Rows carry entries at already-eliminated columns (they are
                // only dropped lazily, when a merge touches them), so the
                // threshold must be computed over ACTIVE entries -- counting
                // dead ones inflates `tau` until every live entry looks
                // unacceptable and the search falsely reports rank deficiency.
                // The active WIDTH comes from the maintained counter.
                let mut rmax = T::zero();
                for &(c, v) in row.iter() {
                    if active_col[c as usize] {
                        rmax = rmax.max(v.abs());
                    }
                }
                let tau = T::from_f64(PIVOT_REL_TAU).unwrap_or_else(T::zero) * rmax;
                let width = (act_len[ri] as u64 - 1).max(1);
                for &(c, v) in row.iter() {
                    let ci = c as usize;
                    if !active_col[ci] {
                        continue;
                    }
                    let av = v.abs();
                    if av < tau || av < floor {
                        continue;
                    }
                    let q = width * u64::from(ccount[ci]).saturating_sub(1).max(1);
                    if q < best_q {
                        best_q = q;
                        best = Some((ri, ci));
                        if q <= width {
                            break 'search;
                        }
                    }
                }
            }
            best
        }
        };
            let (pr, pc) = match best {
                Some(x) => x,
                None => break, // no acceptable pivot left: rank deficiency
            };

            // ── Eliminate: subtract multiples of the pivot row from every
            // other active row sharing the pivot column; the multipliers form
            // this step's L column.
            step_rows.clear();
            step_vals.clear();
            let pivot_row = std::mem::take(&mut rows[pr]);
            let piv_pos = pivot_row
                .binary_search_by_key(&(pc as u32), |&(c, _)| c)
                .expect("pivot candidate re-verified");
            let piv = pivot_row[piv_pos].1;
            let candidates = std::mem::take(&mut col_rows[pc]);
            for cir in candidates.iter().copied() {
                let ir = cir as usize;
                if ir == pr || !active_row[ir] {
                    continue;
                }
                let other = &mut rows[ir];
                let Ok(pos) = other.binary_search_by_key(&(pc as u32), |&(c, _)| c) else {
                    continue; // stale member entry
                };
                let f = other[pos].1 / piv;
                step_rows.push(cir);
                step_vals.push(f);
                // Active-entry delta for this row: the pivot-column entry is
                // dropped unconditionally (-1), fills add one each (+1), and
                // exact cancellations remove one more (-1).
                let mut delta: i32 = -1;
                // other -= f * pivot_row: sorted merge with exact-cancellation
                // pruning. The pivot column itself is dropped unconditionally
                // (f*v rounds, so an exact-zero test could leave a phantom
                // entry in a deactivated column and break triangularity).
                let mut merged: Vec<(u32, T)> = Vec::with_capacity(other.len() + 4);
                let mut ai = other.iter().copied().peekable();
                let mut bi = pivot_row.iter().copied().peekable();
                loop {
                    // Flush whichever side ran dry, then stop.
                    let (next_a, next_b) = match (ai.peek().copied(), bi.peek().copied()) {
                        (Some(a), Some(b)) => (a, b),
                        (Some(a), None) => {
                            let ca = a.0 as usize;
                            // Drop the pivot column unconditionally (see the
                            // Equal branch) and already-eliminated columns
                            // eagerly -- keeping them would inflate every
                            // later search pass over this row.
                            if ca != pc && active_col[ca] {
                                merged.push(a);
                            }
                            ai.next();
                            continue;
                        }
                        (None, Some(b)) => {
                            let cb = b.0 as usize;
                            if cb != pc && active_col[cb] {
                                merged.push((b.0, -f * b.1));
                                ccount[cb] += 1;
                                col_rows[cb].push(cir);
                                delta += 1;
                            }
                            bi.next();
                            continue;
                        }
                        (None, None) => break,
                    };
                    let (ca, va) = next_a;
                    let (cb, vb) = next_b;
                    match ca.cmp(&cb) {
                        std::cmp::Ordering::Less => {
                            if ca as usize != pc && active_col[ca as usize] {
                                merged.push((ca, va));
                            }
                            ai.next();
                        }
                        std::cmp::Ordering::Greater => {
                            if active_col[cb as usize] {
                                merged.push((cb, -f * vb));
                                ccount[cb as usize] += 1;
                                col_rows[cb as usize].push(cir);
                                delta += 1;
                            }
                            bi.next();
                        }
                        std::cmp::Ordering::Equal => {
                            if active_col[ca as usize] && ca as usize != pc {
                                let nv = va - f * vb;
                                if nv == T::zero() {
                                    ccount[ca as usize] -= 1;
                                    delta -= 1;
                                } else {
                                    merged.push((ca, nv));
                                }
                            }
                            ai.next();
                            bi.next();
                        }
                    }
                }
                *other = merged;
                act_len[ir] += delta;
                debug_assert!(act_len[ir] >= 0);
                if act_len[ir] == 1 {
                    singles.push(cir);
                }
            }
            col_rows[pc] = candidates; // restore (taken above)

            active_row[pr] = false;
            active_col[pc] = false;
            ccount[pc] = 0;
            prow.push(pr as u32);
            qcol.push(pc as u32);
            u_diag.push(piv);
            let mut unz = 0usize;
            for &(c, v) in &pivot_row {
                if c as usize != pc {
                    u_col.push(c); // original slot id; remapped at pack
                    u_val.push(v);
                    unz += 1;
                }
            }
            u_counts.push(unz as u32);
            l_counts.push(step_rows.len() as u32);
            for (i, &r) in step_rows.iter().enumerate() {
                l_row.push(r); // original row id; remapped at pack
                l_val.push(step_vals[i]);
            }
            // The pivot row leaves the active set: its entries no longer
            // count toward the active column counts.
            for &(c, _) in &pivot_row {
                if c as usize != pc {
                    ccount[c as usize] -= 1;
                }
            }
            if l_val.len() + u_val.len() > fill_cap {
                return None; // fill guard: caller falls back to dense
            }
            rows[pr] = pivot_row; // retained so the allocation is reused
        }

        // ── Rank deficiency: never-pivoted columns (basis slots) pair with
        // never-pivoted rows. Each completed step consumed exactly one row
        // and one column, so the leftovers are equal in number.
        let deficient: Vec<usize> = (0..m).filter(|&c| active_col[c]).collect();
        let leftover_rows: Vec<usize> = (0..m).filter(|&r| active_row[r]).collect();
        debug_assert_eq!(deficient.len(), leftover_rows.len());

        // ── Pack: remap staged original ids to elimination indices and lay
        // out the solve-ready arrays.
        let nsteps = prow.len();
        let mut rowpos = vec![u32::MAX; m];
        for (s, &r) in prow.iter().enumerate() {
            rowpos[r as usize] = s as u32;
        }
        let mut qinv = vec![u32::MAX; m];
        for (s, &c) in qcol.iter().enumerate() {
            qinv[c as usize] = s as u32;
        }
        // L CSC: group by producing step (entries were appended per step, so
        // replay the per-step counts), remapping rows; sort within a column
        // so accumulation order is row-ascending. On a rank-deficient matrix
        // a multiplier can target a row that never pivots -- such entries
        // belong to the unfactored remainder, so they are dropped here (the
        // caller repairs deficient bases before ever solving).
        let mut l_start: Vec<u32> = Vec::with_capacity(nsteps + 1);
        let mut lr: Vec<u32> = Vec::with_capacity(l_row.len());
        let mut lv: Vec<T> = Vec::with_capacity(l_val.len());
        {
            let mut flat = 0usize;
            for &cnt in &l_counts {
                l_start.push(lr.len() as u32);
                for i in 0..cnt as usize {
                    let er = rowpos[l_row[flat + i] as usize];
                    if er != u32::MAX {
                        lr.push(er);
                        lv.push(l_val[flat + i]);
                    }
                }
                flat += cnt as usize;
            }
            l_start.push(lr.len() as u32);
            debug_assert_eq!(flat, l_row.len());
            // Column-sort each L column by elimination row.
            for s in 0..nsteps {
                let st = l_start[s] as usize;
                let en = l_start[s + 1] as usize;
                let mut pairs: Vec<(u32, T)> = lr[st..en]
                    .iter()
                    .copied()
                    .zip(lv[st..en].iter().copied())
                    .collect();
                pairs.sort_unstable_by_key(|&(r, _)| r);
                for (i, &(r, v)) in pairs.iter().enumerate() {
                    lr[st + i] = r;
                    lv[st + i] = v;
                }
            }
        }
        // U CSR: group by step, remapping columns (dropping entries at
        // never-pivoted columns, same rank-deficiency argument); sort within
        // a row by elimination column.
        let mut u_start: Vec<u32> = Vec::with_capacity(nsteps + 1);
        let mut uc: Vec<u32> = Vec::with_capacity(u_col.len());
        let mut uv: Vec<T> = Vec::with_capacity(u_val.len());
        {
            let mut flat = 0usize;
            for &cnt in &u_counts {
                u_start.push(uc.len() as u32);
                for i in 0..cnt as usize {
                    let ec = qinv[u_col[flat + i] as usize];
                    if ec != u32::MAX {
                        uc.push(ec);
                        uv.push(u_val[flat + i]);
                    }
                }
                flat += cnt as usize;
            }
            u_start.push(uc.len() as u32);
            debug_assert_eq!(flat, u_col.len());
            for s in 0..nsteps {
                let st = u_start[s] as usize;
                let en = u_start[s + 1] as usize;
                let mut pairs: Vec<(u32, T)> = uc[st..en]
                    .iter()
                    .copied()
                    .zip(uv[st..en].iter().copied())
                    .collect();
                pairs.sort_unstable_by_key(|&(c, _)| c);
                for (i, &(c, v)) in pairs.iter().enumerate() {
                    uc[st + i] = c;
                    uv[st + i] = v;
                }
            }
        }

        // Repair perm: pivoted slots -> their pivot row; deficient slots ->
        // distinct leftover rows.
        let mut perm = vec![0usize; m];
        for (s, &r) in prow.iter().enumerate() {
            perm[qcol[s] as usize] = r as usize;
        }
        for (i, &c) in deficient.iter().enumerate() {
            perm[c] = leftover_rows[i];
        }

        Some((
            SparseLu {
                m,
                prow,
                qcol,
                l_start,
                l_row: lr,
                l_val: lv,
                u_start,
                u_col: uc,
                u_val: uv,
                u_diag,
            },
            deficient,
            perm,
        ))
    }

    /// Solve `B x = rhs` into `out` (`out` indexed by basis slot; free slots
    /// of a deficient factor read 0). `scratch` must be at least `m` long;
    /// `rhs`, `out`, `scratch` must be distinct slices.
    pub fn solve_into(&self, rhs: &[T], out: &mut [T], scratch: &mut [T]) {
        let ns = self.prow.len();
        let zero = T::zero();
        for v in out[..self.m].iter_mut() {
            *v = zero;
        }
        if ns == 0 {
            return;
        }
        // Permute rows into elimination space.
        for (s, &r) in self.prow.iter().enumerate() {
            scratch[s] = rhs[r as usize];
        }
        // L forward: unit diagonal, scatter down each column. Skipping a zero
        // component skips its entire column of L -- a unit-vector-style RHS
        // touches only the fill it actually reaches.
        for s in 0..ns {
            let ys = scratch[s];
            if ys == zero {
                continue;
            }
            for idx in self.l_start[s] as usize..self.l_start[s + 1] as usize {
                scratch[self.l_row[idx] as usize] -= self.l_val[idx] * ys;
            }
        }
        // U backward: gather across each row, divide by the diagonal.
        for s in (0..ns).rev() {
            let mut acc = scratch[s];
            for idx in self.u_start[s] as usize..self.u_start[s + 1] as usize {
                acc -= self.u_val[idx] * scratch[self.u_col[idx] as usize];
            }
            scratch[s] = acc / self.u_diag[s];
        }
        // Scatter from elimination space to basis slots.
        for s in 0..ns {
            out[self.qcol[s] as usize] = scratch[s];
        }
    }

    /// Solve `Bᵀ x = rhs` into `out` (`rhs` indexed by basis slot, `out` by
    /// row -- the dense path's orientation). `scratch` at least `m` long;
    /// slices distinct.
    pub fn solve_trans_into(&self, rhs: &[T], out: &mut [T], scratch: &mut [T]) {
        let ns = self.prow.len();
        let zero = T::zero();
        for v in out[..self.m].iter_mut() {
            *v = zero;
        }
        if ns == 0 {
            return;
        }
        // Gather the RHS from slot space into elimination space.
        for (s, &c) in self.qcol.iter().enumerate() {
            scratch[s] = rhs[c as usize];
        }
        // Uᵀ forward: equation s divides by the diagonal, then scatters its
        // contribution to later equations through U's off-diagonal row.
        for s in 0..ns {
            let t = scratch[s] / self.u_diag[s];
            scratch[s] = t;
            for idx in self.u_start[s] as usize..self.u_start[s + 1] as usize {
                scratch[self.u_col[idx] as usize] -= self.u_val[idx] * t;
            }
        }
        // Lᵀ backward: equation s subtracts the already-computed later
        // components through L's column (rows below the diagonal).
        for s in (0..ns).rev() {
            let mut acc = scratch[s];
            for idx in self.l_start[s] as usize..self.l_start[s + 1] as usize {
                acc -= self.l_val[idx] * scratch[self.l_row[idx] as usize];
            }
            scratch[s] = acc;
        }
        // Scatter from elimination space to rows.
        for (s, &r) in self.prow.iter().enumerate() {
            out[r as usize] = scratch[s];
        }
    }
}

#[cfg(test)]
mod tests {
    //! Reference checks against a naive dense Gaussian solve, plus
    //! representation-level edge cases (identity bases, rank deficiency,
    //! fill bail).

    use super::*;

    struct Basis {
        col_start: Vec<usize>,
        row_idx: Vec<usize>,
        val: Vec<f64>,
    }

    impl Basis {
        fn from_dense(mat: &[Vec<f64>]) -> (Basis, Vec<Vec<f64>>) {
            let m = mat.len();
            let mut col_start = vec![0usize];
            let mut row_idx = Vec::new();
            let mut val = Vec::new();
            for col in 0..m {
                for (i, row) in mat.iter().enumerate() {
                    if row[col] != 0.0 {
                        row_idx.push(i);
                        val.push(row[col]);
                    }
                }
                col_start.push(row_idx.len());
            }
            (
                Basis {
                    col_start,
                    row_idx,
                    val,
                },
                mat.to_vec(),
            )
        }

        fn build(&self, m: usize) -> (SparseLu<f64>, Vec<usize>, Vec<usize>) {
            SparseLu::build(m, &self.col_start, &self.row_idx, &self.val, &(0..m).collect::<Vec<_>>())
                .expect("basis should factor sparsely")
        }
    }

    fn residual(mat: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> f64 {
        let m = mat.len();
        let mut r = 0.0f64;
        for i in 0..m {
            let mut s = -rhs[i];
            for j in 0..m {
                s += mat[i][j] * x[j];
            }
            r = r.max(s.abs());
        }
        r
    }

    #[test]
    fn identity_basis_solves() {
        let m = 5;
        let mat: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..m).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let (basis, dense) = Basis::from_dense(&mat);
        let (lu, deficient, perm) = basis.build(m);
        assert!(deficient.is_empty());
        assert_eq!(perm, (0..m).collect::<Vec<_>>());
        assert_eq!(lu.offdiag_nnz(), 0);
        let rhs = [1.0, -2.0, 3.25, 0.0, 7.0];
        let mut x = vec![0.0; m];
        let mut scratch = vec![0.0; m];
        lu.solve_into(&rhs, &mut x, &mut scratch);
        assert!(residual(&dense, &x, &rhs) < 1e-12);
        let mut xt = vec![0.0; m];
        lu.solve_trans_into(&rhs, &mut xt, &mut scratch);
        assert!(residual_trans_ref(&dense, &xt, &rhs) < 1e-12);
    }

    fn residual_trans_ref(mat: &[Vec<f64>], x: &[f64], rhs: &[f64]) -> f64 {
        let m = mat.len();
        let mut r = 0.0f64;
        for j in 0..m {
            let mut s = -rhs[j];
            for i in 0..m {
                s += mat[i][j] * x[i];
            }
            r = r.max(s.abs());
        }
        r
    }

    #[test]
    fn random_sparse_matches_dense_reference_forward_and_transposed() {
        use iconic_core::rng::Lcg;
        // Draw mapping preserved from the retired local generator:
        // top-53 bits folded onto a [-1, 1] cent grid.
        fn f(rng: &mut Lcg) -> f64 {
            ((rng.next_u64() >> 11) % 20001) as f64 / 10000.0 - 1.0
        }
        let mut rng = Lcg::new(0x1234_5678_9abc_def0);
        for trial in 0..20 {
            // Sizes and densities in the regime the caller gates to (m >= 96,
            // a few nonzeros per row): well under the fill bail.
            let m = 96 + trial * 11;
            let nz_per_row = 1 + trial % 3;
            // Random sparse nonsingular-ish basis: diagonally dominant so the
            // reference factors cleanly.
            let mut mat = vec![vec![0.0; m]; m];
            for (i, row) in mat.iter_mut().enumerate() {
                row[i] = 2.0 + f(&mut rng).abs();
                for _ in 0..nz_per_row {
                    let j = (rng.next_u64() as usize) % m;
                    if j != i {
                        let v = f(&mut rng);
                        if v.abs() > 0.2 {
                            row[j] = v;
                        }
                    }
                }
            }
            let (basis, dense) = Basis::from_dense(&mat);
            let (lu, deficient, perm) = basis.build(m);
            assert!(deficient.is_empty(), "diagonally dominant basis is nonsingular");
            // perm must agree with the factor's own permutation data.
            for s in 0..lu.prow.len() {
                assert_eq!(perm[lu.qcol[s] as usize], lu.prow[s] as usize);
            }
            let rhs: Vec<f64> = (0..m).map(|_| f(&mut rng)).collect();
            let mut x = vec![0.0; m];
            let mut scratch = vec![0.0; m];
            lu.solve_into(&rhs, &mut x, &mut scratch);
            let scale = rhs.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
            assert!(
                residual(&dense, &x, &rhs) < 1e-8 * scale,
                "trial {trial}: fwd residual {}",
                residual(&dense, &x, &rhs)
            );
            let mut xt = vec![0.0; m];
            lu.solve_trans_into(&rhs, &mut xt, &mut scratch);
            assert!(
                residual_trans_ref(&dense, &xt, &rhs) < 1e-8 * scale,
                "trial {trial}: trans residual {}",
                residual_trans_ref(&dense, &xt, &rhs)
            );
        }
    }

    #[test]
    fn singular_basis_reports_deficient_slots_and_repairs_cleanly() {
        // Row 2 = row 0 + row 1 (over the nonzero pattern), so the last two
        // columns cannot both pivot; the factor must report deficiency and a
        // valid distinct-row perm.
        let m = 6;
        let mut mat = vec![vec![0.0; m]; m];
        for (i, row) in mat.iter_mut().enumerate() {
            row[i] = 3.0;
        }
        mat[2] = mat[0]
            .iter()
            .zip(&mat[1])
            .map(|(a, b)| a + b)
            .collect::<Vec<f64>>();
        let (basis, _dense) = Basis::from_dense(&mat);
        let (lu, deficient, perm) = basis.build(m);
        assert!(!deficient.is_empty(), "rank-5 basis must report deficiency");
        // Deficient slots get distinct leftover rows; pivoted slots their own.
        let mut seen = vec![false; m];
        for &slot in &deficient {
            let r = perm[slot];
            assert!(r < m && !seen[r], "repair rows must be distinct");
            seen[r] = true;
        }
        for s in 0..lu.prow.len() {
            assert_eq!(perm[lu.qcol[s] as usize], lu.prow[s] as usize);
        }
        // Solves stay total (no panic, finite output).
        let rhs = vec![1.0; m];
        let mut x = vec![0.0; m];
        let mut scratch = vec![0.0; m];
        lu.solve_into(&rhs, &mut x, &mut scratch);
        assert!(x.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn dense_input_bails_to_caller() {
        let m = 60;
        let mat: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..m).map(|j| 1.0 + ((i + j) % 7) as f64 * 0.25).collect())
            .collect();
        let (basis, _) = Basis::from_dense(&mat);
        assert!(SparseLu::build(
            m,
            &basis.col_start,
            &basis.row_idx,
            &basis.val,
            &(0..m).collect::<Vec<_>>()
        )
        .is_none());
    }

    #[test]
    fn unit_vector_rhs_touches_minimal_fill() {
        // e_0-style RHS through an upper bidiagonal factor: the forward solve
        // must reach everything (fill chains), but the result stays exact.
        // Sized past the structural density bail (2m entries vs 0.2 m²).
        let m = 16;
        let mut mat = vec![vec![0.0; m]; m];
        for i in 0..m {
            mat[i][i] = 2.0;
            if i + 1 < m {
                mat[i][i + 1] = 1.0; // U superdiagonal after any ordering
            }
        }
        let (basis, dense) = Basis::from_dense(&mat);
        let (lu, deficient, _) = basis.build(m);
        assert!(deficient.is_empty());
        let mut rhs = vec![0.0; m];
        rhs[0] = 1.0;
        let mut x = vec![0.0; m];
        let mut scratch = vec![0.0; m];
        lu.solve_into(&rhs, &mut x, &mut scratch);
        assert!(residual(&dense, &x, &rhs) < 1e-12);
    }
}
