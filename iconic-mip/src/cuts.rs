#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::empty_line_after_doc_comments
)]
//! Cutting-plane generation: cover cuts, MIR cuts, clique cuts, subtour elimination.
use crate::bounds::recover_route_capacity;
use crate::VarType;
use iconic_core::{Cone, Scalar};
use iconic_linalg::DenseMatrix;
use iconic_simplex::{CscCols, DualSolver, Status as DsStatus};
// Deterministic hashing: these collections' iteration order shapes cut
// selection and generation order, and thus the search trajectory. std's
// RandomState seeds per-process randomness into that order; FxHash has a
// fixed seed. See the determinism note in lib.rs's imports.
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Debug)]
pub struct Cut<T: Scalar> {
    pub row: Vec<T>,
    pub rhs: T,
    pub active: bool,
    pub violation: T,
    pub age: usize,
}

impl<T: Scalar> Cut<T> {
    /// A freshly generated cut: active, new.
    pub fn new(row: Vec<T>, rhs: T, violation: T) -> Self {
        Self {
            row,
            rhs,
            active: true,
            violation,
            age: 0,
        }
    }
}

/// Violation of `row·x ≤ rhs` at `x` (`row·x − rhs`, positive = violated).
pub(crate) fn violation_of<T: Scalar>(row: &[T], x: &[T], rhs: T) -> T {
    let mut viol = -rhs;
    for j in 0..x.len().min(row.len()) {
        viol += row[j] * x[j];
    }
    viol
}

/// Cut-selection score weights (Turner/Koch/Serrano/Winkler, "Adaptive Cut
/// Selection in MILP", Open J. Math. Optim. 2023):
/// `score = l1·dcd + l2·eff + l3·isp + l4·obp`.
///
/// `default()` mirrors the paper's SCIP-8.0 vector {0, 1.0, 0.1, 0.1}
/// (efficacy-dominant); `aggressive()` is the paper's per-instance-best
/// vector {0.3, 0.2, 0.25, 0.25} (dcd-heavy). The paper's Theorem 3.1 shows
/// no fixed vector is universally best, so these are a tuning knob, not a
/// solved choice — the suite A/B gate decides which ships as the default.
#[derive(Clone, Copy, Debug)]
pub struct CutWeights {
    pub dcd: f64,
    pub eff: f64,
    pub isp: f64,
    pub obp: f64,
}

impl CutWeights {
    /// The paper's SCIP-8.0 default vector: efficacy-dominant, no dcd term.
    pub fn scip_default() -> Self {
        Self {
            dcd: 0.0,
            eff: 1.0,
            isp: 0.1,
            obp: 0.1,
        }
    }

    /// The paper's mean per-instance-best vector: dcd-heavy.
    pub fn aggressive() -> Self {
        Self {
            dcd: 0.3,
            eff: 0.2,
            isp: 0.25,
            obp: 0.25,
        }
    }
}

impl Default for CutWeights {
    fn default() -> Self {
        Self::scip_default()
    }
}

/// The three `x_lp`-independent score components of a cut (paper §2), plus
/// the squared norm (shared with the orthogonality gate). One O(n) pass.
///
/// - `eff` = violation / ‖alpha‖ (efficacy, at the violation stored at insert)
/// - `isp` = #{j integer: alpha_j ≠ 0} / #{j: alpha_j ≠ 0} (integer support)
/// - `obp` = |alphaᵀq| / (‖alpha‖·‖q‖) (objective parallelism)
fn score_components<T: Scalar>(
    row: &[T],
    violation: T,
    obj: &[T],
    obj_norm: T,
    var_types: &[VarType],
) -> (T, T, T, T) {
    let zero = T::zero();
    let tiny = T::from_f64(1e-10).expect("scalar literal");
    let tiny_den = T::from_f64(1e-30).expect("scalar literal");
    let mut norm2 = zero;
    let mut int_nz = 0usize;
    let mut total_nz = 0usize;
    let mut dot_obj = zero;
    for (j, &coef) in row.iter().enumerate() {
        if coef.abs() <= tiny {
            continue;
        }
        norm2 += coef * coef;
        total_nz += 1;
        if j < var_types.len() && var_types[j] != VarType::Continuous {
            int_nz += 1;
        }
        if j < obj.len() {
            dot_obj += coef * obj[j];
        }
    }
    let norm = norm2.sqrt();
    let eff = if norm > zero {
        violation / (norm + tiny_den)
    } else {
        zero
    };
    let isp = if total_nz > 0 {
        T::from_usize(int_nz).unwrap() / T::from_usize(total_nz).unwrap()
    } else {
        zero
    };
    let obp = if norm > zero && obj_norm > zero {
        dot_obj.abs() / (norm * obj_norm + tiny_den)
    } else {
        zero
    };
    (norm2, eff, isp, obp)
}

/// The weighted sum `l1·dcd + l2·eff + l3·isp + l4·obp` (paper's score with
/// the `CutWeights` lambdas).
fn weighted_score_of<T: Scalar>(w: CutWeights, eff: T, isp: T, obp: T, dcd: T) -> T {
    T::from_f64(w.dcd).expect("scalar literal") * dcd
        + T::from_f64(w.eff).expect("scalar literal") * eff
        + T::from_f64(w.isp).expect("scalar literal") * isp
        + T::from_f64(w.obp).expect("scalar literal") * obp
}

pub struct CutPool<T: Scalar> {
    pub cuts: Vec<Cut<T>>,
    norms: Vec<T>,
    /// Cached `eff` (efficacy) per cut, computed at insert.
    effs: Vec<T>,
    /// Cached `isp` (integer support) per cut, computed at insert.
    isps: Vec<T>,
    /// Cached `obp` (objective parallelism) per cut, computed at insert.
    obps: Vec<T>,
    /// Score weights for replacement/trim/selection (`CutWeights`).
    weights: CutWeights,
    max_cuts: usize,
    obj: Vec<T>,             // objective vector (for obj-parallelism scoring)
    obj_norm: T,             // cached ||obj||
    var_types: Vec<VarType>, // variable types (for integer-support scoring)
}

impl<T: Scalar> CutPool<T> {
    pub fn new(max_cuts: usize, obj: &[T], var_types: &[VarType]) -> Self {
        let obj_norm = obj.iter().fold(T::zero(), |acc, &v| acc + v * v).sqrt();
        Self {
            cuts: Vec::new(),
            norms: Vec::new(),
            effs: Vec::new(),
            isps: Vec::new(),
            obps: Vec::new(),
            weights: CutWeights::default(),
            max_cuts,
            obj: obj.to_vec(),
            obj_norm,
            var_types: var_types.to_vec(),
        }
    }

    /// Override the score weights used by replacement, trimming and node
    /// selection. The admission gate (violation/sparsify/range/nnz/
    /// orthogonality) is untouched — the weights only rank cuts that
    /// already passed it.
    pub fn set_weights(&mut self, w: CutWeights) {
        self.weights = w;
    }

    /// The pool's admission capacity (`max_cuts_per_round × cut_rounds`).
    /// Callers use it to pause cut separation while the pool is full: at
    /// capacity, `add` can only replace residents, and replaced cuts below
    /// a round's `prev_count` watermark are invisible to `select_for_node`
    /// and can never reach an LP.
    pub fn capacity(&self) -> usize {
        self.max_cuts
    }

    /// The weighted score of pool cut `idx` (no dcd term — dcd needs the
    /// incumbent direction and is only computable at selection time).
    fn pool_score(&self, idx: usize) -> T {
        weighted_score_of(
            self.weights,
            self.effs[idx],
            self.isps[idx],
            self.obps[idx],
            T::zero(),
        )
    }

    /// Append a cut to `ICONIC_DUMP_CUTS`, so every generated cut can be checked for
    /// validity independently: a cut `alpha' x <= rhs` is valid exactly when
    /// `max { alpha' x : x integer-feasible } <= rhs`, and that maximum is itself a MIP.
    /// Enumeration only reaches ~15 binaries; this reaches any size.
    fn log_cut(cut: &Cut<T>) {
        let Some(path) = std::env::var_os("ICONIC_DUMP_CUTS") else {
            return;
        };
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let row: Vec<String> = cut
                .row
                .iter()
                .map(|v| format!("{:?}", v.to_f64().unwrap_or(0.0)))
                .collect();
            let _ = writeln!(
                f,
                "{{\"rhs\":{:?},\"row\":[{}]}}",
                cut.rhs.to_f64().unwrap_or(0.0),
                row.join(",")
            );
        }
    }

    pub fn add(&mut self, cut: Cut<T>) -> bool {
        Self::log_cut(&cut);
        let min_viol = T::from_f64(1e-4).expect("scalar literal");
        if cut.violation < min_viol {
            return false;
        }
        let tiny = T::from_f64(1e-10).expect("scalar literal");
        let new_nnz = cut.row.iter().filter(|c| c.abs() > tiny).count();
        self.admit(cut, new_nnz)
    }

    /// Aggregate-cut admission: the same gates as [`CutPool::add`], except the
    /// nonzero-growth cap is measured on the cut's GROUP-level support — the
    /// number of variable groups (identical-coefficient copies, e.g. one
    /// item's bin-copies in a multiple knapsack) it spans — rather than the
    /// materialized row. The item-level aggregate covers of
    /// `generate_aggregate_surrogate_covers` materialize every copy of every
    /// covered item, so their row density (e.g. 60 nonzeros on a 100-variable
    /// multiple-knapsack LP) exceeds the pool-mean-proportional cap that
    /// `add` enforces — measured: the single aggregate cover that cuts the
    /// multiknap_i25_b4 root LP from -773.718 to -772.183 was rejected by
    /// `add`'s cap (60 > max(20, 2·17.7)) and never reached the LP. The
    /// group-level complexity (15 cover groups + 10 lifted = 25) is the
    /// structural size of the inequality, which is what the cap exists to
    /// bound; the materialized density is an artifact of the copy structure.
    pub fn add_aggregate(&mut self, cut: Cut<T>, group_nnz: usize) -> bool {
        Self::log_cut(&cut);
        let min_viol = T::from_f64(1e-4).expect("scalar literal");
        if cut.violation < min_viol {
            return false;
        }
        self.admit(cut, group_nnz)
    }

    /// Structural-size admission: the same gates as [`CutPool::add`], except
    /// the nonzero-growth cap is measured on the cut's STRUCTURAL support —
    /// for subtour/RCI rows, the subset cardinality |S| — rather than the
    /// materialized row. A rounded capacity inequality over a customer
    /// subset S materializes |S|·(|S|−1) arc coefficients (the complete
    /// directed graph on S), which exceeds the pool-mean-proportional cap
    /// of `add`: measured on cvrp_n15_k4, every big-subset RCI row was
    /// rejected (30+ materialized nonzeros against a cap of 20 set by the
    /// pair/clique cuts that fill the pool first), the root bound stuck at
    /// the plain-DFJ value ~5.73, and the RCI-closed LP equals the known
    /// optimum 8.07 on all three suite CVRPs. The structural size
    /// (≤ n_cities, which routing models keep ≤ 16) is what the cap exists
    /// to bound; the density is an artifact of the arc representation.
    pub fn add_structural(&mut self, cut: Cut<T>, structural_nnz: usize) -> bool {
        Self::log_cut(&cut);
        let min_viol = T::from_f64(1e-4).expect("scalar literal");
        if cut.violation < min_viol {
            return false;
        }
        self.admit(cut, structural_nnz)
    }

    /// Shared admission pipeline after the minimum-violation gate: sparsify,
    /// coefficient-range filter, nonzero-growth cap (on the supplied nnz
    /// measure), orthogonality gate, then insert or score-replace.
    fn admit(&mut self, mut cut: Cut<T>, new_nnz: usize) -> bool {
        // ── Sparsify: drop numerically negligible coefficients ──────────
        let tiny = T::from_f64(1e-10).expect("scalar literal");
        for coef in cut.row.iter_mut() {
            if coef.abs() < tiny {
                *coef = T::zero();
            }
        }

        // ── Coefficient-range filter ──
        let coeff_range_limit = T::from_f64(1e8).expect("scalar literal");
        let mut max_abs = T::zero();
        let mut min_nz_abs = T::from_f64(1e100).expect("scalar literal");
        for coef in cut.row.iter() {
            let a = coef.abs();
            if a <= tiny {
                continue;
            }
            if a > max_abs {
                max_abs = a;
            }
            if a < min_nz_abs {
                min_nz_abs = a;
            }
        }
        if min_nz_abs > T::from_f64(1e99).expect("scalar literal") {
            return false;
        }
        if max_abs / min_nz_abs > coeff_range_limit {
            return false;
        }

        // ── Nonzero growth cap ─────────────────────────────────────
        // A cut far denser than the pool's existing cuts is usually a
        // re-statement of the LP rows it spans rather than a genuinely new
        // inequality, and it costs every LP solve it joins. Cap a cut's
        // support at `max(FLOOR, NNZ_GROWTH_FACTOR × mean pool nnz)`:
        // dense cuts are admitted only in proportion to how dense the pool
        // already is (settings-free constants).
        //
        // The absolute floor itself scales with the formulation width
        // (`n_columns/4`, min 20): "too dense" is a statement about how much
        // of the formulation a cut re-aggregates, not about a raw nonzero
        // count. On the Glover-linearized quadratic-knapsack shape the
        // extended LP is 4–7× wider than the original variable count (one
        // continuous `w_ij` per nonzero objective pair), and the tableau
        // Gomory cuts there span 40–50 of the 274 columns (~16% density) —
        // all rejected under the fixed 20 floor (measured: every family
        // emitted ZERO root cuts on qkp_n40 while each candidate carried
        // violation 0.6, and the tree paid 695 nodes for the missing bound).
        // At quarter-width the floor only rises past 20 above n=80, leaving
        // every smaller instance bit-identical (mdk's harmful 30–60-nnz
        // aggregated rows on n≤60 formulations are still filtered).
        const NNZ_FLOOR: usize = 20;
        const NNZ_GROWTH_FACTOR: f64 = 2.0;
        let width_floor = NNZ_FLOOR.max(self.var_types.len() / 4);
        let mean_nnz = if self.cuts.is_empty() {
            0.0
        } else {
            self.cuts
                .iter()
                .map(|c| c.row.iter().filter(|v| v.abs() > tiny).count() as f64)
                .sum::<f64>()
                / self.cuts.len() as f64
        };
        let nnz_cap = (width_floor as f64).max(NNZ_GROWTH_FACTOR * mean_nnz) as usize;
        if new_nnz > nnz_cap {
            return false;
        }

        // ── Orthogonality gate ─────────────────────────────────────
        // A candidate cut nearly parallel to any existing pool cut adds no
        // new direction to the cutting space: reject it outright (hard gate,
        // not a score penalty — near-duplicate rows only bloat the LP). This
        // is the classic cut-manager orthogonality test, run as a hard
        // rejection against EVERY pool cut; the threshold is |cos| > 0.894
        // (the comparison is on squared cosine, dot² > COS_SQ_THRESH·‖a‖²‖b‖²,
        // to avoid two square roots). The nominal threshold in the literature
        // is 0.9; the slightly tighter 0.894 is kept deliberately — A/B
        // measured that loosening to 0.9 admits near-duplicates that crowd
        // the pool and regress node counts (mdk_n50_k5: 265 → 462 nodes).
        const COS_SQ_THRESH: f64 = 0.80;
        let (na, new_eff, new_isp, new_obp) = score_components(
            &cut.row,
            cut.violation,
            &self.obj,
            self.obj_norm,
            &self.var_types,
        );
        let cos_sq_thresh = T::from_f64(COS_SQ_THRESH).expect("scalar literal");
        let tiny_norm = T::from_f64(1e-30).expect("scalar literal");
        for (existing, &nb) in self.cuts.iter().zip(self.norms.iter()) {
            let mut dot = T::zero();
            for (a, b) in cut.row.iter().zip(existing.row.iter()) {
                dot += *a * *b;
            }
            if dot * dot > cos_sq_thresh * (na * nb + tiny_norm) {
                return false;
            }
        }
        if self.cuts.len() < self.max_cuts {
            self.cuts.push(cut);
            self.norms.push(na);
            self.effs.push(new_eff);
            self.isps.push(new_isp);
            self.obps.push(new_obp);
            return true;
        }
        // Pool is full — weighted-score replacement. The full score uses the
        // cached components (the dcd term is 0 here: no incumbent direction
        // at insert time).
        //
        // `max_cuts == 0` (cuts disabled) lands here with an empty pool,
        // where there is no cut to evict and nowhere to put this one.
        let Some(worst_idx) = self.worst_cut_index() else {
            return false;
        };
        let new_score = weighted_score_of(self.weights, new_eff, new_isp, new_obp, T::zero());
        let worst_score = self.pool_score(worst_idx);
        if new_score <= worst_score {
            // Nothing was stored: report the rejection. Callers read the
            // return as "the cut entered the pool" (`generate_mixing_cuts`
            // stops enumerating candidates for a group on `true`), so a
            // false positive here silently dropped that group's remaining
            // candidates.
            return false;
        }
        self.cuts[worst_idx] = cut;
        self.norms[worst_idx] = na;
        self.effs[worst_idx] = new_eff;
        self.isps[worst_idx] = new_isp;
        self.obps[worst_idx] = new_obp;
        true
    }

    /// Select the cuts to copy into a node's LP, replacing insertion-order
    /// addition. Implements the greedy default-selector of the
    /// Turner/Koch/Serrano/Winkler "Adaptive Cut Selection in MILP" paper
    /// (Open J. Math. Optim. 2023): score the candidates
    /// (`l1·dcd + l2·eff + l3·isp + l4·obp`), take the best, drop every
    /// remaining candidate within the orthogonality threshold of the taken
    /// set, and repeat until the count limit or the nonzero budget is
    /// exhausted.
    ///
    /// Only cuts `[prev_count..]` (this round's new cuts) are eligible —
    /// older pool cuts were already copied when they were new. `dcd`
    /// measures a cut's progress toward the incumbent direction
    /// (`viol / |alphaᵀy|`, `y = (x̂ − x_lp)/‖x̂ − x_lp‖`); with no incumbent
    /// (or an incumbent equal to the LP point) the direction is undefined
    /// and dcd = 0. The admission gate (violation/sparsify/range/nnz/
    /// orthogonality) is untouched — this only ranks cuts that passed it.
    pub fn select_for_node(
        &self,
        prev_count: usize,
        limit: usize,
        nnz_limit: usize,
        x_lp: &[T],
        incumbent: Option<&[T]>,
    ) -> Vec<usize> {
        let zero = T::zero();
        let eps = T::from_f64(1e-10).expect("scalar literal");
        let tiny_norm = T::from_f64(1e-30).expect("scalar literal");
        let cos_sq_thresh = T::from_f64(0.80).expect("scalar literal");

        // Direction y = (x̂ − x_lp)/‖x̂ − x_lp‖ for dcd.
        let dir: Vec<T> = match incumbent {
            Some(xh) => {
                let mut d: Vec<T> = xh.iter().zip(x_lp.iter()).map(|(a, b)| *a - *b).collect();
                let mut nrm = zero;
                for v in &d {
                    nrm += *v * *v;
                }
                let s = nrm.sqrt();
                if s <= eps {
                    Vec::new()
                } else {
                    for v in d.iter_mut() {
                        *v /= s;
                    }
                    d
                }
            }
            None => Vec::new(),
        };

        let w_dcd = T::from_f64(self.weights.dcd).expect("scalar literal");
        let w_eff = T::from_f64(self.weights.eff).expect("scalar literal");
        let w_isp = T::from_f64(self.weights.isp).expect("scalar literal");
        let w_obp = T::from_f64(self.weights.obp).expect("scalar literal");

        // Score every candidate at the current LP point.
        let mut cands: Vec<(usize, T)> = Vec::new();
        for i in prev_count..self.cuts.len() {
            if !self.cuts[i].active {
                continue;
            }
            let cut = &self.cuts[i];
            // Violation at the CURRENT point (paper's viol = alpha·x_lp − rhs).
            let viol: T = cut
                .row
                .iter()
                .zip(x_lp.iter())
                .fold(-cut.rhs, |acc, (&a, &b)| acc + a * b)
                .max(zero);
            let dcd = if dir.is_empty() {
                zero
            } else {
                let mut dot = zero;
                for (&a, &d) in cut.row.iter().zip(dir.iter()) {
                    dot += a * d;
                }
                if dot.abs() <= tiny_norm {
                    zero
                } else {
                    viol / dot.abs()
                }
            };
            let score =
                w_dcd * dcd + w_eff * self.effs[i] + w_isp * self.isps[i] + w_obp * self.obps[i];
            cands.push((i, score));
        }

        let mut taken: Vec<usize> = Vec::new();
        let mut nnz_used = 0usize;
        let tiny = T::from_f64(1e-10).expect("scalar literal");
        while !cands.is_empty() && taken.len() < limit {
            // Best remaining candidate.
            let mut best_pos = 0usize;
            for (p, &(_, sc)) in cands.iter().enumerate().skip(1) {
                if sc > cands[best_pos].1 {
                    best_pos = p;
                }
            }
            let (best_idx, _) = cands.swap_remove(best_pos);
            let nz = self.cuts[best_idx]
                .row
                .iter()
                .filter(|c| c.abs() > tiny)
                .count();
            if nnz_used + nz > nnz_limit {
                continue; // too dense for the remaining budget: skip, try next
            }
            // Orthogonality against the already-taken set.
            let na = self.norms[best_idx];
            let mut ok = true;
            for &t in &taken {
                let nb = self.norms[t];
                let mut dot = zero;
                for (a, b) in self.cuts[best_idx].row.iter().zip(self.cuts[t].row.iter()) {
                    dot += *a * *b;
                }
                if dot * dot > cos_sq_thresh * (na * nb + tiny_norm) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            taken.push(best_idx);
            nnz_used += nz;
            // Drop the rest within the orthogonality threshold of the taken cut.
            cands.retain(|&(i, _)| {
                let nb = self.norms[i];
                let mut dot = zero;
                for (a, b) in self.cuts[i].row.iter().zip(self.cuts[best_idx].row.iter()) {
                    dot += *a * *b;
                }
                dot * dot <= cos_sq_thresh * (nb * na + tiny_norm)
            });
        }
        taken
    }

    /// Index of the lowest-scored cut in the pool (empty pool → `None`).
    /// The score is the weighted sum of the cached components (the dcd term
    /// is 0 — no incumbent direction outside a node context).
    fn worst_cut_index(&self) -> Option<usize> {
        self.cuts
            .iter()
            .enumerate()
            .map(|(i, _)| (i, self.pool_score(i)))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Trim pool to at most `max_cuts` by removing lowest-scored cuts.
    pub fn trim_to_max(&mut self) {
        while self.cuts.len() > self.max_cuts {
            let Some(worst_idx) = self.worst_cut_index() else {
                break;
            };
            self.cuts.remove(worst_idx);
            self.norms.remove(worst_idx);
            self.effs.remove(worst_idx);
            self.isps.remove(worst_idx);
            self.obps.remove(worst_idx);
        }
    }

    /// Age all cuts and remove those exceeding max_age.
    pub fn age_and_purge(&mut self, max_age: usize, x: &[T]) {
        let eps = T::from_f64(1e-8).expect("scalar literal");
        for cut in &mut self.cuts {
            if violation_of(&cut.row, x, cut.rhs) > eps {
                cut.age = 0;
            } else {
                cut.age += 1;
                if cut.age > max_age {
                    cut.active = false;
                }
            }
        }
        let mut nc = Vec::with_capacity(self.cuts.len());
        let mut nn = Vec::with_capacity(self.norms.len());
        let mut ne = Vec::with_capacity(self.effs.len());
        let mut ni = Vec::with_capacity(self.isps.len());
        let mut no = Vec::with_capacity(self.obps.len());
        for ((cut, norm), (eff, (isp, obp))) in self.cuts.drain(..).zip(self.norms.drain(..)).zip(
            self.effs
                .drain(..)
                .zip(self.isps.drain(..).zip(self.obps.drain(..))),
        ) {
            if cut.active || cut.age <= max_age {
                nc.push(cut);
                nn.push(norm);
                ne.push(eff);
                ni.push(isp);
                no.push(obp);
            }
        }
        self.cuts = nc;
        self.norms = nn;
        self.effs = ne;
        self.isps = ni;
        self.obps = no;
    }
}

/// One greedy pass of the shared cover-builder skeleton: sort `items` (index,
/// weight) by the given ordering, fill until the running weight exceeds
/// `cap + eps`, then trim from the end while it still does. Returns `None`
/// unless a genuine cover emerged (`sum > cap + eps`, ≥ 2 items) — proceeding
/// on a non-cover produced cuts that excluded feasible points.
///
/// The efficiency order precomputes `value/weight` per item (the divisor is
/// item-invariant, so sorting the ratio is identical to dividing per
/// comparison, O(k) divisions instead of O(k log k)).
#[derive(Clone, Copy)]
enum CoverOrder {
    /// Weight descending.
    Weight,
    /// LP value descending (caller supplies values).
    Value,
    /// value/weight descending (caller supplies values).
    Efficiency,
}

fn greedy_cover<T: Scalar + PartialOrd>(
    items: &mut [(usize, T)],
    order: CoverOrder,
    x: &[T],
    weights: &dyn Fn(usize) -> T,
    cap: T,
    eps: T,
    zero: T,
) -> Option<(Vec<usize>, T)> {
    match order {
        CoverOrder::Weight => items.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        }),
        CoverOrder::Value => items.sort_unstable_by(|a, b| {
            x[b.0]
                .partial_cmp(&x[a.0])
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        CoverOrder::Efficiency => items.sort_unstable_by(|a, b| {
            let ea = x[a.0] / a.1;
            let eb = x[b.0] / b.1;
            eb.partial_cmp(&ea).unwrap_or(std::cmp::Ordering::Equal)
        }),
    }
    let mut cover = Vec::new();
    let mut sum = zero;
    for &(j, _w) in items.iter() {
        cover.push(j);
        sum += weights(j);
        if sum > cap + eps {
            break;
        }
    }
    if sum <= cap + eps || cover.len() < 2 {
        return None;
    }
    while cover.len() > 1 {
        let last = cover[cover.len() - 1];
        if sum - weights(last) > cap + eps {
            sum -= weights(last);
            cover.pop();
        } else {
            break;
        }
    }
    Some((cover, sum))
}

pub fn generate_cover_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-8).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let n = x.len();
    let m = b.len();
    for i in 0..m {
        let mut has_bin = false;
        for j in 0..n {
            if a.get(i, j).abs() > eps && var_types[j] == VarType::Binary {
                has_bin = true;
                break;
            }
        }
        if !has_bin {
            continue;
        }
        // Effective capacity: a row `sum_pos w_j x_j + sum_neg a_j x_j <= b[i]`
        // (a_j<0 for j in the negative set) implies the RELAXED inequality
        // `sum_pos w_j x_j <= b[i] + sum_neg (-a_j)*ub[j]`, since every
        // negative term contributes at least `a_j*ub[j]` (its most negative
        // possible value is bounded by its own upper bound, valid for any
        // variable type -- not just binary). This is exactly the structure of
        // a "conditional capacity" row (e.g. bin-packing's `sum w_i x_ik -
        // capacity*y_k <= 0`, facility-location-style capacity rows): with
        // b[i]=0 and all weights positive, the raw `aij<=bi` filter below
        // would exclude every item and cover cuts could never fire at all --
        // confirmed directly: zero cover cuts generated on bin-packing's
        // capacity rows before this fix, every round, on every instance.
        // Unbounded ub[j] naturally makes effective_bi huge (no special case
        // needed): if a negative-coefficient variable has no finite upper
        // bound, no meaningful cover exists regardless, and this correctly
        // finds none.
        // Gated to rows with AT MOST ONE negative-coefficient term: that's
        // the actual "conditional capacity" shape (one indicator variable
        // switching the row's effective capacity), and it's a narrow,
        // deliberate gate -- rows with MANY scattered negative coefficients
        // (e.g. generic mixed-sign side constraints, not a clean capacity
        // indicator) aren't this pattern, and applying the same treatment
        // there measured as a real regression: on almost_knap_n30's "extra
        // rows" (random coefficients in [-5,15], ~1/4 negative among 30
        // variables), computing an effective capacity from ALL of them
        // caused a 56x slowdown (61ms -> 3.4s, same iteration count) from
        // many more candidate covers being found and pool-inserted, with no
        // corresponding benefit (the row isn't a real capacity constraint in
        // that shape). Bin-packing's actual motivating case has exactly one.
        let mut neg_count = 0usize;
        let mut effective_bi = b[i];
        for j in 0..n {
            let aij = a.get(i, j);
            if aij < -eps {
                neg_count += 1;
                effective_bi += (-aij) * ub[j];
            }
        }
        // A cover is a set whose combined weight cannot fit, so the capacity it is
        // measured against has to be one the row's *other* terms cannot relax. With a
        // negative coefficient present that is `effective_bi`, not `b[i]`: setting every
        // member of C to 1 pushes the positive part above `b[i]`, and the negative terms
        // then absorb the excess, so the row is satisfied and the cut forbids a feasible
        // combination.
        //
        // The gate below used to fall back to the raw `b[i]` for rows with two or more
        // negative coefficients (to avoid a measured 56x slowdown from the relaxed
        // capacity finding far more candidate covers there). That trades a slowdown for
        // wrong answers. Such rows are skipped instead: no cheap valid cover exists for
        // them, and skipping costs only cuts that were never valid. Verified by
        // maximising each cut's own left-hand side over the integer-feasible set --
        // maxsat_v25_c105 has exactly 38 rows with two negative coefficients and produced
        // exactly 38 invalid cover cuts, all of which disappear here.
        if neg_count > 1 {
            continue;
        }
        let bi = effective_bi;
        if bi <= zero {
            continue;
        }
        // Generate up to 3 covers per constraint using
        // different greedy orderings. Each cover produces a different cut.
        let mut items: Vec<(usize, T)> = Vec::new();
        for j in 0..n {
            let aij = a.get(i, j);
            if aij > eps && aij <= bi && var_types[j] == VarType::Binary {
                items.push((j, aij));
            }
        }
        // Three greedy orderings: LP-value desc, weight desc, efficiency desc.
        let mut seen_covers: FxHashSet<Vec<usize>> = FxHashSet::default();

        for pass in 0u8..3 {
            let order = [
                CoverOrder::Value,
                CoverOrder::Weight,
                CoverOrder::Efficiency,
            ][pass as usize];
            // A non-cover here means the greedy loop ran out of items before
            // the weight ever exceeded `bi` — proceeding anyway produced cuts
            // excluding genuinely feasible points (confirmed by exhaustive
            // enumeration; see greedy_cover's contract).
            let Some((cover, _sum)) =
                greedy_cover(&mut items, order, x, &|j| a.get(i, j), bi, eps, zero)
            else {
                continue;
            };
            // Skip duplicate covers.
            let mut sorted_cover = cover.clone();
            sorted_cover.sort_unstable();
            if !seen_covers.insert(sorted_cover) {
                continue;
            }

            // Lifting IS performed here, by `lift_cover` below. This comment used to say the
            // opposite -- that lifting had been "dropped ... for a guarantee of correctness" --
            // seven lines above the call that does it, which is how it came to be the leading
            // suspect for every later unsoundness (wrongly: the invalid cuts on mdk_n30_k3 and
            // maxsat_v25_c105 came from the capacity and the packing block, not from here).
            //
            // What was actually removed is a *different*, unsound lifting: one that computed
            // each non-cover coefficient independently against the unlifted base cover, so two
            // lifted variables could both be 1 without ever having been checked against each
            // other (measured then at 171/640 cuts excluding a feasible point). `lift_cover`
            // replaces it with proper sequential lifting -- each coefficient accounts for every
            // previously lifted variable, via a value-indexed knapsack DP -- which is sound.
            //
            // The base (unlifted) cover cut is valid for any cover on its own; the lifted
            // coefficients strengthen it and carry their own correctness argument.
            let mut row = vec![zero; n];
            for &j in &cover {
                row[j] = one;
            }
            let cs = cover.len();
            let rhs = T::from_usize(cs - 1).unwrap();
            let candidates: Vec<usize> = (0..n)
                .filter(|&j| {
                    !cover.contains(&j) && a.get(i, j) > eps && var_types[j] == VarType::Binary
                })
                .collect();
            for (j, alpha) in lift_cover(i, a, bi, &cover, &candidates) {
                row[j] = alpha;
            }
            // Packing-set strengthening removed: as written it was unsound.
            //
            // The argument it relied on works for *one* packing set in isolation. With `Σ_{j∈G}
            // x_j ≤ 1` and two or more cover members inside `G`, setting some `j ∈ G \ C` to 1
            // forces every member of `C ∩ G` to 0, so `Σ_C x ≤ |C| − 2` and adding `x_j` with
            // coefficient 1 keeps the cut at `|C| − 1`. It does not survive two packing sets: if
            // `G1` and `G2` each hold two cover members, one member of each may be 1 at the
            // same time, contributing 2 while the cover still contributes `|C| − 2` -- total
            // `|C|`, above the right-hand side. It also overwrote coefficients that sequential
            // lifting had already assigned, discarding the accounting that made them valid.
            //
            // Verified by maximising each cut's own left-hand side over the integer-feasible
            // set: on setpack_hard_n50_d20 (a set-packing instance, so packing sets everywhere)
            // **26 of 26** cuts from this path were invalid, and none are produced now.
            //
            // A correct packing-lifted cover inequality exists in the literature, but it is a
            // derivation over the packing structure, not coefficient-1 terms grafted onto a lifted
            // cover cut. Deriving it properly is separate work; emitting invalid cuts in the
            // meantime costs answers.
            let viol = violation_of(&row, x, rhs);
            if viol > eps {
                pool.add(Cut::new(row, rhs, viol));
            }
        } // end ordering loop
    }
}

/// Sequential lifting: strengthens a base cover cut `Σ_{j∈cover} x_j ≤
/// |cover|-1` (valid for row `a·x ≤ b`) into `Σ_{j∈cover} x_j +
/// Σ_{j∈candidates} α_j x_j ≤ |cover|-1`, correctly accounting for
/// previously-lifted variables at each step (unlike an earlier attempt --
/// see the comment above this function's call site -- that computed each
/// α_j independently against the unlifted base cover and was found unsound
/// by exhaustive enumeration on 27% of generated cuts).
///
/// For each candidate j (processed in decreasing-weight order), the
/// standard formula is `α_j = (|cover|-1) − z_j`, where `z_j` is the
/// maximum Σα_i achievable by a 0/1 selection from the *already-lifted*
/// set (cover ∪ previously-lifted candidates) subject to Σ(their weights)
/// ≤ b − a_j. Computed via a **value-indexed** knapsack DP (`w[v]` =
/// minimum weight to exactly achieve lifted-value `v`): lifted coefficients are small exact integers, so indexing by them needs no discretization, whereas indexing by the row's real-valued weights would require rounding (as the removed
/// `sequential_lift_cover` did, rounding to hundredths) -- and rounding a
/// continuous weight can silently over-lift a coefficient into an invalid
/// cut exactly the way the unsound attempt did.
fn lift_cover<T: Scalar + PartialOrd>(
    row_idx: usize,
    a: &DenseMatrix<T>,
    b: T,
    cover: &[usize],
    candidates: &[usize],
) -> Vec<(usize, T)> {
    let zero = T::zero();
    let max_val = cover.len() - 1;
    if max_val == 0 {
        return Vec::new();
    }
    let mut lifted: Vec<(T, usize)> = cover.iter().map(|&j| (a.get(row_idx, j), 1usize)).collect();
    let mut result: Vec<(usize, T)> = Vec::new();

    let mut order: Vec<usize> = candidates.to_vec();
    order.sort_by(|&x, &y| {
        a.get(row_idx, y)
            .partial_cmp(&a.get(row_idx, x))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Value-indexed DP over `lifted`, cached across candidates: a zero lift
    // leaves `lifted` unchanged, and consecutive candidates reused an
    // identical table that was being rebuilt from scratch each time.
    // dp_cache[v] = minimum weight to exactly achieve lifted-value v using a
    // 0/1 subset of `lifted[..dp_len]`; w[0] = 0 (the empty selection) is
    // seeded once and persists -- it is the base case of every build.
    let mut dp_cache: Vec<Option<T>> = vec![None; max_val + 1];
    dp_cache[0] = Some(zero);
    let mut dp_len = 0usize;

    for &j in &order {
        let aj = a.get(row_idx, j);
        let budget = b - aj;
        let alpha = if budget < zero {
            // a_j alone already exceeds what's left after b -- x_j can
            // never be 1 alongside the cover in any feasible point, so it
            // may be lifted all the way to the cut's own bound.
            max_val
        } else {
            // Fold only the entries added since the last build into the
            // cache; older entries are already reflected.
            while dp_len < lifted.len() {
                let (wi, vi) = lifted[dp_len];
                dp_len += 1;
                dp_fold(&mut dp_cache, wi, vi);
            }
            max_val - dp_best_value(&dp_cache, budget)
        };
        if alpha > 0 {
            result.push((j, T::from_usize(alpha).unwrap()));
            lifted.push((aj, alpha));
        }
    }
    result
}

/// Fold one (weight, value) entry into the value-indexed lifting DP:
/// `dp[v]` = minimum weight to exactly achieve value `v` with a 0/1 subset
/// of the entries folded so far.
fn dp_fold<T: Scalar>(dp: &mut [Option<T>], wi: T, vi: usize) {
    let max_val = dp.len() - 1;
    if vi == 0 || vi > max_val {
        return;
    }
    for v in (vi..=max_val).rev() {
        if let Some(prev) = dp[v - vi] {
            let cand = prev + wi;
            if dp[v].is_none_or(|cur| cand < cur) {
                dp[v] = Some(cand);
            }
        }
    }
}

/// Largest achievable value whose minimum weight fits `budget` (the lifted
/// coefficient is then `max_val − z`).
fn dp_best_value<T: Scalar>(dp: &[Option<T>], budget: T) -> usize {
    let mut z = 0usize;
    for (v, wv) in dp.iter().enumerate() {
        if let Some(wv) = wv {
            if *wv <= budget {
                z = v;
            }
        }
    }
    z
}

/// The dual-weighted surrogate combination row: `w = Σ |πᵢ|·a[i,:]`,
/// `rhs = Σ |πᵢ|·b[i]`. `build_surrogate_row` returns `None` unless ≥2 rows
/// contribute and `rhs > 0` (a single-row surrogate reproduces the ordinary
/// cover cut).
struct SurrogateRow<T> {
    w: Vec<T>,
    rhs: T,
}

/// Surrogate cover cuts: form a single surrogate knapsack row from the LP
/// duals and feed it through the existing cover-cut pipeline.
///
/// Algorithm:
/// 1. For each original row `i` whose coefficients are all nonnegative, add
///    `|dual_pi[i]| * a[i,j]` to the surrogate weight of column `j`.
/// 2. The surrogate RHS is `Σ_i |dual_pi[i]| * b[i]`.
/// 3. The resulting surrogate row is a valid knapsack constraint (all
///    coefficients ≥ 0) for the same binary variables. Passing it through
///    `generate_cover_cuts` — including sequential lifting — produces
///    inequalities that are valid for the surrogate, hence valid for the
///    original LP (since the surrogate is a conical combination with positive
///    multipliers of original rows).
/// 4. Only fired when at least two rows contribute (a single-row surrogate
///    reproduces the ordinary cover cut, which has already been generated).

fn build_surrogate_row<T: Scalar + PartialOrd>(
    dual_pi: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    n: usize,
    eps: T,
    zero: T,
) -> Option<SurrogateRow<T>> {
    let m = b.len();
    let mut w = vec![zero; n];
    let mut rhs = zero;
    let mut contributing = 0usize;
    for i in 0..m {
        let pi = dual_pi.get(i).copied().unwrap_or(zero);
        let weight = pi.abs();
        if weight <= eps {
            continue;
        }
        // Row must be all-nonnegative to qualify as a knapsack row component.
        let all_nonneg = (0..n).all(|j| a.get(i, j) >= -eps);
        if !all_nonneg {
            continue;
        }
        for j in 0..n {
            w[j] += weight * a.get(i, j);
        }
        rhs += weight * b[i];
        contributing += 1;
    }
    (contributing >= 2 && rhs > zero).then_some(SurrogateRow { w, rhs })
}

pub fn generate_surrogate_cover_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    dual_pi: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    if dual_pi.is_empty() {
        return;
    }
    let eps = T::from_f64(1e-8).expect("scalar literal");
    let zero = T::zero();
    let one = T::one();
    let n = ub.len();

    // Sign convention: the surrogate weights are |pi| — conical combinations
    // are sign-agnostic. The bare `pi > eps` filter was convention-dependent:
    // main node LPs (minimization, Ax <= b) have non-positive pi on every
    // row (verified on multiknap: capacity rows -1.51, tight partition rows
    // -45..-62) and never fired there -- measured on multiknap_i25_b4, where
    // the aggregate knapsack covers are exactly the cuts that move the root
    // bound (the per-bin covers provably cannot: all 4x2213 minimal covers
    // added to the LP leave its optimum at -773.7180135006 to the digit,
    // because the 4 identical bins let the LP redistribute any fractional
    // fill; the aggregate-row cover "at most 14 of these 15" is what closes
    // it). But RENS/RINS sub-MIPs and distance LPs use the opposite
    // convention (positive duals on tight rows), where the old filter DID
    // fire -- a bare sign flip regressed those sub-MIP trajectories
    // (sched_j15_m5: 1013 -> 1359 nodes at the same optimum). |pi| preserves
    // both.
    let Some(surr) = build_surrogate_row(dual_pi, a, b, n, eps, zero) else {
        return;
    };
    let SurrogateRow { w, rhs } = surr;
    // Fire the surrogate only where the ITEM-LEVEL structure it was built
    // for exists: some surrogate weight shared by two or more variables
    // (a multiple knapsack's equal-weight bin-copies). Without it the pass
    // degenerates to weak variable-level covers on the surrogate row --
    // new behavior on every problem the sign fix activates (measured on
    // sched_j15_m5: the variable-level pass found no covers but its
    // activation alone is the kind of trajectory perturbation this solver
    // is documented to be sensitive to), with no demonstrated value.
    let has_group = {
        let wtol = T::from_f64(1e-9).expect("scalar literal");
        let mut groups: Vec<(T, usize)> = Vec::new(); // (weight, member count)
        for j in 0..n {
            if w[j] <= eps || var_types[j] != VarType::Binary {
                continue;
            }
            let wj = w[j];
            let mut found = false;
            for (wg, cnt) in groups.iter_mut() {
                let diff = (wj - *wg).abs();
                let scale = wj.abs().max(wg.abs()).max(one);
                if diff <= wtol * scale {
                    *cnt += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                groups.push((wj, 1));
            }
        }
        groups.iter().any(|&(_, cnt)| cnt >= 2)
    };
    if !has_group {
        return;
    }

    // Item-level aggregate covers FIRST: on a multiple-knapsack surrogate the
    // variable-level covers below are structurally weak (measured: the
    // multiknap_i25_b4 root LP optimum does not move even with every minimal
    // per-bin cover and the variable-level surrogate covers added -- the LP
    // slides a split item's fraction between bins to dodge any single-copy
    // cover), while ONE item-level cover ("at most 14 of these 15 items")
    // cuts it. The grouping by equal surrogate weight is exact for the
    // multiple-knapsack shape (an item's bin-copies share the partition-row
    // dual and the equal capacity-row duals, so their surrogate weights
    // coincide bit-for-bit) and sound in general (see the function's doc).
    generate_aggregate_surrogate_covers(x, &w, rhs, ub, var_types, pool);

    // Build a temporary 1-row DenseMatrix and feed it through the existing
    // cover-cut pipeline.
    let surrogate_a = DenseMatrix::from_row_major(1, n, w);
    let surrogate_b = vec![rhs];
    generate_cover_cuts(x, &surrogate_a, &surrogate_b, ub, var_types, pool);

}

/// Item-level aggregate covers on the surrogate row.
///
/// The variable-level cover pipeline is structurally weak on multiple-
/// knapsack problems: an item's bin-copies are separate variables, so a
/// cover contains individual copies and the LP can slide a split item's
/// fraction between bins to dodge any single-copy cover. Measured on
/// multiknap_i25_b4 (100 binary vars, 25 items x 4 bins, 4 capacity rows):
/// the root LP optimum -773.7180135006 does not move to the digit even with
/// all 4x2213 minimal per-bin covers AND the variable-level surrogate covers
/// added; the single ITEM-level aggregate cover "at most 14 of these 15
/// items" (the 14 items at y=1 plus the fractional item) cuts it to
/// -772.183, the 16-item variants to -770.958, the 17-item to -769.586.
///
/// Construction: group variables with identical weight on the surrogate row
/// (in a multiple knapsack the 4 bin-copies of one item share the
/// partition-row dual and the equal capacity-row duals, so their surrogate
/// weights coincide bit-for-bit). On the group level the surrogate is
/// `Σ_j w_j y_j ≤ rhs` with `y_j = Σ_{copies} x_v`, which is the
/// variable-level row restated, so covers found over GROUPS with the
/// aggregate LP values `y_j` and the cut `Σ_{j∈C} y_j ≤ |C|−1` materialized
/// over all copies are valid for the surrogate, hence for the original LP --
/// for ANY grouping by equal weight (the pass is therefore also safe when
/// equal weights are coincidence on other problem classes). Sequential
/// lifting over the remaining groups uses the same value-indexed DP as
/// `lift_cover` (lifted coefficients are small exact integers; the DP is
/// indexed by value, so the real-valued surrogate weights need no
/// discretization).
fn generate_aggregate_surrogate_covers<T: Scalar + PartialOrd>(
    x: &[T],
    w: &[T],
    rhs: T,
    _ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-8).expect("scalar literal");
    let zero = T::zero();
    let one = T::one();
    let n = x.len();
    if rhs <= zero {
        return;
    }
    // Group variables by identical surrogate weight (relative tolerance --
    // copies are built from identical arithmetic so they coincide exactly;
    // the tolerance only guards against reassociation noise).
    let wtol = T::from_f64(1e-9).expect("scalar literal");
    let mut groups: Vec<(T, Vec<usize>)> = Vec::new();
    'outer: for j in 0..n {
        if w[j] <= eps || var_types[j] != VarType::Binary {
            continue;
        }
        let wj = w[j];
        for (wg, members) in groups.iter_mut() {
            let diff = (wj - *wg).abs();
            let scale = wj.abs().max(wg.abs()).max(one);
            if diff <= wtol * scale {
                members.push(j);
                continue 'outer;
            }
        }
        groups.push((wj, vec![j]));
    }
    if groups.len() < 2 {
        return;
    }
    // Aggregate LP values y_g = sum of member x's, and group weights.
    let mut gy: Vec<T> = Vec::with_capacity(groups.len());
    for (_, members) in &groups {
        let mut s = zero;
        for &v in members {
            s += x[v];
        }
        gy.push(s);
    }
    let gws: Vec<T> = groups.iter().map(|(wg, _)| *wg).collect();

    // Three greedy orderings over groups: aggregate LP value desc, weight
    // desc, efficiency desc.
    let mut items: Vec<(usize, T)> = (0..groups.len()).map(|g| (g, gws[g])).collect();
    let mut seen: FxHashSet<Vec<usize>> = FxHashSet::default();
    for pass in 0u8..3 {
        let ord = [
            CoverOrder::Value,
            CoverOrder::Weight,
            CoverOrder::Efficiency,
        ][pass as usize];
        let Some((cover, _)) = greedy_cover(&mut items, ord, &gy, &|g| gws[g], rhs, eps, zero)
        else {
            continue;
        };
        let mut key = cover.clone();
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        // Cut over groups: sum_{C} y + sum_{lifted} alpha_j y_j <= |C|-1,
        // materialized over all copies of each group.
        let rhs_cut = T::from_usize(cover.len() - 1).unwrap();
        let mut row = vec![zero; n];
        for &g in &cover {
            for &v in &groups[g].1 {
                row[v] = one;
            }
        }
        // Sequential lifting over the non-cover groups (value-indexed DP on
        // the group level, same construction as `lift_cover`).
        let max_val = cover.len() - 1;
        let mut lifted: Vec<(T, usize)> = Vec::new(); // (weight, value)
        for &g in &cover {
            lifted.push((gws[g], 1usize));
        }
        let mut cands: Vec<usize> = (0..groups.len())
            .filter(|&g| !cover.contains(&g) && gws[g] > eps)
            .collect();
        cands.sort_unstable_by(|&a, &b| {
            gws[b]
                .partial_cmp(&gws[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if max_val > 0 {
            for &g in &cands {
                let budget = rhs - gws[g];
                let alpha = if budget < zero {
                    max_val
                } else {
                    // w[v] = minimum weight to exactly achieve value v.
                    let mut wmin: Vec<Option<T>> = vec![None; max_val + 1];
                    wmin[0] = Some(zero);
                    for &(wi, vi) in &lifted {
                        dp_fold(&mut wmin, wi, vi);
                    }
                    max_val - dp_best_value(&wmin, budget)
                };
                if alpha > 0 {
                    for &v in &groups[g].1 {
                        row[v] = T::from_usize(alpha).unwrap();
                    }
                    lifted.push((gws[g], alpha));
                }
            }
        }
        let viol = violation_of(&row, x, rhs_cut);
        if viol > eps {
            // Admission by group-level support: the materialized row has
            // every copy of every covered group (60 nonzeros on a 100-var
            // multiple-knapsack LP), which `add`'s pool-mean-proportional
            // cap rejects -- measured, see `CutPool::add_aggregate`. The
            // group-level complexity (cover + lifted groups) is the cap's
            // intended measure. `lifted` is seeded with the cover groups,
            // so its length already includes the cover: group_nnz =
            // lifted.len() (cover + lifted-with-positive-alpha).
            let group_nnz = lifted.len();
            pool.add_aggregate(
                Cut::new(row, rhs_cut, viol),
                group_nnz,
            );
        }
    }
}

/// Strong Chvátal–Gomory cuts: CG cuts strengthened by sequential coefficient
/// lifting.
///
/// For a row `Σ a_j x_j ≤ b` with all-nonnegative coefficients on binary
/// variables, the CG cut from multiplier π > 0, `Σ ⌊π a_j⌋ x_j ≤ ⌊π b⌋`, is
/// valid (the left-hand side is an integer bounded by π·b). It is never
/// violated at a row-feasible fractional point, though: `Σ ⌊πa_j⌋x_j ≤
/// π·Σa_jx_j ≤ πb` with an integer left-hand side, so the cut's strength has
/// to come from lifting. Each coefficient can be raised from `⌊πa_j⌋` to
/// `⌊πb⌋ − f_j`, where `f_j` is the maximum the *other* (already-lifted)
/// coefficients can reach in any point of the row's LP relaxation with x_j =
/// 1, computed by a greedy fractional-knapsack fill of capacity `b − a_j`
/// (the "implied bounds" the row itself places on the others). The fill is an
/// upper bound on the integer-feasible contribution, so the raise is sound;
/// each raise accounts for every previously-raised coefficient (sequential
/// lifting), so the induction holds. The lifted cut can violate the LP point
/// even where the plain CG cut cannot — that is the whole point.
///
/// Multipliers are tried from `{t/b : t = 1..4}` (deterministic, mirroring
/// the zero-half divisor set). Rows are also combined with nonnegative dual
/// weights into one surrogate row first (the surrogate-cover construction),
/// so the cut sees cross-row structure a single row cannot.
///
/// Scale gates: rows with |rhs| or coefficient magnitudes beyond a sane
/// scale are skipped — fractional fills and lifts on badly-scaled data are
/// exactly where CG-strengthening implementations have shipped numerics
/// bugs. Lifting is restricted to binary support: the `x_j ∈ {0,1}` case is
/// the one the one-shot raise formula certifies, and the suite's knapsack
/// rows are binary.
pub fn generate_strong_cg_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    dual_pi: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    lb: &[T],
    ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-8).expect("scalar literal");
    let zero = T::zero();
    let n = x.len();
    let m = b.len();

    let max_support = 80usize;

    // Single-row passes: the base CG cut is per-row, so every qualifying row
    // is separated on its own before the surrogate combines rows.
    for i in 0..m {
        strong_cg_on_row(x, i, a, b, lb, ub, var_types, pool, max_support);
    }

    // Surrogate row: same construction as the surrogate-cover generator;
    // exposes multi-row knapsack structure.
    if let Some(surr) = build_surrogate_row(dual_pi, a, b, n, eps, zero) {
        let surrogate_a = DenseMatrix::from_row_major(1, n, surr.w);
        let surrogate_b = vec![surr.rhs];
        strong_cg_on_row(
            x,
            0,
            &surrogate_a,
            &surrogate_b,
            lb,
            ub,
            var_types,
            pool,
            max_support,
        );
    }
}

/// Strong-CG separation on a single row (see [`generate_strong_cg_cuts`]).
fn strong_cg_on_row<T: Scalar + PartialOrd>(
    x: &[T],
    row_idx: usize,
    a: &DenseMatrix<T>,
    b: &[T],
    lb: &[T],
    ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
    max_support: usize,
) {
    let eps = T::from_f64(1e-8).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let n = x.len();
    let rhs = b[row_idx];

    // Scale gates (see the generator's doc comment). `rhs` must be positive
    // (partial_cmp, so a NaN rhs — the row is corrupt — also fails the gate)
    // and not absurdly large; the coefficients must span less than ~8 orders
    // of magnitude so the fills stay numerically meaningful.
    let scale_ok = rhs
        .partial_cmp(&zero)
        .is_some_and(|o| o == std::cmp::Ordering::Greater)
        && rhs < T::from_f64(1e8).expect("scalar literal");
    if !scale_ok {
        return;
    }

    // CG-roundable support: binary-typed variables with lb >= 0 and a
    // positive coefficient. A negative coefficient or a negative lb breaks
    // the `⌊πa⌋x ≤ πax` step of the CG argument (those rows are the
    // surrogate/cover machinery's domain); a non-binary variable breaks the
    // lifting certificate (the one-shot raise formula is for x_j ∈ {0,1}).
    let mut support: Vec<usize> = Vec::new();
    let mut min_nz = T::from_f64(1e100).expect("scalar literal");
    let mut max_abs = zero;
    for j in 0..n {
        let aij = a.get(row_idx, j);
        if aij.abs() <= eps {
            continue;
        }
        if aij < -eps {
            return;
        }
        let is_binary =
            var_types[j] == VarType::Binary || (var_types[j].is_integer() && ub[j] <= one + eps);
        if !is_binary {
            return;
        }
        if lb[j] < -eps {
            return;
        }
        if aij > max_abs {
            max_abs = aij;
        }
        if aij < min_nz {
            min_nz = aij;
        }
        support.push(j);
    }
    if support.len() < 2 || support.len() > max_support {
        return;
    }
    if max_abs < T::from_f64(1e-4).expect("scalar literal") {
        return;
    }
    if max_abs / min_nz > T::from_f64(1e8).expect("scalar literal") {
        return;
    }

    // Lifting order: decreasing coefficient weight (strongest blockers
    // first), the same convention as `lift_cover`.
    let mut order = support.clone();
    order.sort_by(|&x, &y| {
        a.get(row_idx, y)
            .partial_cmp(&a.get(row_idx, x))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for t in 1u8..=4u8 {
        // Candidate multiplier π = t / b. The CG cut is `Σ⌊πa⌋x ≤ ⌊πb⌋`.
        let pi = T::from_usize(t as usize).unwrap() / rhs;
        let beta = (pi * rhs).floor();
        if beta < one {
            continue; // no rounding effect (⌊πb⌋ = 0)
        }

        let mut cut_row = vec![zero; n];
        for &j in &support {
            let c = (pi * a.get(row_idx, j)).floor();
            if c > zero {
                cut_row[j] = c;
            }
        }

        // Sequential coefficient lifting. For each variable j, `f_j` — the
        // largest the other coefficients can reach over the row's LP
        // relaxation with x_j = 1 — bounds how far c_j may be raised:
        //   f_j = max{ Σ_{k≠j} cut[k]·x_k : Σ_{k≠j} a_k x_k ≤ b − a_j,
        //               0 ≤ x_k ≤ u_k }
        // computed by a greedy fractional-knapsack fill. Since f_j is an
        // upper bound on the integer-feasible contribution, raising c_j to
        // β − f_j keeps the cut valid for every previously-lifted account.
        for &j in &order {
            let cj = cut_row[j];
            // Gate before work: only a coefficient below beta can be raised,
            // and the raise is bounded by the fill computation. Skipping the
            // fill for already-saturated coefficients changes nothing — its
            // result was discarded by this same test.
            if cj >= beta {
                continue;
            }
            let aij = a.get(row_idx, j);
            let cap = rhs - aij;
            let f = fractional_fill(cap, &support, &cut_row, a, row_idx, ub, eps, j);
            if beta - f > cj {
                cut_row[j] = beta - f;
            }
        }

        // The strengthened cut can violate the LP point even though the
        // plain CG cut never can (see the generator's doc comment); check
        // the strengthened cut's violation and add if it fires.
        let viol = violation_of(&cut_row, x, beta);
        if viol > eps {
            pool.add(Cut::new(cut_row, beta, viol));
        }
    }
}

/// Greedy fractional-knapsack fill: the maximum of `Σ cut[k]·x_k` over
/// `Σ_{k≠exclude} a_k x_k ≤ cap`, `0 ≤ x_k ≤ u_k`, treating x as continuous.
/// This is the LP relaxation of the integer selection problem, so it is a
/// valid *upper bound* on the integer maximum — exactly what strong-CG
/// lifting needs (conservative raises stay sound).
fn fractional_fill<T: Scalar + PartialOrd>(
    cap: T,
    support: &[usize],
    cut_row: &[T],
    a: &DenseMatrix<T>,
    row_idx: usize,
    ub: &[T],
    eps: T,
    exclude: usize,
) -> T {
    let zero = T::zero();
    let one = T::one();
    if cap <= zero {
        return zero;
    }
    let mut items: Vec<(T, T, T, usize)> = Vec::with_capacity(support.len());
    for &j in support {
        if j == exclude {
            continue;
        }
        let c = cut_row[j];
        let w = a.get(row_idx, j);
        if c <= eps || w <= eps {
            continue;
        }
        // (value, weight, efficiency, index) — efficiency filled below as
        // the sort key; the index reads the bound during the fill itself.
        items.push((c, w, zero, j));
    }
    // Sort by efficiency (value per unit of weight), descending; the greedy
    // fill is optimal for the single-constraint fractional knapsack. The
    // ratio is precomputed per item — dividing once instead of twice per
    // comparison.
    for it in items.iter_mut() {
        it.2 = it.0 / it.1;
    }
    items.sort_unstable_by(|p, q| q.2.partial_cmp(&p.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut rem = cap;
    let mut val = zero;
    for &(c, w, _, j) in &items {
        if rem <= eps {
            break;
        }
        let take = (rem / w).min(ub[j]).min(one);
        if take <= zero {
            continue;
        }
        val += c * take;
        rem -= w * take;
    }
    val
}

// Mixed-integer rounding, restricted to rows where every variable with a
// nonzero coefficient is integer-typed (i.e. the classical Gomory fractional
// cut, verified sound by exhaustive search: 17M+ random feasible-point checks,
// zero violations). NOT extended to genuinely mixed rows: the single-row MIR
// formula for a continuous variable is only valid when the row comes from an
// optimal simplex TABLEAU (nonbasic variables sitting at their bounds) --
// applying it directly to a raw problem row, as an earlier version of this
// function did, is unsound whenever a continuous variable has a *positive*
// coefficient: that variable can alone absorb the entire RHS (set every other
// variable to 0, `s = b/a_j`), and no finite rescaling of `a_j` keeps the
// rounded cut valid at that point since `b/floor(b) > 1`. Proven both
// analytically and by grid search before this restriction was added; see
// commit message / project memory for the derivation.
pub fn generate_mir_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    lb: &[T],
    ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-7).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let n = x.len();
    let m = b.len();
    for i in 0..m {
        let mut ax = zero;
        for j in 0..n {
            ax += a.get(i, j) * x[j];
        }
        if b[i] - ax > T::from_f64(1e-3).expect("scalar literal") {
            continue;
        }
        // Bound-shift the continuous variables to their most-constraining
        // bounds before rounding (Marchand–Wolsey): for a ≤ row, a positive
        // coefficient is most constraining at the variable's LOWER bound, a
        // negative one at its UPPER bound. Shifted variables enter the cut
        // through the x⁻-term of the MIR inequality (see below); the shift
        // only needs finite bounds where it is applied.
        let rhs = b[i];
        let mut b_shift = rhs;
        for j in 0..n {
            let aij = a.get(i, j);
            if aij.abs() < eps || var_types[j].is_integer() {
                continue;
            }
            if aij > zero && lb[j] > -T::from_f64(crate::INF_BOUND).expect("scalar literal") {
                b_shift -= aij * lb[j];
            } else if aij < zero && ub[j] < T::from_f64(crate::INF_BOUND).expect("scalar literal") {
                b_shift -= aij * ub[j];
            }
        }
        let rfloor = b_shift.floor();
        let fb = b_shift - rfloor;
        if fb < eps {
            continue;
        }
        let denom = one - fb;
        if denom < eps {
            continue;
        }
        // The MIR rounding formula is derived under x_j ≥ 0 (a negative
        // variable reverses the coefficient rounding, so the cut can exclude
        // row-feasible points). Skip rows whose support contains a variable
        // with lb < 0 -- mirrors the `cg_roundable` gate in
        // generate_zerohalf_cuts. Verified unsound without it: the row
        // x0 + 1.5·x1 ≤ 2.5 with x1 ∈ [-1,5] integer produces the "cut"
        // x0 + x1 ≤ 2, which excludes the feasible point (x0=4, x1=-1).
        let all_nonneg_lb = (0..n).all(|j| a.get(i, j).abs() < eps || lb[j] >= -eps);
        if !all_nonneg_lb {
            continue;
        }
        let mut row = vec![zero; n];
        let mut hn = false;
        for j in 0..n {
            let aij = a.get(i, j);
            if aij.abs() < eps {
                continue;
            }
            if var_types[j].is_integer() {
                // MIR coefficient on the integer part (same formula as the
                // all-integer case; the slides' set X = {Σ a_j y_j + x⁺ ≤ b + x⁻}
                // with f = b−⌊b⌋, f_j = a_j−⌊a_j⌋ gives
                // Σ(⌊a_j⌋ + (f_j−f)⁺/(1−f)) y_j ≤ ⌊b⌋ + x⁻/(1−f)).
                let afl = aij.floor();
                let fa = aij - afl;
                let coef = if fa <= fb {
                    afl
                } else {
                    afl + (fa - fb) / denom
                };
                if coef.abs() > eps {
                    row[j] = coef;
                    hn = true;
                }
            } else if aij < zero {
                // Continuous variable with NEGATIVE coefficient: it belongs to
                // the x⁻ aggregate on the right of the set's inequality, so the
                // MIR gives it the coefficient aij/(1−f) on y_j directly. For a
                // variable shifted from its upper bound (u_j finite), the same
                // term read on the shifted w_j = u_j − y_j contributes
                // (−aij)/(1−f)·(u_j − y_j) — folded below into the RHS.
                let coef = aij / denom;
                if coef.abs() > eps {
                    row[j] = coef;
                    hn = true;
                }
            }
            // Continuous variable with POSITIVE coefficient (c > 0): it is the
            // x⁺ aggregate, which does not appear in the MIR inequality at all
            // (the fractional slack is absorbed by x⁻ / the rounded RHS) --
            // coefficient 0. Verified: the row x + y ≤ 2.5 (x integer, y ≥ 0)
            // admits no stronger MIR cut than x ≤ 2.
        }
        // RHS: ⌊b'⌋ − Σ_{c<0, u_j finite} c_j·u_j/(1−f) (the shifted w_j parts;
        // for the unshifted negative-coefficient variables the term is already
        // on the left via row[j] = c_j/(1−f)).
        let mut cut_rhs = rfloor;
        for j in 0..n {
            let aij = a.get(i, j);
            if aij < -eps && !var_types[j].is_integer() && ub[j] < T::from_f64(crate::INF_BOUND).expect("scalar literal") {
                cut_rhs -= aij * ub[j] / denom;
            }
        }
        if !hn {
            continue;
        }
        let viol = violation_of(&row, x, cut_rhs);
        if viol > eps {
            if std::env::var_os("ICONIC_MIR_TRACE").is_some() {
                let nz: Vec<usize> = (0..n)
                    .filter(|&j| row[j].abs() > T::from_f64(1e-9).expect("scalar literal"))
                    .collect();
                eprintln!("[mir] row {i} rhs={cut_rhs:?} viol={viol:?} support={nz:?}");
            }
            pool.add(Cut::new(row, cut_rhs, viol));
        }
    }
}

/// Gomory mixed-integer cuts separated from the node LP's simplex tableau.
///
/// For each tableau row whose basic variable is a fractional integer
/// variable (`x_B + Σ ā_j v_j = x̄_B`, `f₀ = x̄_B − ⌊x̄_B⌋ ∈ (0,1)`), the GMI
/// cut is the MIR of the row's ≤ direction `Σ ā_j v_j + x_B ≤ x̄_B` (the
/// equality implies both directions, so the ≤ row is valid for every
/// feasible point; the MIR's validity is the one verified for
/// [`generate_mir_cuts`]):
///
///   Σ β_j x_j + Σ γ_j y_j ≥ f₀   over the nonbasic variables, with
///     β_j = f_j if f_j ≤ f₀ else f₀(1−f_j)/(1−f₀)   (integer nonbasic, f_j
///           = frac(ā_j), valid for x_j ≥ 0 whatever its bound status),
///     γ_j = c_j if c_j > 0 else −c_j·f₀/(1−f₀)      (continuous nonbasic).
///
/// The key difference from the row-based MIR: the tableau row exists for
/// every fractional basic integer variable, even when no original constraint
/// row is tight at the LP optimum — which is exactly the case (measured on
/// tsptw_n10) where Gomory cuts close the root relaxation and the
/// row-based generators cannot fire.
///
/// Slack columns are converted back to the original rows they stand for
/// (`s_r = b_r − A_r·x`), so the cut is a valid inequality over the MIP's
/// original variables. Rows whose tableau touches a cut-row slack (r ≥
/// m_orig) are skipped — converting those needs the cut row itself.
pub fn generate_gomory_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    tableau: &[iconic_simplex::TableauRow<T>],
    n_orig: usize,
    m_orig: usize,
    a: &DenseMatrix<T>,
    b: &[T],
    lb: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-7).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let n = x.len();
    for tr in tableau {
        if tr.basic_col >= n_orig {
            continue; // basic slack: not an original integer variable
        }
        if !var_types[tr.basic_col].is_integer() {
            continue;
        }
        let xb = tr.basic_val;
        let f0 = xb - xb.floor();
        if f0 < eps || one - f0 < eps {
            continue;
        }
        // Soundness gates: integer nonbasics must have lb ≥ 0 (the MIR
        // derivation assumes x_j ≥ 0); no cut-row slacks (conversion needs
        // the cut row itself).
        let mut ok = true;
        for &(j, _, _, _) in &tr.coeffs {
            if j < n_orig {
                if var_types[j].is_integer() && lb[j] < -eps {
                    ok = false;
                    break;
                }
            } else if j - n_orig >= m_orig {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // The tableau row is expressed in the SHIFTED nonbasic values
        // (x_j − x̄_j at their bound values): x_B + Σ ā_j·(x_j − x̄_j) = xb[r].
        // For at-upper nonbasics, x_j − u_j = −x'_j with x'_j = u_j − x_j ≥ 0,
        // so the complemented row reads x_B − Σ_{ub} ā_j·x'_j + Σ_{lb} ā_j·x_j
        // = xb[r] — the RHS is xb[r] itself (the shift is already in the
        // tableau), NOT xb[r] − Σ ā_j·u_j. An earlier version subtracted the
        // u_j terms again, double-shifting the RHS: f₀ became frac(xb − Σāu)
        // instead of frac(xb) (measured: 0.0292 vs the correct 0.7939 on a
        // knapsack root) and the derived cut removed the true optimum
        // (measured: −17536.796 → −17535.740).
        let f0c = xb - xb.floor();
        if f0c < eps || one - f0c < eps {
            continue;
        }
        let denom = one - f0c;
        // Coefficients over the LP variables (≥ form).
        let mut beta_lp = vec![zero; n_orig + m_orig];
        let mut ge_rhs = f0c;
        let mut has = false;
        for &(j, c, at_upper, u_j) in &tr.coeffs {
            // Effective coefficient on the (possibly complemented) variable.
            let c_eff = if at_upper && u_j < T::from_f64(crate::INF_BOUND).expect("scalar literal") {
                -c
            } else {
                c
            };
            let is_int = j < n_orig && var_types[j].is_integer();
            let coef = if is_int {
                let f_j = c_eff - c_eff.floor();
                if f_j <= f0c {
                    f_j
                } else {
                    f0c * (one - f_j) / denom
                }
            } else if c_eff > zero {
                c_eff
            } else {
                -c_eff * f0c / denom
            };
            if coef.abs() > eps {
                if at_upper && u_j < T::from_f64(crate::INF_BOUND).expect("scalar literal") {
                    // x'_j = u_j − x_j: the cut's coefficient flips sign on
                    // x_j and the constant u_j·coef moves to the RHS.
                    beta_lp[j] = -coef;
                    ge_rhs -= coef * u_j;
                } else {
                    // v_j = x_j − l_j (the tableau is in shifted coordinates):
                    // the constant −coef·l_j moves to the RHS. This was
                    // missing for variables with nonzero lower bounds —
                    // measured on milp_n20_i8_m15, whose continuous variables
                    // have lb = −10: the derived cut dropped +γ·10 per such
                    // nonbasic and removed the true optimum (measured
                    // −379.680 → −310.68). Only the regression instances (all
                    // lb = 0) exercised this path.
                    beta_lp[j] = coef;
                    if j < n_orig {
                        ge_rhs += coef * lb[j];
                    }
                }
                has = true;
            }
        }
        if !has {
            continue;
        }
        // Convert to original space (≤ form):
        //   Σ β_j x_j + Σ_r β_slack_r·(b_r − A_r·x) ≥ f₀
        // → Σ_j (Σ_r β_slack_r·A_rj − β_j) x_j ≤ Σ_r β_slack_r·b_r − f₀
        let mut row = vec![zero; n];
        let mut rhs = -ge_rhs;
        for r in 0..m_orig {
            let bs = beta_lp[n_orig + r];
            if bs.abs() <= eps {
                continue;
            }
            rhs += bs * b[r];
            for j in 0..n {
                let a_rj = a.get(r, j);
                if a_rj != zero {
                    row[j] += bs * a_rj;
                }
            }
        }
        for j in 0..n {
            row[j] = row[j] - beta_lp[j];
        }
        let viol = violation_of(&row, x, rhs);
        if viol > eps {
            if std::env::var_os("ICONIC_GOMORY_DUMP").is_some() {
                eprintln!(
                    "[gomory-dump] basic={} xb={} f0c={:?} row={:?} rhs={}",
                    tr.basic_col, tr.basic_val, f0c, row, rhs
                );
            }
            pool.add(Cut::new(row, rhs, viol));
        }
    }
}

/// Tableau intersection cuts (disjunctive cuts from a single tableau
/// row — the cheap variant of the lift-and-project cut-generating LP).
///
/// For a fractional binary variable `x_j` that is basic in the current node
/// LP, its tableau row `x_B + Σ_k c_k·z_k = x̄_B` (over the shifted nonbasic
/// deviations `z_k ≥ 0`) is a valid equation of the LP relaxation. Every
/// integer-feasible point has `x_B ∈ {0,1}`, hence lies on one of the two
/// sides of the disjunction `x_B ≤ 0` or `x_B ≥ 1`. On side 0 the row gives
/// `Σ c_k·z_k ≥ x̄_B` and on side 1 `Σ c_k·z_k ≤ x̄_B − 1`; the intersection
/// cut
///
/// ```text
///     Σ_{c_k>0} (c_k/x̄_B)·z_k + Σ_{c_k<0} (−c_k/(1−x̄_B))·z_k ≥ 1
/// ```
/// is implied by whichever side's inequality is active (one of the two sums
/// alone reaches 1 on its side), so it is valid for the disjunctive hull of
/// the variable — hence for the integer hull — and it cuts off the fractional
/// LP point itself (`z = 0` violates it by 1 in the shifted metric). This is
/// the classic "basic intersection cut": the node's own basis supplies the
/// row, so there is no cut-generating LP at all.
///
/// Soundness gates mirror the Gomory separator: nonbasic deviations are only
/// meaningful at their true bounds (at-upper variables with huge artificial
/// upper bounds are skipped), and the tableau row identity must hold at the
/// LP point (a stale factorization produces garbage rows — the documented
/// hot-start failure mode). Cut-row slacks ARE converted here (unlike the
/// Gomory separator), because `cut_rows` gives the exact row definitions the
/// tableau's slack columns refer to.
pub fn generate_intersection_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    tableau: &[iconic_simplex::TableauRow<T>],
    n_orig: usize,
    a: &DenseMatrix<T>,
    b: &[T],
    lb: &[T],
    var_types: &[VarType],
    cut_rows: &[std::sync::Arc<(Vec<T>, T)>],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-7).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let m_orig = b.len();
    let n_cuts = cut_rows.len();
    let m_aug = m_orig + n_cuts;
    let huge = T::from_f64(crate::INF_BOUND).expect("scalar literal");
    for tr in tableau {
        let bc = tr.basic_col;
        if bc >= n_orig || var_types[bc] != VarType::Binary {
            continue;
        }
        let xb = tr.basic_val;
        let f0 = xb - xb.floor();
        let f0_low = T::from_f64(1e-4).expect("scalar literal");
        if f0 < f0_low || one - f0 < f0_low {
            continue;
        }
        let mut ok = true;
        for &(_, _, at_upper, u_j) in &tr.coeffs {
            if at_upper && u_j > huge {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // Tableau row identity at the LP point:
        // x*[bc] + Σ c_k·(shifted deviation at x*) ≈ xb.
        // Slack nonbasic columns are not part of the original-space point
        // `x`; their deviation at x* is the row residual b_r − A_r·x* (the
        // shifted lower bound is 0). At-upper slacks are impossible here —
        // their upper bounds are the 1e20 artificials gated out above.
        let mut lhs = xb;
        let mut scale = xb.abs() + one;
        for &(j, c, at_upper, u_j) in &tr.coeffs {
            let dev = if j >= n_orig {
                let r = j - n_orig;
                let (b_r, cut_row) = if r < m_orig {
                    (b[r], None)
                } else if r - m_orig >= cut_rows.len() {
                    // The tableau's slack columns can exceed the node's
                    // current cut rows when the tableau is stale relative to
                    // the node's cut state (row count mismatch). The row
                    // identity is then unknowable -- drop the whole row
                    // rather than emit a cut from a half-known tableau row.
                    if std::env::var_os("ICONIC_CUT_DEBUG").is_some() {
                        eprintln!(
                            "[cutdbg] stale slack col j={j} r={r} m_orig={m_orig} n_cuts={} n_orig={n_orig}",
                            cut_rows.len(),
                        );
                    }
                    ok = false;
                    break;
                } else {
                    (cut_rows[r - m_orig].1, Some(&cut_rows[r - m_orig].0))
                };
                let mut ax = b_r;
                match cut_row {
                    None => {
                        for k in 0..n_orig {
                            let a_rk = a.get(r, k);
                            if a_rk != zero {
                                ax -= a_rk * x[k];
                            }
                        }
                    }
                    Some(cr) => {
                        for (k, &v) in cr.iter().enumerate() {
                            if v != zero {
                                ax -= v * x[k];
                            }
                        }
                    }
                }
                ax
            } else if at_upper {
                u_j - x[j]
            } else {
                x[j] - lb[j]
            };
            lhs -= c * dev;
            scale += c.abs() * dev.abs();
        }
        if !ok {
            continue; // stale slack column: the row identity is unknowable
        }
        if (lhs - x[bc]).abs() > T::from_f64(1e-6).expect("scalar literal") * scale {
            continue;
        }
        // Intersection-cut coefficients on the ≥ 0 deviations, then convert
        // to original space (≥ form over the LP variables, bound shifts into
        // the RHS; at-upper variables are complemented).
        let mut beta_lp = vec![zero; n_orig + m_aug];
        let mut ge_rhs = one;
        let mut has = false;
        for &(j, c, at_upper, u_j) in &tr.coeffs {
            let c_eff = if at_upper { -c } else { c };
            let gamma = if c_eff > zero {
                c_eff / f0
            } else {
                (-c_eff) / (one - f0)
            };
            if gamma.abs() <= T::from_f64(1e-10).expect("scalar literal") {
                continue;
            }
            has = true;
            if at_upper {
                beta_lp[j] = -gamma;
                ge_rhs -= gamma * u_j;
            } else {
                beta_lp[j] = gamma;
                if j < n_orig {
                    ge_rhs += gamma * lb[j];
                }
            }
        }
        if !has {
            continue;
        }
        // Back-substitute the slack variables s_r = b_r − A_r·x of every LP
        // row (original rows and materialized cut rows) and convert to the ≤
        // form over the original variables, exactly like the Gomory
        // separator's row conversion.
        let mut row = vec![zero; n_orig];
        let mut rhs = -ge_rhs;
        for r in 0..m_aug {
            let bs = beta_lp[n_orig + r];
            if bs.abs() <= eps {
                continue;
            }
            let (b_r, cut_row) = if r < m_orig {
                (b[r], None)
            } else {
                (cut_rows[r - m_orig].1, Some(&cut_rows[r - m_orig].0))
            };
            rhs += bs * b_r;
            match cut_row {
                None => {
                    for j in 0..n_orig {
                        let a_rj = a.get(r, j);
                        if a_rj != zero {
                            row[j] += bs * a_rj;
                        }
                    }
                }
                Some(cr) => {
                    for (j, &v) in cr.iter().enumerate() {
                        if v != zero {
                            row[j] += bs * v;
                        }
                    }
                }
            }
        }
        for j in 0..n_orig {
            row[j] = row[j] - beta_lp[j];
        }
        let viol = violation_of(&row, x, rhs);
        if viol > eps {
            pool.add(Cut::new(row, rhs, viol));
        }
    }
}

/// Lift-and-project disjunctive cuts: the cut-generating LP over the
/// `x_j ∈ {0,1}` disjunction, solved per fractional binary variable.
///
/// Any inequality `αᵀx ≤ β` valid on both sides `P⁰ = P ∩ {x_j = 0}` and
/// `P¹ = P ∩ {x_j = 1}` of the current LP relaxation `P` is valid for the
/// integer hull (every integer point lies on one of the two sides). Validity
/// on side `s` is certified by row multipliers `uˢ` (≥ 0 on ≤ rows, free on
/// equality rows) with `α = (uˢ)ᵀA + vˢ·e_j` and `β ≥ (uˢ)ᵀb + s·vˢ` — the
/// Farkas dual of "β ≥ max{αᵀx : x ∈ Pˢ}". The separation LP maximizes the
/// violation `αᵀx* − β` at the fractional point over that family, with the
/// standard `−1 ≤ α ≤ 1` normalization; its optimum is positive exactly when
/// a violated cut exists:
///
/// ```text
///     max αᵀx* − β
///     s.t. α = (u⁰)ᵀA + v⁰·e_j,  β ≥ (u⁰)ᵀb          (side 0)
///          α = (u¹)ᵀA + v¹·e_j,  β ≥ (u¹)ᵀb + v¹     (side 1)
///          u ≥ 0 / free per row type, −1 ≤ α ≤ 1
/// ```
/// The LP is small — 2(m+2n) + n + 3 variables and 2n + 2 rows: the
/// per-variable bounds `l ≤ x ≤ u` are included as rows too, so the cut is
/// valid against the node's actual box, which a rows-only formulation misses
/// (the textbook bound-multiplier example). It is solved cold with the
/// in-process simplex; the current LP is `A`, `b` plus the materialized cut
/// rows (`cut_rows`), so cuts strengthen as rounds tighten the relaxation.
/// The extracted cut's coefficients are recomputed from the side-0
/// multipliers and its RHS from both validity rows, making the validity
/// constraints exact by construction; residual gates bound the difference
/// from the LP's own point and from the side-1 identity (numerical
/// cancellation is the only unsoundness risk, and it is gated out).
pub fn generate_lift_and_project_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    eq_mask: &[bool],
    lb: &[T],
    ub: &[T],
    var_types: &[VarType],
    cut_rows: &[std::sync::Arc<(Vec<T>, T)>],
    max_candidates: usize,
    pool: &mut CutPool<T>,
) {
    let n = x.len();
    let m_orig = b.len();
    let n_cuts = cut_rows.len();
    let m_aug = m_orig + n_cuts + 2 * n; // original + cut + bound rows
                                         // Cost guard: the cut LP has O(m + n) variables and O(n) rows. Beyond
                                         // this size it is no longer cheap per candidate, so the family defers to
                                         // the tableau-only intersection-cut variant.
    if 2 * m_aug + n + 3 > 1200 || n > 400 {
        return;
    }
    let eps = T::from_f64(1e-4).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let half = T::from_f64(0.5).expect("scalar literal");
    // Fractional binary candidates, most-fractional first (deterministic
    // order; ties keep their original index order).
    let mut cands: Vec<(usize, T)> = Vec::new();
    for (j, &v) in x.iter().enumerate() {
        if j >= var_types.len() || var_types[j] != VarType::Binary {
            continue;
        }
        if v > eps && one - v > eps {
            cands.push((j, (v - half).abs()));
        }
    }
    if cands.is_empty() {
        return;
    }
    cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(max_candidates.max(1));

    // Augmented row data: original rows, then cut rows, then the per-variable
    // bound rows (−x_k ≤ −l_k and x_k ≤ u_k).
    let mut b_aug = b.to_vec();
    for cut in cut_rows.iter() {
        b_aug.push(cut.1);
    }
    for k in 0..n {
        b_aug.push(-lb[k]); // −x_k ≤ −l_k
        b_aug.push(ub[k]); // x_k ≤ u_k
    }
    let n_rows = 2 * n + 2;
    let n_vars = 2 * m_aug + n + 3;
    let n_cols = n_vars + n_rows;
    let ub_mult = T::from_f64(1e4).expect("scalar literal");
    let neg_ub = -ub_mult;

    for &(j, _) in &cands {
        // Column layout: u⁰ (m_aug), u¹ (m_aug), v⁰, v¹, α (n), β, slacks.
        let mut col_entries: Vec<Vec<(usize, T)>> = vec![Vec::new(); n_cols];
        let mut c_obj = vec![zero; n_cols];
        for k in 0..n {
            c_obj[2 * m_aug + 2 + k] = -x[k]; // maximize αᵀx* − β
        }
        c_obj[2 * m_aug + n + 2] = one;
        let v0_col = 2 * m_aug;
        let v1_col = 2 * m_aug + 1;
        // α-def rows (equalities): α_k − Σ_r uˢ_r·A_rk − vˢ·δ_jk = 0.
        for k in 0..n {
            let alpha_col = 2 * m_aug + 2 + k;
            col_entries[alpha_col].push((k, one));
            col_entries[alpha_col].push((n + k, one));
        }
        col_entries[v0_col].push((j, -one));
        col_entries[v1_col].push((n + j, -one));
        col_entries[v1_col].push((2 * n + 1, one));
        // u⁰_r / u¹_r entries from the augmented row coefficients.
        // The validity rows express β ≥ uˢᵀb (+ s·vˢ) as
        // uˢᵀb + s·vˢ − β ≤ 0 (the solver's rows are A·x + slack = b with
        // slack ≥ 0, i.e. A·x ≤ b — the earlier sign mistake let β = 0 with
        // u = 0.5·(1,1,1) pass, and the LP reported the loose cut αᵀx ≤ 1.5
        // with zero violation).
        for r in 0..m_orig {
            let (u0c, u1c) = (r, m_aug + r);
            col_entries[u0c].push((2 * n, b_aug[r]));
            col_entries[u1c].push((2 * n + 1, b_aug[r]));
            for k in 0..n {
                let a_rk = a.get(r, k);
                if a_rk != zero {
                    col_entries[u0c].push((k, -a_rk));
                    col_entries[u1c].push((n + k, -a_rk));
                }
            }
        }
        for (ci, cut) in cut_rows.iter().enumerate() {
            let r = m_orig + ci;
            let (u0c, u1c) = (r, m_aug + r);
            col_entries[u0c].push((2 * n, b_aug[r]));
            col_entries[u1c].push((2 * n + 1, b_aug[r]));
            for (k, &v) in cut.0.iter().enumerate() {
                if v != zero {
                    col_entries[u0c].push((k, -v));
                    col_entries[u1c].push((n + k, -v));
                }
            }
        }
        for k in 0..n {
            let r_lo = m_orig + n_cuts + 2 * k;
            let r_hi = r_lo + 1;
            let (u0lo, u1lo) = (r_lo, m_aug + r_lo);
            let (u0hi, u1hi) = (r_hi, m_aug + r_hi);
            // −x_k ≤ −l_k: A_r = −e_k, b = −l_k.
            col_entries[u0lo].push((k, one));
            col_entries[u0lo].push((2 * n, b_aug[r_lo]));
            col_entries[u1lo].push((n + k, one));
            col_entries[u1lo].push((2 * n + 1, b_aug[r_lo]));
            // x_k ≤ u_k: A_r = +e_k, b = u_k.
            col_entries[u0hi].push((k, -one));
            col_entries[u0hi].push((2 * n, b_aug[r_hi]));
            col_entries[u1hi].push((n + k, -one));
            col_entries[u1hi].push((2 * n + 1, b_aug[r_hi]));
        }
        // β enters the two validity rows (coefficient −1 in uˢᵀb − β ≤ 0);
        // slack columns are the identity.
        let beta_col = 2 * m_aug + n + 2;
        col_entries[beta_col].push((2 * n, -one));
        col_entries[beta_col].push((2 * n + 1, -one));
        for r in 0..n_rows {
            col_entries[n_vars + r].push((r, one));
        }
        // Variable bounds.
        let mut l_bounds = vec![zero; n_cols];
        let mut u_bounds = vec![ub_mult; n_cols];
        for r in 0..m_aug {
            let is_eq = r < m_orig && r < eq_mask.len() && eq_mask[r];
            if is_eq {
                l_bounds[r] = neg_ub;
                l_bounds[m_aug + r] = neg_ub;
            }
        }
        for k in 0..n {
            u_bounds[2 * m_aug + 2 + k] = one;
            l_bounds[2 * m_aug + 2 + k] = -one;
        }
        // The x_j-fixing multipliers v⁰/v¹ and the cut RHS β are free: a cut
        // whose validity on one side needs a negative fixing multiplier or a
        // negative β (rows with negative b) must stay expressible.
        l_bounds[2 * m_aug] = neg_ub;
        l_bounds[2 * m_aug + 1] = neg_ub;
        l_bounds[2 * m_aug + n + 2] = neg_ub;
        // Equality rows (α-def) get pinned [0,0] slacks; the two validity
        // rows' slacks are free nonnegative.
        for r in 0..2 * n {
            u_bounds[n_vars + r] = zero;
        }
        // Build the CSC (per-column rows are already ascending; sort defensively).
        let mut col_start = vec![0usize; n_cols + 1];
        for (c, entries) in col_entries.iter().enumerate() {
            col_start[c + 1] = col_start[c] + entries.len();
        }
        let mut row_idx = vec![0usize; col_start[n_cols]];
        let mut vals = vec![zero; col_start[n_cols]];
        for (c, entries) in col_entries.iter().enumerate() {
            let mut sorted = entries.clone();
            sorted.sort_by_key(|&(r, _)| r);
            for (i, &(r, v)) in sorted.iter().enumerate() {
                row_idx[col_start[c] + i] = r;
                vals[col_start[c] + i] = v;
            }
        }
        let mut solver = DualSolver::from_csc(
            c_obj,
            CscCols {
                col_start,
                row_idx,
                val: vals,
            },
            vec![zero; n_rows],
            l_bounds,
            u_bounds,
            n_rows,
            n_cols,
        );
        // A hard iteration cap: degenerate cut LPs (many pinned equality
        // slack columns) can otherwise grind to the solver's default
        // 8e6-iteration safety net — measured at ~15s per grinding
        // candidate on tsptw_n10's root, which alone blew the whole solve
        // budget. Converged candidates finish in a few hundred to ~1500
        // pivots; a non-converged candidate is skipped (its cut is lost,
        // the other candidates' are not).
        solver.set_max_iters(2_000);
        let sol = solver.cold_solve();
        if sol.status != DsStatus::Optimal {
            continue;
        }
        let xs = &sol.x;
        let u0 = &xs[0..m_aug];
        let u1 = &xs[m_aug..2 * m_aug];
        let v0 = xs[v0_col];
        let v1 = xs[v1_col];
        let alpha_sol = &xs[2 * m_aug + 2..2 * m_aug + 2 + n];
        // Recomputed cut coefficients from the side-0 multipliers (exact
        // identity) and the side-1 mirror for the coupling residual.
        let mut alpha0 = vec![zero; n];
        let mut alpha1 = vec![zero; n];
        for r in 0..m_aug {
            let (u0r, u1r) = (u0[r], u1[r]);
            if u0r == zero && u1r == zero {
                continue;
            }
            if r < m_orig {
                for k in 0..n {
                    let a_rk = a.get(r, k);
                    if a_rk != zero {
                        alpha0[k] += u0r * a_rk;
                        alpha1[k] += u1r * a_rk;
                    }
                }
            } else if r < m_orig + n_cuts {
                let cr = &cut_rows[r - m_orig].0;
                for (k, &v) in cr.iter().enumerate() {
                    if v != zero {
                        alpha0[k] += u0r * v;
                        alpha1[k] += u1r * v;
                    }
                }
            } else {
                let bk = (r - m_orig - n_cuts) / 2;
                let is_lo = (r - m_orig - n_cuts) % 2 == 0;
                // Bound row −x_bk ≤ −l_bk (A = −e_bk) or x_bk ≤ u_bk (A = +e_bk):
                // α gets +u·A_rk.
                let s = if is_lo { -one } else { one };
                alpha0[bk] += u0r * s;
                alpha1[bk] += u1r * s;
            }
        }
        alpha0[j] += v0;
        alpha1[j] += v1;
        // Residual gates: the cut's coefficients must agree with the LP's own
        // α (no cancellation garbage) and the two sides' identities (validity
        // on side 1 holds to this residual).
        let mut res = zero;
        let mut a_max = zero;
        for k in 0..n {
            let av = alpha0[k].abs();
            if av > a_max {
                a_max = av;
            }
            let d0 = (alpha0[k] - alpha1[k]).abs();
            let d1 = (alpha0[k] - alpha_sol[k]).abs();
            if d0 > res {
                res = d0;
            }
            if d1 > res {
                res = d1;
            }
        }
        if res > T::from_f64(1e-5).expect("scalar literal") * (one + a_max) {
            continue;
        }
        // Multiplier sign feasibility (exact validity needs u ≥ 0 on the ≤
        // rows, free on equalities).
        let tiny = T::from_f64(1e-7).expect("scalar literal");
        let mut ok = true;
        for r in 0..m_aug {
            let is_eq = r < m_orig && r < eq_mask.len() && eq_mask[r];
            if !is_eq && (u0[r] < -tiny || u1[r] < -tiny) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // Exact RHS from the validity rows (β ≥ max of both sides' values).
        let mut beta0 = zero;
        let mut beta1 = v1;
        for r in 0..m_aug {
            beta0 += u0[r] * b_aug[r];
            beta1 += u1[r] * b_aug[r];
        }
        let beta = if beta0 > beta1 { beta0 } else { beta1 };
        let viol = violation_of(&alpha0, x, beta);
        if viol > T::from_f64(1e-4).expect("scalar literal") {
            pool.add(Cut::new(alpha0, beta, viol));
        }
    }
}

/// Mixing cuts: for a group of rows sharing an identical coefficient
/// pattern on a common support `c·x` and differing only in one distinct
/// binary "mixing" variable `y_i` each — rows `y_i + c·x ⋛ b_i` — the
/// mixing inequality combines the group into one stronger row. The family
/// is subtle (its form depends on the row direction and the shared term's
/// range, and it degenerates for shapes where every binary is forced), so
/// no closed-form derivation is trusted: candidate cuts are enumerated
/// (coefficient of the shared term ∈ {−1, 0, +1} × a few RHS variants) and
/// each candidate is validated EXACTLY — over every binary assignment, with
/// the shared term's feasible interval per assignment computed from the
/// TRUE variable bounds and the ACTUAL row constraints (direction-agnostic),
/// checked at the interval endpoints (a linear cut holds on an interval iff
/// it holds at its endpoints). Only validated candidates that separate the
/// LP point are emitted; an invalid cut can never enter the pool. The
/// earlier window-based validation was unsound (a cut valid on a window can
/// be invalid beyond it — measured on the maxsat family, where it proved a
/// false optimum); endpoint validation over the true bounds closes that.
pub fn generate_mixing_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    lb: &[T],
    ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let m = b.len();
    let n = x.len();
    let eps = T::from_f64(1e-9).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    if m > 300 || n > 400 {
        return;
    }
    let rows: Vec<(Vec<usize>, Vec<T>, T)> = (0..m)
        .map(|r| {
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            for j in 0..n {
                let v = a.get(r, j);
                if v != zero {
                    cols.push(j);
                    vals.push(v);
                }
            }
            (cols, vals, b[r])
        })
        .collect();

    let mut grouped = vec![false; m];
    for r1 in 0..m {
        if grouped[r1] || rows[r1].0.len() < 2 {
            continue;
        }
        let mut group: Vec<usize> = vec![r1];
        for r2 in (r1 + 1)..m {
            if grouped[r2] {
                continue;
            }
            let (c1, v1, _) = &rows[r1];
            let (c2, v2, _) = &rows[r2];
            let mut i1 = 0usize;
            let mut i2 = 0usize;
            let mut diff1: Option<usize> = None;
            let mut diff2: Option<usize> = None;
            let mut ok = true;
            while i1 < c1.len() && i2 < c2.len() {
                if c1[i1] == c2[i2] {
                    if v1[i1] != v2[i2] {
                        ok = false;
                        break;
                    }
                    i1 += 1;
                    i2 += 1;
                } else if c1[i1] < c2[i2] {
                    if diff1.is_some() {
                        ok = false;
                        break;
                    }
                    diff1 = Some(c1[i1]);
                    i1 += 1;
                } else {
                    if diff2.is_some() {
                        ok = false;
                        break;
                    }
                    diff2 = Some(c2[i2]);
                    i2 += 1;
                }
            }
            if ok {
                while i1 < c1.len() {
                    if diff1.is_some() {
                        ok = false;
                        break;
                    }
                    diff1 = Some(c1[i1]);
                    i1 += 1;
                }
                while i2 < c2.len() {
                    if diff2.is_some() {
                        ok = false;
                        break;
                    }
                    diff2 = Some(c2[i2]);
                    i2 += 1;
                }
            }
            if !ok {
                continue;
            }
            let (d1, d2) = match (diff1, diff2) {
                (Some(a_), Some(b_)) if a_ != b_ => (a_, b_),
                _ => continue,
            };
            if var_types.get(d1).map(|t| t.is_integer()).unwrap_or(false)
                && var_types.get(d2).map(|t| t.is_integer()).unwrap_or(false)
            {
                group.push(r2);
            }
        }
        if group.len() < 2 {
            continue;
        }
        let shared: Vec<usize> = rows[r1]
            .0
            .iter()
            .copied()
            .filter(|&j| group.iter().all(|&r| rows[r].0.contains(&j)))
            .collect();
        if shared.is_empty() {
            continue;
        }
        // Shared term's true box range from the variable bounds.
        let mut t_val = zero;
        let mut t_lo_box = zero;
        let mut t_hi_box = zero;
        for &j in &shared {
            let coef = a.get(r1, j);
            t_val += coef * x[j];
            let (lo_j, hi_j) = (lb[j], ub[j]);
            if coef > zero {
                t_lo_box += coef * lo_j;
                t_hi_box += coef * hi_j;
            } else {
                t_lo_box += coef * hi_j;
                t_hi_box += coef * lo_j;
            }
        }
        // Mixing variables: each row's column outside the shared support,
        // with its ACTUAL coefficient in the row (the mixing derivation
        // assumes coefficient 1; other coefficients change the per-row
        // bound the validation enforces).
        let grp: Vec<(usize, T, T)> = group
            .iter()
            .map(|&r| {
                let yj = rows[r]
                    .0
                    .iter()
                    .position(|&j| !shared.contains(&j))
                    .map(|p| rows[r].0[p])
                    .unwrap();
                let ycoef = rows[r].1[rows[r].0.iter().position(|&j| j == yj).unwrap()];
                (yj, b[r], ycoef)
            })
            .collect();
        let k = grp.len();
        // The validation enumerates 2^k binary assignments per candidate; a
        // star-shaped group (vcover rows sharing a high-degree vertex) can
        // reach k = 20+ and explode the per-node cost (measured: the
        // separator was 95% of vcover_n40_p3's 26.9s solve). Cap the group
        // size so the validation stays bounded — larger groups' rows are
        // individually weaker anyway.
        if k > 14 {
            continue;
        }
        // Candidate enumeration: shared-term coefficient gamma and RHS.
        // Precompute the group's row vector once: Σ y_i + gamma·t has
        // coefficient 1 at each mixing binary and gamma·a[r1,j] at each
        // shared column. Its LP activity is an O(n) pass — checking it
        // BEFORE the exponential validation skips the 2^k enumeration for
        // every candidate that isn't violated here (the common case; the
        // validation is the separator's cost center). A violated-at-x
        // candidate is still exhaustively validated before emission, so
        // soundness is untouched: this is a filter, not a weakening.
        let mut y_part = vec![zero; n];
        for &(yj, _, _) in grp.iter() {
            y_part[yj] += one;
        }
        let mut emitted = false;
        for gamma in [T::from_f64(-1.0).expect("scalar literal"), zero, one] {
            if emitted {
                break;
            }
            for &j in &shared {
                y_part[j] = gamma * a.get(r1, j);
            }
            let mut lhs0 = zero;
            for j in 0..n {
                lhs0 += y_part[j] * x[j];
            }
            for beta_delta in [T::from_f64(-1.0).expect("scalar literal"), zero, one] {
                if emitted {
                    break;
                }
                // Cut: Σ y_i + gamma·t ≤ k − 1 + b_max + beta_delta.
                let mut b_max = grp[0].1;
                for &(_, bi, _) in grp.iter().skip(1) {
                    b_max = b_max.max(bi);
                }
                let rhs = T::from_usize(k - 1).unwrap() + b_max + beta_delta;
                // Violation gate: t's box range is [t_lo_box, t_hi_box], so
                // the largest achievable lhs is lhs0 + gamma·(gamma>0 ? t_hi_box
                // : t_lo_box); skip the exponential validation unless even
                // that overflows the rhs. (Exact upper bound for both signs
                // of gamma; conservative only through float rounding.)
                let t_extreme = if gamma > zero { t_hi_box } else { t_lo_box };
                let best_lhs = lhs0 + gamma * t_extreme;
                if best_lhs <= rhs + T::from_f64(1e-5).expect("scalar literal") {
                    continue;
                }
                // Exact validation over every binary assignment, at the
                // endpoints of the assignment's feasible t-interval, using
                // the ACTUAL row constraints (direction-agnostic).
                let mut valid = true;
                // The canonical form is all ≤ rows: y_i + t ≤ b_i (the group
                // rows share the term c·x = t and differ only in the binary
                // y_i). For a fixed assignment, the feasible t-interval is
                // [t_lo_box, min(t_hi_box, min_i(b_i − y_i))] — the rows
                // bound t from above (≤), the box bounds both sides. A
                // linear cut holds on the interval iff it holds at its
                // endpoints, so checking both endpoints (when the interval
                // is nonempty) is exact for this family.
                let mut checked_any = false;
                'validate: for mask in 0u32..(1u32 << k) {
                    let mut lhs_const = zero;
                    let mut t_hi = t_hi_box;
                    for (gi, &(_, bi, ycoef)) in grp.iter().enumerate() {
                        let yi = if (mask >> gi) & 1 == 1 { one } else { zero };
                        lhs_const += yi;
                        // ≤ row: ycoef·y_i + t ≤ b_i → t ≤ b_i − ycoef·y_i.
                        // MIR-strengthened-row layer (the U-cut inside MIR):
                        // for a unit y-coefficient binary row with fractional
                        // b_i, the mixed-integer rounding of the RHS gives
                        // t ≤ ⌊b_i⌋ + f_i·(1 − y_i) — valid (y=0 → t ≤ b_i;
                        // y=1 → t ≤ ⌊b_i⌋, implied by the row's t ≤ b_i − 1)
                        // and the f_i-scaling is what gives the mixing family
                        // its cutting power on fractional-RHS shapes.
                        let bound = if ycoef == one {
                            let floored = bi.floor();
                            let f = bi - floored;
                            if f > eps && f < one - eps {
                                floored + f * (one - yi)
                            } else {
                                bi - ycoef * yi
                            }
                        } else {
                            bi - ycoef * yi
                        };
                        if bound < t_hi {
                            t_hi = bound;
                        }
                    }
                    if t_hi < t_lo_box - eps {
                        continue; // assignment infeasible on the whole box
                    }
                    checked_any = true;
                    let t_lo = t_lo_box;
                    for &t in &[t_lo, t_hi] {
                        let lhs = lhs_const + gamma * t;
                        if lhs > rhs + eps {
                            valid = false;
                            break 'validate;
                        }
                    }
                }
                // A cut whose every assignment is infeasible on the box has
                // been checked on NOTHING — emitting it would be pure
                // speculation (measured on the vcover presolved shape: a
                // group whose rows force every y_i below the box range
                // emitted an unvalidated cut and the search proved a wrong
                // optimum). Unvalidated cuts never emit.
                if !valid || !checked_any {
                    continue;
                }
                // Violation at the LP point.
                let mut row = vec![zero; n];
                for &(yj, _, _) in grp.iter() {
                    row[yj] += one;
                }
                for &j in &shared {
                    row[j] += gamma * a.get(r1, j);
                }
                let mut lhs = zero;
                for j in 0..n {
                    lhs += row[j] * x[j];
                }
                let violation = lhs - rhs;
                if violation > T::from_f64(1e-5).expect("scalar literal") {
                    if std::env::var_os("ICONIC_CUT_LOG").is_some() {
                        eprintln!("[mixing] emitting gamma={:?} rhs={:?} viol={:?} k={} b_max={:?} t=[{:?},{:?}] row={:?}",
                            gamma.to_f64().unwrap_or(0.0), rhs.to_f64().unwrap_or(0.0),
                            violation.to_f64().unwrap_or(0.0), k, b_max.to_f64().unwrap_or(0.0),
                            t_lo_box.to_f64().unwrap_or(0.0), t_hi_box.to_f64().unwrap_or(0.0),
                            row.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect::<Vec<_>>());
                    }
                    let cut = Cut {
                        row,
                        rhs,
                        active: true,
                        violation,
                        age: 0,
                    };
                    if pool.add(cut) {
                        emitted = true;
                        break;
                    }
                }
            }
        }
        for &r in &group {
            grouped[r] = true;
        }
    }
}

/// {0,½}-Chvátal-Gomory cuts. Two complementary passes: For each ≤ constraint
/// Σ a_j·x_j ≤ b with nonnegative integer variables, the inequality
/// Σ ⌊a_j/d⌋·x_j ≤ ⌊b/d⌋ is always valid (standard CG rounding with
/// divisor d). The d=2 case is the classic zero-half cut; d=4,8 can close
/// rank-1 gaps the d=2 cut misses. (Multiple divisors are typically tried in
/// CG-family cut generators.)
///
/// **Pass 2 — pairwise GF(2) aggregation:** The real power of zero-half
/// separation: combine two tight rows that each have odd coefficients,
/// eliminating variables whose odd-coefficient contributions cancel
/// modulo 2. This finds cuts that no single row can produce — the core
/// CG-family cut generators.) The core technique: Gomory→mod-2 elimination,
/// the Koster–Zymolka–Kutschka algorithm, and
/// SymmetricDifference + EliminateVarUsingRow.
/// Only applied for m ≤ 150 to stay cheap.
///
/// Both passes are purely combinatorial — no LP solve, no basis access.
pub fn generate_zerohalf_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    lb: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-9).expect("scalar literal");
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).expect("scalar literal");
    let n = x.len();
    let m = b.len();

    // Whether Chvátal-Gomory rounding may be applied to a row at all.
    //
    // From `Σ a_j x_j ≤ b`, dividing by `d` and using `⌊a_j/d⌋ ≤ a_j/d` gives
    // `Σ ⌊a_j/d⌋ x_j ≤ b/d` -- but only when every `x_j` in the support is **nonnegative**,
    // since a negative `x_j` reverses that inequality. Rounding the right-hand side down
    // then needs the left-hand side to be an integer, which needs every support variable
    // to be **integer**.
    //
    // Both passes below checked only that the row contained *at least one* integer
    // variable (`.any(...)`), which is not the precondition -- the doc comment above
    // states the real one. On a mixed-integer row the rounded left-hand side is not an
    // integer and the floored right-hand side simply cuts into the feasible region.
    // Verified by maximising each cut's own left-hand side over the integer-feasible set:
    // on milp_n20_i8_m15 (8 integer variables among 20, so 12 continuous) **12 of 12**
    // generated zero-half cuts were invalid.
    let cg_roundable = |i: usize| -> bool {
        (0..n).all(|j| {
            let aij = a.get(i, j);
            aij.abs() <= eps || (var_types[j].is_integer() && lb[j] >= zero)
        })
    };

    // ── Pass 1: single-row k-cuts (d = 2, 4, 8, 3, 5) ─────────────────
    // The odd primes 3 and 5 are the single-row form of the mod-k cut family
    // (mod-k cuts): CG rounding with any integer divisor d is valid
    // for nonnegative integer variables, and odd divisors catch congruences
    // the even ones cannot (e.g. a row where every weight and the RHS are
    // congruent mod 3 rounds nontrivially at d=3 but not at d=2).
    let divisors: [T; 5] = [
        two,
        T::from_f64(4.0).expect("scalar literal"),
        T::from_f64(8.0).expect("scalar literal"),
        T::from_f64(3.0).expect("scalar literal"),
        T::from_f64(5.0).expect("scalar literal"),
    ];

    for &div in &divisors {
        for i in 0..m {
            let rhs = b[i];
            let half_rhs = (rhs / div).floor();
            if half_rhs <= eps {
                continue;
            }
            if !cg_roundable(i) {
                continue;
            }
            // Skip if RHS/div is already integer (no rounding effect).
            if (rhs / div - (rhs / div).floor()).abs() < eps {
                continue;
            }

            let mut row = vec![zero; n];
            let mut changed = false;
            for j in 0..n {
                let aij = a.get(i, j);
                if aij.abs() <= eps {
                    continue;
                }
                let coef = (aij / div).floor();
                if (coef - aij / div).abs() > eps {
                    changed = true;
                }
                row[j] = coef;
            }
            if !changed {
                continue;
            }

            let viol = violation_of(&row, x, half_rhs);
            if viol > eps {
                pool.add(Cut::new(row, half_rhs, viol));
            }
        }
    }

    // ── Pass 2: pairwise GF(2) row aggregation ────────────────────────
    // Only try when the constraint set is small enough that O(m²·n) is
    // cheap. Both rows must be "active" (slack < 1) — a row that is far
    // from tight contributes nothing to a violated combination.
    let max_pair_m = 150;
    if m > max_pair_m {
        return;
    }

    // Pre-compute odd-coefficient sets + slack + odd-RHS for each row.
    let mut odd_cols: Vec<Vec<usize>> = vec![vec![]; m];
    let mut odd_rhs = vec![false; m];
    let mut row_slack = vec![T::zero(); m];

    for i in 0..m {
        let rhs = b[i];
        let rhs_int = (rhs / two).floor();
        odd_rhs[i] = rhs_int
            .to_f64()
            .is_some_and(|f| (f as i64).rem_euclid(2) != 0);

        // Slack: how far the constraint is from binding at the current x.
        let mut activity = zero;
        for j in 0..n {
            activity += a.get(i, j) * x[j];
        }
        row_slack[i] = rhs - activity;
        if row_slack[i] < eps {
            row_slack[i] = zero;
        }
        // Only build odd-coefficient sets for tight rows.
        if row_slack[i] >= one {
            continue;
        }

        if !cg_roundable(i) {
            continue;
        }

        for j in 0..n {
            let aij = a.get(i, j);
            if aij.abs() <= eps {
                continue;
            }
            let aij_int = aij.floor();
            if aij_int
                .to_f64()
                .is_some_and(|f| (f as i64).rem_euclid(2) != 0)
            {
                odd_cols[i].push(j);
            }
        }
    }

    // Try every pair of tight rows. Both must be CG-roundable: the aggregated row's
    // support is the union of the two, so a single non-roundable row makes the rounded
    // aggregate invalid. Gating only the odd-column precomputation above was not enough --
    // this loop filters on slack alone, so a row that failed the precondition still
    // reached it, which is why the first attempt at this fix left 8 of 8 cuts invalid.
    for i in 0..m {
        if row_slack[i] >= one {
            continue;
        }
        if !cg_roundable(i) {
            continue;
        }

        for k in (i + 1)..m {
            if row_slack[k] >= one {
                continue;
            }
            if !cg_roundable(k) {
                continue;
            }

            // Combined slack: must be < 1 for a violated cut.
            let combined_slack = row_slack[i] + row_slack[k];
            if combined_slack >= one {
                continue;
            }

            // Compute the aggregated constraint: Σ (aᵢⱼ + aₖⱼ) xⱼ ≤ bᵢ + bₖ,
            // then apply CG rounding with divisor 2.
            let combined_rhs_floor = ((b[i] + b[k]) / two).floor();
            if combined_rhs_floor <= eps {
                continue;
            }

            let mut row = vec![zero; n];
            let mut has_any = false;
            for j in 0..n {
                let coef = ((a.get(i, j) + a.get(k, j)) / two).floor();
                if coef.abs() > eps {
                    row[j] = coef;
                    has_any = true;
                }
            }
            if !has_any {
                continue;
            }

            let viol = violation_of(&row, x, combined_rhs_floor);
            if viol > eps {
                pool.add(Cut::new(row, combined_rhs_floor, viol));
            }
        }
    }
}

/// Greedy clique extension over the conflict graph of `pairs`: each pair
/// seeds a clique, extended by any binary variable conflicting with every
/// current member; deduped, sorted, size > 2 only. Bounded by 200k
/// outer-loop operations and n <= 2000 so the O(|E|·n) extension stays cheap.
fn extend_cliques(
    pairs: &[(usize, usize)],
    var_types: &[VarType],
    n: usize,
) -> Vec<Vec<usize>> {
    let mut adj = vec![vec![false; n]; n];
    for &(j, k) in pairs {
        adj[j][k] = true;
        adj[k][j] = true;
    }
    let mut cliques: Vec<Vec<usize>> = Vec::new();
    let mut work = 0usize;
    let work_limit = 200_000usize;
    'seeds: for &(j, k) in pairs {
        if work > work_limit {
            break;
        }
        let mut clique = vec![j, k];
        for l in 0..n {
            if l == j || l == k || var_types[l] != VarType::Binary {
                continue;
            }
            work += clique.len();
            if work > work_limit {
                break 'seeds;
            }
            if clique.iter().all(|&c| adj[c][l]) {
                clique.push(l);
            }
        }
        if clique.len() > 2 {
            clique.sort();
            cliques.push(clique);
        }
    }
    cliques.sort();
    cliques.dedup();
    cliques
}

pub fn generate_clique_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
    probing_conflicts: &[(usize, usize)],
) {
    let eps = T::from_f64(1e-8).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let n = var_types.len();
    let m = b.len();
    let mut cf: Vec<(usize, usize)> = Vec::new();
    let mut covering_pairs: FxHashSet<(usize, usize)> = FxHashSet::default();
    for i in 0..m {
        let rhs = b[i];
        // `x_j=x_k=1` only PROVES a conflict (row violated) if a_j+a_k>rhs is
        // true no matter what every OTHER variable in the row does. A negative
        // coefficient elsewhere can push the row sum back under rhs even with
        // x_j=x_k=1 (its minimum contribution, at the variable's own upper
        // bound, is negative) -- this function has no access to variable
        // bounds to compute that minimum precisely, so the only sound choice
        // without one is to require every coefficient in the row be
        // non-negative before treating any pairwise sum as a proven conflict.
        // Confirmed empirically: allowing negative coefficients elsewhere gave
        // a 72% unsound rate (2759/3842 generated cuts excluded a point that
        // genuinely satisfies the original row).
        // Set-covering rows (all coeffs -1, RHS -1): x_i + x_j >= 1 means
        // x_i and x_j can't both be 0. Complements to packing: (1-x_i)+(1-x_j)<=1.
        let is_covering = rhs <= -one + eps
            && (0..n).all(|l| {
                let al = a.get(i, l);
                al >= -one - eps && al <= -one + eps || al.abs() < eps
            });
        if is_covering {
            for j in 0..n {
                if var_types[j] != VarType::Binary {
                    continue;
                }
                if a.get(i, j) > -eps {
                    continue;
                }
                for k in (j + 1)..n {
                    if var_types[k] != VarType::Binary {
                        continue;
                    }
                    if a.get(i, k) > -eps {
                        continue;
                    }
                    covering_pairs.insert((j.min(k), j.max(k)));
                }
            }
            continue;
        }
        if (0..n).any(|l| a.get(i, l) < -eps) {
            continue;
        }
        // Clique detection is gated behind structural presolve that
        // builds the conflict graph once (not per cut round). Without that
        // pre-built graph, a single row with many binaries burns O(k²) time
        // for near-zero value (most pairs don't conflict). Cap at 200
        // binaries per row, matching the existing clique-merging n≤2000 cap.
        let bin_count: usize = (0..n)
            .filter(|&j| var_types[j] == VarType::Binary && a.get(i, j) > eps)
            .count();
        if bin_count > 200 {
            continue;
        }
        for j in 0..n {
            if var_types[j] != VarType::Binary {
                continue;
            }
            let aj = a.get(i, j);
            if aj <= eps {
                continue;
            }
            for k in (j + 1)..n {
                if var_types[k] != VarType::Binary {
                    continue;
                }
                if a.get(i, k) <= eps {
                    continue;
                }
                if aj + a.get(i, k) > rhs + eps {
                    cf.push((j, k));
                }
            }
        }
    }
    // Probing-derived conflict edges (root probing presolve implications:
    // `x_j = 1 -> x_k = 0` and the contrapositive of `x_j = 0 -> x_k = 1`).
    // Same soundness as the row-derived edges — both endpoints are proven to
    // never be simultaneously 1 in any feasible point of the problem the
    // probing ran on — so they join the same conflict graph and the same
    // clique merging. The edges were recorded on an earlier problem state
    // (the root presolved problem), so re-validate the endpoints
    // defensively: in-bounds and binary.
    for &(j, k) in probing_conflicts {
        if j >= n || k >= n {
            continue;
        }
        if var_types[j] != VarType::Binary || var_types[k] != VarType::Binary {
            continue;
        }
        cf.push((j, k));
    }
    cf.sort();
    cf.dedup();

    // Clique merging: a pairwise conflict only captures one edge of the
    // conflict graph (the graph with a node per binary variable and an edge
    // between any two that provably can't both be 1). Multiple rows each
    // contributing a *different* edge of the same underlying clique --
    // Three pairwise conflicts `x1+x2<=1`, `x1+x3<=1`, `x2+x3<=1`
    // merge into the single, strictly stronger `x1+x2+x3<=1` -- used to only ever get emitted as three separate,
    // weaker 2-variable cuts. Build the conflict graph from every detected
    // edge, then greedily extend each seed pair by any other variable that
    // conflicts with *every* current member (soundness: a clique inequality
    // `sum_{j in C} x_j <= 1` is valid exactly when every pair within C
    // conflicts, which is what "every pair we've already proven sound" gives
    // for free -- no new soundness argument needed beyond the pairwise one
    // above). Bounded by an explicit work limit (200k outer-loop operations)
    // and an outright size cap (n ≤ 2000) so the O(|E|·n) greedy extension
    // stays cheap on large set-packing models.
    let mut covered_pairs: FxHashSet<(usize, usize)> = FxHashSet::default();
    if !cf.is_empty() && n <= 2000 {
        let cliques = extend_cliques(&cf, var_types, n);

        for clique in &cliques {
            for ai in 0..clique.len() {
                for bi in (ai + 1)..clique.len() {
                    covered_pairs.insert((clique[ai], clique[bi]));
                }
            }
            let viol = clique.iter().fold(-one, |acc, &j| acc + x[j]);
            if viol <= eps {
                continue;
            }
            let mut row = vec![zero; n];
            for &j in clique {
                row[j] = one;
            }
            pool.add(Cut::new(row, one, viol));
        }
    }

    // Covering clique cuts: for set-covering constraints (x_i + x_j >= 1),
    // a clique C implies sum_{j in C} x_j >= |C|-1 (at most one can be 0).
    // Rewritten as -sum_{j in C} x_j <= -(|C|-1) for the <= cut format.
    if !covering_pairs.is_empty() && n <= 2000 {
        let cf_cov: Vec<(usize, usize)> = covering_pairs.into_iter().collect();
        let cliques = extend_cliques(&cf_cov, var_types, n);
        for clique in &cliques {
            let k = clique.len();
            let rhs_neg = -T::from_usize(k - 1).unwrap();
            let viol = rhs_neg + clique.iter().fold(zero, |acc, &j| acc + x[j]);
            if viol <= eps {
                continue;
            }
            let mut row = vec![zero; n];
            for &j in clique {
                row[j] = -one;
            }
            pool.add(Cut::new(row, rhs_neg, viol));
        }
    }

    // A conflict pair is only worth spending a cut-pool slot on when it's
    // actually violated at the CURRENT LP point (`x[j]+x[k] > 1`), matching
    // every other generator here (cover/MIR/subtour all compute a real,
    // x-dependent violation). Previously this used a hardcoded
    // `violation: one`, independent of `x` (the parameter was `_x`, unused)
    // -- every structurally-valid conflict pair was unconditionally added
    // regardless of whether it cut off anything at the current point. On a
    // TSP degree-constraint row (`sum_j x_ij = 1`, all coefficients 1), every
    // one of the ~n_cities-1 pairs is a "conflict" by this row-structure
    // test, but since the row already forces at most one of them to be 1,
    // NONE of these pairs is ever actually violated by any feasible point of
    // that row (fractional or integer) -- confirmed directly: on a 12-city
    // TSP instance this generated 1320 permanently-non-violated cuts that
    // filled the cut pool's cap and crowded out the 5 genuinely-violated
    // subtour-elimination cuts also generated that round.
    for &(j, k) in &cf {
        // Dominated by an already-emitted larger clique above: any point
        // satisfying that stronger inequality automatically satisfies this
        // pair, so re-adding it is at best redundant and at worst crowds out
        // a cut-pool slot a genuinely distinct cut could have used.
        if covered_pairs.contains(&(j, k)) {
            continue;
        }
        let viol = x[j] + x[k] - one;
        if viol <= eps {
            continue;
        }
        let mut row = vec![zero; n];
        row[j] = one;
        row[k] = one;
        pool.add(Cut::new(row, one, viol));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arc-variable index for the 4-city instances below (n_cities=4 fixed).
    fn x_idx_4c(i: usize, j: usize) -> usize {
        let n_cities = 4usize;
        if j < i {
            i * (n_cities - 1) + j
        } else {
            i * (n_cities - 1) + j - 1
        }
    }

    /// Builds a 4-city TSP-MTZ instance (same layout as
    /// iconic-bench::mip::gen_tsp_mtz: degree rows then MTZ rows) and an `x`
    /// with a KNOWN disconnected 2+2 subtour (0<->1 fully selected, 2<->3
    /// fully selected, no arcs between {0,1} and {2,3}) -- feasible for every
    /// degree constraint (each city's in/out-degree is exactly 1) but not a
    /// single tour. Returns (a, b, cones, var_types, x).
    #[allow(clippy::type_complexity)]
    fn build_4city_disconnected_subtour() -> (
        DenseMatrix<f64>,
        Vec<f64>,
        Vec<Cone>,
        Vec<VarType>,
        Vec<f64>,
    ) {
        let n_cities = 4usize;
        let n_arcs = n_cities * (n_cities - 1);
        let n = n_arcs + n_cities - 1;
        let u_idx = |i: usize| -> usize { n_arcs + i - 1 };
        let m_deg = 2 * n_cities;
        let m_mtz = (n_cities - 1) * (n_cities - 2);
        let m = m_deg + m_mtz;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        let mut bv = vec![0.0; m];
        for i in 0..n_cities {
            for j in 0..n_cities {
                if i != j {
                    a.set(i, x_idx_4c(i, j), 1.0);
                }
            }
            bv[i] = 1.0;
            for k in 0..n_cities {
                if k != i {
                    a.set(n_cities + i, x_idx_4c(k, i), 1.0);
                }
            }
            bv[n_cities + i] = 1.0;
        }
        let big_n = (n_cities - 1) as f64;
        let rhs_mtz = (n_cities - 2) as f64;
        let mut row = m_deg;
        for i in 1..n_cities {
            for j in 1..n_cities {
                if i != j {
                    a.set(row, x_idx_4c(i, j), big_n);
                    a.set(row, u_idx(i), 1.0);
                    a.set(row, u_idx(j), -1.0);
                    bv[row] = rhs_mtz;
                    row += 1;
                }
            }
        }
        let mut vt = vec![VarType::Continuous; n];
        for i in 0..n_arcs {
            vt[i] = VarType::Binary;
        }
        let cones = vec![Cone::Zero(m_deg), Cone::NonNegative(m_mtz)];

        let mut x = vec![0.0; n];
        x[x_idx_4c(0, 1)] = 1.0;
        x[x_idx_4c(1, 0)] = 1.0;
        x[x_idx_4c(2, 3)] = 1.0;
        x[x_idx_4c(3, 2)] = 1.0;
        (a, bv, cones, vt, x)
    }

    #[test]
    fn subtour_cut_fires_on_a_disconnected_2plus2_pattern() {
        let (a, b, cones, vt, x) = build_4city_disconnected_subtour();
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_subtour_cuts(&x, &a, &b, &cones, &vt, &mut pool, true);
        assert!(
            !pool.cuts.is_empty(),
            "must detect at least one subtour from a disconnected 2+2 pattern"
        );
    }

    /// Soundness check: every cut generated on the disconnected 2+2 pattern
    /// above must never exclude a genuine tour. With 4 cities there are
    /// (4-1)!=6 distinct directed Hamiltonian cycles (fix city 0, permute
    /// the other 3) -- exhaustively check all 6 against every generated cut.
    #[test]
    fn subtour_cuts_never_exclude_a_genuine_tour() {
        let (a, b, cones, vt, x) = build_4city_disconnected_subtour();
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_subtour_cuts(&x, &a, &b, &cones, &vt, &mut pool, true);
        assert!(!pool.cuts.is_empty());

        let n = a.ncols;
        let others = [1usize, 2, 3];
        let mut perms: Vec<[usize; 3]> = Vec::new();
        for &p0 in &others {
            for &p1 in &others {
                for &p2 in &others {
                    if p0 != p1 && p1 != p2 && p0 != p2 {
                        perms.push([p0, p1, p2]);
                    }
                }
            }
        }
        assert_eq!(perms.len(), 6);
        for perm in &perms {
            let seq = [0usize, perm[0], perm[1], perm[2]];
            let mut tour_x = vec![0.0; n];
            for k in 0..4 {
                let (from, to) = (seq[k], seq[(k + 1) % 4]);
                tour_x[x_idx_4c(from, to)] = 1.0;
            }
            for cut in &pool.cuts {
                let lhs: f64 = (0..n).map(|j| cut.row[j] * tour_x[j]).sum();
                assert!(
                    lhs <= cut.rhs + 1e-9,
                    "cut (rhs={}) excludes genuine tour {:?} (lhs={lhs})",
                    cut.rhs,
                    seq
                );
            }
        }
    }

    /// Regression: `generate_subtour_cuts` used to iterate candidate
    /// subtour components in `HashMap::values()` order, which depends on
    /// Rust's per-process-randomized hash seed -- on instances with more
    /// violated components than the cut pool's cap could hold, WHICH cuts
    /// survived (pool cap + orthogonality filter) varied run to run on the
    /// identical instance. Components are now processed in a deterministic
    /// order (by minimum city index), so two independent calls on the same
    /// input must produce identical cuts (same rows, in the same order).
    #[test]
    fn subtour_cut_generation_is_deterministic_across_calls() {
        let (a, b, cones, vt, x) = build_4city_disconnected_subtour();
        let mut pool_a: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_subtour_cuts(&x, &a, &b, &cones, &vt, &mut pool_a, true);
        let mut pool_b: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_subtour_cuts(&x, &a, &b, &cones, &vt, &mut pool_b, true);
        let rows_a: Vec<&Vec<f64>> = pool_a.cuts.iter().map(|c| &c.row).collect();
        let rows_b: Vec<&Vec<f64>> = pool_b.cuts.iter().map(|c| &c.row).collect();
        assert_eq!(
            rows_a, rows_b,
            "identical input must produce cuts in the same order"
        );
    }

    /// Builds a 4-customer 2-vehicle CVRP in the same layout as
    /// iconic-bench::mip::gen_cvrp: depot rows 0/1 ask for k=2 departures
    /// (skipped by the subtour recovery, which only takes RHS-1 degree rows),
    /// four customer in/out degree-row pairs, and MTZ load rows with capacity
    /// `cap`. Demands [15,15,5,5] with cap 22 mean the two 15-demand
    /// customers may never share a route: the two-customer set {a,b} has
    /// ceil(30/22) = 2 required routes, so the capacity cut reads
    /// x_ab + x_ba <= 0. Returns (a, b, cones, var_types).
    #[allow(clippy::type_complexity)]
    fn build_4cust_2veh_cvrp() -> (DenseMatrix<f64>, Vec<f64>, Vec<Cone>, Vec<VarType>) {
        let n_cities = 4usize;
        let nc = n_cities + 1;
        let na = nc * (nc - 1);
        let n = na + nc;
        let dem = [0.0, 15.0, 15.0, 5.0, 5.0];
        let cap = 22.0;
        let x_i = |i: usize, j: usize| -> usize {
            if j < i {
                i * (nc - 1) + j
            } else {
                i * (nc - 1) + j - 1
            }
        };
        let m_deg = 2 * nc;
        let m_mtz = n_cities * (n_cities - 1);
        let m = m_deg + m_mtz;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        let mut bv = vec![0.0; m];
        // Depot: exactly k departures and k arrivals (rows 0/1, RHS k != 1).
        for j in 1..nc {
            a.set(0, x_i(0, j), 1.0);
            a.set(1, x_i(j, 0), 1.0);
        }
        bv[0] = 2.0;
        bv[1] = 2.0;
        // Customers: in/out degree rows, RHS 1, cities 0..3 = nodes 1..4.
        for i in 0..n_cities {
            for j in 0..nc {
                if j != i + 1 {
                    a.set(2 + i * 2, x_i(i + 1, j), 1.0);
                    a.set(3 + i * 2, x_i(j, i + 1), 1.0);
                }
            }
            bv[2 + i * 2] = 1.0;
            bv[3 + i * 2] = 1.0;
        }
        // MTZ load rows (i and j both customers), capacity `cap`, RHS cap - d_j.
        let mut row = m_deg;
        for i in 0..n_cities {
            for j in 0..n_cities {
                if i != j {
                    a.set(row, x_i(i + 1, j + 1), cap);
                    a.set(row, na + i + 1, 1.0);
                    a.set(row, na + j + 1, -1.0);
                    bv[row] = cap - dem[j + 1];
                    row += 1;
                }
            }
        }
        let mut vt = vec![VarType::Continuous; n];
        for i in 0..na {
            vt[i] = VarType::Binary;
        }
        (a, bv, vec![Cone::Zero(m_deg), Cone::NonNegative(m_mtz)], vt)
    }

    /// The two-customer capacity cut: with demands [15,15,5,5] and capacity
    /// 22, customers 0 and 1 (both demand 15) cannot share a vehicle, so a
    /// fractional point setting both arcs between them to 0.6 must be cut
    /// with x_01 + x_10 <= 0.
    #[test]
    fn capacity_cut_forbids_an_overcapacity_pair() {
        let (a, b, cones, vt) = build_4cust_2veh_cvrp();
        let nc = 5usize;
        let na = nc * (nc - 1);
        let n = na + nc;
        let x_i = |i: usize, j: usize| -> usize {
            if j < i {
                i * (nc - 1) + j
            } else {
                i * (nc - 1) + j - 1
            }
        };
        let mut x = vec![0.0; n];
        x[x_i(1, 2)] = 0.6; // both 15-demand customers
        x[x_i(2, 1)] = 0.6;
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_subtour_cuts(&x, &a, &b, &cones, &vt, &mut pool, true);
        let pair = pool
            .cuts
            .iter()
            .find(|c| c.rhs < 0.5 && c.row[x_i(1, 2)] > 0.5 && c.row[x_i(2, 1)] > 0.5);
        assert!(
            pair.is_some(),
            "must cut x_01 + x_10 <= 0: cuts = {:?}",
            pool.cuts
                .iter()
                .map(|c| (c.rhs, c.row[x_i(1, 2)], c.row[x_i(2, 1)]))
                .collect::<Vec<_>>()
        );
    }

    /// Soundness: every cut generated on the 4-customer CVRP must hold on
    /// every feasible route set. Demands [15,15,5,5] with capacity 22 admit
    /// exactly the balanced splits {15,5}/{15,5} (four of them); each split's
    /// two routes can be traversed in either direction, giving 4 * 2 * 2 = 16
    /// feasible arc assignments, enumerated exhaustively below.
    #[test]
    fn capacity_cuts_never_exclude_a_feasible_route_set() {
        let (a, b, cones, vt) = build_4cust_2veh_cvrp();
        let nc = 5usize;
        let na = nc * (nc - 1);
        let n = na + nc;
        let x_i = |i: usize, j: usize| -> usize {
            if j < i {
                i * (nc - 1) + j
            } else {
                i * (nc - 1) + j - 1
            }
        };
        // Fractional points to separate from (exercises every code path).
        let mut points: Vec<Vec<f64>> = Vec::new();
        let mut frac = vec![0.0; n];
        frac[x_i(1, 2)] = 0.6;
        frac[x_i(2, 1)] = 0.6;
        points.push(frac);
        let mut cycle = vec![0.0; n];
        for (f, t) in [(1usize, 2usize), (2, 3), (3, 1)] {
            cycle[x_i(f, t)] = 1.0; // 3-cycle over {1,2,3}: dem 35 > 22 -> k=2
        }
        points.push(cycle);
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        for x in &points {
            generate_subtour_cuts(x, &a, &b, &cones, &vt, &mut pool, true);
        }
        assert!(!pool.cuts.is_empty(), "must find at least one capacity cut");

        // All feasible route sets: the 15-demand customers are nodes 1,2 and the
        // 5-demand are 3,4; capacity 22 forces each route to hold one of each,
        // and either direction of each route is feasible, giving 16 points.
        let mut feasible: Vec<[usize; 4]> = Vec::new();
        let splits: Vec<([usize; 2], [usize; 2])> = vec![
            ([1, 3], [2, 4]),
            ([1, 4], [2, 3]),
            ([2, 3], [1, 4]),
            ([2, 4], [1, 3]),
        ];
        for (r1, r2) in &splits {
            for r1r in 0..2 {
                for r2r in 0..2 {
                    let s1 = if r1r == 0 {
                        [r1[0], r1[1]]
                    } else {
                        [r1[1], r1[0]]
                    };
                    let s2 = if r2r == 0 {
                        [r2[0], r2[1]]
                    } else {
                        [r2[1], r2[0]]
                    };
                    feasible.push([s1[0], s1[1], s2[0], s2[1]]);
                }
            }
        }
        assert_eq!(feasible.len(), 16, "16 feasible route points expected");
        for seq in &feasible {
            let mut fx = vec![0.0; n];
            // Depot -> first of route 1 -> second -> depot, then route 2.
            fx[x_i(0, seq[0])] = 1.0;
            fx[x_i(seq[0], seq[1])] = 1.0;
            fx[x_i(seq[1], 0)] = 1.0;
            fx[x_i(0, seq[2])] = 1.0;
            fx[x_i(seq[2], seq[3])] = 1.0;
            fx[x_i(seq[3], 0)] = 1.0;
            for cut in &pool.cuts {
                let lhs: f64 = (0..n).map(|j| cut.row[j] * fx[j]).sum();
                assert!(
                    lhs <= cut.rhs + 1e-9,
                    "cut (rhs={}) excludes feasible routes {:?} (lhs={lhs})",
                    cut.rhs,
                    seq
                );
            }
        }
    }

    /// The root's full subset enumeration must catch a violation that neither
    /// the pair cuts nor the x > 0.5 connected components see: all six arcs of
    /// a 3-city instance at exactly 0.5. Every pair sums to 1.0 (its cut rhs,
    /// not violated), and no arc exceeds 0.5 so the union-find components are
    /// singletons — but the 3-city set holds internal arcs summing to 3.0
    /// against a rhs of |S| − 1 = 2, so the subset enumeration must cut it.
    #[test]
    fn subtour_full_enumeration_cuts_a_half_cycle_that_components_miss() {
        let (a, b, cones, vt, mut x) = build_4city_disconnected_subtour();
        for v in x.iter_mut() {
            *v = 0.5;
        }
        let mut cheap: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_subtour_cuts(&x, &a, &b, &cones, &vt, &mut cheap, false);
        let mut full: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_subtour_cuts(&x, &a, &b, &cones, &vt, &mut full, true);
        assert!(
            cheap.cuts.is_empty(),
            "pair/component separation must miss the all-0.5 point, got {:?}",
            cheap.cuts.iter().map(|c| c.rhs).collect::<Vec<_>>()
        );
        assert!(
            full.cuts.len() >= 1,
            "full enumeration must cut the all-0.5 point"
        );
    }

    #[test]
    fn cover_cut() {
        let n = 4;
        let a = DenseMatrix::from_row_major(1, n, vec![6.0, 5.0, 4.0, 3.0]);
        let vt = vec![VarType::Binary; n];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_cover_cuts(
            &vec![0.6, 0.6, 0.0, 0.0],
            &a,
            &vec![10.0],
            &vec![1.0; n],
            &vt,
            &mut p,
        );
        assert!(p.cuts.len() > 0);
    }
    #[test]
    fn mir_cut() {
        let n = 2;
        let a = DenseMatrix::from_row_major(1, n, vec![1.3, 1.8]);
        let vt = vec![VarType::Integer; n];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_mir_cuts(
            &vec![0.7, 0.8833333333333333],
            &a,
            &vec![2.5],
            &vec![0.0; n],
            &vec![1e20; n],
            &vt,
            &mut p,
        );
        assert!(p.cuts.len() > 0);
    }

    /// Regression: `generate_mir_cuts` used to apply the continuous-variable
    /// coefficient formula (`a_j/(1-f_0)`) to EVERY row regardless of whether
    /// it actually contained a continuous variable, deriving the classic
    /// single-row MIR rounding from a raw problem row rather than a simplex
    /// tableau row (where nonbasic variables sit at their bounds). That's
    /// unsound whenever a continuous variable has a positive coefficient: set
    /// every other variable to 0 and that one variable alone can realize
    /// `x_j = b/a_j` (row exactly tight, fully feasible since x_j>=0 and
    /// continuous), but `a_j/(1-f_0) * (b/a_j) = b/(1-f_0) > floor(b)`
    /// whenever `b` has ANY fractional part -- so the "cut" always excludes
    /// this feasible point. Caught via a real instance (a lot-sizing balance
    /// row `x_t - I_t = d_t`, both continuous, zero integer variables): the
    /// generated cut excluded the trivially-feasible `x_t=d_t, I_t=0`, and
    /// with several such cuts accumulated the ROOT relaxation -- which must
    /// stay feasible as long as the MIP itself is feasible -- was reported
    /// infeasible. Fixed by only generating a cut when every nonzero-
    /// coefficient variable in the row is integer-typed (the classical
    /// Gomory fractional cut, verified sound by 17M+ exhaustive
    /// feasible-point checks across random rows).
    #[test]
    fn mir_cut_skips_rows_with_a_continuous_variable() {
        // x_0 - x_1 = 5.00005 (both continuous, lb=0): a real lot-sizing-
        // shaped row. The old code generated an unsound cut from this; the
        // fixed code must generate none.
        let a = DenseMatrix::from_row_major(1, 2, vec![1.0, -1.0]);
        let vt = vec![VarType::Continuous, VarType::Continuous];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        // x = (5.00005, 0) satisfies the row at equality -- feed it as the
        // current LP point so the violation check (if a cut were generated)
        // would fire.
        generate_mir_cuts(
            &vec![5.00005, 0.0],
            &a,
            &vec![5.00005],
            &vec![0.0; 2],
            &vec![1e20; 2],
            &vt,
            &mut p,
        );
        assert_eq!(
            p.cuts.len(),
            0,
            "must not generate a cut from an all-continuous row"
        );
    }

    /// Same soundness property, but exhaustively: for 500 random rows with a
    /// mix of integer and continuous variables (so `generate_mir_cuts` only
    /// acts on the pure-integer ones), any generated cut must never exclude
    /// a point that is feasible for the source row (respecting integrality
    /// and x_j>=0).
    #[test]
    fn mir_cuts_never_exclude_a_row_feasible_point() {
        use iconic_core::rng::XorShift;
        let mut rng = XorShift::new(777);
        let mut checked_nontrivially = 0;
        for _ in 0..500 {
            let n = 2 + rng.pick(3);
            let a_vals: Vec<f64> = (0..n)
                .map(|_| {
                    let s = if rng.pick(2) == 0 { 1.0 } else { -1.0 };
                    s * rng.uniform(0.2, 4.0)
                })
                .collect();
            let a = DenseMatrix::from_row_major(1, n, a_vals.clone());
            let b = rng.uniform(1.0, 15.0);
            let vt: Vec<VarType> = (0..n)
                .map(|_| {
                    if rng.pick(2) == 0 {
                        VarType::Integer
                    } else {
                        VarType::Continuous
                    }
                })
                .collect();
            let mut p: CutPool<f64> = CutPool::new(10, &[], &[]);
            // Feed a plausible fractional LP point (row tight) as `x`.
            let x_lp: Vec<f64> = (0..n)
                .map(|j| if j == 0 { b / a_vals[0].max(0.1) } else { 0.0 })
                .collect();
            generate_mir_cuts(
                &x_lp,
                &a,
                &vec![b],
                &vec![0.0; n],
                &vec![1e20; n],
                &vt,
                &mut p,
            );
            if p.cuts.is_empty() {
                continue;
            }
            checked_nontrivially += 1;
            let cut = &p.cuts[0];
            // Brute-force integer assignments (small range) + the exact
            // remaining slack on ONE continuous variable, checking the row
            // is satisfied, then assert the cut is not violated.
            let int_idx: Vec<usize> = (0..n).filter(|&j| vt[j].is_integer()).collect();
            let cont_idx: Vec<usize> = (0..n).filter(|&j| !vt[j].is_integer()).collect();
            let ranges: Vec<i64> = (0..int_idx.len()).map(|_| 0).collect();
            let _ = ranges;
            fn recurse(
                depth: usize,
                int_idx: &[usize],
                cont_idx: &[usize],
                a: &[f64],
                b: f64,
                cur: &mut Vec<f64>,
                cut_row: &[f64],
                cut_rhs: f64,
            ) {
                if depth == int_idx.len() {
                    let used: f64 = int_idx.iter().map(|&j| a[j] * cur[j]).sum();
                    let remaining = b - used;
                    // Try putting all remaining budget on each continuous var in turn (>=0 required).
                    if cont_idx.is_empty() {
                        if used <= b + 1e-9 {
                            let lhs: f64 = (0..cur.len()).map(|j| cut_row[j] * cur[j]).sum();
                            assert!(
                                lhs <= cut_rhs + 1e-6,
                                "cut excludes feasible point {:?} (lhs={lhs}, rhs={cut_rhs})",
                                cur
                            );
                        }
                        return;
                    }
                    for &cj in cont_idx {
                        if a[cj].abs() < 1e-12 {
                            continue;
                        }
                        let val = remaining / a[cj];
                        if val < -1e-9 {
                            continue;
                        }
                        let mut trial = cur.clone();
                        trial[cj] = val.max(0.0);
                        let row_val: f64 = (0..trial.len()).map(|j| a[j] * trial[j]).sum();
                        if row_val > b + 1e-6 {
                            continue;
                        }
                        let lhs: f64 = (0..trial.len()).map(|j| cut_row[j] * trial[j]).sum();
                        assert!(
                            lhs <= cut_rhs + 1e-6,
                            "cut excludes feasible point {:?} (lhs={lhs}, rhs={cut_rhs})",
                            trial
                        );
                    }
                    return;
                }
                let j = int_idx[depth];
                for v in 0..6 {
                    cur[j] = v as f64;
                    recurse(depth + 1, int_idx, cont_idx, a, b, cur, cut_row, cut_rhs);
                }
                cur[j] = 0.0;
            }
            let mut cur = vec![0.0; n];
            recurse(
                0, &int_idx, &cont_idx, &a_vals, b, &mut cur, &cut.row, cut.rhs,
            );
        }
        assert!(
            checked_nontrivially > 0,
            "test setup produced no cuts to check at all"
        );
    }
    #[test]
    fn clique_cut() {
        let n = 2;
        let a = DenseMatrix::from_row_major(1, n, vec![2.0, 2.0]);
        let vt = vec![VarType::Binary; n];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        // x=[0.7,0.7] genuinely violates the derived x_0+x_1<=1 cut (sum 1.4>1);
        // x=[0.5,0.5] (the previous value) sums to exactly 1 and is NOT
        // violated -- it only "passed" before because violation was a
        // hardcoded constant, independent of x.
        generate_clique_cuts(&vec![0.7, 0.7], &a, &vec![3.0], &vt, &mut p, &[]);
        assert!(p.cuts.len() > 0);
    }

    /// Regression: `generate_clique_cuts` used to report a hardcoded
    /// `violation: one` for every structurally-valid conflict pair,
    /// regardless of whether it was actually violated at the current LP
    /// point (the function's first parameter was `_x`, unused). On a row
    /// like a TSP degree constraint (`sum_j x_ij = 1`, all coefficients 1),
    /// every pair of variables is a structurally-valid "conflict" (any two
    /// summing to 2 > 1), but the row itself already forces at most one of
    /// them to be 1 -- so NONE of these pairs is ever actually violated by
    /// any feasible point, fractional or integer. Unconditionally adding
    /// them anyway wasted cut-pool budget: on a 12-city TSP instance this
    /// generated 1320 permanently-non-violated cuts that filled the pool's
    /// cap and crowded out the 5 genuinely-useful subtour-elimination cuts
    /// generated in the same round.
    #[test]
    fn clique_cut_skips_pairs_not_violated_at_the_current_point() {
        let n = 3;
        // sum x_j = 1, all binary: every pair is a structural "conflict",
        // but x=[1/3,1/3,1/3] satisfies the row exactly and violates none.
        let a = DenseMatrix::from_row_major(1, n, vec![1.0, 1.0, 1.0]);
        let vt = vec![VarType::Binary; n];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_clique_cuts(
            &vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            &a,
            &vec![1.0],
            &vt,
            &mut p,
            &[],
        );
        assert!(
            p.cuts.is_empty(),
            "no pair is violated by a point that merely satisfies the row, got {:?}",
            p.cuts.iter().map(|c| &c.row).collect::<Vec<_>>()
        );
    }

    /// Regression: `generate_cover_cuts`' sequential-lifting step computed each
    /// non-cover variable's coefficient independently against the ORIGINAL
    /// cover's weight profile, rather than accounting for previously-lifted
    /// variables (proper sequential lifting). Two independently "lifted"
    /// variables could each individually look safe yet, taken TOGETHER, exceed
    /// the row's capacity while the cut only forbade them one at a time.
    /// weights=[9,8,15,12,15], b=19: cover={2,4} (15+15=30>19), rhs=1; the old
    /// lifting gave BOTH non-cover items 0 and 1 coefficient 1 (row=[1,1,1,0,1]
    /// with item 3 still 0), producing "x0+x1+x2+x4<=1" -- but x0=x1=1 alone
    /// (weight 9+8=17<=19) is feasible for the true row and violates that cut.
    #[test]
    fn cover_cut_lifting_does_not_allow_two_lifted_vars_together() {
        let n = 5;
        let a = DenseMatrix::from_row_major(1, n, vec![9.0, 8.0, 15.0, 12.0, 15.0]);
        let vt = vec![VarType::Binary; n];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_cover_cuts(
            &vec![0.9, 0.9, 0.1, 0.1, 0.1],
            &a,
            &vec![19.0],
            &vec![1.0; n],
            &vt,
            &mut p,
        );
        for cut in &p.cuts {
            // x0=x1=1, everything else 0: feasible for the true row (17<=19).
            let x = [1.0, 1.0, 0.0, 0.0, 0.0];
            let lhs: f64 = (0..n).map(|j| cut.row[j] * x[j]).sum();
            assert!(
                lhs <= cut.rhs + 1e-9,
                "cut {:?} <= {} excludes the feasible point x0=x1=1 (lhs={lhs})",
                cut.row,
                cut.rhs
            );
        }
    }

    /// Regression: the greedy cover-building loop never checked that the
    /// accumulated weight actually EXCEEDED the row's capacity -- a cover is
    /// defined by its weight exceeding capacity, so a set the loop merely ran
    /// out of items while building (total weight <= capacity) is not a cover
    /// at all. weights=[20,16,17,10,3] (only items with weight<=b=14 are
    /// eligible: indices 3,4, weights 10+3=13<=14) used to be accepted as a
    /// 2-item "cover" (13 never exceeded 14), producing "x3+x4<=1" even though
    /// x3=x4=1 (weight 13<=14) is feasible for the true row.
    #[test]
    fn cover_cut_requires_weight_to_actually_exceed_capacity() {
        let n = 5;
        let a = DenseMatrix::from_row_major(1, n, vec![20.0, 16.0, 17.0, 10.0, 3.0]);
        let vt = vec![VarType::Binary; n];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_cover_cuts(
            &vec![0.1, 0.1, 0.1, 0.9, 0.9],
            &a,
            &vec![14.0],
            &vec![1.0; n],
            &vt,
            &mut p,
        );
        for cut in &p.cuts {
            let x = [0.0, 0.0, 0.0, 1.0, 1.0]; // weight 10+3=13 <= 14, feasible
            let lhs: f64 = (0..n).map(|j| cut.row[j] * x[j]).sum();
            assert!(
                lhs <= cut.rhs + 1e-9,
                "cut {:?} <= {} excludes the feasible point x3=x4=1 (lhs={lhs})",
                cut.row,
                cut.rhs
            );
        }
    }

    /// Regression: `generate_cover_cuts` used the row's raw RHS `b[i]`
    /// directly as the capacity to build covers against. On a row with a
    /// negative-coefficient term (e.g. `sum w_i x_i - C*y <= 0`, the
    /// "conditional capacity" shape of a bin-packing capacity row or a
    /// facility-location-style capacity constraint), the raw RHS is 0 (or
    /// negative) even though the TRUE effective capacity is C (since y<=1
    /// always) -- with all w_i>0, the `aij<=bi` item filter excluded every
    /// candidate and cover cuts could never fire at all. Confirmed directly:
    /// generate_cover_cuts produced zero cuts on bin-packing's capacity rows
    /// on every instance tried. Fixed by computing an effective capacity
    /// `b[i] + sum_{a_j<0} (-a_j)*ub[j]` (valid since `x_j<=ub[j]` bounds
    /// each negative term's most-negative contribution, for any variable
    /// type, not just binary) and using that throughout instead of the raw
    /// b[i]. This test verifies soundness exhaustively: for a small
    /// 3-item-plus-indicator row, every one of the 2^4=16 binary
    /// combinations satisfying the TRUE row must also satisfy every
    /// generated cut.
    #[test]
    fn cover_cut_effective_capacity_handles_negative_coefficient_indicator() {
        let n = 4; // x0,x1,x2 (weighted items) + y (indicator, coefficient -capacity)
        let capacity = 10.0;
        let a = DenseMatrix::from_row_major(1, n, vec![6.0, 5.0, 4.0, -capacity]);
        let vt = vec![VarType::Binary; n];
        let ub = vec![1.0; n];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        // x_frac chosen to make x0,x1,x2 look attractive to the greedy cover
        // orderings; y=1 (bin "open") is the only value that can make the
        // row's true capacity reach `capacity`.
        generate_cover_cuts(&vec![0.9, 0.9, 0.9, 1.0], &a, &vec![0.0], &ub, &vt, &mut p);
        assert!(
            !p.cuts.is_empty(),
            "must find a cover once the effective capacity accounts for the indicator"
        );

        for mask in 0u32..16 {
            let x = [
                (mask & 1) as f64,
                ((mask >> 1) & 1) as f64,
                ((mask >> 2) & 1) as f64,
                ((mask >> 3) & 1) as f64,
            ];
            let row_lhs = 6.0 * x[0] + 5.0 * x[1] + 4.0 * x[2] - capacity * x[3];
            if row_lhs > 0.0 + 1e-9 {
                continue;
            } // not feasible for the true row
            for cut in &p.cuts {
                let lhs: f64 = (0..n).map(|j| cut.row[j] * x[j]).sum();
                assert!(
                    lhs <= cut.rhs + 1e-9,
                    "cut (rhs={}) excludes true-row-feasible point {:?} (lhs={lhs})",
                    cut.rhs,
                    x
                );
            }
        }
    }

    /// Same property as the hand-built case above, but exhaustively over 200
    /// random rows with a negative-coefficient "indicator" term: for each,
    /// brute-force every binary combination (n<=14, so 2^n is enumerable),
    /// and check every generated cut against every point that is feasible
    /// for the TRUE row (including the negative term).
    #[test]
    fn cover_cut_effective_capacity_never_excludes_a_feasible_point() {
        use iconic_core::rng::Lcg;
        let mut rng = Lcg::new(0xD1B54A32D192ED03);
        let mut checked_nontrivially = 0;
        for trial in 0..200 {
            let n_items = 3 + (trial % 6); // 3..=8 weighted items
            let n = n_items + 1; // + one negative-coefficient indicator
            let weights: Vec<f64> = (0..n_items).map(|_| rng.uniform(1.0, 30.0)).collect();
            let capacity = rng.uniform(0.3, 0.7) * weights.iter().sum::<f64>();
            let mut a_data = weights.clone();
            a_data.push(-capacity);
            let a = DenseMatrix::from_row_major(1, n, a_data);
            let vt = vec![VarType::Binary; n];
            let ub = vec![1.0; n];
            let x_frac: Vec<f64> = (0..n_items)
                .map(|_| rng.uniform(0.0, 1.0))
                .collect::<Vec<_>>()
                .into_iter()
                .chain(std::iter::once(1.0))
                .collect();
            let mut p: CutPool<f64> = CutPool::new(1000, &[], &[]);
            generate_cover_cuts(&x_frac, &a, &vec![0.0], &ub, &vt, &mut p);
            if p.cuts.is_empty() {
                continue;
            }
            checked_nontrivially += 1;
            for mask in 0u32..(1u32 << n) {
                let x: Vec<f64> = (0..n).map(|j| ((mask >> j) & 1) as f64).collect();
                let row_lhs: f64 = (0..n).map(|j| a.get(0, j) * x[j]).sum();
                if row_lhs > 1e-9 {
                    continue;
                } // not feasible for the true row
                for cut in &p.cuts {
                    let lhs: f64 = (0..n).map(|j| cut.row[j] * x[j]).sum();
                    assert!(lhs <= cut.rhs + 1e-9, "trial {trial}: cut (rhs={}) excludes true-row-feasible point {:?} (lhs={lhs})", cut.rhs, x);
                }
            }
        }
        assert!(
            checked_nontrivially > 0,
            "test setup produced no cuts to check at all"
        );
    }

    /// Regression: `generate_clique_cuts` inferred a conflict between x_j and
    /// x_k from `a_j+a_k>rhs` alone, ignoring every OTHER coefficient in the
    /// row. A negative coefficient elsewhere can bring the row sum back under
    /// rhs even with x_j=x_k=1 (its minimum contribution, at the variable's
    /// own upper bound, is negative), making the inferred conflict unsound.
    /// a=[12,4,9,2,-17,-1,6], b=14: x0=x2=1 (a0+a2=21>14) used to be flagged
    /// as a conflict ("x0+x2<=1"), but x0=x2=1 with x4=1 (a4=-17) gives row
    /// sum 12+9-17=4<=14 -- genuinely feasible, and excluded by that cut.
    #[test]
    fn clique_cut_ignores_negative_coefficients_elsewhere_in_row() {
        let n = 7;
        let a = DenseMatrix::from_row_major(1, n, vec![12.0, 4.0, 9.0, 2.0, -17.0, -1.0, 6.0]);
        let vt = vec![VarType::Binary; n];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_clique_cuts(&vec![0.5; n], &a, &vec![14.0], &vt, &mut p, &[]);
        assert!(
            p.cuts.is_empty(),
            "no clique cut should fire on a row with a negative coefficient without bound info, got {:?}",
            p.cuts.iter().map(|c| &c.row).collect::<Vec<_>>()
        );
    }

    /// Exhaustive soundness check for sequential lifting (`lift_cover`),
    /// matching the methodology that caught the previous lifting
    /// implementation's unsoundness (documented above
    /// `cover_cut_lifting_does_not_allow_two_lifted_vars_together`: 171/640
    /// generated cuts, 27%, excluded a feasible point). For many random
    /// small knapsack rows (n<=14, so 2^n<=16384 -- exhaustively
    /// enumerable), generates cover cuts (now including lifted
    /// coefficients) and checks every one of the 2^n binary points: any
    /// point satisfying the ORIGINAL row must also satisfy every generated
    /// cut. A single counterexample means a cut is unsound (excludes a
    /// truly feasible point), which would make the B&B tree prune a
    /// feasible -- possibly optimal -- solution.
    #[test]
    fn lifted_cover_cuts_never_exclude_a_feasible_point() {
        use iconic_core::rng::Lcg;
        let mut rng = Lcg::new(0x9E3779B97F4A7C15);
        let mut total_cuts = 0usize;
        let mut lifted_nontrivially = 0usize;
        for trial in 0..300 {
            let n = 6 + (trial % 9); // 6..=14
            let weights: Vec<f64> = (0..n).map(|_| rng.uniform(1.0, 30.0)).collect();
            let total: f64 = weights.iter().sum();
            let b = rng.uniform(0.3, 0.7) * total;
            let a = DenseMatrix::from_row_major(1, n, weights.clone());
            let vt = vec![VarType::Binary; n];
            // A fractional "LP point" to drive the greedy cover-selection
            // orderings; doesn't need to be a real LP solution for this test.
            let x_frac: Vec<f64> = (0..n).map(|_| rng.uniform(0.0, 1.0)).collect();
            let mut p: CutPool<f64> = CutPool::new(1000, &[], &[]);
            generate_cover_cuts(&x_frac, &a, &vec![b], &vec![1.0; n], &vt, &mut p);
            total_cuts += p.cuts.len();
            for cut in &p.cuts {
                let nnz = cut.row.iter().filter(|&&v| v != 0.0).count();
                // rhs = |cover|-1, so |cover| = rhs+1; more nonzeros than
                // that means at least one non-cover variable was lifted in.
                if (nnz as f64) > cut.rhs + 1.5 {
                    lifted_nontrivially += 1;
                }
            }
            for cut in &p.cuts {
                for mask in 0u32..(1u32 << n) {
                    let x: Vec<f64> = (0..n)
                        .map(|j| if (mask >> j) & 1 == 1 { 1.0 } else { 0.0 })
                        .collect();
                    let row_sum: f64 = (0..n).map(|j| weights[j] * x[j]).sum();
                    if row_sum > b + 1e-9 {
                        continue;
                    } // not feasible for the original row
                    let cut_lhs: f64 = (0..n).map(|j| cut.row[j] * x[j]).sum();
                    if cut_lhs > cut.rhs + 1e-6 {
                        panic!(
                            "trial {trial} (n={n}, b={b:.3}): cut {:?} <= {} excludes feasible point {:?} (row_sum={row_sum:.3}, cut_lhs={cut_lhs:.3})",
                            cut.row, cut.rhs, x
                        );
                    }
                }
            }
        }
        assert!(
            total_cuts > 0,
            "sanity: the trials should have generated at least some cover cuts"
        );
        assert!(
            lifted_nontrivially > 0,
            "sanity: lifting should have added at least one non-cover variable to at least one cut across 300 trials -- {total_cuts} cuts generated but none were lifted, suggesting lift_cover is a silent no-op"
        );
    }

    /// Regression test for clique merging: three pairwise set-packing
    /// constraints `x0+x1<=1`, `x0+x2<=1`, `x1+x2<=1` are really one
    /// triangle in the conflict graph, mergeable into the single, strictly
    /// stronger `x0+x1+x2<=1`. Before this fix, `generate_clique_cuts` only
    /// ever emitted the three separate pairwise cuts (one edge of the
    /// triangle each) and had no mechanism to recognize or merge the larger
    /// structure.
    ///
    /// `x = (0.4, 0.4, 0.4)` is the discriminating point: it satisfies every
    /// individual pairwise row/cut (0.4+0.4=0.8 <= 1) but violates the merged
    /// triangle cut (0.4+0.4+0.4=1.2 > 1) -- a point only the merged clique
    /// actually excludes, proving genuine strengthening rather than a
    /// cosmetic rewrite of the same information.
    #[test]
    fn clique_merging_combines_a_pairwise_triangle_into_one_stronger_cut() {
        let n = 3;
        let a =
            DenseMatrix::from_row_major(3, n, vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let b = vec![1.0, 1.0, 1.0];
        let var_types = vec![VarType::Binary; n];
        let x = vec![0.4, 0.4, 0.4];

        // Sanity: x is feasible for every original row (this is the LP
        // relaxation point the cut generator sees, not a point already
        // excluded by the rows themselves).
        for i in 0..3 {
            let row_sum: f64 = (0..n).map(|j| a.get(i, j) * x[j]).sum();
            assert!(
                row_sum <= b[i] + 1e-9,
                "row {i} should be satisfied by x, got {row_sum}"
            );
        }

        let mut pool = CutPool::new(50, &[], &[]);
        generate_clique_cuts(&x, &a, &b, &var_types, &mut pool, &[]);

        let merged = pool
            .cuts
            .iter()
            .find(|c| c.row.iter().filter(|&&v| v > 0.5).count() == 3);
        assert!(
            merged.is_some(),
            "expected a merged 3-variable clique cut (x0+x1+x2<=1); pool contains: {:?}",
            pool.cuts
                .iter()
                .map(|c| (&c.row, c.rhs))
                .collect::<Vec<_>>()
        );
        let merged = merged.unwrap();
        assert!(
            (merged.rhs - 1.0).abs() < 1e-9,
            "merged clique rhs should be 1, got {}",
            merged.rhs
        );
        assert!(
            merged.violation > 0.1,
            "merged clique should be violated by x=(0.4,0.4,0.4) (sum=1.2>1), got violation={}",
            merged.violation
        );

        // Every pairwise cut this merged clique subsumes should have been
        // skipped, not re-added as a separate, weaker, redundant entry.
        let pairwise_2var_cuts = pool
            .cuts
            .iter()
            .filter(|c| c.row.iter().filter(|&&v| v > 0.5).count() == 2)
            .count();
        assert_eq!(
            pairwise_2var_cuts, 0,
            "the three pairwise cuts covered by the merged clique should all have been \
             skipped as dominated, found {pairwise_2var_cuts} separate 2-variable cuts instead"
        );
    }

    /// Probing-derived conflict edges must reach the clique separator: the
    /// row scan alone finds nothing here (the only row has a negative
    /// coefficient, which the pairwise scan refuses), but the edge (0,1)
    /// proven by root probing produces the cut x0+x1<=1.
    #[test]
    fn clique_cuts_accept_probing_conflict_edges() {
        let n = 3;
        // Row x0 - x1 <= 1.5: mixed sign, the row scan skips it entirely.
        let a = DenseMatrix::from_row_major(1, n, vec![1.0, -1.0, 0.0]);
        let b = vec![1.5];
        let var_types = vec![VarType::Binary; n];
        let x = vec![0.7, 0.7, 0.0]; // violates x0+x1<=1 (sum 1.4), satisfies the row
        assert!(
            a.get(0, 0) * x[0] + a.get(0, 1) * x[1] <= b[0] + 1e-9,
            "x must satisfy the original row"
        );

        let mut pool = CutPool::new(50, &[], &[]);
        generate_clique_cuts(&x, &a, &b, &var_types, &mut pool, &[(0, 1)]);

        let found = pool
            .cuts
            .iter()
            .any(|c| c.rhs == 1.0 && c.row[0] == 1.0 && c.row[1] == 1.0 && c.row[2] == 0.0);
        assert!(
            found,
            "the probing edge (0,1) must produce x0+x1<=1; pool contains: {:?}",
            pool.cuts
                .iter()
                .map(|c| (&c.row, c.rhs))
                .collect::<Vec<_>>()
        );
        // Without the probing edges, no cut is derivable from this problem.
        let mut pool2 = CutPool::new(50, &[], &[]);
        generate_clique_cuts(&x, &a, &b, &var_types, &mut pool2, &[]);
        assert!(
            pool2.cuts.is_empty(),
            "no row-derived conflict exists; the probing edge is what made the cut, \
             got {:?}",
            pool2
                .cuts
                .iter()
                .map(|c| (&c.row, c.rhs))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn separate_ls_cuts_detects_3_period_uls() {
        // 3-period uncapacitated lot-sizing: prod[t] + inv[t-1] - inv[t] = d[t]
        // with linking prod[t] ≤ M[t]·y[t].  Variables: prod0,prod1,prod2,
        // inv0,inv1,inv2, y0,y1,y2 (9 cols).  Rows: 3 flow balance + 3 linking = 6.
        let n = 9usize;
        let mut a = DenseMatrix::<f64>::zeros(6, n);
        let mut b_vec = vec![0.0; 6];
        let d = [10.0, 15.0, 5.0]; // demands
                                   // Flow balance rows
        a.set(0, 0, 1.0);
        a.set(0, 3, -1.0);
        b_vec[0] = d[0]; // prod0 - inv0 = 10
        a.set(1, 1, 1.0);
        a.set(1, 3, 1.0);
        a.set(1, 4, -1.0);
        b_vec[1] = d[1]; // prod1+inv0-inv1=15
        a.set(2, 2, 1.0);
        a.set(2, 4, 1.0);
        a.set(2, 5, -1.0);
        b_vec[2] = d[2]; // prod2+inv1-inv2=5
                         // Linking rows: prod[t] ≤ 30·y[t] → prod[t] - 30·y[t] ≤ 0
        a.set(3, 0, 1.0);
        a.set(3, 6, -30.0); // prod0 ≤ 30·y0
        a.set(4, 1, 1.0);
        a.set(4, 7, -30.0); // prod1 ≤ 30·y1
        a.set(5, 2, 1.0);
        a.set(5, 8, -30.0); // prod2 ≤ 30·y2
        let var_types = vec![
            VarType::Continuous,
            VarType::Continuous,
            VarType::Continuous,
            VarType::Continuous,
            VarType::Continuous,
            VarType::Continuous,
            VarType::Binary,
            VarType::Binary,
            VarType::Binary,
        ];

        // LP solution: y=[0.33, 0.5, 0.17] (fractional setups), produces
        // fractional inventory/production.  Should violate (ℓ,S) for a=0,b=1
        // and a=0,b=2.
        let x = vec![
            10.0, 15.0, 5.0, // prod all demand in each period
            0.0, 0.0, 0.0, // zero inventory
            0.33, 0.67, 0.17, // fractional setups
        ];

        let cones = vec![Cone::Zero(3), Cone::NonNegative(3)];
        let mut pool = CutPool::new(100, &[], &[]);
        let added = separate_lot_sizing_ls_cuts(&x, &a, &b_vec, &cones, &var_types, &mut pool);
        assert!(added > 0, "should generate at least one (l,S) cut");
        // First cut should be for a=0,b=0: the (ℓ,S) lower bound
        // Σ D_{tb}·y_t ≥ d[0] (LHS = 10*0.33 = 3.3 < 10 → violated), stored
        // negated for the pool's `row·x ≤ rhs` convention, so rhs = −10.
        assert!(pool.cuts.iter().any(|c| c.rhs == -10.0));
    }

    /// Regression: `generate_subtour_cuts` used to fire on assignment problems
    /// whose degree rows are indistinguishable from a TSP's. A 4-worker/
    /// 4-job derangement assignment (self-assignment forbidden) has the same
    /// `Σ x_ij = 1` rows and the same complete-minus-self-loop arc set as a
    /// 4-city TSP, yet the 2-cycle matching x01=x10=x23=x32=1 is FEASIBLE --
    /// and the DFJ cut `Σ_{i,j∈S} x_ij ≤ |S|-1` (lhs 2 vs rhs 1 for S={0,1})
    /// excludes it. The model is nothing but the degree rows here, which
    /// cannot distinguish a tour requirement from an assignment; the
    /// generator must emit no cuts.
    #[test]
    fn subtour_cuts_skip_a_derangement_assignment() {
        let n_cities = 4usize;
        let n_arcs = n_cities * (n_cities - 1);
        let n = n_arcs;
        let m_deg = 2 * n_cities;
        let mut a = DenseMatrix::<f64>::zeros(m_deg, n);
        let mut bv = vec![0.0; m_deg];
        for i in 0..n_cities {
            for j in 0..n_cities {
                if i != j {
                    a.set(i, x_idx_4c(i, j), 1.0);
                }
            }
            bv[i] = 1.0;
            for k in 0..n_cities {
                if k != i {
                    a.set(n_cities + i, x_idx_4c(k, i), 1.0);
                }
            }
            bv[n_cities + i] = 1.0;
        }
        let vt = vec![VarType::Binary; n];
        let cones = vec![Cone::Zero(m_deg)];
        let mut x = vec![0.0; n];
        x[x_idx_4c(0, 1)] = 1.0;
        x[x_idx_4c(1, 0)] = 1.0;
        x[x_idx_4c(2, 3)] = 1.0;
        x[x_idx_4c(3, 2)] = 1.0;
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_subtour_cuts(&x, &a, &bv, &cones, &vt, &mut pool, true);
        assert!(
            pool.cuts.is_empty(),
            "derangement assignment must get no DFJ cuts"
        );
    }

    /// Regression: `generate_disjunctive_triangle_cuts` used to accept ANY
    /// 3-nonzero row of the shape (+1 cont, −1 cont, integer coeff) as a
    /// job-shop disjunction without checking the coefficient/RHS. Rows like
    /// `s_i - s_j + 2·y_ij ≤ 3` are not disjunctions: the all-y=1 point is
    /// feasible for every such row yet violates the triangle "cut"
    /// y_a+y_b+y_c ≤ 2. Only rows matching the true big-M job-shop form
    /// `s_i - s_j + M·y ≤ M - p_i` (binary coefficient positive and strictly
    /// larger than a nonnegative RHS) may produce triangle cuts.
    #[test]
    fn triangle_cuts_skip_non_disjunctive_rows() {
        // 6 rows s_i - s_j + 2*y_ij <= 3, all continuous s_i/s_j, binary y.
        let n = 9; // s0,s1,s2, y01,y10,y02,y20,y12,y21
        let mut a = DenseMatrix::<f64>::zeros(6, n);
        let b = vec![3.0; 6];
        let rows: [(usize, usize, usize); 6] = [
            (0, 1, 3),
            (1, 0, 4),
            (0, 2, 5),
            (2, 0, 6),
            (1, 2, 7),
            (2, 1, 8),
        ];
        for (r, (si, sj, y)) in rows.iter().enumerate() {
            a.set(r, *si, 1.0);
            a.set(r, *sj, -1.0);
            a.set(r, *y, 2.0);
        }
        let vt: Vec<VarType> = (0..3)
            .map(|_| VarType::Continuous)
            .chain((0..6).map(|_| VarType::Binary))
            .collect();
        let x = vec![3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_disjunctive_triangle_cuts(&x, &a, &b, &vt, &mut pool);
        assert!(
            pool.cuts.is_empty(),
            "rows s_i-s_j+2y<=3 are not disjunctive; got {} cuts",
            pool.cuts.len()
        );
    }

    /// Positive counterpart: genuine job-shop disjunctions (s_i - s_j + M·y
    /// ≤ M - p_i, M > p_i) must still fire the triangle cut. Three jobs on
    /// one machine, p = [2,3,4], M = 9; fractional y's at 0.8 make each
    /// triangle sum 2.4 > 2.
    #[test]
    fn triangle_cuts_fire_on_true_job_shop_disjunctions() {
        let p = [2.0, 3.0, 4.0];
        let m = 9.0;
        let n = 9; // s0,s1,s2, y01,y10,y02,y20,y12,y21
        let mut a = DenseMatrix::<f64>::zeros(6, n);
        let mut b = vec![0.0; 6];
        let rows: [(usize, usize, usize); 6] = [
            (0, 1, 3),
            (1, 0, 4),
            (0, 2, 5),
            (2, 0, 6),
            (1, 2, 7),
            (2, 1, 8),
        ];
        // s_i - s_j + M*y_ij <= M - p_i
        let p_of = |c: usize| p[c];
        for (r, (si, sj, y)) in rows.iter().enumerate() {
            a.set(r, *si, 1.0);
            a.set(r, *sj, -1.0);
            a.set(r, *y, m);
            b[r] = m - p_of(*si);
        }
        let vt: Vec<VarType> = (0..3)
            .map(|_| VarType::Continuous)
            .chain((0..6).map(|_| VarType::Binary))
            .collect();
        // Fractional y's at 0.8: each triangle sums to 2.4 > 2.
        let x = vec![0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8];
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_disjunctive_triangle_cuts(&x, &a, &b, &vt, &mut pool);
        assert!(
            !pool.cuts.is_empty(),
            "true job-shop disjunctions must produce triangle cuts"
        );
        for c in &pool.cuts {
            assert!(
                (c.rhs - 2.0).abs() < 1e-9,
                "triangle cut must be y_a+y_b+y_c <= 2, got rhs={}",
                c.rhs
            );
        }
    }

    /// Regression: `separate_lot_sizing_ls_cuts` stored the (ℓ,S) inequality
    /// `inv[a-1] + Σ_{t∈[a,b]} D_tb·y_t ≥ D_ab` in the WRONG DIRECTION
    /// (`Σ D·y + inv ≤ d_ab`), but the pool's convention is `row·x ≤ rhs`.
    /// The stored cut then excluded feasible points: carry-in inventory
    /// combined with a setup on [a,b] makes the lhs exceed D_ab. Here the
    /// fractional LP point triggers the cuts, and every stored cut must
    /// satisfy the feasible integer point prod=[30,0,0], inv=[20,5,0],
    /// y=[1,1,0] (for (a,b)=(1,2): lhs = 20 + 20·1 = 40 > D_ab = 20).
    #[test]
    fn lot_sizing_cuts_never_exclude_a_feasible_point() {
        let n = 9usize;
        let mut a = DenseMatrix::<f64>::zeros(6, n);
        let mut b_vec = vec![0.0; 6];
        let d = [10.0, 15.0, 5.0];
        a.set(0, 0, 1.0);
        a.set(0, 3, -1.0);
        b_vec[0] = d[0];
        a.set(1, 1, 1.0);
        a.set(1, 3, 1.0);
        a.set(1, 4, -1.0);
        b_vec[1] = d[1];
        a.set(2, 2, 1.0);
        a.set(2, 4, 1.0);
        a.set(2, 5, -1.0);
        b_vec[2] = d[2];
        a.set(3, 0, 1.0);
        a.set(3, 6, -30.0);
        a.set(4, 1, 1.0);
        a.set(4, 7, -30.0);
        a.set(5, 2, 1.0);
        a.set(5, 8, -30.0);
        let var_types = vec![
            VarType::Continuous,
            VarType::Continuous,
            VarType::Continuous,
            VarType::Continuous,
            VarType::Continuous,
            VarType::Continuous,
            VarType::Binary,
            VarType::Binary,
            VarType::Binary,
        ];
        let cones = vec![Cone::Zero(3), Cone::NonNegative(3)];
        let x = vec![10.0, 15.0, 5.0, 0.0, 0.0, 0.0, 0.33, 0.67, 0.17];
        let x_feas = vec![30.0, 0.0, 0.0, 20.0, 5.0, 0.0, 1.0, 1.0, 0.0];
        let mut pool = CutPool::new(100, &[], &[]);
        let added = separate_lot_sizing_ls_cuts(&x, &a, &b_vec, &cones, &var_types, &mut pool);
        assert!(added > 0);
        for c in &pool.cuts {
            let lhs: f64 = (0..n).map(|j| c.row[j] * x_feas[j]).sum();
            assert!(
                lhs <= c.rhs + 1e-9,
                "stored cut (rhs={}) excludes the feasible point (lhs={lhs})",
                c.rhs
            );
        }
    }

    /// Regression: `generate_mir_cuts` had no access to variable lower
    /// bounds and rounded any all-integer row, but the MIR formula is
    /// derived under x_j ≥ 0. On `x0 + 1.5·x1 ≤ 2.5` with x1 ∈ [-1,5]
    /// integer it emits the "cut" x0 + x1 ≤ 2, which excludes the
    /// row-feasible point (x0=4, x1=-1). Rows whose support contains a
    /// variable with lb < 0 must be skipped.
    #[test]
    fn mir_cuts_skip_rows_with_negative_lb_support() {
        let a = DenseMatrix::from_row_major(1, 2, vec![1.0, 1.5]);
        let vt = vec![VarType::Integer; 2];
        let lb = vec![0.0, -1.0];
        let mut p: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_mir_cuts(
            &vec![4.0, -1.0],
            &a,
            &vec![2.5],
            &lb,
            &vec![1e20; 2],
            &vt,
            &mut p,
        );
        assert_eq!(
            p.cuts.len(),
            0,
            "MIR on a row with a negative-lb integer is unsound"
        );
    }

    /// Strong-CG strictly dominates plain CG. Row `4x0+4x1+4x2+9x3 ≤ 11`
    /// (binary), multiplier π = 1/11: the plain CG cut has ALL coefficients
    /// zero (`⌊4/11⌋ = ⌊9/11⌋ = 0`) and is therefore never violated at any
    /// row-feasible point. Sequential lifting raises x3 to 1 (its f_3 = 0:
    /// no other item fits in the remaining capacity 2), then x0..x2 to 2/9
    /// each (blocked at β − f_j with f_j = 7/9 from x3's lifted coefficient)
    /// -- the strengthened cut `2/9·x0 + 2/9·x1 + 2/9·x2 + x3 ≤ 1` IS
    /// violated at the row-feasible fractional point x = (0.5, 0, 0, 1)
    /// (value 10/9 > 1). Every one of the 2^4 = 16 binary points satisfying
    /// the row must satisfy the cut (exhaustive validity).
    #[test]
    fn strong_cg_cut_strictly_dominates_plain_cg_and_is_valid() {
        let n = 4;
        let a = DenseMatrix::from_row_major(1, n, vec![4.0, 4.0, 4.0, 9.0]);
        let b = vec![11.0];
        let lb = vec![0.0; n];
        let ub = vec![1.0; n];
        let vt = vec![VarType::Binary; n];
        // Row-feasible: 4·0.5 + 9·1 = 11. The plain CG cut from π = 1/11 is
        // `0 ≤ 1` (all ⌊a_j/11⌋ = 0) -- violation −1; the lifted cut has
        // value 2/9·0.5 + 1 = 10/9, violation 1/9 > 0.
        let x = vec![0.5, 0.0, 0.0, 1.0];
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &vt);
        generate_strong_cg_cuts(&x, &[], &a, &b, &lb, &ub, &vt, &mut pool);
        assert!(!pool.cuts.is_empty(), "strong-CG must fire on this row");
        // The three other multipliers' cuts are proportional to this one and
        // must be rejected by the pool's orthogonality gate, leaving exactly
        // the t=1 lifted cut.
        assert_eq!(
            pool.cuts.len(),
            1,
            "expected exactly the t=1 lifted cut in the pool, got {:?}",
            pool.cuts
                .iter()
                .map(|c| (&c.row, c.rhs))
                .collect::<Vec<_>>()
        );
        let cut = &pool.cuts[0];
        assert_eq!(cut.rhs, 1.0);
        assert!(
            (cut.row[3] - 1.0).abs() < 1e-9,
            "x3 lifted to 1, got {}",
            cut.row[3]
        );
        for j in 0..3 {
            assert!(
                (cut.row[j] - 2.0 / 9.0).abs() < 1e-9,
                "x{j} lifted to 2/9, got {}",
                cut.row[j]
            );
        }
        assert!(
            cut.violation > 1e-4,
            "lifted cut must be violated at x, got {}",
            cut.violation
        );

        // Exhaustive validity: every feasible binary point satisfies the cut.
        for mask in 0u32..16 {
            let xi: Vec<f64> = (0..n).map(|j| ((mask >> j) & 1) as f64).collect();
            let row_sum: f64 = xi
                .iter()
                .enumerate()
                .map(|(j, &v)| [4.0, 4.0, 4.0, 9.0][j] * v)
                .sum();
            if row_sum > b[0] + 1e-9 {
                continue;
            }
            let lhs: f64 = (0..n).map(|j| cut.row[j] * xi[j]).sum();
            assert!(
                lhs <= cut.rhs + 1e-9,
                "cut (rhs={}) excludes feasible point {:?} (lhs={lhs})",
                cut.rhs,
                xi
            );
        }
    }

    /// Exhaustive soundness of the strong-CG separator: for 200 random
    /// knapsack rows (n ≤ 11 binary, so 2^n is enumerable), generate strong-CG
    /// cuts at a fractional point and check every generated cut against every
    /// binary point that satisfies the ORIGINAL row. A single counterexample
    /// means a cut excludes a genuinely feasible point. Also asserts cuts
    /// actually fired (pool non-empty across trials) so the test is not
    /// vacuous.
    #[test]
    fn strong_cg_cuts_never_exclude_a_feasible_point() {
        use iconic_core::rng::Lcg;
        let mut rng = Lcg::new(0xC0FFEE);
        let mut total_cuts = 0usize;
        for trial in 0..200 {
            let n = 5 + (trial % 7); // 5..=11
            let weights: Vec<f64> = (0..n).map(|_| rng.uniform(1.0, 30.0)).collect();
            let total: f64 = weights.iter().sum();
            let b = rng.uniform(0.3, 0.7) * total;
            let a = DenseMatrix::from_row_major(1, n, weights.clone());
            let vt = vec![VarType::Binary; n];
            let lb = vec![0.0; n];
            let ub = vec![1.0; n];
            // Fractional LP point biased toward the heavy items (the shape a
            // fractional knapsack solution takes), to give cuts a chance to
            // violate.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&x, &y| {
                weights[y]
                    .partial_cmp(&weights[x])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut x = vec![0.0; n];
            for (rank, &j) in order.iter().enumerate() {
                x[j] = if rank < 4 {
                    rng.uniform(0.5, 1.0)
                } else {
                    rng.uniform(0.0, 0.3)
                };
            }
            let mut pool: CutPool<f64> = CutPool::new(1000, &[], &vt);
            generate_strong_cg_cuts(&x, &[], &a, &vec![b], &lb, &ub, &vt, &mut pool);
            total_cuts += pool.cuts.len();
            for cut in &pool.cuts {
                for mask in 0u32..(1u32 << n) {
                    let xi: Vec<f64> = (0..n).map(|j| ((mask >> j) & 1) as f64).collect();
                    let row_sum: f64 = (0..n).map(|j| weights[j] * xi[j]).sum();
                    if row_sum > b + 1e-9 {
                        continue;
                    }
                    let cut_lhs: f64 = (0..n).map(|j| cut.row[j] * xi[j]).sum();
                    assert!(cut_lhs <= cut.rhs + 1e-6,
                        "trial {trial} (n={n}, b={b:.3}): cut {:?} <= {} excludes feasible point {:?} (row_sum={row_sum:.3}, cut_lhs={cut_lhs:.3})",
                        cut.row, cut.rhs, xi);
                }
            }
        }
        assert!(total_cuts > 0, "sanity: strong-CG should fire on some of the 200 trials (plain CG alone can never violate a row-feasible point)");
    }

    /// The pool's nonzero growth cap: a candidate cut with more than
    /// max(width/4 floor, 2 × mean pool nnz) nonzeros must be rejected.
    /// Pool at n=40 (width floor 20) with one 2-nnz cut → cap 20: a 30-nnz
    /// candidate is rejected, a 20-nnz one is accepted; the cap grows with
    /// the pool's own density. A second test pins the width scaling: the
    /// same 30-nnz candidate is accepted on a 274-column extended
    /// formulation (floor 68), the Glover-QKP shape where the fixed-20
    /// floor silently rejected every tableau Gomory cut (measured: zero
    /// root cuts on qkp_n40).
    #[test]
    fn cut_pool_nnz_growth_cap_rejects_dense_cuts() {
        let n = 40;
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &vec![VarType::Binary; n]);
        // Seed one sparse cut (2 nonzeros, rhs 1) — mean pool nnz = 2.
        let mut sparse_row = vec![0.0; n];
        sparse_row[0] = 1.0;
        sparse_row[1] = 1.0;
        assert!(pool.add(Cut::new(sparse_row, 1.0, 0.5)));
        assert_eq!(pool.cuts.len(), 1);

        // Dense candidate (30 nonzeros) — cap is max(20, 2·2) = 20 → reject.
        let mut dense_row = vec![0.0; n];
        for j in 0..30 {
            dense_row[j] = 1.0;
        }
        assert!(
            !pool.add(Cut::new(dense_row, 1.0, 0.5)),
            "30-nnz cut must be rejected when the pool mean is 2 (cap 20)"
        );

        // At-cap candidate (20 nonzeros) — accepted.
        let mut at_cap_row = vec![0.0; n];
        for j in 0..20 {
            at_cap_row[j] = 1.0;
        }
        assert!(
            pool.add(Cut::new(at_cap_row, 1.0, 0.5)),
            "20-nnz cut is exactly at the cap and must be accepted"
        );

        // Width scaling: the same 30-nnz candidate on a 274-column
        // formulation (Glover-QKP extended space) has floor 68 → accepted.
        let n_ext = 274;
        let mut pool_ext: CutPool<f64> = CutPool::new(100, &[], &vec![VarType::Binary; n_ext]);
        let mut wide_dense = vec![0.0; n_ext];
        for j in 0..30 {
            wide_dense[j] = 1.0;
        }
        assert!(
            pool_ext.add(Cut::new(wide_dense, 1.0, 0.5)),
            "30-nnz cut on a 274-wide formulation must be accepted (floor n/4 = 68)"
        );
    }

    /// The pool's orthogonality gate (hard rejection at |cos| > 0.894, the
    /// squared-cosine threshold 0.80): a candidate whose row vector is
    /// nearly parallel to ANY existing pool cut is rejected; a more
    /// orthogonal candidate is accepted. Near-parallel cuts add no new
    /// direction to the cutting space and only bloat the LP.
    #[test]
    fn cut_pool_orthogonality_gate_rejects_parallel_cuts() {
        let n = 4;
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &vec![VarType::Binary; n]);
        assert!(pool.add(Cut::new(vec![1.0, 0.0, 0.0, 0.0], 1.0, 0.5)));

        // [1, 0.1, 0, 0] vs [1, 0, 0, 0]: cos = 1/√1.01 ≈ 0.995 > 0.894 → reject.
        assert!(
            !pool.add(Cut::new(vec![1.0, 0.1, 0.0, 0.0], 1.0, 0.5)),
            "cos ≈ 0.995 must be rejected by the orthogonality gate"
        );

        // [0.5, 0.5, 0, 0] vs [1, 0, 0, 0]: cos = 0.5/√0.5 ≈ 0.707 ≤ 0.894 → accept.
        assert!(
            pool.add(Cut::new(vec![0.5, 0.5, 0.0, 0.0], 1.0, 0.5)),
            "cos ≈ 0.707 must pass the orthogonality gate"
        );
        assert_eq!(pool.cuts.len(), 2);
    }
}

/// Subtour-elimination cuts (Dantzig-Fulkerson-Johnson), separated purely from the
/// equality-row pattern -- no problem-specific knowledge of which variables represent
/// arcs. Detects a degree-1 bipartite assignment structure (as any directed-arc
/// formulation over a node set produces: `Σ_j x_ij = 1` per "out" node i, `Σ_i x_ij = 1`
/// per "in" node j, `x_ij` binary, self-loops excluded -- TSP/VRP arc formulations all
/// share this shape) among the leading `Cone::Zero` rows, recovers each arc variable's
/// endpoints via the unique self-exclusion bijection between out-rows and in-rows, then
/// separates violated `Σ_{i,j∈S, i≠j} x_ij ≤ |S|-1` constraints from the connected
/// components of the current LP solution's `x > 0.5` support. Bails out (adds nothing)
/// at the first sign the matrix doesn't cleanly match this shape, rather than guessing.
pub fn generate_subtour_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    cones: &[Cone],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
    full: bool,
) {
    let Some((n_cities, arcs, arc_owner)) = recover_tour_arcs(a, b, cones, var_types) else {
        return;
    };
    let n = x.len();
    let one = T::one();

    // Demands and the shared route capacity, when the formulation carries
    // load-accumulation rows (`c·x_ij + u_i − u_j ≤ c − d_j`, the MTZ form). Their
    // presence strengthens the DFJ cut: any feasible route set visits a customer
    // subset S in blocks each carrying at most Q load, so S holds at most
    // |S| − ceil(dem(S)/Q) internal arcs — the classic rounded capacity inequality
    // of capacitated vehicle routing. Without load rows the capacity term degrades
    // to 1 and the cut is the plain |S| − 1, so a pure TSP is unchanged.
    let cap_info = recover_route_capacity(a, b, cones, n, n_cities, &arc_owner);

    // The root (full = true) gets the exact separation: every proper subset is
    // tested, so no violated rounded capacity inequality is missed. The tree
    // (full = false) keeps the cheap candidate families — pairs and connected
    // components — whose evaluation is linear in the arc count, since a node's
    // fractional point changes little between siblings.
    let min_viol = T::from_f64(1e-4).unwrap();
    if full {
        // Subset enumeration by increasing size (deterministic; small sets
        // are the classic violations, and the early-exit cap also bounds
        // the work). The full family is 2^n_cities subsets — fine at the
        // root for the sizes routing models reach (<= 16 cities). Every
        // proper subset's cut is the plain DFJ |S| − 1 for a single tour,
        // or the rounded capacity inequality |S| − ceil(dem(S)/Q) when the
        // load rows are readable.
        let mut x_arc: Vec<Vec<T>> = vec![vec![T::zero(); n_cities]; n_cities];
        let mut arc_var: Vec<Vec<Option<usize>>> = vec![vec![None; n_cities]; n_cities];
        for &(v, from, to) in &arcs {
            x_arc[from][to] = x[v];
            arc_var[from][to] = Some(v);
        }
        let mut dem_s: Vec<T> = vec![T::zero(); 1usize << n_cities];
        let mut arc_s: Vec<T> = vec![T::zero(); 1usize << n_cities];
        // Singleton base cases for the incremental accumulation below.
        if let Some((dem, _, _)) = &cap_info {
            for (i, d) in dem.iter().enumerate().take(n_cities) {
                dem_s[1usize << i] = *d;
            }
        }
        let mut added = 0usize;
        for size in 2..n_cities {
            // Enumerate masks with `size` bits in increasing order
            // (Gosper's hack).
            let mut mask = (1usize << size) - 1;
            while mask < (1usize << n_cities) {
                // Accumulate demand and internal arcs from the subset
                // without its least-significant member.
                let lsb = mask & mask.wrapping_neg();
                let i = lsb.trailing_zeros() as usize;
                let prev = mask ^ lsb;
                if let Some((dem, _, _)) = &cap_info {
                    dem_s[mask] = dem_s[prev] + dem[i];
                }
                let mut s = arc_s[prev];
                let mut pm = prev;
                while pm != 0 {
                    let j = pm.trailing_zeros() as usize;
                    pm &= pm - 1;
                    s = s + x_arc[i][j] + x_arc[j][i];
                }
                arc_s[mask] = s;
                let k = match &cap_info {
                    Some((_, cap, _)) => min_routes_required(dem_s[mask], *cap),
                    None => 1,
                };
                let rhs = T::from_usize(size.saturating_sub(k)).unwrap();
                let violation = arc_s[mask] - rhs;
                if violation > min_viol {
                    let mut row = vec![T::zero(); n];
                    let mut mm = mask;
                    while mm != 0 {
                        let ii = mm.trailing_zeros() as usize;
                        mm &= mm - 1;
                        let mut jm = mask;
                        while jm != 0 {
                            let jj = jm.trailing_zeros() as usize;
                            jm &= jm - 1;
                            if ii != jj {
                                if let Some(v) = arc_var[ii][jj] {
                                    row[v] = one;
                                }
                            }
                        }
                    }
                    // The structural size (the subset cardinality |S|) is the
                    // meaningful support measure: the materialized row is
                    // |S|·(|S|−1) arc coefficients, which the pool-mean cap
                    // of `add` rejects for |S| ≥ 6 once the pool is full of
                    // pair/clique rows — the reason big-subset RCI cuts never
                    // reached the LP and cvrp root bounds stuck at the
                    // plain-DFJ value.
                    pool.add_structural(
                        Cut {
                            row,
                            rhs,
                            active: true,
                            violation,
                            age: 0,
                        },
                        size,
                    );
                    added += 1;
                    if added >= 80 {
                        return;
                    }
                }
                let c = mask & mask.wrapping_neg();
                let r = mask + c;
                mask = (((r ^ mask) >> 2) / c) | r;
            }
        }
        return;
    }

    // Two-customer sets: for S = {i,j} the internal arcs are exactly x_ij and x_ji,
    // so the cut reads x_ij + x_ji ≤ 2 − ceil((d_i+d_j)/Q). This is the 2-cycle
    // elimination (the LP loves to set both directions of a pair to 0.5), and it
    // degenerates to x_ij + x_ji ≤ 0 — no arc either way — when the two customers'
    // combined demand exceeds the capacity and no route may serve them together.
    if let Some((dem, cap, _)) = &cap_info {
        let mut added = 0usize;
        for i in 0..n_cities {
            for j in (i + 1)..n_cities {
                let k = min_routes_required(dem[i] + dem[j], *cap);
                let rhs = T::from_usize(2usize.saturating_sub(k)).unwrap();
                let mut row = vec![T::zero(); n];
                let mut sum = T::zero();
                for &(v, from, to) in &arcs {
                    if (from == i && to == j) || (from == j && to == i) {
                        row[v] = one;
                        sum += x[v];
                    }
                }
                let violation = sum - rhs;
                if violation > min_viol {
                    pool.add_structural(
                        Cut {
                            row,
                            rhs,
                            active: true,
                            violation,
                            age: 0,
                        },
                        2,
                    );
                    added += 1;
                    // The pool's own cap guards the round; this local cap keeps one
                    // generator from crowding out the rest of the chain on a model
                    // where every pair is violated.
                    if added >= 24 {
                        return;
                    }
                }
            }
        }
    }

    // Union-find over cities using the current LP solution's selected (x > 0.5) arcs.
    let mut parent: Vec<usize> = (0..n_cities).collect();
    fn find(parent: &mut [usize], mut u: usize) -> usize {
        while parent[u] != u {
            parent[u] = parent[parent[u]];
            u = parent[u];
        }
        u
    }
    let half = T::from_f64(0.5).unwrap();
    for &(j, from, to) in &arcs {
        if x[j] > half {
            let (ru, rv) = (find(&mut parent, from), find(&mut parent, to));
            if ru != rv {
                parent[ru] = rv;
            }
        }
    }

    // Each proper (not full, not singleton) connected component is a candidate
    // subtour: the induced arc set can hold at most |members| − k(members) arcs,
    // where k is the minimum number of routes the subset's demand requires
    // (k = 1 without capacity rows, i.e. the plain |members| − 1 DFJ bound).
    let mut comp_members: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for city in 0..n_cities {
        let r = find(&mut parent, city);
        comp_members.entry(r).or_default().push(city);
    }
    // Iterate components in a deterministic order (by their smallest city
    // index -- each Vec is already built in increasing-city order above, so
    // `members[0]` is that minimum) rather than `comp_members.values()`'s
    // HashMap order, which depends on Rust's per-process hash-seed
    // randomization. With the cut pool's cap and orthogonality filter, which
    // components get processed first determines which cuts survive when
    // there's contention -- leaving this at HashMap order made cut
    // generation (and everything downstream in the B&B search) vary
    // run-to-run on the same instance.
    let mut ordered_comps: Vec<&Vec<usize>> = comp_members.values().collect();
    ordered_comps.sort_by_key(|members| members[0]);
    let min_viol = T::from_f64(1e-4).unwrap();
    for members in ordered_comps {
        if members.len() < 2 || members.len() >= n_cities {
            continue;
        }
        let in_comp: FxHashSet<usize> = members.iter().copied().collect();
        let mut row = vec![T::zero(); n];
        let mut sum = T::zero();
        for &(j, from, to) in &arcs {
            if in_comp.contains(&from) && in_comp.contains(&to) {
                row[j] = one;
                sum += x[j];
            }
        }
        let k = match &cap_info {
            Some((dem, cap, _)) => {
                let dem_s: T = members
                    .iter()
                    .map(|&c| dem[c])
                    .fold(T::zero(), |a, d| a + d);
                min_routes_required(dem_s, *cap)
            }
            None => 1,
        };
        let rhs = T::from_usize(members.len().saturating_sub(k)).unwrap();
        let violation = sum - rhs;
        if violation > min_viol {
            // Structural size = the component's member count (same rationale
            // as the enumeration family above: the materialized row spans
            // |members|·(|members|−1) arcs).
            pool.add_structural(
                Cut {
                    row,
                    rhs,
                    active: true,
                    violation,
                    age: 0,
                },
                members.len(),
            );
        }
    }
}

/// `ceil(dem / cap)` in `T` arithmetic: the minimum number of routes that must serve a
/// set of customers whose total demand is `dem`, each route carrying at most `cap`.
fn min_routes_required<T: Scalar + PartialOrd>(dem: T, cap: T) -> usize {
    if dem <= T::zero() || cap <= T::zero() {
        return 1;
    }
    let ratio = dem / cap;
    let fl = ratio.floor();
    let k = fl.to_usize().unwrap_or(0);
    if ratio - fl <= T::from_f64(1e-9).unwrap() {
        k.max(1)
    } else {
        k + 1
    }
}

/// Cheap probe for the tour/vehicle-routing shape: `recover_tour_arcs` with the
/// recovered structure discarded. Used by `generate_root_cuts` to run the
/// subtour/RCI family FIRST on routing models — there it is the strongest cut
/// family, and running it after the clique/MIR families lets their rows fill
/// the round's pool budget before any subtour row is seen.
pub(crate) fn probe_tour_shape<T: Scalar + PartialOrd>(
    a: &DenseMatrix<T>,
    b: &[T],
    cones: &[Cone],
    var_types: &[VarType],
) -> bool {
    recover_tour_arcs(a, b, cones, var_types).is_some()
}

/// Recover a pure tour structure from equality rows: one binary arc variable per
/// ordered pair of distinct cities, linking an "out" row (`Σ_j x_ij = 1`) to an "in"
/// row (`Σ_j x_ji = 1`), with every city's out-row missing exactly its own in-row (no
/// self-loop arc exists). Returns `(n_cities, arcs, arc_owner)` with `arcs` as
/// `(variable, from_city, to_city)` and `arc_owner` as `variable -> (from, to)`.
/// Bails out (returns `None`) at the first sign the matrix doesn't cleanly match,
/// rather than guessing.
#[allow(clippy::type_complexity)]
fn recover_tour_arcs<T: Scalar + PartialOrd>(
    a: &DenseMatrix<T>,
    b: &[T],
    cones: &[Cone],
    var_types: &[VarType],
) -> Option<(
    usize,
    Vec<(usize, usize, usize)>,
    FxHashMap<usize, (usize, usize)>,
)> {
    let eps = T::from_f64(1e-6).unwrap();
    let one = T::one();
    let n = var_types.len();

    // 1. Equality (Zero-cone) rows form a prefix in canonical cone order.
    let n_eq = crate::n_eq_rows(cones);
    if n_eq < 4 {
        return None;
    }

    // 2. Degree-1 binary rows: Σ_{j: a[r][j]==1} x_j == 1, every such j binary, no
    //    other nonzero entries in the row.
    let mut degree_rows: Vec<usize> = Vec::new();
    let mut row_vars: Vec<Vec<usize>> = Vec::new();
    for r in 0..n_eq {
        if (b[r] - one).abs() > eps {
            continue;
        }
        let mut vars = Vec::new();
        let mut shape_ok = true;
        for j in 0..n {
            let aij = a.get(r, j);
            if aij.abs() <= eps {
                continue;
            }
            if (aij - one).abs() > eps || var_types[j] != VarType::Binary {
                shape_ok = false;
                break;
            }
            vars.push(j);
        }
        if shape_ok && vars.len() >= 2 {
            degree_rows.push(r);
            row_vars.push(vars);
        }
    }
    if degree_rows.len() < 4 {
        return None;
    }
    // The pure degree-equalities shape cannot distinguish a tour problem (a
    // single connected cycle, where any proper subset S holds at most |S|-1
    // arcs) from an assignment / permutation problem (any bijection is
    // feasible and S can hold |S| arcs). Concretely: a 4-worker/4-job
    // derangement assignment has exactly the same `Σ x_ij = 1` rows and the
    // same complete-minus-self-loop arc set as a 4-city TSP, yet the 2-cycle
    // matching x01=x10=x23=x32=1 is feasible and violates the S={0,1} "cut"
    // (lhs 2 vs rhs 1). The DFJ cut is only sound when something beyond the
    // degree rows forces a single tour (MTZ-style coupling rows, capacity
    // rows, ...). A model consisting of exactly the degree rows is skipped
    // rather than guessed at.
    if b.len() == degree_rows.len() {
        return None;
    }

    // 3. Row-to-row adjacency: a binary variable touching exactly two degree rows is
    //    the arc-variable linking them (any other multiplicity doesn't fit the shape
    //    and is excluded from the graph, not treated as an error).
    let mut var_rows: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (ri, vars) in row_vars.iter().enumerate() {
        for &j in vars {
            var_rows.entry(j).or_default().push(degree_rows[ri]);
        }
    }
    let row_index: FxHashMap<usize, usize> = degree_rows
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i))
        .collect();
    let k = degree_rows.len();
    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); k];
    let mut var_edge: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    for (&j, rows) in &var_rows {
        if rows.len() != 2 {
            continue;
        }
        let (ia, ib) = (row_index[&rows[0]], row_index[&rows[1]]);
        if ia == ib {
            continue;
        }
        adj[ia].insert(ib);
        adj[ib].insert(ia);
        var_edge.insert((ia.min(ib), ia.max(ib)), j);
    }

    // 4. Bipartite 2-coloring: a variable only ever links a row of one class to a row
    //    of the other (two out-rows, or two in-rows, never share an arc-variable).
    let mut color: Vec<i8> = vec![-1; k];
    for start in 0..k {
        if color[start] != -1 {
            continue;
        }
        color[start] = 0;
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            let cu = color[u];
            for &v in &adj[u] {
                if color[v] == -1 {
                    color[v] = 1 - cu;
                    stack.push(v);
                } else if color[v] == cu {
                    return None; // odd cycle: not this pattern
                }
            }
        }
    }
    let out_rows: Vec<usize> = (0..k).filter(|&i| color[i] == 0).collect();
    let in_rows: Vec<usize> = (0..k).filter(|&i| color[i] == 1).collect();
    if out_rows.is_empty() || out_rows.len() != in_rows.len() {
        return None;
    }
    let n_cities = out_rows.len();
    if n_cities < 3 {
        return None;
    }

    // 5. Self-pairing via exclusion: a clean square assignment has every out-row
    //    connected to every in-row except its own (no self-loop arc exists). Verify
    //    this holds everywhere and that the resulting "missing" map is a bijection --
    //    otherwise bail rather than guess at a pairing.
    let mut city_of_out: FxHashMap<usize, usize> = FxHashMap::default();
    let mut city_of_in: FxHashMap<usize, usize> = FxHashMap::default();
    for (city_id, &o) in out_rows.iter().enumerate() {
        let missing: Vec<usize> = in_rows
            .iter()
            .copied()
            .filter(|i| !adj[o].contains(i))
            .collect();
        if missing.len() != 1 {
            return None;
        }
        let i = missing[0];
        if city_of_in.contains_key(&i) {
            return None; // not a bijection
        }
        city_of_out.insert(o, city_id);
        city_of_in.insert(i, city_id);
    }
    if city_of_in.len() != n_cities {
        return None;
    }

    // 6. Recover each arc variable's (from_city, to_city).
    let mut arcs: Vec<(usize, usize, usize)> = Vec::new();
    let mut arc_owner: FxHashMap<usize, (usize, usize)> = FxHashMap::default();
    for (&(ia, ib), &j) in &var_edge {
        let (from, to) = if color[ia] == 0 {
            (city_of_out[&ia], city_of_in[&ib])
        } else {
            (city_of_out[&ib], city_of_in[&ia])
        };
        arcs.push((j, from, to));
        arc_owner.insert(j, (from, to));
    }
    if arcs.len() < n_cities {
        return None;
    }
    // The arc set must be the complete directed graph minus self-loops:
    // exactly one variable per ordered pair (i,j), i≠j. Partial arc sets --
    // the common case for assignment problems, where a worker can only take
    // a subset of jobs -- are not a plain tour formulation, and the
    // component-union argument that makes the DFJ cut valid does not apply;
    // skip rather than guess.
    let mut pair_seen: FxHashSet<(usize, usize)> = FxHashSet::default();
    for &(_, from, to) in &arcs {
        pair_seen.insert((from, to));
    }
    if arcs.len() != n_cities * (n_cities - 1) {
        return None;
    }
    for i in 0..n_cities {
        for j in 0..n_cities {
            if i != j && !pair_seen.contains(&(i, j)) {
                return None;
            }
        }
    }
    Some((n_cities, arcs, arc_owner))
}

/// Disjunctive triangle cuts for job-shop scheduling (Applegate-Cook 1991).
///
/// Detects big-M disjunctive rows `s_i − s_j + M·y_ij ≤ M − p_i` (Type A:
/// i before j) and their reverse `s_j − s_i + M·y_ji ≤ M − p_j` (Type B).
/// For any three jobs i,j,k on the same machine, the triangle inequality
/// `y_ij + y_jk + y_ki ≤ 2` is valid and prevents the LP from setting all
/// sequencing variables to 0.5 — the dominant cause of B&B explosion on job
/// shop instances.
pub fn generate_disjunctive_triangle_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-8).unwrap();
    let one = T::one();
    let zero = T::zero();
    let n = var_types.len();
    let m = b.len();
    let mut disj_pairs: Vec<(usize, usize, usize)> = Vec::new(); // (s_i, s_j, y)
    for i in 0..m {
        let mut nz: Vec<(usize, T)> = Vec::new();
        for j in 0..n {
            let aij = a.get(i, j);
            if aij.abs() > eps {
                nz.push((j, aij));
            }
        }
        if nz.len() == 3 {
            let mut cont_pos = None;
            let mut cont_neg = None;
            let mut bin = None;
            let mut bin_coef = zero;
            for &(j, aij) in &nz {
                if aij > eps && (aij - one).abs() < eps && !var_types[j].is_integer() {
                    cont_pos = Some(j);
                } else if aij < -eps && (aij + one).abs() < eps && !var_types[j].is_integer() {
                    cont_neg = Some(j);
                } else if var_types[j].is_integer() {
                    bin = Some(j);
                    bin_coef = aij;
                }
            }
            if let (Some(si), Some(sj), Some(y)) = (cont_pos, cont_neg, bin) {
                // The row must be a genuine big-M disjunction in the job-shop
                // form s_i - s_j + M·y ≤ M - p_i with M > p_i > 0: the binary
                // coefficient is positive and strictly larger than a
                // nonnegative RHS (M - rhs = p_i > 0). Any other 3-nonzero
                // (+1 cont, −1 cont, integer) row -- e.g. s_i - s_j + 2·y_ij
                // ≤ 3, where y=1 still leaves slack -- is not a disjunction:
                // the all-y=1 point is feasible for every such row yet
                // violates the triangle cut y_a+y_b+y_c ≤ 2. Skip it.
                if bin_coef > eps && b[i] >= zero && bin_coef > b[i] {
                    disj_pairs.push((si, sj, y));
                }
            }
        }
    }
    if disj_pairs.len() < 6 {
        return;
    }
    // Group into machines by shared continuous variables
    let mut machine_jobs: Vec<FxHashSet<usize>> = Vec::new();
    let mut machine_ys: Vec<FxHashMap<(usize, usize), usize>> = Vec::new();
    for &(si, sj, y) in &disj_pairs {
        let mut found = None;
        for (midx, jobs) in machine_jobs.iter().enumerate() {
            if jobs.contains(&si) || jobs.contains(&sj) {
                found = Some(midx);
                break;
            }
        }
        if let Some(midx) = found {
            machine_jobs[midx].insert(si);
            machine_jobs[midx].insert(sj);
            machine_ys[midx].insert((si, sj), y);
        } else {
            let mut jset = FxHashSet::default();
            jset.insert(si);
            jset.insert(sj);
            machine_jobs.push(jset);
            let mut ys = FxHashMap::default();
            ys.insert((si, sj), y);
            machine_ys.push(ys);
        }
    }
    for (midx, jobs) in machine_jobs.iter().enumerate() {
        if jobs.len() < 3 {
            continue;
        }
        let jl: Vec<usize> = jobs.iter().copied().collect();
        let ys = &machine_ys[midx];
        for a in 0..jl.len() {
            for b in (a + 1)..jl.len() {
                for c in (b + 1)..jl.len() {
                    let (i, j, k) = (jl[a], jl[b], jl[c]);
                    for tri in &[[(i, j), (j, k), (k, i)], [(i, k), (k, j), (j, i)]] {
                        if let (Some(&y1), Some(&y2), Some(&y3)) =
                            (ys.get(&tri[0]), ys.get(&tri[1]), ys.get(&tri[2]))
                        {
                            let mut row = vec![zero; n];
                            row[y1] = one;
                            row[y2] = one;
                            row[y3] = one;
                            let viol = x[y1] + x[y2] + x[y3] - T::from_f64(2.0).unwrap();
                            if viol > T::from_f64(1e-4).unwrap() {
                                pool.add(Cut::new(row, T::from_f64(2.0).unwrap(), viol));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Cross-row implication cuts (implication-based bound tightening).
///
/// Detects pairwise implications x_j=1 ⇒ x_k=0 that the single-row clique
/// detector misses. The single-row check (a_ij + a_ik > b_i) catches direct
/// conflicts within one constraint; this probes each binary variable by
/// tentatively fixing it to 1 and propagating through ALL constraints to
/// find variables forced to 0. The resulting implication x_j + x_k ≤ 1 is
/// valid across the entire problem, not just within one row.
///
/// Uses a single forward propagation pass per candidate (not full FBBT
/// iteration to convergence — that's the probing presolve's job). The
/// existing CutPool orthogonality filter automatically deduplicates against
/// cuts already generated by the clique detector.
pub fn generate_implied_bound_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    var_types: &[VarType],
    lb: &[T],
    ub: &[T],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-10).unwrap();
    let zero = T::zero();
    let one = T::one();
    let n = x.len();
    let m = b.len();

    // Only probe binary variables that are unfixed.
    let candidates: Vec<usize> = (0..n)
        .filter(|&j| var_types[j] == VarType::Binary && ub[j] - lb[j] > eps)
        .collect();
    if candidates.is_empty() {
        return;
    }

    // Limit: probing every binary is O(n_bin * m * n). For large n_bin,
    // only probe the most connected variables (rank by
    // column density).
    let max_probes = 30usize;
    let probes: Vec<usize> = if candidates.len() <= max_probes {
        candidates
    } else {
        let mut ranked: Vec<(usize, usize)> = candidates
            .iter()
            .map(|&j| {
                let nz = (0..m).filter(|&i| a.get(i, j).abs() > eps).count();
                (j, nz)
            })
            .collect();
        ranked.sort_by_key(|(_, nz)| std::cmp::Reverse(*nz));
        ranked.truncate(max_probes);
        ranked.into_iter().map(|(j, _)| j).collect()
    };

    let mut imp_pairs: Vec<(usize, usize)> = Vec::new();

    // Row classification is input-independent: a row is usable for sound
    // binary implications iff its coefficients are all nonnegative. Hoisted
    // out of the probe loop — it was recomputed per row per propagation pass
    // (O(m·n) each) although `a` never changes across probes or passes.
    let usable_rows: Vec<usize> = (0..m)
        .filter(|&i| (0..n).all(|k| a.get(i, k) >= -eps))
        .collect();
    if usable_rows.is_empty() {
        return;
    }

    for &j in &probes {
        // Start with current bounds, fix x_j = 1.
        let mut tmp_ub = ub.to_vec();
        tmp_ub[j] = one;

        // Single forward propagation pass: for each row with all-nonnegative
        // coefficients, check if any remaining binary variable must be 0.
        let mut changed = true;
        while changed {
            changed = false;
            for &i in &usable_rows {

                let rhs = b[i];
                // The maximum possible contribution if all remaining free
                // binaries were set to 1. Variables with zero upper bound
                // contribute nothing.
                let mut max_contrib = zero;
                for k in 0..n {
                    if tmp_ub[k] > eps {
                        max_contrib += a.get(i, k).max(zero);
                    }
                }
                // If even the max possible contribution can't exceed rhs,
                // no fixing is possible from this row.
                if max_contrib <= rhs + eps {
                    continue;
                }

                // Check each binary for necessary fixing.
                for k in 0..n {
                    if k == j || var_types[k] != VarType::Binary {
                        continue;
                    }
                    if tmp_ub[k] <= eps {
                        continue;
                    }
                    let a_ik = a.get(i, k);
                    if a_ik <= eps {
                        continue;
                    }
                    // If a_ik alone (with x_j already at 1) exceeds rhs, x_k must be 0.
                    let a_ij = a.get(i, j);
                    if a_ij + a_ik > rhs + eps {
                        tmp_ub[k] = zero;
                        changed = true;
                        imp_pairs.push((j, k));
                    }
                }
            }
        }
    }

    // Sort, deduplicate, generate cuts.
    imp_pairs.sort();
    imp_pairs.dedup();

    for (j, k) in imp_pairs {
        let mut row = vec![zero; n];
        row[j] = one;
        row[k] = one;
        let violation = x[j] + x[k] - one;
        if violation > eps {
            pool.add(Cut {
                row,
                rhs: one,
                active: true,
                violation,
                age: 0,
            });
        }
    }
}

/// Projected implied bound cuts: tighten `x_j ≤ l'_j` from knapsack rows
/// against the variable bounds, emitted as a cut (pool row) rather than a
/// bound change.
///
/// For a knapsack row `Σ_k a_k·x_k ≤ b` with all `a_k ≥ 0`, projecting
/// every other variable out of the row at its lower bound gives the implied
/// upper bound
///     l'_j = (b − Σ_{k≠j} a_k·lb_k) / a_j,
/// which every LP-feasible point satisfies (`a_j x_j ≤ b − Σ_{k≠j} a_k x_k
/// ≤ b − Σ_{k≠j} a_k lb_k`). As an LP inequality it is therefore never
/// violated; its cutting power comes from integrality — an *integer* x_j
/// bounded by l'_j is in fact bounded by `⌊l'_j⌋`, and a fractional LP
/// point with `x_j ∈ (⌊l'_j⌋, l'_j]` violates the floored cut. The bound is
/// projected against the *current* bounds: after reduced-cost fixing or any
/// other bound tightening, the same row yields strictly stronger cuts.
///
/// Only ever discards integer points with `x_j > ⌊l'_j⌋`, which — being
/// integer and above the LP-valid bound l'_j — do not exist in the integer
/// hull of the row, so every emitted cut holds for all integer-feasible
/// points (validated exhaustively in the tests). Continuous variables get
/// no cut: their implied bound is LP-valid and hence never violated.
pub fn generate_projected_implied_bound_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    lb: &[T],
    ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-10).unwrap();
    let zero = T::zero();
    let one = T::one();
    let n = x.len();
    let m = b.len();

    for i in 0..m {
        // Knapsack rows only: no negative coefficients (a mixed-sign row
        // does not bound any single variable from the others' lower bounds).
        let mut knapsack = true;
        for j in 0..n {
            if a.get(i, j) < -eps {
                knapsack = false;
                break;
            }
        }
        if !knapsack {
            continue;
        }
        let rhs = b[i];
        // Project the row's variables out at their lower bounds.
        let mut s_lo = zero;
        for j in 0..n {
            let aij = a.get(i, j);
            if aij > eps {
                s_lo += aij * lb[j];
            }
        }
        if !s_lo.is_finite() {
            continue;
        }
        for j in 0..n {
            let aij = a.get(i, j);
            if aij <= eps {
                continue;
            }
            if var_types[j] == VarType::Continuous {
                continue;
            }
            if ub[j] - lb[j] <= eps {
                continue;
            } // already fixed
            let l_j = (rhs - (s_lo - aij * lb[j])) / aij;
            if !l_j.is_finite() {
                continue;
            }
            // The floor is the whole point: l_j itself is LP-valid (never
            // violated), floor(l_j) is integer-valid and can cut a
            // fractional x_j in the band (floor(l_j), l_j].
            if l_j < ub[j] - T::from_f64(1e-6).unwrap() {
                let new_ub = l_j.floor();
                let violation = x[j] - new_ub;
                if violation > T::from_f64(1e-4).unwrap() {
                    let mut row = vec![zero; n];
                    row[j] = one;
                    pool.add(Cut {
                        row,
                        rhs: new_ub,
                        active: true,
                        violation,
                        age: 0,
                    });
                }
            }
        }
    }
}

/// Flow cover cuts: target fixed-charge / lot-sizing
/// constraints of the form Σ a_j·x_j − b·y ≤ d where y is binary. These
/// appear in production planning, facility location, and job-shop scheduling.
///
/// The simplified flow cover inequality for one positive-continuous and one
/// negative-binary per row (the most common lot-sizing shape):
///   x_j ≤ min(d + b, a_j·u_j)·y_j   when d ≥ 0 and b > 0
/// This is tighter than the original big-M x_j ≤ M·y_j because it uses the
/// variable's actual upper bound u_j rather than the generic M.
///
/// Full flow cover also handles multi-variable rows and lifting.
/// This implementation targets the single-positive-single-negative case
/// (lot-sizing/big-M) which is the most common and impactful.
pub fn generate_flow_cover_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a_mat: &DenseMatrix<T>,
    b: &[T],
    ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-9).unwrap();
    let zero = T::zero();
    let one = T::one();
    let n = x.len();
    let m = b.len();

    for i in 0..m {
        let rhs = b[i];
        if rhs < -eps {
            continue;
        } // only ≤ constraints with non-negative RHS

        // Count positive and negative coefficients; only handle the simple
        // one-positive-one-negative case (lot-sizing / big-M).
        let mut pos: Option<(usize, T)> = None;
        let mut neg: Option<(usize, T)> = None;
        let mut other_nz = 0usize;
        for j in 0..n {
            let aij = a_mat.get(i, j);
            if aij > eps {
                if pos.is_none() {
                    pos = Some((j, aij));
                } else {
                    other_nz += 1;
                }
            } else if aij < -eps {
                if neg.is_none() {
                    neg = Some((j, -aij));
                } else {
                    other_nz += 1;
                }
            }
        }
        if other_nz > 0 {
            continue;
        }
        let (p_j, a_pos) = match pos {
            Some(v) => v,
            None => continue,
        };
        let (n_j, a_neg) = match neg {
            Some(v) => v,
            None => continue,
        };
        if var_types[n_j] != VarType::Binary {
            continue;
        }

        // Original constraint: a_pos·x_j − a_neg·y ≤ rhs
        // Equivalent to: x_j ≤ (rhs + a_neg·y) / a_pos
        // When y=0: x_j ≤ rhs / a_pos
        // When y=1: x_j ≤ (rhs + a_neg) / a_pos
        //
        // Flow cover tightening: x_j ≤ min(rhs/a_pos, u_j)·(1−y) + min((rhs+a_neg)/a_pos, u_j)·y
        // For the typical case rhs ≥ 0:
        //   y=0 ⇒ x_j ≤ min(rhs/a_pos, u_j)
        //   y=1 ⇒ x_j ≤ min((rhs+a_neg)/a_pos, u_j)
        // This can be written as: x_j − α·y ≤ β where α and β depend on the bounds.

        let x_ub = ub[p_j];
        if x_ub >= T::from_f64(crate::INF_BOUND).unwrap() {
            continue;
        } // unbounded, can't tighten

        // Simple flow cover inequality: x_j ≤ u_j·y  (if rhs=0 and a_pos ≤ a_neg)
        // or more generally:  x_j − (u_j − rhs/a_pos)·y ≤ rhs/a_pos
        let bound_y0 = (rhs / a_pos).min(x_ub); // max x_j when y=0
        let bound_y1 = ((rhs + a_neg) / a_pos).min(x_ub); // max x_j when y=1

        // Flow cover cut: x_j − (bound_y1 − bound_y0)·y ≤ bound_y0
        let alpha = bound_y1 - bound_y0; // coefficient on y (how much x_j can increase when y=1)
        let beta = bound_y0;

        // Generate the cut: x_j − alpha·y ≤ beta
        // Expressed in standard form: 1·x_j + (−alpha)·y ≤ beta
        let mut row = vec![zero; n];
        row[p_j] = one;
        row[n_j] = -alpha;

        let violation = x[p_j] - alpha * x[n_j] - beta;
        if violation > eps {
            pool.add(Cut {
                row,
                rhs: beta,
                active: true,
                violation,
                age: 0,
            });
        }
    }
}

/// Multi-row cover cuts: when two knapsack rows share variables,
/// a cover in one row can be strengthened using weights from the other.
/// Multi-dimensional knapsack cover separation: when two knapsack rows
/// share variables, combining them yields a stronger cut than either
/// row’s cover alone -- information single-row cover cuts structurally
/// cannot access.
///
/// For each pair of knapsack rows (i,k) the two are aggregated with unit multipliers:
/// `(a_i + a_k)·x ≤ b_i + b_k` is implied by the pair, so a cover with respect to the
/// summed weights and summed capacity certifies infeasibility and `Σ_{j∈C} x_j ≤ |C|−1`
/// is valid.
///
/// The aggregation has to be a genuine one. This previously took `w_j = max(a_ij, a_kj)`
/// against `cap = min(b_i, b_k)`, which certifies nothing: `Σ_C max(a_ij, a_kj)` is an
/// upper bound on *both* row activities at once, so exceeding `min(b_i, b_k)` is
/// consistent with neither row being violated -- and the cut then forbids a combination
/// both rows permit. Verified against the integer hull (maximising each cut's own
/// left-hand side over the feasible set, which is exact at any size, unlike enumeration):
/// 344 of 772 cuts generated on mdk_n30_k3 were invalid, every one of them a pure cover
/// whose support was not a cover for any row -- for instance nine items summing to 159.9,
/// 160.5 and 136.8 against capacities of 186.8, 202.7 and 180.7.
///
/// Also generates covers from the fully aggregated row (sum of all
/// all-nonneg knapsack rows), which gives each item its total resource
/// consumption weight against the total resource budget.
pub fn generate_multirow_cover_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-8).unwrap();
    let one = T::one();
    let zero = T::zero();
    let n = x.len();
    let m = b.len();

    // Structure-driven multi-row detection (packing sets, flow paths) avoids
    // brute-force pairwise O(m²) scans. Without that infrastructure, cap m
    // to prevent pathological blow-up on instances with many constraints.
    // The shared.len()<2 guard catches most non-matching pairs, but the O(n)
    // column scan per pair still costs. m≤300 covers the entire current suite.
    if m > 300 {
        return;
    }

    for i in 0..m {
        if b[i] <= zero {
            continue;
        }
        // Row i: all-nonneg, with binary variables.
        let mut all_nonneg = true;
        for j in 0..n {
            if a.get(i, j) < -eps {
                all_nonneg = false;
                break;
            }
        }
        if !all_nonneg {
            continue;
        }

        for k in i + 1..m {
            if b[k] <= zero {
                continue;
            }
            let mut all_nonneg_k = true;
            for j in 0..n {
                if a.get(k, j) < -eps {
                    all_nonneg_k = false;
                    break;
                }
            }
            if !all_nonneg_k {
                continue;
            }

            // Find shared binary variables between the two rows.
            let mut shared: Vec<(usize, T, T)> = Vec::new();
            for j in 0..n {
                if var_types[j] != VarType::Binary {
                    continue;
                }
                let aij = a.get(i, j);
                let akj = a.get(k, j);
                if aij > eps && akj > eps {
                    shared.push((j, aij, akj));
                }
            }
            if shared.len() < 2 {
                continue;
            }

            // Cover with respect to the unit aggregation of the two rows: weight
            // `a_ij + a_kj` against capacity `b_i + b_k`. Restricting the cover to the
            // shared columns stays sound because both rows are all-nonneg, so every
            // omitted term only adds to the activity.
            let cap = b[i] + b[k];
            shared.sort_by(|p, q| {
                let wp = p.1 + p.2;
                let wq = q.1 + q.2;
                x[q.0]
                    .partial_cmp(&x[p.0])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| wq.partial_cmp(&wp).unwrap_or(std::cmp::Ordering::Equal))
            });

            let mut cover = Vec::new();
            let mut sum = zero;
            for &(j, aij, akj) in &shared {
                cover.push(j);
                sum += aij + akj;
                if sum > cap + eps {
                    break;
                }
            }
            if sum <= cap + eps || cover.len() < 2 {
                continue;
            }
            // Make it minimal.
            while cover.len() > 1 {
                let last = cover[cover.len() - 1];
                let w_last = a.get(i, last) + a.get(k, last);
                if sum - w_last > cap + eps {
                    sum -= w_last;
                    cover.pop();
                } else {
                    break;
                }
            }

            let cs = cover.len();
            let mut row = vec![zero; n];
            for &j in &cover {
                row[j] = one;
            }
            let rhs = T::from_usize(cs - 1).unwrap();
            let viol = violation_of(&row, x, rhs);
            if viol > eps {
                pool.add(Cut::new(row, rhs, viol));
            }
        }
    }

    // ── Fully aggregated row (sum of all all-nonneg knapsack rows) ──────
    // Summing every all-nonneg ≤ row gives a single valid ≤ constraint.
    // For multi-dimensional knapsack (MDK), this aggregated row finds
    // covers no single-row or pair-based combination can detect: an item's
    // aggregated weight is its TOTAL resource consumption, and the
    // aggregated capacity is the TOTAL resource budget. The aggregated-row
    // cover inequality Σ_{j∈C} x_j ≤ |C|−1 is valid for the summed constraint
    // and therefore also for every original row that contributed to it.
    {
        let mut agg_w = vec![zero; n];
        let mut agg_rhs = zero;
        let mut n_krows: usize = 0;
        for i in 0..m {
            if b[i] <= zero {
                continue;
            }
            if (0..n).any(|j| a.get(i, j) < -eps) {
                continue;
            }
            n_krows += 1;
            agg_rhs += b[i];
            for j in 0..n {
                let aij = a.get(i, j);
                if aij > eps {
                    agg_w[j] += aij;
                }
            }
        }
        if n_krows >= 2 {
            let cap = agg_rhs;
            let mut items: Vec<(usize, T)> = Vec::new();
            for j in 0..n {
                if var_types[j] != VarType::Binary {
                    continue;
                }
                if agg_w[j] > eps && agg_w[j] <= cap {
                    items.push((j, agg_w[j]));
                }
            }
            if items.len() >= 2 {
                // Build up to 3 diverse covers via greedy orderings, matching
                // the single-row `generate_cover_cuts` strategy.
                let mut seen_covers: FxHashSet<Vec<usize>> = FxHashSet::default();

                for pass in 0u8..3 {
                    let ord = [
                        CoverOrder::Value,
                        CoverOrder::Weight,
                        CoverOrder::Efficiency,
                    ][pass as usize];
                    let Some((cover, _)) =
                        greedy_cover(&mut items, ord, x, &|j| agg_w[j], cap, eps, zero)
                    else {
                        continue;
                    };
                    let mut sorted = cover.clone();
                    sorted.sort_unstable();
                    if !seen_covers.insert(sorted) {
                        continue;
                    }

                    let cs = cover.len();
                    let candidates: Vec<usize> = (0..n)
                        .filter(|&j| {
                            !cover.contains(&j) && agg_w[j] > eps && var_types[j] == VarType::Binary
                        })
                        .collect();
                    let synth_a = DenseMatrix::from_row_major(1, n, agg_w.clone());
                    let mut row = vec![zero; n];
                    for &j in &cover {
                        row[j] = one;
                    }
                    for (j, alpha) in lift_cover(0, &synth_a, cap, &cover, &candidates) {
                        row[j] = alpha;
                    }
                    let rhs = T::from_usize(cs - 1).unwrap();
                    let viol = violation_of(&row, x, rhs);
                    if viol > eps {
                        pool.add(Cut::new(row, rhs, viol));
                    }
                }
            }
        }
    }
}

// ── (ℓ,S) inequality separation for uncapacitated lot-sizing ──────────────
//
// The standard (ℓ,S) inequality (Barany–Van Roy–Wolsey 1984) for single-item
// uncapacitated lot-sizing (ULS / Wagner-Whitin):
//
//   inv[a−1] + Σ_{t=a}^{b} D_{tb} · y_t  ≥  D_{ab}      (1 ≤ a ≤ b ≤ T)
//
// where D_{ab} = Σ_{k=a}^{b} d_k is the cumulative demand from a to b, and
// d_k is the demand in period k.  These inequalities make the LP relaxation
// describe the convex hull of integer solutions — the LP is ALWAYS integral
// once they are added, turning ULS from a T-decision B&B problem into a
// root-node solve.
//
// Separation is O(T²) per item — trivial (400 checks for T=20, < 0.1 ms).
// Structure detection: flow-balance rows (prod + inv_in − inv_out = demand)
// matched to linking rows (prod − M·setup ≤ 0, setup is binary) by
// production-column identity.

/// Try to detect single-item lot-sizing structure and separate (ℓ,S) cuts.
pub fn separate_lot_sizing_ls_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    cones: &[Cone],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) -> usize {
    let eps = T::from_f64(1e-8).unwrap();
    let zero = T::zero();
    let one = T::one();
    let n = x.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return 0;
    }

    // Zero-cone (equality) rows form a prefix in canonical cone order; the
    // flow-balance equalities must live among them (linking rows are ≤
    // inequalities and live after them).
    let n_eq = crate::n_eq_rows(cones);

    // Phase 1: detect lot-sizing structure from the constraint matrix.
    // Flow-balance row (Zero-cone equality): exactly the {+1 prod, +1 inv_in,
    // −1 inv_out} pattern, RHS = demand > 0 (first period has no inv_in).
    // Linking row: exactly 2 nonzeros {+1 prod, −M·setup}, RHS = 0, setup
    // binary, prod column a known production column.
    let mut prod_col: Vec<usize> = Vec::new();
    let mut inv_col: Vec<usize> = Vec::new();
    let mut setup_col: Vec<usize> = Vec::new();
    let mut demand: Vec<T> = Vec::new();
    let mut periods = 0usize;

    // Map from production column → period index (for matching linking rows)
    let mut prod_to_period: FxHashMap<usize, usize> = FxHashMap::default();

    for i in 0..m {
        let mut nz: Vec<(usize, T)> = Vec::new();
        for j in 0..n {
            let v = a.get(i, j);
            if v.abs() > eps {
                nz.push((j, v));
            }
        }
        let nzn = nz.len();

        // Flow balance: prod + inv_in − inv_out = demand, with the exact
        // coefficient pattern above (no loose |v|>0 matches -- a wrong-sign
        // or scaled inventory column must not be detected as one).
        if i < n_eq && (nzn == 2 || nzn == 3) && b[i] > eps {
            let mut pos: Vec<usize> = Vec::new();
            let mut neg: Vec<usize> = Vec::new();
            let mut exact = true;
            for &(j, v) in &nz {
                if (v - one).abs() <= eps {
                    pos.push(j);
                } else if (v + one).abs() <= eps {
                    neg.push(j);
                } else {
                    exact = false;
                }
            }
            let expected_pos = if nzn == 2 { 1 } else { 2 };
            if exact
                && pos.len() == expected_pos
                && neg.len() == 1
                && pos.iter().all(|&j| var_types[j] == VarType::Continuous)
                && var_types[neg[0]] == VarType::Continuous
            {
                let p = periods;
                prod_to_period.insert(pos[0], p);
                prod_col.push(pos[0]);
                inv_col.push(neg[0]);
                demand.push(b[i]);
                setup_col.push(0); // placeholder, filled by linking row
                periods += 1;
            }
        } else if nzn == 2 && b[i].abs() < eps {
            // Linking row: prod − M·setup ≤ 0, exactly {+1 prod, −M·setup},
            // the prod coefficient matching the +1 pattern of the flow
            // balance and the column being one of the detected production
            // columns; setup is binary with M > 0.
            let mut prod_j = None;
            let mut setup_j = None;
            let mut exact = true;
            for &(j, v) in &nz {
                if (v - one).abs() <= eps && var_types[j] == VarType::Continuous {
                    prod_j = Some(j);
                } else if v < -eps && var_types[j] == VarType::Binary {
                    setup_j = Some(j);
                } else {
                    exact = false;
                }
            }
            if exact {
                if let (Some(pj), Some(sj)) = (prod_j, setup_j) {
                    if let Some(&p) = prod_to_period.get(&pj) {
                        setup_col[p] = sj;
                    }
                }
            }
        }
    }

    // Need at least 2 periods with complete detection (every period has a setup).
    if periods < 2 {
        return 0;
    }
    if setup_col.contains(&0) {
        return 0;
    }
    // (ℓ,S) separation active — cuts will be generated if violated

    // Phase 2: separate (ℓ,S) cuts.
    // Precompute cumulative demand D_{tb} for all t ≤ b.
    let mut cum = vec![vec![T::zero(); periods]; periods];
    for a in 0..periods {
        let mut s = T::zero();
        for b in a..periods {
            s += demand[b];
            cum[a][b] = s;
        }
    }

    let mut added = 0usize;

    for a in 0..periods {
        for b in a..periods {
            let d_ab = cum[a][b];
            if d_ab <= eps {
                continue;
            }

            // LHS = inv[a-1] + Σ_{t=a}^{b} D_{tb} * y[t]
            // a==0: no carry-in inventory, first-period demand is handled by
            // the setup term Σ D_{tb}·y_t which already covers all production.
            let mut lhs = T::zero();
            if a > 0 {
                lhs += x[inv_col[a - 1]];
            }
            for t in a..=b {
                lhs += cum[t][b] * x[setup_col[t]];
            }

            let violation = d_ab - lhs;
            if violation > eps {
                // The (ℓ,S) inequality is a LOWER bound: inv[a-1] +
                // Σ_{t∈[a,b]} D_tb·y_t ≥ D_ab (carry-in inventory plus
                // production capacity via setups must cover the demand of
                // [a,b]). The pool's convention is `row·x ≤ rhs`, so store it
                // negated: −inv[a-1] − Σ D_tb·y_t ≤ −D_ab. The previous
                // version stored the ≥ inequality as-is (`row·x ≤ d_ab`),
                // which excluded every feasible point whose lhs exceeds
                // D_ab -- carry-in inventory combined with a setup on [a,b]
                // (verified: at the feasible prod=[30,0,0], inv=[20,5,0],
                // y=[1,1,0] point, the (a,b)=(1,2) "cut" had lhs 40 > rhs 20).
                let mut row = vec![zero; n];
                if a > 0 {
                    row[inv_col[a - 1]] = -one;
                }
                for t in a..=b {
                    row[setup_col[t]] = -cum[t][b];
                }
                pool.add(Cut {
                    row,
                    rhs: -d_ab,
                    active: true,
                    violation,
                    age: 0,
                });
                added += 1;
            }
        }
    }

    // done
    added
}

// ═══════════════════════════════════════════════════════════════════════════════
// Set-Packing Cover Cuts
// ═══════════════════════════════════════════════════════════════════════════════
//
// Set-packing constraints: Σ_{j∈G} x_j ≤ 1 for a set G of binary variables.
// When set-packing structure coexists with a knapsack constraint aᵀx ≤ b,
// the standard cover inequality can be strengthened by exploiting the fact
// that at most one member of each packing group can be set to 1.
//
// Sound set-packing cover inequality (the correct formula, not the unsound
// version that was previously disabled):
//
//   For a cover C with Σ_{j∈C} a_j > b, and packing groups G_1,…,G_g:
//   - Let C_i = C ∩ G_i, k_i = |C_i|.
//   - From each G_i with k_i ≥ 2, keep only the max-weight variable
//     j*_i = argmax_{j∈C_i} a_j, drop the other k_i−1 from the LHS.
//   - Check if the packing-cover (max-weight reps + non-packing members)
//     still exceeds b: Σ_i a_{j*_i} + Σ_{j∈C\∪G_i} a_j > b.
//   - If yes, the inequality is valid:
//     Σ_i x_{j*_i} + Σ_{j∈C\∪G_i} x_j ≤ |C| − Σ_i (k_i−1) − 1
//
// The key insight: dropping k_i−1 members from a packing group reduces both the
// LHS (fewer variables) and the RHS (fewer variables needed to violate the
// knapsack, since at most one per group can be 1 anyway).
//
// Without the "packing-cover still exceeds b" check, the inequality could be
// invalid: if the max-weight reps alone don't exceed b, then the cover only
// violates the knapsack because of the extra packing-group members, and
// dropping them changes the cover's fundamental property.

/// Generate set-packing cover cuts: strengthen standard cover cuts using
/// set-packing structure (rows where Σ_{j∈G} x_j ≤ 1, all binary, all
/// coefficients 1.0).
///
/// For each knapsack-like row, detect packing groups intersecting the covers
/// found, apply the sound drop-reps+reduce-rhs strengthening, and add the
/// resulting cuts to the pool. Only applies when the packing-cover (max-weight
/// one rep per group + non-packing cover members) still exceeds the knapsack
/// capacity.
pub fn generate_packing_cover_cuts<T: Scalar + PartialOrd>(
    x: &[T],
    a: &DenseMatrix<T>,
    b: &[T],
    _ub: &[T],
    var_types: &[VarType],
    pool: &mut CutPool<T>,
) {
    let eps = T::from_f64(1e-8).unwrap();
    let one = T::one();
    let zero = T::zero();
    let n = x.len();
    let m = b.len();

    // Detect set-packing structure.
    let mut packing_membership: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut packing_groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..m {
        if (b[i] - one).abs() > eps {
            continue;
        }
        let mut group = Vec::new();
        let mut valid = true;
        for j in 0..n {
            let aij = a.get(i, j);
            if aij.abs() < eps {
                continue;
            }
            if (aij - one).abs() > eps {
                valid = false;
                break;
            }
            if var_types[j] != VarType::Binary {
                valid = false;
                break;
            }
            group.push(j);
        }
        if valid && group.len() >= 2 {
            for &j in &group {
                packing_membership[j].push(packing_groups.len());
            }
            packing_groups.push(group);
        }
    }
    if packing_groups.is_empty() {
        return;
    }

    // For each knapsack-like row, find covers and strengthen.
    // Pre-allocate per-row weight cache reused across orderings.
    let mut weight_of: Vec<T> = vec![zero; n];
    let mut seen_covers: FxHashSet<Vec<usize>> = FxHashSet::default();

    for i in 0..m {
        if b[i] <= zero {
            continue;
        }
        let mut has_bin = false;
        let mut has_packing_overlap = false;
        let mut all_nonneg = true;
        for j in 0..n {
            let aij = a.get(i, j);
            if aij < -eps {
                all_nonneg = false;
                break;
            }
            if aij > eps && var_types[j] == VarType::Binary {
                has_bin = true;
                if !packing_membership[j].is_empty() {
                    has_packing_overlap = true;
                }
            }
        }
        if !all_nonneg || !has_bin || !has_packing_overlap {
            continue;
        }

        // Build weight cache: O(1) weight lookups for the rest of this row.
        let mut items: Vec<(usize, T)> = Vec::new();
        for j in 0..n {
            let aij = a.get(i, j);
            weight_of[j] = aij;
            if aij > eps && aij <= b[i] && var_types[j] == VarType::Binary {
                items.push((j, aij));
            }
        }
        if items.len() < 2 {
            continue;
        }

        let bi = b[i];
        seen_covers.clear();

        // Three orderings — weight desc, LP value desc, efficiency desc.
        for pass in 0u8..3 {
            let ord = [
                CoverOrder::Weight,
                CoverOrder::Value,
                CoverOrder::Efficiency,
            ][pass as usize];
            let Some((cover, _)) =
                greedy_cover(&mut items, ord, x, &|j| weight_of[j], bi, eps, zero)
            else {
                continue;
            };

            // Deduplicate
            let mut sorted_cover = cover.clone();
            sorted_cover.sort_unstable();
            if !seen_covers.insert(sorted_cover) {
                continue;
            }

            // Compute packing-group intersections.
            let mut packing_intersections: Vec<(usize, usize, T, usize)> = Vec::new();

            for gidx in 0..packing_groups.len() {
                let mut max_w = zero;
                let mut max_j = usize::MAX;
                let mut k = 0usize;
                for &j in &cover {
                    if packing_membership[j].contains(&gidx) {
                        let w = weight_of[j];
                        k += 1;
                        if w > max_w {
                            max_w = w;
                            max_j = j;
                        }
                    }
                }
                if k >= 2 {
                    packing_intersections.push((gidx, k, max_w, max_j));
                }
            }
            if packing_intersections.is_empty() {
                continue;
            }

            // Build the set of variables to KEEP (max-weight reps + non-packing)
            let mut keep: FxHashSet<usize> = FxHashSet::default();
            let mut total_drop = 0usize;
            let mut max_rep_weight = zero;

            for &(_, k, max_w, max_j) in &packing_intersections {
                keep.insert(max_j);
                max_rep_weight += max_w;
                total_drop += k - 1;
            }

            let mut non_packing_weight = zero;
            for &j in &cover {
                let mut in_intersecting = false;
                for &(gidx, _, _, _) in &packing_intersections {
                    if packing_membership[j].contains(&gidx) {
                        in_intersecting = true;
                        break;
                    }
                }
                if !in_intersecting {
                    keep.insert(j);
                    non_packing_weight += weight_of[j];
                }
            }

            // Packing-cover check: the cover must still exceed capacity once
            // each packing group keeps only its max-weight rep.
            if max_rep_weight + non_packing_weight <= bi + eps {
                continue;
            }

            // Build the strengthened cut: keep one max-weight rep per group.
            let cs = cover.len();
            if total_drop + 1 > cs {
                continue;
            }
            let rhs_int = cs - 1 - total_drop;
            let rhs = T::from_usize(rhs_int).unwrap();

            let mut row = vec![zero; n];
            for &j in &keep {
                row[j] = one;
            }

            let viol = violation_of(&row, x, rhs);
            if viol > eps {
                pool.add(Cut::new(row, rhs, viol));
            }
        }
    }
}

#[cfg(test)]
mod mixing_tests {
    use super::*;
    use iconic_linalg::DenseMatrix;

    fn vartypes(n: usize) -> Vec<VarType> {
        vec![VarType::Binary; n]
    }

    /// Whatever the separator emits must be valid on the FULL feasible
    /// region of the group: every binary assignment × every feasible t in
    /// the true box range, re-checked independently (the separator's own
    /// exact validation, duplicated here as belt and braces).
    #[test]
    fn mixing_separator_emits_only_valid_cuts() {
        // ≤ rows with a shared term: y_1 + x0 ≤ 1.2, y_2 + x0 ≤ 1.0,
        // y_3 + 2·x0 ≤ 1.5 (the third row's y-coefficient is 1 but its
        // shared coefficient differs from the first two — only rows 1-2
        // group). x0 ∈ [0, 1].
        let n = 4usize;
        let mut a = DenseMatrix::zeros(2, n);
        let b = vec![1.2f64, 1.0];
        a.set(0, 0, 1.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 2, 1.0);
        let lb = vec![0.0f64, 0.0, 0.0, 0.0];
        let ub = vec![1.0f64, 1.0, 1.0, 1.0];
        let x = vec![0.5f64, 0.5, 0.5, 0.5];
        let mut p = CutPool::new(50, &vec![0.0; n], &vartypes(n));
        generate_mixing_cuts(&x, &a, &b, &lb, &ub, &vartypes(n), &mut p);
        for cut in &p.cuts {
            // Exhaustive re-check over y-assignments and a t-grid over the
            // true box range [t_lo, t_hi] = [0, 1] (shared x0 ∈ [0,1]).
            for mask in 0u32..4 {
                for ti in 0..=100 {
                    let t = 1.0 * (ti as f64) / 100.0;
                    // Feasible iff y_1 + t ≤ 1.2 and y_2 + t ≤ 1.0 (≤ rows,
                    // y-coefficient 1 in both).
                    let y1 = if mask & 1 == 1 { 1.0 } else { 0.0 };
                    let y2 = if (mask >> 1) & 1 == 1 { 1.0 } else { 0.0 };
                    if y1 + t > 1.2 + 1e-9 || y2 + t > 1.0 + 1e-9 {
                        continue;
                    }
                    let mut lhs = cut.row[0] * t; // x0 is the shared variable
                    lhs += cut.row[1] * y1 + cut.row[2] * y2 + cut.row[3] * 0.0;
                    assert!(
                        lhs <= cut.rhs + 1e-6,
                        "invalid mixing cut: lhs {lhs} > rhs {} at mask {mask} t {t}",
                        cut.rhs
                    );
                }
            }
        }
    }

    /// A group with a NON-unit y-coefficient: the validation must use the
    /// actual coefficient (the row 2·y + x0 ≤ 2.4 is a different constraint
    /// than y + x0 ≤ 1.2), and any emitted cut must still be valid.
    #[test]
    fn mixing_separator_handles_nonunit_y_coefficients() {
        let n = 4usize;
        let mut a = DenseMatrix::zeros(2, n);
        let b = vec![2.4f64, 1.0];
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0); // 2·y_1 + x0 ≤ 2.4
        a.set(1, 0, 1.0);
        a.set(1, 2, 1.0); // y_2 + x0 ≤ 1.0
        let lb = vec![0.0f64; n];
        let ub = vec![1.0f64; n];
        let x = vec![0.3f64, 0.7, 0.7, 0.5];
        let mut p = CutPool::new(50, &vec![0.0; n], &vartypes(n));
        generate_mixing_cuts(&x, &a, &b, &lb, &ub, &vartypes(n), &mut p);
        for cut in &p.cuts {
            for mask in 0u32..4 {
                for ti in 0..=100 {
                    let t = 1.0 * (ti as f64) / 100.0;
                    let y1 = if mask & 1 == 1 { 1.0 } else { 0.0 };
                    let y2 = if (mask >> 1) & 1 == 1 { 1.0 } else { 0.0 };
                    if 2.0 * y1 + t > 2.4 + 1e-9 || y2 + t > 1.0 + 1e-9 {
                        continue;
                    }
                    let mut lhs = cut.row[0] * t;
                    lhs += cut.row[1] * y1 + cut.row[2] * y2;
                    assert!(lhs <= cut.rhs + 1e-6, "invalid cut at mask {mask} t {t}");
                }
            }
        }
    }
}

#[cfg(test)]
mod mixing_guard_tests {
    use super::*;
    use iconic_linalg::DenseMatrix;

    /// A group whose rows force every binary above the box range — every
    /// assignment's feasible t-interval is empty — must emit NOTHING (the
    /// unvalidated-emission hole: measured on the vcover presolved shape,
    /// where an unvalidated cut moved a Solved optimum from 16 to 17).
    #[test]
    fn mixing_separator_emits_nothing_when_no_assignment_is_checkable() {
        // Rows y_1 − x0 ≤ −3, y_2 − x0 ≤ −3 (shared −x0, t = −x0 ∈ [−1, 0]):
        // t ≤ −3 + y_i forces every y_i below the box range → all intervals
        // empty → the cut must not emit.
        let n = 3usize;
        let mut a = DenseMatrix::zeros(2, n);
        let b = vec![-3.0f64, -3.0];
        a.set(0, 0, -1.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, -1.0);
        a.set(1, 2, 1.0);
        let lb = vec![0.0f64; n];
        let ub = vec![1.0f64; n];
        let x = vec![0.5f64, 0.5, 0.5];
        let mut p = CutPool::new(50, &vec![0.0; n], &vec![VarType::Binary; n]);
        generate_mixing_cuts(&x, &a, &b, &lb, &ub, &vec![VarType::Binary; n], &mut p);
        assert!(
            p.cuts.is_empty(),
            "an unvalidated cut must never be emitted"
        );
    }
}

#[cfg(test)]
mod projected_implied_tests {
    use super::*;
    use iconic_core::rng::XorShift;

    /// The projected implied bound cut must fire on a fractional LP point
    /// once a fixing has raised the projected-out variables' lower bounds,
    /// and every emitted cut must hold on every integer-feasible point
    /// (exhaustive over the box).
    ///
    /// Row: `2·x0 + 3·x1 + 4·x2 + 5·x3 ≤ 8` with x0 pinned at 1 (lb=ub=1).
    /// Projecting x0 out at its lower bound: l'_3 = (8 − 2·1)/5 = 1.2, so
    /// the integer cut is x3 ≤ 1 — violated at the fractional LP point
    /// x = (1, 1/6, 0, 1.1) (row is tight: 2 + 0.5 + 5.5 = 8). The other
    /// targets produce nothing: l'_1 = (8−2)/3 = 2 ≥ ub_1, l'_2 = 1.5 with
    /// floor 1 = ub_2, x0 is fixed.
    #[test]
    fn projected_implied_bound_cut_fires_after_a_fixing_and_holds_exhaustively() {
        let n = 4;
        let mut a = DenseMatrix::<f64>::zeros(1, n);
        for (j, v) in [2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            a.set(0, j, *v);
        }
        let b = vec![8.0];
        let var_types = vec![
            VarType::Binary,
            VarType::Binary,
            VarType::Binary,
            VarType::Integer,
        ];
        let lb = vec![1.0, 0.0, 0.0, 0.0];
        let ub = vec![1.0, 1.0, 1.0, 2.0];
        let x = vec![1.0, 1.0 / 6.0, 0.0, 1.1];
        let mut pool = CutPool::new(100, &[], &var_types);
        generate_projected_implied_bound_cuts(&x, &a, &b, &lb, &ub, &var_types, &mut pool);

        assert_eq!(pool.cuts.len(), 1, "exactly one cut expected: x3 <= 1");
        let cut = &pool.cuts[0];
        assert!((cut.row[3] - 1.0).abs() < 1e-9, "cut must be on x3");
        assert!(
            (cut.rhs - 1.0).abs() < 1e-9,
            "cut must be x3 <= 1, got rhs {}",
            cut.rhs
        );
        assert!(cut
            .row
            .iter()
            .enumerate()
            .filter(|&(j, _v)| j != 3)
            .all(|(_, v)| v.abs() < 1e-9));

        // Exhaustive validity: every integer point in the box that satisfies
        // the row must satisfy every emitted cut.
        for x1 in [0.0, 1.0] {
            for x2 in [0.0, 1.0] {
                for x3 in [0.0, 1.0, 2.0] {
                    let pt = [1.0, x1, x2, x3];
                    let row_val: f64 = (0..n).map(|j| a.get(0, j) * pt[j]).sum();
                    if row_val > b[0] + 1e-9 {
                        continue;
                    }
                    for c in &pool.cuts {
                        let lhs: f64 = (0..n).map(|j| c.row[j] * pt[j]).sum();
                        assert!(
                            lhs <= c.rhs + 1e-9,
                            "cut x3 <= {} excludes the row-feasible integer point {:?} (lhs {lhs})",
                            c.rhs,
                            pt
                        );
                    }
                }
            }
        }
    }

    /// A plain 0-1 knapsack with all lower bounds zero gets no projected
    /// implied bound cuts: l'_j = b/a_j ≥ 1 = ub_j for every variable (any
    /// j with b/a_j < 1 would have been fixed by presolve), so there is
    /// nothing to tighten. Guards against the family firing unconditionally
    /// and perturbing every knapsack solve.
    #[test]
    fn projected_implied_bound_cuts_never_fire_on_plain_binary_knapsacks() {
        let n = 4;
        let mut a = DenseMatrix::<f64>::zeros(1, n);
        for (j, v) in [2.0, 2.0, 2.0, 3.0].iter().enumerate() {
            a.set(0, j, *v);
        }
        let b = vec![7.0];
        let var_types = vec![VarType::Binary; n];
        let lb = vec![0.0; n];
        let ub = vec![1.0; n];
        // LP-feasible fractional point (row value 6 <= 7).
        let x = vec![0.5, 0.5, 0.5, 1.0];
        let mut pool = CutPool::new(100, &[], &var_types);
        generate_projected_implied_bound_cuts(&x, &a, &b, &lb, &ub, &var_types, &mut pool);
        assert!(
            pool.cuts.is_empty(),
            "plain 0-1 knapsack must get no cuts, got {}",
            pool.cuts.len()
        );
    }

    /// Randomized soundness: over random knapsack rows, bounds and variable
    /// types, every emitted cut must hold on every integer-feasible point in
    /// the box (exhaustive over the small n used here).
    #[test]
    fn projected_implied_bound_cuts_are_valid_on_all_integer_points() {
        for trial in 0u64..200 {
            let mut rng = XorShift::new(trial.wrapping_mul(2654435761).wrapping_add(7));
            let n = 2 + rng.pick(4);
            let m = 1 + rng.pick(2);
            let mut a = DenseMatrix::<f64>::zeros(m, n);
            for i in 0..m {
                for j in 0..n {
                    let mut v = rng.uniform(0.5, 6.0);
                    if rng.pick(3) == 0 {
                        v = -v;
                    } // occasional negative coeff -> not a knapsack row
                    a.set(i, j, v);
                }
            }
            let lb: Vec<f64> = (0..n).map(|_| rng.pick(3) as f64).collect();
            let ub: Vec<f64> = (0..n)
                .map(|j| lb[j] + 1.0 + rng.pick(4) as f64)
                .collect();
            let var_types: Vec<VarType> = (0..n)
                .map(|_| {
                    if rng.pick(3) == 0 {
                        VarType::Continuous
                    } else {
                        VarType::Integer
                    }
                })
                .collect();
            let b: Vec<f64> = (0..m)
                .map(|i| {
                    let lo_sum: f64 = (0..n).map(|j| a.get(i, j).max(0.0) * lb[j]).sum();
                    lo_sum + rng.uniform(0.5, 9.0)
                })
                .collect();
            // LP-feasible point: all at lower bounds satisfies the row by construction.
            let x: Vec<f64> = lb.clone();

            let mut pool = CutPool::new(100, &[], &var_types);
            generate_projected_implied_bound_cuts(&x, &a, &b, &lb, &ub, &var_types, &mut pool);

            // Exhaustive enumeration of integer points in the box.
            let mut idx = vec![0usize; n];
            loop {
                let pt: Vec<f64> = (0..n).map(|j| idx[j] as f64).collect();
                let row_ok = (0..m).all(|i| {
                    let rv: f64 = (0..n).map(|j| a.get(i, j) * pt[j]).sum();
                    rv <= b[i] + 1e-9
                });
                if row_ok {
                    for c in &pool.cuts {
                        let lhs: f64 = (0..n).map(|j| c.row[j] * pt[j]).sum();
                        assert!(
                            lhs <= c.rhs + 1e-9,
                            "trial {trial}: cut (rhs {}) excludes the row-feasible integer point {pt:?} (lhs {lhs})",
                            c.rhs
                        );
                    }
                }
                let mut k = 0;
                while k < n {
                    idx[k] += 1;
                    let lim = (ub[k] - lb[k]).max(0.0).floor() as usize + 1;
                    if idx[k] < lim {
                        break;
                    }
                    idx[k] = 0;
                    k += 1;
                }
                if k == n {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod disjunctive_tests {
    use super::*;

    /// Every cut family's soundness contract, checked exhaustively: any cut
    /// the family emits must be satisfied by every integer-feasible point,
    /// and must cut off the fractional LP point it was separated from.
    fn assert_cuts_sound_and_violated(
        pool: &CutPool<f64>,
        integer_points: &[Vec<f64>],
        x_star: &[f64],
    ) {
        for cut in &pool.cuts {
            for p in integer_points {
                let lhs: f64 = cut.row.iter().zip(p.iter()).map(|(c, &v)| c * v).sum();
                assert!(
                    lhs <= cut.rhs + 1e-6,
                    "cut must be valid for the integer hull: row·p = {lhs} > rhs {}",
                    cut.rhs
                );
            }
            let viol: f64 = cut
                .row
                .iter()
                .zip(x_star.iter())
                .map(|(c, &v)| c * v)
                .sum::<f64>()
                - cut.rhs;
            assert!(
                viol > 1e-5,
                "the cut must cut off the fractional LP point it was separated from: viol = {viol}"
            );
        }
    }

    /// Intersection cut: the cut from a fractional basic binary's
    /// tableau row is valid for the integer hull (exhaustive over the two
    /// integer points) and cuts off the fractional LP point.
    ///
    /// Instance: `2x0 + x1 − x2 = 1` (equality), binaries in [0,1]. Integer
    /// points: (0,1,0) and (1,0,1). Fractional LP point (0.5, 0, 0). The
    /// tableau row of the basic fractional binary x0 (value 0.5) with x1, x2
    /// nonbasic at their lower bounds 0 reads `x0 + 0.5·x1 − 0.5·x2 = 0.5`;
    /// the disjunction x0 ∈ {0,1} forces x1 + x2 = 1 on both sides, and the
    /// intersection cut x1 + x2 ≥ 1 separates (0.5, 0, 0).
    #[test]
    fn intersection_intersection_cut_is_sound_and_cuts_the_fractional_point() {
        let a = DenseMatrix::from_row_major(1, 3, vec![2.0, 1.0, -1.0]);
        let b = vec![1.0];
        let lb = vec![0.0; 3];
        let vt = vec![VarType::Binary; 3];
        let x_star = vec![0.5, 0.0, 0.0];
        let tableau = vec![iconic_simplex::TableauRow {
            basic_col: 0,
            basic_val: 0.5,
            coeffs: vec![(1, 0.5, false, 0.0), (2, -0.5, false, 0.0)],
        }];
        let mut pool: CutPool<f64> = CutPool::new(10, &[], &[]);
        generate_intersection_cuts(&x_star, &tableau, 3, &a, &b, &lb, &vt, &[], &mut pool);
        assert!(
            !pool.cuts.is_empty(),
            "the intersection cut must fire on the fractional basic binary"
        );
        assert_cuts_sound_and_violated(&pool, &[vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 1.0]], &x_star);
        // The cut from this row is exactly x1 + x2 ≥ 1 (≤ form: −x1 − x2 ≤ −1).
        let cut = &pool.cuts[0];
        assert!(
            (cut.row[1] + 1.0).abs() < 1e-9 && (cut.row[2] + 1.0).abs() < 1e-9,
            "expected the cut −x1 − x2 ≤ −1, got {:?}",
            cut.row
        );
    }

    /// Tableau intersection cuts end-to-end through a real simplex solve: solve a tiny
    /// fractional LP, extract the tableau rows of fractional basic binaries
    /// exactly like `solve_node_lp_simplex` does, and check the generated
    /// cuts against the full integer hull by enumeration.
    ///
    /// Instance: `x0 + x1 + x2 ≥ 1.5` (row −x0−x1−x2 ≤ −1.5), binaries in
    /// [0,1]. Every vertex has one variable at 0.5 and the other two at 0/1,
    /// so any optimal basis holds a fractional basic binary. Integer points:
    /// all points with ≥ 2 ones (4 points).
    #[test]
    fn intersection_cuts_from_a_real_simplex_tableau_are_sound() {
        let a = DenseMatrix::from_row_major(1, 3, vec![-1.0, -1.0, -1.0]);
        let b = vec![-1.5];
        let lb = vec![0.0; 3];
        let vt = vec![VarType::Binary; 3];
        // LP relaxation (min 0): the simplex lands on a fractional vertex.
        let c = vec![0.0, 0.0, 0.0, 0.0];
        let mut solver = DualSolver::from_csc(
            c,
            CscCols {
                col_start: vec![0, 1, 2, 3, 4],
                row_idx: vec![0, 0, 0, 0],
                val: vec![-1.0, -1.0, -1.0, 1.0],
            },
            vec![-1.5],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0, 1e20],
            1,
            4,
        );
        let sol = solver.cold_solve();
        assert_eq!(sol.status, iconic_simplex::Status::Optimal);
        let x_lp = sol.x;
        assert!(
            (0..3).any(|k| x_lp[k] > 0.0 && x_lp[k] < 1.0),
            "expected a fractional vertex, got {:?}",
            x_lp
        );
        // Extract tableau rows for fractional basic original binaries (the
        // node-LP extraction convention).
        let mut tableau: Vec<iconic_simplex::TableauRow<f64>> = Vec::new();
        for r in 0..1 {
            let bc = solver.basic_col(r);
            if bc >= 3 || vt[bc] != VarType::Binary {
                continue;
            }
            let xb: f64 = solver.basic_val(r);
            let f = xb - xb.floor();
            if f > 1e-7 && 1.0 - f > 1e-7 {
                tableau.push(solver.tableau_row(r));
            }
        }
        assert!(
            !tableau.is_empty(),
            "no fractional basic binary in the tableau"
        );
        let mut pool: CutPool<f64> = CutPool::new(10, &[], &[]);
        generate_intersection_cuts(&x_lp, &tableau, 3, &a, &b, &lb, &vt, &[], &mut pool);
        assert!(!pool.cuts.is_empty(), "the intersection cut must fire");
        let integer_points = vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        assert_cuts_sound_and_violated(&pool, &integer_points, &x_lp);
    }

    /// Lift-and-project: the cut-generating LP over a fractional binary's
    /// {0,1} disjunction emits a cut valid for the integer hull (exhaustive
    /// over all 4 integer points) that cuts off the fractional vertex.
    ///
    /// Instance: `P = {x0 + x1 ≤ 1, x0 + x2 ≤ 1, x1 + x2 ≤ 1}`, binaries.
    /// The integer hull is the tetrahedron {x ≥ 0, x0 + x1 + x2 ≤ 1} (its 4
    /// extreme points), while the LP vertex (0.5, 0.5, 0.5) lies outside it
    /// — the disjunctive cut x0 + x1 + x2 ≤ 1 separates it and is expressible
    /// in the separation LP (side 0 uses the x1 + x2 ≤ 1 row, side 1 the
    /// degree-style combination), so the family must fire.
    #[test]
    fn lift_and_project_cut_is_sound_and_cuts_the_fractional_point() {
        let a =
            DenseMatrix::from_row_major(3, 3, vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let b = vec![1.0; 3];
        let eq_mask = vec![false; 3];
        let lb = vec![0.0; 3];
        let ub = vec![1.0; 3];
        let vt = vec![VarType::Binary; 3];
        let x_star = vec![0.5, 0.5, 0.5];
        let mut pool: CutPool<f64> = CutPool::new(10, &[], &[]);
        generate_lift_and_project_cuts(&x_star, &a, &b, &eq_mask, &lb, &ub, &vt, &[], 3, &mut pool);
        assert!(
            !pool.cuts.is_empty(),
            "a violated disjunctive cut must exist"
        );
        let integer_points = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert_cuts_sound_and_violated(&pool, &integer_points, &x_star);
    }

    /// Lift-and-project negative case: when the fractional point lies in the
    /// disjunctive hull of every variable (a convex combination of one
    /// x_j = 0 point and one x_j = 1 point exists for every j), no cut can
    /// exist and the family must stay silent.
    ///
    /// Instance: `x0 + x1 + x2 + x3 ≥ 2`, binaries, x* = (0.5, 0.5, 0.5,
    /// 0.5) = ½·(0, 2/3, 2/3, 2/3) + ½·(1, 1/3, 1/3, 1/3), and both points
    /// are feasible for the respective sides of every variable's split.
    #[test]
    fn lift_and_project_stays_silent_when_the_point_is_in_the_disjunctive_hull() {
        let a = DenseMatrix::from_row_major(1, 4, vec![-1.0, -1.0, -1.0, -1.0]);
        let b = vec![-2.0];
        let eq_mask = vec![false];
        let lb = vec![0.0; 4];
        let ub = vec![1.0; 4];
        let vt = vec![VarType::Binary; 4];
        let x_star = vec![0.5, 0.5, 0.5, 0.5];
        let mut pool: CutPool<f64> = CutPool::new(10, &[], &[]);
        generate_lift_and_project_cuts(&x_star, &a, &b, &eq_mask, &lb, &ub, &vt, &[], 4, &mut pool);
        assert!(
            pool.cuts.is_empty(),
            "no disjunctive cut may exist for a point in the disjunctive hull"
        );
    }

    /// Soundness under cut-row slacks: the intersection-cut conversion must handle
    /// nonbasic slack columns of materialized cut rows exactly like original
    /// rows (the Gomory separator skips these; the disjunctive separator
    /// must not, or it would stop firing after the first cut round). The cut
    /// is checked exhaustively against the integer hull of a problem with a
    /// cut row already active.
    #[test]
    fn intersection_converts_cut_row_slacks_soundly() {
        // Original row: x0 + x1 + x2 ≤ 1 (≤). Cut row (already materialized):
        // x0 − x2 ≤ 0 (valid below). Integer points of the system
        // {x0 + x1 + x2 ≤ 1, x0 ≤ x2, binaries}: (0,0,0), (0,1,0), (0,0,1) —
        // the cut x0 ≤ 0 is a facet of their hull.
        let a = DenseMatrix::from_row_major(1, 3, vec![1.0, 1.0, 1.0]);
        let b = vec![1.0];
        let lb = vec![0.0; 3];
        let vt = vec![VarType::Binary; 3];
        let cut_rows: Vec<std::sync::Arc<(Vec<f64>, f64)>> =
            vec![std::sync::Arc::new((vec![1.0, 0.0, -1.0], 0.0))];
        // Fractional point (0.5, 0, 0.5) makes both rows tight; with x0 and
        // x2 basic and x1, s0 (row-0 slack), s1 (cut-row slack) nonbasic at
        // their lower bounds, the row of x0 is
        //   x0 + 0.5·x1 + 0.5·s0 + 0.5·s1 = 0.5
        // (from 2x0 = 1 − x1 − s0 − s1), whose nonbasic slack coefficients
        // exercise the cut-row slack conversion.
        let x_star = vec![0.5, 0.0, 0.5];
        let tableau = vec![iconic_simplex::TableauRow {
            basic_col: 0,
            basic_val: 0.5,
            coeffs: vec![
                (1, 0.5, false, 0.0),
                (3, 0.5, false, 0.0),
                (4, 0.5, false, 0.0),
            ],
        }];
        let mut pool: CutPool<f64> = CutPool::new(10, &[], &[]);
        generate_intersection_cuts(&x_star, &tableau, 3, &a, &b, &lb, &vt, &cut_rows, &mut pool);
        let integer_points = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        for cut in &pool.cuts {
            for p in &integer_points {
                let lhs: f64 = cut.row.iter().zip(p.iter()).map(|(c, &v)| c * v).sum();
                assert!(
                    lhs <= cut.rhs + 1e-6,
                    "cut must be valid: row·p = {lhs} > {}",
                    cut.rhs
                );
            }
            let viol: f64 = cut
                .row
                .iter()
                .zip(x_star.iter())
                .map(|(c, &v)| c * v)
                .sum::<f64>()
                - cut.rhs;
            assert!(
                viol > 1e-5,
                "the cut must cut off the fractional point: viol = {viol}"
            );
        }
        assert!(
            !pool.cuts.is_empty(),
            "the separator must fire with cut-row slacks present"
        );
    }
}

#[cfg(test)]
mod audits {
    use super::*;

    // ── Mixed-row MIR soundness audit ─────────────────────────────────
    // Brute force: for random small mixed MIPs, enumerate every integer
    // assignment in the box, and for each verify via an LP over the
    // continuous variables that no feasible point violates any generated
    // MIR cut. Same methodology as the F1/F2 audits.
    /// Append `x_j <= ub_j` / `-x_j <= -lb_j` rows for every finite bound of
    /// the continuous columns, as dense rows on `a` with RHS pushed onto `b`.
    /// Both audit LP builders below need exactly this.
    fn push_finite_bound_rows(
        a: &mut DenseMatrix<f64>,
        b: &mut Vec<f64>,
        lb: &[f64],
        ub: &[f64],
        cols: std::ops::Range<usize>,
    ) {
        for j in cols {
            if ub[j] < crate::INF_BOUND {
                let r = b.len();
                b.push(ub[j]);
                a.set(r, j, 1.0);
            }
            if lb[j] > -crate::INF_BOUND {
                let r = b.len();
                b.push(-lb[j]);
                a.set(r, j, -1.0);
            }
        }
    }
    /// Max of `Σ ψ_c y_c` over the continuous polyhedron of a single row.
    /// Returns +inf if unbounded (audit fails loudly rather than silently).
    fn max_cont_lhs(
        row_coeffs: &[(usize, f64)],
        cut_coeffs: &[(usize, f64)],
        b_rem: f64,
        lb: &[f64],
        ub: &[f64],
        n: usize,
    ) -> f64 {
        // solve max Σ ψ y  s.t.  Σ a y ≤ b_rem, l ≤ y ≤ u  via iconic LP
        use iconic_api::{solve as api_solve, ConeProgram};
        use iconic_core::{Cone, Settings};
        let mut a = DenseMatrix::zeros(1, n);
        let mut q = vec![0.0; n];
        let mut has = false;
        for &(c, v) in row_coeffs {
            a.set(0, c, v);
            has = true;
        }
        if !has {
            return f64::NEG_INFINITY;
        }
        for &(c, v) in cut_coeffs {
            q[c] = -v;
        } // minimize −ψy
        // bounds as nonneg rows: y_j ≤ u_j → row +y_j ≤ u_j ; y_j ≥ l_j → −y_j ≤ −l_j
        let mut a2 = DenseMatrix::zeros(1 + 2 * n, n);
        for j in 0..n {
            a2.set(0, j, a.get(0, j));
        }
        let mut b2 = vec![b_rem];
        push_finite_bound_rows(&mut a2, &mut b2, lb, ub, 0..n);
        let cones = vec![Cone::NonNegative(b2.len())];
        let prog = ConeProgram {
            p: DenseMatrix::zeros(n, n),
            q,
            a: a2,
            a_csc: None,
            b: b2,
            cones,
        };
        let mut st = Settings::default();
        st.presolve = false;
        match api_solve(&prog, &st) {
            Ok(sol) => {
                if sol.status == iconic_core::Status::PrimalInfeasible {
                    // No feasible point for this integer assignment: the cut is
                    // not obligated to hold (there is nothing to exclude).
                    return f64::NEG_INFINITY;
                }
                // Verify the returned point actually satisfies the row to a
                // real tolerance before trusting the max (an infeasible LP can
                // come back "Optimal" with a garbage iterate).
                let mut row_res = 0.0;
                for &(c, v) in row_coeffs {
                    row_res += v * sol.x[c];
                }
                if row_res > b_rem + 1e-5 {
                    return f64::NEG_INFINITY;
                }
                let mut best = 0.0;
                for &(c, v) in cut_coeffs {
                    best -= v * sol.x[c];
                }
                if sol.status == iconic_core::Status::DualInfeasible {
                    return f64::INFINITY;
                }
                best
            }
            Err(_) => f64::INFINITY,
        }
    }

    #[test]
    fn mixed_mir_cuts_never_exclude_a_feasible_point() {
        use iconic_core::rng::Lcg;
        let mut rng = Lcg::new(0x5eed);
        let (n_int, n_cont, trials) = (3usize, 3usize, 400usize);
        let n = n_int + n_cont;
        let mut total_cuts = 0usize;
        for _ in 0..trials {
            let lb = vec![0.0; n];
            let mut ub = vec![1.0; n];
            for j in 0..n_int {
                ub[j] = (1 + (rng.next_u64() % 3)) as f64;
            }
            // Tighter continuous bounds so the MIR cuts actually bite.
            for j in n_int..n {
                ub[j] = 0.5 + 1.5 * rng.unit();
            }
            let mut a = DenseMatrix::zeros(1, n);
            let mut b = vec![0.0];
            let mut var_types = vec![VarType::Binary; n_int];
            var_types.extend(std::iter::repeat_n(VarType::Continuous, n_cont));
            // random row: integer coeffs in [-3,3], continuous in [-2,2]
            for j in 0..n_int {
                a.set(0, j, (rng.next_u64() % 7) as f64 - 3.0);
            }
            for j in n_int..n {
                a.set(0, j, (rng.next_u64() % 5) as f64 - 2.0);
            }
            // RHS tight at the fractional separation point x = 0.3·ones, so
            // the row is active and cuts actually fire (the regression test must not be
            // vacuous).
            let mut ax = 0.0;
            for j in 0..n {
                ax += a.get(0, j) * 0.3;
            }
            b[0] = ax;
            // Only keep rows whose RHS fractional part is well inside (0,1) —
            // near-integral RHS produce no MIR cut.
            let fb0 = b[0] - b[0].floor();
            if !(0.05..=0.95).contains(&fb0) {
                continue;
            }
            // skip rows with negative-lb support (the MIR gate skips them too)
            if (0..n).any(|j| a.get(0, j).abs() > 1e-9 && lb[j] < -1e-9) {
                continue;
            }
            let mut pool = CutPool::<f64>::new(64, &vec![0.0; n], &var_types);
            for _sep in 0..3 {
                let x: Vec<f64> = (0..n).map(|_| 0.1 + 0.8 * rng.unit()).collect();
                generate_mir_cuts(&x, &a, &b, &lb, &ub, &var_types, &mut pool);
            }
            total_cuts += pool.cuts.len();
            for cut in pool.cuts {
                // enumerate integer assignments
                let mut idx = vec![0usize; n_int];
                loop {
                    let mut int_lhs = 0.0;
                    for (k, &v) in idx.iter().enumerate() {
                        int_lhs += a.get(0, k) * v as f64;
                    }
                    let mut cont_row: Vec<(usize, f64)> = vec![];
                    let mut cont_cut: Vec<(usize, f64)> = vec![];
                    for j in n_int..n {
                        let aij = a.get(0, j);
                        let cj = cut.row[j];
                        if aij.abs() > 1e-9 {
                            cont_row.push((j, aij));
                        }
                        if cj.abs() > 1e-9 {
                            cont_cut.push((j, cj));
                        }
                    }
                    let b_rem = b[0] - int_lhs;
                    let maxc = max_cont_lhs(&cont_row, &cont_cut, b_rem, &lb, &ub, n);
                    let mut int_cut = 0.0;
                    for (k, &v) in idx.iter().enumerate() {
                        int_cut += cut.row[k] * v as f64;
                    }
                    let total = int_cut + maxc;
                    assert!(
                        total <= cut.rhs + 1e-5,
                        "MIR cut excludes a feasible point: row {:?} b={} cut rhs={} at int {:?}: max LHS {:.6} > rhs",
                        (0..n).map(|j| a.get(0, j)).collect::<Vec<_>>(), b[0], cut.rhs, idx, total
                    );
                    // next assignment
                    let mut k = 0;
                    while k < n_int {
                        idx[k] += 1;
                        if idx[k] <= ub[k] as usize {
                            break;
                        }
                        idx[k] = 0;
                        k += 1;
                    }
                    if k == n_int {
                        break;
                    }
                }
            }
        }
        eprintln!("[audit] {trials} trials, {total_cuts} MIR cuts audited");
        assert!(
            total_cuts > 0,
            "audit vacuous: no MIR cut fired in {trials} trials"
        );
    }

    // ── Gomory tableau-cut soundness audit ─────────────────────────────
    // For random small mixed MIPs: solve the LP relaxation with the dual
    // simplex, extract the fractional tableau rows, generate the Gomory cuts,
    // then verify each cut against the WHOLE LP relaxation by maximizing its
    // violation over the polyhedron (a cut is valid iff that max ≤ 0).

    /// Max of (cut·x − rhs) over the continuous variables with the integer
    /// part fixed at `int_assign` (0 = infeasible assignment → NEG_INFINITY,
    /// nothing to check). A Gomory cut MAY cut off fractional LP points —
    /// its validity is over the MIP's integer hull — so the validity check must fix
    /// the integers and maximize over the continuous part only.
    fn max_cut_violation(
        row: &[f64],
        rhs: f64,
        a: &DenseMatrix<f64>,
        b: &[f64],
        lb: &[f64],
        ub: &[f64],
        int_assign: &[usize],
        n_int: usize,
    ) -> f64 {
        use iconic_api::{solve as api_solve, ConeProgram};
        use iconic_core::{Cone, Settings};
        let (m, n) = (a.nrows, a.ncols);
        // original rows + bound rows; integer part fixed into the RHS
        let mut a3 = DenseMatrix::zeros(m + 2 * n, n);
        for i in 0..m {
            for j in 0..n {
                a3.set(i, j, a.get(i, j));
            }
        }
        let mut b3: Vec<f64> = b
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let mut r = v;
                for k in 0..n_int {
                    r -= a.get(i, k) * int_assign[k] as f64;
                }
                r
            })
            .collect();
        push_finite_bound_rows(&mut a3, &mut b3, lb, ub, n_int..n);
        let cones = vec![Cone::NonNegative(b3.len())];
        let mut q = vec![0.0; n];
        for j in 0..n {
            q[j] = -row[j];
        } // maximize row·x − rhs → minimize −row·x
        let prog = ConeProgram {
            p: DenseMatrix::zeros(n, n),
            q,
            a: a3,
            a_csc: None,
            b: b3,
            cones,
        };
        let mut st = Settings::default();
        st.presolve = false;
        match api_solve(&prog, &st) {
            Ok(sol) => {
                if sol.status == iconic_core::Status::DualInfeasible {
                    return f64::INFINITY;
                }
                let mut act = 0.0;
                for j in n_int..n {
                    act += row[j] * sol.x[j];
                }
                for k in 0..n_int {
                    act += row[k] * int_assign[k] as f64;
                }
                act - rhs
            }
            Err(_) => f64::INFINITY,
        }
    }

    #[test]
    fn gomory_cuts_are_valid_for_the_lp_relaxation() {
        use iconic_simplex::{CscCols, DualSolver};
        let mut state = iconic_core::rng::Lcg::new(0x600d);
        let (n_int, n_cont, trials) = (3usize, 2usize, 60usize);
        let n = n_int + n_cont;
        let mut audited = 0usize;
        for t in 0..trials {
            let m = 2 + state.pick(3);
            let lb = vec![0.0; n];
            let mut ub = vec![1.0; n];
            for j in 0..n_int {
                ub[j] = 1.0 + (state.pick(3) as f64);
            }
            for j in n_int..n {
                ub[j] = 0.5 + 1.5 * state.unit();
            }
            let mut a = DenseMatrix::zeros(m, n);
            let mut b = vec![0.0; m];
            let mut var_types = vec![VarType::Binary; n_int];
            var_types.extend(std::iter::repeat_n(VarType::Continuous, n_cont));
            for i in 0..m {
                for j in 0..n {
                    let v = if j < n_int {
                        (state.pick(7) as f64) - 3.0
                    } else {
                        (state.pick(5) as f64) - 2.0
                    };
                    a.set(i, j, v);
                }
                b[i] = (state.pick(20) as f64) / 5.0 - 1.0;
            }
            // Build the simplex-form LP: conic rows (all nonneg → slack cols) + bounds.
            let n_lp = n + m;
            let mut col_start = vec![0usize; n_lp + 1];
            let mut row_idx = Vec::new();
            let mut val = Vec::new();
            for j in 0..n {
                for i in 0..m {
                    let v = a.get(i, j);
                    if v.abs() > 1e-12 {
                        row_idx.push(i);
                        val.push(v);
                    }
                }
                col_start[j + 1] = row_idx.len();
            }
            for r in 0..m {
                row_idx.push(r);
                val.push(1.0);
                col_start[n + r + 1] = row_idx.len();
            }
            let mut l = vec![0.0; n_lp];
            let mut u = vec![1e20; n_lp];
            for j in 0..n {
                l[j] = lb[j];
                u[j] = ub[j];
            }
            let c = vec![0.0; n_lp];
            let b_lp = b.clone();
            let mut solver: DualSolver<f64> = DualSolver::from_csc(
                c.clone(),
                CscCols {
                    col_start: col_start.clone(),
                    row_idx: row_idx.clone(),
                    val: val.clone(),
                },
                b_lp.clone(),
                l.clone(),
                u.clone(),
                m,
                n_lp,
            );
            let sol = solver.cold_solve();
            if sol.status != iconic_simplex::Status::Optimal {
                continue;
            }
            // Extract fractional basic integer tableau rows
            let mut tableau = Vec::new();
            for r in 0..m {
                let bc = solver.basic_col(r);
                if bc >= n || !var_types[bc].is_integer() {
                    continue;
                }
                let xb = solver.basic_val(r);
                let f = xb - xb.floor();
                if f > 1e-7 && 1.0 - f > 1e-7 {
                    tableau.push(solver.tableau_row(r));
                }
            }
            if tableau.is_empty() {
                continue;
            }
            let x_lp: Vec<f64> = sol.x[..n].to_vec();
            let mut pool = CutPool::<f64>::new(64, &vec![0.0; n], &var_types);
            generate_gomory_cuts(&x_lp, &tableau, n, m, &a, &b, &lb, &var_types, &mut pool);
            for cut in pool.cuts {
                // Enumerate every integer assignment; per assignment, the
                // continuous max of (cut·x − rhs) must be ≤ 0 (or the
                // assignment is infeasible, in which case there is nothing
                // to exclude).
                let mut idx = vec![0usize; n_int];
                let mut invalid: Option<(Vec<usize>, f64)> = None;
                'outer: loop {
                    let viol = max_cut_violation(&cut.row, cut.rhs, &a, &b, &lb, &ub, &idx, n_int);
                    if viol.is_finite() && viol > 1e-5 {
                        invalid = Some((idx.clone(), viol));
                        break 'outer;
                    }
                    let mut k = 0;
                    while k < n_int {
                        idx[k] += 1;
                        if idx[k] <= ub[k] as usize {
                            break;
                        }
                        idx[k] = 0;
                        k += 1;
                    }
                    if k == n_int {
                        break;
                    }
                }
                assert!(
                    invalid.is_none(),
                    "Gomory cut excludes an integer-feasible point (trial {t}): assign {:?} max violation {:.6}, row={:?} rhs={}",
                    invalid.as_ref().unwrap().0, invalid.as_ref().unwrap().1, cut.row, cut.rhs
                );
                audited += 1;
            }
        }
        eprintln!("[gomory-audit] {trials} trials, {audited} cuts audited");
        assert!(audited > 0, "audit vacuous: no Gomory cut generated");
    }

    #[test]
    fn cut_weights_presets() {
        let d = CutWeights::default();
        assert_eq!((d.dcd, d.eff, d.isp, d.obp), (0.0, 1.0, 0.1, 0.1));
        let a = CutWeights::aggressive();
        assert_eq!((a.dcd, a.eff, a.isp, a.obp), (0.3, 0.2, 0.25, 0.25));
    }

    /// The greedy node selector must rank by the weighted score (dcd =
    /// progress toward the incumbent) and respect both the count limit and
    /// the nonzero budget. Cuts c1/c2 point along the incumbent direction
    /// (high dcd); c0 does not (dcd = 0). With the aggressive weights the
    /// dcd term decides: c1 and c2 are chosen before c0.
    #[test]
    fn select_for_node_ranks_by_dcd_and_respects_limits() {
        let n = 3;
        let vt = vec![VarType::Binary; n];
        let mut pool = CutPool::new(50, &[], &vt);
        // x_lp = (1,1,1); incumbent x̂ = (1,0,0) → direction (0,-1,-1)/√2.
        let x_lp = vec![1.0, 1.0, 1.0];
        let inc = vec![1.0, 0.0, 0.0];
        // Three pairwise-orthogonal cuts, same rhs, same stored violation:
        //   c0: x0 <= 0.5  → alpha·y = 0            → dcd = 0
        //   c1: x1 <= 0.5  → |alpha·y| = 1/√2       → dcd = 0.5/0.7071 ≈ 0.707
        //   c2: x2 <= 0.5  → same as c1
        for row in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let viol: f64 = row.iter().zip(&x_lp).map(|(&a, &b)| a * b).sum::<f64>() - 0.5;
            assert!(pool.add(Cut::new(row.to_vec(), 0.5, viol)));
        }
        pool.set_weights(CutWeights::aggressive());
        let sel = pool.select_for_node(0, 2, 1000, &x_lp, Some(&inc));
        assert_eq!(sel.len(), 2);
        assert!(
            sel.contains(&1) && sel.contains(&2),
            "the incumbent-direction cuts must outrank c0, got {sel:?}"
        );
        // Count limit: only the single best cut fits.
        let sel1 = pool.select_for_node(0, 1, 1000, &x_lp, Some(&inc));
        assert_eq!(sel1.len(), 1);
        assert!(sel1[0] == 1 || sel1[0] == 2);
        // Nonzero budget: each cut has 1 nonzero; budget 1 admits one cut.
        let selb = pool.select_for_node(0, 10, 1, &x_lp, Some(&inc));
        assert_eq!(selb.len(), 1);
        // No incumbent: dcd = 0 for all, all three cuts qualify (tie order
        // is not specified).
        let seln = pool.select_for_node(0, 10, 1000, &x_lp, None);
        let mut seln_sorted = seln.clone();
        seln_sorted.sort_unstable();
        assert_eq!(seln_sorted, vec![0, 1, 2]);
        // Older cuts (before prev_count) are not eligible.
        let sel_prev = pool.select_for_node(2, 10, 1000, &x_lp, Some(&inc));
        assert_eq!(sel_prev, vec![2]);
    }

    /// The surrogate cover generator filters rows by "positive dual". Node
    /// LPs are minimizations (Ax <= b), so the dual simplex pi is
    /// non-positive on every row — the old `pi > eps` filter skipped all of
    /// them and the surrogate never fired on the main node LPs. The
    /// convention-agnostic |pi| weight fires on both conventions.
    /// Regression: with all-negative duals the surrogate must contribute
    /// rows and produce covers (the multiknap root case).
    #[test]
    fn surrogate_covers_fire_with_negative_duals() {
        // Two capacity rows of a multiple knapsack: items 0,1 fit one per
        // bin (weights 10 each, cap 15). Partition rows first.
        let n = 4;
        let m = 4;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        // partition rows
        a.set(0, 0, 1.0);
        a.set(0, 2, 1.0);
        a.set(1, 1, 1.0);
        a.set(1, 3, 1.0);
        // capacity rows
        a.set(2, 0, 10.0);
        a.set(2, 1, 10.0);
        a.set(3, 2, 10.0);
        a.set(3, 3, 10.0);
        let b = vec![1.0, 1.0, 15.0, 15.0];
        // Minimization-form duals: non-positive, tight rows negative.
        let dual_pi = vec![-0.5, -0.5, -1.0, -1.0];
        let ub = vec![1.0; 4];
        let vt = vec![VarType::Binary; 4];
        let x = vec![1.0, 1.0, 0.5, 0.5];
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_surrogate_cover_cuts(&x, &dual_pi, &a, &b, &ub, &vt, &mut pool);
        assert!(
            !pool.cuts.is_empty(),
            "negative duals (the minimization convention) must drive the surrogate"
        );
        for cut in &pool.cuts {
            // Soundness on the surrogate row: any 0/1 point with three
            // variables selected violates the surrogate row itself
            // (3 x 10.5 > rhs 31), so a cover cut may never allow three.
            for mask in 0u32..16 {
                let xx: Vec<f64> = (0..4).map(|j| ((mask >> j) & 1) as f64).collect();
                let row_lhs: f64 = (0..4).map(|j| 10.5 * xx[j]).sum();
                if row_lhs <= 31.0 + 1e-9 {
                    let lhs: f64 = (0..4).map(|j| cut.row[j] * xx[j]).sum();
                    assert!(
                        lhs <= cut.rhs + 1e-9,
                        "cut excludes a surrogate-feasible point {:?}",
                        xx
                    );
                }
            }
        }
    }

    /// Item-level aggregate covers: variables with equal surrogate weight
    /// are one item's copies; the cover is found over the GROUP level and
    /// materialized over every copy. Regression: the multiknap_i25_b4 root
    /// LP optimum does not move even with all per-bin covers (the LP slides
    /// a split item's fraction between bins), while the item-level cover
    /// "at most |C|-1 of these items" cuts it — this test pins the
    /// construction (all-copies support, rhs |C|-1) that the per-bin
    /// pipeline cannot produce.
    #[test]
    fn aggregate_surrogate_cover_materializes_all_copies() {
        // 4 variables = 2 items x 2 bins; the copies of each item share
        // their surrogate weight (10.5 and 9.0). Item 0 at 1.0, item 1
        // split 0.5/0.5.
        let x = vec![1.0, 0.5, 0.0, 0.5];
        let w = vec![10.5, 10.5, 9.0, 9.0];
        let rhs = 15.0;
        let ub = vec![1.0; 4];
        let vt = vec![VarType::Binary; 4];
        let mut pool: CutPool<f64> = CutPool::new(100, &[], &[]);
        generate_aggregate_surrogate_covers(&x, &w, rhs, &ub, &vt, &mut pool);
        assert!(!pool.cuts.is_empty(), "must find the item-level cover");
        let cut = &pool.cuts[0];
        assert_eq!(cut.rhs, 1.0, "rhs must be |C|-1 for a 2-item cover");
        for j in 0..4 {
            assert!(
                cut.row[j] > 0.0,
                "every copy of every covered item must be in the cut"
            );
        }
        let viol: f64 = (0..4).map(|j| cut.row[j] * x[j]).sum::<f64>() - cut.rhs;
        assert!(viol > 1e-6, "the cover must be violated at the point");
    }
}
