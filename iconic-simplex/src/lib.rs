//! Proper bounded dual simplex with hot-start for MIP node LPs, built for the
//! numerical demands of branch-and-bound: product-form-of-the-inverse (PFI)
//! basis updates, periodic refactorization, iterative refinement of the
//! triangular solves, a dual-feasibility crash at cold-start, and Harris-style
//! ratio testing with a bounded-degradation recovery path.
use iconic_core::Scalar;
use std::fmt::Debug;

mod sparse_lu;
use sparse_lu::SparseLu;

/// Cap on the product-form-of-inverse eta chain length before a full refactor.
/// Bounds both the per-solve eta-application cost (O(MAX_ETAS * m) worst case)
/// and numerical error accumulation from the growing product of rank-1 updates.
///
/// With unconditional LU reuse in hot_solve, the inherited factor is at most
/// MAX_ETAS updates old. A value of 100 keeps the inherited LU fresh enough
/// for reliable pivoting while allowing enough updates between refactorizations
/// to amortize the O(m³) factorization cost.
const MAX_ETAS: usize = 100;

/// Absolute pivot floor shared by the LU factorization's repair pass and the
/// backward triangular solve's singular-pivot detection.
///
/// `factor_once` records any pivot below this as deficient and repairs the
/// basis (substituting the dependent column's logical unit column, whose
/// pivot is then exactly 1), so a pivot below this threshold should never
/// occur in a healthy basis. When `bwd_solve_dense` still meets one, the
/// basis is genuinely singular, and it must surface that as a non-finite
/// sentinel -- which the solve's NaN guards turn into a refactor /
/// IterationLimit -- rather than silently clamping to 0.0, which yields a
/// finite-but-wrong solution that evades every guard.
const PIVOT_TOL: f64 = 1e-20;

/// Dense-factor size gate for the LAPACK (`dgetrf`) mirror: the m² f64 copy
/// doubles the basis-matrix footprint, so beyond this the hand-rolled path
/// (and its single m² `lu` array) stays in effect. Node LPs — the hot path —
/// stay far below the gate.
const BLAS_MAX_M: usize = 4096;

/// Minimum dimension for the sparse Markowitz LU path. Below this the packed
/// sparse representation's setup overhead is not worth avoiding the dense
/// loops that are already fast at small `m` (and the BLAS mirror covers them).
const SPARSE_LU_MIN_M: usize = 96;

/// A/B knob: `ICONIC_NO_SPARSE_LU` routes every factorization through the
/// dense/BLAS paths (the sparse-vs-dense equivalence tests toggle the same
/// switch at runtime via [`set_sparse_lu_forced`]).
static SPARSE_LU_FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn sparse_lu_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    SPARSE_LU_FORCED.load(std::sync::atomic::Ordering::Relaxed)
        && *ENV.get_or_init(|| std::env::var_os("ICONIC_NO_SPARSE_LU").is_none())
}

/// Test/A-B override for [`sparse_lu_enabled`] (the env var is read once).
#[doc(hidden)]
pub fn set_sparse_lu_forced(on: bool) {
    SPARSE_LU_FORCED.store(on, std::sync::atomic::Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Infeasible,
    IterationLimit,
}

#[derive(Clone, Debug)]
pub struct Solution<T: Scalar> {
    pub status: Status,
    pub x: Vec<T>,
    pub obj: T,
    pub iters: usize,
    pub pi: Vec<T>,
}

/// Why the last solve returned a non-Optimal verdict, keyed by return site.
/// The MIP layer sees IterationLimit at 86 pivots against a 200k cap and
/// cannot tell from the outside whether that is cap exhaustion (raise it), a
/// dead basis (numerics), or a Phase-1 breakdown (algorithm) -- each has a
/// different fix. Set at every early-return site in `dual_loop`; read with
/// [`DualSolver::take_fail_reason`]. Diagnostic-only; zero cost when unread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FailReason {
    None,
    Deadline,
    DeadBasisXb,
    Phase1NoEntering,
    Phase1NoLeaving,
    RatioBreakdown,
    SingularPi,
    CapExhausted,
}

thread_local! {
    static LAST_FAIL_REASON: std::cell::Cell<FailReason> =
        const { std::cell::Cell::new(FailReason::None) };
}

fn set_fail(r: FailReason) {
    LAST_FAIL_REASON.with(|c| c.set(r));
}


/// Read and clear the last non-Optimal verdict's reason.
pub fn take_fail_reason() -> FailReason {
    LAST_FAIL_REASON.with(|c| c.replace(FailReason::None))
}

/// A row of the simplex tableau (see [`DualSolver::tableau_row`]).
#[derive(Clone, Debug)]
pub struct TableauRow<T: Scalar> {
    /// Column index of the basic variable.
    pub basic_col: usize,
    /// Current value of the basic variable.
    pub basic_val: T,
    /// `(nonbasic column, coefficient, at_upper, upper_bound)` tuples. The
    /// bound status lets the Gomory separator complement at-upper variables
    /// (`x_j = u_j − x'_j`) for the strengthened cut.
    pub coeffs: Vec<(usize, T, bool, T)>,
}

#[derive(Clone, Debug)]
pub struct HotBasis<T: Scalar> {
    pub basic: Vec<usize>,
    pub at_upper: Vec<bool>,
    pub lu: Vec<T>,
    pub perm: Vec<usize>,
    /// The row-swap sequence of the factorization: `swaps[k]` is the partner
    /// of elimination step `k` (or `usize::MAX` when step `k` pivoted in
    /// place). This is exactly LAPACK's `ipiv` in 0-based form, so a child
    /// solver inheriting the factor can reconstruct the LAPACK pivot
    /// sequence (`ipiv[k] = swaps[k] + 1` in 1-based) without re-factoring —
    /// the BLAS warm-start path's requirement. The hand-rolled triangular
    /// solves only ever need the final `perm`, so this stays cheap to carry
    /// (m `usize`).
    pub swaps: Vec<usize>,
    pub n_etas: usize,
    pub eta_p: Vec<usize>,
    pub eta_alpha: Vec<T>,
    /// Sparse eta mirror (see `DualSolver::eta_idx`/`eta_val`/`eta_start`):
    /// the same chain stored as per-eta nonzero lists, so a child inheriting
    /// this basis can apply etas at O(nnz) instead of O(m) each.
    pub eta_start: Vec<usize>,
    pub eta_idx: Vec<u32>,
    pub eta_val: Vec<T>,
    /// Packed Markowitz LU when the parent's base factor came from the
    /// sparse path. While `Some`, `lu` holds stale data (the sparse path
    /// does not maintain the dense array), so a child inheriting this basis
    /// must route its base solves through this factor (which `hot_solve`
    /// does) rather than rebuilding the LAPACK mirror.
    pub sparse: Option<SparseLu<T>>,
}

/// LAPACK `dgetrf` mirror state for the BLAS-accelerated dense-LU path.
/// Holds the factored matrix in LAPACK's convention (row-major, unit-lower
/// `L` below the diagonal in the pivoted rows, `U` in the pivot rows), the
/// 1-based pivot permutation, and the RHS conversion scratch. `singular`
/// records any |U_kk| <= PIVOT_TOL at factorization time: the hand-rolled
/// backward solve emits its NaN sentinel on such a pivot and the caller's
/// guards turn it into a refactor / honest failure — LAPACK `dgetrs` would
/// divide silently, so the BLAS path defers to the hand path on a singular
/// factor.
struct BlasLu {
    lu: Vec<f64>,
    rhs: Vec<f64>,
    ipiv: Vec<i32>,
    singular: bool,
}

/// A matrix in compressed-sparse-column form: column `j` owns
/// `row_idx[col_start[j]..col_start[j + 1]]` with the matching values in `val`.
///
/// This is the form the solver works in throughout, so callers that can build it
/// directly should, rather than handing over a dense array to be converted.
#[derive(Clone, Debug)]
pub struct CscCols<T> {
    pub col_start: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub val: Vec<T>,
}

pub struct DualSolver<T: Scalar> {
    n: usize,
    m: usize,
    c: Vec<T>,
    b: Vec<T>,
    l: Vec<T>,
    u: Vec<T>,
    // Sparse column storage for A (CSC format): col j's nonzeros are at
    // a_row_idx[a_col_start[j]..a_col_start[j+1]] with values in a_val.
    // Slack columns (indices n-m..n) are identity vectors — stored as
    // a single nonzero at their slack row.
    a_col_start: Vec<usize>,
    a_row_idx: Vec<usize>,
    a_val: Vec<T>,
    // Per column: its first row index when that column's rows are *contiguous*
    // (`row_idx` runs base, base+1, ...), else `usize::MAX`.
    //
    // The pricing loop is the solver's hottest, and reading a column through CSC costs
    // an index load and a gather per entry. A contiguous column needs neither: row `k`
    // of the column is row `base + k`, so `pi`/`rho` can be walked as slices and the loop
    // vectorizes. This covers both ends of the density range that matter -- every slack
    // column (one entry) and every dense structural column (all m rows) is contiguous --
    // which is what a dense-storage fast path used to be for, without keeping a dense
    // `m * n` copy of A alive to serve it.
    a_col_contig: Vec<usize>,
    basis: Vec<usize>,
    at_upper: Vec<bool>,
    in_basis: Vec<bool>,
    lu: Vec<T>,
    perm: Vec<usize>,
    /// Row-swap sequence of the current factor (see [`HotBasis::swaps`]).
    swaps: Vec<usize>,
    /// LAPACK-form (dgetrf) mirror of the factor for the BLAS-accelerated
    /// path: `Some` when `T == f64`, BLAS is enabled at construction and
    /// `m` is under the dense-factor size gate. The basis matrix and every
    /// triangular solve then run through LAPACK `dgetrf`/`dgetrs`, which
    /// are 5-20x the hand-rolled loops on the dense m×m factor (the node-LP
    /// shape that matters — measured: the extended QKP node LPs spend
    /// ~80% of their solve time in dense triangular solves).
    blas: Option<BlasLu>,
    /// Sparse Markowitz LU of the current base factor ([`sparse_lu::SparseLu`]),
    /// `Some` whenever the last `factor_once` took the sparse path. While
    /// present it IS the base factor: the dense `lu` array holds stale data
    /// from an earlier factorization (not maintained on the sparse path), so
    /// every base solve dispatches through this factor first and the LAPACK
    /// mirror is neither rebuilt nor consulted. Dropped again by any
    /// factorization that falls back to the dense/BLAS paths.
    sparse_lu: Option<SparseLu<T>>,
    xb: Vec<T>,
    tol: T,
    max_iters: usize,
    iters: usize,
    deadline: Option<std::time::Instant>,
    /// Stall-triggered RHS perturbation: when the largest bound violation makes
    /// no progress for [`PERTURB_STALL_LIMIT`] iterations, the RHS is
    /// temporarily perturbed by ~2e-4 relative (deterministic pattern) to break
    /// the degeneracy, and restored as soon as the vertex is escaped — every
    /// verdict and returned point is computed on the exact RHS (see
    /// [`Self::restore_exact_rhs`]). Default on; disable for callers that
    /// cannot tolerate even a transiently different RHS.
    pub perturb_on_stall: bool,
    // Perturbation state (per solve, managed by dual_loop).
    perturb_active: bool,
    perturb_iters: usize,
    perturb_baseline: T,
    perturb_orig_b: Vec<T>,
    perturb_rng: u64,
    /// How many times the stall-triggered perturbation fired during this
    /// solve (test-visible).
    perturb_episodes: usize,
    /// Consecutive perturbation episodes that ended WITHOUT escaping their
    /// stall. Each failure escalates the next episode's magnitude and length;
    /// a successful escape resets it to zero. Capped at MAX_PERTURB_FAILURES,
    /// after which later stalls defer to the Bland latch instead of burning
    /// another episode.
    perturb_failures: usize,
    /// Magnitude multiplier for the NEXT perturbation episode: starts at 1,
    /// multiplied by the escalation factor on every failed episode, reset on a
    /// successful escape. A 10-pivot episode at the base ~2e-4 magnitude is a
    /// single sample of one random pattern; on genuinely stuck degenerate
    /// vertices that single sample often fails while a larger nudge succeeds —
    /// measured node LPs otherwise latched into Bland's rule for tens of
    /// thousands of smallest-index pivots after ONE failed episode.
    perturb_scale: f64,
    // Product-form-of-the-inverse (PFI) eta chain: the current basis inverse is
    // B_t^{-1} = E_t^{-1} ... E_1^{-1} B_0^{-1}, where B_0 = self.lu/self.perm (the
    // last full factor()) and each E_i is a rank-1 "eta" matrix (identity except
    // column eta_p[i] replaced by eta_alpha[i] = B_{i-1}^{-1} times the entering
    // column) recording one pivot. Reset (n_etas = 0) on every factor(). This
    // replaces a previous dense "Forrest-Tomlin bump" scheme that mutated self.lu
    // in place and required re-triangularizing column p using OTHER columns'
    // pivots without repivoting -- which fails deterministically (not just as a
    // numerical rarity) whenever the entering column's sparsity pattern doesn't
    // happen to touch row p, since a fixed elimination order can hit a zero pivot
    // even for a perfectly well-conditioned matrix. PFI has no such failure mode:
    // an eta is valid whenever alpha[p] (the SAME quantity the ratio test already
    // filters via alpha_tol before selecting the pivot) is above tolerance.
    n_etas: usize,
    eta_p: Vec<usize>,     // leaving slot for each eta, length MAX_ETAS
    eta_alpha: Vec<T>,     // flattened m-length alpha vectors, MAX_ETAS*m capacity (dense mirror)
    eta_idx: Vec<u32>,     // flattened nonzero indices per eta (sparse form)
    eta_val: Vec<T>,       // flattened nonzero values per eta (sparse form)
    eta_start: Vec<usize>, // offsets into eta_idx/eta_val, length MAX_ETAS+1

    // Pre-allocated LU scratch buffers (pre-allocated buffers)
    spike: Vec<T>,  // new column's raw entries / base-solve scratch (used by add_eta)
    rho_ep: Vec<T>, // ep vector for lu_solve_trans in ratio tests
    lu_z: Vec<T>,   // internal to lu_solve / lu_solve_trans
    lu_y: Vec<T>,   // internal to lu_solve / lu_solve_trans
    lu_pi: Vec<T>,  // internal to lu_solve_trans (result buffer)
    bt_rho: Vec<T>, // B^T*rho refinement residual scratch (dual_loop's iterative refinement)
    eta_scratch: Vec<T>, // transformed-cbt scratch for the transpose eta chain (lu_solve_trans)
    pi_buf: Vec<T>, // pre-allocated dual variables (length m), avoids per-iteration alloc
                    // Sparse L column storage (CSC format for strict lower triangular factor)
    devex_row: Vec<T>, // dual-Devex weights for basic variables, indexed by row
    devex_col: Vec<T>, // dual-Devex weights for nonbasic variables, indexed by column
    devex_enabled: bool, // A/B knob: ICONIC_NO_DEVEX freezes every weight at 1
}

fn devex_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ICONIC_NO_DEVEX").is_none())
}

impl<T: Scalar + PartialOrd + Debug> DualSolver<T> {
    /// Cap the pivot count for this solve, returning `IterationLimit` beyond it.
    ///
    /// Distinct from [`set_deadline`]: a pivot cap is deterministic and independent of
    /// machine load, which is what strong branching wants. Its probe LPs are cheap
    /// *estimates* feeding pseudo-costs, so truncating one costs an estimate rather than
    /// correctness -- whereas leaving them uncapped lets a single degenerate probe run
    /// for seconds.
    pub fn set_max_iters(&mut self, max_iters: usize) {
        self.max_iters = max_iters;
    }

    /// Stop pivoting once this instant passes, returning `IterationLimit`.
    ///
    /// The iteration cap alone is a safety net set high enough (8 million) to be no
    /// bound in practice. That suits a one-shot LP, but branch-and-bound solves one LP
    /// per node under a wall-clock budget and only tests that budget *between* nodes, so
    /// a single degenerate node can run for minutes with nothing able to interrupt it --
    /// measured at 1157s against a 3s limit on `facloc_f8_c20`, all of it in one node.
    ///
    /// A fixed pivot cap is the wrong instrument: sized to protect a pathological node it
    /// also truncates the root LP, which is solved once, drives every bound below it, and
    /// legitimately needs far more pivots. Bounding by time cuts only what actually
    /// overruns.
    pub fn set_deadline(&mut self, deadline: std::time::Instant) {
        self.deadline = Some(deadline);
    }

    pub fn new(c: &[T], a: &[T], b: &[T], l: &[T], u: &[T], m: usize, n: usize) -> Self {
        Self::from_owned(
            c.to_vec(),
            a.to_vec(),
            b.to_vec(),
            l.to_vec(),
            u.to_vec(),
            m,
            n,
        )
    }

    /// As [`new`], but takes ownership of the problem data instead of copying it.
    ///
    /// Callers that build the LP themselves -- branch-and-bound rebuilds one per node --
    /// already hold freshly allocated vectors, and the borrowing constructor copies all
    /// five again. The dense `a` is `m * n`, so that is the largest single allocation
    /// and memcpy in node setup, paid twice per node for nothing.
    ///
    /// Prefer [`from_csc`] where the caller can build column storage directly: the
    /// conversion below reads a row-major array down its columns, which is a cache miss
    /// per element, and the dense array it reads has to be materialized first.
    pub fn from_owned(
        c: Vec<T>,
        a: Vec<T>,
        b: Vec<T>,
        l: Vec<T>,
        u: Vec<T>,
        m: usize,
        n: usize,
    ) -> Self {
        let slack_start = n - m;
        // Build sparse column storage (CSC) for A, matching standard sparse-LU design.
        // Slack columns (n-m..n) are identity vectors: store as a single
        // nonzero at their slack row. Structural columns store actual nonzeros.
        let eps_nz = T::from_f64(1e-15).expect("scalar literal");
        let mut a_col_start = vec![0usize; n + 1];
        let mut a_row_idx: Vec<usize> = Vec::new();
        let mut a_val: Vec<T> = Vec::new();
        for j in 0..n {
            a_col_start[j] = a_row_idx.len();
            if j >= slack_start {
                // Slack column: identity vector, 1 nonzero at row (j - slack_start)
                a_row_idx.push(j - slack_start);
                a_val.push(T::one());
            } else {
                // Structural column: scan for nonzeros
                for i in 0..m {
                    let v = a[i * n + j];
                    if v.abs() > eps_nz {
                        a_row_idx.push(i);
                        a_val.push(v);
                    }
                }
            }
        }
        a_col_start[n] = a_row_idx.len();
        Self::from_csc(
            c,
            CscCols {
                col_start: a_col_start,
                row_idx: a_row_idx,
                val: a_val,
            },
            b,
            l,
            u,
            m,
            n,
        )
    }

    /// Build from column storage directly, which is the form the solver actually uses.
    ///
    /// The dense `m * n` array the other constructors take is never read after setup --
    /// every kernel in the solve goes through the CSC arrays. Materializing it costs an
    /// `m * n` allocation, an `m * n` zeroing and an `m * n` strided read per node LP,
    /// against a matrix whose nonzeros are typically a few per row. On sudoku_9x9
    /// (m = 354, n = 1083) that is 383k elements handled to deliver ~1.4k nonzeros, and
    /// branch-and-bound rebuilds it at every node and every strong-branching probe.
    ///
    /// Columns `n - m .. n` are the slack/artificial identity block and must be present
    /// in `a_col_start`/`a_row_idx`/`a_val` like any other column.
    pub fn from_csc(
        c: Vec<T>,
        a: CscCols<T>,
        b: Vec<T>,
        l: Vec<T>,
        u: Vec<T>,
        m: usize,
        n: usize,
    ) -> Self {
        let CscCols {
            col_start: a_col_start,
            row_idx: a_row_idx,
            val: a_val,
        } = a;
        let tol = T::from_f64(1e-14).expect("scalar literal");
        let mut in_basis = vec![false; n];
        let slack_start = n - m;
        for b in &mut in_basis[slack_start..n] {
            *b = true;
        }

        let a_col_contig: Vec<usize> = (0..n)
            .map(|j| {
                let (s, e) = (a_col_start[j], a_col_start[j + 1]);
                if s == e {
                    return usize::MAX;
                }
                let base = a_row_idx[s];
                if a_row_idx[s..e]
                    .iter()
                    .enumerate()
                    .all(|(k, &r)| r == base + k)
                {
                    base
                } else {
                    usize::MAX
                }
            })
            .collect();

        Self {
            n,
            m,
            c,
            b,
            l,
            u,
            a_col_start,
            a_row_idx,
            a_val,
            a_col_contig,
            basis: (slack_start..n).collect(),
            at_upper: vec![false; n],
            in_basis,
            // The dense factor arrays are LAZY: the packed Markowitz sparse
            // factor is the default base once m >= SPARSE_LU_MIN_M, and while
            // it is live neither dense array is ever read (`solve_base_into`
            // dispatches sparse -> BLAS mirror -> hand). Allocating two m^2
            // zeroed buffers per solver cost ~1ms per node LP at m~700 --
            // measurable across a 600-node tree -- for data the sparse path
            // leaves stale by contract (see `factor_once_sparse`). Every
            // dense WRITER ensures the size it needs first (`ensure_lu`, and
            // the mirror's own resize below); readers only run after a
            // writer ran, or behind a live sparse factor that never reaches
            // them.
            lu: Vec::new(),
            perm: (0..m).collect(),
            swaps: vec![usize::MAX; m],
            // The dense LAPACK mirror doubles the basis-matrix footprint
            // (m² f64), so it is gated to sizes where the dense factor is
            // the point of the exercise. Node LPs (the MIP hot path) stay
            // well under the gate.
            blas: if m <= BLAS_MAX_M
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
                && iconic_linalg::blas::blas_enabled()
            {
                Some(BlasLu {
                    lu: Vec::new(),
                    rhs: vec![0.0f64; m],
                    ipiv: vec![0i32; m],
                    singular: false,
                })
            } else {
                None
            },
            sparse_lu: None,
            xb: vec![T::zero(); m],
            tol,
            max_iters: 8_000_000,
            iters: 0,
            deadline: None, // rely on convergence; limit is a safety net
            perturb_on_stall: true,
            perturb_active: false,
            perturb_iters: 0,
            perturb_baseline: T::zero(),
            perturb_orig_b: Vec::new(),
            perturb_rng: 0x2545_F491_4F6C_DD1D,
            perturb_episodes: 0,
            perturb_failures: 0,
            perturb_scale: 1.0,
            n_etas: 0,
            eta_p: vec![0usize; MAX_ETAS],
            eta_alpha: vec![T::zero(); MAX_ETAS * m],
            eta_idx: Vec::with_capacity(MAX_ETAS * 16),
            eta_val: Vec::with_capacity(MAX_ETAS * 16),
            eta_start: vec![0usize; MAX_ETAS + 1],
            devex_row: vec![T::one(); m],
            devex_col: vec![T::one(); n],
            devex_enabled: devex_enabled(),
            spike: vec![T::zero(); m],
            rho_ep: vec![T::zero(); m],
            lu_z: vec![T::zero(); m],
            lu_y: vec![T::zero(); m],
            lu_pi: vec![T::zero(); m],
            bt_rho: vec![T::zero(); m],
            eta_scratch: vec![T::zero(); m],
            pi_buf: vec![T::zero(); m],
        }
    }

    /// Toggle the stall-triggered RHS perturbation (default on).
    pub fn set_perturb_on_stall(&mut self, on: bool) {
        self.perturb_on_stall = on;
    }

    /// Tighten one structural variable's box (`lb[j] = lo`, `ub[j] = hi`).
    ///
    /// This is the branch-and-bound node step: the constraint matrix, objective
    /// and row count are untouched, only a variable's allowed range shrinks. A
    /// basis optimal for the old box stays **dual feasible** for the new one —
    /// reduced costs do not depend on bounds — so the caller can re-optimize
    /// with [`Self::hot_solve`] from [`Self::export_basis`]: the dual loop
    /// starts from whatever primal bound violations the tightening introduced,
    /// which is exactly the violation-driven pivoting it is built for. Loosening
    /// a bound is also sound this way but pointless (the old optimum remains
    /// feasible), and *moving* a bound past the current basic value of a basic
    /// variable is what hot_solve's re-derivation of `xb` picks up.
    ///
    /// Only ever called between solves; there is no consistent intermediate
    /// state to reason about before the next `hot_solve`/`cold_solve`.
    pub fn set_var_bound(&mut self, j: usize, lo: T, hi: T) {
        debug_assert!(j < self.n, "column index out of range");
        self.l[j] = lo;
        self.u[j] = hi;
        // The variable may be nonbasic at the bound it just left; `hot_solve`
        // copies `at_upper` from the exported parent wholesale and the ratio
        // test reads `l`/`u` fresh each pivot, so no cached state needs
        // invalidating here.
    }

    /// Apply the stall-triggered RHS perturbation: save the exact RHS, add a
    /// deterministic relative perturbation per row, and re-derive the basic
    /// values. The magnitude is large enough to break the exact ties of a
    /// degenerate vertex but small enough that the escape it buys leads back
    /// to the same optimal basis. `baseline` is the pre-perturbation best
    /// worst-violation: the perturbation is considered to have escaped the
    /// vertex once the worst violation improves past it.
    ///
    /// The episode's magnitude scales with `perturb_scale`: consecutive failed
    /// episodes escalate (a 10-pivot sample of one pattern is not evidence the
    /// vertex is unescapable, only that one nudge was too small or unlucky),
    /// and a successful escape resets the scale. The escalation is bounded —
    /// after MAX_PERTURB_FAILURES failures later stalls go to Bland's rule,
    /// which keeps the termination guarantee.
    fn apply_rhs_perturbation(&mut self, baseline: T) {
        let m = self.m;
        self.perturb_orig_b.resize(m, T::zero());
        self.perturb_orig_b.copy_from_slice(&self.b);
        // Advance the deterministic LCG: a fresh pattern per episode, so a
        // perturbation that fails to escape can be retried differently.
        let mut r = self
            .perturb_rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for i in 0..m {
            r = r
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((r >> 11) as f64) / ((1u64 << 53) as f64);
            let mag =
                2e-4 * self.perturb_scale * (1.0 + self.b[i].abs().to_f64().expect("scalar literal"));
            let delta = (2.0 * u - 1.0) * mag;
            self.b[i] += T::from_f64(delta).expect("scalar literal");
        }
        self.perturb_rng = r;
        self.perturb_active = true;
        self.perturb_iters = 0;
        self.perturb_baseline = baseline;
        self.perturb_episodes += 1;
        self.xb = self.compute_xb();
    }

    /// Restore the exact RHS and re-derive the basic values. No-op when no
    /// perturbation is active. Called before every termination verdict and
    /// returned point: the perturbation is a temporary tie-breaker, never part
    /// of the LP whose solution the caller sees.
    fn restore_exact_rhs(&mut self) {
        if !self.perturb_active {
            return;
        }
        self.b.copy_from_slice(&self.perturb_orig_b);
        self.perturb_active = false;
        self.xb = self.compute_xb();
    }


    /// One primal-style Phase-1 pivot: bring the dual-infeasible nonbasic
    /// `j` into the basis (it moves off its bound), choosing the leaving row
    /// by the minimum ratio test. This is the standard repair for a basis
    /// that is BOTH primal-infeasible (some basic variable violates its
    /// bounds) and dual-infeasible (a nonbasic has a wrong-signed reduced
    /// cost): the dual ratio test cannot price such a basis (wrong-signed
    /// columns are skipped) and bound-flipping cannot fix columns sitting at
    /// an INFINITE opposite bound -- cut slacks [0, inf) are exactly that
    /// shape. Pivoting the offending column in restores dual feasibility by
    /// construction (the wrong-signed column is now basic) without touching
    /// the objective or the LP.
    ///
    /// Returns true when the pivot was applied (basis/xb updated; caller must
    /// recompute pi). Returns false when no basic row limits j's movement --
    /// with the LP bounded this is a numerical breakdown, not a certificate:
    /// the caller reports IterationLimit ("no definitive answer").
    fn phase1_pivot(&mut self, j: usize, use_blands_rule: bool) -> bool {
        let zero = T::zero();
        let huge = T::from_f64(1e19).expect("scalar literal");
        // Compute alpha = B^-1 A_j (scatter the sparse column, then FTRAN).
        let mut aj = vec![zero; self.m];
        let start = self.a_col_start[j];
        let end = self.a_col_start[j + 1];
        for idx in start..end {
            aj[self.a_row_idx[idx]] = self.a_val[idx];
        }
        let alpha = self.lu_solve(&aj);
        let push_up = !self.at_upper[j];
        // Minimum ratio test over all four bound-hit cases; bounds beyond
        // +/-1e19 never limit movement (see crash_dual_feasible's rule).
        let finite_cut = T::from_f64(1e19).expect("scalar literal");
        // Bland's leaving band, mirroring the main loop's `tie` constant:
        // under Bland's rule the leaving choice must be the smallest
        // basic-variable index among the min-ratio band for the termination
        // guarantee (row positions are arbitrary; variable indices are the
        // rule's substance).
        let bland_tie = T::from_f64(1e-12).expect("scalar literal");
        let tol = self.tol;
        let mut min_ratio = huge;
        let mut leave_p: Option<usize> = None;
        let mut leave_at_upper = false;
        let mut leave_bj = usize::MAX;
        for (i, &ai) in alpha.iter().enumerate() {
            if ai.abs() < tol {
                continue;
            }
            let bj = self.basis[i];
            let (ratio, hits_upper) = if push_up {
                if ai > zero && self.l[bj] > -finite_cut {
                    ((self.xb[i] - self.l[bj]) / ai, false)
                } else if ai < zero && self.u[bj] < finite_cut {
                    ((self.u[bj] - self.xb[i]) / (-ai), true)
                } else {
                    continue;
                }
            } else if ai > zero && self.u[bj] < finite_cut {
                ((self.u[bj] - self.xb[i]) / ai, true)
            } else if ai < zero && self.l[bj] > -finite_cut {
                ((self.xb[i] - self.l[bj]) / (-ai), false)
            } else {
                continue;
            };
            if Self::bland_leaving_prefer(
                use_blands_rule,
                ratio,
                min_ratio,
                self.basis[i],
                leave_bj,
                bland_tie,
            ) {
                min_ratio = ratio;
                leave_p = Some(i);
                leave_at_upper = hits_upper;
                leave_bj = self.basis[i];
            }
        }
        let p = match leave_p {
            Some(p) => p,
            None => return false,
        };
        self.at_upper[self.basis[p]] = leave_at_upper;
        self.update_basis_entry(p, j);
        // PFI eta update reusing alpha (the same discipline as the in-loop
        // Phase-1 arm); refactor when the chain is full or alpha[p] is too
        // small to commit.
        let refactored = if self.n_etas < MAX_ETAS && self.add_eta_from(p, &alpha) {
            self.n_etas += 1;
            false
        } else {
            self.factor();
            true
        };
        // Incremental primal update with the bounded-refresh discipline.
        if refactored {
            self.xb = self.compute_xb();
        } else {
            for (xi, &ai) in self.xb.iter_mut().zip(alpha.iter()) {
                if push_up {
                    *xi -= min_ratio * ai;
                } else {
                    *xi += min_ratio * ai;
                }
            }
            let entering_val = if push_up {
                self.l[j] + min_ratio
            } else {
                self.u[j] - min_ratio
            };
            self.xb[p] = entering_val;
        }
        true
    }

    /// Update the in_basis array when basis[p] changes from old_col to new_col.
    fn update_basis_entry(&mut self, p: usize, new_col: usize) {
        let old = self.basis[p];
        self.in_basis[old] = false;
        self.in_basis[new_col] = true;
        self.basis[p] = new_col;
    }

    /// LU factorization with SIMD-friendly blocked update.
    /// The inner loop is structured for autovectorization by LLVM:
    /// chunked row updates with unit stride access enable AVX2/AVX-512.
    /// Establishes a fresh base B_0 = self.lu/self.perm, so the PFI eta chain
    /// (self.eta_p/self.eta_alpha) built on top of it is cleared here.
    fn factor_once(&mut self) -> Vec<usize> {
        self.n_etas = 0;
        // Any fresh factorization invalidates a previously inherited sparse
        // factor (a warm-started solver may carry one even with this
        // solver's gate off) — clear it before dispatch so the eta chain is
        // never built on a base different from the one every base solve
        // routes through.
        self.sparse_lu = None;
        if sparse_lu_enabled() && self.m >= SPARSE_LU_MIN_M {
            if let Some(deficient) = self.factor_once_sparse() {
                return deficient;
            }
            // Sparse bailed (too dense / fill blowout): the dense fallback
            // below is now the unambiguous base.
        }
        if self.blas.is_some() && iconic_linalg::blas::blas_enabled() {
            return self.factor_once_blas();
        }
        self.factor_once_hand()
    }

    /// Markowitz sparse LU variant of [`factor_once`]: build the basis
    /// matrix's CSC view (columns already live in the solver's `a_*`
    /// storage), factor it sparsely, and record the result as the base
    /// factor. Returns `None` when the basis is too dense for the packed
    /// representation (the caller falls back to its dense paths); otherwise
    /// the deficient-slot list, with `self.perm` filled to the repair
    /// contract (`perm[slot]` = the row whose slack substitutes a dependent
    /// slot) exactly as the hand path does.
    fn factor_once_sparse(&mut self) -> Option<Vec<usize>> {
        let m = self.m;
        // Structural density pre-check without touching the factor work.
        let mut nnz = 0usize;
        for &col in &self.basis[..m] {
            nnz += self.a_col_start[col + 1] - self.a_col_start[col];
        }
        if nnz as f64 > sparse_lu::FILL_DENSITY_BAIL * (m as f64) * (m as f64) {
            return None;
        }
        let (slu, deficient, perm) = SparseLu::build(
            m,
            &self.a_col_start,
            &self.a_row_idx,
            &self.a_val,
            &self.basis,
        )?;
        // The dense `lu` array is left stale while the sparse factor is
        // live (documented on the field): every consumer of the base factor
        // dispatches through `sparse_lu` first, and warm starts carry the
        // sparse factor itself. `swaps` stays all-MAX — there is no LAPACK
        // swap sequence here, and the only consumer that reconstructs a
        // mirror from `swaps` is skipped when a warm start inherits this
        // factor.
        self.perm.copy_from_slice(&perm);
        self.swaps.fill(usize::MAX);
        self.sparse_lu = Some(slu);
        Some(deficient)
    }

    /// The hand-rolled dense LU with partial pivoting (rows `perm[i]` in
    /// elimination order, unit-lower multipliers below the diagonal, `U` in
    /// the pivot rows — LAPACK's convention, so the two paths are
    /// interchangeable). Records the row-swap sequence in `self.swaps` for
    /// the warm-start mirror reconstruction. Returns the rows whose pivot
    /// fell below the PIVOT_TOL repair floor.
    /// Grow the dense factor array to m^2 when a dense writer needs it.
    /// The array starts EMPTY (see the constructor note): the sparse base
    /// factor never touches it, so a fresh solver pays nothing. `resize`
    /// zero-fills, which is also what `factor_once_hand`'s own
    /// `self.lu.fill(T::zero())` would do -- one memset either way, and only
    /// on the paths that actually read/write the dense layout.
    #[inline]
    fn ensure_lu(&mut self) {
        let want = self.m * self.m;
        if self.lu.len() != want {
            self.lu.resize(want, T::zero());
        }
    }

    fn factor_once_hand(&mut self) -> Vec<usize> {
        let m = self.m;
        self.ensure_lu();
        // Load basis matrix columns from CSC storage: slack columns are identity
        // (1 nonzero), structural columns typically 1-5% dense. The `m < 100` case used
        // to load from a dense copy of A instead, on the grounds that contiguous access
        // vectorizes better -- but the access it made was `a[k * n + col]`, one column of
        // a *row-major* array, so it strode `n` elements per step and read `m` cache
        // lines to gather `m` values. It also required keeping the whole dense `m * n`
        // array alive purely for this one loop.
        self.lu.fill(T::zero());
        for i in 0..m {
            let col = self.basis[i];
            let start = self.a_col_start[col];
            let end = self.a_col_start[col + 1];
            for idx in start..end {
                let row = self.a_row_idx[idx];
                self.lu[row * m + i] = self.a_val[idx];
            }
        }
        self.perm = (0..m).collect();
        self.swaps.fill(usize::MAX);
        let eps = T::from_f64(PIVOT_TOL).expect("scalar literal");
        let mut deficient: Vec<usize> = Vec::new();

        // Block size for L2 cache-friendly factorization
        const BLK: usize = 32;

        for kk in (0..m).step_by(BLK) {
            let k_end = (kk + BLK).min(m);
            // Factor the diagonal block
            for k in kk..k_end {
                let mut pv = self.lu[self.perm[k] * m + k].abs();
                let mut pr = k;
                for i in (k + 1)..m {
                    let v = self.lu[self.perm[i] * m + k].abs();
                    if v > pv {
                        pv = v;
                        pr = i;
                    }
                }
                // No acceptable pivot in this column: the basic variable in this
                // position is linearly dependent on the rest of the basis. Record it
                // for repair -- skipping the elimination leaves a zero on U's
                // diagonal, which the triangular solves later divide by.
                if pv < eps {
                    deficient.push(k);
                    continue;
                }
                self.perm.swap(k, pr);
                self.swaps[k] = pr;
                let pk = self.perm[k];
                let piv = self.lu[pk * m + k];

                // SIMD-friendly: chunk the inner elimination loop for autovectorization.
                // LLVM can autovectorize this into AVX2/AVX-512 instructions.
                let mut i = k + 1;
                // Process 4 rows at a time (autovectorizable chunk)
                while i + 4 <= m {
                    let pi0 = self.perm[i];
                    let pi1 = self.perm[i + 1];
                    let pi2 = self.perm[i + 2];
                    let pi3 = self.perm[i + 3];
                    let f0 = self.lu[pi0 * m + k] / piv;
                    let f1 = self.lu[pi1 * m + k] / piv;
                    let f2 = self.lu[pi2 * m + k] / piv;
                    let f3 = self.lu[pi3 * m + k] / piv;
                    self.lu[pi0 * m + k] = f0;
                    self.lu[pi1 * m + k] = f1;
                    self.lu[pi2 * m + k] = f2;
                    self.lu[pi3 * m + k] = f3;
                    // Inner update: contiguous access for autovectorization
                    for j in (k + 1)..m {
                        let uj = self.lu[pk * m + j];
                        self.lu[pi0 * m + j] -= f0 * uj;
                        self.lu[pi1 * m + j] -= f1 * uj;
                        self.lu[pi2 * m + j] -= f2 * uj;
                        self.lu[pi3 * m + j] -= f3 * uj;
                    }
                    i += 4;
                }
                // Remainder
                for i in i..m {
                    let pi = self.perm[i];
                    let f = self.lu[pi * m + k] / piv;
                    self.lu[pi * m + k] = f;
                    for j in (k + 1)..m {
                        self.lu[pi * m + j] = self.lu[pi * m + j] - f * self.lu[pk * m + j];
                    }
                }
            }
        }
        deficient
    }

    /// LAPACK `dgetrf` variant of [`factor_once`]: gather the basis columns
    /// into the f64 mirror, factor in place, copy back to the T-typed `lu`
    /// (the HotBasis export and hand-path solves read it), reconstruct
    /// `perm` and the swap sequence from LAPACK's `ipiv`, and scan U's
    /// diagonal for the 1e-20 repair floor — LAPACK reports only exactly-zero
    /// pivots, so the deficient list is built from the same floor the hand
    /// path's partial-pivoting scan applies.
    fn factor_once_blas(&mut self) -> Vec<usize> {
        let m = self.m;
        if self.blas.as_ref().is_some_and(|b| b.lu.len() != m * m) {
            let b = self.blas.as_mut().expect("checked above");
            b.lu.resize(m * m, 0.0);
        }
        // The copy-back below writes the full row-major `self.lu`.
        self.ensure_lu();
        let b = self.blas.as_mut().expect("gated by blas.is_some()");
        b.lu.fill(0.0);
        // Gather the basis into COLUMN-major layout (`index = row + col*m`),
        // which is what LAPACK factors natively: the mirror then holds the
        // factor of the basis itself (not its transpose), so dgetrs solves
        // directly and the copy-back into the row-major `self.lu` is a
        // plain transpose (the hand-rolled path reads row-major).
        for i in 0..m {
            let col = self.basis[i];
            for idx in self.a_col_start[col]..self.a_col_start[col + 1] {
                b.lu[self.a_row_idx[idx] + i * m] = self.a_val[idx].to_f64().expect("scalar literal");
            }
        }
        // dgetrf aborts at the first *exactly* zero pivot (info = k+1)
        // where the hand path records a deficient row and continues; the
        // repair floor below catches the near-zero case either way. On an
        // exact-zero abort the factor output is unusable, so fall back to
        // the hand factor for this basis and rebuild the mirror from it
        // (the hand factor's layout is the same convention, and the mirror's
        // ipiv is derived from the recorded swap sequence, so the BLAS
        // solves remain valid on the repaired factor).
        let ipiv = match iconic_linalg::blas::getrf(m, m, &mut b.lu) {
            Some(p) => p,
            None => {
                let deficient = self.factor_once_hand();
                if self.blas.is_some() && iconic_linalg::blas::blas_enabled() {
                    self.rebuild_blas_mirror();
                }
                return deficient;
            }
        };
        // Copy the column-major factor back into the row-major `self.lu`
        // (a transpose: the hand path reads `lu[row*m + col]`).
        for r in 0..m {
            for c in 0..m {
                self.lu[r * m + c] = T::from_f64(b.lu[r + c * m]).expect("scalar literal");
            }
        }
        // Store LAPACK's pivot sequence for the solves (`getrs` reads the
        // ipiv of the *mirror*, not the perm — an all-zero ipiv made every
        // solve apply garbage interchanges).
        b.ipiv[..m].copy_from_slice(&ipiv[..m]);
        // Reconstruct perm and the swap sequence from ipiv: at step k
        // (0-based) dgetrf interchanged positions k and ipiv[k] (1-based);
        // applying the sequence to the identity gives the original row at
        // each output position — the hand path's `perm` convention.
        self.perm = (0..m).collect();
        let mut deficient = Vec::new();
        for (k, &ip1) in ipiv.iter().enumerate().take(m) {
            let ip = ip1 - 1;
            if ip == k as i32 {
                self.swaps[k] = usize::MAX;
            } else {
                self.perm.swap(k, ip as usize);
                self.swaps[k] = ip as usize;
            }
            // U's diagonal sits at OUTPUT position (k, k) of the LAPACK
            // array (column-major index k + k*m) — NOT at (perm[k], k):
            // the pivot row occupies output row k, wherever it came from.
            if b.lu[k + k * m].abs() <= PIVOT_TOL {
                deficient.push(k);
            }
        }
        b.singular = !deficient.is_empty();
        deficient
    }

    /// Rebuild the LAPACK mirror from the T-typed factor without
    /// re-factorizing: the warm-start path (`hot_solve`) inherits the
    /// parent's factored LU and perm, so the mirror is reconstructed by
    /// conversion (transposed: row-major to column-major) + `ipiv`
    /// derivation from the swap sequence (exactly what `dgetrs` applies, so
    /// the inherited factor solves consistently).
    fn rebuild_blas_mirror(&mut self) {
        let m = self.m;
        self.ensure_lu();
        if self.blas.as_ref().is_some_and(|b| b.lu.len() != m * m) {
            let b = self.blas.as_mut().expect("checked above");
            b.lu.resize(m * m, 0.0);
        }
        let b = self.blas.as_mut().expect("gated by blas.is_some()");
        for r in 0..m {
            for c in 0..m {
                b.lu[r + c * m] = self.lu[r * m + c].to_f64().expect("scalar literal");
            }
        }
        let mut singular = false;
        for k in 0..m {
            b.ipiv[k] = if self.swaps[k] == usize::MAX {
                (k + 1) as i32
            } else {
                (self.swaps[k] + 1) as i32
            };
            if self.lu[self.perm[k] * m + k].abs().to_f64().expect("scalar literal") <= PIVOT_TOL {
                singular = true;
            }
        }
        b.singular = singular;
    }

    /// Solve the base (pre-eta-chain) factor `B0 * x = rhs` into `lu_y`:
    /// the packed Markowitz factor when one is live (`O(nnz)` per RHS),
    /// LAPACK `dgetrs` when the BLAS mirror is live and the factor is
    /// healthy, else the hand-rolled triangular solves (whose backward pass
    /// emits the NaN sentinel on a below-floor pivot). `scratch` serves the
    /// sparse path only; disjoint-field associated fn, per the file's borrow
    /// convention.
    #[allow(clippy::too_many_arguments)]
    fn solve_base_into(
        rhs: &[T],
        lu: &[T],
        perm: &[usize],
        lu_y: &mut [T],
        m: usize,
        scratch: &mut [T],
        sparse: Option<&SparseLu<T>>,
        blas: Option<&mut BlasLu>,
    ) {
        if let Some(s) = sparse {
            s.solve_into(rhs, lu_y, scratch);
            return;
        }
        if let Some(b) = blas {
            if !b.singular && iconic_linalg::blas::blas_enabled() {
                for (out, &v) in b.rhs[..m].iter_mut().zip(rhs.iter()) {
                    *out = v.to_f64().expect("scalar literal");
                }
                // The mirror holds the column-major factor of the basis, so
                // `B x = b` is dgetrs with `trans = 'N'`.
                iconic_linalg::blas::getrs(m, &b.lu, &b.ipiv, &mut b.rhs, false);
                for (out, &v) in lu_y[..m].iter_mut().zip(b.rhs.iter()) {
                    *out = T::from_f64(v).expect("scalar literal");
                }
                return;
            }
        }
        let tol = T::from_f64(PIVOT_TOL).expect("scalar literal");
        Self::lu_solve_into_dense(lu, perm, rhs, lu_y, m, tol);
    }

    /// Solve the base factor's transpose `B0ᵀ * x = cbt` into `lu_y` (BLAS
    /// `dgetrs` transposed, else the hand-rolled transpose triangular
    /// solves, which use `lu_z` and `spike` as scratch — disjoint from the
    /// caller's buffers by convention).
    #[allow(clippy::too_many_arguments)]
    fn solve_base_trans_into(
        cbt: &[T],
        lu: &[T],
        perm: &[usize],
        lu_y: &mut [T],
        lu_z: &mut [T],
        spike: &mut [T],
        m: usize,
        sparse: Option<&SparseLu<T>>,
        blas: Option<&mut BlasLu>,
    ) {
        if let Some(s) = sparse {
            s.solve_trans_into(cbt, lu_y, lu_z);
            return;
        }
        if let Some(b) = blas {
            if !b.singular && iconic_linalg::blas::blas_enabled() {
                for (out, &v) in b.rhs[..m].iter_mut().zip(cbt.iter()) {
                    *out = v.to_f64().expect("scalar literal");
                }
                // `Bᵀ x = c` is dgetrs with `trans = 'T'` on the column-major
                // factor of the basis.
                iconic_linalg::blas::getrs(m, &b.lu, &b.ipiv, &mut b.rhs, true);
                for (out, &v) in lu_y[..m].iter_mut().zip(b.rhs.iter()) {
                    *out = T::from_f64(v).expect("scalar literal");
                }
                return;
            }
        }
        Self::lu_solve_trans_into_dense(lu, perm, cbt, lu_y, lu_z, spike, m);
    }

    /// Factor the basis, repairing it if it turns out to be singular.
    ///
    /// A basis can go rank-deficient mid-solve. The factorization then finds no
    /// acceptable pivot in some column, and simply skipping that elimination step
    /// leaves a zero on U's diagonal -- which the triangular solves divide by, so
    /// `pi` and `rho` come back non-finite. Every subsequent comparison against a
    /// NaN is false, so the dual ratio test rejects every column and reports
    /// `Infeasible`: a numerical breakdown dressed up as a proof. Branch-and-bound
    /// believes it and prunes a feasible subtree.
    ///
    /// The standard remedy is to repair the basis rather than to factor a singular
    /// one: a dependent basic variable is replaced by the logical (slack) column of
    /// its pivot row, which is a unit vector and therefore restores rank, and the
    /// basis is factored again. Each round strictly increases the number of logical
    /// columns in the basis, so this terminates.
    fn factor(&mut self) {
        let slack_start = self.n - self.m;
        let huge = T::from_f64(-1e19).expect("scalar literal");
        let mut any_repair = false;
        // At most one repair per row, plus one final clean factorization.
        for _build in 0..=self.m {
            let deficient = self.factor_once();
            if deficient.is_empty() {
                // A repair changed which variables are basic, so the cached primal
                // values belong to the old basis. Refresh them against the new one.
                if any_repair {
                    self.xb = self.compute_xb();
                }
                return;
            }
            let mut repaired = false;
            for k in deficient {
                let row = self.perm[k];
                let logical = slack_start + row;
                // Already basic elsewhere -- this position cannot be repaired with it.
                if self.in_basis[logical] {
                    continue;
                }
                let old = self.basis[k];
                self.in_basis[old] = false;
                self.basis[k] = logical;
                self.in_basis[logical] = true;
                // The evicted variable becomes nonbasic at a finite bound: its lower
                // one when that is finite, otherwise its upper.
                self.at_upper[old] = !(self.l[old].is_finite() && self.l[old] > huge);
                repaired = true;
                any_repair = true;
            }
            // Nothing left to substitute: fall through with the best factor available
            // and let the caller's non-finite guard turn it into an honest failure.
            if !repaired {
                if any_repair {
                    self.xb = self.compute_xb();
                }
                return;
            }
        }
    }

    /// Apply the forward PFI eta chain (creation order, oldest to newest) to `x`
    /// in place: `x <- E_{n_etas}^{-1}(...(E_1^{-1} x)...)`. Each `E_i^{-1}` is the
    /// Sherman-Morrison inverse of a rank-1 eta `E_i = I + (alpha_i - e_{p_i})
    /// e_{p_i}^T`: `beta = x[p_i] / alpha_i[p_i]`, then `x[k] -= beta*alpha_i[k]`
    /// for `k != p_i` and `x[p_i] = beta`. Associated fn (explicit slice args, not
    /// `&mut self`) so callers can pass disjoint struct-field borrows (eta_p/
    /// eta_alpha read while a different field is written), matching
    /// lu_solve_into/lu_solve_trans_into's existing calling convention.
    #[allow(clippy::too_many_arguments)]
    fn apply_etas_forward(
        eta_p: &[usize],
        eta_start: &[usize],
        eta_idx: &[u32],
        eta_val: &[T],
        eta_alpha: &[T],
        n_etas: usize,
        m: usize,
        x: &mut [T],
    ) {
        if m == 0 || n_etas == 0 {
            return;
        }
        let zero = T::zero();
        for e in 0..n_etas {
            let pt = eta_p[e];
            let (s0, s1) = (eta_start[e], eta_start[e + 1]);
            // Per-eta format choice. Skipping true zeros is arithmetically
            // identical to the dense pass (each component receives exactly one
            // independent update either way), so the two arms are bit-stable
            // and interchangeable; the split is purely about memory traffic --
            // the sparse form pays gather/scatter per nonzero, the dense form
            // a contiguous m-length pass. Below half density sparse wins.
            // An empty stored range marks a dense alpha (see commit_eta).
            if s1 == s0 || 2 * (s1 - s0) >= m {
                let ab = e * m;
                let beta = x[pt] / eta_alpha[ab + pt];
                if beta == zero {
                    continue;
                }
                for i in 0..m {
                    if i == pt {
                        x[i] = beta;
                    } else {
                        x[i] -= beta * eta_alpha[ab + i];
                    }
                }
            } else {
                // The pivot entry's position within the stored nonzeros.
                // commit_eta guarantees |alpha[p]| above tolerance, so the
                // scan always finds it.
                let mut pk = s0;
                while pk < s1 && eta_idx[pk] as usize != pt {
                    pk += 1;
                }
                debug_assert!(pk < s1, "eta pivot entry missing from sparse form");
                if pk >= s1 {
                    continue;
                }
                let beta = x[pt] / eta_val[pk];
                if beta == zero {
                    continue;
                }
                for k in s0..s1 {
                    let ii = eta_idx[k] as usize;
                    if ii == pt {
                        x[ii] = beta;
                    } else {
                        x[ii] -= beta * eta_val[k];
                    }
                }
            }
        }
    }

    /// Apply the transpose PFI eta chain (newest to oldest) to `x` in place:
    /// `x <- E_1^{-T}(...(E_{n_etas}^{-T} x)...)`. `E_i^{-T}` only ever changes
    /// component `p_i`: `gamma = (alpha_i . x - x[p_i]) / alpha_i[p_i]`, then
    /// `x[p_i] -= gamma`. Associated fn for the same disjoint-borrow reason as
    /// `apply_etas_forward`.
    #[allow(clippy::too_many_arguments)]
    fn apply_etas_transpose(
        eta_p: &[usize],
        eta_start: &[usize],
        eta_idx: &[u32],
        eta_val: &[T],
        eta_alpha: &[T],
        n_etas: usize,
        m: usize,
        x: &mut [T],
    ) {
        if m == 0 || n_etas == 0 {
            return;
        }
        let zero = T::zero();
        for e in (0..n_etas).rev() {
            let pt = eta_p[e];
            let (s0, s1) = (eta_start[e], eta_start[e + 1]);
            // Same per-eta format choice as apply_etas_forward (see there).
            // An empty stored range marks a dense alpha (see commit_eta).
            if s1 == s0 || 2 * (s1 - s0) >= m {
                let ab = e * m;
                let mut dot = zero;
                for i in 0..m {
                    dot += eta_alpha[ab + i] * x[i];
                }
                let gamma = (dot - x[pt]) / eta_alpha[ab + pt];
                x[pt] -= gamma;
            } else {
                let mut pk = s0;
                while pk < s1 && eta_idx[pk] as usize != pt {
                    pk += 1;
                }
                debug_assert!(pk < s1, "eta pivot entry missing from sparse form");
                if pk >= s1 {
                    continue;
                }
                let mut dot = zero;
                for k in s0..s1 {
                    dot += eta_val[k] * x[eta_idx[k] as usize];
                }
                let gamma = (dot - x[pt]) / eta_val[pk];
                x[pt] -= gamma;
            }
        }
    }

    /// Validate `alpha[p]` (expected in `self.lu_y[..m]`) and, if above tolerance,
    /// commit it as the next eta. Shared tail of `add_eta_from` and the main
    /// pivot path, which computes `alpha` into `lu_y` inline before calling this.
    fn commit_eta(&mut self, p: usize) -> bool {
        // Tighter than the ratio test's alpha_tol (1e-10): under normal operation
        // alpha[p] is exactly the quantity the ratio test already validated above
        // alpha_tol before selecting this pivot, so this should essentially never
        // fire except on genuine numerical breakdown -- a wider safety margin
        // below the ratio test's own threshold, erring toward the always-correct
        // full-refactor fallback rather than accepting a borderline eta.
        let eps = T::from_f64(1e-12).expect("scalar literal");
        if self.lu_y[p].abs() < eps {
            return false;
        }
        let m = self.m;
        let slot = self.n_etas;
        let base = slot * m;
        self.eta_alpha[base..base + m].copy_from_slice(&self.lu_y[..m]);
        self.eta_p[slot] = p;
        // Sparse form: record every exactly-nonzero entry (skipping true
        // zeros is arithmetically identical to the dense pass, so formats
        // are bit-stable and interchangeable -- see apply_etas_forward).
        // Materialized only below half density; a dense alpha keeps the
        // contiguous layout and pays no scatter-list bookkeeping.
        if slot == 0 {
            self.eta_idx.clear();
            self.eta_val.clear();
        }
        self.eta_start[slot] = self.eta_idx.len();
        let zero = T::zero();
        let mut nz = 0usize;
        for i in 0..m {
            if self.lu_y[i] != zero {
                nz += 1;
            }
        }
        if 2 * nz < m {
            for i in 0..m {
                let v = self.lu_y[i];
                if v != zero {
                    self.eta_idx.push(i as u32);
                    self.eta_val.push(v);
                }
            }
        }
        // A dense eta is encoded by leaving its [start, end) range empty --
        // impossible for a real sparse eta, whose pivot entry guarantees at
        // least one stored nonzero.
        self.eta_start[slot + 1] = self.eta_idx.len();
        true
    }

    /// Append a PFI eta from an already-computed `alpha = B_current^{-1} a_{q}`,
    /// avoiding redundant derivation when the caller (e.g. the Phase-1
    /// primal-pivot ratio test) already needed the full alpha vector.
    fn add_eta_from(&mut self, p: usize, alpha: &[T]) -> bool {
        self.lu_y[..self.m].copy_from_slice(alpha);
        self.commit_eta(p)
    }

    /// Triangular solve: writes result into out. Pre-allocated associated fn (pre-allocated buffers).
    fn lu_solve_into_dense(lu: &[T], perm: &[usize], rhs: &[T], out: &mut [T], m: usize, tol: T) {
        Self::fwd_solve_dense(lu, perm, rhs, out, m);
        Self::bwd_solve_dense(lu, perm, out, m, tol);
    }
    fn fwd_solve_dense(lu: &[T], perm: &[usize], rhs: &[T], out: &mut [T], m: usize) {
        let mut i = 0;
        while i + 4 <= m {
            let (pi0, pi1, pi2, pi3) = (perm[i], perm[i + 1], perm[i + 2], perm[i + 3]);
            let (r0, r1, r2, r3) = (pi0 * m, pi1 * m, pi2 * m, pi3 * m);
            let (mut s0, mut s1, mut s2, mut s3) = (rhs[pi0], rhs[pi1], rhs[pi2], rhs[pi3]);
            for j in 0..i {
                s0 -= lu[r0 + j] * out[j];
                s1 -= lu[r1 + j] * out[j];
                s2 -= lu[r2 + j] * out[j];
                s3 -= lu[r3 + j] * out[j];
            }
            out[i] = s0;
            let y0 = s0;
            s1 -= lu[r1 + i] * y0;
            out[i + 1] = s1;
            let y1 = s1;
            s2 = s2 - lu[r2 + i] * y0 - lu[r2 + i + 1] * y1;
            out[i + 2] = s2;
            let y2 = s2;
            s3 = s3 - lu[r3 + i] * y0 - lu[r3 + i + 1] * y1 - lu[r3 + i + 2] * y2;
            out[i + 3] = s3;
            i += 4;
        }
        for i in i..m {
            let pi = perm[i];
            let mut s = rhs[pi];
            let row = pi * m;
            for j in 0..i {
                s -= lu[row + j] * out[j];
            }
            out[i] = s;
        }
    }
    // A pivot at or below `tol` (the shared PIVOT_TOL / factor-repair floor)
    // means the basis is singular: emitting the quotient would divide by a
    // numerically meaningless pivot, but clamping to 0.0 (the old behavior)
    // silently produced a finite-but-wrong solution that evaded every
    // non-finite guard. Emit a NaN sentinel instead so the existing recovery
    // paths (the xb/pi/rho NaN guards, which refactor once and then report
    // IterationLimit) engage. The transposed solve already behaves this way
    // -- it divides unconditionally, so a zero pivot naturally yields
    // non-finite output.
    fn bwd_solve_dense(lu: &[T], perm: &[usize], y: &mut [T], m: usize, tol: T) {
        let nan = T::nan();
        let mut i = m;
        while i >= 4 {
            i -= 4;
            let (pi0, pi1, pi2, pi3) = (perm[i], perm[i + 1], perm[i + 2], perm[i + 3]);
            let (r0, r1, r2, r3) = (pi0 * m, pi1 * m, pi2 * m, pi3 * m);
            let (mut s0, mut s1, mut s2, mut s3) = (y[i], y[i + 1], y[i + 2], y[i + 3]);
            for j in (i + 4)..m {
                let oj = y[j];
                s0 -= lu[r0 + j] * oj;
                s1 -= lu[r1 + j] * oj;
                s2 -= lu[r2 + j] * oj;
                s3 -= lu[r3 + j] * oj;
            }
            let d3 = lu[r3 + i + 3];
            y[i + 3] = if d3.abs() > tol { s3 / d3 } else { nan };
            let x3 = y[i + 3];
            s2 -= lu[r2 + i + 3] * x3;
            let d2 = lu[r2 + i + 2];
            y[i + 2] = if d2.abs() > tol { s2 / d2 } else { nan };
            let x2 = y[i + 2];
            s1 -= lu[r1 + i + 2] * x2 + lu[r1 + i + 3] * x3;
            let d1 = lu[r1 + i + 1];
            y[i + 1] = if d1.abs() > tol { s1 / d1 } else { nan };
            let x1 = y[i + 1];
            s0 -= lu[r0 + i + 1] * x1 + lu[r0 + i + 2] * x2 + lu[r0 + i + 3] * x3;
            let d0 = lu[r0 + i];
            y[i] = if d0.abs() > tol { s0 / d0 } else { nan };
        }
        for i in (0..i).rev() {
            let pi = perm[i];
            let row = pi * m;
            let mut s = y[i];
            for j in (i + 1)..m {
                s -= lu[row + j] * y[j];
            }
            let d = lu[row + i];
            y[i] = if d.abs() > tol { s / d } else { nan };
        }
    }

    /// Solve `B_current * x = rhs`: the base triangular solve against `self.lu`/
    /// `self.perm`, then the PFI eta chain (forward, oldest to newest). Uses
    /// pre-allocated self.lu_y buffer (pre-allocated buffers).
    fn lu_solve(&mut self, rhs: &[T]) -> Vec<T> {
        // The backward solve's singular-pivot floor is the factor-repair
        // threshold (PIVOT_TOL), not self.tol: a healthy basis never contains
        // a pivot below it, so encountering one means singular, and the solve
        // must report it rather than silently clamp (see bwd_solve_dense).
        // The BLAS mirror path preserves that contract: a factor flagged
        // singular at factorization time defers to the hand solve, whose
        // backward pass emits the NaN sentinel.
        let m = self.m;
        Self::solve_base_into(
            rhs,
            &self.lu,
            &self.perm,
            &mut self.lu_y,
            m,
            &mut self.lu_z,
            self.sparse_lu.as_ref(),
            self.blas.as_mut(),
        );
        Self::apply_etas_forward(&self.eta_p, &self.eta_start, &self.eta_idx, &self.eta_val, &self.eta_alpha, self.n_etas, m, &mut self.lu_y);
        self.lu_y[..self.m].to_vec()
    }

    /// Transpose triangular solve: writes result into out, uses z_buf/y_buf as scratch.
    /// Pre-allocated associated fn (pre-allocated buffers).
    fn lu_solve_trans_into_dense(
        lu: &[T],
        perm: &[usize],
        cbt: &[T],
        out: &mut [T],
        z_buf: &mut [T],
        y_buf: &mut [T],
        m: usize,
    ) {
        // Forward: z = L^{-T} cbt
        for i in 0..m {
            let pi = perm[i];
            let mut s = cbt[i];
            for j in 0..i {
                s -= lu[perm[j] * m + i] * z_buf[j];
            }
            z_buf[i] = s / lu[pi * m + i];
        }
        // Backward: y = U^{-T} z
        for i in (0..m).rev() {
            let mut s = z_buf[i];
            for j in (i + 1)..m {
                s -= lu[perm[j] * m + i] * y_buf[j];
            }
            y_buf[i] = s;
        }
        // Permute: out = P^T y
        for i in 0..m {
            out[perm[i]] = y_buf[i];
        }
    }
    /// Compute xb = B^{-1}(b - A_N * x_N) with iterative refinement.
    /// Refining every LU solve is cheap relative to the factorization it protects: one
    /// step reduces error from epsilon_mach*kappa(B) (~1e-8) to
    /// epsilon_mach*kappa(B)^{2/3} (~1e-12).
    fn compute_xb(&mut self) -> Vec<T> {
        let (m, n) = (self.m, self.n);
        // Build RHS: b - A_N * x_N (CSC sparse column access)
        let mut rhs = self.b.clone();
        for j in 0..n {
            if self.in_basis[j] {
                continue;
            }
            let xj = if self.at_upper[j] {
                self.u[j]
            } else {
                self.l[j]
            };
            if xj != T::zero() {
                let start = self.a_col_start[j];
                let end = self.a_col_start[j + 1];
                for idx in start..end {
                    rhs[self.a_row_idx[idx]] -= self.a_val[idx] * xj;
                }
            }
        }
        let mut xb = self.lu_solve(&rhs);
        // Iterative refinement: r = rhs - B*xb, solve B*dx = r, xb += dx
        let refine_tol = T::from_f64(1e-12).expect("scalar literal");
        for _ in 0..2 {
            // Compute B * xb (CSC sparse column access over basis columns)
            let mut bx = vec![T::zero(); m];
            for (&xj, &col) in xb.iter().zip(self.basis.iter()) {
                let start = self.a_col_start[col];
                let end = self.a_col_start[col + 1];
                for idx in start..end {
                    bx[self.a_row_idx[idx]] += self.a_val[idx] * xj;
                }
            }
            // Residual r = rhs - B*xb
            let mut r_norm = T::zero();
            for i in 0..m {
                let ri = rhs[i] - bx[i];
                bx[i] = ri; /* reuse as residual */
                if ri.abs() > r_norm {
                    r_norm = ri.abs();
                }
            }
            // Relative stopping: ||r||_inf must be meaningfully large to refine
            let xb_norm = xb.iter().fold(
                T::zero(),
                |acc, &v| if v.abs() > acc { v.abs() } else { acc },
            );
            let thresh = refine_tol * (T::one() + xb_norm);
            if r_norm < thresh {
                break;
            }
            // Solve correction: B * dx = r
            let dx = self.lu_solve(&bx);
            for i in 0..m {
                xb[i] += dx[i];
            }
        }
        xb
    }

    /// Write dual variables pi = B^{-T} c_B into self.pi_buf (no allocation).
    fn compute_pi(&mut self) {
        let m = self.m;
        for i in 0..m {
            self.eta_scratch[i] = self.c[self.basis[i]];
        }
        Self::apply_etas_transpose(
            &self.eta_p,
            &self.eta_start,
            &self.eta_idx,
            &self.eta_val,
            &self.eta_alpha,
            self.n_etas,
            m,
            &mut self.eta_scratch,
        );
        let m = self.m;
        Self::solve_base_trans_into(
            &self.eta_scratch,
            &self.lu,
            &self.perm,
            &mut self.lu_y,
            &mut self.lu_z,
            &mut self.spike,
            m,
            self.sparse_lu.as_ref(),
            self.blas.as_mut(),
        );
        self.pi_buf[..m].copy_from_slice(&self.lu_y[..m]);
    }

    fn reduced_cost(&self, j: usize, pi: &[T]) -> T {
        let mut dot = T::zero();
        let start = self.a_col_start[j];
        let end = self.a_col_start[j + 1];
        let base = self.a_col_contig[j];
        if base != usize::MAX {
            let vals = &self.a_val[start..end];
            let pis = &pi[base..base + vals.len()];
            for k in 0..vals.len() {
                dot += pis[k] * vals[k];
            }
        } else {
            for idx in start..end {
                dot += pi[self.a_row_idx[idx]] * self.a_val[idx];
            }
        }
        self.c[j] - dot
    }

    /// Fused dot products: computes both reduced_cost (c_j - pi^T a_j) and
    /// alpha (rho^T a_j) in a single traversal of column j (fused ratio test).
    ///
    /// CSC throughout. The dense-column special case this used to carry indexed
    /// `a[i * n + j]` -- one column of a row-major array, a cache line per element --
    /// so it was slower than the sparse path it was meant to beat even at full density,
    /// where CSC reads `m` contiguous values and gathers `pi`/`rho` by sequential row
    /// indices.
    fn reduced_cost_and_alpha(&self, j: usize, pi: &[T], rho: &[T]) -> (T, T) {
        let mut dot_pi = T::zero();
        let mut dot_rho = T::zero();
        let start = self.a_col_start[j];
        let end = self.a_col_start[j + 1];
        let base = self.a_col_contig[j];
        if base != usize::MAX {
            // Contiguous rows: index-free, so this compiles to a straight vectorizable
            // triple of slices rather than a gather.
            let vals = &self.a_val[start..end];
            let pis = &pi[base..base + vals.len()];
            let rhos = &rho[base..base + vals.len()];
            for k in 0..vals.len() {
                let aij = vals[k];
                dot_pi += pis[k] * aij;
                dot_rho += rhos[k] * aij;
            }
        } else {
            for idx in start..end {
                let row = self.a_row_idx[idx];
                let aij = self.a_val[idx];
                dot_pi += pi[row] * aij;
                dot_rho += rho[row] * aij;
            }
        }
        (self.c[j] - dot_pi, dot_rho)
    }

    /// True iff the current basis is dual-feasible at tolerance `opt_tol`:
    /// every nonbasic (non-fixed) variable's reduced cost has the sign its
    /// current bound requires -- nonnegative at its lower bound, nonpositive
    /// at its upper bound. This is exactly the test the entering scan of
    /// dual_loop's Phase-1 branch applies, extracted so the infeasibility
    /// certificate can verify it: a dual-simplex infeasibility proof is only
    /// valid from a dual-feasible basis, so the certificate checks must
    /// re-confirm dual feasibility before declaring Infeasible. Fixed columns
    /// (l == u) are excluded here as they are from entering: they cannot move,
    /// so they are dual-feasible at any reduced cost.
    fn is_dual_feasible(&self, pi: &[T], opt_tol: T) -> bool {
        for j in 0..self.n {
            if self.in_basis[j] {
                continue;
            }
            if self.u[j] - self.l[j] < self.tol {
                continue;
            }
            let dj = self.reduced_cost(j, pi);
            if !self.at_upper[j] && dj < -opt_tol {
                return false;
            }
            if self.at_upper[j] && dj > opt_tol {
                return false;
            }
        }
        true
    }

    /// Restore dual feasibility by flipping each dual-infeasible nonbasic to
    /// its OPPOSITE finite bound (the bound-flipping step of the BFRT ratio
    /// test, Fourer 1994 / Maros): a variable's reduced-cost sign requirement
    /// is determined by WHICH bound it sits at, so relabeling at_upper
    /// reverses the requirement without touching the basis, the objective, or
    /// the LP itself -- the point moves along the edge, the duals stay valid.
    ///
    /// This repairs the exact state that otherwise deadlocks the dual ratio
    /// test: a warm-started basis made dual-infeasible by added cut rows
    /// (which shift pi) while a bound violation exists. The dual ratio test
    /// skips wrong-signed columns (their ratio is negative and not clampable),
    /// so it can find no entering column at all; the Phase-1 primal-style
    /// repair only engages when NO row violates its bounds, so a node holding
    /// both defects fell through to "numerical breakdown" -- measured on
    /// cvrp_n15_k4: 8-11 give-ups per solve, each followed by a ~500ms cold
    /// IPM re-solve of an LP the repaired dual simplex solves in tens of
    /// pivots.
    ///
    /// Returns true iff the basis is dual-feasible after flipping (every
    /// wrong-signed column had a finite opposite bound to flip to). The
    /// caller must refresh `xb` -- flipped columns moved by u_j - l_j.
    fn flip_to_dual_feasibility(&mut self, opt_tol: T) -> bool {
        let finite_cut = T::from_f64(1e19).expect("scalar literal");
        // Reduced costs first: reduced_cost borrows pi_buf, which must not be
        // alive while at_upper is mutated.
        let pi = self.pi_buf.clone();
        let mut flip = Vec::new();
        let mut blocked = 0usize;
        for j in 0..self.n {
            if self.in_basis[j] || self.u[j] - self.l[j] < self.tol {
                continue;
            }
            let dj = self.reduced_cost(j, &pi);
            let wrong_signed =
                (!self.at_upper[j] && dj < -opt_tol) || (self.at_upper[j] && dj > opt_tol);
            if !wrong_signed {
                continue;
            }
            // Both bounds finite is the flip precondition; node LPs are boxed
            // but stay defensive against a half-infinite column.
            if self.l[j] < -finite_cut || self.u[j] > finite_cut {
                blocked += 1;
                continue;
            }
            flip.push(j);
        }
        for j in flip {
            self.at_upper[j] = !self.at_upper[j];
        }
        blocked == 0 && self.is_dual_feasible(&self.pi_buf, opt_tol)
    }

    /// Leaving-row selection rule for the Phase-1 (primal-style) ratio test.
    ///
    /// Outside Bland's rule, the strict minimum ratio wins (Harris-style
    /// tie-breaking on the alpha magnitude is not used here). Under Bland's
    /// rule the leaving choice must be the smallest basic-variable index
    /// among the min-ratio band: the anti-cycling argument of Bland's rule
    /// is about variable indices, while row positions in the basis are
    /// arbitrary, so breaking an exact tie by scan order provides no
    /// termination guarantee. `bland_tie` mirrors the main loop's `tie`
    /// constant (1e-12). Extracted as an associated fn so the rule is
    /// unit-testable; `dual_loop`'s Phase-1 branch is the only caller.
    fn bland_leaving_prefer(
        use_blands_rule: bool,
        ratio: T,
        min_ratio: T,
        bj: usize,
        leave_bj: usize,
        bland_tie: T,
    ) -> bool {
        if use_blands_rule {
            ratio < min_ratio - bland_tie || (ratio <= min_ratio + bland_tie && bj < leave_bj)
        } else {
            ratio < min_ratio
        }
    }

    fn extract_x(&self) -> Vec<T> {
        let mut x = vec![T::zero(); self.n];
        for i in 0..self.m {
            x[self.basis[i]] = self.xb[i];
        }
        for (j, xj) in x.iter_mut().enumerate() {
            if !self.in_basis[j] {
                *xj = if self.at_upper[j] {
                    self.u[j]
                } else {
                    self.l[j]
                };
            }
        }
        x
    }

    fn compute_obj(&self) -> T {
        let x = self.extract_x();
        self.c
            .iter()
            .zip(&x)
            .fold(T::zero(), |acc, (&cj, &xj)| acc + cj * xj)
    }

    /// Dual-feasibility "crash": for the initial all-slack basis, pick each
    /// nonbasic variable's starting bound (lower vs upper) by the sign of its
    /// reduced cost, instead of unconditionally lower bound. Without this, a
    /// nonbasic variable with a negative reduced cost at its lower bound
    /// starts the whole basis dual-infeasible, which the ratio test's sign
    /// convention cannot resolve as an ordinary dual pivot -- the only
    /// recourse is `dual_loop`'s Phase-1 branch (a primal-style pivot to
    /// bring in the dual-infeasible variable), and on LPs where MANY
    /// structural variables start this way (any LP whose objective has mixed
    /// signs -- not a rare/adversarial case), Phase-1 fires repeatedly before
    /// the first real dual pivot, and was observed to occasionally
    /// misconverge (see fuzz_sparse_lp_solution_is_primal_feasible /
    /// fuzz_tiny_lp_matches_brute_force_optimum's documented pre-existing
    /// gap). Crashing to a dual-feasible start when possible removes the
    /// NEED for those Phase-1 pivots in the common case, rather than
    /// asking Phase-1 to reliably recover from a start that was never
    /// dual-feasible to begin with.
    ///
    /// Only flips a variable to its upper bound when that bound is finite
    /// (below the `1e19` "effectively infinite" convention used throughout
    /// this crate) -- flipping to an unbounded upper bound would seed a huge,
    /// meaningless xb entry. Variables with a negative reduced cost and no
    /// finite upper bound are left at their lower bound (no worse than
    /// before this crash existed); `dual_loop`'s Phase-1 branch still
    /// handles any residual dual infeasibility exactly as it always has.
    fn crash_dual_feasible(&mut self) {
        let huge = T::from_f64(1e19).expect("scalar literal");
        let neg_tol = T::from_f64(-1e-10).expect("scalar literal");
        let pos_tol = T::from_f64(1e-10).expect("scalar literal");
        // Multiple passes: changing bounds changes duals, enabling further fixes.
        for _pass in 0..3 {
            self.compute_pi();
            let pi = &self.pi_buf[..self.m];
            let mut changed = false;
            for j in 0..self.n {
                if self.in_basis[j] {
                    continue;
                }
                let dj = self.reduced_cost(j, pi);
                if dj < neg_tol && self.u[j] < huge && !self.at_upper[j] {
                    self.at_upper[j] = true;
                    changed = true;
                } else if dj > pos_tol && self.l[j] > -huge && self.at_upper[j] {
                    self.at_upper[j] = false;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    pub fn cold_solve(&mut self) -> Solution<T> {
        // "Cold" means from the all-slack basis. `factor()` factors whatever basis is
        // currently loaded, so without this reset a `cold_solve` after a failed warm start
        // merely refactors the basis that failed -- a warm start wearing the wrong name,
        // and useless as a retry. On a freshly constructed solver this is a no-op.
        let slack_start = self.n - self.m;
        for i in 0..self.m {
            self.basis[i] = slack_start + i;
        }
        for f in self.in_basis.iter_mut() {
            *f = false;
        }
        for i in 0..self.m {
            let bi = self.basis[i];
            self.in_basis[bi] = true;
        }
        self.n_etas = 0;
        self.factor();
        self.crash_dual_feasible();
        self.xb = self.compute_xb();
        self.dual_loop()
    }

    pub fn hot_solve(&mut self, parent: &HotBasis<T>) -> Solution<T> {
        self.basis.copy_from_slice(&parent.basic);
        self.at_upper.copy_from_slice(&parent.at_upper);
        // Rebuild in_basis from parent basis
        for j in 0..self.n {
            self.in_basis[j] = false;
        }
        for &b in &self.basis {
            self.in_basis[b] = true;
        }
        self.iters = 0;

        // Inherit the parent's LU factorization unconditionally: the
        // inherited factorization is applied via forward/backward triangular solves.
        // With MAX_ETAS=100, the inherited factor is never more than 100
        // updates old — always fresh enough for reliable pivoting.
        // The existing refactor trigger (n_etas >= MAX_ETAS) and ratio-test
        // recovery handle the rare case of a degraded inherited factor.
        //
        // A sparse-path parent exports NO dense factor at all (see
        // `export_basis`: the dense array holds stale data there), so the
        // hand-off is "sparse present" rather than "dense non-empty".
        let reuse_ok = !parent.lu.is_empty() || parent.sparse.is_some();
        if reuse_ok {
            // Inherit parent's LU and PFI chain for warm re-optimization.
            // When the parent factored sparsely, the dense `lu` copy is
            // skipped (stale data — see `sparse_lu`): this solver inherits
            // the packed factor itself and routes every base solve through
            // it until its own next `factor_once`.
            if parent.sparse.is_none() {
                debug_assert!(!parent.lu.is_empty());
                self.ensure_lu();
                self.lu.copy_from_slice(&parent.lu);
            }
            self.perm.copy_from_slice(&parent.perm);
            if parent.swaps.len() == self.m {
                self.swaps.copy_from_slice(&parent.swaps);
            } else {
                // A stale/foreign parent (e.g. a hand-path basis carried by
                // a BLAS-path child): the swap sequence is unknown, so the
                // mirror cannot be reconstructed consistently — refactor.
                self.sparse_lu = None;
                self.factor();
                self.xb = self.compute_xb();
                return self.dual_loop();
            }
            self.n_etas = parent.n_etas;
            self.eta_p[..parent.n_etas].copy_from_slice(&parent.eta_p[..parent.n_etas]);
            let m = self.m;
            self.eta_alpha[..parent.n_etas * m]
                .copy_from_slice(&parent.eta_alpha[..parent.n_etas * m]);
            // Sparse mirror: take the parent's flattened lists and shift the
            // per-eta offsets to local origin 0 (the child appends after them).
            self.eta_idx.clear();
            self.eta_idx.extend_from_slice(&parent.eta_idx);
            self.eta_val.clear();
            self.eta_val.extend_from_slice(&parent.eta_val);
            for e in 0..=parent.n_etas {
                self.eta_start[e] = parent.eta_start[e];
            }
            self.sparse_lu = parent.sparse.clone();
            // Rebuild the LAPACK mirror from the inherited factor (no
            // re-factorization): convert, derive ipiv from the swap
            // sequence, rescan the diagonal for the singular flag. A sparse
            // parent has no dense factor to rebuild from — its solves go
            // through the inherited packed factor instead.
            if self.blas.is_some()
                && iconic_linalg::blas::blas_enabled()
                && parent.sparse.is_none()
            {
                self.rebuild_blas_mirror();
            }
            // Recompute xb with the inherited factorization and updated bounds
            self.xb = self.compute_xb();
        } else {
            // No (or too stale) parent LU — refactor from scratch.
            self.factor();
            self.xb = self.compute_xb();
        }
        self.dual_loop()
    }

    /// Warm re-optimization for an LP that EXTENDED the parent's constraint
    /// set. `parent` is a basis of a smaller LP (fewer rows) whose columns
    /// are all still valid columns of this solver; this LP appends
    /// `self.m - parent.basic.len()` new rows with identity slack columns at
    /// the trailing block — exactly the MIP cut rounds' shape (`build_node_lp`
    /// appends one cut row and one cut slack per cut).
    ///
    /// The parent's basic columns stay basic; the new rows' slack columns
    /// join as basic (column `n - m + r` is row `r`'s identity column, the
    /// same invariant `from_csc`'s all-slack start relies on); the basis is
    /// refactored once (the inherited LU is parent-dimension and the new rows
    /// invalidate it anyway); then the dual loop runs from the resulting
    /// primal-infeasible start. A violated cut puts its slack below zero,
    /// which is exactly the starting state dual simplex's bound-violation-
    /// driven pivoting is built for (the all-slack cold start has the same
    /// shape when rows are violated).
    pub fn hot_solve_extended(&mut self, parent: &HotBasis<T>) -> Solution<T> {
        let parent_m = parent.basic.len();
        debug_assert!(parent_m < self.m, "hot_solve_extended requires new rows");
        self.basis[..parent_m].copy_from_slice(&parent.basic);
        // `at_upper` is indexed by COLUMN (size n), unlike `basis` (size m).
        // The parent's columns keep their bound side; the new slack columns
        // default to their lower bound (they are basic, so the entry is
        // irrelevant until a pivot makes one nonbasic).
        let n_parent = parent.at_upper.len();
        debug_assert!(n_parent <= self.n);
        self.at_upper[..n_parent].copy_from_slice(&parent.at_upper);
        for j in n_parent..self.n {
            self.at_upper[j] = false;
        }
        for j in 0..self.n {
            self.in_basis[j] = false;
        }
        for r in 0..parent_m {
            self.in_basis[self.basis[r]] = true;
        }
        // Trailing identity block: column (n - m + r) is the identity column
        // of row r. n - m is constant across cut rounds because each new row
        // appends exactly one slack column.
        let slack_start = self.n - self.m;
        for r in parent_m..self.m {
            let col = slack_start + r;
            debug_assert!(col < self.n && !self.in_basis[col]);
            self.basis[r] = col;
            // The new slack columns' at_upper entries were already zeroed by
            // the `n_parent..self.n` loop above (they are exactly that
            // range); keep the explicit write for the reader.
            self.at_upper[col] = false; // slack at its lower bound (0)
            self.in_basis[col] = true;
        }
        self.iters = 0;
        self.factor();
        self.xb = self.compute_xb();
        self.dual_loop()
    }

    pub fn export_basis(&self) -> HotBasis<T> {
        // When the packed sparse factor is live, the dense `lu` array holds
        // STALE data (the sparse path never maintains it — see
        // `factor_once_sparse`). Every consumer of an inherited basis routes
        // base solves through `sparse` while it is `Some`, so the m² clone is
        // dead weight on exactly the hot path that exports a basis per node:
        // hand over an empty vector instead (the same convention `hot_solve`
        // already checks for with `!parent.lu.is_empty()`).
        let lu_out = if self.sparse_lu.is_some() {
            Vec::new()
        } else {
            self.lu.clone()
        };
        HotBasis {
            basic: self.basis.clone(),
            at_upper: self.at_upper.clone(),
            lu: lu_out,
            perm: self.perm.clone(),
            swaps: self.swaps.clone(),
            n_etas: self.n_etas,
            eta_p: self.eta_p[..self.n_etas].to_vec(),
            eta_alpha: self.eta_alpha[..self.n_etas * self.m].to_vec(),
            eta_start: {
                let mut s = self.eta_start[..self.n_etas + 1].to_vec();
                // The flattened sparse arrays are inherited wholesale; shift
                // the offsets to the child's local origin.
                let base = s[0];
                for off in s.iter_mut() {
                    *off -= base;
                }
                s
            },
            eta_idx: self.eta_idx[self.eta_start[0]..self.eta_start[self.n_etas]].to_vec(),
            eta_val: self.eta_val[self.eta_start[0]..self.eta_start[self.n_etas]].to_vec(),
            sparse: self.sparse_lu.clone(),
        }
    }

    /// One row of the current simplex tableau: the basic variable `basic_col`
    /// (a column index into the LP's A) with current value `basic_val`, and
    /// the coefficients `ā_j = wᵀ·A_j` of the nonbasic columns, so the row
    /// reads `x_basic + Σ_j ā_j·x_j = basic_val`.
    ///
    /// `w = B⁻ᵀ·e_r` is computed through the same machinery the ratio test
    /// uses: the transpose PFI eta chain (newest to oldest) followed by the
    /// base LU transpose solve. Used by the MIP layer's Gomory mixed-integer
    /// cut separator, which needs the fractional basic rows of the node LP.
    /// Replace the objective vector (same length). The basis and primal
    /// solution stay valid — only the reduced costs change — so a subsequent
    /// hot solve re-optimizes from the current basis. Used by root-level
    /// optimization-based bound tightening, which solves min/max x_j per
    /// integer variable with warm starts.
    pub fn set_objective(&mut self, c: Vec<T>) {
        debug_assert_eq!(c.len(), self.n);
        self.c = c;
    }

    /// Column index of the basic variable in tableau row `r`.
    pub fn basic_col(&self, r: usize) -> usize {
        self.basis[r]
    }

    /// Current value of the basic variable in tableau row `r`.
    pub fn basic_val(&self, r: usize) -> T {
        self.xb[r]
    }

    pub fn tableau_row(&mut self, r: usize) -> TableauRow<T> {
        let m = self.m;
        let mut w = vec![T::zero(); m];
        w[r] = T::one();
        Self::apply_etas_transpose(&self.eta_p, &self.eta_start, &self.eta_idx, &self.eta_val, &self.eta_alpha, self.n_etas, m, &mut w);
        let mut out = vec![T::zero(); m];
        Self::solve_base_trans_into(
            &w,
            &self.lu,
            &self.perm,
            &mut out,
            &mut self.lu_z,
            &mut self.spike,
            m,
            self.sparse_lu.as_ref(),
            self.blas.as_mut(),
        );
        let mut coeffs: Vec<(usize, T, bool, T)> = Vec::new();
        let zero = T::zero();
        let nz_tol = T::from_f64(1e-12).expect("scalar literal");
        for j in 0..self.n {
            if self.in_basis[j] {
                continue;
            }
            let mut acc = zero;
            for k in self.a_col_start[j]..self.a_col_start[j + 1] {
                acc += out[self.a_row_idx[k]] * self.a_val[k];
            }
            if acc.abs() > nz_tol {
                coeffs.push((j, acc, self.at_upper[j], self.u[j]));
            }
        }
        TableauRow {
            basic_col: self.basis[r],
            basic_val: self.xb[r],
            coeffs,
        }
    }

    fn dual_loop(&mut self) -> Solution<T> {
        let zero = T::zero();
        let one = T::one();
        let huge = T::from_f64(1e20).expect("scalar literal");
        // Numerical defenses against degenerate/near-singular pivots
        let alpha_tol = T::from_f64(1e-10).expect("scalar literal"); // relaxed alpha tolerance
        let harris_delta = T::from_f64(1e-7).expect("scalar literal"); // Harris ratio band
        let refine_tol = T::from_f64(1e-12).expect("scalar literal"); // iterative refinement threshold
        let opt_tol = self.tol * T::from_f64(1e6).expect("scalar literal"); // scaled optimality tolerance
                                                            // One-shot recovery: if ratio test fails, refactor and retry ONCE
        let mut ratio_recovery_attempted = false;
        // The bound-flip repair (see flip_to_dual_feasibility) is tried once
        // per factorization cycle: after a refactor re-derives pi from
        // scratch, the state that made flips useful is gone and the guard
        // must reset.
        let mut flip_attempted = false;
        // Same once-per-recovery-cycle discipline for the Phase-1 pivot
        // repair engaged when flips cannot fix the basis (infinite opposite
        // bounds). Reset alongside `flip_attempted` after each refactor.
        let mut p1_attempted = false;
        let mut nan_recovery_attempted = false;
        // Anti-cycling: see the leaving-variable selection below.
        let mut best_worst = huge;
        let mut stall_count = 0usize;
        let mut use_blands_rule = false;
        // Incremental-update bookkeeping: pi and xb are advanced per pivot by
        // the standard dual-simplex recurrences and periodically recomputed
        // exactly (every refactor, and at latest every this-many pivots) so
        // roundoff can never accumulate across more than a bounded number of
        // incremental steps. pi must also be recomputed after Phase-1
        // (primal-style) pivots, which do not have rho at hand.
        let mut pivots_since_refresh = 0usize;
        let mut pi_stale = true;
        const PI_XB_REFRESH_EVERY: usize = 64;
        const STALL_LIMIT: usize = 150;
        // The RHS perturbation fires well before Bland's latch (150): a
        // degenerate vertex is usually escapable with a tiny nudge, and the
        // perturbation's restore-on-improvement keeps the LP exact, whereas
        // Bland's slows the whole tail of the solve.
        const PERTURB_STALL_LIMIT: usize = 50;
        // Episode cap: if the perturbed RHS has not escaped the vertex within
        // this many pivots, restore and let a fresh stall re-perturb with the
        // next (different) pattern.
        const PERTURB_EPISODE_ITERS: usize = 10;
        // Escalation ladder for failed episodes. A failed episode is weak
        // evidence: it sampled ONE pattern at ONE magnitude for 10 pivots.
        // The next episode gets a larger nudge and a longer window;
        // after MAX_PERTURB_FAILURES consecutive failures the mechanism
        /// stands down (Bland's latch takes over, keeping termination).
        const PERTURB_ESCALATION: f64 = 5.0;
        const MAX_PERTURB_FAILURES: usize = 6;
        // Stall iterations before the dual-Devex weighted leaving-row rule
        // engages (plain largest-violation until then). Sits at the bottom of
        // the anti-stall ladder: devex weights -> RHS perturbation (50) ->
        // Bland's rule (150).
        const DEVEX_STALL_GATE: usize = 2;
        // Bland's rule is a *recovery* from a degenerate patch, not a permanent
        // downgrade. It used to be one-way: the first 150 non-improving iterations
        // latched it on for the rest of the solve. Degeneracy in a MIP node LP is
        // usually transient (a plateau in the max bound violation, which dual simplex
        // has no reason to decrease monotonically in the first place), so that latch
        // fires on ordinary problems and then crawls -- the root LP of stein_v10_t4
        // ran 81408 iterations into the 30s deadline instead of solving.
        //
        // So: leave Bland's as soon as the violation genuinely improves (the
        // degenerate vertex has been escaped), and re-enter on the next stall. To keep
        // the termination guarantee that made this one-way to begin with, only
        // MAX_BLAND_EPISODES escapes are allowed -- after that the latch is permanent
        // and Bland's runs continuously to the end, which is what the guarantee needs.
        let mut bland_episodes = 0usize;
        const MAX_BLAND_EPISODES: usize = 8;

        // Resolved once per process, not once per node LP: `dual_loop` runs at every
        // branch-and-bound node, and an environment lookup there is pure overhead.
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trace = *TRACE.get_or_init(|| std::env::var_os("ICONIC_SIMPLEX_TRACE").is_some());
        for _ in 0..self.max_iters {
            self.iters += 1;
            if trace && self.iters.is_multiple_of(5000) {
                let nviol = (0..self.m)
                    .filter(|&i| {
                        let bj = self.basis[i];
                        self.xb[i] < self.l[bj] - self.tol || self.xb[i] > self.u[bj] + self.tol
                    })
                    .count();
                let nan_xb = self.xb[..self.m].iter().filter(|v| !v.is_finite()).count();
                let big_xb = self.xb[..self.m]
                    .iter()
                    .filter(|v| v.is_finite() && v.abs() > T::from_f64(1e15).expect("scalar literal"))
                    .count();
                eprintln!(
                    "      [trace] it={} obj={:?} nviol={} bland={} etas={} nan_xb={} big_xb={}",
                    self.iters,
                    self.compute_obj(),
                    nviol,
                    use_blands_rule,
                    self.n_etas,
                    nan_xb,
                    big_xb
                );
            }
            // Every 256 pivots so the clock read is negligible against the pivot cost.
            if self.iters.is_multiple_of(256) {
                if let Some(d) = self.deadline {
                    if std::time::Instant::now() >= d {
                        set_fail(FailReason::Deadline);
                        self.restore_exact_rhs();
                        return Solution {
                            status: Status::IterationLimit,
                            x: self.extract_x(),
                            obj: self.compute_obj(),
                            iters: self.iters,
                            pi: Vec::new(),
                        };
                    }
                }
            }
            // A basis that `factor`'s repair could not rescue leaves zeros on U's
            // diagonal, and the triangular solves turn those into non-finite entries
            // of `xb`. Every bound test below compares against `xb[i]`, and every
            // comparison against a NaN is false, so those rows silently register as
            // *satisfied*: the solver sees a nearly-feasible basis, pivots on whichever
            // one or two rows are still finite, and never makes progress. On the root
            // LP of stein_v10_t4 that was 101 of 138 basic variables NaN from the first
            // few hundred iterations onward, spinning 81408 iterations into the 30s
            // deadline -- the whole solve budget spent on a basis that was already dead.
            //
            // The guard further down catches exactly this for `pi`/`rho` but not for
            // `xb`, which is what the leaving-variable selection actually reads. Refactor
            // once; if that does not restore it, report `IterationLimit` -- "no definitive
            // answer", which the caller already handles by falling back to the IPM -- so
            // the failure costs milliseconds instead of the entire time limit.
            if !self.xb[..self.m].iter().all(|v| v.is_finite()) {
                if nan_recovery_attempted {
                    set_fail(FailReason::DeadBasisXb);
                    self.restore_exact_rhs();
                    return Solution {
                        status: Status::IterationLimit,
                        x: self.extract_x(),
                        obj: self.compute_obj(),
                        iters: self.iters,
                        pi: Vec::new(),
                    };
                }
                nan_recovery_attempted = true;
                self.factor();
                self.xb = self.compute_xb();
                if !self.xb[..self.m].iter().all(|v| v.is_finite()) {
                    set_fail(FailReason::DeadBasisXb);
                    self.restore_exact_rhs();
                    return Solution {
                        status: Status::IterationLimit,
                        x: self.extract_x(),
                        obj: self.compute_obj(),
                        iters: self.iters,
                        pi: Vec::new(),
                    };
                }
            }

            // ── Phase I: find leaving variable (bound violation in xb) ──
            //
            // Largest-violation selection by default: fastest on the well-conditioned
            // majority. It has no cycling protection on its own, though, so a degenerate
            // LP can pivot among equally-violated rows indefinitely -- the comment that
            // used to sit here claimed "degeneracy is handled by proactive cost
            // perturbation", but no perturbation code exists anywhere in this file. It was
            // removed and the claim was left behind, so nothing has guaranteed termination
            // since.
            //
            // That is not hypothetical: any change perturbing the starting basis can hit
            // it. Extending a parent basis over newly added cut rows -- a routine warm-start
            // technique, and mathematically sound here (the extended basis is block
            // triangular and nonsingular) -- sent `pseudo_cost_branching_...` into a cycle
            // that only the 3600s default deadline would have ended.
            //
            // So: track the best (smallest) worst-violation seen, and after STALL_LIMIT
            // iterations without improvement switch to Bland's rule, choosing the smallest
            // basic-variable index. Bland's provably terminates. It is slower than
            // largest-violation, which is exactly why it is a fallback and not the default
            // -- on a problem that never stalls it never engages.
            //
            // Deliberately *without* the bound perturbation that accompanied this before:
            // that shifted `self.b` permanently, with no clean-up pass to undo it, which
            // silently changes the LP whose objective the branch-and-bound tree prunes on.
            let mut p_opt = None;
            let mut worst = zero;
            // Dual-Devex selection score (largest viol²/weight), used only
            // while stalling (see DEVEX_STALL_GATE below).
            let mut best_score = zero;
            let mut best_viol_sel = zero;
            // Engage the weighted rule only after two consecutive
            // non-improving iterations: one flat iteration is ordinary noise,
            // two is a degenerate plateau. Weight updates always run so the
            // framework is warm the moment the gate opens.
            let use_devex_sel =
                self.devex_enabled && !use_blands_rule && stall_count >= DEVEX_STALL_GATE;
            if use_blands_rule {
                let mut best_bj = usize::MAX;
                for i in 0..self.m {
                    let bj = self.basis[i];
                    let viol = if self.xb[i] < self.l[bj] - self.tol {
                        self.l[bj] - self.xb[i]
                    } else if self.xb[i] > self.u[bj] + self.tol {
                        self.xb[i] - self.u[bj]
                    } else {
                        continue;
                    };
                    if viol > worst {
                        worst = viol;
                    }
                    if bj < best_bj {
                        best_bj = bj;
                        p_opt = Some(i);
                    }
                }
                // Escaped the degenerate patch — go back to largest-violation.
                if worst < best_worst - self.tol {
                    best_worst = worst;
                    stall_count = 0;
                    if bland_episodes < MAX_BLAND_EPISODES {
                        use_blands_rule = false;
                    }
                }
            } else {
                for i in 0..self.m {
                    let bj = self.basis[i];
                    let viol = if self.xb[i] < self.l[bj] - self.tol {
                        self.l[bj] - self.xb[i]
                    } else if self.xb[i] > self.u[bj] + self.tol {
                        self.xb[i] - self.u[bj]
                    } else {
                        continue;
                    };
                    if viol > worst {
                        worst = viol;
                    }
                    // Dual-Devex leaving choice (Forrest–Goldfarb 1992), gated
                    // to degenerate plateaus: while the worst violation is
                    // making progress, plain largest-violation selection is
                    // kept bit-identical (an unconditional weight-based rule
                    // reshuffles every MIP trajectory — measured 3-30x work
                    // regressions on families that solve cleanly without it).
                    // Once stalling, prefer the row whose violation is large
                    // relative to its reference-framework weight — a proxy for
                    // how much the basis inverse row grows by pivoting on it.
                    // Selection only; dual feasibility comes from the entering
                    // column's true-ratio test, which is untouched. Progress
                    // (stall_count reset) restores plain selection.
                    if use_devex_sel {
                        let score = viol * viol / self.devex_row[i];
                        if p_opt.is_none() || score > best_score {
                            best_score = score;
                            p_opt = Some(i);
                        }
                    } else if p_opt.is_none() || viol > best_viol_sel {
                        best_viol_sel = viol;
                        p_opt = Some(i);
                    }
                }
                if worst < best_worst - self.tol {
                    best_worst = worst;
                    stall_count = 0;
                } else {
                    stall_count += 1;
                    if stall_count > STALL_LIMIT {
                        use_blands_rule = true;
                        bland_episodes += 1;
                    }
                    // Stall-triggered RHS perturbation: break the degenerate
                    // vertex with a tiny RHS nudge instead of immediately
                    // latching Bland's. Only when no perturbation is active and
                    // Bland's has not latched (Bland's already guarantees
                    // termination — the perturbation is the cheaper escape).
                    if !use_blands_rule
                        && self.perturb_on_stall
                        && !self.perturb_active
                        && self.perturb_failures < MAX_PERTURB_FAILURES
                        && stall_count == PERTURB_STALL_LIMIT
                    {
                        self.apply_rhs_perturbation(best_worst);
                        stall_count = 0;
                    }
                }
            }
            // Perturbation escape: the perturbed RHS did its job once the worst
            // violation improves past the pre-perturbation baseline (the
            // degenerate vertex was escaped), or the episode budget is spent.
            // Restore the exact RHS and re-evaluate the restored basis from the
            // top — every subsequent decision is on the true LP again.
            if self.perturb_active {
                self.perturb_iters += 1;
                let escaped = worst < self.perturb_baseline - self.tol;
                if escaped || self.perturb_iters >= PERTURB_EPISODE_ITERS * (1 + self.perturb_failures) {
                    if !escaped {
                        // The episode did not escape the stall: escalate —
                        // the next episode gets a larger nudge and a longer
                        // window. After enough consecutive failures stand
                        // down so the Bland latch (which the perturbation's
                        // stall-counter resets would otherwise delay
                        // indefinitely) can take over.
                        self.perturb_failures += 1;
                        self.perturb_scale *= PERTURB_ESCALATION;
                    } else {
                        self.perturb_failures = 0;
                        self.perturb_scale = 1.0;
                    }
                    self.restore_exact_rhs();
                    stall_count = 0;
                    continue;
                }
            }
            let p = match p_opt {
                Some(p) => p,
                None => {
                    // A bound-violation-free basis under the perturbed RHS is not a
                    // verdict on the true LP: restore the exact RHS and re-evaluate
                    // before any optimality/infeasibility conclusion.
                    if self.perturb_active {
                        self.restore_exact_rhs();
                        continue;
                    }
                    // No bound violation → check optimality.
                    // If dual-infeasible, perform a primal-simplex pivot to bring
                    // a dual-infeasible nonbasic into the basis (Phase 1 dual simplex).
                    self.compute_pi();
                    // Find the most dual-infeasible nonbasic variable
                    let pi = &self.pi_buf[..self.m];
                    let mut enter_j: Option<usize> = None;
                    let mut worst_viol = zero;
                    for j in 0..self.n {
                        if self.in_basis[j] {
                            continue;
                        }
                        // Same exclusion as the dual ratio test: a fixed column cannot move,
                        // so it is dual-feasible at any reduced cost. Entering one here gives
                        // a zero-length primal step, and since the basis is otherwise
                        // unchanged the next iteration selects it again -- a non-terminating
                        // loop with no bound violation to drive progress.
                        if self.u[j] - self.l[j] < self.tol {
                            continue;
                        }
                        let dj = self.reduced_cost(j, pi);
                        let viol = if !self.at_upper[j] && dj < -opt_tol {
                            -dj
                        } else if self.at_upper[j] && dj > opt_tol {
                            dj
                        } else {
                            continue;
                        };
                        if viol > worst_viol {
                            worst_viol = viol;
                            enter_j = Some(j);
                        }
                    }
                    let j = match enter_j {
                        Some(j) => j,
                        None => {
                            // Only optimal if the point is real. Every test that would have
                            // found a violation here -- both the bound checks on `xb` and the
                            // reduced-cost scan above -- compares against a value that may be
                            // NaN, and a NaN comparison is false, so a corrupt basis reports
                            // "nothing violated" and reaches this arm. Declaring optimality on
                            // it hands back a solution vector full of NaNs.
                            if !self.xb[..self.m].iter().all(|v| v.is_finite()) {
                                set_fail(FailReason::Phase1NoEntering);
                                self.restore_exact_rhs();
                                return Solution {
                                    status: Status::IterationLimit,
                                    x: self.extract_x(),
                                    obj: self.compute_obj(),
                                    iters: self.iters,
                                    pi: Vec::new(),
                                };
                            }
                            self.restore_exact_rhs();
                            return Solution {
                                status: Status::Optimal,
                                x: self.extract_x(),
                                obj: self.compute_obj(),
                                iters: self.iters,
                                pi: self.pi_buf[..self.m].to_vec(),
                            };
                        }
                    };
                    // Primal simplex pivot: j enters the basis. The pivot
                    // mechanics (FTRAN, four-case minimum-ratio leaving test,
                    // PFI eta update, bounded-refresh primal update) are the
                    // shared `phase1_pivot` method -- also engaged from the
                    // breakdown path below, where the same repair must run
                    // while bound violations still exist.
                    if !self.phase1_pivot(j, use_blands_rule) {
                        // "No basic variable limits the entering variable's movement"
                        // is not a valid primal-infeasibility certificate: that would
                        // require a dual-feasible basis (the dual-simplex Farkas
                        // certificate), and this Phase-1 branch is entered precisely
                        // because the basis is dual-INfeasible -- the entering
                        // variable was selected for its wrong-signed reduced cost.
                        // With nothing stopping it, the variable either travels to
                        // its own other bound or the objective is unbounded; neither
                        // conclusion is primal infeasibility. Report IterationLimit
                        // ("no definitive answer") instead of a false proof.
                        set_fail(FailReason::Phase1NoLeaving);
                        self.restore_exact_rhs();
                        return Solution {
                            status: Status::IterationLimit,
                            x: self.extract_x(),
                            obj: self.compute_obj(),
                            iters: self.iters,
                            pi: Vec::new(),
                        };
                    }
                    // No rho at hand in this branch: the next iteration's pi
                    // recompute is one BTRAN, exactly what this branch always paid.
                    pi_stale = true;
                    continue;
                }
            };

            let lv = self.basis[p];
            let viol_dir = if self.xb[p] < self.l[lv] { one } else { -one };

            // ── Compute pi (simplex multipliers, stored in self.pi_buf) ──
            // Exact recompute only on demand; the common path inherits pi
            // from the previous iteration's incremental update.
            if pi_stale || pivots_since_refresh >= PI_XB_REFRESH_EVERY {
                self.compute_pi();
            }

            // ── Compute rho = B^{-T} e_p with iterative refinement ──
            // Refining every transpose solve: one step cuts error by ~4 orders
            for i in 0..self.m {
                self.rho_ep[i] = zero;
            }
            self.rho_ep[p] = one;
            // Bypasses the lu_solve_trans wrapper to write straight into lu_pi (no
            // Vec allocation), so the eta-transpose must be applied here directly.
            Self::apply_etas_transpose(
                &self.eta_p,
                &self.eta_start,
                &self.eta_idx,
                &self.eta_val,
                &self.eta_alpha,
                self.n_etas,
                self.m,
                &mut self.rho_ep,
            );
            let m = self.m;
            Self::solve_base_trans_into(
                &self.rho_ep,
                &self.lu,
                &self.perm,
                &mut self.lu_y,
                &mut self.lu_z,
                &mut self.spike,
                m,
                self.sparse_lu.as_ref(),
                self.blas.as_mut(),
            );
            self.lu_pi[..m].copy_from_slice(&self.lu_y[..m]);
            // Refine rho: residual r = e_p - B^T * rho, solve B^T * drho = r.
            // Uses the pre-allocated `bt_rho` field (not a fresh Vec) -- this runs at
            // least once per dual-simplex iteration, called from every B&B node, so a
            // per-iteration heap allocation here is pure hot-path waste.
            for _ in 0..2 {
                // (B^T rho)[j] = sum_i B[i][j] * rho[i] = sum_i A[i][basis[j]] * rho[i]
                // CSC sparse: O(nnz_basis_cols) instead of O(m²)
                for j in 0..self.m {
                    let col = self.basis[j];
                    let mut acc = zero;
                    let start = self.a_col_start[col];
                    let end = self.a_col_start[col + 1];
                    for idx in start..end {
                        acc += self.a_val[idx] * self.lu_pi[self.a_row_idx[idx]];
                    }
                    self.bt_rho[j] = acc;
                }
                let mut r_norm = zero;
                for j in 0..self.m {
                    let rj = if j == p { one } else { zero } - self.bt_rho[j];
                    self.bt_rho[j] = rj; // reuse buffer
                    if rj.abs() > r_norm {
                        r_norm = rj.abs();
                    }
                }
                let rho_norm = (0..self.m).fold(zero, |acc, j| {
                    let v = self.lu_pi[j].abs();
                    if v > acc {
                        v
                    } else {
                        acc
                    }
                });
                if r_norm < refine_tol * (one + rho_norm) {
                    break;
                }
                // Solve correction: B^T * drho = r  →  drho = B^{-T} r.
                // Disjoint-field borrow: bt_rho (read) vs lu_z/lu_y/rho_ep (written) are
                // separate struct fields passed as separate arguments (not through a
                // &mut self receiver), so the borrow checker accepts this without a clone.
                Self::apply_etas_transpose(
                    &self.eta_p,
                    &self.eta_start,
                    &self.eta_idx,
                    &self.eta_val,
                    &self.eta_alpha,
                    self.n_etas,
                    self.m,
                    &mut self.bt_rho,
                );
                let m = self.m;
                Self::solve_base_trans_into(
                    &self.bt_rho,
                    &self.lu,
                    &self.perm,
                    &mut self.lu_y,
                    &mut self.lu_z,
                    &mut self.spike,
                    m,
                    self.sparse_lu.as_ref(),
                    self.blas.as_mut(),
                );
                for j in 0..m {
                    self.lu_pi[j] = self.lu_pi[j] + self.lu_y[j];
                }
            }
            // A singular basis makes the transposed solves divide by a ~zero pivot and
            // return non-finite `pi`/`rho`. Every downstream comparison against a NaN
            // alpha is false, so the pricing loop below rejects *every* column -- by the
            // sign test, in both directions at once, which is impossible for a finite
            // alpha -- and the ratio test concludes "no entering column" and returns
            // `Infeasible`. That is reported as a proof of infeasibility when it is a
            // numerical breakdown, and branch-and-bound then prunes a feasible subtree.
            //
            // Observed on a 50x81 job-shop node LP that HiGHS solves to optimality
            // (obj 22.327590): all 50 entries of both `pi` and `rho` were NaN while
            // `xb` and `c` were clean, and the solver reported Infeasible after 120
            // iterations.
            //
            // Refactor once -- an accumulated eta chain can be the whole problem -- and
            // if the basis is still singular report `IterationLimit`, which callers
            // already treat as "no definitive answer" rather than as proof.
            if !self.lu_pi[..self.m].iter().all(|v| v.is_finite())
                || !self.pi_buf[..self.m].iter().all(|v| v.is_finite())
            {
                if nan_recovery_attempted {
                    set_fail(FailReason::SingularPi);
                    self.restore_exact_rhs();
                    return Solution {
                        status: Status::IterationLimit,
                        x: self.extract_x(),
                        obj: self.compute_obj(),
                        iters: self.iters,
                        pi: Vec::new(),
                    };
                }
                nan_recovery_attempted = true;
                self.factor();
                self.xb = self.compute_xb();
                continue;
            }
            // Borrow instead of clone: reduced_cost_and_alpha takes &self plus these
            // slices, and lu_pi/pi_buf aren't mutated again before the pricing loop.
            let rho: &[T] = &self.lu_pi[..self.m];
            let pi: &[T] = &self.pi_buf[..self.m];

            // ── Harris two-pass ratio test ──
            // Price on the TRUE ratio d_j/|alpha_jp|: the dual-simplex entering
            // column must minimize the true ratio to preserve dual feasibility of
            // the next basis. The reduced costs of the other nonbasics transform
            // as d'_j = d_j - (alpha_jq/alpha_pq) d_q, which stays correctly
            // signed for every j exactly when q minimizes d_j/|alpha_jp| among the
            // eligible columns. (A DSE-scaled ratio d_j/(sqrt(w_j)|alpha_jp|)
            // would reorder the candidates and can leave the next basis
            // dual-infeasible -- which then voids the "no entering column"
            // Infeasible certificate downstream, since the basis must be
            // dual-feasible for it to certify anything. DSE weights were
            // therefore never wired into this pricing.)
            let mut best_ratio = huge;
            let mut best_alpha_abs = zero;
            let mut q_opt: Option<usize> = None;
            // Winner's reduced cost and SIGNED pivot-row entry, kept for the
            // incremental dual/pi update after the pivot (theta_D = d_q/alpha_pq).
            let mut best_dj = zero;
            let mut best_alpha = zero;

            for j in 0..self.n {
                if self.in_basis[j] {
                    continue;
                }
                // A fixed column (l == u) has no room to move, so it is dual-feasible
                // whatever its reduced cost and can never be a useful entering variable.
                // Selecting one yields a pivot whose primal step the variable cannot
                // take.
                if self.u[j] - self.l[j] < self.tol {
                    continue;
                }
                let (dj, alpha) = self.reduced_cost_and_alpha(j, pi, rho);
                if alpha.abs() < alpha_tol {
                    continue;
                }
                let ratio = if viol_dir > zero {
                    if !self.at_upper[j] && alpha < -alpha_tol {
                        dj / (-alpha)
                    } else if self.at_upper[j] && alpha > alpha_tol {
                        -dj / alpha
                    } else {
                        continue;
                    }
                } else {
                    if !self.at_upper[j] && alpha > alpha_tol {
                        dj / alpha
                    } else if self.at_upper[j] && alpha < -alpha_tol {
                        -dj / (-alpha)
                    } else {
                        continue;
                    }
                };
                // A negative ratio means `dj` carries the wrong sign for its bound. When
                // it does so by no more than `opt_tol` that is roundoff, not dual
                // infeasibility: `is_dual_feasible` accepts such a basis, and the Phase-1
                // pass above leaves those columns alone (it acts only past `-opt_tol`).
                // Dropping them here made the ratio test stricter than every other test in
                // the loop, so a basis all of them call dual-feasible could still yield
                // "no entering column" -- which the caller reads as breakdown, or, when
                // the certificate check passes, as a proof of infeasibility. Clamping to
                // zero puts the column where a zero reduced cost belongs: at the minimum.
                let ratio = if ratio < zero {
                    if dj.abs() <= opt_tol {
                        zero
                    } else {
                        continue;
                    }
                } else {
                    ratio
                };
                let alpha_abs = alpha.abs();
                // Harris selection: strictly better ratio, or same-band max-pivot tiebreak
                // Bland's rule needs the smallest index on BOTH the leaving row and the
                // entering column. It was applied to the row only, which does not
                // terminate: mdk_n30_k3 repeated a three-objective orbit
                // (5.138036187255768 -> 4.88420400442798 -> 5.220696956476742) for all
                // 8e6 iterations of the safety net, with the Bland's latch already on.
                // Columns are scanned in increasing j, so among those attaining the
                // minimum ratio the first one *is* the smallest index -- dropping the
                // largest-pivot tie-break is exactly Bland's tie-break.
                let tie = if use_blands_rule {
                    T::from_f64(1e-12).expect("scalar literal")
                } else {
                    harris_delta
                };
                if ratio < best_ratio - tie {
                    best_ratio = ratio;
                    best_alpha_abs = alpha_abs;
                    best_dj = dj;
                    best_alpha = alpha;
                    q_opt = Some(j);
                } else if !use_blands_rule
                    && ratio < best_ratio + harris_delta
                    && alpha_abs > best_alpha_abs
                {
                    best_alpha_abs = alpha_abs;
                    best_dj = dj;
                    best_alpha = alpha;
                    q_opt = Some(j);
                }
            }

            let q = match q_opt {
                Some(q) => q,
                None => {
                    // Ratio test failed — likely LU degradation.
                    // Setback recovery: refactor and retry once before declaring infeasibility.
                    if ratio_recovery_attempted {
                        // "A primal-infeasible row with no entering column" is only
                        // a valid infeasibility certificate when the current basis
                        // is dual-feasible (the dual-simplex Farkas certificate).
                        // After Phase-1 (primal-style) pivots -- which restore dual
                        // feasibility for the entering variable but not for every
                        // nonbasic -- the basis need not be dual-feasible, and then
                        // a failed ratio test is numerical breakdown, not a proof:
                        // the 50x81 job-shop node LP that previously reported false
                        // Infeasible (HiGHS solves it to optimality, obj 22.327590)
                        // reached exactly this state. Verify the certificate by
                        // re-deriving pi and checking every nonbasic reduced cost;
                        // without it, report IterationLimit ("no definitive
                        // answer"), which callers already treat as such rather than
                        // as proof.
                        self.compute_pi();
                        let status = if self.is_dual_feasible(&self.pi_buf[..self.m], opt_tol) {
                            set_fail(FailReason::None);
                            Status::Infeasible
                        } else if !flip_attempted && {
                            flip_attempted = true;
                            self.flip_to_dual_feasibility(opt_tol)
                        } {
                            // Once per solve: a second empty ratio test on an
                            // already-flipped basis means alpha filtering (not
                            // dual infeasibility) is excluding everything, and
                            // re-flipping is a no-op -- retrying would loop.
                            // The ratio test's empty result was a dual-infeasible
                            // basis (wrong-signed columns are skipped as negative-
                            // ratio), not LU degradation. Flipping those columns to
                            // their opposite bounds restores dual feasibility without
                            // touching the basis -- resume the dual loop on the same
                            // leaving row (the bound violation is untouched by the
                            // flips' reduced costs). xb must be recomputed: flipped
                            // columns moved by u_j - l_j.
                            self.xb = self.compute_xb();
                            continue;
                        } else if !p1_attempted && {
                            // Bound-flip could not repair (the wrong-signed
                            // columns sit at INFINITE opposite bounds -- cut
                            // slacks [0, inf)). The remaining sound repair is
                            // a primal-style pivot bringing one offending
                            // column into the basis: dual feasibility is
                            // restored by construction and the loop's own
                            // leaving-row logic takes over from there. One
                            // attempt per recovery cycle; a pivot that finds
                            // no leaving row means genuine breakdown.
                            p1_attempted = true;
                            let pi_snap = self.pi_buf.clone();
                            let mut worst_j: Option<usize> = None;
                            let mut worst_v = zero;
                            for jj in 0..self.n {
                                if self.in_basis[jj] || self.u[jj] - self.l[jj] < self.tol {
                                    continue;
                                }
                                let dj = self.reduced_cost(jj, &pi_snap);
                                let v = (!self.at_upper[jj]
                                    && dj < -opt_tol)
                                    .then(|| -dj)
                                    .or_else(|| {
                                        (self.at_upper[jj] && dj > opt_tol).then_some(dj)
                                    });
                                if let Some(v) = v {
                                    if v > worst_v {
                                        worst_v = v;
                                        worst_j = Some(jj);
                                    }
                                }
                            }
                            match worst_j {
                                Some(jj) => self.phase1_pivot(jj, false) && {
                                    self.xb = self.compute_xb();
                                    true
                                },
                                None => false,
                            }
                        } {
                            continue;
                        } else {
                            set_fail(FailReason::RatioBreakdown);
                            Status::IterationLimit
                        };
                        // The Infeasible return now carries the basis's dual
                        // variables: they form the Farkas certificate that the
                        // MIP layer verifies exactly (A^T z against the box),
                        // which lets a verified verdict prune without any
                        // re-solve. The IterationLimit return keeps pi empty
                        // (no certificate exists).
                        let pi_out = if status == Status::Infeasible {
                            self.pi_buf[..self.m].to_vec()
                        } else {
                            Vec::new()
                        };
                        self.restore_exact_rhs();
                        return Solution {
                            status,
                            x: self.extract_x(),
                            obj: self.compute_obj(),
                            iters: self.iters,
                            pi: pi_out,
                        };
                    }
                    ratio_recovery_attempted = true;
                    self.factor();
                    self.xb = self.compute_xb();
                    // A refactor re-derives everything: both repairs are fair game again.
                    flip_attempted = false;
                    p1_attempted = false;
                    continue;
                }
            };
            // Successful pivot — reset recovery flag
            ratio_recovery_attempted = false;

            // ── Pivot ──
            self.at_upper[lv] = viol_dir <= zero;
            self.update_basis_entry(p, q);

            // ── Eta append ──
            // Compute alpha = B_current^{-1} a_q into lu_y: spike scatter →
            // base LU solve → PFI eta chain. This is the data commit_eta
            // below validates and stores as the next eta.
            {
                let m = self.m;
                for i in 0..m {
                    self.spike[i] = T::zero();
                }
                let start = self.a_col_start[q];
                let end = self.a_col_start[q + 1];
                for idx in start..end {
                    self.spike[self.a_row_idx[idx]] = self.a_val[idx];
                }
                let m = self.m;
            Self::solve_base_into(
                    &self.spike,
                    &self.lu,
                    &self.perm,
                    &mut self.lu_y,
                    m,
                    &mut self.lu_z,
                    self.sparse_lu.as_ref(),
                    self.blas.as_mut(),
                );
                Self::apply_etas_forward(
                    &self.eta_p,
                    &self.eta_start,
                    &self.eta_idx,
                    &self.eta_val,
                    &self.eta_alpha,
                    self.n_etas,
                    m,
                    &mut self.lu_y,
                );
            }
            // ── Dual-Devex weight update ──
            // `lu_y` holds beta = B^{-1} a_q for the pre-pivot basis (the FTRAN
            // above ran before this pivot's eta was appended), so the
            // Forrest–Goldfarb dual reference-framework recurrences apply
            // directly: rows grow at most by (beta_i/beta_p)^2 * w_q, the
            // entering variable's basic weight is w_q/beta_p^2, and the
            // leaving variable's nonbasic weight is w_p_old/beta_p^2. All
            // floored at 1 (a unit-weight reference framework). Weights above
            // DEVEX_RESET lose their scaling meaning entirely, so the whole
            // framework restarts at 1 — standard Devex practice.
            if self.devex_enabled {
                const DEVEX_RESET: f64 = 1e8;
                let bp = self.lu_y[p];
                let bp_abs = bp.to_f64().unwrap_or(f64::NAN);
                if bp_abs.is_finite() && bp_abs > 1e-12 {
                    let one = T::one();
                    let wq = self.devex_col[q];
                    let w_p_old = self.devex_row[p];
                    // Growth of the surviving rows scales with the OLD pivot-row
                    // weight (B~^-1_i = B^-1_i - (beta_i/beta_p) B^-1_p); the new
                    // row-p weight is where the entering variable's own weight
                    // enters. Swapping the two is a mis-scaled framework that
                    // over-/under-weights rows depending on wq vs w_p_old.
                    let sq = w_p_old / (bp * bp);
                    let sq_enter = wq / (bp * bp);
                    let mut finite = true;
                    let mut wmax: f64 = 0.0;
                    for i in 0..self.m {
                        let t = self.lu_y[i];
                        let tf = t.to_f64().unwrap_or(f64::NAN);
                        if !tf.is_finite() {
                            finite = false;
                            break;
                        }
                        if i != p {
                            let cand = t * t * sq;
                            if cand > self.devex_row[i] {
                                self.devex_row[i] = cand;
                            }
                        }
                        let wf = self.devex_row[i].to_f64().unwrap_or(0.0);
                        if wf > wmax {
                            wmax = wf;
                        }
                    }
                    if finite {
                        let new_wp = if sq_enter > one { sq_enter } else { one };
                        self.devex_row[p] = new_wp;
                        self.devex_col[lv] = if sq > one { sq } else { one };
                        if new_wp.to_f64().unwrap_or(0.0) > wmax {
                            wmax = new_wp.to_f64().unwrap_or(0.0);
                        }
                        if wmax > DEVEX_RESET {
                            self.devex_row.iter_mut().for_each(|w| *w = one);
                            self.devex_col.iter_mut().for_each(|w| *w = one);
                        }
                    }
                }
            }

            // PFI eta append (O(m) per pivot), full refactor every MAX_ETAS pivots.
            let refactored = if self.n_etas < MAX_ETAS && self.commit_eta(p) {
                self.n_etas += 1;
                false
            } else {
                self.factor();
                true
            };
            // ── Incremental xb and pi updates ──
            // Standard dual-simplex recurrences, replacing a full
            // BTRAN-with-refinement for xb and one for pi on EVERY pivot
            // (measured: those two were 2 of the ~4 O(m^2) solves per pivot).
            // theta_P = (xb[p] - bound_of_lv) / beta_p shifts the basics;
            // the entering variable takes position p at its own bound value
            // plus theta_P. theta_D = d_q / alpha_pq advances the dual
            // multipliers along rho (already in lu_pi). Exact recompute at
            // every refactorization and at latest every PI_XB_REFRESH_EVERY
            // pivots bounds the accumulated roundoff; a disagreement between
            // the FTRAN pivot element and the pricing alpha beyond a relative
            // 1e-6 (the ftran/btran consistency invariant) falls back to the
            // exact recompute on the spot.
            let bp = self.lu_y[p];
            let bp_consistent = bp.is_finite()
                && bp.abs() > T::from_f64(1e-9).expect("scalar literal")
                && (bp - best_alpha).abs()
                    <= T::from_f64(1e-6).expect("scalar literal") * (one + best_alpha.abs());
            if refactored || pivots_since_refresh >= PI_XB_REFRESH_EVERY || !bp_consistent {
                pivots_since_refresh = 0;
                self.xb = self.compute_xb();
                self.compute_pi();
                pi_stale = false;
            } else {
                pivots_since_refresh += 1;
                // Leaving variable's target bound: it was violating l or u.
                let b_lv = if viol_dir > zero { self.l[lv] } else { self.u[lv] };
                let theta_p = (self.xb[p] - b_lv) / bp;
                if theta_p.is_finite() {
                    for i in 0..self.m {
                        self.xb[i] = self.xb[i] - theta_p * self.lu_y[i];
                    }
                    let x_q_bound = if self.at_upper[q] { self.u[q] } else { self.l[q] };
                    self.xb[p] = x_q_bound + theta_p;
                    // Dual multipliers along rho (lu_pi holds B^-T e_p for the
                    // PRE-pivot basis, which is the rho this pivot defines).
                    let theta_d = best_dj / best_alpha;
                    if theta_d.is_finite() {
                        for i in 0..self.m {
                            self.pi_buf[i] = self.pi_buf[i] + theta_d * self.lu_pi[i];
                        }
                        pi_stale = false;
                    } else {
                        pi_stale = true;
                    }
                } else {
                    pivots_since_refresh = 0;
                    self.xb = self.compute_xb();
                    pi_stale = true;
                }
            }
        }
        self.restore_exact_rhs();
        set_fail(FailReason::CapExhausted);
        Solution {
            status: Status::IterationLimit,
            x: self.extract_x(),
            obj: self.compute_obj(),
            iters: self.iters,
            pi: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Warm re-optimization across ROW ADDITIONS (the MIP cut-rounds shape):
    /// LP2 = LP1 plus one new row `x0 >= 0.4` with its identity slack. The
    /// parent's LP1 basis (the row-1 slack) is still valid; the new slack
    /// joins as basic with a negative value (the new row is violated at the
    /// parent's optimum) and the dual loop must restore feasibility to the
    /// shifted optimum — matching a cold solve exactly.
    #[test]
    fn hot_solve_extended_matches_cold_on_extended_lp() {
        // LP1: min x0 s.t. x0 <= 0.5, x0 in [0, 10].
        let a1 = vec![1.0, 1.0]; // row-major m x n: [x0, slack]
        let c1 = vec![1.0, 0.0];
        let b1 = vec![0.5];
        let l1 = vec![0.0, 0.0];
        let u1 = vec![10.0, 1e20];
        let mut s1: DualSolver<f64> = DualSolver::new(&c1, &a1, &b1, &l1, &u1, 1, 2);
        let sol1 = s1.cold_solve();
        assert!(matches!(sol1.status, Status::Optimal));
        assert!((sol1.x[0] - 0.0).abs() < 1e-6, "LP1 optimum x0=0, got {}", sol1.x[0]);
        let parent = s1.export_basis();

        // LP2: LP1 + x0 >= 0.4 (row -x0 + s2 = -0.4).
        let a2 = vec![1.0, 1.0, 0.0, -1.0, 0.0, 1.0];
        let c2 = vec![1.0, 0.0, 0.0];
        let b2 = vec![0.5, -0.4];
        let l2 = vec![0.0, 0.0, 0.0];
        let u2 = vec![10.0, 1e20, 1e20];
        let mut s2c: DualSolver<f64> = DualSolver::new(&c2, &a2, &b2, &l2, &u2, 2, 3);
        let cold = s2c.cold_solve();
        assert!(matches!(cold.status, Status::Optimal));
        assert!(
            (cold.x[0] - 0.4).abs() < 1e-6,
            "LP2 optimum x0=0.4, got {}",
            cold.x[0]
        );
        assert!((cold.obj - 0.4).abs() < 1e-6);

        let mut s2h: DualSolver<f64> = DualSolver::new(&c2, &a2, &b2, &l2, &u2, 2, 3);
        let hot = s2h.hot_solve_extended(&parent);
        assert!(
            matches!(hot.status, Status::Optimal),
            "extended warm start must reach Optimal, got {:?}",
            hot.status
        );
        assert!(
            (hot.x[0] - cold.x[0]).abs() < 1e-6,
            "extended warm start must match cold, got {} vs {}",
            hot.x[0],
            cold.x[0]
        );
        assert!(
            (hot.obj - cold.obj).abs() < 1e-6,
            "extended warm start objective must match cold"
        );
        assert!(
            hot.iters <= 10,
            "warm start should re-optimize in a handful of pivots, took {}",
            hot.iters
        );
        // The final basis must be a valid basis of the new LP's 2 rows (the
        // violated new slack pivots OUT of the basis on the way to the
        // optimum — that is the dual simplex working, not a defect).
        let b_hot = s2h.export_basis();
        assert_eq!(b_hot.basic.len(), 2, "basis must span the new LP's 2 rows");
        assert!(
            b_hot.basic.iter().all(|&c| c < 3),
            "basis columns must index the new LP's columns, got {:?}",
            b_hot.basic
        );
        assert!(
            b_hot.basic[0] != b_hot.basic[1],
            "basis must contain distinct columns, got {:?}",
            b_hot.basic
        );
    }

    #[test]
    fn cold_solve_simple() {
        let a = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let c = vec![-1.0, -2.0, 0.0, 0.0];
        let b = vec![10.0, 7.5];
        let l = vec![0.0; 4];
        let u = vec![1e20_f64; 4];
        let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, 2, 4);
        let sol = s.cold_solve();
        assert!(matches!(sol.status, Status::Optimal));
        assert!((sol.obj + 17.5).abs() < 1e-5);
    }
    #[test]
    fn hot_start_tighten_bound() {
        let a = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let c = vec![-1.0, -2.0, 0.0, 0.0];
        let b = vec![10.0, 7.5];
        let l = vec![0.0; 4];
        let uo = vec![1e20_f64; 4];
        let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &uo, 2, 4);
        let sol1 = s.cold_solve();
        assert!(matches!(sol1.status, Status::Optimal));
        let basis = s.export_basis();
        let mut ut = uo.clone();
        ut[1] = 5.0;
        let mut s2: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &ut, 2, 4);
        let sol2 = s2.hot_solve(&basis);
        assert!(matches!(sol2.status, Status::Optimal));
        assert!((sol2.obj + 15.0).abs() < 1e-5);
        assert!((sol2.x[0] - 5.0).abs() < 1e-5);
        assert!((sol2.x[1] - 5.0).abs() < 1e-5);
        assert!(sol2.iters <= 10);
    }

    /// Phase-1 (primal-style) pivot whose leaving variable hits its FINITE
    /// upper bound. The two upper-bound-hit branches of the Phase-1 ratio test
    /// used to record `leave_at_upper=false` (the fix claimed by 21150cc's
    /// message but absent from its diff), corrupting the leaving variable's
    /// bound side for all subsequent iterations.
    ///
    /// Instance: `min 0.5 x1 - 2 x2` s.t. `x1 + x2 + s1 = 5`, `x1 - x2 + s2 = 3`
    /// with `x1 in [0,4]`, `x2 in [0,inf)`, `s1 in [0,inf)`, `s2 in [0,6]`.
    /// x = 0 is feasible. x2 is dual-infeasible at its lower bound and the
    /// crash cannot flip it (its upper bound is the 1e20 sentinel), so
    /// Phase-1 enters x2; alpha = [1, -1], and the minimum ratio is s2's
    /// UPPER hit: `(u[s2] - xb[1]) / 1 = (6-3)/1 = 3 < 5` (s1's lower hit).
    /// The pivot must therefore leave s2 nonbasic at its upper bound, and the
    /// solve must reach Optimal (x1 then also enters, true optimum
    /// x = [1, 4, 0, 6], obj = -7.5). With `leave_at_upper=false` the
    /// corrupted basis (s2 "at lower" 0 with a negative reduced cost) drives
    /// the solver into a false Infeasible declaration on a feasible LP.
    #[test]
    fn phase1_upper_bound_hit_leaving_records_at_upper() {
        let a = vec![1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
        let c = vec![0.5, -2.0, 0.0, 0.0];
        let b = vec![5.0, 3.0];
        let l = vec![0.0; 4];
        let u = vec![4.0, 1e20, 1e20, 6.0];
        let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, 2, 4);
        let sol = s.cold_solve();
        assert!(
            matches!(sol.status, Status::Optimal),
            "expected Optimal, got {:?}",
            sol.status
        );
        // The leaving variable s2 must sit at its upper bound, not its lower.
        assert!(
            (sol.x[3] - 6.0).abs() < 1e-6,
            "s2 should be at its upper bound 6 after the upper-hit pivot, got {}",
            sol.x[3]
        );
        // True optimum: x = [1, 4, 0, 6] (s2 saturates its upper bound),
        // obj = 0.5*1 - 2*4 = -7.5.
        assert!(
            (sol.obj + 7.5).abs() < 1e-5,
            "expected obj -7.5, got {}",
            sol.obj
        );
        for i in 0..2 {
            let ax = a[i * 4] * sol.x[0]
                + a[i * 4 + 1] * sol.x[1]
                + a[i * 4 + 2] * sol.x[2]
                + a[i * 4 + 3] * sol.x[3];
            assert!((ax - b[i]).abs() < 1e-6, "row {i}: Ax={ax} != b={}", b[i]);
        }
        for j in 0..4 {
            assert!(
                sol.x[j] >= l[j] - 1e-8 && sol.x[j] <= u[j] + 1e-8,
                "var {j} out of bounds: {}",
                sol.x[j]
            );
        }
    }

    /// Phase-1 pivot whose entering column is numerically invisible to the
    /// ratio test (every alpha below the ratio tolerance): "no basic variable
    /// limits the movement" used to be declared Infeasible unconditionally.
    /// That is only a valid primal-infeasibility certificate from a
    /// DUAL-feasible basis, and this Phase-1 branch is reached precisely
    /// because the basis is dual-infeasible (the entering variable x2 was
    /// selected for its wrong-signed reduced cost; the crash cannot fix it
    /// because its upper bound is the 1e20 sentinel). x = 0 is feasible
    /// (slacks s = b), so Infeasible would be a false proof; the honest
    /// answer is IterationLimit ("no definitive answer").
    #[test]
    fn phase1_no_leaving_row_reports_iteration_limit_not_infeasible() {
        let a = vec![1.0, 5e-15, 1.0, 0.0, 1.0, -5e-15, 0.0, 1.0];
        let c = vec![0.5, -2.0, 0.0, 0.0];
        let b = vec![1.0, 1.0];
        let l = vec![0.0; 4];
        let u = vec![1e20; 4];
        let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, 2, 4);
        let sol = s.cold_solve();
        assert!(
            matches!(sol.status, Status::IterationLimit),
            "expected IterationLimit (basis is dual-infeasible, no certificate), got {:?}",
            sol.status
        );
    }

    /// A sub-tolerance U diagonal used to be clamped to 0.0 in the backward
    /// solve, returning a finite-but-wrong result that evaded every non-finite
    /// guard. It must surface as a non-finite sentinel (aligned with the
    /// factor-repair threshold PIVOT_TOL) so the recovery paths engage.
    #[test]
    fn bwd_solve_dense_subtolerance_pivot_yields_nonfinite() {
        let m = 2;
        // L = [[1,0],[0.5,1]], U = [[1, 0],[0, 1e-21]]: the second pivot
        // (1e-21) is below PIVOT_TOL = 1e-20, so the basis is singular.
        let lu = vec![1.0, 0.0, 0.5, 1e-21];
        let perm = vec![0usize, 1];
        let mut y = vec![2.0, 3.0];
        DualSolver::<f64>::bwd_solve_dense(&lu, &perm, &mut y, m, PIVOT_TOL);
        assert!(
            !y.iter().all(|v| v.is_finite()),
            "expected a non-finite sentinel from a sub-tolerance pivot, got {y:?}"
        );
    }

    /// Phase-1 leaving selection with an exact ratio tie: both rows tie at
    /// ratio 2, so the tie-break decides the leaving row, and the solve must
    /// complete correctly (a degenerate Phase-1 tie is the shape that used to
    /// misdeclare Infeasible -- see phase1_no_leaving_row_reports...).
    #[test]
    fn phase1_ratio_tie_solves_correctly() {
        // min 0.5 x1 - 2 x2 s.t. x1 + x2 + s1 = 2, x1 + x2 + s2 = 2
        // x1 in [0,4] (finite upper), x2 in [0,inf), slacks in [0,inf).
        // Crash cannot flip x2 (sentinel upper bound), so Phase-1 enters it;
        // alpha = [1, 1] gives both rows the exact ratio (2-0)/1 = 2.
        let a = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let c = vec![0.5, -2.0, 0.0, 0.0];
        let b = vec![2.0, 2.0];
        let l = vec![0.0; 4];
        let u = vec![4.0, 1e20, 1e20, 1e20];
        let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, 2, 4);
        let sol = s.cold_solve();
        assert!(
            matches!(sol.status, Status::Optimal),
            "expected Optimal on the tie instance, got {:?}",
            sol.status
        );
        // True optimum: x2 = 2 (both constraints saturate with s = 0),
        // obj = -4.
        assert!(
            (sol.obj + 4.0).abs() < 1e-5,
            "expected obj -4, got {}",
            sol.obj
        );
        for i in 0..2 {
            let ax = a[i * 4] * sol.x[0]
                + a[i * 4 + 1] * sol.x[1]
                + a[i * 4 + 2] * sol.x[2]
                + a[i * 4 + 3] * sol.x[3];
            assert!((ax - b[i]).abs() < 1e-6, "row {i}: Ax={ax} != b={}", b[i]);
        }
    }

    /// The Phase-1 leaving tie-break rule itself: under Bland's rule, an exact
    /// ratio tie must select the smallest basic-variable index (row positions
    /// in the basis are arbitrary, so scan order is not Bland's rule), while
    /// a strictly better ratio wins regardless of index, and outside Bland's
    /// rule ties are irrelevant (strict minimum only).
    #[test]
    fn phase1_bland_leaving_prefers_smallest_basic_index_on_tie() {
        let tie = 1e-12;
        // First candidate selected: strict min vs the initial huge bound.
        assert!(DualSolver::<f64>::bland_leaving_prefer(
            true,
            2.0,
            1e20,
            3,
            usize::MAX,
            tie
        ));
        // Exact tie, smaller basic-variable index (2) at a later row wins.
        assert!(
            DualSolver::<f64>::bland_leaving_prefer(true, 2.0, 2.0, 2, 3, tie),
            "Bland tie-break must prefer basic variable 2 over 3"
        );
        // Exact tie, larger index loses.
        assert!(!DualSolver::<f64>::bland_leaving_prefer(
            true, 2.0, 2.0, 4, 3, tie
        ));
        // Strictly better ratio wins regardless of index.
        assert!(DualSolver::<f64>::bland_leaving_prefer(
            true, 1.5, 2.0, 3, 3, tie
        ));
        // Outside Bland's rule: strict minimum only, ties are scan-order.
        assert!(!DualSolver::<f64>::bland_leaving_prefer(
            false, 2.0, 2.0, 2, 3, tie
        ));
    }

    // ---- PFI eta-chain correctness fuzz tests -------------------------------
    //
    // update_lu was replaced by a product-form-of-inverse (PFI) eta chain
    // (see add_eta/apply_etas_forward/apply_etas_transpose above) after
    // profiling found the old dense "bump elimination" scheme failing 99.6%
    // of calls (always at the very first elimination step, always an exact
    // zero pivot) on real MIP node-LP relaxations -- not a numerical rarity
    // but the deterministic outcome whenever a sparse entering column's
    // nonzero rows don't happen to include the row about to leave. These
    // tests generate random LPs with exactly that sparse-column shape (the
    // structure that broke the old scheme) and validate the new one directly
    // against ground truth, independent of whatever the solver itself
    // computes internally: primal feasibility from first principles, and (for
    // tiny instances) the exact optimum via brute-force vertex enumeration.

    /// Uniform [lo, hi) draw from the shared LCG.
    fn lcg_f64(g: &mut iconic_core::rng::Lcg, lo: f64, hi: f64) -> f64 {
        g.uniform(lo, hi)
    }

    /// Random sparse equality-constrained bounded LP: `m` rows, `k` structural
    /// columns (each with up to `nnz_per_col` nonzero rows) plus `m` trailing
    /// identity (slack) columns used as the initial basis -- the shape that
    /// stresses the PFI eta chain (general/sparse structural columns entering
    /// one at a time against a mostly-slack basis, matching the real
    /// set-packing MIP relaxations that triggered the original bug). All `n`
    /// variables get finite bounds so a bounded-polytope brute force is exact.
    /// `(a, c, b, lb, ub)` — the dense row-major constraint matrix, objective,
    /// right-hand side, and the lower/upper bound vectors.
    type SparseLp = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

    fn random_sparse_lp(seed: u64, m: usize, k: usize, nnz_per_col: usize) -> SparseLp {
        let mut g = iconic_core::rng::Lcg::new(seed);
        let n = k + m;
        let mut a = vec![0.0f64; m * n];
        for j in 0..k {
            let mut rows_used = std::collections::HashSet::new();
            for _ in 0..nnz_per_col.min(m) {
                let mut r = (g.next_u64() as usize) % m;
                while rows_used.contains(&r) {
                    r = (r + 1) % m;
                }
                rows_used.insert(r);
                a[r * n + j] = lcg_f64(&mut g, -3.0, 3.0);
            }
        }
        for i in 0..m {
            a[i * n + (k + i)] = 1.0;
        }
        let c: Vec<f64> = (0..n).map(|_| lcg_f64(&mut g, -5.0, 5.0)).collect();
        let b: Vec<f64> = (0..m).map(|_| lcg_f64(&mut g, -10.0, 10.0)).collect();
        let l: Vec<f64> = (0..n).map(|_| lcg_f64(&mut g, -5.0, 0.0)).collect();
        let u: Vec<f64> = l
            .iter()
            .map(|&lo| lo + lcg_f64(&mut g, 1.0, 15.0))
            .collect();
        (a, c, b, l, u)
    }

    // History: this fuzzer originally surfaced a PRE-EXISTING robustness gap
    // unrelated to the PFI eta-chain rewrite above -- confirmed by splicing
    // this same test onto the pre-PFI `update_lu` implementation (checked out
    // from git history) and reproducing an identical failure. Root cause:
    // `DualSolver::new` used to always start every nonbasic variable at its
    // LOWER bound without verifying the resulting basis was dual-feasible
    // (nonbasic-at-lower needs a nonnegative reduced cost); when it wasn't,
    // the ratio test's sign convention rejected the only viable pivot
    // direction and either stalled (IterationLimit) or misreported Infeasible
    // on a feasible problem. Fixed by `crash_dual_feasible` (see its doc):
    // choosing each nonbasic variable's starting bound by the sign of its
    // reduced cost. A SEPARATE bug (a missing within-block dependency in
    // `lu_solve_into`'s 4-wide unrolled forward/backward substitution -- see
    // its doc) was found and fixed at the same time, root-caused via this
    // exact fuzzer regressing once the crash changed the pivot trajectory
    // enough to hit it. With both fixes, this asserts SOUNDNESS *and*
    // completeness cleanly: zero notes across a 30000-seed sweep (versus
    // routine false Infeasible/IterationLimit before either fix landed).
    #[test]
    fn fuzz_sparse_lp_solution_is_primal_feasible() {
        for seed in 0..30000u64 {
            let m = 4 + (seed as usize % 20); // 4..24, enough pivots to exercise the eta chain
            let k = m + 2 + (seed as usize % 6); // more structural columns than rows
            let (a, c, b, l, u) = random_sparse_lp(seed * 7919 + 1, m, k, 2);
            let n = k + m;
            let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
            let sol = s.cold_solve();
            // Any well-posed solve on these modest sizes should reach a definitive
            // answer (Optimal or Infeasible) inside the iteration budget -- hitting
            // it is always a robustness failure, unlike Infeasible (which can be a
            // correct answer for a random LP with no feasible region).
            assert!(
                !matches!(sol.status, Status::IterationLimit),
                "seed {seed}: hit iteration limit (m={m}, n={n})"
            );
            if matches!(sol.status, Status::Optimal) {
                for i in 0..m {
                    let mut ax = 0.0;
                    for j in 0..n {
                        ax += a[i * n + j] * sol.x[j];
                    }
                    assert!(
                        (ax - b[i]).abs() < 1e-6,
                        "seed {seed} row {i}: Ax={ax} != b={} (m={m}, n={n})",
                        b[i]
                    );
                }
                for j in 0..n {
                    assert!(
                        sol.x[j] >= l[j] - 1e-6 && sol.x[j] <= u[j] + 1e-6,
                        "seed {seed} var {j}: x={} out of [{},{}] (m={m}, n={n})",
                        sol.x[j],
                        l[j],
                        u[j]
                    );
                }
            }
        }
    }

    /// The m>=50 path (where DSE-weighted pricing used to be active) had zero
    /// fuzz coverage. The DSE-scaled ratio test does not preserve dual
    /// feasibility -- the entering column must minimize the TRUE ratio
    /// d_j/|alpha_jp| -- so pricing is now on the unscaled ratio. Assert
    /// soundness (Optimal implies a primal-feasible point) and that a
    /// well-posed random LP never burns the iteration budget, at the sizes
    /// where the old weighted-ratio pricing would have engaged.
    #[test]
    fn fuzz_m50_path_lp_solution_is_primal_feasible() {
        for seed in 0..400u64 {
            let m = 50 + (seed as usize % 25); // 50..75, the DSE-active path
            let k = m + 2 + (seed as usize % 6); // more structural columns than rows
            let (a, c, b, l, u) = random_sparse_lp(seed * 2687 + 99, m, k, 2);
            let n = k + m;
            let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
            let sol = s.cold_solve();
            assert!(
                !matches!(sol.status, Status::IterationLimit),
                "seed {seed}: hit iteration limit (m={m}, n={n})"
            );
            if matches!(sol.status, Status::Optimal) {
                for i in 0..m {
                    let mut ax = 0.0;
                    for j in 0..n {
                        ax += a[i * n + j] * sol.x[j];
                    }
                    assert!(
                        (ax - b[i]).abs() < 1e-6,
                        "seed {seed} row {i}: Ax={ax} != b={} (m={m}, n={n})",
                        b[i]
                    );
                }
                for j in 0..n {
                    assert!(
                        sol.x[j] >= l[j] - 1e-6 && sol.x[j] <= u[j] + 1e-6,
                        "seed {seed} var {j}: x={} out of [{},{}] (m={m}, n={n})",
                        sol.x[j],
                        l[j],
                        u[j]
                    );
                }
            }
        }
    }

    /// Plain Gaussian elimination with partial pivoting (test-only reference
    /// helper for the brute-force vertex enumeration below).
    fn solve_dense_square(mat: &mut [f64], rhs: &mut [f64], m: usize) -> Option<Vec<f64>> {
        for k in 0..m {
            let mut piv_row = k;
            let mut piv_val = mat[k * m + k].abs();
            for i in (k + 1)..m {
                let v = mat[i * m + k].abs();
                if v > piv_val {
                    piv_val = v;
                    piv_row = i;
                }
            }
            if piv_val < 1e-10 {
                return None;
            }
            if piv_row != k {
                for j in 0..m {
                    mat.swap(k * m + j, piv_row * m + j);
                }
                rhs.swap(k, piv_row);
            }
            let pv = mat[k * m + k];
            for i in (k + 1)..m {
                let f = mat[i * m + k] / pv;
                if f == 0.0 {
                    continue;
                }
                for j in k..m {
                    mat[i * m + j] -= f * mat[k * m + j];
                }
                rhs[i] -= f * rhs[k];
            }
        }
        let mut x = vec![0.0; m];
        for i in (0..m).rev() {
            let mut s = rhs[i];
            for j in (i + 1)..m {
                s -= mat[i * m + j] * x[j];
            }
            x[i] = s / mat[i * m + i];
        }
        Some(x)
    }

    /// Next k-combination of {0,...,n-1} in lexicographic order (standard
    /// textbook algorithm). Returns false once `combo` is the last combination.
    fn next_combination(combo: &mut [usize], n: usize) -> bool {
        let k = combo.len();
        let mut i = k;
        loop {
            if i == 0 {
                return false;
            }
            i -= 1;
            if combo[i] != i + n - k {
                break;
            }
        }
        combo[i] += 1;
        for j in (i + 1)..k {
            combo[j] = combo[j - 1] + 1;
        }
        true
    }

    /// Exhaustive vertex enumeration for `min c^Tx s.t. Ax=b, l<=x<=u`: for
    /// every way of choosing which `n-m` variables are nonbasic (fixed at a
    /// bound) and which bound each sits at, solve the resulting m×m system for
    /// the rest and keep the best feasible objective. Exact (not a heuristic)
    /// because every variable is finite-bounded, so the feasible region is a
    /// bounded polytope and its optimum is attained at one of these vertices --
    /// only tractable for tiny m,n, used solely as a ground-truth oracle.
    fn brute_force_lp_optimum(
        a: &[f64],
        c: &[f64],
        b: &[f64],
        l: &[f64],
        u: &[f64],
        m: usize,
        n: usize,
    ) -> Option<f64> {
        let mut best: Option<f64> = None;
        let nonbasic_count = n - m;
        let mut combo: Vec<usize> = (0..nonbasic_count).collect();
        loop {
            for bits in 0u32..(1u32 << nonbasic_count) {
                let mut x = vec![0.0f64; n];
                let basic: Vec<usize> = (0..n).filter(|j| !combo.contains(j)).collect();
                for (idx, &j) in combo.iter().enumerate() {
                    x[j] = if (bits >> idx) & 1 == 1 { u[j] } else { l[j] };
                }
                let mut rhs = b.to_vec();
                for &j in &combo {
                    for i in 0..m {
                        rhs[i] -= a[i * n + j] * x[j];
                    }
                }
                let mut mat = vec![0.0f64; m * m];
                for (col_idx, &j) in basic.iter().enumerate() {
                    for i in 0..m {
                        mat[i * m + col_idx] = a[i * n + j];
                    }
                }
                if let Some(xb) = solve_dense_square(&mut mat, &mut rhs, m) {
                    let mut feasible = true;
                    for (idx, &j) in basic.iter().enumerate() {
                        x[j] = xb[idx];
                        if x[j] < l[j] - 1e-7 || x[j] > u[j] + 1e-7 {
                            feasible = false;
                            break;
                        }
                    }
                    if feasible {
                        let obj: f64 = (0..n).map(|j| c[j] * x[j]).sum();
                        best = Some(best.map_or(obj, |b0: f64| b0.min(obj)));
                    }
                }
            }
            if !next_combination(&mut combo, n) {
                break;
            }
        }
        best
    }

    // History: before `crash_dual_feasible` and the `lu_solve_into` fix
    // (see their docs), a wide seed sweep here found the pre-existing
    // dual-feasibility gap wasn't limited to false Infeasible/IterationLimit
    // (completeness) -- it could also terminate at a "locally optimal
    // looking" point that isn't the true LP optimum (soundness). Both
    // fixes together closed it: zero mismatches across an 8000-seed sweep,
    // so every arm below is now a hard assertion (previously the two
    // "wrong answer" arms only logged a note, tolerating the then-known,
    // now-fixed gap).
    #[test]
    fn fuzz_tiny_lp_matches_brute_force_optimum() {
        for seed in 0..8000u64 {
            let m = 2 + (seed as usize % 3); // 2..4
            let k = 1 + (seed as usize % 3); // 1..3 structural columns
            let (a, c, b, l, u) = random_sparse_lp(seed * 104729 + 17, m, k, 2);
            let n = k + m;
            let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
            let sol = s.cold_solve();
            let truth = brute_force_lp_optimum(&a, &c, &b, &l, &u, m, n);
            match (matches!(sol.status, Status::Optimal), truth) {
                (true, Some(opt)) => assert!((sol.obj - opt).abs() < 1e-5,
                    "seed {seed}: solver obj={} but brute-force optimum={opt} (m={m}, n={n})", sol.obj),
                (true, None) => panic!("seed {seed}: solver reported Optimal ({}) but the LP is infeasible per brute force (m={m}, n={n})", sol.obj),
                (false, Some(opt)) => panic!("seed {seed}: solver reported {:?} but brute force found feasible optimum {opt} (m={m}, n={n})", sol.status),
                (false, None) => {} // both agree: no feasible vertex
            }
        }
    }

    /// ftran and btran must be solving against the *same* basis. For any nonbasic
    /// column `j` and any basis position `p`,
    ///
    ///     rho^T a_j,  where rho = B^-T e_p   (btran -- what the ratio test prices on)
    ///     (B^-1 a_j)_p                       (ftran -- what the pivot element actually is)
    ///
    /// are the same scalar by definition. A basis update applied in one direction and
    /// not the other breaks this identity while leaving every single-direction test
    /// green: that is how a half-implemented LU update once shipped here. Its own test
    /// checked only the ftran against a fresh factorization, so nothing caught that the
    /// btran was solving against a different matrix. The ratio test then priced on a
    /// `rho` that had run away to ~1e19 while the ftran stayed at O(1), node LPs
    /// returned bounds ~38 above their true optimum, and branch-and-bound pruned the
    /// subtrees holding the optimum while still reporting a proof.
    #[test]
    fn ftran_and_btran_agree_on_the_pivot_element() {
        // Whitebox test of the DENSE representation (reads `s.lu`/`s.perm`
        // directly), so pin the sparse-LU gate off: while a SparseLu factor is
        // live the dense array is deliberately stale (see `sparse_lu`). The
        // representation-independent FTRAN/BTRAN identity over the eta chain
        // is covered for the sparse factor by
        // `sparse_lu_path_matches_dense_path_cold_and_warm`.
        set_sparse_lu_forced(false);
        // Clause-like sparse rows: many pivots, dense-ish fill -- the regime where an
        // incremental basis update accumulates corrections rather than staying trivial.
        let m: usize = 105;
        let ns = 130;
        let n = m + ns;
        let mut a_val = vec![0.0f64; m * n];
        for i in 0..m {
            a_val[i * n + ns + i] = 1.0;
        }
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for i in 0..m {
            for _ in 0..3 {
                let j = (next() % ns as u64) as usize;
                let v = ((next() % 7) as f64) - 3.0;
                a_val[i * n + j] = if v == 0.0 { 1.5 } else { v };
            }
        }
        let c_full: Vec<f64> = (0..n)
            .map(|j| {
                if j < ns {
                    -(((j % 11) + 1) as f64)
                } else {
                    0.0
                }
            })
            .collect();
        let b_vec: Vec<f64> = (0..m).map(|i| ((i % 13) as f64) + 1.0).collect();
        let l = vec![0.0; n];
        let u: Vec<f64> = (0..n).map(|j| if j < ns { 1.0 } else { 1e20 }).collect();

        let mut s: DualSolver<f64> = DualSolver::new(&c_full, &a_val, &b_vec, &l, &u, m, n);
        // Solve first, so the identity is checked at a basis actually reached through
        // many pivots -- with a live update chain -- not at the trivial initial one.
        let sol = s.cold_solve();
        // x = 0, s = b >= 0 is feasible by construction, so `Infeasible` here is not a
        // borderline numerical call -- it is impossible. The one-directional update made
        // the solver report exactly that.
        assert!(
            matches!(sol.status, Status::Optimal),
            "cold_solve returned {:?} on an LP where x=0, s=b>=0 is feasible",
            sol.status
        );

        let mut worst = 0.0f64;
        for p in [0usize, 1, m / 3, m / 2, m - 1] {
            for i in 0..m {
                s.rho_ep[i] = 0.0;
            }
            s.rho_ep[p] = 1.0;
            DualSolver::<f64>::apply_etas_transpose(
                &s.eta_p,
                &s.eta_start,
                &s.eta_idx,
                &s.eta_val,
                &s.eta_alpha,
                s.n_etas,
                m,
                &mut s.rho_ep,
            );
            DualSolver::<f64>::lu_solve_trans_into_dense(
                &s.lu,
                &s.perm,
                &s.rho_ep,
                &mut s.lu_pi,
                &mut s.lu_z,
                &mut s.lu_y,
                m,
            );
            let rho = s.lu_pi[..m].to_vec();

            for j in 0..n {
                if s.in_basis[j] {
                    continue;
                }
                let mut alpha_row = 0.0;
                for idx in s.a_col_start[j]..s.a_col_start[j + 1] {
                    alpha_row += s.a_val[idx] * rho[s.a_row_idx[idx]];
                }
                for i in 0..m {
                    s.spike[i] = 0.0;
                }
                for idx in s.a_col_start[j]..s.a_col_start[j + 1] {
                    s.spike[s.a_row_idx[idx]] = s.a_val[idx];
                }
                DualSolver::<f64>::lu_solve_into_dense(
                    &s.lu,
                    &s.perm,
                    &s.spike,
                    &mut s.lu_y,
                    m,
                    PIVOT_TOL,
                );
                DualSolver::<f64>::apply_etas_forward(
                    &s.eta_p,
                &s.eta_start,
                &s.eta_idx,
                &s.eta_val,
                &s.eta_alpha,
                s.n_etas,
                    m,
                    &mut s.lu_y,
                );
                let alpha_col = s.lu_y[p];

                let scale = 1.0f64.max(alpha_row.abs()).max(alpha_col.abs());
                let rel = (alpha_row - alpha_col).abs() / scale;
                if rel > worst {
                    worst = rel;
                }
                assert!(rel < 1e-9,
                    "p={p} j={j}: btran gives alpha={alpha_row:e}, ftran gives {alpha_col:e} (rel {rel:e})");
            }
        }
        assert!(worst < 1e-9, "worst ftran/btran disagreement {worst:e}");
        // Restore the default (tests may run in any order).
        set_sparse_lu_forced(true);
    }

    /// A cold_solve on a mid-size LP produces a genuinely optimal, primal-feasible
    /// solution -- the basis-update chain is exercised across many pivots here.
    #[test]
    fn cold_solve_correct_on_medium_lp() {
        // random-ish LP, large enough to need many basis updates
        let m = 50;
        let n = m + 4;
        let a_val: Vec<f64> = (0..(m * n))
            .map(|i| {
                let col = i % n;
                let row = i / n;
                if col >= 4 {
                    if row == col - 4 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    // One LCG step from a position-derived seed (sparsity pattern).
                    let s = ((col * 7919 + row * 104729 + 12345) as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if s.is_multiple_of(5) {
                        (s as f64 / u64::MAX as f64) * 3.0 - 1.5
                    } else {
                        0.0
                    }
                }
            })
            .collect();
        let c_full: Vec<f64> = (0..n)
            .map(|j| if j < 4 { -((j + 1) as f64) } else { 0.0 })
            .collect();
        let b: Vec<f64> = (0..m).map(|i| (i as f64) * 0.5 + 2.0).collect();
        let l: Vec<f64> = vec![0.0; n];
        let u: Vec<f64> = vec![1e20; n];

        let mut s: DualSolver<f64> = DualSolver::new(&c_full, &a_val, &b, &l, &u, m, n);
        let sol = s.cold_solve();
        assert!(
            matches!(sol.status, Status::Optimal),
            "expected Optimal, got {:?}",
            sol.status
        );
        // The objective must be feasible: check Ax = b
        for i in 0..m {
            let mut ax = 0.0;
            for j in 0..n {
                ax += a_val[i * n + j] * sol.x[j];
            }
            assert!((ax - b[i]).abs() < 1e-6, "row {i}: Ax={ax} != b={}", b[i]);
        }
        // Verify solution is nonnegative
        for j in 0..n {
            assert!(sol.x[j] >= -1e-8, "var {j} negative: {}", sol.x[j]);
        }
    }

    /// The stall-triggered RHS perturbation's restore discipline: applying it
    /// changes the RHS (and the basic values), and restoring brings back the
    /// EXACT original RHS and basic values — the perturbation must never leak
    /// into the LP the caller sees.
    #[test]
    fn perturb_restore_is_exact() {
        let a = vec![1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
        let c = vec![0.5, -2.0, 0.0, 0.0];
        let b = vec![5.0, 3.0];
        let l = vec![0.0; 4];
        let u = vec![4.0, 1e20, 1e20, 6.0];
        let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, 2, 4);
        let sol = s.cold_solve();
        assert!(matches!(sol.status, Status::Optimal));
        let xb_before = s.xb.clone();
        let b_before = s.b.clone();
        // Apply the perturbation directly (as dual_loop would at a stall).
        s.apply_rhs_perturbation(1.0);
        assert_eq!(s.perturb_episodes, 1);
        assert!(s.perturb_active);
        // The RHS changed by ~2e-4 relative, and the basic values followed.
        let mut changed = false;
        for (i, (&bi, &bb)) in s.b.iter().zip(b_before.iter()).enumerate() {
            if (bi - bb).abs() > 1e-12 {
                changed = true;
            }
            assert!(
                (bi - bb).abs() <= 2e-4 * (1.0 + bb.abs()) * 1.01,
                "row {i}: perturbation exceeds its 2e-4 relative magnitude"
            );
        }
        assert!(changed, "the perturbation must actually change the RHS");
        assert!(
            (s.xb[0] - xb_before[0]).abs() > 0.0,
            "the basic values must follow the perturbed RHS"
        );
        // Restore: exact RHS and exact basic values.
        s.restore_exact_rhs();
        assert!(!s.perturb_active);
        for (i, (&bi, &bb)) in s.b.iter().zip(b_before.iter()).enumerate() {
            assert_eq!(bi, bb, "row {i}: RHS must be bit-exact after restore");
        }
        for (i, (&xi, &xb)) in s.xb.iter().zip(xb_before.iter()).enumerate() {
            assert_eq!(xi, xb, "row {i}: xb must be bit-exact after restore");
        }
        // And a solve that went through a perturbation episode returns the same
        // solution as an unperturbed solve (deterministic, exact after restore).
        let mut s2: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, 2, 4);
        let sol2 = s2.cold_solve();
        assert_eq!(sol2.obj, sol.obj);
        assert!((sol2.obj + 7.5).abs() < 1e-5);
    }

    /// The perturbation is invisible on problems that never stall: a fuzz of
    /// deterministic random LPs (including exact-duplicate constraint rows, the
    /// degenerate shape that ties the leaving-row selection) must produce the
    /// same status and the same objective with the perturbation on and off —
    /// it only ever fires mid-stall, and the restore keeps the answer exact.
    #[test]
    fn perturb_on_off_agree_on_random_lps() {
        let mut rnd = iconic_core::rng::Lcg::new(0);
        let mut trials_with_episodes = 0usize;
        for trial in 0..60 {
            let n = 8;
            let m = 6;
            let mut a = vec![0.0; m * n];
            for i in 0..m / 2 {
                for j in 0..n {
                    let v = 2.0 * rnd.unit() - 1.0;
                    a[(2 * i) * n + j] = v;
                    a[(2 * i + 1) * n + j] = v; // exact duplicate row: tied leaving violations
                }
            }
            let c: Vec<f64> = (0..n).map(|_| 2.0 * rnd.unit() - 1.0).collect();
            let l = vec![0.0; n];
            let u = vec![1e20; n];
            let b: Vec<f64> = (0..m)
                .map(|i| {
                    let mut ax = 0.0;
                    for j in 0..n {
                        ax += a[i * n + j] * (0.5 + 0.5 * rnd.unit());
                    }
                    ax + 1.0
                })
                .collect();
            let mut on: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
            on.set_max_iters(20000);
            let sol_on = on.cold_solve();
            let mut off: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
            off.set_perturb_on_stall(false);
            off.set_max_iters(20000);
            let sol_off = off.cold_solve();
            assert_eq!(
                sol_on.status, sol_off.status,
                "trial {trial}: status mismatch"
            );
            if matches!(sol_on.status, Status::Optimal) {
                assert!(
                    (sol_on.obj - sol_off.obj).abs() < 1e-7,
                    "trial {trial}: objectives differ: {} vs {}",
                    sol_on.obj,
                    sol_off.obj
                );
            }
            if on.perturb_episodes > 0 {
                trials_with_episodes += 1;
            }
        }
        // The perturbation must at least be capable of firing on this
        // degenerate shape without corrupting anything (it fires when a stall
        // happens; the exact count is instance-dependent).
        let _ = trials_with_episodes;
    }

    /// Primal-feasibility check of a solution against the dense LP data:
    /// every variable within its bounds and `|(Ax)_i - b_i|` small where the
    /// row's slack is basic at its bound... — actually, the general check:
    /// `x` satisfies `l <= x <= u` and `A x` lies in the interval implied by
    /// `b` and the slack columns (each slack column is the identity at
    /// `n - m + i`, so `(A x)_i + s_i = b_i` with the slack basic value
    /// recovered as `b_i - (A x)_i` — for a solution whose status is
    /// Optimal the slack is within [l, u] of the slack column).
    #[allow(clippy::too_many_arguments)]
    fn assert_primal_feasible(
        a: &[f64],
        b: &[f64],
        l: &[f64],
        u: &[f64],
        m: usize,
        n: usize,
        x: &[f64],
        status: Status,
        tol: f64,
    ) {
        if !matches!(status, Status::Optimal) {
            return;
        }
        for j in 0..n {
            assert!(
                x[j] >= l[j] - tol && x[j] <= u[j] + tol,
                "x[{j}]={} outside [{}, {}]",
                x[j],
                l[j],
                u[j]
            );
        }
        let mut ax = vec![0.0f64; m];
        for i in 0..m {
            for j in 0..n {
                ax[i] += a[i * n + j] * x[j];
            }
            // Slack column i is identity at n - m + i; its bound interval is
            // [l[n-m+i], u[n-m+i]] and it must satisfy ax_i + s_i = b_i.
            let s = b[i] - ax[i];
            assert!(
                s >= l[n - m + i] - tol && s <= u[n - m + i] + tol,
                "slack of row {i} = {s} outside [{}, {}]",
                l[n - m + i],
                u[n - m + i]
            );
        }
    }

    /// The LAPACK mirror path must solve the same LP to the same status and
    /// objective as the hand-rolled dense-LU path, on a genuinely sparse
    /// random LP whose basis is far from identity (so the factorization and
    /// every pivot go through the dense LU). Exercises factor_once_blas
    /// (cold), the transposed and forward base solves, the eta chain, and
    /// the warm-start mirror rebuild (`hot_solve` inheriting the parent's
    /// factor + swap sequence).
    #[test]
    fn blas_path_matches_hand_path_cold_and_warm() {
        // Gate on the BLAS being actually compiled+linked and enabled; on a
        // BLAS-less machine the mirror is None and the test vacuously passes.
        if !iconic_linalg::blas::blas_enabled() {
            return;
        }
        let (a, c, b, l, u) = random_sparse_lp(99, 40, 26, 3);
        let m = 40;
        let n = 26 + m;

        // Hand path (BLAS off for this thread) vs BLAS path.
        iconic_linalg::blas::set_blas_enabled(false);
        let mut hand: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
        hand.set_max_iters(20000);
        let sol_hand = hand.cold_solve();
        let hand_basis = hand.export_basis();

        iconic_linalg::blas::set_blas_enabled(true);
        let mut blas: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
        assert!(
            blas.blas.is_some(),
            "BLAS enabled but the mirror was not allocated"
        );
        blas.set_max_iters(20000);
        let sol_blas = blas.cold_solve();

        assert_eq!(
            sol_hand.status, sol_blas.status,
            "cold status differs: hand {:?} vs blas {:?}",
            sol_hand.status, sol_blas.status
        );
        if matches!(sol_hand.status, Status::Optimal) {
            assert!(
                (sol_hand.obj - sol_blas.obj).abs() < 1e-7,
                "cold objectives differ: {} vs {}",
                sol_hand.obj,
                sol_blas.obj
            );
        }
        assert_primal_feasible(&a, &b, &l, &u, m, n, &sol_blas.x, sol_blas.status, 1e-7);

        // Warm start: perturbed RHS -> child LP, hot-solve from the parent
        // basis on both paths (the BLAS path inherits the factor and must
        // rebuild its mirror from the swap sequence).
        let mut b2 = b.clone();
        for v in b2.iter_mut().take(10) {
            *v += 0.37;
        }
        iconic_linalg::blas::set_blas_enabled(false);
        let mut hand2: DualSolver<f64> = DualSolver::new(&c, &a, &b2, &l, &u, m, n);
        hand2.set_max_iters(20000);
        let hot_hand = hand2.hot_solve(&hand_basis);

        iconic_linalg::blas::set_blas_enabled(true);
        let mut blas2: DualSolver<f64> = DualSolver::new(&c, &a, &b2, &l, &u, m, n);
        blas2.set_max_iters(20000);
        let hot_blas = blas2.hot_solve(&hand_basis);

        assert_eq!(
            hot_hand.status, hot_blas.status,
            "warm status differs: hand {:?} vs blas {:?}",
            hot_hand.status, hot_blas.status
        );
        if matches!(hot_hand.status, Status::Optimal) {
            assert!(
                (hot_hand.obj - hot_blas.obj).abs() < 1e-7,
                "warm objectives differ: {} vs {}",
                hot_hand.obj,
                hot_blas.obj
            );
        }
        assert_primal_feasible(&a, &b2, &l, &u, m, n, &hot_blas.x, hot_blas.status, 1e-7);

        // The reverse inheritance: a HAND-factored parent basis feeding a
        // BLAS-path child (the parent's swaps must be carried and
        // reconstructed) — and the BLAS-path parent feeding a hand child.
        iconic_linalg::blas::set_blas_enabled(true);
        let mut blas3: DualSolver<f64> = DualSolver::new(&c, &a, &b2, &l, &u, m, n);
        blas3.set_max_iters(20000);
        let hot_blas_of_blas_parent = blas3.hot_solve(&blas.export_basis());
        assert_eq!(
            hot_hand.status, hot_blas_of_blas_parent.status,
            "BLAS-parent -> BLAS-child warm status differs"
        );
        if matches!(hot_hand.status, Status::Optimal) {
            assert!(
                (hot_hand.obj - hot_blas_of_blas_parent.obj).abs() < 1e-7,
                "BLAS-parent -> BLAS-child warm objective differs"
            );
        }
        assert_primal_feasible(
            &a,
            &b2,
            &l,
            &u,
            m,
            n,
            &hot_blas_of_blas_parent.x,
            hot_blas_of_blas_parent.status,
            1e-7,
        );

        iconic_linalg::blas::set_blas_enabled(false);
        let mut hand3: DualSolver<f64> = DualSolver::new(&c, &a, &b2, &l, &u, m, n);
        hand3.set_max_iters(20000);
        let hot_hand_of_blas_parent = hand3.hot_solve(&blas.export_basis());
        assert_eq!(
            hot_hand.status, hot_hand_of_blas_parent.status,
            "BLAS-parent -> hand-child warm status differs"
        );
        if matches!(hot_hand.status, Status::Optimal) {
            assert!(
                (hot_hand.obj - hot_hand_of_blas_parent.obj).abs() < 1e-7,
                "BLAS-parent -> hand-child warm objective differs"
            );
        }
        // Restore the thread-local override (tests may run in any order).
        iconic_linalg::blas::set_blas_enabled(false);
    }

    /// The sparse Markowitz-LU path must solve the same LP to the same status
    /// and objective as the dense/BLAS path, on an LP above the
    /// `SPARSE_LU_MIN_M` gate so every factorization and base solve routes
    /// through `SparseLu`. Exercises the cold factorization (including the
    /// `perm` repair contract on the initial basis), both solve directions,
    /// the eta chain over a sparse base solve, and warm-start inheritance
    /// (`HotBasis` carries the sparse factor; children skip the dense-lu
    /// copy and mirror rebuild). Cross-parent cases cover sparse-parent ->
    /// dense-child (the parent's `swaps` must still drive a dense refactor
    /// when the child's gate is off) and dense-parent -> sparse-child.
    #[test]
    fn sparse_lu_path_matches_dense_path_cold_and_warm() {
        let (a, c, b, l, u) = random_sparse_lp(7, 160, 120, 3);
        let m = 160;
        let n = 120 + m;

        // Reference: sparse LU disabled -> the dense/BLAS path.
        set_sparse_lu_forced(false);
        let mut dense: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
        dense.set_max_iters(20000);
        let sol_dense = dense.cold_solve();
        let dense_basis = dense.export_basis();

        // Sparse path (the default).
        set_sparse_lu_forced(true);
        let mut sparse: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
        sparse.set_max_iters(20000);
        let sol_sparse = sparse.cold_solve();
        assert!(
            sparse.sparse_lu.is_some(),
            "sparse gate on but no factor survived the cold solve"
        );

        assert_eq!(
            sol_dense.status, sol_sparse.status,
            "cold status differs: dense {:?} vs sparse {:?}",
            sol_dense.status, sol_sparse.status
        );
        if matches!(sol_dense.status, Status::Optimal) {
            assert!(
                (sol_dense.obj - sol_sparse.obj).abs() < 1e-7,
                "cold objectives differ: {} vs {}",
                sol_dense.obj,
                sol_sparse.obj
            );
        }
        assert_primal_feasible(&a, &b, &l, &u, m, n, &sol_sparse.x, sol_sparse.status, 1e-7);

        // Warm start: perturbed RHS -> child LP, hot-solve from the parent
        // basis on both paths. The sparse child inherits the parent's
        // SparseLu and applies etas over it; the dense child rebuilds from
        // the swap sequence.
        let mut b2 = b.clone();
        for v in b2.iter_mut().take(20) {
            *v += 0.41;
        }
        set_sparse_lu_forced(false);
        let mut dense2: DualSolver<f64> = DualSolver::new(&c, &a, &b2, &l, &u, m, n);
        dense2.set_max_iters(20000);
        let hot_dense = dense2.hot_solve(&dense_basis);

        set_sparse_lu_forced(true);
        let mut sparse2: DualSolver<f64> = DualSolver::new(&c, &a, &b2, &l, &u, m, n);
        sparse2.set_max_iters(20000);
        let hot_sparse = sparse2.hot_solve(&sparse.export_basis());

        assert_eq!(
            hot_dense.status, hot_sparse.status,
            "warm status differs: dense {:?} vs sparse {:?}",
            hot_dense.status, hot_sparse.status
        );
        if matches!(hot_dense.status, Status::Optimal) {
            assert!(
                (hot_dense.obj - hot_sparse.obj).abs() < 1e-7,
                "warm objectives differ: {} vs {}",
                hot_dense.obj,
                hot_sparse.obj
            );
        }
        assert_primal_feasible(&a, &b2, &l, &u, m, n, &hot_sparse.x, hot_sparse.status, 1e-7);

        // Cross inheritance: sparse-parent basis feeding a dense-path child
        // (HotBasis.sparse is Some but the child's gate is off -> the child
        // must refactor densely from the swap sequence), and a dense-parent
        // basis feeding a sparse-path child.
        set_sparse_lu_forced(false);
        let mut dense3: DualSolver<f64> = DualSolver::new(&c, &a, &b2, &l, &u, m, n);
        dense3.set_max_iters(20000);
        let hot_dense_of_sparse_parent = dense3.hot_solve(&sparse.export_basis());
        assert_eq!(
            hot_dense.status, hot_dense_of_sparse_parent.status,
            "sparse-parent -> dense-child warm status differs"
        );
        if matches!(hot_dense.status, Status::Optimal) {
            assert!(
                (hot_dense.obj - hot_dense_of_sparse_parent.obj).abs() < 1e-7,
                "sparse-parent -> dense-child warm objective differs"
            );
        }

        set_sparse_lu_forced(true);
        let mut sparse3: DualSolver<f64> = DualSolver::new(&c, &a, &b2, &l, &u, m, n);
        sparse3.set_max_iters(20000);
        let hot_sparse_of_dense_parent = sparse3.hot_solve(&dense_basis);
        assert_eq!(
            hot_dense.status, hot_sparse_of_dense_parent.status,
            "dense-parent -> sparse-child warm status differs"
        );
        if matches!(hot_dense.status, Status::Optimal) {
            assert!(
                (hot_dense.obj - hot_sparse_of_dense_parent.obj).abs() < 1e-7,
                "dense-parent -> sparse-child warm objective differs"
            );
        }
        assert_primal_feasible(
            &a,
            &b2,
            &l,
            &u,
            m,
            n,
            &hot_sparse_of_dense_parent.x,
            hot_sparse_of_dense_parent.status,
            1e-7,
        );

        // Restore the default (tests may run in any order).
        set_sparse_lu_forced(true);
    }
}
