//! Phase timing for `solve_mip`, so "which part of the solve is slow" is a
//! measurement rather than a guess.
//!
//! The MIP suite contains instances that spend tens of seconds while exploring
//! *one* node — all of that time is root work (presolve passes, heuristics, cut
//! rounds, strong branching), and the per-instance report alone cannot say which.
//! Accumulators are thread-local and reported to stderr when `ICONIC_MIP_PROFILE`
//! is set; the timing itself is always collected, which costs one clock read per
//! phase against phases that are individually at least an LP solve.

use std::cell::RefCell;
use std::thread::LocalKey;
use std::time::Instant;

/// Phase slots. Keep in sync with [`NAMES`].
pub const PRESOLVE: usize = 0;
pub const ROOT_HEUR: usize = 1;
pub const RC_FIX: usize = 2;
pub const NODE_LP: usize = 3;
pub const NODE_PRESOLVE: usize = 4;
pub const CUTS: usize = 5;
pub const STRONG_BRANCH: usize = 6;
pub const PERIODIC_HEUR: usize = 7;
pub const ROOT_OBBT: usize = 8;
/// The conic-IPM fallback that re-solves a node LP the dual simplex gave up
/// on. Sits INSIDE the [`NODE_LP`] slot (nested, like strong-branching probes),
/// but unlike the simplex split it had no instrumentation at all -- on
/// cvrp_n15_k4 six fallback solves were ~7s of a 16.5s solve hiding behind a
/// "node LP" line whose simplex accounted for 1.4s.
pub const NODE_IPM: usize = 9;

pub const SLOTS: usize = 10;

pub const NAMES: [&str; SLOTS] = [
    "root presolve",
    "root heuristics",
    "root rc-fixing",
    "node LP",
    "node presolve/probing",
    "cuts",
    "strong branching",
    "periodic heuristics",
    "root OBBT",
    "..of which IPM fallback",
];

thread_local! {
    static ACC: RefCell<[u64; SLOTS]> = const { RefCell::new([0; SLOTS]) };
}

/// Add `ns` nanoseconds to a phase slot.
#[inline]
pub fn add(slot: usize, ns: u64) {
    ACC.with(|a| a.borrow_mut()[slot] += ns);
}

// ── Per-family cut-generation timing ─────────────────────────────────────
//
// The `CUTS` slot cannot attribute cost between the ~13 separator families
// (plus the pool selection/copy step) that run per cut round. This is the
// first-class instrumentation for cut tuning: measured, the multiknap
// in-tree battery was ~1.65ms/node with no single family dominating and
// disabling several families changing nothing — which family is the cost is
// exactly what these counters answer.

pub const F_COVER: usize = 0;
pub const F_MIXING: usize = 1;
pub const F_PACKING: usize = 2;
pub const F_SURROGATE: usize = 3;
pub const F_STRONGCG: usize = 4;
pub const F_CLIQUE: usize = 5;
pub const F_MIR: usize = 6;
pub const F_GOMORY: usize = 7;
pub const F_INTERSECTION: usize = 8;
pub const F_LANDP: usize = 9;
pub const F_IMPLIED: usize = 10; // implied-bound + projected-implied-bound
pub const F_FLOW: usize = 11;
pub const F_MULTIROW: usize = 12;
pub const F_SUBTOUR: usize = 13;
pub const F_TRIANGLE: usize = 14;
pub const F_ZEROHALF: usize = 15;
pub const F_SELECT: usize = 16; // pool selection + copy into node LPs
pub const CUT_FAMILIES: usize = 17;

pub const FAM_NAMES: [&str; CUT_FAMILIES] = [
    "cover",
    "mixing",
    "packing",
    "surrogate",
    "strong-CG",
    "clique",
    "MIR",
    "Gomory",
    "intersection",
    "lift-and-project",
    "implied-bound",
    "flow-cover",
    "multirow",
    "subtour",
    "triangle",
    "zero-half",
    "selection/copy",
];

thread_local! {
    static FAM: RefCell<[u64; CUT_FAMILIES]> = const { RefCell::new([0; CUT_FAMILIES]) };
}

/// Add `ns` nanoseconds to a cut-family slot.
#[inline]
pub fn fam_add(slot: usize, ns: u64) {
    FAM.with(|f| f.borrow_mut()[slot] += ns);
}

/// Read and clear one thread-local accumulator slot array.
fn take_slot<const N: usize>(cell: &'static LocalKey<RefCell<[u64; N]>>) -> [u64; N] {
    cell.with(|s| {
        let mut b = s.borrow_mut();
        let out = *b;
        *b = [0; N];
        out
    })
}

fn take_fams() -> [u64; CUT_FAMILIES] {
    take_slot(&FAM)
}

/// Counted events, reported alongside the phase times. A phase can be expensive either
/// because each call is slow or because it is called far more often than expected, and
/// the timings alone cannot tell those apart: most of tsptw_n6 sat in the node-LP slot
/// while the simplex accounted for almost none of it, which is a count question -- the
/// simplex was giving up on nearly every node and a cold IPM solve was picking up each
/// one.
pub const CNT_SIMPLEX_FAIL: usize = 0;
pub const CNT_SIMPLEX_INFEAS: usize = 1;
pub const CNT_SIMPLEX_ITERLIM: usize = 2;
pub const CNT_IPM_FALLBACK: usize = 3;
pub const CNT_SUBMIP: usize = 4;
/// The IPM fallback found a solution for a node the simplex had called `Infeasible`.
/// This is the count that decides whether the simplex's infeasibility conclusion can be
/// trusted: everything else only says how *often* it is consulted.
pub const CNT_IPM_OVERTURNED: usize = 5;
pub const CNT_INACCURATE_LP: usize = 6;
pub const CNT_NODE_WARM: usize = 7;
pub const CNT_NODE_WARMEXT: usize = 8;
pub const CNT_NODE_COLD: usize = 9;
pub const CNT_NODE_WARM_RETRY: usize = 10;
/// Infeasible verdicts settled by the tableau-row Farkas certificate (no
/// cold re-solve, no IPM). The certificate is the cheap path; this count
/// against [`CNT_SIMPLEX_INFEAS`] shows how much of the confirmation work
/// the certificate already absorbs.
pub const CNT_FARKAS_CERT: usize = 11;
pub const COUNTERS: usize = 12;

pub const COUNTER_NAMES: [&str; COUNTERS] = [
    "simplex gave up",
    "  ..reported Infeasible",
    "  ..reported IterLimit",
    "IPM fallback solves",
    "sub-MIPs run",
    "  ..IPM overturned it",
    "node LPs below tolerance",
    "node LPs warm-started",
    "  ..via row-addition extension",
    "node LPs solved cold",
    "warm IterLimit -> cold retry won",
    "infeas certified by Farkas row",
];

thread_local! {
    static CNT: RefCell<[u64; COUNTERS]> = const { RefCell::new([0; COUNTERS]) };
}

/// Count one occurrence of a diagnostic event.
#[inline]
pub fn bump(counter: usize) {
    CNT.with(|c| c.borrow_mut()[counter] += 1);
}

fn take_counts() -> [u64; COUNTERS] {
    take_slot(&CNT)
}

/// Read and clear the accumulators.
pub fn take() -> [u64; SLOTS] {
    take_slot(&ACC)
}

/// True when `ICONIC_MIP_PROFILE` is set in the environment.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ICONIC_MIP_PROFILE").is_some())
}

/// Print the phase breakdown against a total wall time, then clear it. Phases
/// nest (a node LP inside a strong-branching probe is counted once, under strong
/// branching), so the slots sum to at most the total; the remainder is reported
/// as "unattributed".
pub fn report(label: &str, total_secs: f64, nodes: usize) {
    let acc = take();
    if !enabled() {
        return;
    }
    let total_ns = (total_secs * 1e9) as u64;
    eprintln!("── profile: {label} ({total_secs:.3}s, {nodes} nodes) ──");
    let mut attributed = 0u64;
    for (i, name) in NAMES.iter().enumerate() {
        attributed += acc[i];
        if acc[i] == 0 {
            continue;
        }
        let secs = acc[i] as f64 / 1e9;
        eprintln!(
            "   {name:<24} {secs:>9.3}s  {:>5.1}%",
            100.0 * secs / total_secs.max(1e-12)
        );
    }
    let rest = total_ns.saturating_sub(attributed) as f64 / 1e9;
    eprintln!(
        "   {:<24} {rest:>9.3}s  {:>5.1}%",
        "unattributed",
        100.0 * rest / total_secs.max(1e-12)
    );
    let counts = take_counts();
    for (i, name) in COUNTER_NAMES.iter().enumerate() {
        if counts[i] != 0 {
            eprintln!("   {name:<24} {:>9}", counts[i]);
        }
    }
    let fams = take_fams();
    let fam_total: u64 = fams.iter().sum();
    if fam_total > 0 {
        eprintln!("   ── cut families ──");
        for (i, name) in FAM_NAMES.iter().enumerate() {
            if fams[i] == 0 {
                continue;
            }
            let secs = fams[i] as f64 / 1e9;
            eprintln!(
                "   {name:<24} {secs:>9.3}s  {:>5.1}% of cut work",
                100.0 * fams[i] as f64 / fam_total as f64
            );
        }
    }
}

#[inline]
pub fn start() -> Instant {
    Instant::now()
}

/// RAII accumulator: owns a clock started at construction and adds the
/// elapsed time to `slot` when dropped. For functions with many return paths
/// (`solve_node_lp_ipm` accepts or rejects at four points) one scoped guard
/// beats instrumenting every `return`.
///
/// Deliberately NOT same-slot-reentrant: nesting a slot inside itself
/// double-counts. Nested *different* slots (NODE_IPM inside NODE_LP) are
/// exactly what the report wants to show.
pub struct ScopedAdd {
    slot: usize,
    t0: Instant,
}

impl ScopedAdd {
    #[inline]
    pub fn over(slot: usize) -> Self {
        ScopedAdd {
            slot,
            t0: Instant::now(),
        }
    }
}

impl Drop for ScopedAdd {
    fn drop(&mut self) {
        add(self.slot, self.t0.elapsed().as_nanos() as u64);
    }
}
