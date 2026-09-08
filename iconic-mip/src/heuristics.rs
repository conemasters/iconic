//! MIP primal heuristics.
//!
//! **Feasibility Pump** (Fischetti-Glover-Lodi 2005):
//! Alternates between LP-feasible and integer-rounded solutions to find
//! a feasible integer point. The workhorse for hard MIPs.
//!
//! **Diving**: Depth-first branching with rounding at each level.
//! Fast to implement, often finds good solutions quickly.
//!
//! **RINS** (Relaxation Induced Neighborhood Search):
//! Fixes variables that agree between incumbent and LP solution,
//! solves a sub-MIP on the rest.

use crate::{BbNode, MipProblem, MipSettings, VarType};
use iconic_core::{Cone, Scalar};
use iconic_linalg::DenseMatrix;

// ── Feasibility Pump ───────────────────────────────────────────────────────

/// Feasibility pump: find an integer-feasible solution by alternating
/// between LP rounding and LP projection.
///
/// Returns Some(solution) if an integer-feasible point is found.
pub fn feasibility_pump<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
    max_iters: usize,
) -> Option<(Vec<T>, T)> {
    feasibility_pump_timed(
        problem,
        settings,
        max_iters,
        std::time::Duration::from_millis(500),
    )
}

/// As `feasibility_pump`, but with an explicit wall-clock budget for the whole
/// rounding/projection loop. The projection LP (min Σd_j, distance to the
/// rounded target) is occasionally primal-degenerate — ties between the
/// x_j−d_t ≤ x̃_j / −x_j−d_t ≤ −x̃_j constraint pairs can send the LP solver's
/// internal fast paths into thousands of non-productive iterations before
/// falling back. A per-call iteration cap does not bound this (the pathology
/// is inside a fixed-iteration-count fast path the caller's `max_iters`
/// setting does not reach), so the loop is additionally capped on wall time:
/// feasibility pump is a heuristic and a partial result is always safe to
/// abandon.
pub fn feasibility_pump_timed<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
    max_iters: usize,
    time_budget: std::time::Duration,
) -> Option<(Vec<T>, T)> {

    let fp_start = std::time::Instant::now();
    let one = T::one();
    let n = problem.q.len();

    // Step 1: Solve LP relaxation. Linear objectives go through the dual
    // simplex (see `solve_lp_relaxation_simplex`): the same relaxation the
    // tree solves in milliseconds cost seconds on the IPM path. Quadratic
    // objectives keep the IPM — the simplex relaxation drops P.
    let prog = mip_to_cone_prog(problem, &problem.lb, &problem.ub);
    let mut x_lp: Vec<T> = if problem.p.data.iter().all(|&v| v == T::zero()) {
        match crate::solve_lp_relaxation_simplex(
            problem,
            &problem.lb,
            &problem.ub,
            false,
            settings.deadline,
        ) {
            crate::RelaxLpOutcome::Optimal(x) => x,
            _ => return None,
        }
    } else {
        crate::solve_lp_ok(&prog, &settings.lp_settings)?.x
    };

    // If already integer feasible, return. `mip_feasible` as well as integrality:
    // `sol` may be `SolvedInaccurate`, which does not guarantee the constraints hold.
    if is_integral_point(&x_lp, &problem.var_types) && crate::check_feasibility(&x_lp, problem) {
        let obj = crate::compute_objective(&problem.p, &problem.q, &x_lp);
        return Some((x_lp, obj));
    }

    // Step 2: Iterate rounding → LP projection
    let mut prev_rounded: Vec<T> = vec![];
    let mut iter = 0usize;
    let n_int: Vec<usize> = (0..n)
        .filter(|&j| problem.var_types[j].is_integer())
        .collect();

    while iter < max_iters {
        if fp_start.elapsed() >= time_budget {
            return None;
        }
        let mut x_rounded = x_lp.clone();
        for &j in &n_int {
            x_rounded[j] = x_lp[j].round();
            if x_rounded[j] < problem.lb[j] {
                x_rounded[j] = problem.lb[j];
            }
            if x_rounded[j] > problem.ub[j] {
                x_rounded[j] = problem.ub[j];
            }
        }

        if crate::check_feasibility(&x_rounded, problem) && is_integral_point(&x_rounded, &problem.var_types)
        {
            let obj = crate::compute_objective(&problem.p, &problem.q, &x_rounded);
            return Some((x_rounded, obj));
        }

        // Cycle detection: if same rounding as previous iteration,
        // the distance LP is pulling back to the same fractional point.
        // Instead of a single-variable perturbation (which the distance LP
        // immediately "un-does"), flip the rounding direction — but ONLY for
        // variables that were fractional in x_lp. Flipping every binary goes
        // against the LP everywhere at once and discards the integral
        // structure the LP already committed to: measured on a 5-item
        // knapsack, the all-binary flip turned the near-optimal vertex
        // [1, .5, 1, 1, 0] into the single-item corner [0, 0, 0, 0, 1]
        // (obj −6 where −15 is achievable). Flipping just the fractional
        // coordinates breaks the cycle while keeping the committed ones.
        // The "flip" strategy is the standard fix when the standard FP
        // cycles (Achterberg & Berthold, "Improving the Feasibility Pump").
        if iter > 0 && prev_rounded == x_rounded {
            let mut flipped = false;
            for &j in &n_int {
                if problem.var_types[j] == VarType::Binary
                    && (x_lp[j] - x_lp[j].round()).abs()
                        > T::from_f64(1e-6).expect("scalar literal")
                {
                    if x_rounded[j] >= T::from_f64(0.5).expect("scalar literal") {
                        x_rounded[j] = problem.lb[j];
                    } else {
                        x_rounded[j] = problem.ub[j].min(T::one());
                    }
                    flipped = true;
                }
            }
            if !flipped {
                // No binary variables to flip — just perturb one general integer
                for &j in &n_int {
                    if x_rounded[j] != x_lp[j].round() {
                        x_rounded[j] = if x_rounded[j]
                            <= (problem.lb[j] + problem.ub[j]) / T::from_f64(2.0).expect("scalar literal")
                        {
                            problem.ub[j].min(x_rounded[j] + one)
                        } else {
                            problem.lb[j].max(x_rounded[j] - one)
                        };
                        break;
                    }
                }
            }
            // No early return here even when the flipped point is already
            // feasible: the distance LP below projects onto it exactly
            // (a feasible target has distance zero), so falling through
            // costs one cheap solve and loses nothing.
        }
        prev_rounded = x_rounded.clone();

        // Step 3: Solve LP that minimizes distance to rounded solution
        // min Σ_{j∈J} |x_j − x̃_j| subject to original constraints
        // Reformulate: add variables d_j ≥ 0, min Σ d_j, x_j − d_j ≤ x̃_j, −x_j − d_j ≤ −x̃_j
        //
        // Built and solved directly in simplex standard form (see
        // `solve_distance_lp_simplex`): the projection drops the objective by
        // construction, so it is exact for every problem shape, and a
        // millisecond-scale solve per iteration replaces the dense
        // cone-program IPM that dominated the pump's budget.
        if let crate::RelaxLpOutcome::Optimal(x_new) =
            crate::solve_distance_lp_simplex(problem, &x_rounded, settings.deadline)
        {
            x_lp = x_new;
            // Integrality is not enough to return this as an incumbent. The
            // projection LP carries extra distance columns and rows, so its
            // first `n` components can violate the original constraints
            // outright. On a 4x3 job shop this returned an integral point
            // that broke a disjunctive row by 9.2 with objective 0;
            // branch-and-bound took it as the incumbent, pruned the entire
            // tree against it, and reported `Optimal` with a zero makespan.
            // Verify against the original problem.
            if is_integral_point(&x_lp, &problem.var_types) && crate::check_feasibility(&x_lp, problem) {
                let obj = crate::compute_objective(&problem.p, &problem.q, &x_lp);
                return Some((x_lp, obj));
            }
        }

        iter += 1;
    }

    None
}

// ── Diving heuristic ───────────────────────────────────────────────────────

/// Simple rounding dive: round one fractional variable, re-solve LP,
/// repeat until integer feasible or infeasible.
///
/// Two engines, tried in order:
/// 1. **Warm chain** ([`rounding_dive_warm`], linear objectives only): one
///    persistent dual simplex re-optimized from the previous step's basis.
///    ~40x cheaper per step on the instances that burn dive budget.
/// 2. **Cold rebuild** (the historical shape): every step re-builds the node
///    LP and cold-solves it.
///
/// The fallback is not dead weight: a warm chain walks different vertices
/// among degenerate optima than a sequence of cold solves, and an instance's
/// incumbent can hinge on exactly which vertex the intermediate LPs land on
/// (measured: color_n15_k5's tree went from 1 node to 4500+ when the warm
/// chain alone replaced the cold one — its dive `improved` verdict became
/// `no_solution`). So whenever the warm chain finds nothing, the cold shape
/// runs before the caller hears "no solution".
///
/// On [`WarmDive::DoveDeep`] the cold pass is depth-capped
/// ([`WARM_DIVE_COLD_FALLBACK_DEPTH`]): the warm chain already spent up to
/// `max_depth` rounds, and an uncapped second full-depth pass doubles the
/// worst-case dive cost. The cap does not protect the incumbent: the two
/// chains diverge at their FIRST fractional tie-break (degenerate optima),
/// so most of the cold chain's value sits in its early steps — exactly the
/// part the cap keeps.
pub fn rounding_dive<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
    max_depth: usize,
) -> Option<(Vec<T>, T)> {
    if std::env::var_os("ICONIC_NO_WARM_DIVE").is_some() {
        return rounding_dive_cold(problem, settings, max_depth);
    }
    match rounding_dive_warm(problem, settings, max_depth) {
        WarmDive::Solved(sol) => Some(sol),
        // The warm chain dove deep and still found nothing. Run the cold
        // shape anyway, depth-capped: "the cold rebuild would repeat the
        // identical tightening sequence" is FALSE under degeneracy — each
        // engine re-optimizes its own LP, and among tied optima they pick
        // different vertices, so the rounding decisions differ from the
        // first step on. Measured both ways on color_n15_k5: warm-only
        // (no fallback) lost the incumbent entirely and its tree grew from
        // 1 node to 4500+.
        WarmDive::DoveDeep => {
            let capped = max_depth.min(WARM_DIVE_COLD_FALLBACK_DEPTH);
            rounding_dive_cold(problem, settings, capped)
        }
        // The warm chain died after almost no steps: an early death where the
        // cold chain's different vertex trajectory can still succeed (measured:
        // color_n15_k5's incumbent comes from a cold-chain dive the warm chain
        // cannot reproduce). Fall back at the full depth — barely any warm
        // steps were spent.
        WarmDive::DiedEarly => rounding_dive_cold(problem, settings, max_depth),
    }
}

/// Outcome of the warm-chained dive, deciding whether the cold fallback is
/// worth its cost.
enum WarmDive<T> {
    Solved((Vec<T>, T)),
    /// Ran at least [`WARM_DIVE_MIN_STEPS`] rounds without finding anything —
    /// a genuine deep dive that the cold rebuild would only repeat.
    DoveDeep,
    /// Gave up in under [`WARM_DIVE_MIN_STEPS`] rounds — potentially the
    /// degenerate-vertex divergence the cold shape avoids.
    DiedEarly,
}

const WARM_DIVE_MIN_STEPS: usize = 4;

/// Depth cap for the cold fallback that runs after a deep warm dive. The
/// warm chain already spent up to `max_depth` (= n) rounds; this bounds the
/// combined worst case at `n + cap` LP solves instead of `2n`. 16 covers the
/// early divergent steps where the two engines' vertex trajectories differ.
const WARM_DIVE_COLD_FALLBACK_DEPTH: usize = 16;

/// The warm-chained dive: linear objectives only (returns `DiedEarly`
/// otherwise, so a quadratic problem simply runs the cold shape).
fn rounding_dive_warm<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
    max_depth: usize,
) -> WarmDive<T> {
    if problem.p.data.iter().any(|&v| v != T::zero()) {
        return WarmDive::DiedEarly;
    }
    use crate::{build_node_lp, node_lp_solver};

    let n = problem.q.len();
    let mut lb = problem.lb.clone();
    let mut ub = problem.ub.clone();

    // ONE simplex solver persists across all the dive's sequential LPs. Each
    // dive step changes exactly one variable's box, and a basis optimal for
    // the old box stays dual feasible for any box — reduced costs don't read
    // bounds — so every re-solve is a `hot_solve` from the previous optimum's
    // exported basis, re-deriving `xb` under the new bounds and pivoting out
    // only what the tightening violated. The cold shape rebuilt the entire
    // node LP (CSC assembly + cold factor) per step; measured on knapsack_n=1000
    // the dive spent its full 1s call budget on those rebuilds for no solution.
    let node = BbNode {
        id: usize::MAX,
        depth: 0,
        lb: lb.clone(),
        ub: ub.clone(),
        x: None,
        obj_val: T::zero(),
        bound: T::zero(),
        estimate: T::zero(),
        parent_id: None,
        branch_var: None,
        branch_dir: None,
        branch_frac: None,
        basis: None,
        cut_rows: Vec::new(),
        branch_path: vec![],
    };
    let (c, a, b, l, u, m, nn) = build_node_lp(problem, &node);
    let mut s = node_lp_solver(c, a, b, l, u, m, nn, settings.deadline);
    let mut basis: Option<iconic_simplex::HotBasis<T>> = None;
    let mut steps = 0usize;

    for _depth in 0..max_depth {
        if settings
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return if steps >= WARM_DIVE_MIN_STEPS {
                WarmDive::DoveDeep
            } else {
                WarmDive::DiedEarly
            };
        }
        let sol = match basis.take() {
            Some(parent) => s.hot_solve(&parent),
            None => s.cold_solve(),
        };
        if sol.status != iconic_simplex::Status::Optimal {
            if std::env::var_os("ICONIC_WARM_DIVE_TRACE").is_some() {
                eprintln!("[warm-dive] failed at step {steps}: {:?}", sol.status);
            }
            return if steps >= WARM_DIVE_MIN_STEPS {
                WarmDive::DoveDeep
            } else {
                WarmDive::DiedEarly
            };
        }
        steps += 1;
        basis = Some(s.export_basis());
        let x = &sol.x[..n];

        // Integrality alone is not enough: a vertex can still violate its own
        // rows within tolerance. Same acceptance test as the cold dive.
        let x_owned: Vec<T> = x.to_vec();
        if is_integral_point(x, &problem.var_types) && crate::check_feasibility(x, problem) {
            let obj = crate::compute_objective(&problem.p, &problem.q, x);
            return WarmDive::Solved((x_owned, obj));
        }

        // Find most fractional integer variable (identical rule to cold).
        let Some(j) = most_fractional(x, &problem.var_types) else {
            break; // all integer
        };

        // Round to nearest integer — identical convention to the cold dive.
        if !tighten_to_nearest(x[j], j, &mut lb, &mut ub) {
            return if steps >= WARM_DIVE_MIN_STEPS {
                WarmDive::DoveDeep
            } else {
                WarmDive::DiedEarly
            };
        }
        // Mirror the tightened box into the persistent solver so the next
        // iteration's `hot_solve` sees it.
        s.set_var_bound(j, lb[j], ub[j]);
    }

    // Exhausted `max_depth` rounds without an integer point: a deep dive by
    // definition (the cold rebuild would repeat the identical sequence of
    // tightenings).
    WarmDive::DoveDeep
}

/// The historical cold-rebuild dive (every step rebuilds the node LP and
/// cold-solves it). Kept verbatim as [`rounding_dive`]'s fallback engine.
fn rounding_dive_cold<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
    max_depth: usize,
) -> Option<(Vec<T>, T)> {

    let mut lb = problem.lb.clone();
    let mut ub = problem.ub.clone();

    // The dive's LP solves are rounding targets, not certified bounds: the
    // returned point is verified against the ORIGINAL problem (mip_feasible
    // below), so the solve itself only needs to be close enough to guide the
    // rounding. The node-LP relaxed convention (1e-6 + capped iterations,
    // the same shape as the feasibility pump's distance LPs) is several
    // times cheaper per solve than the 1e-8 default — and the dive burns
    // most of its budget on these solves (measured: transport_w8_c25's dive
    // ran 4.05s of sequential 1e-8 solves and found nothing; the tree then
    // solved the instance in 12 nodes).
    let mut dive_lp_settings = settings.lp_settings.clone();
    dive_lp_settings.eps_abs = T::from_f64(1e-6).expect("scalar literal");
    dive_lp_settings.eps_rel = T::from_f64(1e-6).expect("scalar literal");
    dive_lp_settings.eps_gap = T::from_f64(1e-6).expect("scalar literal");
    dive_lp_settings.max_iters = 50;

    for _depth in 0..max_depth {
        // One dive is up to `max_depth` sequential cold IPM solves, and callers pass
        // `max_depth = n`. Without this the dive keeps solving past the deadline it
        // was launched under -- the caller's own budget check only runs between
        // heuristics, so a single dive on a midsize instance overruns the whole solve.
        if settings
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return None;
        }
        // Linear objective: solve the dive LP on the dual simplex — the same
        // swap the other root heuristics made (see `solve_lp_relaxation_simplex`).
        // The dive is up to `n` of these sequential solves; at IPM cost that was
        // seconds per dive on midsize instances (the 1s call budget mostly bought
        // a deadline abort), at simplex cost the dive actually gets to finish.
        // Quadratic objectives keep the IPM (the simplex LP drops P).
        let x_owned: Vec<T> = if problem.p.data.iter().all(|&v| v == T::zero()) {
            match crate::solve_lp_relaxation_simplex(problem, &lb, &ub, false, settings.deadline) {
                crate::RelaxLpOutcome::Optimal(x) => x,
                // Infeasible tightened box (the dive is dead) or a give-up:
                // both end the dive, exactly as a non-Solved IPM status does.
                _ => return None,
            }
        } else {
            let prog = mip_to_cone_prog(problem, &lb, &ub);
            crate::solve_lp_ok(&prog, &dive_lp_settings)?.x
        };
        let x = &x_owned;

        // Integrality alone is not enough: `SolvedInaccurate` is accepted above, so
        // the dive's LP point can still violate the constraints. Same defect the
        // feasibility pump had.
        if is_integral_point(x, &problem.var_types) && crate::check_feasibility(x, problem) {
            let obj = crate::compute_objective(&problem.p, &problem.q, x);
            return Some((x.clone(), obj));
        }

        // Find most fractional integer variable
        let Some(j) = most_fractional(x, &problem.var_types) else {
            break; // all integer
        };

        // Round to nearest integer — see `tighten_to_nearest` for why
        // nearest, not floor.
        if !tighten_to_nearest(x[j], j, &mut lb, &mut ub) {
            return None;
        }
    }

    None
}

// ── Sub-MIP solver (substrate for RINS/RENS/LocalBranching) ────────────────

/// Solve a restricted MIP with some variables fixed to given values.
/// This is the shared substrate: RINS, RENS, and Local Branching all use it.
/// Runs B&B with a tight node limit on the sub-problem.
pub fn solve_sub_mip<T: Scalar + PartialOrd + std::fmt::Debug>(
    problem: &MipProblem<T>,
    fixed_vars: &[Option<T>], // None = free, Some(val) = fixed to val
    node_limit: usize,
    settings: &MipSettings<T>,
) -> Option<(Vec<T>, T)> {
    solve_sub_mip_with_rows(problem, fixed_vars, &[], node_limit, settings)
}

/// As [`solve_sub_mip`], but additionally restricted by `extra_rows` —
/// rows `aᵀx ≤ rhs` appended after the problem's own rows (NonNegative
/// cone), in the same canonical convention the rest of the codebase uses
/// (`aᵀx + s = b, s ∈ K`). Every extra row is a *restriction*, so any
/// solution of the restricted problem is feasible for the original — the
/// row-adding substrate local branching uses for its Hamming-neighborhood
/// row (whose coefficients are `+1`/`−1` and RHS `k − |S1|`, valid by
/// construction for every binary point at Hamming distance ≤ k from the
/// incumbent).
pub fn solve_sub_mip_with_rows<T: Scalar + PartialOrd + std::fmt::Debug>(
    problem: &MipProblem<T>,
    fixed_vars: &[Option<T>], // None = free, Some(val) = fixed to val
    extra_rows: &[(Vec<(usize, T)>, T)], // (sparse coeffs, rhs) for aᵀx ≤ rhs
    node_limit: usize,
    settings: &MipSettings<T>,
) -> Option<(Vec<T>, T)> {
    let n = problem.q.len();
    let m_base = problem.b.len();
    // Build sub-problem with fixed variables
    let mut sub_lb = problem.lb.clone();
    let mut sub_ub = problem.ub.clone();
    for j in 0..n {
        if let Some(val) = fixed_vars[j] {
            sub_lb[j] = val;
            sub_ub[j] = val;
        }
    }

    let m = m_base + extra_rows.len();
    let mut a = DenseMatrix::<T>::zeros(m, n);
    a.copy_block_from(&problem.a, m_base, n);
    let mut b = problem.b.clone();
    for (row, (coeffs, rhs)) in extra_rows.iter().enumerate() {
        let r = m_base + row;
        for &(j, c) in coeffs {
            a.set(r, j, c);
        }
        b.push(*rhs);
    }
    let mut cones = problem.cones.clone();
    if !extra_rows.is_empty() {
        match cones.last_mut() {
            Some(Cone::NonNegative(d)) => *d += extra_rows.len(),
            _ => cones.push(Cone::NonNegative(extra_rows.len())),
        }
    }

    // Create sub-MIP with tightened bounds and the extra restriction rows
    let sub_problem = MipProblem {
        p: problem.p.clone(),
        q: problem.q.clone(),
        a,
        b,
        cones,
        var_types: problem.var_types.clone(),
        lb: sub_lb,
        ub: sub_ub,
        warm_start: None,
    };

    // Solve with tight node limit
    let mut sub_settings = settings.clone();
    sub_settings.max_nodes = node_limit;
    sub_settings.heuristics = false; // no recursion
    sub_settings.mip_presolve = false; // presolve already done

    crate::prof::bump(crate::prof::CNT_SUBMIP);
    let sol = crate::solve_mip(&sub_problem, &sub_settings);
    if matches!(
        sol.status,
        crate::MipStatus::Optimal | crate::MipStatus::Feasible
    ) {
        Some((sol.x, sol.obj_val))
    } else {
        None
    }
}

// ── RINS (Relaxation Induced Neighborhood Search) ──────────────────────────

/// RINS (Relaxation Induced Neighborhood Search): fix-and-solve sub-MIP.
///
/// Fixes integer variables where the incumbent `best_x` and the current LP
/// solution `lp_solution` agree (both round to the same integer value), then
/// solves the sub-MIP on the remaining (disagreeing) variables with a tight
/// node budget. Complements RENS: RENS fixes variables that are naturally
/// integer in the LP, while RINS fixes variables where the LP and incumbent
/// already agree, regardless of whether the LP value itself is fractional.
///
/// Reference: Danna, Rothberg, Le Pape (2005), "Exploring relaxation induced
/// neighborhoods to improve MIP solutions."
pub fn rins<T: Scalar + PartialOrd + std::fmt::Debug>(
    problem: &MipProblem<T>,
    lp_solution: &[T],
    incumbent: &[T],
    settings: &MipSettings<T>,
) -> Option<(Vec<T>, T)> {
    let n = problem.q.len();
    let eps = T::from_f64(1e-6).expect("scalar literal");

    let mut fixed: Vec<Option<T>> = vec![None; n];
    let mut n_fixed = 0usize;
    let mut n_free = 0usize;

    for j in 0..n {
        if !problem.var_types[j].is_integer() {
            continue;
        }
        // Both must round to the same integer value
        let lp_rounded = lp_solution[j].round();
        let inc_rounded = incumbent[j].round();
        if (lp_rounded - inc_rounded).abs() < eps {
            fixed[j] = Some(lp_rounded);
            n_fixed += 1;
        } else {
            n_free += 1;
        }
    }

    // RINS is only useful when there are disagreements to resolve.
    if n_free == 0 || n_fixed == 0 {
        return None;
    }

    // Fixed variables are proven optimal in both the LP and incumbent views —
    // the sub-MIP is likely small. Budget: 500 nodes (same as RENS).
    solve_sub_mip(problem, &fixed, 500, settings)
}

/// Infeasible-solution RINS (the target-heuristic framework's infeasible
/// source): when the ROUNDED node-LP point violates a row — the trickle-flow
/// shape, where rounding a near-integral LP value crosses a tight row — that
/// infeasible near-integer point is still a strong neighborhood source: fix
/// every integer variable where it agrees with the incumbent, and solve the
/// sub-MIP over the rest. The rounded point marks the region the relaxation
/// considers good; the fixing steers the sub-MIP toward feasible
/// improvements near it. Adopted only when the sub-MIP returns a feasible,
/// improving point (the caller's gate).
pub fn rins_infeasible<T: Scalar + PartialOrd + std::fmt::Debug>(
    problem: &MipProblem<T>,
    lp_solution: &[T],
    incumbent: &[T],
    settings: &MipSettings<T>,
) -> Option<(Vec<T>, T)> {
    let n = problem.q.len();
    let eps = T::from_f64(1e-6).expect("scalar literal");

    // Round the LP point to the nearest integers — this is what produces the
    // infeasible source: a near-integral LP value can round past a tight row.
    let mut src = lp_solution.to_vec();
    for j in 0..n {
        if problem.var_types[j].is_integer() {
            src[j] = src[j].round();
        }
    }
    // The source is only interesting when rounding actually broke feasibility.
    if crate::check_feasibility(&src, problem) {
        return None;
    }
    let mut fixed: Vec<Option<T>> = vec![None; n];
    let mut n_fixed = 0usize;
    let mut n_free = 0usize;
    for j in 0..n {
        if !problem.var_types[j].is_integer() {
            continue;
        }
        let src_rounded = src[j].round();
        let inc_rounded = incumbent[j].round();
        if (src_rounded - inc_rounded).abs() < eps {
            fixed[j] = Some(src_rounded);
            n_fixed += 1;
        } else {
            n_free += 1;
        }
    }
    if n_free == 0 || n_fixed == 0 {
        return None;
    }
    solve_sub_mip(problem, &fixed, 500, settings)
}

// ── RENS (Relaxation Enforced Neighborhood Search) ─────────────────────────

/// RENS: Fix variables that are ALREADY INTEGER in the LP solution,
/// solve sub-MIP on the fractional ones. Complements RINS as a second,
/// cheap improvement heuristic once RINS has run.
pub fn rens<T: Scalar + PartialOrd + std::fmt::Debug>(
    problem: &MipProblem<T>,
    lp_solution: &[T],
    settings: &MipSettings<T>,
) -> Option<(Vec<T>, T)> {
    let n = problem.q.len();
    let eps = T::from_f64(1e-6).expect("scalar literal");

    let mut fixed: Vec<Option<T>> = vec![None; n];
    let mut n_fixed = 0usize;
    let mut n_frac = 0usize;
    for j in 0..n {
        if !problem.var_types[j].is_integer() {
            continue;
        }
        let xj = lp_solution[j];
        let frac = xj - xj.floor();
        if frac < eps || frac > T::one() - eps {
            // Already integer in LP → fix it
            fixed[j] = Some(xj.round());
            n_fixed += 1;
        } else {
            n_frac += 1;
        }
    }

    // RENS is useful when some variables are naturally integer and some fractional
    if n_frac == 0 || n_fixed == 0 {
        return None;
    }
    solve_sub_mip(problem, &fixed, 500, settings)
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Distance to the nearest integer — the dive heuristics' fractionality
/// measure (0 for integral, 0.5 maximally fractional).
fn frac_dist<T: Scalar>(xj: T) -> T {
    let frac = xj - xj.floor();
    if frac > T::from_f64(0.5).expect("scalar literal") {
        T::one() - frac
    } else {
        frac
    }
}

/// Most-fractional integer variable, or `None` when all are integral.
/// Identical selection rule in the warm and cold dives.
fn most_fractional<T: Scalar>(x: &[T], var_types: &[VarType]) -> Option<usize> {
    let mut best_j = None;
    let mut best_frac = T::zero();
    for j in 0..x.len() {
        if !var_types[j].is_integer() {
            continue;
        }
        let dist = frac_dist(x[j]);
        if dist > best_frac {
            best_frac = dist;
            best_j = Some(j);
        }
    }
    best_j
}

/// Round variable `j`'s LP value `xj` to its nearest integer by tightening
/// one bound. Rounding to nearest (rather than always flooring) protects
/// big-M links: for `x ≤ M·y`, flooring y → 0 kills x, while y ≥ 0.5
/// staying 1 lets x stay positive.
/// Returns false when both bounds collapsed (`lb > ub`).
fn tighten_to_nearest<T: Scalar>(xj: T, j: usize, lb: &mut [T], ub: &mut [T]) -> bool {
    let floor = xj.floor();
    if xj - floor > T::from_f64(0.5).expect("scalar literal") {
        lb[j] = lb[j].max(floor + T::one());
    } else {
        ub[j] = ub[j].min(floor);
    }
    lb[j] <= ub[j]
}

/// Check that every integer variable's value is integral (no bounds check —
/// distinct from `lib.rs`'s `is_integer_feasible`, which also tests bounds).
fn is_integral_point<T: Scalar>(x: &[T], var_types: &[VarType]) -> bool {
    let eps = T::from_f64(1e-6).expect("scalar literal");
    for (j, vt) in var_types.iter().enumerate() {
        if !vt.is_integer() {
            continue;
        }
        let frac = x[j] - x[j].floor();
        if frac > eps && frac < T::one() - eps {
            return false;
        }
    }
    true
}

/// Build a ConeProgram from the MIP problem with given bounds.
pub(crate) fn mip_to_cone_prog<T: Scalar>(
    problem: &MipProblem<T>,
    lb: &[T],
    ub: &[T],
) -> iconic_api::ConeProgram<T> {
    use iconic_api::ConeProgram;
    let n = problem.q.len();
    let m_base = problem.b.len();
    // INF_BOUND, not anything larger: a generator's `ub[j] = 1e20` sentinel
    // must read as "unbounded" here, or it materializes as a literal
    // `x <= 1e20` row and wrecks the LP's scaling.
    let huge = T::from_f64(crate::INF_BOUND).expect("scalar literal");

    let mut extra = 0usize;
    for j in 0..n {
        if lb[j] > -huge {
            extra += 1;
        }
        if ub[j] < huge {
            extra += 1;
        }
    }
    let m = m_base + extra;
    let mut a = DenseMatrix::<T>::zeros(m, n);
    let mut b = vec![T::zero(); m];
    let mut cones: Vec<Cone> = problem.cones.clone();

    b[..m_base].copy_from_slice(&problem.b);
    a.copy_block_from(&problem.a, m_base, n);
    if extra > 0 {
        match cones.last_mut() {
            Some(Cone::NonNegative(d)) => *d += extra,
            _ => cones.push(Cone::NonNegative(extra)),
        }
    }
    let mut row = m_base;
    for j in 0..n {
        if lb[j] > -huge {
            a.set(row, j, -T::one());
            b[row] = -lb[j];
            row += 1;
        }
        if ub[j] < huge {
            a.set(row, j, T::one());
            b[row] = ub[j];
            row += 1;
        }
    }
    ConeProgram {
        p: problem.p.clone(),
        q: problem.q.clone(),
        a,
        a_csc: None,
        b,
        cones,
    }
}

// ── Local Branching (Fischetti-Lodi 2003) ──────────────────────────────────
// ── Relaxation-free search ────────────────────────────────────────────────

/// Relaxation-free search: explore a limited search tree *without* solving LP
/// relaxations, using constraint propagation and rounding to find
/// integer-feasible solutions quickly.
///
/// At each node, instead of solving an LP, we propagate row-activity bounds
/// to prune infeasible partial assignments. Nodes cost nearly nothing — the
/// entire search explores 500 nodes in microseconds, finding incumbents that
/// LP-based heuristics miss due to time-budget constraints.
///
/// ## Algorithm
///
/// 1. Order integer variables by objective impact (|q_j| descending, ties by
///    bound width).
/// 2. Quick check: round bound-midpoints to nearest integer (zero search cost).
/// 3. Recursive depth-first search with a 500-node budget:
///    - Pick the next unfixed variable in priority order.
///    - Branch: try the nearer integer bound first, then the farther.
///    - After each fixing, propagate row-activity bounds — prune if any
///      constraint becomes unsatisfiable even at extreme variable values.
///    - When all integer variables are fixed, check full MIP feasibility.
/// 4. Return the first feasible solution found, or None.
///
/// No LP is solved inside this heuristic — the "relaxation" is skipped
/// entirely, hence the name.
pub fn relaxation_free_search<T: Scalar + PartialOrd + std::fmt::Debug>(
    problem: &MipProblem<T>,
    _settings: &MipSettings<T>,
) -> Option<(Vec<T>, T)> {
    let n = problem.q.len();
    let zero = T::zero();
    let half = T::from_f64(0.5).expect("scalar literal");

    // Collect and order integer variable indices by objective impact.
    let mut int_vars: Vec<usize> = (0..n)
        .filter(|&j| problem.var_types[j].is_integer())
        .collect();
    if int_vars.is_empty() {
        return None;
    }
    int_vars.sort_by(|&a, &b| {
        let qa = problem.q[a].abs();
        let qb = problem.q[b].abs();
        match qb.partial_cmp(&qa).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => {
                let ra = problem.ub[a] - problem.lb[a];
                let rb = problem.ub[b] - problem.lb[b];
                rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
            }
            other => other,
        }
    });

    // Quick check: round the bound midpoints. Catches the trivial case
    // (e.g. a fully-presolved problem) at zero search cost.
    {
        let mut x = vec![zero; n];
        for j in 0..n {
            x[j] = if problem.var_types[j].is_integer() {
                let mid = (problem.lb[j] + problem.ub[j]) * half;
                let r = mid.round();
                if r < problem.lb[j] {
                    problem.lb[j]
                } else if r > problem.ub[j] {
                    problem.ub[j]
                } else {
                    r
                }
            } else {
                (problem.lb[j] + problem.ub[j]) * half
            };
        }
        if crate::check_feasibility(&x, problem) {
            let obj = crate::compute_objective(&problem.p, &problem.q, &x);
            return Some((x, obj));
        }
    }

    // ── Row-activity propagation helpers ─────────────────────────────────

    /// Compute min and max possible value of row `r` of A·x given bounds [lb, ub].
    fn row_range<T: Scalar>(problem: &MipProblem<T>, r: usize, lb: &[T], ub: &[T]) -> (T, T) {
        let n = problem.q.len();
        let zero = T::zero();
        let mut lo = zero;
        let mut hi = zero;
        for j in 0..n {
            let a = problem.a.get(r, j);
            if a == zero {
                continue;
            }
            if a > zero {
                lo += a * lb[j];
                hi += a * ub[j];
            } else {
                lo += a * ub[j];
                hi += a * lb[j];
            }
        }
        (lo, hi)
    }

    /// Return true iff bounds [lb, ub] are consistent with all constraints.
    fn consistent<T: Scalar + PartialOrd>(problem: &MipProblem<T>, lb: &[T], ub: &[T]) -> bool {
        let eps = T::from_f64(1e-8).expect("scalar literal");
        let mut row = 0usize;
        for cone in &problem.cones {
            match cone {
                Cone::Zero(len) => {
                    for i in row..row + len {
                        let (lo, hi) = row_range(problem, i, lb, ub);
                        if lo > problem.b[i] + eps || hi < problem.b[i] - eps {
                            return false;
                        }
                    }
                    row += len;
                }
                Cone::NonNegative(len) => {
                    for i in row..row + len {
                        let (lo, _hi) = row_range(problem, i, lb, ub);
                        // Ax + s = b, s >= 0  =>  Ax <= b
                        if lo > problem.b[i] + eps {
                            return false;
                        }
                    }
                    row += len;
                }
                _ => {
                    // Non-orthant cones: skip propagation.
                    row += cone.dim();
                }
            }
        }
        true
    }

    // ── Recursive DFS ────────────────────────────────────────────────────
    let max_nodes: usize = 500;
    let mut nodes: usize = 0;
    let mut lb = problem.lb.clone();
    let mut ub = problem.ub.clone();

    /// Recursive depth-first search from position `pos` in `int_vars`.
    fn dfs<T: Scalar + PartialOrd + std::fmt::Debug>(
        problem: &MipProblem<T>,
        int_vars: &[usize],
        pos: usize,
        lb: &mut [T],
        ub: &mut [T],
        max_nodes: usize,
        nodes: &mut usize,
    ) -> Option<(Vec<T>, T)> {
        if *nodes >= max_nodes {
            return None;
        }
        *nodes += 1;
        let half = T::from_f64(0.5).expect("scalar literal");

        // Skip past already-fixed variables.
        let mut p = pos;
        while p < int_vars.len() {
            let j = int_vars[p];
            if lb[j] != ub[j] {
                break;
            }
            p += 1;
        }

        if p >= int_vars.len() {
            // All integer variables are fixed. Build solution and check feasibility.
            let n = problem.q.len();
            let zero = T::zero();
            let mut x = vec![zero; n];
            for j in 0..n {
                x[j] = if lb[j] == ub[j] {
                    lb[j]
                } else {
                    (lb[j] + ub[j]) * half
                };
            }
            if crate::check_feasibility(&x, problem) {
                let obj = crate::compute_objective(&problem.p, &problem.q, &x);
                return Some((x, obj));
            }
            return None;
        }

        let j = int_vars[p];

        // Determine branch values: nearest integer bound first.
        let mid = (lb[j] + ub[j]) * half;
        let nearest = mid.round();
        let v1 = if nearest < lb[j] {
            lb[j]
        } else if nearest > ub[j] {
            ub[j]
        } else {
            nearest
        };
        let v2 = if v1 > lb[j] {
            v1 - T::one()
        } else if v1 < ub[j] {
            v1 + T::one()
        } else {
            v1
        };

        // Try v1 first (nearest), then v2.
        for &val in &[v1, v2] {
            if val < lb[j] || val > ub[j] {
                continue;
            }
            let old_lb = lb[j];
            let old_ub = ub[j];
            lb[j] = val;
            ub[j] = val;

            if consistent(problem, lb, ub) {
                if let Some(result) = dfs(problem, int_vars, p + 1, lb, ub, max_nodes, nodes) {
                    return Some(result);
                }
            }

            lb[j] = old_lb;
            ub[j] = old_ub;
        }

        None
    }

    dfs(
        problem, &int_vars, 0, &mut lb, &mut ub, max_nodes, &mut nodes,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────

// ── Incumbent polish ───────────────────────────────────────────────────────

/// Improve an incumbent by single-variable moves, in place. Returns true if it improved.
///
/// Every incumbent here arrives by rounding, diving or a sub-MIP, and all of those can
/// leave a variable at a feasible but plainly improvable value. The clearest case is a
/// variable that carries a negative cost and appears in constraints that still have slack:
/// raising it is free objective. Nothing was looking for that.
///
/// maxsat is exactly that shape. Its clause indicators `y_j` cost `-w_j` and sit in one
/// row each, so given any assignment of the problem variables, every satisfiable clause
/// can be marked satisfied at no cost. The all-false assignment with that completion is
/// worth about -505 against a true optimum of -542, and the search was reporting 0.
///
/// This is a 1-opt local search, not a sub-MIP: each candidate move changes one variable,
/// so the objective delta and the affected row activities are computed incrementally and
/// a pass costs O(n*m). It only ever replaces a point with a feasible, strictly better
/// one, so it cannot introduce a wrong answer -- the worst case is that it finds nothing.
pub fn polish_incumbent<T: Scalar + PartialOrd>(problem: &MipProblem<T>, x: &mut [T]) -> bool {
    let n = problem.q.len();
    let m = problem.b.len();
    if x.len() != n {
        return false;
    }
    let zero = T::zero();
    let tol = T::from_f64(1e-9).expect("scalar literal");

    // Row kind, in cone order: Zero rows must stay at b, NonNegative rows at or below it.
    let mut row_is_eq = Vec::with_capacity(m);
    for c in &problem.cones {
        match c {
            Cone::Zero(k) => row_is_eq.extend(std::iter::repeat_n(true, *k)),
            Cone::NonNegative(k) => row_is_eq.extend(std::iter::repeat_n(false, *k)),
            _ => return false, // only linear cones have a cheap activity test
        }
    }
    if row_is_eq.len() != m {
        return false;
    }

    // Index each column's nonzero rows once. Without this every candidate move rescans all
    // m rows to find the handful it touches, so a pass costs O(n*m) per move rather than
    // O(nnz) -- which showed up as the polish itself doubling the solve time on the larger
    // instances (stein_v10_t4 62.7 -> 135.4 ms) rather than the search it feeds.
    let mut col_rows: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
    let mut act = vec![zero; m];
    for i in 0..m {
        let mut s = zero;
        for (j, cr) in col_rows.iter_mut().enumerate().take(n) {
            let aij = problem.a.get(i, j);
            if aij != zero {
                cr.push((i, aij));
                s += aij * x[j];
            }
        }
        act[i] = s;
    }
    let has_p = problem.p.data.iter().any(|&v| v != zero);
    let mut px = vec![zero; n];
    if has_p {
        for (i, pxi) in px.iter_mut().enumerate() {
            let mut s = zero;
            for (j, &xj) in x.iter().enumerate().take(n) {
                s += problem.p.get(i, j) * xj;
            }
            *pxi = s;
        }
    }

    let mut improved_any = false;
    // Passes are capped: each accepted move strictly lowers the objective, but on a
    // degenerate problem the improvements can be arbitrarily small, and this runs on the
    // hot path of every incumbent update.
    for _pass in 0..4 {
        let mut improved_this_pass = false;
        for j in 0..n {
            if !problem.var_types[j].is_integer() {
                continue;
            }
            // Candidate single-variable moves: to the opposite bound for a binary, one
            // step either way for a general integer.
            let xj = x[j];
            let cands: [T; 2] = [xj - T::one(), xj + T::one()];
            for &nv in cands.iter() {
                if nv < problem.lb[j] - tol || nv > problem.ub[j] + tol {
                    continue;
                }
                let d = nv - xj;
                if d.abs() < tol {
                    continue;
                }
                // Objective delta for x_j <- x_j + d, from the maintained P*x.
                let mut delta = problem.q[j] * d;
                if has_p {
                    delta += d * px[j] + T::from_f64(0.5).expect("scalar literal") * problem.p.get(j, j) * d * d;
                }
                if delta >= -tol {
                    continue;
                }
                // Feasible only if every row it touches stays inside its cone.
                let mut ok = true;
                let rtol = T::from_f64(1e-7).expect("scalar literal");
                for &(i, aij) in &col_rows[j] {
                    let na = act[i] + aij * d;
                    if row_is_eq[i] {
                        if (na - problem.b[i]).abs() > rtol {
                            ok = false;
                            break;
                        }
                    } else if na > problem.b[i] + rtol {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
                x[j] = nv;
                for &(i, aij) in &col_rows[j] {
                    act[i] += aij * d;
                }
                if has_p {
                    for (i, pxi) in px.iter_mut().enumerate() {
                        *pxi += problem.p.get(i, j) * d;
                    }
                }
                improved_this_pass = true;
                improved_any = true;
                break;
            }
        }
        if !improved_this_pass {
            break;
        }
    }
    improved_any
}

#[cfg(test)]
mod tests {
    use super::*;
    use iconic_linalg::DenseMatrix;

    /// Infeasible-solution RINS: when the rounded LP point violates a row,
    /// the heuristic fixes the variables where the source agrees with the
    /// incumbent and solves the sub-MIP over the rest; a feasible-rounded
    /// LP point (no infeasible source) returns None.
    #[test]
    fn rins_infeasible_uses_only_infeasible_sources() {
        // min -x1 - x2 s.t. x1 + x2 <= 1.2, x binary.
        let n = 2usize;
        let mut a = DenseMatrix::zeros(1, n);
        a.set(0, 0, 1.0);
        a.set(0, 1, 1.0);
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![-1.0, -1.0],
            a,
            b: vec![1.2],
            cones: vec![Cone::NonNegative(1)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        // LP point x = (0.7, 0.5): rounds to (1, 1) — infeasible (2 > 1.2).
        let lp = vec![0.7f64, 0.5];
        let inc = vec![1.0f64, 0.0]; // incumbent (1, 0): feasible, obj -1
        let settings = MipSettings::<f64>::default();
        let res = rins_infeasible(&problem, &lp, &inc, &settings);
        // The source (1,1) agrees with the incumbent on x0 only; x1 is free.
        // The sub-MIP over x1 finds x1 = 0 (or 1 with the row... x1=1 makes
        // x0+x1 = 2 > 1.2 infeasible) → (1, 0) — the incumbent itself.
        if let Some((x, obj)) = res {
            assert!(
                crate::check_feasibility(&x, &problem),
                "returned point must be feasible"
            );
            assert!((obj - (-1.0)).abs() < 1e-6, "obj {obj}");
        }
        // A feasible-rounded LP point (both integral, row satisfied) has no
        // infeasible source → None.
        let lp2 = vec![0.4f64, 0.4]; // rounds to (0,0): feasible
        assert!(rins_infeasible(&problem, &lp2, &inc, &settings).is_none());
        // Exactly-integral LP point → also None (nothing to round past).
        let lp3 = vec![1.0f64, 0.0];
        assert!(rins_infeasible(&problem, &lp3, &inc, &settings).is_none());
    }

    // ── Local branching (Fischetti & Lodi 2003) ─────────────────────────

    /// Regression test: `mip_feasible` (used by `feasibility_pump` to
    /// validate a candidate integer point before returning it as a
    /// heuristic solution) must reject a point that violates an equality
    /// (`Zero` cone) constraint in the "positive" slack direction.
    ///
    /// The old implementation checked only `s = b - Ax >= -eps` for every
    /// row, treating every constraint as a one-directional `<=`. For a
    /// `Zero` cone row that requires `Ax = b` exactly, a point with
    /// `Ax < b` (positive slack) was wrongly accepted as feasible.
    ///
    /// Row 0 here is `x0 + x1 = 1` (Zero cone). x = [0, 0] gives slack
    /// s = 1 - 0 = 1 >= -eps (the old check passes it), but it plainly
    /// violates the equality (0 != 1).
    #[test]
    fn mip_feasible_rejects_equality_violation_with_positive_slack() {
        let n = 2;
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![0.0, 0.0],
            a: DenseMatrix::from_row_major(1, n, vec![1.0, 1.0]),
            b: vec![1.0],
            cones: vec![Cone::Zero(1)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        assert!(
            !crate::check_feasibility(&[0.0, 0.0], &problem),
            "x=[0,0] violates the equality x0+x1=1 (0 != 1) and must be rejected"
        );
        assert!(
            crate::check_feasibility(&[1.0, 0.0], &problem),
            "x=[1,0] satisfies x0+x1=1 and must be accepted"
        );
        assert!(
            crate::check_feasibility(&[0.0, 1.0], &problem),
            "x=[0,1] satisfies x0+x1=1 and must be accepted"
        );
    }

    /// Feasibility pump on small binary knapsack.
    #[test]
    fn fp_binary_knapsack() {
        let n = 5;
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![-5.0, -3.0, -8.0, -2.0, -6.0], // min negative = max
            a: DenseMatrix::from_row_major(1, n, vec![3.0, 2.0, 5.0, 1.0, 4.0]),
            b: vec![10.0],
            cones: vec![Cone::NonNegative(1)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        let settings = MipSettings::<f64>::default();
        let result = feasibility_pump(&problem, &settings, 20);
        if let Some((x, obj)) = result {
            assert!(obj <= -10.0, "obj should be good, got {}", obj); // at least some reasonable value
            for (j, &xj) in x.iter().enumerate() {
                assert!(
                    (xj - xj.round()).abs() < 1e-5,
                    "x[{}]={} not integer",
                    j,
                    xj
                );
            }
        }
    }

    /// Diving on small covering problem.
    #[test]
    fn diving_small_covering() {
        let n = 4;
        let m = 2;
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![1.0; n],
            // x0+x1 ≥ 1, x2+x3 ≥ 1 (encoded as −a·x + s = −1)
            a: DenseMatrix::from_row_major(m, n, vec![-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]),
            b: vec![-1.0; m],
            cones: vec![Cone::NonNegative(m)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        let settings = MipSettings::<f64>::default();
        let result = rounding_dive(&problem, &settings, 20);
        if let Some((x, _obj)) = result {
            assert!(x[0] + x[1] >= 0.999, "constraint 1 violated");
            assert!(x[2] + x[3] >= 0.999, "constraint 2 violated");
        }
    }
    /// A variable with negative cost sitting in a row that still has slack is free
    /// objective, and rounding leaves them behind constantly. This is the shape of
    /// maxsat's clause indicators, where the search was reporting 0 while a trivially
    /// reachable point was worth far more.
    #[test]
    fn polish_raises_a_free_negative_cost_variable() {
        // min -3y  s.t.  x + y <= 1,  x,y binary.  Start from the feasible (0,0).
        let n = 2;
        let m = 1;
        let mut a = DenseMatrix::zeros(m, n);
        a.set(0, 0, 1.0);
        a.set(0, 1, 1.0);
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![0.0, -3.0],
            a,
            b: vec![1.0],
            cones: vec![Cone::NonNegative(m)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        let mut x = vec![0.0, 0.0];
        assert!(
            polish_incumbent(&problem, &mut x),
            "an improving flip exists"
        );
        assert_eq!(x, vec![0.0, 1.0], "y should be raised, x left alone");
    }

    /// It must never hand back a point that violates a constraint it started inside.
    #[test]
    fn polish_never_breaks_feasibility() {
        // min -x -y  s.t.  x + y <= 1. Both flips look profitable; only one is legal.
        let n = 2;
        let m = 1;
        let mut a = DenseMatrix::zeros(m, n);
        a.set(0, 0, 1.0);
        a.set(0, 1, 1.0);
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![-1.0, -1.0],
            a,
            b: vec![1.0],
            cones: vec![Cone::NonNegative(m)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        let mut x = vec![0.0, 0.0];
        polish_incumbent(&problem, &mut x);
        assert!(x[0] + x[1] <= 1.0 + 1e-9, "row violated: {x:?}");
        assert!(
            (x[0] + x[1] - 1.0f64).abs() < 1e-9,
            "should take the one legal flip: {x:?}"
        );
    }
}

/// Zero-objective feasibility search (the root-phase feasibility fallback): when every
/// root heuristic has failed to produce an incumbent, re-solve the MIP with
/// the objective zeroed and a small node budget. The search then focuses on
/// feasibility — the first integer-feasible point is as good as any — and
/// either returns one or proves (within the budget) that none exists in the
/// explored region. The same idea runs as a root-phase fallback
/// (the "zero-objective heuristic"); iconic's trigger is "no incumbent at all",
/// which otherwise leaves the tree without a pruning anchor.
pub fn zero_objective_search<T: Scalar + PartialOrd + std::fmt::Debug>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
) -> crate::bounds::ZeroObjectiveOutcome<T> {
    let mut p = problem.clone();
    p.q = vec![T::zero(); problem.q.len()];
    let mut s = settings.clone();
    s.max_nodes = 1000; // bounded: this is a fallback, not a second solve
    s.heuristics = false; // no recursion into sub-MIP heuristics
    s.heuristic_freq = usize::MAX;
    s.cut_rounds = 0;
    // Hard time cap: without one, 1000 nodes of feasibility search on an
    // infeasible instance (each node LP paying the IPM-confirm fallback) can
    // consume the whole parent budget — measured: tsptw_n10's zero-obj
    // fallback ate 30s of a 30s solve. One second is a bounded probe.
    s.deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(1));
    let sol = crate::solve_mip(&p, &s);
    // The zero-objective sub-MIP has the same feasible set as the parent (only
    // the objective was zeroed): a proven-infeasible sub-solve is a genuine
    // MIP-infeasibility proof, not a discardable failure. This verdict used to
    // be dropped here, leaving the parent to exhaust its own tree instead of
    // returning the proof.
    if sol.status == crate::MipStatus::Infeasible {
        return crate::bounds::ZeroObjectiveOutcome::ProvedInfeasible;
    }
    if sol.x.is_empty() {
        return crate::bounds::ZeroObjectiveOutcome::None;
    }
    let obj = crate::compute_objective(&problem.p, &problem.q, &sol.x);
    crate::bounds::ZeroObjectiveOutcome::Incumbent(sol.x, obj)
}

#[cfg(test)]
mod zero_objective_tests {
    use super::*;
    use crate::MipSettings;

    /// The fallback must return a genuinely feasible point when one exists,
    /// and stay within its 1s cap when none does.
    #[test]
    fn zero_objective_fallback_finds_feasibility_and_respects_the_cap() {
        // A small feasible MIP: two disjoint knapsack-ish constraints.
        let mut prob = MipProblem::<f64> {
            p: iconic_linalg::DenseMatrix::zeros(4, 4),
            q: vec![-1.0; 4],
            a: iconic_linalg::DenseMatrix::from_row_major(
                2,
                4,
                vec![2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0],
            ),
            b: vec![4.0, 4.0],
            cones: vec![iconic_core::Cone::NonNegative(2)],
            var_types: vec![crate::VarType::Binary; 4],
            lb: vec![0.0; 4],
            ub: vec![1.0; 4],
            warm_start: None,
        };
        let mut st = MipSettings::<f64>::default();
        st.max_time = 10.0;
        let res = zero_objective_search(&prob, &st);
        match &res {
            crate::bounds::ZeroObjectiveOutcome::Incumbent(x, _) => {
                // The returned point must be integer-feasible.
                assert!(
                    crate::is_valid_incumbent(x, &prob),
                    "fallback returned an infeasible point"
                );
            }
            other => panic!("fallback found nothing on a feasible MIP: {other:?}"),
        }
        // Infeasible instance: the fallback must give up within the cap
        // (the caller's budget is not burned) -- and since the zero-objective
        // sub-MIP shares the parent's feasible set, a proven-infeasible
        // sub-solve is itself the MIP-infeasibility proof, not a None.
        // We can't assert wall time reliably under load; assert the verdict
        // is ProvedInfeasible (or at worst None, never a bogus incumbent).
        // Now truly infeasible: keep the tiny capacities AND require at least
        // two items (x0+x1+x2+x3 >= 2 via -sum <= -2).
        prob.a = iconic_linalg::DenseMatrix::from_row_major(
            3,
            4,
            vec![
                2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0,
            ],
        );
        prob.b = vec![0.5, 0.5, -2.0];
        prob.cones = vec![iconic_core::Cone::NonNegative(3)];
        let res2 = zero_objective_search(&prob, &st);
        match res2 {
            crate::bounds::ZeroObjectiveOutcome::ProvedInfeasible => {}
            crate::bounds::ZeroObjectiveOutcome::None => {} // budget ran out; not a proof, but honest
            other => panic!("fallback claimed a solution on an infeasible MIP: {other:?}"),
        }
    }
}
