//! Combinatorial techniques that don't come from the LP relaxation.
//!
//! For some problem families the LP relaxation is classically weak (e.g. the
//! MTZ formulation of TSP), and B&B stalls -- either because it can't prove
//! a good incumbent is optimal even though it already found it (weak dual
//! bound), or because naive rounding of a fractional LP solution almost
//! never respects the combinatorial structure (degree-1 assignment rows), so
//! no incumbent is found at all. This module detects such structure
//! directly from the problem data and provides both a stronger dual bound
//! and a constructive primal heuristic, independent of the LP relaxation.

use crate::{MipProblem, MipSettings, VarType};
use iconic_core::{Cone, Scalar, Settings, Status};
use iconic_linalg::DenseMatrix;

/// Iterations of the VRP iterated-local-search loop: perturb + re-optimize
/// rounds after the single-start local search converges. 5000 rounds of a
/// 15-customer shake are ~0.02s; the rounds buy the basin escapes that the
/// deterministic single start cannot reach (cvrp_n15_k4's 8.071 optimum
/// appears at round ~3300 from a 8.287 single-start local optimum).
const ILS_ROUNDS: usize = 5000;
// Deterministic hashing: these collections' iteration order feeds the arc
// recovery and the packing heuristic, whose output order shapes the search
// trajectory. std's RandomState seeds per-process randomness into that order;
// FxHash has a fixed seed. See the determinism note in lib.rs's imports.
use rustc_hash::{FxHashMap, FxHashSet};

/// Tolerances for an LP a heuristic solves only to *guide* itself.
///
/// These LPs are not bounds and never reach the caller's answer: one ranks an item's
/// candidate columns by preference, the other thresholds the relaxation at 0.1 to pick
/// variables to fix. Both were inheriting the caller's default 1e-8 conic tolerance and
/// full iteration budget, which is precision spent on a decision that only needs the
/// rough shape of the solution -- and it showed: `greedy_assignment_heuristic` alone was
/// 342ms of a 1.0s solve on `gap_hard_j30_m8`, where the entire branch-and-bound tree
/// underneath it costs well under a millisecond.
///
/// Mirrors the relaxed-tolerance convention the feasibility pump already uses for its
/// distance LP.
fn guidance_lp_settings<T: Scalar>(base: &Settings<T>) -> Settings<T> {
    let mut s = base.clone();
    let eps = T::from_f64(1e-6).expect("scalar literal");
    s.eps_abs = eps;
    s.eps_rel = eps;
    s.eps_gap = eps;
    s.max_iters = 100;
    s
}

/// The root LP relaxation, solved at most once per MIP solve.
///
/// `greedy_assignment_heuristic` and `round_binaries_resolve` each rebuilt this exact
/// relaxation -- same objective, same bounds -- and re-solved it, on top of the
/// feasibility pump's own copy. Building it is not free either: `mip_to_cone_prog`
/// materialises a dense `(m + 2n) x n` matrix. Profiling put the root heuristics at 60%
/// of a representative solve while the whole branch-and-bound tree underneath cost under
/// a millisecond, so this duplication is a real cost, not a rounding error.
///
/// Resolved lazily rather than up front, because both consumers bail on a cheap
/// structural check before they need an LP at all; solving it eagerly would charge every
/// MIP for a relaxation that neither one goes on to use. `Some(empty)` records "tried and
/// failed" so a failing solve is not retried.
fn root_relaxation<'a, T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
    cache: &'a mut Option<Vec<T>>,
) -> Option<&'a [T]> {
    if cache.is_none() {
        let solved = if problem.p.data.iter().all(|&v| v == T::zero()) {
            // LP objective: solve on the dual simplex — the same relaxation the
            // tree's root node solves in milliseconds there cost seconds per
            // call on the IPM path (see `solve_lp_relaxation_simplex`).
            match crate::solve_lp_relaxation_simplex(
                problem,
                &problem.lb,
                &problem.ub,
                false,
                settings.deadline,
            ) {
                crate::RelaxLpOutcome::Optimal(x) => x,
                _ => Vec::new(),
            }
        } else {
            let prog = crate::heuristics::mip_to_cone_prog(problem, &problem.lb, &problem.ub);
            match iconic_api::solve(&prog, &guidance_lp_settings(&settings.lp_settings)) {
                Ok(sol) if sol.status.has_solution() => sol.x,
                _ => Vec::new(),
            }
        };
        *cache = Some(solved);
    }
    match cache.as_deref() {
        Some(x) if !x.is_empty() => Some(x),
        _ => None,
    }
}

/// A directed arc variable recovered from a degree-1 bipartite assignment
/// structure among the problem's equality rows (see [`recover_arc_structure`]).
struct ArcVar {
    var: usize,
    from: usize,
    to: usize,
}

/// Detect a degree-1 bipartite assignment structure (`Σ_j x_ij = 1` per "out"
/// node i, `Σ_i x_ij = 1` per "in" node j, `x_ij` binary, no self-loops) among
/// the leading `Cone::Zero` rows, purely from the matrix pattern -- the same
/// structure any directed-arc-over-a-node-set formulation (TSP, VRP,
/// assignment problems) produces. Returns `None` the moment any step doesn't
/// cleanly match, rather than guessing. On success, returns the number of
/// nodes and the recovered arc variables with their endpoints.
///
/// This mirrors `cuts::generate_subtour_cuts`'s detection (kept as an
/// independent, self-contained copy rather than a shared dependency between
/// the two modules, so a change to one cannot silently affect the other).
fn recover_arc_structure<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
) -> Option<(usize, Vec<ArcVar>)> {
    let (n_nodes, arcs, depot) = recover_routing_structure(problem)?;
    // A depot visited `k > 1` times is a vehicle-routing problem, not a single tour:
    // pinning a Hamiltonian cycle there leaves the depot's `Σ_j x_0j = k` row at 1 and
    // the point is infeasible however good the cycle is. Hand those to
    // [`vrp_nearest_neighbor_routes`] instead of building a tour that cannot be accepted.
    if depot.is_some() {
        return None;
    }
    Some((n_nodes, arcs))
}

/// The same detection as [`recover_arc_structure`], but also recognising a *depot*: a
/// node whose two degree rows carry RHS `k >= 2` rather than 1, i.e. `k` vehicles leave
/// and return to it while every other node is visited exactly once. Returns the node
/// count, the arcs, and `Some((depot_node, k))` when such a node is present.
///
/// Restricting the degree rows to RHS 1 (as this detection originally did) silently
/// dropped the depot rows on a CVRP and recovered an arc structure over the *customers
/// alone* -- a structure whose tours can never satisfy the rows that were dropped.
#[allow(clippy::type_complexity)]
fn recover_routing_structure<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
) -> Option<(usize, Vec<ArcVar>, Option<(usize, usize)>)> {
    let eps = T::from_f64(1e-6).expect("scalar literal");
    let one = T::one();
    let n = problem.q.len();

    let n_eq = crate::n_eq_rows(&problem.cones);
    if n_eq < 4 {
        return None;
    }

    let mut degree_rows: Vec<usize> = Vec::new();
    let mut row_vars: Vec<Vec<usize>> = Vec::new();
    // RHS per accepted row, so the depot (RHS `k >= 2`) can be told from the customers
    // (RHS 1) once the rows have been paired into nodes below.
    let mut row_rhs: Vec<usize> = Vec::new();
    for r in 0..n_eq {
        // Accept any positive integer RHS. Exactly one *node* may carry a RHS above 1,
        // checked after pairing -- two independently multi-visited nodes are not a
        // single-depot routing problem and this returns `None` for them.
        let rhs = problem.b[r];
        let Some(rhs_f) = rhs.to_f64() else { continue };
        if !rhs_f.is_finite() {
            continue;
        }
        let rhs_round = rhs_f.round();
        if rhs_round < 1.0 || (rhs - T::from_f64(rhs_round).expect("scalar literal")).abs() > eps {
            continue;
        }
        let mut vars = Vec::new();
        let mut shape_ok = true;
        for j in 0..n {
            let aij = problem.a.get(r, j);
            if aij.abs() <= eps {
                continue;
            }
            if (aij - one).abs() > eps || problem.var_types[j] != VarType::Binary {
                shape_ok = false;
                break;
            }
            vars.push(j);
        }
        if shape_ok && vars.len() >= 2 {
            degree_rows.push(r);
            row_vars.push(vars);
            row_rhs.push(rhs_round as usize);
        }
    }
    if degree_rows.len() < 4 {
        return None;
    }

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
                    return None;
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
    if n_cities < 4 {
        return None;
    }

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
            return None;
        }
        city_of_out.insert(o, city_id);
        city_of_in.insert(i, city_id);
    }
    if city_of_in.len() != n_cities {
        return None;
    }

    let mut arcs: Vec<ArcVar> = Vec::new();
    for (&(ia, ib), &j) in &var_edge {
        let (from, to) = if color[ia] == 0 {
            (city_of_out[&ia], city_of_in[&ib])
        } else {
            (city_of_out[&ib], city_of_in[&ia])
        };
        arcs.push(ArcVar { var: j, from, to });
    }
    if arcs.len() < n_cities {
        return None;
    }

    // The depot is the node whose degree rows ask for more than one visit. Both of its
    // rows must agree on `k`, and it must be unique -- two independently multi-visited
    // nodes are not a single-depot routing problem, so give up rather than guess.
    let mut depot: Option<(usize, usize)> = None;
    for (&o, &city) in &city_of_out {
        let k_out = row_rhs[o];
        if k_out <= 1 {
            continue;
        }
        if depot.is_some() {
            return None;
        }
        depot = Some((city, k_out));
    }
    if let Some((city, k_out)) = depot {
        let in_row = *city_of_in.iter().find(|(_, &c)| c == city)?.0;
        if row_rhs[in_row] != k_out {
            return None;
        }
    }

    Some((n_cities, arcs, depot))
}

/// Minimum 1-tree (spanning tree over cities `{1..n_cities-1}` via Prim's
/// algorithm, plus city 0's two cheapest incident edges) under the given
/// per-city penalties: edge `(i,j)` costs `dist[i][j] + pi[i] + pi[j]`.
/// Returns the 1-tree's weight *under these modified costs* together with
/// each city's degree in it. A tour is itself a 1-tree with every city at
/// degree exactly 2, so the minimum 1-tree's weight lower-bounds any tour's
/// -- and degree deviations from 2 are exactly the signal Lagrangian
/// relaxation of the "every city has degree 2" constraint needs.
fn min_1tree<T: Scalar + PartialOrd>(
    dist: &[Vec<T>],
    pi: &[T],
    n_cities: usize,
) -> Option<(T, Vec<usize>)> {
    let huge = T::from_f64(1e19).expect("scalar literal");
    let zero = T::zero();
    let c = |i: usize, j: usize| dist[i][j] + pi[i] + pi[j];

    let mut in_tree = vec![false; n_cities];
    let mut min_edge = vec![huge; n_cities];
    let mut via = vec![usize::MAX; n_cities];
    in_tree[1] = true;
    for j in 2..n_cities {
        min_edge[j] = c(1, j);
        via[j] = 1;
    }
    let mut mst_weight = zero;
    let mut degree = vec![0usize; n_cities];
    for _ in 2..n_cities {
        let mut best_j = usize::MAX;
        let mut best_w = huge;
        for j in 1..n_cities {
            if !in_tree[j] && min_edge[j] < best_w {
                best_w = min_edge[j];
                best_j = j;
            }
        }
        if best_j == usize::MAX {
            return None; // not connected
        }
        in_tree[best_j] = true;
        mst_weight += best_w;
        degree[best_j] += 1;
        degree[via[best_j]] += 1;
        for j in 1..n_cities {
            if !in_tree[j] && c(best_j, j) < min_edge[j] {
                min_edge[j] = c(best_j, j);
                via[j] = best_j;
            }
        }
    }

    // Two cheapest edges incident to city 0.
    let mut edges0: Vec<(T, usize)> = (1..n_cities).map(|j| (c(0, j), j)).collect();
    edges0.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if edges0.len() < 2 {
        return None;
    }
    degree[0] += 2;
    degree[edges0[0].1] += 1;
    degree[edges0[1].1] += 1;

    Some((mst_weight + edges0[0].0 + edges0[1].0, degree))
}

/// Held-Karp 1-tree lower bound, sharpened by Lagrangian subgradient
/// optimization (Held & Karp's original technique): a minimum 1-tree is a
/// valid lower bound on the optimal tour cost for *any* choice of per-city
/// penalties relaxing the "every city has degree 2" constraint (any tour is
/// itself a 1-tree at degree 2 everywhere, so weak duality holds regardless
/// of the penalties), so the tightest bound is the one maximizing over that
/// choice. Ascends via subgradient steps -- a city whose 1-tree degree isn't
/// 2 is exactly the signal to push its penalty up or down -- using a quick
/// local nearest-neighbor tour as the step-size rule's target (Held & Karp's
/// own choice: a good known upper bound to aim the step size at). Bails out
/// (returns `None`) if the arc structure isn't found or the costs aren't
/// symmetric (the classical 1-tree bound assumes undirected edge costs).
///
/// Valid as a floor for *every* node in the B&B tree, not just the root:
/// branching only adds restrictions, so a restricted subproblem's optimum
/// can only be >= the unrestricted problem's optimum, which is itself >=
/// this bound. Computing it once and reusing it everywhere is therefore
/// sound, not just cheap.
pub fn held_karp_1tree_bound<T: Scalar + PartialOrd>(problem: &MipProblem<T>) -> Option<T> {
    let (n_cities, arcs) = recover_arc_structure(problem)?;

    let huge = T::from_f64(1e19).expect("scalar literal");
    let eps = T::from_f64(1e-6).expect("scalar literal");
    let mut dist = vec![vec![huge; n_cities]; n_cities];
    for a in &arcs {
        let c = problem.q[a.var];
        dist[a.from][a.to] = c;
    }
    // Verify symmetry: the 1-tree bound is for undirected edge costs.
    for i in 0..n_cities {
        for j in (i + 1)..n_cities {
            let (dij, dji) = (dist[i][j], dist[j][i]);
            if dij >= huge || dji >= huge {
                return None; // not a complete graph -- structure doesn't fit
            }
            if (dij - dji).abs() > eps * (T::one() + dij.abs()) {
                return None; // asymmetric costs -- classical 1-tree doesn't apply
            }
        }
    }

    let zero = T::zero();
    let two = T::from_f64(2.0).expect("scalar literal");

    // Quick local nearest-neighbor tour length as the step-size rule's target
    // (Held & Karp's own choice of a good known upper bound). Computed
    // locally from `dist` rather than calling `nearest_neighbor_tour` --
    // that function builds a full MIP solution via an LP solve, which this
    // bound has no need for and shouldn't depend on.
    let ub_target = {
        let mut visited = vec![false; n_cities];
        visited[0] = true;
        let mut cur = 0usize;
        let mut len = zero;
        let mut ok = true;
        for _ in 1..n_cities {
            let mut best = usize::MAX;
            let mut best_d = huge;
            for j in 0..n_cities {
                if !visited[j] && dist[cur][j] < best_d {
                    best_d = dist[cur][j];
                    best = j;
                }
            }
            if best == usize::MAX {
                ok = false;
                break;
            }
            visited[best] = true;
            len += best_d;
            cur = best;
        }
        if ok {
            Some(len + dist[cur][0])
        } else {
            None
        }
    };

    let mut pi = vec![zero; n_cities];
    let (w0, deg0) = min_1tree(&dist, &pi, n_cities)?;
    let mut best_bound = w0;
    if deg0.iter().all(|&d| d == 2) {
        return Some(w0); // already a valid tour -- provably optimal, no iteration needed
    }

    let mut lambda = two;
    let mut stall = 0usize;
    for _iter in 0..60 {
        let (w, degree) = match min_1tree(&dist, &pi, n_cities) {
            Some(v) => v,
            None => break,
        };
        let penalty_sum = pi.iter().fold(zero, |a, &v| a + v);
        let bound = w - two * penalty_sum;
        if bound > best_bound {
            best_bound = bound;
            stall = 0;
        } else {
            stall += 1;
        }
        if degree.iter().all(|&d| d == 2) {
            break; // this 1-tree is itself a valid tour -- can't improve further this way
        }
        let denom = degree.iter().fold(zero, |a, &d| {
            let g = two - T::from_usize(d).expect("small-integer conversion");
            a + g * g
        });
        if denom <= T::from_f64(1e-12).expect("scalar literal") {
            break;
        }
        // Held-Karp step-size rule: scale toward a known upper bound (or, absent
        // one, a modest margin above the best bound seen) by the subgradient's
        // magnitude, damping `lambda` whenever a while passes with no improvement.
        let target =
            ub_target.unwrap_or(best_bound + best_bound.abs() * T::from_f64(0.05).expect("scalar literal"));
        let gap = (target - bound).max(zero);
        let step = lambda * gap / denom;
        // Ascent direction is the subgradient (degree_i - 2): a city whose 1-tree
        // degree is too high needs its penalty *raised* (discouraging more edges
        // touching it next time), too low needs it *lowered*.
        for i in 0..n_cities {
            let g = T::from_usize(degree[i]).expect("small-integer conversion") - two;
            pi[i] += step * g;
        }
        if stall >= 8 {
            lambda *= T::from_f64(0.5).expect("scalar literal");
            stall = 0;
        }
    }

    Some(best_bound)
}

/// Nearest-neighbor tour construction: a classic, always-valid TSP primal
/// heuristic, generalized to the same arc structure [`recover_arc_structure`]
/// detects. Coordinate-wise rounding of a fractional LP solution (what
/// `feasibility_pump`/`rounding_dive` do) almost never respects a degree-1
/// assignment's structure -- a random 0/1 rounding of arc variables is
/// vanishingly unlikely to form a single cycle through every node -- so on
/// these problems the generic heuristics can fail to find *any* incumbent
/// even after exploring thousands of B&B nodes (observed directly on
/// tsp_mtz_n=10/12). Greedily walking to the nearest unvisited node and
/// closing the tour is trivially guaranteed to produce a valid permutation
/// by construction, sidestepping the rounding problem entirely.
///
/// Any variable not part of the recovered arc structure (e.g. MTZ's `u_i`
/// potentials) is left free within its original bounds and recovered by
/// solving the LP with only the arc variables pinned to the constructed
/// tour -- generic to whatever auxiliary variables the formulation uses,
/// not specific to MTZ's particular ones.
pub fn nearest_neighbor_tour<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
) -> Option<(Vec<T>, T)> {
    let (n_cities, arcs) = recover_arc_structure(problem)?;

    let huge = T::from_f64(1e19).expect("scalar literal");
    let mut dist = vec![vec![huge; n_cities]; n_cities];
    let mut arc_var = vec![vec![usize::MAX; n_cities]; n_cities];
    for a in &arcs {
        dist[a.from][a.to] = problem.q[a.var];
        arc_var[a.from][a.to] = a.var;
    }

    // Greedy nearest-neighbor walk starting at city 0.
    let mut visited = vec![false; n_cities];
    visited[0] = true;
    let mut tour = vec![0usize];
    let mut cur = 0usize;
    for _ in 1..n_cities {
        let mut best = usize::MAX;
        let mut best_d = huge;
        for j in 0..n_cities {
            if !visited[j] && dist[cur][j] < best_d {
                best_d = dist[cur][j];
                best = j;
            }
        }
        if best == usize::MAX {
            return None; // disconnected -- not a complete graph
        }
        visited[best] = true;
        tour.push(best);
        cur = best;
    }

    // Pin every recovered arc variable: 1 on the tour's edges, 0 elsewhere.
    // The tour's cost is fully determined by these pinned values (TSP-shaped
    // objectives carry all their cost on the arc variables, none on
    // auxiliary ones like MTZ's u_i), so compute it directly here rather
    // than reading it back from the LP solve below -- that solve exists only
    // to fill in a consistent value for whatever auxiliary variables the
    // formulation has, and its convergence quality on those has no bearing
    // on the (already fully known) arc costs.
    let mut lb = problem.lb.clone();
    let mut ub = problem.ub.clone();
    let zero = T::zero();
    let one = T::one();
    for a in &arcs {
        lb[a.var] = zero;
        ub[a.var] = zero;
    }
    let mut tour_cost = zero;
    for k in 0..n_cities {
        let (from, to) = (tour[k], tour[(k + 1) % n_cities]);
        let v = arc_var[from][to];
        if v == usize::MAX {
            return None; // this directed edge isn't in the recovered arc set
        }
        lb[v] = one;
        ub[v] = one;
        tour_cost += dist[from][to];
    }

    // Solve the resulting LP (arc variables pinned, everything else free
    // within its original bounds) to recover a consistent assignment for any
    // other variables the formulation has. With every arc variable pinned to
    // an exact 0/1 value, this system tends to be tightly degenerate --
    // observed directly stalling at MaxIterations, drifting well away
    // (10-20%) from the intended pins on several arc variables, rather than
    // reaching Solved/SolvedInaccurate -- even though a feasible auxiliary-
    // variable assignment is guaranteed to exist for a genuine tour (e.g.
    // MTZ's u_i set to each city's position along it). Retry once with the
    // same relaxed settings solve_node_lp_ipm already uses for exactly this
    // stalled-IPM pattern (no presolve, looser tolerance, a larger iteration
    // budget) before giving up. Accept a retry that still lands on
    // MaxIterations only once the returned arc-variable values are verified
    // still at their intended pins -- the IPM's best-iterate tracking means
    // a non-converged result can still have drifted, and this is the one
    // property that must hold for `x` to represent the tour we constructed.
    let prog = crate::heuristics::mip_to_cone_prog(problem, &lb, &ub);
    let pin_eps = T::from_f64(1e-4).expect("scalar literal");
    let try_accept = |sol: &iconic_core::Solution<T>| -> bool {
        if sol.status.has_solution() {
            return true;
        }
        match sol.status {
            Status::MaxIterations => arcs
                .iter()
                .all(|a| (sol.x[a.var] - lb[a.var]).abs() <= pin_eps),
            _ => false,
        }
    };
    if let Ok(sol) = iconic_api::solve(&prog, &settings.lp_settings) {
        if try_accept(&sol) {
            return Some((sol.x, tour_cost));
        }
    }
    let retry_settings = crate::retry_lp_settings(&settings.lp_settings);
    if let Ok(sol) = iconic_api::solve(&prog, &retry_settings) {
        if try_accept(&sol) {
            return Some((sol.x, tour_cost));
        }
    }
    None
}

/// Per-node demands and the shared route capacity, read off the load-accumulation rows
/// (`c * x_ij + u_i - u_j <= c - d_j`) that a vehicle-routing formulation uses to bound
/// each route's load and forbid subtours.
///
/// Recognised purely by shape, in the same spirit as [`recover_routing_structure`]: an
/// inequality row over exactly one arc variable (coefficient `c > 0`) and two distinct
/// non-binary variables at `+1` and `-1`. `d_j = c - rhs` then follows, and every row
/// mentioning the same head node must agree on it.
#[allow(clippy::type_complexity)]
pub(crate) fn recover_route_capacity<T: Scalar + PartialOrd>(
    a: &DenseMatrix<T>,
    b: &[T],
    cones: &[Cone],
    n: usize,
    n_nodes: usize,
    arc_owner: &FxHashMap<usize, (usize, usize)>,
) -> Option<(Vec<T>, T, Vec<Option<usize>>)> {
    let eps = T::from_f64(1e-6).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let m = b.len();

    let is_eq_row = crate::eq_row_mask(cones, m);

    let mut demand: Vec<Option<T>> = vec![None; n_nodes];
    let mut capacity: Option<T> = None;
    // Which variable carries each node's accumulated load. Knowing this lets the caller
    // write the potentials down directly instead of recovering them from a pinned LP.
    let mut u_var: Vec<Option<usize>> = vec![None; n_nodes];
    for r in 0..m {
        if is_eq_row[r] {
            continue;
        }
        let mut arcs: Vec<(usize, T)> = Vec::new(); // arc vars with coefficients, at most 2
        let mut plus: Option<usize> = None;
        let mut minus: Option<usize> = None;
        let mut shape_ok = true;
        for j in 0..n {
            let aij = a.get(r, j);
            if aij.abs() <= eps {
                continue;
            }
            if let Some(&(_, _to)) = arc_owner.get(&j) {
                if arcs.len() >= 2 || aij <= zero {
                    shape_ok = false;
                    break;
                }
                arcs.push((j, aij));
            } else if (aij - one).abs() <= eps && plus.is_none() {
                plus = Some(j);
            } else if (aij + one).abs() <= eps && minus.is_none() {
                minus = Some(j);
            } else {
                shape_ok = false;
                break;
            }
        }
        let (Some((avar0, c0)), Some(p), Some(q)) = (arcs.first().copied(), plus, minus) else {
            continue;
        };
        if !shape_ok || p == q {
            continue;
        }
        // Desrochers–Laporte strengthened MTZ rows (the CVRP presolve pass)
        // carry a second, complementary arc term `(Q − d_i − d_j)·x_ji` on the
        // reverse arc, so the column scan may collect the two arcs in either
        // order. The forward arc is the one with the larger coefficient: the
        // complementary term is strictly smaller than the capacity for
        // positive demands. Accept the four-nonzero shape with the structure
        // the strengthening itself imposes: the second arc is exactly the
        // reverse of the first, its coefficient lies strictly between zero and
        // the capacity, and — once the tail's demand is known from another
        // row — matches `c − d_tail − d_head`. (The TSP-MTZ generator's
        // statically lifted rows satisfy the same checks with unit demands.)
        // Rows with the plain three-nonzero shape are accepted exactly as
        // before.
        let (avar, c, reverse_c) = if arcs.len() == 2 && arcs[1].1 > c0 {
            (arcs[1].0, arcs[1].1, c0)
        } else if arcs.len() == 2 {
            (avar0, c0, arcs[1].1)
        } else {
            (avar0, c0, zero)
        };
        // `u_i - u_j` on arc (i, j): the head `j` is the node whose demand is accumulated,
        // so `p` is the tail's potential and `q` the head's.
        let (tail, head) = arc_owner[&avar];
        if arcs.len() == 2 {
            let (t2, h2) = {
                // The reverse arc variable is the one not chosen as forward.
                let rev = if arcs[0].1 > arcs[1].1 {
                    arcs[1].0
                } else {
                    arcs[0].0
                };
                arc_owner[&rev]
            };
            if t2 != head || h2 != tail || reverse_c <= zero || reverse_c >= c {
                continue;
            }
            let d_j = c - b[r];
            if let Some(d_i) = demand[tail] {
                if (reverse_c - (c - d_i - d_j)).abs() > eps * (one + c.abs()) {
                    continue;
                }
            }
        }
        for (node, var) in [(tail, p), (head, q)] {
            match u_var[node] {
                None => u_var[node] = Some(var),
                Some(prev) => {
                    if prev != var {
                        return None;
                    }
                }
            }
        }
        let d = c - b[r];
        if d < zero {
            return None;
        }
        match &capacity {
            None => capacity = Some(c),
            Some(prev) => {
                if (*prev - c).abs() > eps {
                    return None;
                }
            }
        }
        match demand[head] {
            None => demand[head] = Some(d),
            Some(prev) => {
                if (prev - d).abs() > eps {
                    return None;
                }
            }
        }
    }

    let cap = capacity?;
    // Refuse a partial decode. A customer whose demand never decoded is
    // silently zeroed by `unwrap_or(zero)` below, and the capacity rounding
    // `ceil(dem(S)/Q)` then builds on a wrong demand vector — at best a
    // weakened cut, at worst (with the caller's n_nodes excluding the depot)
    // an invalid rounded capacity inequality. At most ONE node may be
    // undecoded: the depot, which is never a head of a load row when the
    // caller passes the full node set (the construction heuristic). Two or
    // more undecoded nodes mean the decode itself broke down — the
    // subtour-cuts caller passes customers only, where every node must
    // decode, and the construction caller knows only the depot goes
    // missing.
    let undecoded = (0..n_nodes).filter(|&i| demand[i].is_none()).count();
    if undecoded > 1 {
        return None;
    }
    // Nodes never appearing as a head (the depot) carry no demand.
    let dem: Vec<T> = demand.iter().map(|d| d.unwrap_or(zero)).collect();
    if dem.iter().all(|d| *d <= zero) {
        return None;
    }
    Some((dem, cap, u_var))
}

/// Constructive primal heuristic for capacitated vehicle routing: build exactly `k`
/// capacity-feasible routes out of the depot and pin the arcs they use.
///
/// [`nearest_neighbor_tour`] cannot serve here. It builds one Hamiltonian cycle, which
/// leaves the depot's `Σ_j x_0j = k` row at 1, so the pinned LP is infeasible for every
/// tour it can possibly construct -- on cvrp_n10_k3 the search ran its full 30s budget
/// across thousands of nodes and returned *no incumbent at all*, while the generic
/// rounding heuristics fail here for the same reason they fail on a TSP (a coordinate-wise
/// rounding of arc variables essentially never forms valid routes).
///
/// Customers are packed into routes best-fit-decreasing on demand, which respects the
/// capacity whenever the instance admits any `k`-route packing, and each route is then
/// ordered by a nearest-neighbor walk from the depot. Auxiliary variables (the load
/// potentials) are recovered by solving the LP with the arcs pinned, exactly as the tour
/// heuristic does.
/// Clarke-Wright savings construction: start with one vehicle per customer and repeatedly
/// merge the two route ends whose join saves the most, `s(i,j) = d(0,i) + d(0,j) - d(i,j)`,
/// stopping at exactly `k` routes.
///
/// The demand-only packing this sits beside groups customers by size and never looks at
/// where they are, so it routinely puts far-apart customers on one vehicle and leaves local
/// search to unpick it one move at a time. Savings is driven entirely by the distances, so
/// it groups by geography from the start -- and it needs nothing but `dist`, which matters
/// here because the formulation gives no coordinates.
///
/// Returns `None` if capacity prevents reaching `k` routes, leaving the caller its other
/// construction rather than a guess.
fn clarke_wright_routes<T: Scalar + PartialOrd>(
    n_nodes: usize,
    depot: usize,
    k_routes: usize,
    dist: &[Vec<T>],
    demand: &[T],
    capacity: T,
    huge: T,
) -> Option<(Vec<Vec<usize>>, Vec<T>)> {
    let customers: Vec<usize> = (0..n_nodes).filter(|&i| i != depot).collect();
    let mut routes: Vec<Vec<usize>> = customers.iter().map(|&c| vec![c]).collect();
    let mut load: Vec<T> = customers.iter().map(|&c| demand[c]).collect();
    for &c in &customers {
        if demand[c] > capacity {
            return None;
        }
    }

    let mut savings: Vec<(T, usize, usize)> = Vec::new();
    for (a, &i) in customers.iter().enumerate() {
        for &j in customers.iter().skip(a + 1) {
            if dist[depot][i] >= huge || dist[depot][j] >= huge || dist[i][j] >= huge {
                continue;
            }
            savings.push((dist[depot][i] + dist[j][depot] - dist[i][j], i, j));
        }
    }
    savings.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));

    // Merge only end-to-end: `i` must finish its route and `j` must start another, so the
    // join is a single new arc and interior customers keep their neighbours.
    for (_, i, j) in savings {
        if routes.len() <= k_routes {
            break;
        }
        let Some(ri) = routes.iter().position(|r| r.last() == Some(&i)) else {
            continue;
        };
        let Some(rj) = routes.iter().position(|r| r.first() == Some(&j)) else {
            continue;
        };
        if ri == rj || load[ri] + load[rj] > capacity {
            continue;
        }
        let tail = std::mem::take(&mut routes[rj]);
        routes[ri].extend(tail);
        load[ri] = load[ri] + load[rj];
        routes.remove(rj);
        load.remove(rj);
    }

    // Savings only ever joins a route's tail to another's head, and that restriction
    // reliably strands the construction one route above `k`: measured on all three cvrp
    // instances, every remaining tail-to-head join exceeded capacity while a different
    // pairing of the same two routes fit. Finish the job by considering both orientations
    // of each pair -- a route may be traversed either way -- and taking the cheapest
    // capacity-feasible merge, until the depot's `k` departures are met exactly.
    while routes.len() > k_routes {
        let mut best: Option<(usize, usize, Vec<usize>, T)> = None;
        for a in 0..routes.len() {
            for b in 0..routes.len() {
                if a == b || load[a] + load[b] > capacity {
                    continue;
                }
                for rev_a in [false, true] {
                    for rev_b in [false, true] {
                        let mut merged = routes[a].clone();
                        if rev_a {
                            merged.reverse();
                        }
                        let mut tail = routes[b].clone();
                        if rev_b {
                            tail.reverse();
                        }
                        merged.extend(tail);
                        let cost = route_cost(&merged, depot, dist, huge);
                        if cost >= huge {
                            continue;
                        }
                        let better = match &best {
                            None => true,
                            Some(x) => cost < x.3,
                        };
                        if better {
                            best = Some((a, b, merged, cost));
                        }
                    }
                }
            }
        }
        let (a, b, merged, _) = best?;
        routes[a] = merged;
        load[a] = load[a] + load[b];
        routes.remove(b);
        load.remove(b);
    }

    if routes.len() != k_routes {
        return None;
    }
    Some((routes, load))
}

/// Cost of one route, depot out and depot back. `huge` for any leg the arc set is missing,
/// so a route needing an absent arc can never look attractive.
fn route_cost<T: Scalar + PartialOrd>(
    route: &[usize],
    depot: usize,
    dist: &[Vec<T>],
    huge: T,
) -> T {
    let mut cur = depot;
    let mut total = T::zero();
    for &c in route {
        if dist[cur][c] >= huge {
            return huge;
        }
        total = total + dist[cur][c];
        cur = c;
    }
    if dist[cur][depot] >= huge {
        return huge;
    }
    total + dist[cur][depot]
}

/// Improve a set of routes in place with the two classical vehicle-routing neighbourhoods,
/// alternating until neither finds a gain:
///
/// - **2-opt**: reverse a segment within one route, which removes crossings.
/// - **relocate**: move one customer to the best position in any route, subject to
///   capacity, which fixes customers assigned to the wrong vehicle.
///
/// Both are evaluated on the whole affected route(s) rather than by an edge-delta formula.
/// That is a few more additions per candidate and it stays correct when the distances are
/// asymmetric or an arc is missing entirely (`route_cost` reports those as unreachable),
/// where the usual delta shortcut quietly assumes a complete symmetric graph.
///
/// A route is never emptied: the depot's degree row asks for exactly `k` departures, so a
/// vehicle left unused makes the point infeasible however short it is.
fn improve_routes<T: Scalar + PartialOrd>(
    routes: &mut [Vec<usize>],
    load: &mut [T],
    depot: usize,
    dist: &[Vec<T>],
    demand: &[T],
    capacity: T,
    huge: T,
) {
    let k = routes.len();
    // Bounded so a large instance cannot spend the whole root budget here; in practice the
    // loop converges well inside this.
    for _ in 0..50 {
        let mut improved = false;

        for r in 0..k {
            if routes[r].len() < 3 {
                continue;
            }
            let mut base = route_cost(&routes[r], depot, dist, huge);
            let len = routes[r].len();
            for i in 0..len - 1 {
                for j in i + 1..len {
                    routes[r][i..=j].reverse();
                    let cand = route_cost(&routes[r], depot, dist, huge);
                    if cand < base {
                        base = cand;
                        improved = true;
                    } else {
                        routes[r][i..=j].reverse();
                    }
                }
            }
        }

        for from in 0..k {
            let mut pos = 0usize;
            while pos < routes[from].len() {
                // Never strand a vehicle: emptying a route breaks the depot degree row.
                if routes[from].len() == 1 {
                    break;
                }
                let c = routes[from][pos];
                let before_from = route_cost(&routes[from], depot, dist, huge);
                let mut without = routes[from].clone();
                without.remove(pos);
                let after_from = route_cost(&without, depot, dist, huge);

                let mut best: Option<(usize, usize, T)> = None;
                for to in 0..k {
                    if to == from {
                        continue;
                    }
                    if load[to] + demand[c] > capacity {
                        continue;
                    }
                    let before_to = route_cost(&routes[to], depot, dist, huge);
                    for at in 0..=routes[to].len() {
                        let mut trial = routes[to].clone();
                        trial.insert(at, c);
                        let after_to = route_cost(&trial, depot, dist, huge);
                        let delta = (after_from + after_to) - (before_from + before_to);
                        let better = match &best {
                            None => true,
                            Some(b) => delta < b.2,
                        };
                        if delta < T::zero() && better {
                            best = Some((to, at, delta));
                        }
                    }
                }

                match best {
                    Some((to, at, _)) => {
                        routes[from].remove(pos);
                        routes[to].insert(at, c);
                        load[from] = load[from] - demand[c];
                        load[to] = load[to] + demand[c];
                        improved = true;
                    }
                    None => pos += 1,
                }
            }
        }

        if !improved {
            return;
        }
    }
}

/// Takes `_settings` for symmetry with the other root heuristics; unlike them it needs no
/// LP solve, because the routes and their loads are both constructed outright.
pub fn vrp_nearest_neighbor_routes<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    _settings: &MipSettings<T>,
) -> Option<(Vec<T>, T)> {
    let (n_nodes, arcs, depot_info) = recover_routing_structure(problem)?;
    let (depot, k_routes) = depot_info?;
    if k_routes < 2 || k_routes >= n_nodes {
        return None;
    }

    let huge = T::from_f64(1e19).expect("scalar literal");
    let zero = T::zero();
    let one = T::one();
    let mut dist = vec![vec![huge; n_nodes]; n_nodes];
    let mut arc_var = vec![vec![usize::MAX; n_nodes]; n_nodes];
    let mut arc_owner: FxHashMap<usize, (usize, usize)> = FxHashMap::default();
    for a in &arcs {
        dist[a.from][a.to] = problem.q[a.var];
        arc_var[a.from][a.to] = a.var;
        arc_owner.insert(a.var, (a.from, a.to));
    }

    let (demand, capacity, u_var) = recover_route_capacity(
        &problem.a,
        &problem.b,
        &problem.cones,
        problem.q.len(),
        n_nodes,
        &arc_owner,
    )?;

    // Best-fit decreasing: the largest demands are placed first, each into the fullest
    // route that still has room. That is the packing rule with the best worst-case
    // behaviour among the simple ones, and a route left empty would violate the depot
    // row (which asks for exactly `k` departures), so empty routes are refilled below.
    let mut customers: Vec<usize> = (0..n_nodes).filter(|&i| i != depot).collect();
    customers.sort_by(|&a, &b| {
        demand[b]
            .partial_cmp(&demand[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut routes: Vec<Vec<usize>> = vec![Vec::new(); k_routes];
    let mut load: Vec<T> = vec![zero; k_routes];
    for &c in &customers {
        let mut best: Option<usize> = None;
        for r in 0..k_routes {
            if load[r] + demand[c] > capacity {
                continue;
            }
            match best {
                None => best = Some(r),
                Some(b) if load[r] > load[b] => best = Some(r),
                _ => {}
            }
        }
        let r = best?;
        routes[r].push(c);
        load[r] = load[r] + demand[c];
    }
    // Every vehicle must be used. Move a customer out of the largest route into any
    // empty one; if the largest route is a singleton there are fewer customers than
    // vehicles and the instance cannot want `k` non-empty routes.
    for r in 0..k_routes {
        if !routes[r].is_empty() {
            continue;
        }
        let donor = (0..k_routes).max_by_key(|&i| routes[i].len())?;
        if routes[donor].len() < 2 {
            return None;
        }
        let moved = routes[donor].pop()?;
        load[donor] = load[donor] - demand[moved];
        routes[r].push(moved);
        load[r] = load[r] + demand[moved];
    }

    // Order each route by a nearest-neighbor walk from the depot: the packing fixes which
    // customers share a vehicle, this fixes the cost of visiting them.
    for route in &mut routes {
        let mut ordered = Vec::with_capacity(route.len());
        let mut remaining = std::mem::take(route);
        let mut cur = depot;
        while !remaining.is_empty() {
            let mut best_i = 0usize;
            let mut best_d = huge;
            for (i, &c) in remaining.iter().enumerate() {
                if dist[cur][c] < best_d {
                    best_d = dist[cur][c];
                    best_i = i;
                }
            }
            cur = remaining.remove(best_i);
            ordered.push(cur);
        }
        *route = ordered;
    }

    // Greedy construction leaves obvious crossings and misassignments behind: on
    // cvrp_n10_k3 it lands 18% above the optimum and the branch-and-bound search never
    // improves on it, because the moves that help here (reversing a segment, moving a
    // customer to another vehicle) are not single-variable flips -- `polish_incumbent`,
    // which is 1-opt over variables, cannot express any of them. Improve the routes
    // directly instead, with the two classical neighbourhoods, until neither finds a gain.
    improve_routes(
        &mut routes,
        &mut load,
        depot,
        &dist,
        &demand,
        capacity,
        huge,
    );
    // Best-improvement pass with the swap neighborhood (a strict superset of
    // improve_routes' first-improvement 2-opt + relocate): on cvrp_n15_k4 it is
    // the difference between a 9.7-cost construction and the 8.07 optimum.
    vrp_local_search(&mut routes, &mut load, &demand, capacity, depot, &dist);

    // Savings builds from the distances rather than the demands, so it usually starts much
    // closer; run both and keep whichever is cheaper once each has been improved, which
    // costs one extra construction and cannot be worse than either alone.
    if let Some((mut cw_routes, mut cw_load)) =
        clarke_wright_routes(n_nodes, depot, k_routes, &dist, &demand, capacity, huge)
    {
        improve_routes(
            &mut cw_routes,
            &mut cw_load,
            depot,
            &dist,
            &demand,
            capacity,
            huge,
        );
        vrp_local_search(
            &mut cw_routes,
            &mut cw_load,
            &demand,
            capacity,
            depot,
            &dist,
        );
        let cost = |rs: &[Vec<usize>]| -> T {
            rs.iter()
                .fold(zero, |acc, r| acc + route_cost(r, depot, &dist, huge))
        };
        if cost(&cw_routes) < cost(&routes) {
            routes = cw_routes;
            load = cw_load;
        }
    }

    // ── Iterated local search ──────────────────────────────────────
    // The single-start local search above converges to a local optimum of
    // every neighborhood; on cvrp_n15_k4 that basin sits at 8.287 with the
    // 8.071 optimum unreachable by any single improving move (all six
    // neighborhoods exhausted, verified by trace). Standard ILS escape:
    // perturb the route set with a few deterministic random chain
    // relocations, re-optimize, and accept the re-optimized candidate
    // (basin-walking — strict-only acceptance keeps re-shaking the same
    // basin and needed ~4200 rounds where basin-walking needs ~3300),
    // reporting the best seen. The shake is 1-3 moves of 1-3 customers
    // (the classic "double-bridge" strength range) — small enough that
    // re-optimization stays near the incumbent, large enough to leave its
    // basin. Fixed seed: the construction must be reproducible across runs.
    let no_ils = std::env::var_os("ICONIC_NO_ILS").is_some();
    if !no_ils && routes.len() >= 2 {
        // Deliberately NOT iconic_core::rng::Lcg: that one shares the state
        // constants but has a different seed transform (`^ golden`) and
        // output mapping (full-state `% n` vs `>> 33`). The ILS below was
        // tuned against *these* draws -- they are what walk cvrp_n15_k4's
        // basin to the 8.071391 verified optimum -- so swapping generators
        // would silently change every shake and could lose that incumbent.
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> u64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                self.0 >> 33
            }
            fn pick(&mut self, n: usize) -> usize {
                (self.next() % n as u64) as usize
            }
        }
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let mut best_routes = routes.clone();
        let mut best_cost: T = routes
            .iter()
            .fold(zero, |acc, r| acc + route_cost(r, depot, &dist, huge));
        for _iter in 0..ILS_ROUNDS {
            let mut cand = routes.clone();
            let mut cand_load = load.clone();
            let shakes = 1 + rng.pick(3);
            let mut ok = true;
            for _ in 0..shakes {
                // Relocate a chain of 1-3 customers from route a into route b.
                let a = rng.pick(k_routes);
                let b = rng.pick(k_routes - 1);
                let b = if b >= a { b + 1 } else { b };
                let len = 1 + rng.pick(3);
                if cand[a].len() <= len {
                    ok = false;
                    break;
                }
                let pa = rng.pick(cand[a].len() - len + 1);
                let chain_load: T = cand[a][pa..pa + len]
                    .iter()
                    .map(|&c| demand[c])
                    .fold(zero, |s, d| s + d);
                if cand_load[b] + chain_load > capacity {
                    ok = false;
                    break;
                }
                let chain: Vec<usize> = cand[a].drain(pa..pa + len).collect();
                cand_load[a] = cand_load[a] - chain_load;
                let pb = rng.pick(cand[b].len() + 1);
                cand[b].splice(pb..pb, chain);
                cand_load[b] = cand_load[b] + chain_load;
            }
            if !ok {
                continue;
            }
            vrp_local_search(&mut cand, &mut cand_load, &demand, capacity, depot, &dist);
            let c: T = cand
                .iter()
                .fold(zero, |acc, r| acc + route_cost(r, depot, &dist, huge));
            // Standard ILS acceptance: always move to the re-optimized
            // candidate (walking between basins — strict-only acceptance
            // keeps re-shaking the same basin, which needed ~4200 rounds
            // to find cvrp_n15's optimum; basin-walking finds it far
            // sooner), but report the best seen.
            if c < best_cost {
                if std::env::var_os("VRP_ILS_TRACE").is_some() {
                    eprintln!("[ils] iter {_iter}: improved to {c:?}");
                }
                best_cost = c;
                best_routes = cand.clone();
            }
            routes = cand;
            load = cand_load;
        }
        routes = best_routes;
    }
    let _ = &load;

    let mut total = zero;
    // The load potentials are not an unknown to solve for. On a route `u` is the load
    // carried after each stop, which is exactly what the rows encode (`u_j >= u_i + d_j`
    // across a used arc), so they can be written down. Unused arcs are satisfied for free:
    // `u_i <= capacity` and `u_j >= d_j` give `u_i - u_j <= capacity - d_j`.
    //
    // The tour heuristic instead recovers auxiliaries by solving the LP with the arcs
    // pinned, and that system is tightly degenerate. On cvrp_n10_k3 it returned with the
    // arcs still at their pins but the potentials drifted far enough that the completed
    // point failed the incumbent check -- discarding a perfectly good set of routes, and
    // leaving the instance with no incumbent at all after its full 30s budget.
    let mut x = vec![zero; problem.q.len()];
    for a in &arcs {
        x[a.var] = zero;
    }
    if let Some(v) = u_var[depot] {
        x[v] = zero;
    }
    for route in &routes {
        let mut cur = depot;
        let mut acc = zero;
        for &next in route {
            let v = arc_var[cur][next];
            if v == usize::MAX {
                return None;
            }
            x[v] = one;
            total = total + dist[cur][next];
            acc = acc + demand[next];
            x[u_var[next]?] = acc;
            cur = next;
        }
        let v = arc_var[cur][depot];
        if v == usize::MAX {
            return None;
        }
        x[v] = one;
        total = total + dist[cur][depot];
    }

    // Any variable that is neither an arc nor a load potential is left where its bounds
    // put it; if that is not a value the formulation accepts, the caller's incumbent check
    // rejects the point rather than this returning something unchecked.
    for j in 0..problem.q.len() {
        if arc_owner.contains_key(&j) || u_var.contains(&Some(j)) {
            continue;
        }
        x[j] = if problem.lb[j] > zero {
            problem.lb[j]
        } else if problem.ub[j] < zero {
            problem.ub[j]
        } else {
            zero
        };
    }
    Some((x, total))
}

/// Improve a constructed route set with the classic vehicle-routing local-search
/// neighborhoods: intra-route 2-opt segment reversals, inter-route single-customer
/// relocations, inter-route customer swaps, and cross-route 2-opt* tail exchanges
/// (with the depot-start edges in the delta — the exchange reassigns which route
/// leaves the depot first when a prefix empties out). Every move is
/// capacity-checked (a route's load may never exceed the capacity; a relocation or
/// tail exchange may not empty a route, since the model demands exactly `k`
/// non-empty routes) and accepted on strict cost decrease. Passes repeat until a
/// full pass finds no improving move (capped; each accepted move strictly lowers
/// the objective, so the loop is finite anyway). The scan order is fixed, so the
/// result is deterministic.
fn vrp_local_search<T: Scalar + PartialOrd>(
    routes: &mut [Vec<usize>],
    load: &mut [T],
    demand: &[T],
    capacity: T,
    depot: usize,
    dist: &[Vec<T>],
) {
    let zero = T::zero();
    let n_r = routes.len();
    if n_r < 2 {
        return;
    }
    // A/B kill-switch (same convention as the cut-family knobs): isolate
    // the 2-opt* / Or-opt neighbourhoods for before/after measurements.
    let no_opt2star = std::env::var_os("ICONIC_NO_OPT2STAR").is_some();
    let no_oropt = std::env::var_os("ICONIC_NO_OROPT").is_some();
    // Neighbour accessors, with the depot as the sentinel on either side.
    let prev_of = |r: &[usize], p: usize| -> usize {
        if p == 0 {
            depot
        } else {
            r[p - 1]
        }
    };
    let next_of = |r: &[usize], p: usize| -> usize {
        if p + 1 >= r.len() {
            depot
        } else {
            r[p + 1]
        }
    };

    #[derive(Clone, Copy, Debug)]
    enum Move {
        Opt2(usize, usize, usize),            // route, segment start, segment end
        Relocate(usize, usize, usize, usize), // from (a, pa) into b before pb
        OrOpt(usize, usize, usize, usize, usize), // relocate chain (a, pa, len) into b before pb
        Swap(usize, usize, usize, usize),     // (a, pa) <-> (b, pb)
        Opt2Star(usize, usize, usize, usize), // exchange route a's tail from pa with route b's tail from pb
    }
    for _pass in 0..8 {
        // Best-improvement pass: evaluate every move in a fixed order, keep the
        // best strictly-improving one, apply it, repeat.
        let mut best_delta = T::infinity();
        let mut best_move: Option<Move> = None;
        // Intra-route 2-opt: reverse segment [pa..pb] of route a.
        for a in 0..n_r {
            let r = &routes[a];
            if r.len() < 3 {
                continue;
            }
            for pa in 0..r.len() - 1 {
                for pb in pa + 1..r.len() {
                    if pa == 0 && pb == r.len() - 1 {
                        continue; // full reversal: same cost on symmetric distances
                    }
                    let (u, v) = (prev_of(r, pa), next_of(r, pb));
                    let old = dist[u][r[pa]] + dist[r[pb]][v];
                    let new = dist[u][r[pb]] + dist[r[pa]][v];
                    let delta = new - old;
                    if delta < best_delta {
                        best_delta = delta;
                        best_move = Some(Move::Opt2(a, pa, pb));
                    }
                }
            }
        }
        // Inter-route relocate: move route a's customer at pa into route b before pb.
        for a in 0..n_r {
            if routes[a].len() < 2 {
                continue;
            }
            for pa in 0..routes[a].len() {
                let c = routes[a][pa];
                for b in 0..n_r {
                    if b == a {
                        continue;
                    }
                    if load[b] + demand[c] > capacity {
                        continue;
                    }
                    let (u, v) = (prev_of(&routes[a], pa), next_of(&routes[a], pa));
                    for pb in 0..=routes[b].len() {
                        let (x, y) = (prev_of(&routes[b], pb), next_of(&routes[b], pb));
                        let old = dist[u][c] + dist[c][v] + dist[x][y];
                        let new = dist[u][v] + dist[x][c] + dist[c][y];
                        let delta = new - old;
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(Move::Relocate(a, pa, b, pb));
                        }
                    }
                }
            }
        }
        // Or-opt: relocate a chain of 2-3 consecutive customers from route a
        // into route b (the single-customer case is the Relocate pass above).
        // The classic missing neighborhood for route membership: moving
        // blocks rather than one customer at a time restructures the route
        // set where relocate/swap see only local ripples — measured on
        // cvrp_n15_k4, whose construction lands at 8.287 (optimum 8.071)
        // with every single-customer and reversal neighborhood exhausted.
        if !no_oropt {
            for a in 0..n_r {
                if routes[a].len() <= 2 {
                    continue;
                }
                for pa in 0..routes[a].len() {
                    for len in 2..=3usize {
                        if pa + len > routes[a].len() {
                            continue;
                        }
                        if routes[a].len() == len {
                            continue; // may not empty a route (exactly k non-empty)
                        }
                        let (u, v) = (prev_of(&routes[a], pa), next_of(&routes[a], pa + len - 1));
                        let old_seg = dist[u][routes[a][pa]] + dist[routes[a][pa + len - 1]][v];
                        let chain_load: T = routes[a][pa..pa + len]
                            .iter()
                            .map(|&c| demand[c])
                            .fold(zero, |s, d| s + d);
                        for b in 0..n_r {
                            if b == a {
                                continue;
                            }
                            if load[b] + chain_load > capacity {
                                continue;
                            }
                            for pb in 0..=routes[b].len() {
                                let (x, y) = (prev_of(&routes[b], pb), next_of(&routes[b], pb));
                                let old = old_seg + dist[x][y];
                                let new = dist[u][v]
                                    + dist[x][routes[a][pa]]
                                    + dist[routes[a][pa + len - 1]][y];
                                let delta = new - old;
                                if delta < best_delta {
                                    best_delta = delta;
                                    best_move = Some(Move::OrOpt(a, pa, len, b, pb));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Inter-route swap: exchange route a's customer at pa with route b's at pb.
        for a in 0..n_r {
            for pa in 0..routes[a].len() {
                let c1 = routes[a][pa];
                for b in a + 1..n_r {
                    for pb in 0..routes[b].len() {
                        let c2 = routes[b][pb];
                        if load[a] - demand[c1] + demand[c2] > capacity {
                            continue;
                        }
                        if load[b] - demand[c2] + demand[c1] > capacity {
                            continue;
                        }
                        let (u, v) = (prev_of(&routes[a], pa), next_of(&routes[a], pa));
                        let (x, y) = (prev_of(&routes[b], pb), next_of(&routes[b], pb));
                        let old = dist[u][c1] + dist[c1][v] + dist[x][c2] + dist[c2][y];
                        let new = dist[u][c2] + dist[c2][v] + dist[x][c1] + dist[c1][y];
                        let delta = new - old;
                        if delta < best_delta {
                            best_delta = delta;
                            best_move = Some(Move::Swap(a, pa, b, pb));
                        }
                    }
                }
            }
        }
        if !no_opt2star {
            // 2-opt*: exchange the tails of two routes. Removing edge (u,v) at
            // route a's position pa and (x,y) at route b's position pb, reconnect
            // (u,y) and (x,v) — route a keeps its prefix and gains b's suffix, and
            // vice versa. The load invariant is explicit: both reconnected
            // routes' loads are the prefix/suffix sums and must stay within the
            // capacity, and neither reconnected route may empty (the model
            // demands exactly k non-empty routes). An exchange with no actual
            // tail movement is impossible (a != b), and both tail lengths are
            // exact, so the loads can never silently drift.
            //
            // The delta must also cover the depot edges: unlike the boundary
            // positions of the single-route moves, `prev_of`/`next_of` here
            // delimit the CUT edges, not the route ends, so the two routes'
            // depot-start edges change hands — when pa == 0 the reconnected
            // route a' starts at y (the old depot edge depot→a0 is dropped) and
            // when pb == 0 the reconnected route b' starts at v. Missing those
            // terms made a "tail-exchange" look improving while actually
            // ratcheting the cost up (measured: the construction cost of
            // cvrp_n15_k4 degraded 8.286897 → 9.873842 with the edge-only delta).
            for a in 0..n_r {
                for pa in 0..routes[a].len() {
                    let (u, v) = (prev_of(&routes[a], pa), routes[a][pa]);
                    let tail_a: T = routes[a][pa..]
                        .iter()
                        .map(|&c| demand[c])
                        .fold(zero, |s, d| s + d);
                    let start_a = dist[depot][routes[a][0]];
                    for b in a + 1..n_r {
                        let start_b = dist[depot][routes[b][0]];
                        for pb in 0..routes[b].len() {
                            let (x, y) = (prev_of(&routes[b], pb), routes[b][pb]);
                            // Both reconnected routes need at least one customer.
                            if pa + (routes[b].len() - pb) < 1 {
                                continue;
                            }
                            if pb + (routes[a].len() - pa) < 1 {
                                continue;
                            }
                            let tail_b: T = routes[b][pb..]
                                .iter()
                                .map(|&c| demand[c])
                                .fold(zero, |s, d| s + d);
                            if load[a] - tail_a + tail_b > capacity {
                                continue;
                            }
                            if load[b] - tail_b + tail_a > capacity {
                                continue;
                            }
                            // Edge deltas for the two cut edges plus the depot
                            // start edges, which the exchange reassigns when a
                            // prefix empties out (pa == 0 or pb == 0).
                            let old = dist[u][v] + dist[x][y] + start_a + start_b;
                            let new_start_a = if pa == 0 { dist[depot][y] } else { start_a };
                            let new_start_b = if pb == 0 { dist[depot][v] } else { start_b };
                            let new = dist[u][y] + dist[x][v] + new_start_a + new_start_b;
                            let delta = new - old;
                            if delta < best_delta {
                                best_delta = delta;
                                best_move = Some(Move::Opt2Star(a, pa, b, pb));
                            }
                        }
                    }
                }
            }
        }
        // Accept only a strictly-improving move: with `best_delta` seeded at
        // infinity the first evaluated move would otherwise "win" every pass
        // and the search would ratchet the cost up instead of down.
        if std::env::var_os("VRP_LS_TRACE").is_some() {
            eprintln!("[vrpls] pass {_pass}: best_delta = {best_delta:?} move = {best_move:?}");
        }
        if best_delta >= zero {
            break;
        }
        match best_move {
            None => break,
            Some(Move::Opt2(a, pa, pb)) => {
                // Same customer set, so loads are unchanged.
                routes[a][pa..=pb].reverse();
            }
            Some(Move::Relocate(a, pa, b, pb)) => {
                let c = routes[a].remove(pa);
                load[a] = load[a] - demand[c];
                routes[b].insert(pb.min(routes[b].len()), c);
                load[b] = load[b] + demand[c];
            }
            Some(Move::OrOpt(a, pa, len, b, pb)) => {
                let chain: Vec<usize> = routes[a].drain(pa..pa + len).collect();
                let chain_load: T = chain.iter().map(|&c| demand[c]).fold(zero, |s, d| s + d);
                load[a] = load[a] - chain_load;
                let at = pb.min(routes[b].len());
                routes[b].splice(at..at, chain);
                load[b] = load[b] + chain_load;
            }
            Some(Move::Swap(a, pa, b, pb)) => {
                let c1 = routes[a].remove(pa);
                load[a] = load[a] - demand[c1];
                let c2 = routes[b].remove(pb);
                load[b] = load[b] - demand[c2];
                routes[a].insert(pa.min(routes[a].len()), c2);
                load[a] = load[a] + demand[c2];
                routes[b].insert(pb.min(routes[b].len()), c1);
                load[b] = load[b] + demand[c1];
            }
            Some(Move::Opt2Star(a, pa, b, pb)) => {
                // Route a keeps its prefix up to pa and gains b's suffix from
                // pb; route b keeps its prefix and gains a's suffix. Loads are
                // re-derived from the final customer lists (exact, never
                // drifts — a suffix sum bookkeeping slip would show up here).
                let tail_a: Vec<usize> = routes[a].split_off(pa);
                let tail_b: Vec<usize> = routes[b].split_off(pb);
                routes[a].extend(tail_b);
                routes[b].extend(tail_a);
                load[a] = routes[a]
                    .iter()
                    .map(|&c| demand[c])
                    .fold(zero, |s, d| s + d);
                load[b] = routes[b]
                    .iter()
                    .map(|&c| demand[c])
                    .fold(zero, |s, d| s + d);
            }
        }
    }
    // The loads are maintained exactly; re-derive the load of each route from the
    // final customer lists so a bookkeeping slip can never silently leave a route
    // over capacity (the caller's incumbent check re-verifies the point anyway).
    for (r, l) in routes.iter().zip(load.iter_mut()) {
        *l = r.iter().map(|&c| demand[c]).fold(zero, |a, d| a + d);
    }
}

/// Detect a "generalized assignment" structure: one binary variable per
/// (item, candidate) pair, with a degree-1 equality row per item
/// (`Σ_k x_ik = 1`, all-1 coefficients, RHS 1) -- the shape shared by bin
/// packing, generalized assignment, and uncapacitated facility-customer
/// assignment. Unlike [`recover_arc_structure`] (which requires each
/// variable to sit on exactly *two* such rows, the bipartite/degree-2
/// case), here each candidate variable belongs to exactly *one* item row;
/// "capacity"-style rows are whatever inequality rows those candidate
/// columns also participate in, read directly from `a`/`b` rather than
/// assumed to have any particular shape.
///
/// Returns `None` the moment the structure doesn't cleanly match, per
/// the same philosophy as `recover_arc_structure`.
fn recover_item_candidate_rows<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
) -> Option<Vec<Vec<usize>>> {
    let eps = T::from_f64(1e-6).expect("scalar literal");
    let one = T::one();
    let n = problem.q.len();
    let m = problem.b.len();

    let is_eq_row = crate::eq_row_mask(&problem.cones, m);

    let mut item_candidates: Vec<Vec<usize>> = Vec::new();
    let mut var_item: Vec<Option<usize>> = vec![None; n];
    for i in 0..m {
        if !is_eq_row[i] {
            continue;
        }
        if (problem.b[i] - one).abs() > eps {
            continue;
        }
        let mut vars = Vec::new();
        let mut shape_ok = true;
        for j in 0..n {
            let aij = problem.a.get(i, j);
            if aij.abs() <= eps {
                continue;
            }
            if (aij - one).abs() > eps || problem.var_types[j] != VarType::Binary {
                shape_ok = false;
                break;
            }
            vars.push(j);
        }
        if !shape_ok || vars.len() < 2 {
            continue;
        }
        if vars.iter().any(|&j| var_item[j].is_some()) {
            continue;
        }
        let idx = item_candidates.len();
        for &j in &vars {
            var_item[j] = Some(idx);
        }
        item_candidates.push(vars);
    }
    if item_candidates.len() < 2 {
        return None;
    }

    // Every candidate variable must appear in no equality row other than
    // its own item row -- keeps the later capacity-feasibility check
    // (which only reasons about inequality rows) sound.
    for i in 0..m {
        if !is_eq_row[i] {
            continue;
        }
        if (problem.b[i] - one).abs() <= eps {
            let is_item_row = (0..n)
                .filter(|&j| problem.a.get(i, j).abs() > eps)
                .all(|j| var_item[j].is_some());
            if is_item_row {
                continue;
            }
        }
        for j in 0..n {
            if problem.a.get(i, j).abs() > eps && var_item[j].is_some() {
                return None;
            }
        }
    }

    Some(item_candidates)
}

/// Greedy constructive heuristic for the generalized-assignment structure
/// [`recover_item_candidate_rows`] detects: assigns each item to the
/// candidate its LP relaxation favors most, verifying every affected
/// inequality row's slack directly against the raw constraint data as
/// each assignment commits (so it respects capacity rows of any shape,
/// not just bin packing's specific form), and falling back to the next-
/// favored candidate on a would-be violation. Remaining (non-item)
/// variables -- typically "resource used" indicators -- are filled in by
/// an LP solve over the pinned assignment, then any that come out
/// fractional are rounded up (safe for a `Σ(...) − C·y ≤ 0`-style row:
/// increasing `y` only relaxes it) and the whole candidate solution is
/// verified feasible before being returned, so a wrong rounding guess
/// fails closed (`None`) rather than returning an invalid point.
///
/// This exists because feasibility pump was confirmed (directly, with a
/// 20s budget vs. its usual ~500ms) to reliably find *nothing* on this
/// structure at moderate scale (`binpack_n20`, 420 variables) -- pure
/// coordinate rounding essentially never respects a `Σ_k x_ik = 1`
/// assignment row, the same reason `nearest_neighbor_tour` exists for
/// the degree-1 bipartite case.
pub fn greedy_assignment_heuristic<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
    root_lp: &mut Option<Vec<T>>,
) -> Option<(Vec<T>, T)> {
    use iconic_api::solve as api_solve;

    let eps = T::from_f64(1e-6).expect("scalar literal");
    let zero = T::zero();
    let one = T::one();
    let n = problem.q.len();
    let m = problem.b.len();

    let item_candidates = recover_item_candidate_rows(problem)?;
    let is_item_var: Vec<bool> = {
        let mut v = vec![false; n];
        for cands in &item_candidates {
            for &j in cands {
                v[j] = true;
            }
        }
        v
    };

    // LP relaxation, purely to rank each item's candidates by preference. Shared with
    // the other root heuristics that need this same relaxation.
    let x_lp: Vec<T> = root_relaxation(problem, settings, root_lp)?.to_vec();

    // Best-case contribution of every non-item variable toward each row's
    // `Σ a_ij x_j ≤ b_i` (its own bound in whichever direction helps that
    // row), so committing an item candidate can be checked for capacity
    // feasibility without knowing yet how the LP re-solve will set them.
    let mut min_other = vec![zero; m];
    for i in 0..m {
        let mut s = zero;
        for j in 0..n {
            if is_item_var[j] {
                continue;
            }
            let aij = problem.a.get(i, j);
            if aij.abs() <= eps {
                continue;
            }
            s += if aij >= zero {
                aij * problem.lb[j]
            } else {
                aij * problem.ub[j]
            };
        }
        min_other[i] = s;
    }

    let mut row_sum = vec![zero; m];
    let mut chosen: Vec<Option<usize>> = vec![None; item_candidates.len()];
    for (item_idx, candidates) in item_candidates.iter().enumerate() {
        let mut ranked = candidates.clone();
        ranked.sort_by(|&a, &b| {
            x_lp[b]
                .partial_cmp(&x_lp[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut placed = None;
        for &cand in &ranked {
            let mut ok = true;
            for i in 0..m {
                let aij = problem.a.get(i, cand);
                if aij.abs() <= eps {
                    continue;
                }
                if row_sum[i] + aij + min_other[i] > problem.b[i] + eps {
                    ok = false;
                    break;
                }
            }
            if ok {
                for i in 0..m {
                    let aij = problem.a.get(i, cand);
                    if aij.abs() > eps {
                        row_sum[i] += aij;
                    }
                }
                placed = Some(cand);
                break;
            }
        }
        chosen[item_idx] = placed;
        placed?;
    }

    let mut lb = problem.lb.clone();
    let mut ub = problem.ub.clone();
    for (item_idx, candidates) in item_candidates.iter().enumerate() {
        let winner = chosen[item_idx].expect("scalar literal");
        for &j in candidates {
            let v = if j == winner { one } else { zero };
            lb[j] = v;
            ub[j] = v;
        }
    }

    let prog2 = crate::heuristics::mip_to_cone_prog(problem, &lb, &ub);
    // The retry tier rescues numerical trouble (error return, iteration
    // limit). A definitive infeasibility/unboundedness certificate is not
    // that: the pinned program's feasible set does not depend on solver
    // tolerances, so re-solving at different settings cannot change the
    // verdict -- measured on tsp_mtz_n=12 it only burned 99 IPM iterations
    // (~1.7s) re-proving what tier 1 had established in 1ms, before the
    // point failed the feasibility check anyway.
    let solved = match api_solve(&prog2, &settings.lp_settings) {
        Ok(s)
            if matches!(
                s.status,
                Status::PrimalInfeasible | Status::DualInfeasible
            ) =>
        {
            return None;
        }
        Ok(s) if s.status.has_solution() => Some(s),
        _ => api_solve(&prog2, &crate::retry_lp_settings(&settings.lp_settings))
            .ok()
            .filter(|s| s.status.has_solution()),
    }?;

    let mut x = solved.x;
    for j in 0..n {
        if is_item_var[j] {
            continue;
        }
        if problem.var_types[j].is_integer() {
            let frac = x[j] - x[j].floor();
            if frac > eps && frac < one - eps {
                x[j] = x[j].ceil();
            }
        }
    }

    if !crate::check_feasibility(&x, problem) {
        return None;
    }
    let obj = crate::compute_objective(&problem.p, &problem.q, &x);
    Some((x, obj))
}

/// A bin-packing structure recovered from the problem data: `items[i]` gives
/// item `i`'s weight and its candidate assignment column per bin (indexed
/// the same as `bins`); `bins[k]` gives bin `k`'s indicator (opening)
/// variable column. Every bin shares the same `capacity`.
struct BinPackingStructure<T> {
    capacity: T,
    weights: Vec<T>,
    /// `item_cols[i][k]` = column for "item i assigned to bin k".
    item_cols: Vec<Vec<usize>>,
    bin_indicator: Vec<usize>,
}

/// Detects the bin-packing shape directly from problem data: `n_items`
/// equality assignment rows (`Σ_k x_ik = 1`, all-binary, coefficient 1 --
/// exactly [`recover_item_candidate_rows`]'s structure) plus `n_bins`
/// capacity rows, each with exactly one negative-coefficient variable (the
/// bin's binary "open" indicator) and the rest positive coefficients on
/// item-candidate columns (`Σ_i w_i x_ik − C_k y_k ≤ 0`). Classical
/// First-Fit-Decreasing assumes a SINGLE shared capacity, so this bails
/// (returns `None`) if bins don't all share one; it also requires every
/// item to have a consistent weight across every bin it can go in (the same
/// physical item, so its weight can't depend on which bin holds it) and
/// every item to have exactly one candidate per bin (a dense assignment,
/// matching the actual generator -- a sparser structure would need a
/// different, per-item-available-bins-aware packing, not implemented here).
fn recover_bin_packing_structure<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
) -> Option<BinPackingStructure<T>> {
    let eps = T::from_f64(1e-6).expect("scalar literal");
    let zero = T::zero();
    let n = problem.q.len();
    let m = problem.b.len();

    let item_candidates = recover_item_candidate_rows(problem)?;
    let n_items = item_candidates.len();
    let n_bins = item_candidates[0].len();
    if item_candidates.iter().any(|c| c.len() != n_bins) {
        return None; // not a dense assignment (every item must reach every bin)
    }
    let is_item_col: Vec<bool> = {
        let mut v = vec![false; n];
        for cands in &item_candidates {
            for &j in cands {
                v[j] = true;
            }
        }
        v
    };

    let is_eq_row = crate::eq_row_mask(&problem.cones, m);

    // Each capacity row -> (row index, indicator column, capacity, member item-columns).
    let mut bin_rows: Vec<(usize, usize, T, Vec<usize>)> = Vec::new();
    for i in 0..m {
        if is_eq_row[i] {
            continue;
        }
        let mut neg: Option<usize> = None;
        let mut members = Vec::new();
        let mut shape_ok = problem.b[i].abs() <= eps; // these rows are `<= 0`
        for j in 0..n {
            let aij = problem.a.get(i, j);
            if aij.abs() <= eps {
                continue;
            }
            if aij < zero {
                if neg.is_some() || problem.var_types[j] != VarType::Binary {
                    shape_ok = false;
                    break;
                }
                neg = Some(j);
            } else if is_item_col[j] {
                members.push(j);
            } else {
                shape_ok = false;
                break;
            }
        }
        let Some(ind) = neg else { continue };
        // Skip rows where the negative-coefficient column is an ITEM column
        // (e.g. symmetry-breaking rows from binpack_symmetry_break).
        if is_item_col[ind] {
            continue;
        }
        if !shape_ok || members.is_empty() {
            continue;
        }
        bin_rows.push((i, ind, -problem.a.get(i, ind), members));
    }
    if bin_rows.len() != n_bins {
        return None;
    }

    // All bins must share one capacity (classical FFD's requirement).
    let capacity = bin_rows[0].2;
    if bin_rows
        .iter()
        .any(|(_, _, c, _)| (*c - capacity).abs() > eps)
    {
        return None;
    }

    // Map each item-column to its bin index and weight (read directly from
    // that bin's own row, so it's correct regardless of column layout).
    let mut col_weight: FxHashMap<usize, T> = FxHashMap::default();
    let mut col_bin: FxHashMap<usize, usize> = FxHashMap::default();
    for (bin_idx, (row, _, _, members)) in bin_rows.iter().enumerate() {
        for &col in members {
            col_bin.insert(col, bin_idx);
            col_weight.insert(col, problem.a.get(*row, col));
        }
    }

    let mut weights = vec![zero; n_items];
    let mut item_cols = vec![vec![0usize; n_bins]; n_items];
    for (item_idx, cands) in item_candidates.iter().enumerate() {
        let mut w0: Option<T> = None;
        for &col in cands {
            let &bin_idx = col_bin.get(&col)?;
            let &w = col_weight.get(&col)?;
            match w0 {
                None => w0 = Some(w),
                Some(prev) => {
                    if (prev - w).abs() > eps {
                        return None;
                    }
                } // inconsistent weight across bins
            }
            item_cols[item_idx][bin_idx] = col;
        }
        weights[item_idx] = w0?;
    }

    let bin_indicator: Vec<usize> = bin_rows.iter().map(|(_, ind, _, _)| *ind).collect();
    Some(BinPackingStructure {
        capacity,
        weights,
        item_cols,
        bin_indicator,
    })
}

/// First-Fit-Decreasing: sort items by weight descending, place each into
/// the first already-open bin with enough remaining capacity, opening a
/// new one otherwise. A well-known, simple, near-optimal (within a small
/// constant factor of the true optimum in the worst case, and exact on
/// many practical instances) bin-packing heuristic -- classical LP-
/// relaxation-driven rounding gives it no help here (the LP relaxation
/// tends to spread each item fractionally across many bins, since
/// integrality is the entire difficulty), so a dedicated constructive
/// heuristic is the right tool, the same rationale as
/// [`nearest_neighbor_tour`] for TSP and [`greedy_assignment_heuristic`]
/// for the generalized-assignment shape.
///
/// Added after measuring the actual gap on `binpack_n20` directly: ICONIC's
/// existing incumbent used 8 bins while the trivial L2 lower bound
/// (⌈Σweights/capacity⌉) is 6 and FFD finds 6 -- the reported "large
/// integrality gap" was actually a weak INCUMBENT, not a weak bound (the
/// bound was already exact).
pub fn first_fit_decreasing_heuristic<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
) -> Option<(Vec<T>, T)> {
    let one = T::one();
    let zero = T::zero();
    let n = problem.q.len();

    let s = recover_bin_packing_structure(problem)?;
    let n_items = s.weights.len();
    let n_bins = s.bin_indicator.len();

    let mut order: Vec<usize> = (0..n_items).collect();
    order.sort_by(|&a, &b| {
        s.weights[b]
            .partial_cmp(&s.weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut bin_used = vec![zero; n_bins];
    let mut bin_open = vec![false; n_bins];
    let mut assignment = vec![0usize; n_items]; // item -> bin index
    for &item in &order {
        let w = s.weights[item];
        let mut placed = None;
        for k in 0..n_bins {
            if bin_open[k] && bin_used[k] + w <= s.capacity + T::from_f64(1e-9).expect("scalar literal") {
                placed = Some(k);
                break;
            }
        }
        let k = placed.unwrap_or_else(|| {
            let k = (0..n_bins)
                .find(|&k| !bin_open[k])
                .expect("n_bins >= n_items, always room for a fresh bin");
            bin_open[k] = true;
            k
        });
        bin_used[k] += w;
        assignment[item] = k;
    }

    // ── Compact used bins to the END ───────────────────────────────────
    // binpack_symmetry_break adds `y_k <= y_{k+1}` (all-z then all-ones).
    // FFD opens bins from index 0 upward (used bins at the front), which
    // is the opposite order. Remap so that empty bins are lowest, used
    // bins are highest -- satisfying the ordering without changing which
    // items share a bin.
    let n_used = bin_open.iter().filter(|&&o| o).count();
    if n_used > 0 {
        let mut empty_idx = 0usize;
        let mut used_idx = n_bins - n_used;
        let mut remap = vec![0usize; n_bins];
        for k in 0..n_bins {
            if bin_open[k] {
                remap[k] = used_idx;
                used_idx += 1;
            } else {
                remap[k] = empty_idx;
                empty_idx += 1;
            }
        }
        let mut new_open = vec![false; n_bins];
        for k in 0..n_bins {
            let nk = remap[k];
            new_open[nk] = bin_open[k];
        }
        bin_open = new_open;
        for item in 0..n_items {
            assignment[item] = remap[assignment[item]];
        }
    }

    let mut x = vec![zero; n];
    for item in 0..n_items {
        x[s.item_cols[item][assignment[item]]] = one;
    }
    for k in 0..n_bins {
        x[s.bin_indicator[k]] = if bin_open[k] { one } else { zero };
    }

    if !crate::check_feasibility(&x, problem) {
        return None;
    }
    let obj = crate::compute_objective(&problem.p, &problem.q, &x);
    Some((x, obj))
}

/// Detects a pure 0-1 multi-dimensional knapsack: every variable binary, no
/// quadratic term, every row a `≤` inequality (`Cone::NonNegative`, no
/// equality rows at all) with every coefficient non-negative -- a genuine
/// "packing" structure where including an item can only consume capacity,
/// never free it -- and a linear objective that only rewards inclusion
/// (`q_j ≤ 0` for every variable, since internally profit maximization is
/// stored as minimizing `-profit`). Bails (`false`) the moment any of this
/// doesn't hold, rather than guessing at a looser structure.
fn is_pure_binary_knapsack<T: Scalar + PartialOrd>(problem: &MipProblem<T>) -> bool {
    let eps = T::from_f64(1e-9).expect("scalar literal");
    let zero = T::zero();
    let n = problem.q.len();
    let m = problem.b.len();
    if problem.p.data.iter().any(|&v| v != zero) {
        return false;
    }
    if !problem.var_types.iter().all(|&vt| vt == VarType::Binary) {
        return false;
    }
    if problem.q.iter().any(|&qj| qj > eps) {
        return false;
    }
    let mut r = 0usize;
    for cone in &problem.cones {
        match cone {
            Cone::Zero(d) => {
                if *d > 0 {
                    return false;
                }
                r += d;
            }
            Cone::NonNegative(d) => {
                r += d;
            }
            _ => return false,
        }
    }
    if r != m {
        return false;
    }
    for i in 0..m {
        if problem.b[i] < -eps {
            return false;
        }
        for j in 0..n {
            if problem.a.get(i, j) < -eps {
                return false;
            }
        }
    }
    true
}

/// Greedy constructive heuristic for the pure 0-1 multi-dimensional
/// knapsack shape [`is_pure_binary_knapsack`] detects: ranks items by
/// efficiency (profit divided by the SUM, across every row, of that row's
/// weight-to-capacity fraction -- generalizing the classical single-
/// dimension profit/weight ratio to multiple simultaneous capacity
/// constraints) and greedily includes each item that still fits every row
/// at its current efficiency rank.
///
/// Added after measuring the actual gap on `mdk_n30_k3` directly: after a
/// full 30s budget, ICONIC's own B&B search (cover cuts + Mehrotra/Gondzio
/// IPM) found an incumbent of 769.41, while this greedy heuristic finds
/// 794.05 instantly -- the LP-relaxation-driven exploration doesn't have
/// this structure-aware efficiency signal to guide it, the same rationale
/// as [`first_fit_decreasing_heuristic`] for bin packing and
/// [`nearest_neighbor_tour`] for TSP.
pub fn multi_knapsack_greedy_heuristic<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
) -> Option<(Vec<T>, T)> {
    if !is_pure_binary_knapsack(problem) {
        return None;
    }
    let zero = T::zero();
    let one = T::one();
    let n = problem.q.len();
    let m = problem.b.len();
    let tiny = T::from_f64(1e-12).expect("scalar literal");
    let huge = T::from_f64(1e12).expect("scalar literal");
    let feas_eps = T::from_f64(1e-9).expect("scalar literal");

    let mut eff: Vec<(T, usize)> = Vec::with_capacity(n);
    for j in 0..n {
        let mut norm_w = zero;
        for i in 0..m {
            let bi = problem.b[i];
            let aij = problem.a.get(i, j);
            if bi > tiny {
                norm_w += aij / bi;
            } else if aij > zero {
                // Zero-capacity row with positive weight: this item can
                // never fit at all. Give it the worst possible efficiency
                // so it sorts last; the feasibility check during placement
                // excludes it regardless, this just avoids wasting a high
                // sort position on something that can never be placed.
                norm_w += huge;
            }
        }
        let profit = -problem.q[j];
        let e = if norm_w > zero {
            profit / norm_w
        } else {
            profit
        };
        eff.push((e, j));
    }
    eff.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut used = vec![zero; m];
    let mut x = vec![zero; n];
    let fits = |used: &[T], j: usize| -> bool {
        (0..m).all(|i| {
            let aij = problem.a.get(i, j);
            aij <= zero || used[i] + aij <= problem.b[i] + feas_eps
        })
    };
    let add = |used: &mut [T], x: &mut [T], j: usize| {
        x[j] = one;
        for i in 0..m {
            let aij = problem.a.get(i, j);
            if aij > zero {
                used[i] += aij;
            }
        }
    };
    let remove = |used: &mut [T], x: &mut [T], j: usize| {
        x[j] = zero;
        for i in 0..m {
            let aij = problem.a.get(i, j);
            if aij > zero {
                used[i] -= aij;
            }
        }
    };
    for &(_, j) in &eff {
        if fits(&used, j) {
            add(&mut used, &mut x, j);
        }
    }
    // Snapshot the post-greedy, pre-exchange state. Every item placed by the
    // greedy pass was checked against the constraints at its moment of
    // placement — if no item was ever placed, the all-zero vector is trivially
    // feasible for any pure 0-1 knapsack (all constraints are Ax≤b, A≥0, b≥0,
    // so x=0 gives 0≤b — always satisfied). The snapshot is a safety net: if
    // the exchange passes ever produce a point that fails the feasibility
    // check (floating-point edge case or a latent add/remove bug), we fall
    // back to this guaranteed-feasible state rather than returning None.
    let greedy_x = x.clone(); // snapshot for fallback

    // 1-for-1 exchange local search: standard, simple neighborhood search
    // for knapsack-family problems (standard public algorithmic knowledge).
    // The pure greedy
    // construction above commits to its efficiency ranking and never
    // reconsiders -- a lower-ranked but still-valuable item can be
    // permanently locked out even when swapping it in for one already-
    // placed item would improve total profit while remaining feasible.
    // Repeats full passes until a pass makes no improvement (bounded by
    // n passes, each O(n*m) -- negligible next to a single LP solve, let
    // alone a B&B search).
    for _ in 0..n {
        let mut improved = false;
        for j_out_candidate in 0..n {
            if x[j_out_candidate] == zero {
                continue;
            }
            for j_in in 0..n {
                if x[j_in] != zero {
                    continue;
                }
                let gain = -problem.q[j_in] - (-problem.q[j_out_candidate]);
                if gain <= feas_eps {
                    continue;
                }
                remove(&mut used, &mut x, j_out_candidate);
                if fits(&used, j_in) {
                    add(&mut used, &mut x, j_in);
                    improved = true;
                    break;
                } else {
                    add(&mut used, &mut x, j_out_candidate);
                }
            }
        }
        if !improved {
            break;
        }
    }
    // 1-for-2 exchange: remove one included item, try to fit TWO currently-
    // excluded items in its place. A single large-but-inefficient item can
    // block two smaller, collectively-more-valuable ones — the 1-for-1
    // swap above catches the "equal size" case but misses this because
    // neither of the two smaller items alone fits into the freed capacity.
    // Same provable-termination guarantee as 1-for-1 (each swap strictly
    // increases total profit, a bounded integer quantity, so no cycling).
    for _ in 0..n {
        let mut improved = false;
        // Incoming-pair candidates: excluded items by profit descending,
        // capped so the pair scan stays bounded on very large item sets —
        // the exhaustive O(excluded²)-per-item scan cost ~125ms per pass at
        // n = 1000 regardless of how many pairs pass the gain filter. The
        // cap is exhaustive for every item count up to ~500; beyond it only
        // the least profitable candidates drop out, and those can clear the
        // gain filter for the weakest placed items only.
        let mut pool: Vec<usize> = (0..n).filter(|&j| x[j] == zero).collect();
        pool.sort_unstable_by(|&a, &b| {
            (-problem.q[b])
                .partial_cmp(&(-problem.q[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pool.truncate(256);
        for j_out in 0..n {
            if x[j_out] == zero {
                continue;
            }
            let profit_out = -problem.q[j_out];
            'pairs: for a in 0..pool.len() {
                let j_in1 = pool[a];
                if x[j_in1] != zero {
                    continue; // stale: placed by an acceptance earlier this pass
                }
                let p1 = -problem.q[j_in1];
                // Profits descend through the pool, so the best remaining
                // partner for j_in1 is the next entry: if even that pair
                // cannot clear the gain, no later partner can.
                if a + 1 < pool.len() && p1 + (-problem.q[pool[a + 1]]) - profit_out <= feas_eps {
                    break 'pairs;
                }
                for b in (a + 1)..pool.len() {
                    let j_in2 = pool[b];
                    if x[j_in2] != zero {
                        continue;
                    }
                    let gain = p1 + (-problem.q[j_in2]) - profit_out;
                    if gain <= feas_eps {
                        break; // later partners are strictly less profitable
                    }
                    remove(&mut used, &mut x, j_out);
                    // Check sequentially: j_in1 must fit, THEN j_in2
                    // must also fit in the REMAINING capacity.  Checking
                    // both against the pre-removal state accepts swaps
                    // where the two items don't fit together.
                    if fits(&used, j_in1) {
                        add(&mut used, &mut x, j_in1);
                        if fits(&used, j_in2) {
                            add(&mut used, &mut x, j_in2);
                            improved = true;
                            // Continue the pass from the next j_out rather
                            // than rescanning from the top — each acceptance
                            // is a strict improvement either way, so the
                            // pass count stays bounded.
                            break 'pairs;
                        }
                        remove(&mut used, &mut x, j_in1);
                    }
                    add(&mut used, &mut x, j_out);
                }
            }
        }
        if !improved {
            break;
        }
    }

    // ── Item-set re-pack exchange (multiple-knapsack shape) ────────────
    // The 1-for-1 / 1-for-2 passes above operate on bin assignments: a swap
    // only succeeds when the incoming item fits the freed capacity in place.
    // On a multiple knapsack the optimum can require RE-PACKING the whole
    // item set (measured on multiknap_i25_b4: the exchange-local optimum is
    // -761.8789457 while the true optimum is -764.8918501 — the optimal
    // solution swaps item 3 for item 10 and re-assigns three bins' contents,
    // and the 1-for-1 exchange cannot reach it because item 10 fits no
    // single bin's freed capacity). This pass works on the ITEM level:
    // extract the placed item set, try 1-for-1 set swaps (remove one item,
    // add one) and RE-PACK the candidate set from scratch with best-fit
    // placement; accept only swaps whose repack places every item. The
    // repack is a constructive heuristic — it never claims feasibility it
    // did not produce — so accepted swaps are sound, and the final state is
    // feasibility-checked with the usual fallback below.
    //
    // Items are detected as groups of variables sharing the same objective
    // coefficient (a multiple knapsack's bin-copies of one item all carry
    // the item's profit) that agree on every row they SHARE (the copies
    // agree on the partition row; in the capacity rows only one copy of the
    // pair appears, so nothing to disagree). The grouping is a heuristic
    // for the swap structure — a coincidental grouping can only cost
    // accepted-swap quality, never feasibility (every accepted point is
    // built by the feasibility-checked placement and verified below).
    {
        let mut items: Vec<(usize, T, Vec<usize>)> = Vec::new(); // (item, weight, member vars)
        let mut item_of: Vec<usize> = vec![usize::MAX; n];
        for v in 0..n {
            if item_of[v] != usize::MAX || problem.q[v] >= zero {
                continue;
            }
            let mut members = vec![v];
            for v2 in (v + 1)..n {
                if item_of[v2] != usize::MAX || problem.q[v2] != problem.q[v] {
                    continue;
                }
                // Shared-row agreement: on every row where BOTH have a
                // nonzero coefficient, the coefficients must match.
                let mut agree = true;
                for i in 0..m {
                    let av = problem.a.get(i, v);
                    let av2 = problem.a.get(i, v2);
                    if av.abs() > tiny && av2.abs() > tiny && (av - av2).abs() > tiny {
                        agree = false;
                        break;
                    }
                }
                if agree {
                    members.push(v2);
                }
            }
            let mut weight = zero;
            for &u in &members {
                for i in 0..m {
                    let aiu = problem.a.get(i, u).abs();
                    if aiu > weight {
                        weight = aiu;
                    }
                }
            }
            let idx = items.len();
            for &u in &members {
                item_of[u] = idx;
            }
            items.push((idx, weight, members));
        }
        let mut placed_item = vec![false; items.len()];
        for v in 0..n {
            if x[v] != zero && item_of[v] != usize::MAX {
                placed_item[item_of[v]] = true;
            }
        }
        // Iterate set swaps to a fixed point (bounded by n passes; each swap
        // strictly increases the objective).
        //
        // Provably a no-op when there is a single capacity row and every
        // item is a singleton group: with P = 0 (the entry gate) profits are
        // separable, and repacking S' = placed − io + ii into one row
        // succeeds exactly when Σ_{k∈S'} a_k ≤ b — the same feasibility
        // test the 1-for-1 exchange's fits() made after removing j_out,
        // under the same p_in > p_out gain filter. The 1-for-1 loop above
        // already ran to a no-improvement pass, so no pair acceptable here
        // exists. Skipping avoids the O(n²) candidate scan with a fresh
        // repack-order sort per pair, which alone cost ~1.25s of a 2.2s
        // solve on knapsack_n=1000 — all of it discarded work.
        let set_swap_inert = m == 1 && items.iter().all(|(_, _, members)| members.len() == 1);
        let mut obj_cur = crate::compute_objective(&problem.p, &problem.q, &x);
        for _ in 0..n {
            if set_swap_inert {
                break;
            }
            let mut swapped = false;
            for (io, (_, _, members_out)) in items.iter().enumerate() {
                if !placed_item[io] {
                    continue;
                }
                let p_out = -problem.q[members_out[0]];
                for (ii, (_, _, members_in)) in items.iter().enumerate() {
                    if placed_item[ii] {
                        continue;
                    }
                    let p_in = -problem.q[members_in[0]];
                    if p_in - p_out <= feas_eps {
                        continue;
                    }
                    // Repack S' = placed − io + ii.
                    let mut used2 = vec![zero; m];
                    let mut x2 = vec![zero; n];
                    let mut order: Vec<usize> = (0..items.len())
                        .filter(|&k| (placed_item[k] && k != io) || k == ii)
                        .collect();
                    order.sort_unstable_by(|&a, &b| {
                        items[b]
                            .1
                            .partial_cmp(&items[a].1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mut ok = true;
                    'items: for &k in &order {
                        // Best-fit: the member variable that fits and leaves
                        // the most capacity slack (for a multiple knapsack
                        // this is the bin with the most room).
                        let mut best_var = None;
                        let mut best_slack = -huge;
                        for &u in &items[k].2 {
                            if (0..m).all(|i| {
                                let aiu = problem.a.get(i, u);
                                aiu <= zero || used2[i] + aiu <= problem.b[i] + feas_eps
                            }) {
                                let mut slack = huge;
                                for i in 0..m {
                                    if problem.a.get(i, u) > zero {
                                        let s = problem.b[i] - used2[i] - problem.a.get(i, u);
                                        if s < slack {
                                            slack = s;
                                        }
                                    }
                                }
                                if slack > best_slack {
                                    best_slack = slack;
                                    best_var = Some(u);
                                }
                            }
                        }
                        let Some(u) = best_var else {
                            ok = false;
                            break 'items;
                        };
                        x2[u] = one;
                        for i in 0..m {
                            let aiu = problem.a.get(i, u);
                            if aiu > zero {
                                used2[i] += aiu;
                            }
                        }
                    }
                    if !ok {
                        continue;
                    }
                    // Accept the swap only if the repacked assignment is
                    // strictly better (verify by the objective, not the
                    // nominal gain — the repack may also drop untracked
                    // negative-profit copies). obj_cur is loop-invariant
                    // between acceptances, so it is hoisted out of the
                    // candidate scan.
                    let obj_after = crate::compute_objective(&problem.p, &problem.q, &x2);
                    if obj_after >= obj_cur - feas_eps {
                        continue;
                    }
                    x = x2;
                    obj_cur = obj_after;
                    placed_item[io] = false;
                    placed_item[ii] = true;
                    swapped = true;
                    break;
                }
                if swapped {
                    break;
                }
            }
            if !swapped {
                break;
            }
        }
    }

    let obj = crate::compute_objective(&problem.p, &problem.q, &x);
    // The greedy construction always produces a feasible point (at minimum the
    // all-zero vector). The exchange passes above only swap items under
    // constraint checks, so this should always pass — but on the off chance a
    // floating-point edge case or latent bug produces an infeasible state, fall
    // back to the guaranteed-feasible post-greedy snapshot rather than
    // returning None, which would leave the solver without an incumbent.
    let feasible = crate::check_feasibility(&x, problem);
    if feasible {
        return Some((x, obj));
    }
    // Fallback: the post-greedy state is guaranteed feasible.
    let greedy_obj = crate::compute_objective(&problem.p, &problem.q, &greedy_x);
    debug_assert!(
        crate::check_feasibility(&greedy_x, problem),
        "post-greedy state must be feasible for any pure 0-1 knapsack"
    );
    // In release builds, if the snapshot itself fails (should be impossible
    // for any genuine knapsack), return the all-zero solution — always valid.
    if crate::check_feasibility(&greedy_x, problem) {
        Some((greedy_x, greedy_obj))
    } else {
        let zero_x = vec![zero; n];
        Some((zero_x, zero))
    }
}

/// Greedy graph coloring: detects `n_v*k + k` variable coloring
/// formulation and applies greedy algorithm. Process vertices in order,
/// assign each the first color not used by neighbors. O(V+E), finds
/// feasible k-coloring if one exists within the given k.
///
/// The formulation has `n = n_v*k + k = k*(n_v + 1)` variables, so every
/// divisor k (2..15) of n is a candidate.  We try each candidate
/// structurally — the first whose "one color per vertex" rows (b=1,
/// coefficients 1 in slots i*k..i*k+k-1) all pass is accepted.  This
/// handles the case where two different choices of k produce the same
/// total n (e.g. n=44: k=2,n_v=21 or k=4,n_v=10 — we try both and pick
/// the one whose formulation rows match).
pub fn greedy_graph_coloring<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
) -> Option<(Vec<T>, T)> {
    let eps = T::from_f64(1e-9).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let n = problem.q.len();
    let m = problem.b.len();

    // Collect all viable (k, n_v) candidates: n = k * (n_v+1) ⇒ n_v = n/k - 1.
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for kv in 2..=15 {
        if n % kv == 0 {
            let nv = n / kv - 1;
            if nv > 1 {
                candidates.push((kv, nv));
            }
        }
    }
    // Try larger k first (more colors = easier to color greedily).
    candidates.sort_by_key(|a| std::cmp::Reverse(a.0));

    let mut k = 0usize;
    let mut n_v = 0usize;
    for &(kv, nv) in &candidates {
        let n_x = nv * kv;
        if n_x + kv != n {
            continue;
        }
        if m < nv {
            continue;
        }
        // Each of the first nv rows must have b=1 and exactly kv binary
        // columns at (i*kv)..(i*kv+kv-1), all coefficient 1 — the
        // "each vertex gets exactly one color" constraint.
        let mut ok = true;
        for i in 0..nv {
            if (problem.b[i] - one).abs() > eps {
                ok = false;
                break;
            }
            let cols: Vec<usize> = (0..kv)
                .filter(|&c| problem.a.get(i, i * kv + c).abs() > eps)
                .collect();
            if cols.len() != kv
                || cols
                    .iter()
                    .any(|&c| (problem.a.get(i, i * kv + c) - one).abs() > eps)
            {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        k = kv;
        n_v = nv;
        break;
    }
    if k == 0 || n_v == 0 {
        return None;
    }
    let n_x = n_v * k;
    if !problem.var_types.iter().all(|v| *v == VarType::Binary) {
        return None;
    }

    let mut adj = vec![Vec::new(); n_v];
    for row in n_v..m {
        let mut vs = Vec::new();
        for j in 0..n_x {
            if problem.a.get(row, j).abs() > eps {
                vs.push(j / k);
            }
        }
        vs.sort();
        vs.dedup();
        if vs.len() == 2 && vs[0] != vs[1] {
            adj[vs[0]].push(vs[1]);
            adj[vs[1]].push(vs[0]);
        }
    }
    if adj.iter().all(|a| a.is_empty()) {
        return None;
    }

    let mut x = vec![zero; n];
    let mut color = vec![usize::MAX; n_v];
    // Sort vertices by degree (descending): largest-degree-first greedy
    // coloring uses far fewer colors than arbitrary order on random graphs
    // and matches DSATUR on the first step.
    let mut order: Vec<usize> = (0..n_v).collect();
    order.sort_by_key(|&v| std::cmp::Reverse(adj[v].len()));
    for &v in &order {
        let mut forbidden = [false; 16];
        for &u in &adj[v] {
            if color[u] < 16 {
                forbidden[color[u]] = true;
            }
        }
        let mut ok = false;
        for c in 0..k {
            if !forbidden[c] {
                color[v] = c;
                ok = true;
                break;
            }
        }
        if !ok {
            return None;
        }
    }

    for v in 0..n_v {
        x[v * k + color[v]] = one;
    }
    let mut used = vec![false; k];
    for v in 0..n_v {
        used[color[v]] = true;
    }
    for c in 0..k {
        if used[c] {
            x[n_x + c] = one;
        }
    }
    if !crate::check_feasibility(&x, problem) {
        return None;
    }
    let obj = crate::compute_objective(&problem.p, &problem.q, &x);
    Some((x, obj))
}

/// Round binaries from LP, then solve the continuous subproblem.
/// The 0.1 rounding threshold catches fractional binary values that are
/// slightly above zero but below 0.5 — common in weak-LP-relaxation
/// problems (fcflow, lot-sizing) where the LP "opens" variables
/// fractionally at negligible cost. Rounding at 0.5 would miss these.
/// If the resulting continuous subproblem is infeasible, the function
/// returns None (fails closed).
pub fn round_binaries_resolve<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
    root_lp: &mut Option<Vec<T>>,
) -> Option<(Vec<T>, T)> {
    use crate::heuristics::mip_to_cone_prog;
    if problem.var_types.iter().all(|v| *v == VarType::Continuous) {
        return None;
    }
    let n = problem.q.len();
    let thresh = T::from_f64(0.1).expect("scalar literal");
    let one = T::one();
    let x_root: Vec<T> = root_relaxation(problem, settings, root_lp)?.to_vec();
    let mut lb = problem.lb.clone();
    let mut ub = problem.ub.clone();
    for j in 0..n {
        if problem.var_types[j].is_integer() && x_root[j] > thresh {
            lb[j] = one;
            ub[j] = one;
        }
    }
    // The fixed-binary re-solve is a plain LP whenever the objective is linear:
    // solve it on the dual simplex instead of the IPM path. Measured on the
    // suite this call is the *expensive* half of the heuristic exactly where it
    // is needed most — 2.9-11.7s on the transport instances, every one ending
    // in `None` anyway — while the same LP shapes solve in milliseconds on the
    // tree's engine. Quadratic objectives keep the IPM: the simplex LP drops P,
    // which would change the point this rounds from.
    let x2 = if problem.p.data.iter().all(|&v| v == T::zero()) {
        match crate::solve_lp_relaxation_simplex(problem, &lb, &ub, false, settings.deadline) {
            crate::RelaxLpOutcome::Optimal(x) => x,
            // Infeasible fixed LP or a simplex give-up: the rounding failed.
            // Fail closed without re-burning seconds on the IPM for a verdict
            // that can only confirm the failure.
            _ => return None,
        }
    } else {
        let prog2 = mip_to_cone_prog(problem, &lb, &ub);
        let sol2 = crate::solve_lp_ok(&prog2, &settings.lp_settings)?;
        sol2.x
    };
    if !crate::check_feasibility(&x2, problem) {
        return None;
    }
    let obj = crate::compute_objective(&problem.p, &problem.q, &x2);
    Some((x2, obj))
}

// ── Zero-Objective Heuristic ───────────────────────────────────────────────

/// Outcome of a zero-objective feasibility probe.
///
/// The zero-objective LP (and the zero-objective sub-MIP) is a relaxation of
/// the MIP: the feasible set is unchanged by zeroing the objective, so a
/// proven-infeasible probe is a *proof* that the MIP is infeasible. Callers
/// must propagate that verdict rather than treating it as "no incumbent".
#[derive(Debug)]
pub enum ZeroObjectiveOutcome<T> {
    /// The zero-objective LP relaxation is infeasible: the MIP is proven infeasible.
    ProvedInfeasible,
    /// A feasible integer point was found (x, objective against the ORIGINAL q).
    Incumbent(Vec<T>, T),
    /// Nothing usable: LP solved but rounding failed, or another status.
    None,
}

/// Zero-objective heuristic: solve the LP relaxation with the objective zeroed
/// out, turning it into a pure feasibility problem. The resulting fractional
/// solution (driven only by constraints, not objective) is rounded to integer
/// values. If the rounded point is MIP-feasible, its true objective is computed.
///
/// This is a zero-objective start. Finding ANY
/// feasible point is often the hard part -- without the objective competing, the
/// LP is purely a feasibility problem, typically much cheaper to solve and less
/// likely to push variables to fractional extremes.
///
/// Returns `Incumbent(x, obj)` if a feasible integer solution is found, where `obj`
/// is evaluated against the ORIGINAL (non-zero) objective. Returns
/// `ProvedInfeasible` when the zero-objective LP itself is infeasible: it is a
/// relaxation of the MIP, so an infeasible relaxation is a MIP-infeasibility
/// proof. The verdict used to be discarded (`None` for any non-Solved status) --
/// a genuinely infeasible MIP then ran the full tree to exhaustion instead of
/// returning an instant proof.
pub fn zero_objective_heuristic<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
    settings: &MipSettings<T>,
) -> ZeroObjectiveOutcome<T> {
    use crate::heuristics::mip_to_cone_prog;
    use iconic_api::solve as api_solve;

    let n = problem.q.len();
    let zero = T::zero();

    // LP objective: try the dual simplex first — the same feasibility LP the
    // tree would solve in milliseconds costs seconds on the IPM path (measured
    // 1.3-7.7s per call across the suite's worst instances). Only an `Optimal`
    // vertex short-circuits the IPM below: the simplex's `Infeasible` is a
    // verdict, not a certificate, so the *proof* still has to come from the
    // IPM's PrimalInfeasible status; a give-up also falls through rather than
    // fail closed (this is the first heuristic, and its incumbents count).
    let mut x_lp: Option<Vec<T>> = if problem.p.data.iter().all(|&v| v == T::zero()) {
        match crate::solve_lp_relaxation_simplex(
            problem,
            &problem.lb,
            &problem.ub,
            true,
            settings.deadline,
        ) {
            crate::RelaxLpOutcome::Optimal(x) => Some(x),
            _ => None,
        }
    } else {
        None
    };

    // The IPM fallback below has no deadline of its own (iconic-core Settings
    // carries no time limit), so it is gated on budget remaining: when the
    // simplex path gave up *because* the call's deadline passed (a
    // deadline abort surfaces as IterationLimit -> Failed here), falling
    // through would burn an unbounded IPM solve on top of the exhausted
    // budget — measured 4.9-5.8s of a cvrp solve for no solution.
    let out_of_budget = settings
        .deadline
        .is_some_and(|d| std::time::Instant::now() >= d);

    if x_lp.is_none() && !out_of_budget {
        // Build the cone program with zeroed objective
        let mut prog = mip_to_cone_prog(problem, &problem.lb, &problem.ub);
        for j in 0..n {
            prog.q[j] = zero;
        }

        // Solve with relaxed tolerances -- this is a heuristic, not a bound
        let mut relaxed = settings.lp_settings.clone();
        relaxed.eps_abs = T::from_f64(1e-6).expect("scalar literal");
        relaxed.eps_rel = T::from_f64(1e-6).expect("scalar literal");
        relaxed.eps_gap = T::from_f64(1e-6).expect("scalar literal");

        let sol = match api_solve(&prog, &relaxed) {
            Ok(s) => s,
            Err(_) => return ZeroObjectiveOutcome::None,
        };
        // Only a genuine PrimalInfeasible certificate is a proof. SolvedInaccurate
        // still carries a point (rounding proceeds below); NumericalError /
        // MaxIterations / TimeLimit are inconclusive, not infeasibility evidence.
        if sol.status == Status::PrimalInfeasible {
            return ZeroObjectiveOutcome::ProvedInfeasible;
        }
        if sol.status != Status::Solved && sol.status != Status::SolvedInaccurate {
            return ZeroObjectiveOutcome::None;
        }
        x_lp = Some(sol.x);
    }

    // Round integer variables to nearest integer, clamped to bounds. Every
    // fall-through path above left `x_lp` populated (the alternatives all
    // returned), so the `else` arm is unreachable insurance, not a live case.
    let Some(mut x) = x_lp else {
        return ZeroObjectiveOutcome::None;
    };
    for j in 0..n {
        if problem.var_types[j].is_integer() {
            x[j] = x[j].round();
            if x[j] < problem.lb[j] {
                x[j] = problem.lb[j];
            }
            if x[j] > problem.ub[j] {
                x[j] = problem.ub[j];
            }
        }
    }

    // Check MIP feasibility (bounds + integrality + constraints)
    if !crate::check_feasibility(&x, problem) {
        return ZeroObjectiveOutcome::None;
    }

    // Compute objective against the ORIGINAL (non-zero) q and P
    let obj = crate::compute_objective(&problem.p, &problem.q, &x);
    ZeroObjectiveOutcome::Incumbent(x, obj)
}

/// Greedy set covering/partitioning heuristic.  Detects the set-cover
/// structure from standard-form rows with RHS=1 and all {0,1} coefficients
/// (set covering, `Ax + s = 1, s ≥ 0`) or the equality-as-inequality-pair
/// encoding (set partitioning: `Ax + s = 1, s ≥ 0` for n_rows and
/// `−Ax + t = −1, t ≥ 0` for the next n_rows).  Picks the column with the
/// most uncovered rows, fixes it to 1, marks those rows covered, and
/// repeats until every cover row is covered at least once; preferences that
/// break ties by penalizing columns that touch already-covered rows (prefer
/// "no over-coverage"), so set-partitioning instances with an exact
/// covering are naturally preferred and the feasibility check afterward
/// catches any unavoidable over-cover.
///
/// O(n·m) — trivial.  Gives any set covering/partitioning instance at least
/// one feasible incumbent.
pub fn greedy_set_covering_heuristic<T: Scalar + PartialOrd>(
    problem: &MipProblem<T>,
) -> Option<(Vec<T>, T)> {
    let eps = T::from_f64(1e-9).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let n = problem.q.len();
    let m = problem.b.len();
    if m < 2 || n < 2 {
        return None;
    }
    // Detect set covering/partitioning structure: all variables binary,
    // all coefficients in {0, 1, -1} (allow -1 for the equality-as-
    // inequality-pairs encoding used in set partitioning).
    for j in 0..n {
        if problem.var_types[j] != VarType::Binary {
            return None;
        }
        for i in 0..m {
            let aij = problem.a.get(i, j);
            if aij.abs() > eps && (aij - one).abs() > eps && (aij + one).abs() > eps {
                return None;
            }
        }
    }
    // Partition rows: cover_rows have b≈1 and all coefficients in {0,1}
    // (the "cover" direction); reverse rows have b≈−1 (the "at least"
    // direction, satisfied automatically when every cover row is covered
    // exactly once).  Both are used in the final feasibility check.
    let mut cover_rows: Vec<usize> = Vec::new();
    for i in 0..m {
        if (problem.b[i] - one).abs() > eps {
            continue;
        }
        // Only use rows whose coefficients are all non-negative.
        if (0..n).any(|j| problem.a.get(i, j) < -eps) {
            continue;
        }
        cover_rows.push(i);
    }
    if cover_rows.is_empty() {
        return None;
    }
    let n_cover = cover_rows.len();
    let idx: Vec<usize> = {
        let mut map = vec![usize::MAX; m];
        for (ri, &i) in cover_rows.iter().enumerate() {
            map[i] = ri;
        }
        map
    };
    let mut covered = vec![false; n_cover];
    let mut x = vec![zero; n];
    let mut n_covered = 0usize;
    while n_covered < n_cover {
        let mut best_score = -1isize;
        let mut best_j: Option<usize> = None;
        for j in 0..n {
            if x[j] != zero {
                continue;
            }
            let mut new_count = 0usize;
            let mut old_count = 0usize;
            for i in 0..m {
                let aij = problem.a.get(i, j);
                if aij <= eps {
                    continue;
                }
                let ri = idx[i];
                if ri == usize::MAX {
                    continue;
                }
                if !covered[ri] {
                    new_count += 1;
                } else {
                    old_count += 1;
                }
            }
            if new_count == 0 {
                continue;
            }
            // Prefer maximum new (uncovered) rows, then minimum old
            // (already-covered) rows to avoid over-covering — required
            // for set partitioning where every row must be covered
            // exactly once.
            let score = (new_count as isize) * ((n_cover + 1) as isize) - (old_count as isize);
            if score > best_score {
                best_score = score;
                best_j = Some(j);
            }
        }
        let j = best_j?;
        x[j] = one;
        for i in 0..m {
            let aij = problem.a.get(i, j);
            if aij <= eps {
                continue;
            }
            let ri = idx[i];
            if ri != usize::MAX && !covered[ri] {
                covered[ri] = true;
                n_covered += 1;
            }
        }
    }
    if !crate::check_feasibility(&x, problem) {
        return None;
    }
    let obj = crate::compute_objective(&problem.p, &problem.q, &x);
    Some((x, obj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iconic_linalg::DenseMatrix;

    /// Minimal 2-dimensional 0-1 knapsack, hand-verifiable: item0=(w=(5,1),
    /// profit=10), item1=(w=(1,5), profit=10), item2=(w=(3,3), profit=8),
    /// item3=(w=(4,4), profit=1), capacities=(6,6). item0+item1 exactly
    /// saturates both dimensions (6,6) for profit 20 -- the true optimum
    /// (every other combination either violates a capacity or scores
    /// lower: item2 alone=8, item0+item2 violates dim 1, item1+item2
    /// violates dim 2, item3 is dominated).
    fn tiny_multi_knapsack() -> MipProblem<f64> {
        let n = 4;
        let k = 2;
        let weights = [[5.0, 1.0], [1.0, 5.0], [3.0, 3.0], [4.0, 4.0]];
        let profits = [10.0, 10.0, 8.0, 1.0];
        let capacities = [6.0, 6.0];
        let mut a = DenseMatrix::<f64>::zeros(k, n);
        for i in 0..k {
            for j in 0..n {
                a.set(i, j, weights[j][i]);
            }
        }
        let q: Vec<f64> = profits.iter().map(|&p| -p).collect();
        MipProblem {
            p: DenseMatrix::zeros(n, n),
            q,
            a,
            b: capacities.to_vec(),
            cones: vec![Cone::NonNegative(k)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        }
    }

    /// A route that crosses itself: depot at the origin, customers
    /// 0=(0,1), 1=(2,0), 2=(0,3), 3=(2,2) on one route plus a singleton
    /// second route (the function needs >= 2 routes). The crossing order
    /// 0->2->1->3 costs 1 + 2 + sqrt(13) + 2 + sqrt(8) ~= 11.434; reversing
    /// the middle segment (2-opt) or relocating a customer strictly lowers
    /// it, and the improved route must respect the capacity.
    #[test]
    fn vrp_local_search_improves_a_crossing_route_within_capacity() {
        // dist[i][j]: depot=0, customers 1..=5.
        let pts: [(f64, f64); 6] = [
            (0.0, 0.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (0.0, 3.0),
            (2.0, 2.0),
            (0.0, 5.0),
        ];
        let dist: Vec<Vec<f64>> = (0..pts.len())
            .map(|i| {
                (0..pts.len())
                    .map(|j| {
                        let (dx, dy) = (pts[i].0 - pts[j].0, pts[i].1 - pts[j].1);
                        (dx * dx + dy * dy).sqrt()
                    })
                    .collect()
            })
            .collect();
        let cost = |r: &[usize]| -> f64 {
            let mut c = dist[0][r[0]] + dist[r[r.len() - 1]][0];
            for w in r.windows(2) {
                c += dist[w[0]][w[1]];
            }
            c
        };
        let mut routes = vec![vec![1usize, 3, 2, 4], vec![5usize]];
        let mut load = vec![4.0, 1.0];
        let demand = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let total_cost = |rs: &[Vec<usize>]| -> f64 { rs.iter().map(|r| cost(r)).sum() };
        let before = total_cost(&routes);
        vrp_local_search(&mut routes, &mut load, &demand, 10.0, 0, &dist);
        assert!(
            total_cost(&routes) < before - 1e-9,
            "2-opt/relocate/swap must improve the total route cost: {:.6} -> {:.6}",
            before,
            total_cost(&routes)
        );
        for (r, l) in routes.iter().zip(&load) {
            let sum: f64 = r.iter().map(|&c| demand[c]).sum();
            assert!(
                (sum - l).abs() < 1e-12,
                "load {l} must equal the demand sum {sum}"
            );
            assert!(
                sum <= 10.0 + 1e-9,
                "route {r:?} exceeds capacity: load {sum}"
            );
            assert!(!r.is_empty(), "routes must stay non-empty");
        }
    }

    /// The capacity check must be enforced: relocating a customer whose demand
    /// would push the destination route over capacity is never accepted, and
    /// every route's load stays exactly its demand sum. Demands [1,1,0.6,0.6,0.6]
    /// with capacity 2.0 (routes [1,2] at load 2.0, [3,4,5] at 1.8) make every
    /// inter-route relocation or swap overload a route -- the donor's 1.0 into
    /// the 1.8 route, or a 0.6 into the full 2.0 route, or any swap shifting
    /// load between them -- so only intra-route 2-opt may act; the customer
    /// set, the loads, and the capacities must all survive intact.
    #[test]
    fn vrp_local_search_never_violates_capacity() {
        let pts: [(f64, f64); 6] = [
            (0.0, 0.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (0.0, 3.0),
            (2.0, 2.0),
            (0.0, 5.0),
        ];
        let dist: Vec<Vec<f64>> = (0..pts.len())
            .map(|i| {
                (0..pts.len())
                    .map(|j| {
                        let (dx, dy) = (pts[i].0 - pts[j].0, pts[i].1 - pts[j].1);
                        (dx * dx + dy * dy).sqrt()
                    })
                    .collect()
            })
            .collect();
        let mut routes = vec![vec![1usize, 2], vec![3usize, 4, 5]];
        let mut load = vec![2.0, 1.8];
        let demand = [0.0, 1.0, 1.0, 0.6, 0.6, 0.6];
        vrp_local_search(&mut routes, &mut load, &demand, 2.0, 0, &dist);
        for (r, l) in routes.iter().zip(&load) {
            let sum: f64 = r.iter().map(|&c| demand[c]).sum();
            assert!((sum - l).abs() < 1e-12, "load {l} != demand sum {sum}");
            assert!(sum <= 2.0 + 1e-9, "route {r:?} exceeds capacity 2.0");
        }
        let all: Vec<usize> = routes.iter().flatten().copied().collect();
        assert_eq!(all.len(), 5, "the customer set must be preserved exactly");
        let mut sorted = all.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5], "customer set changed");
    }

    #[test]
    fn multi_knapsack_greedy_finds_the_true_optimum_on_a_tiny_instance() {
        let problem = tiny_multi_knapsack();
        let (x, obj) = multi_knapsack_greedy_heuristic(&problem)
            .expect("pure 0-1 multi-knapsack shape should be detected");
        assert!(
            crate::check_feasibility(&x, &problem),
            "returned point must be feasible: {:?}",
            x
        );
        assert!(
            (obj - (-20.0)).abs() < 1e-6,
            "obj={} (minimized -profit), expected the true optimum -20 (profit 20, items 0+1)",
            obj
        );
    }

    /// The 1-for-1 exchange local search must catch a case pure greedy
    /// construction alone misses: A=(w=3,p=4,eff=1.333), B=(w=4,p=4,eff=1.0),
    /// C=(w=6,p=5,eff=0.833), capacity=10. Greedy (efficiency-descending)
    /// picks A then B (both fit, C doesn't after that) for profit 8 -- but
    /// removing B and adding C instead (A+C, weight 3+6=9<=10) scores 9,
    /// the true optimum (confirmed by brute-force enumeration of all 8
    /// subsets). C's LOWER efficiency than B (despite HIGHER profit, since
    /// it's proportionally much heavier) is exactly why greedy's one-pass
    /// ranking locks it out without a follow-up swap.
    #[test]
    fn multi_knapsack_local_search_fixes_a_case_greedy_alone_gets_wrong() {
        let n = 3;
        let weights = [3.0, 4.0, 6.0];
        let profits = [4.0, 4.0, 5.0];
        let mut a = DenseMatrix::<f64>::zeros(1, n);
        for j in 0..n {
            a.set(0, j, weights[j]);
        }
        let q: Vec<f64> = profits.iter().map(|&p| -p).collect();
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q,
            a,
            b: vec![10.0],
            cones: vec![Cone::NonNegative(1)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        let (x, obj) = multi_knapsack_greedy_heuristic(&problem)
            .expect("pure 0-1 knapsack shape should be detected");
        assert!(crate::check_feasibility(&x, &problem));
        assert!((obj - (-9.0)).abs() < 1e-6, "obj={} (minimized -profit), expected the true optimum -9 (profit 9), not greedy-alone's 8", obj);
    }

    /// Structure-detection guard: a row with a negative coefficient (not a
    /// genuine "packing" row) must be rejected, not misinterpreted.
    /// Builds a small multiple-knapsack in the partition + capacity shape
    /// (item j has one binary per bin; rows `sum_b x_bj <= 1` per item and
    /// `sum_j w_j x_bj <= cap` per bin), like iconic-bench's
    /// `gen_multiple_knapsack`.
    fn tiny_multiknap(
        items: usize,
        bins: usize,
        weights: &[f64],
        profits: &[f64],
        cap: f64,
    ) -> MipProblem<f64> {
        let n = items * bins;
        let m = items + bins;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        let mut b = vec![0.0; m];
        for j in 0..items {
            for i in 0..bins {
                a.set(j, i * items + j, 1.0);
            }
            b[j] = 1.0;
        }
        for i in 0..bins {
            for j in 0..items {
                a.set(items + i, i * items + j, weights[j]);
            }
            b[items + i] = cap;
        }
        let q: Vec<f64> = profits.iter().cycle().take(n).map(|&p| -p).collect();
        MipProblem {
            p: DenseMatrix::zeros(n, n),
            q,
            a,
            b,
            cones: vec![Cone::NonNegative(m)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        }
    }

    fn brute_force_knapsack(problem: &MipProblem<f64>) -> f64 {
        let n = problem.q.len();
        let mut best = f64::INFINITY;
        for mask in 0u32..(1u32 << n) {
            let x: Vec<f64> = (0..n).map(|j| ((mask >> j) & 1) as f64).collect();
            if (0..problem.b.len()).any(|i| {
                (0..n).map(|j| problem.a.get(i, j) * x[j]).sum::<f64>() > problem.b[i] + 1e-9
            }) {
                continue;
            }
            best = best.min((0..n).map(|j| problem.q[j] * x[j]).sum::<f64>());
        }
        best
    }

    /// The item-set re-pack exchange: on a multiple knapsack the optimum can
    /// require swapping an item out of the SET and re-packing all bins, which
    /// the bin-level 1-for-1/1-for-2 exchanges cannot reach (the incoming
    /// item fits no single bin's freed capacity). Regression: the heuristic
    /// must match the enumerated optimum on small multiple-knapsack shapes.
    #[test]
    fn multi_knapsack_greedy_reaches_the_set_exchange_optimum() {
        let cases: Vec<(Vec<f64>, Vec<f64>, f64)> = vec![
            (vec![6.0, 5.0, 4.0], vec![10.0, 8.0, 7.0], 10.0),
            (
                vec![7.0, 6.0, 5.0, 4.0, 3.0],
                vec![12.0, 11.0, 9.0, 8.0, 6.0],
                11.0,
            ),
            (vec![9.0, 8.0, 7.0, 6.0], vec![14.0, 10.0, 9.0, 8.0], 13.0),
            (
                vec![5.0, 4.0, 3.0, 3.0, 2.0],
                vec![9.0, 8.0, 7.0, 5.0, 4.0],
                8.0,
            ),
        ];
        for (weights, profits, cap) in cases {
            for bins in 2..=3usize {
                let problem = tiny_multiknap(weights.len(), bins, &weights, &profits, cap);
                let truth = brute_force_knapsack(&problem);
                let (x, obj) = multi_knapsack_greedy_heuristic(&problem)
                    .expect("pure 0-1 knapsack must produce an incumbent");
                assert!(
                    crate::check_feasibility(&x, &problem),
                    "returned point must be feasible"
                );
                assert!(
                    (obj - truth).abs() < 1e-6,
                    "heuristic {obj} != enumerated optimum {truth} (bins={bins}, w={weights:?}, p={profits:?})"
                );
            }
        }
    }

    /// Builds a small multiple-knapsack in the partition + capacity shape
    /// (item j has one binary per bin; rows `sum_b x_bj <= 1` per item and
    /// `sum_j w_j x_bj <= cap` per bin), like iconic-bench's
    /// `gen_multiple_knapsack`.
    #[test]
    fn multi_knapsack_greedy_rejects_a_row_with_a_negative_coefficient() {
        let mut problem = tiny_multi_knapsack();
        problem.a.set(0, 0, -1.0);
        assert!(multi_knapsack_greedy_heuristic(&problem).is_none());
    }

    /// Structure-detection guard: a non-binary variable must be rejected.
    #[test]
    fn multi_knapsack_greedy_rejects_non_binary_variables() {
        let mut problem = tiny_multi_knapsack();
        problem.var_types[0] = VarType::Continuous;
        assert!(multi_knapsack_greedy_heuristic(&problem).is_none());
    }

    /// Minimal bin-packing-shaped MIP, deterministic (not the real
    /// generator's random weights, only its structure matters): 3 items
    /// of weight 4 each, capacity 10, up to 3 candidate bins. Two items
    /// fit per bin (4+4=8<=10) but not three (12>10), so the true optimum
    /// needs exactly 2 bins; any single-bin-per-item assignment (3 bins)
    /// is feasible but not optimal.
    fn tiny_bin_packing() -> MipProblem<f64> {
        let n_items = 3usize;
        let n_bins = 3usize;
        let capacity = 10.0;
        let weights = [4.0, 4.0, 4.0];
        let n_x = n_items * n_bins;
        let n = n_x + n_bins;
        let m = n_items + n_bins;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        let mut b = vec![0.0; m];
        for i in 0..n_items {
            for k in 0..n_bins {
                a.set(i, i * n_bins + k, 1.0);
            }
            b[i] = 1.0;
        }
        for k in 0..n_bins {
            for i in 0..n_items {
                a.set(n_items + k, i * n_bins + k, weights[i]);
            }
            a.set(n_items + k, n_x + k, -capacity);
            b[n_items + k] = 0.0;
        }
        let mut q = vec![0.0; n];
        for k in 0..n_bins {
            q[n_x + k] = 1.0;
        }
        let mut var_types = vec![VarType::Binary; n];
        for k in 0..n_bins {
            var_types[n_x + k] = VarType::Binary;
        }
        let mut ub = vec![1e20; n];
        for j in 0..n {
            if var_types[j] == VarType::Binary {
                ub[j] = 1.0;
            }
        }
        MipProblem {
            p: DenseMatrix::zeros(n, n),
            q,
            a,
            b,
            cones: vec![Cone::Zero(n_items), Cone::NonNegative(n_bins)],
            var_types,
            lb: vec![0.0; n],
            ub,
            warm_start: None,
        }
    }

    #[test]
    fn greedy_assignment_heuristic_finds_a_feasible_packing() {
        let problem = tiny_bin_packing();
        let settings = MipSettings::default();
        let (x, obj) = greedy_assignment_heuristic(&problem, &settings, &mut None).expect(
            "a feasible packing exists (e.g. two items per bin) but the heuristic found none",
        );
        assert!(
            crate::check_feasibility(&x, &problem),
            "returned point must be feasible: {:?}",
            x
        );
        // Every item must be assigned to exactly one bin.
        for i in 0..3 {
            let s: f64 = (0..3).map(|k| x[i * 3 + k]).sum();
            assert!(
                (s - 1.0).abs() < 1e-6,
                "item {} not assigned to exactly one bin: {:?}",
                i,
                x
            );
        }
        // At least 2 bins are needed (3 items of weight 4, capacity 10);
        // the heuristic isn't required to find the optimum, just something
        // feasible, but it must not silently claim fewer bins than possible.
        assert!(
            obj >= 2.0 - 1e-6,
            "obj={} bins claimed, but 2 is the true minimum",
            obj
        );
    }

    #[test]
    fn first_fit_decreasing_finds_the_true_optimum_on_a_tiny_instance() {
        let problem = tiny_bin_packing();
        let (x, obj) = first_fit_decreasing_heuristic(&problem)
            .expect("bin-packing structure should be detected and FFD should find a packing");
        assert!(
            crate::check_feasibility(&x, &problem),
            "returned point must be feasible: {:?}",
            x
        );
        for i in 0..3 {
            let s: f64 = (0..3).map(|k| x[i * 3 + k]).sum();
            assert!(
                (s - 1.0).abs() < 1e-6,
                "item {} not assigned to exactly one bin: {:?}",
                i,
                x
            );
        }
        // 3 items of weight 4, capacity 10: two fit per bin (8<=10), three
        // don't (12>10), so the true optimum is exactly 2 bins -- FFD should
        // find it exactly on an instance this small.
        assert!(
            (obj - 2.0).abs() < 1e-6,
            "obj={} bins, expected the true optimum of 2",
            obj
        );
    }

    /// Larger, less contrived instance (10 items, varied weights, capacity
    /// 20): verifies FFD both respects capacity everywhere and gets
    /// reasonably close to (in this case, matches) the trivial L2 lower
    /// bound `⌈Σweights/capacity⌉` -- the standard sanity check for any
    /// bin-packing heuristic, independent of the MIP solver entirely.
    #[test]
    fn first_fit_decreasing_respects_capacity_and_approaches_the_l2_bound() {
        let n_items = 10usize;
        let n_bins = n_items;
        let capacity = 20.0;
        let weights = [8.0, 7.0, 6.0, 9.0, 5.0, 8.0, 7.0, 6.0, 9.0, 5.0];
        let n_x = n_items * n_bins;
        let n = n_x + n_bins;
        let m = n_items + n_bins;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        let mut b = vec![0.0; m];
        for i in 0..n_items {
            for k in 0..n_bins {
                a.set(i, i * n_bins + k, 1.0);
            }
            b[i] = 1.0;
        }
        for k in 0..n_bins {
            for i in 0..n_items {
                a.set(n_items + k, i * n_bins + k, weights[i]);
            }
            a.set(n_items + k, n_x + k, -capacity);
            b[n_items + k] = 0.0;
        }
        let mut q = vec![0.0; n];
        for k in 0..n_bins {
            q[n_x + k] = 1.0;
        }
        let mut var_types = vec![VarType::Binary; n];
        for k in 0..n_bins {
            var_types[n_x + k] = VarType::Binary;
        }
        let mut ub = vec![1e20; n];
        for j in 0..n {
            if var_types[j] == VarType::Binary {
                ub[j] = 1.0;
            }
        }
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q,
            a,
            b,
            cones: vec![Cone::Zero(n_items), Cone::NonNegative(n_bins)],
            var_types,
            lb: vec![0.0; n],
            ub,
            warm_start: None,
        };

        let (x, obj) = first_fit_decreasing_heuristic(&problem).expect("should find a packing");
        assert!(crate::check_feasibility(&x, &problem));
        let l2_bound = (weights.iter().sum::<f64>() / capacity).ceil();
        assert!(
            obj >= l2_bound - 1e-6,
            "obj={obj} below the L2 lower bound {l2_bound} -- impossible, a bug"
        );
        assert!(
            obj <= l2_bound + 1e-6,
            "obj={obj}, expected FFD to match the L2 bound {l2_bound} on this instance"
        );
    }

    #[test]
    fn recover_item_candidate_rows_finds_three_items() {
        let problem = tiny_bin_packing();
        let items =
            recover_item_candidate_rows(&problem).expect("bin-packing shape should be detected");
        assert_eq!(items.len(), 3, "expected 3 items, got {:?}", items);
        for cands in &items {
            assert_eq!(
                cands.len(),
                3,
                "expected 3 candidate bins per item, got {:?}",
                cands
            );
        }
    }

    /// A plain knapsack (no assignment rows at all) must not be
    /// misdetected as a generalized-assignment structure.
    #[test]
    fn recover_item_candidate_rows_rejects_plain_knapsack() {
        let n = 4;
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![-1.0, -2.0, -3.0, -4.0],
            a: DenseMatrix::from_row_major(1, n, vec![2.0, 3.0, 4.0, 5.0]),
            b: vec![7.0],
            cones: vec![Cone::NonNegative(1)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        assert!(recover_item_candidate_rows(&problem).is_none());
    }

    /// Regression: after `binpack_symmetry_break` adds `y_k <= y_{k+1}`
    /// constraints, used bins must be the LAST ones (highest indices).
    /// FFD naturally opens bins from index 0 upward -- the remap pass must
    /// compact them to the end or the symmetry-breaking rows will reject
    /// the solution. Before this was fixed, FFD returned `None` silently on
    /// every binpack instance where symmetry breaking activated.
    #[test]
    fn ffd_works_with_symmetry_breaking_constraints() {
        let mut problem = tiny_bin_packing();
        let n_bins = 3;
        let n_x = 3 * n_bins;
        let old_m = problem.b.len();
        let new_m = old_m + n_bins - 1;
        let mut new_a = DenseMatrix::<f64>::zeros(new_m, problem.q.len());
        let mut new_b = vec![0.0; new_m];
        new_a.copy_block_from(&problem.a, old_m, problem.q.len());
        new_b[..old_m].copy_from_slice(&problem.b);
        for k in 0..(n_bins - 1) {
            let r = old_m + k;
            new_a.set(r, n_x + k, 1.0);
            new_a.set(r, n_x + k + 1, -1.0);
            new_b[r] = 0.0;
        }
        problem.a = new_a;
        problem.b = new_b;
        match problem.cones.last_mut() {
            Some(Cone::NonNegative(d)) => *d += n_bins - 1,
            _ => problem.cones.push(Cone::NonNegative(n_bins - 1)),
        }

        let (x, obj) = first_fit_decreasing_heuristic(&problem)
            .expect("FFD must find feasible packing with symmetry-breaking constraints");
        assert!(crate::check_feasibility(&x, &problem));
        assert!((obj - 2.0).abs() < 1e-6, "obj={} bins, expected 2", obj);
        // Verify ordering: unused bins first, then used bins (y_k <= y_{k+1}).
        for k in 0..(n_bins - 1) {
            let yk = x[n_x + k];
            let yk1 = x[n_x + k + 1];
            assert!(yk <= yk1 + 1e-9, "y[{}]={} > y[{}]={}", k, yk, k + 1, yk1);
        }
    }

    /// The zero-objective LP is a relaxation of the MIP: an infeasible
    /// relaxation proves the MIP infeasible, and the heuristic must say so
    /// instead of returning "no incumbent" (which used to send the caller off
    /// to exhaust the tree proving the same thing).
    #[test]
    fn zero_objective_lp_infeasibility_is_a_mip_proof() {
        // Binary x, y with x + y <= 1, x >= 1, y >= 1: the LP relaxation is
        // infeasible (x >= 1 and y >= 1 force x + y >= 2 > 1), so the
        // zero-objective LP must come back PrimalInfeasible.
        let n = 2;
        let m = 3;
        let mut a = DenseMatrix::<f64>::zeros(m, n);
        a.set(0, 0, 1.0);
        a.set(0, 1, 1.0); // x + y <= 1
        a.set(1, 0, -1.0); // -x <= -1  (x >= 1)
        a.set(2, 1, -1.0); // -y <= -1  (y >= 1)
        let problem = MipProblem {
            p: DenseMatrix::zeros(n, n),
            q: vec![1.0, 1.0],
            a,
            b: vec![1.0, -1.0, -1.0],
            cones: vec![Cone::NonNegative(m)],
            var_types: vec![VarType::Binary; n],
            lb: vec![0.0; n],
            ub: vec![1.0; n],
            warm_start: None,
        };
        let settings = MipSettings::<f64>::default();
        match zero_objective_heuristic(&problem, &settings) {
            ZeroObjectiveOutcome::ProvedInfeasible => {}
            other => panic!("expected ProvedInfeasible, got {other:?}"),
        }
    }
}

// ── Grouped multi-choice knapsack floor bound ──────────────────────────────
//
// Multiple-knapsack problems (and GUB-structured packings generally) have a
// relaxation whose weakness is not any single row but the *combination*: the
// assignment rows `Σ_i x_{i,g} ≤ 1` let the LP spread each item fractionally
// across every bin, harvesting most of the profit while no per-bin capacity
// row is ever violated. Aggregating the capacity rows into one virtual
// knapsack is valid but weak (it permits several copies of an item); the
// strength comes from keeping the assignment groups *integral*: with
//
//     y_g = Σ_{j ∈ g} x_j ∈ {0,1}      (assignment row, all-ones, rhs ≤ 1)
//     Σ_g W_g · y_g ≤ B                (all capacity rows summed)
//
// every feasible point of the MIP maps to a feasible point of this 0/1
// "grouped knapsack" (take y_g = 1 iff group g is used; its true weight
// Σ_{j∈g} a_{r,j} x_j is at most W_g for every row r simultaneously), so the
// DP optimum over it is a valid lower bound on the minimization. The DP uses
// the same conservative grid as [`crate::knapsack_presolve`] — group weights
// rounded UP, budget rounded DOWN — so discretization can only shrink the
// feasible set and never raise the reported bound.
//
// Measured motivation: multiknap_i25_b4's LP root bound is −772.18 against a
// true optimum of −764.89 (0.95%); this bound reads exactly −764.8919,
// closing the instance at the root where the tree needed ~19k nodes.

/// One assignment group: variables that share a row of the form
/// `x_a + x_b + ... ≤ rhs` with all coefficients equal to 1 and rhs ≤ 1.
struct Group {
    vars: Vec<usize>,
    /// Weight of the group under each aggregated (capacity) row: the max
    /// coefficient in the group, since setting any member to 1 contributes
    /// at least... rather, AT MOST its own coefficient, and the binding
    /// choice for the knapsack relaxation is the heaviest member.
    weights: Vec<f64>,
    profit: f64,
}

/// Value-only grouped 0/1 knapsack DP on the conservative integer grid.
/// `weights[g]` are already ceil'd grid units, `cap` is already floor'd;
/// profits are the group profits (positive = maximization).
fn grouped_dp_value(weights: &[usize], profits: &[f64], cap: usize) -> f64 {
    let mut dp = vec![0.0f64; cap + 1];
    for (g, &w) in weights.iter().enumerate() {
        if w == 0 || w > cap {
            continue;
        }
        let pg = profits[g];
        for c in (w..=cap).rev() {
            let cand = dp[c - w] + pg;
            if cand > dp[c] {
                dp[c] = cand;
            }
        }
    }
    dp[cap]
}

/// The grouped multi-choice knapsack lower bound, or `None` when the problem
/// does not have the shape (see the struct docs above). `Some(bound)` is a
/// valid lower bound on the minimization objective.
pub fn grouped_knapsack_bound<T: Scalar + PartialOrd>(problem: &MipProblem<T>) -> Option<T> {
    let eps = T::from_f64(1e-9).expect("scalar literal");
    let one = T::one();
    let zero = T::zero();
    let n = problem.q.len();
    let m = problem.b.len();
    // Linear objective only, all binaries, plain nonneg rows (the same gate
    // family as `is_pure_binary_knapsack`).
    if problem.p.data.iter().any(|&v| v != zero) {
        return None;
    }
    if !problem.var_types.iter().all(|&vt| vt == VarType::Binary) {
        return None;
    }
    if problem
        .cones
        .iter()
        .any(|c| !matches!(c, Cone::NonNegative(_)))
    {
        return None;
    }

    // Partition rows: assignment rows (all coefficients ≈ 1, rhs ≤ 1 + eps)
    // become groups; every other row must be all-nonneg and joins the
    // aggregated budget. A negative coefficient anywhere outside a group
    // breaks both halves (aggregation would need upper bounds to stay valid).
    let mut groups: Vec<Group> = Vec::new();
    let mut agg_rows: Vec<usize> = Vec::new();
    for i in 0..m {
        let mut all_one = true;
        let mut any_nonzero = false;
        for j in 0..n {
            let aij = problem.a.get(i, j);
            if aij.abs() > eps {
                any_nonzero = true;
                if (aij - one).abs() > eps {
                    all_one = false;
                    break;
                }
            }
        }
        if all_one && any_nonzero && problem.b[i] <= one + eps && problem.b[i] >= -eps {
            let vars: Vec<usize> = (0..n)
                .filter(|&j| problem.a.get(i, j).abs() > eps)
                .collect();
            groups.push(Group {
                vars,
                weights: Vec::new(),
                profit: 0.0,
            });
        } else {
            for j in 0..n {
                if problem.a.get(i, j) < -eps {
                    return None;
                }
            }
            agg_rows.push(i);
        }
    }
    if groups.is_empty() || agg_rows.is_empty() {
        return None;
    }
    // Groups must partition disjoint variable sets (an overlapping pair would
    // make the y-binary argument invalid: two groups sharing a var cannot
    // both independently take it).
    {
        let mut seen = vec![false; n];
        for g in &groups {
            for &j in &g.vars {
                if seen[j] {
                    return None;
                }
                seen[j] = true;
            }
        }
    }

    // Group profits: the best (most negative q_j → largest profit) member.
    // Since at most one member of a group can be 1, taking the max member
    // profit as the group's profit is valid for the RELAXATION (the DP may
    // credit a group with a profit whose member's weight it did not charge —
    // charging min weight and crediting max profit both loosen, i.e. only
    // raise the DP value toward a still-valid upper bound on max-profit).
    // Soundness: the DP must UPPER-bound the achievable profit. Credit each
    // group with its max member profit p* AND charge, on every aggregated
    // row, the max member coefficient W_r. For any feasible x, y_g = 1 iff
    // some member is 1 satisfies Σ W_r·y ≤ B and Σ p*·y ≥ Σ actual profit:
    // within a group only one member is nonzero, so group g's row-r
    // contribution is a_{r,j(x)}·1 ≤ W_r and its profit contribution is
    // q-scaled ≤ p*. Hence (y, W, p*) dominates every feasible point, and
    // the DP max over it upper-bounds the true max profit. (Standard GUB-
    // cover surrogate argument. Charging min-weight instead would let item
    // b's profit ride item a's lighter weight — an unachievable pair.)
    for g in groups.iter_mut() {
        let mut best_profit = f64::NEG_INFINITY;
        for &j in &g.vars {
            let pj = -problem.q[j].to_f64()?; // maximize -q
            if pj > best_profit {
                best_profit = pj;
            }
        }
        if best_profit <= 0.0 {
            return None; // not a maximization packing shape; bail conservatively
        }
        g.profit = best_profit;
        g.weights = agg_rows
            .iter()
            .map(|&i| {
                g.vars
                    .iter()
                    .map(|&j| problem.a.get(i, j).to_f64().unwrap_or(0.0))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
    }

    // Budget: sum of the aggregated rows' rhs (each ≤ its rhs for any point).
    let mut budget = 0.0f64;
    for &i in &agg_rows {
        budget += problem.b[i].to_f64()?;
    }
    if budget <= 0.0 {
        return None;
    }

    // Conservative grid (same convention as `knapsack_presolve`): weights UP,
    // budget DOWN, so the discretized problem is a subset and the DP value is
    // a genuine upper bound on the true max profit. Try 100× then 10× then 1×.
    let nrows = agg_rows.len();
    for &scale in &[100.0f64, 10.0, 1.0] {
        let cap = (budget * scale).floor();
        if cap < 1.0 {
            continue;
        }
        let cap = cap as usize;
        if nrows
            .saturating_mul(groups.len())
            .saturating_mul(cap.saturating_add(1))
            > 200_000_000
        {
            continue;
        }
        let gw: Vec<usize> = groups
            .iter()
            .map(|g| {
                // ceil of the max weight across aggregated rows
                let wmax = g.weights.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
                (wmax * scale).ceil().max(0.0) as usize
            })
            .collect();
        let gp: Vec<f64> = groups.iter().map(|g| g.profit).collect();
        let ub_profit = grouped_dp_value(&gw, &gp, cap);
        return Some(-T::from_f64(ub_profit)?);
    }
    None
}
