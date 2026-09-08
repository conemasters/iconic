#![allow(clippy::field_reassign_with_default)]
//! CVRP diagnosis: which component loses cvrp_n10_k3 / cvrp_n15_k4?
//!
//! (a) incumbent quality — the route-construction heuristic's objective vs the
//!     brute-force optimum; (b) root bound — LP bound without cuts vs after the cut
//!     rounds, vs the brute-force optimum; (c) tree progress — final obj/bound/nodes.

use iconic_bench::mip::gen_cvrp;
use iconic_core::rng::XorShift;
use iconic_mip::{bounds, solve_mip, MipSettings};

/// True optimum of the generated instance by exhaustive search: every assignment of
/// customers to `k` routes (3^n), route order cost via Held-Karp over the subset
/// (depot-start path, min over the ending customer, + return arc). `n <= 15`.
fn brute_force_optimum(cust: usize, k: usize, seed: u64) -> Option<f64> {
    let nc = cust + 1;
    // Coordinates / demands replicated from gen_cvrp (XorShift).
    let mut rng = XorShift::new(seed);
    let xs: Vec<f64> = (0..nc).map(|_| rng.uniform(0.0, 1.0)).collect();
    let ys: Vec<f64> = (0..nc).map(|_| rng.uniform(0.0, 1.0)).collect();
    let dem: Vec<f64> = std::iter::once(0.0)
        .chain((1..nc).map(|_| rng.uniform(5.0, 25.0)))
        .collect();
    let q_cap = dem.iter().sum::<f64>() / k as f64 * 1.1;
    let dist =
        |i: usize, j: usize| -> f64 { ((xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2)).sqrt() };
    let c: Vec<Vec<f64>> = (0..nc)
        .map(|i| (0..nc).map(|j| dist(i, j)).collect())
        .collect();

    // Held-Karp: path[i][S] = cheapest depot->S path ending at customer i.
    let n_subsets = 1usize << cust;
    let inf = f64::INFINITY;
    let mut path = vec![vec![inf; n_subsets]; cust];
    for i in 0..cust {
        path[i][1 << i] = c[0][i + 1];
    }
    for s in 1..n_subsets {
        for i in 0..cust {
            if s & (1 << i) == 0 {
                continue;
            }
            let prev = s ^ (1 << i);
            if prev == 0 {
                continue;
            }
            let mut best = inf;
            let mut q = prev;
            while q != 0 {
                let j = q.trailing_zeros() as usize;
                q &= q - 1;
                let cand = path[j][prev] + c[j + 1][i + 1];
                if cand < best {
                    best = cand;
                }
            }
            path[i][s] = best;
        }
    }
    let route_cost = |s: usize| -> f64 {
        let mut best = inf;
        let mut q = s;
        while q != 0 {
            let i = q.trailing_zeros() as usize;
            q &= q - 1;
            best = best.min(path[i][s] + c[i + 1][0]);
        }
        best
    };
    let route_load = |s: usize| -> f64 {
        let mut l = 0.0;
        for i in 0..cust {
            if s & (1 << i) != 0 {
                l += dem[i + 1];
            }
        }
        l
    };

    // Enumerate assignments of the first customer to each route (canonical: route 0
    // must contain customer 0, so only 3^(n-1) assignments). The depot rows demand
    // exactly `k` departures, so every route must be non-empty. Each route's cost is
    // the Held-Karp optimum of ITS customers, evaluated at the leaf from the bitset.
    let mut best = inf;
    let mut route = vec![0usize; cust];
    let mut load = vec![0.0; k];
    let mut sets = vec![0usize; k];
    #[allow(clippy::too_many_arguments)]
    fn rec(
        cust: usize,
        k: usize,
        i: usize,
        cap: f64,
        route: &mut [usize],
        load: &mut [f64],
        sets: &mut [usize],
        route_cost: &dyn Fn(usize) -> f64,
        route_load: &dyn Fn(usize) -> f64,
        best: &mut f64,
    ) {
        if i == cust {
            if sets.iter().all(|&s| s != 0) {
                *best = (*best).min((0..k).map(|r| route_cost(sets[r])).sum::<f64>());
            }
            return;
        }
        for r in 0..k {
            if r > 0 && i == 0 {
                continue; // canonical: customer 0 in route 0
            }
            if load[r] + route_load(1 << i) > cap + 1e-9 {
                continue;
            }
            route[i] = r;
            let old_l = load[r];
            load[r] += route_load(1 << i);
            sets[r] |= 1 << i;
            if load[r] <= cap + 1e-9 {
                rec(
                    cust,
                    k,
                    i + 1,
                    cap,
                    route,
                    load,
                    sets,
                    route_cost,
                    route_load,
                    best,
                );
            }
            route[i] = 0;
            load[r] = old_l;
            sets[r] &= !(1 << i);
        }
    }
    rec(
        cust,
        k,
        0,
        q_cap,
        &mut route,
        &mut load,
        &mut sets,
        &route_cost,
        &route_load,
        &mut best,
    );
    if best.is_finite() {
        Some(best)
    } else {
        None
    }
}

fn main() {
    for &(cust, k) in &[(8usize, 2usize), (10, 3), (15, 4)] {
        let prob = gen_cvrp(cust, k, 85000 + cust as u64 + k as u64);
        let name = format!("cvrp_n{cust}_k{k}");
        let settings = MipSettings::<f64>::default();

        // (0) Ground truth.
        let opt = brute_force_optimum(cust, k, 85000 + cust as u64 + k as u64);
        println!(
            "{name:14} brute-force optimum {:?}",
            opt.map(|o| format!("{o:.6}"))
        );

        // (a) Heuristic incumbent.
        match bounds::vrp_nearest_neighbor_routes(&prob, &settings) {
            Some((x, obj)) => {
                let recomputed: f64 = (0..prob.q.len()).map(|j| prob.q[j] * x[j]).sum();
                println!("{name:14} heuristic obj {obj:.6} (q.x {recomputed:.6})");
            }
            None => println!("{name:14} heuristic declined"),
        }

        // (b) LP bounds. With and without the root cut rounds.
        for label in ["no cuts", "cuts"] {
            let mut s = MipSettings::<f64>::default();
            if label == "no cuts" {
                s.cut_rounds = 0;
            }
            s.max_time = 30.0;
            s.max_nodes = 200_000;
            let sol = solve_mip(&prob, &s);
            println!(
                "{name:14} {label:8} status={:?} obj={:.6} bound={:.6} nodes={}",
                sol.status, sol.obj_val, sol.best_bound, sol.nodes
            );
        }
    }
}
