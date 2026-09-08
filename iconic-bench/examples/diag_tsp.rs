#![allow(clippy::field_reassign_with_default)]
//! Isolate which component loses the true optimum on tsp_mtz n=8 (seed 60008):
//! the brute-force-verified tour costs 2.8649, the solver reports 3.4052.

use iconic_bench::mip::gen_tsp_mtz;
use iconic_mip::{check_feasibility, solve_mip, MipSettings};

fn main() {
    let prob = gen_tsp_mtz(8, 60008);

    let run = |label: &str, f: &dyn Fn(&mut MipSettings<f64>)| {
        let mut s = MipSettings::<f64>::default();
        s.max_nodes = 200_000;
        s.max_time = 60.0;
        f(&mut s);
        let sol = solve_mip(&prob, &s);
        let feas = if sol.x.is_empty() {
            "no-x".to_string()
        } else {
            format!("{}", check_feasibility(&sol.x, &prob))
        };
        println!(
            "{label:<28} status={:?} obj={:.6} bound={:.6} nodes={} feasible={feas}",
            sol.status, sol.obj_val, sol.best_bound, sol.nodes
        );
    };

    run("default", &|_s| {});
    run("no heuristics", &|s| s.heuristics = false);
    run("no presolve", &|s| s.mip_presolve = false);
    run("no cuts", &|s| {
        s.cut_rounds = 0;
    });
    run("none of the three", &|s| {
        s.heuristics = false;
        s.mip_presolve = false;
        s.cut_rounds = 0;
    });
}
