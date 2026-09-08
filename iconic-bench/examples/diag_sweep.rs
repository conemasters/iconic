#![allow(clippy::field_reassign_with_default)]
//! Bisect a slow MIP instance by turning components off one at a time.
//!
//! Instrumentation placement proved unreliable on `setpack_hard_n80_d15` (timers were
//! bypassed by `continue` paths and by code after the last marker), so identify the
//! expensive component by removing it instead.

use iconic_bench::mip::build_mip_suite;
use iconic_mip::{solve_mip, BranchingRule, MipSettings};

fn main() {
    let filt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "setpack_hard_n80".into());
    let budget: f64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    // (label, mutator) pairs.
    type Config = (&'static str, fn(&mut MipSettings<f64>));
    let configs: Vec<Config> = vec![
        ("default", |_s| {}),
        ("no cuts", |s| s.cut_rounds = 0),
        ("no heuristics", |s| s.heuristics = false),
        ("no presolve", |s| s.mip_presolve = false),
        ("no tree_cuts", |s| s.tree_cuts = false),
        ("most-fractional branching", |s| {
            s.branching = BranchingRule::MostFractional
        }),
        ("bare (none of them)", |s| {
            s.cut_rounds = 0;
            s.heuristics = false;
            s.mip_presolve = false;
            s.tree_cuts = false;
            s.branching = BranchingRule::MostFractional;
        }),
    ];

    for (name, prob, _) in build_mip_suite() {
        if !name.contains(&filt) {
            continue;
        }
        println!("== {name} (budget {budget}s) ==");
        for (label, apply) in &configs {
            let mut s = MipSettings::<f64>::default();
            s.max_nodes = 100_000;
            s.max_time = budget;
            apply(&mut s);
            let t0 = std::time::Instant::now();
            let sol = solve_mip(&prob, &s);
            println!(
                "  {label:<28} {:>8.1}s  {:>6.1}x  {:?} nodes={}",
                t0.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64() / budget,
                sol.status,
                sol.nodes
            );
        }
    }
}
