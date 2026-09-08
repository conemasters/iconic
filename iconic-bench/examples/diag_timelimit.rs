#![allow(clippy::field_reassign_with_default)]
//! Does `max_time` actually bound the solve? Runs the instances that overran their
//! budget worst and reports requested vs actual wall clock.

use iconic_bench::mip::build_mip_suite;
use iconic_mip::{solve_mip, MipSettings};

fn main() {
    let budget: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let filt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "facloc_f8_c20".into());
    let wanted = [filt];

    println!(
        "{:<24} {:>10} {:>10} {:>9}  status",
        "instance", "budget_s", "actual_s", "over"
    );
    for (name, prob, _) in build_mip_suite() {
        if !wanted.iter().any(|w| name.contains(w)) {
            continue;
        }
        let mut s = MipSettings::<f64>::default();
        s.max_nodes = 10_000;
        s.max_time = budget;

        let t0 = std::time::Instant::now();
        let sol = solve_mip(&prob, &s);
        let el = t0.elapsed().as_secs_f64();
        println!(
            "{name:<24} {budget:>10.1} {el:>10.1} {:>8.1}x  {:?} nodes={} simplex_iters={}",
            el / budget,
            sol.status,
            sol.nodes,
            sol.simplex_iters
        );
    }
}
