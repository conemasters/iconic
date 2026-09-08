#![allow(clippy::field_reassign_with_default)]
//! Diagnostic: is the branch-and-bound loop pivot-bound or setup-bound?
//!
//! Each node rebuilds its LP from scratch (`build_node_lp` fills a fresh dense
//! `m x n` matrix, `DualSolver::new` re-seeds a basis) and only then hot-starts
//! the dual simplex from the parent basis. When the hot start is doing its job
//! it converges in a handful of pivots, so a low pivots-per-node ratio means the
//! per-node *setup* -- not the simplex work it exists to serve -- dominates.
//!
//! Prints per instance: nodes, total simplex iterations, pivots/node, and time.

use iconic_bench::mip::build_mip_suite;
use iconic_mip::{solve_mip, take_node_lp_timing, MipSettings};

fn main() {
    let suite = build_mip_suite();
    let max_time: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    println!(
        "{:<34} {:>6} {:>7} {:>9} {:>10} {:>11}",
        "instance", "nodes", "pivots", "piv/node", "time_ms", "us/node"
    );

    let mut tot_nodes = 0usize;
    let mut tot_piv = 0usize;
    let mut tot_ms = 0.0f64;

    for (name, prob, _known_opt) in &suite {
        let mut settings = MipSettings::<f64>::default();
        settings.max_nodes = 10_000;
        settings.max_time = max_time;

        let t0 = std::time::Instant::now();
        let sol = solve_mip(prob, &settings);
        let ms = t0.elapsed().as_secs_f64() * 1e3;

        let per_node = if sol.nodes > 0 {
            sol.simplex_iters as f64 / sol.nodes as f64
        } else {
            0.0
        };
        let us_node = if sol.nodes > 0 {
            ms * 1e3 / sol.nodes as f64
        } else {
            0.0
        };
        let (setup_ms, solve_ms) = take_node_lp_timing();
        let pct = if setup_ms + solve_ms > 0.0 {
            100.0 * setup_ms / (setup_ms + solve_ms)
        } else {
            0.0
        };
        println!(
            "{:<34} {:>6} {:>7} {:>9.2} {:>10.1} {:>11.1}  setup={setup_ms:8.1}ms solve={solve_ms:8.1}ms setup={pct:.0}%",
            name, sol.nodes, sol.simplex_iters, per_node, ms, us_node
        );

        tot_nodes += sol.nodes;
        tot_piv += sol.simplex_iters;
        tot_ms += ms;
    }

    println!(
        "\nTOTAL nodes={tot_nodes} pivots={tot_piv} piv/node={:.2} time={tot_ms:.0}ms",
        tot_piv as f64 / tot_nodes.max(1) as f64
    );
}
