//! Report the bound the search has after the root node, per instance.
//!
//! When branch-and-bound times out, the cause is either throughput (nodes are slow) or
//! bound quality (the tree is huge because the bound is weak). These need opposite fixes
//! and the suite's time column cannot tell them apart. The root bound can: compare it
//! against the integer optimum and the size of the remaining gap is exactly the work the
//! search still has to do.
//!
//! Prints, per instance, the bound after the root (`max_nodes = 1`), the bound after a
//! full run, and the incumbent each produced. Pass instance-name substrings to filter.
//!
//!     cargo run --release -p iconic-bench --example root_gap -- tsptw maxsat

use iconic_bench::mip::build_mip_suite;
use iconic_mip::{solve_mip, MipSettings};

fn main() {
    let filters: Vec<String> = std::env::args().skip(1).collect();
    let problems = build_mip_suite();

    println!(
        "{:<26}{:>14}{:>14}{:>14}{:>10}",
        "instance", "root_bound", "final_bound", "incumbent", "nodes"
    );
    for (name, prob, _) in &problems {
        if !filters.is_empty() && !filters.iter().any(|f| name.contains(f.as_str())) {
            continue;
        }
        let root = solve_mip(
            prob,
            &MipSettings::<f64> {
                max_nodes: 1,
                max_time: 30.0,
                ..Default::default()
            },
        );
        let full = solve_mip(
            prob,
            &MipSettings::<f64> {
                max_nodes: 10_000,
                max_time: 30.0,
                ..Default::default()
            },
        );
        println!(
            "{:<26}{:>14.6}{:>14.6}{:>14.6}{:>10}",
            name, root.best_bound, full.best_bound, full.obj_val, full.nodes
        );
    }
}
