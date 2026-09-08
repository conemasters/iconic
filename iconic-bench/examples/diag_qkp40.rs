//! Diagnostic: which heuristic found qkp_n40's incumbent, and at what cost?
//!
//! The root-heuristic budget changes (per-call budgets, pump max_iters
//! 50->10, pump moved to run last) moved qkp_n40's incumbent from -1316.998
//! to -1307.526. This prints the full heuristic diary so the
//! regression can be attributed to a specific heuristic before A/B-ing the
//! budget.
//!
//! Usage: `cargo run --release -p iconic-bench --example diag_qkp40 [max_time]`
//! Set ICONIC_FP_TRACE=1 for per-iteration pump/dive traces.

use iconic_bench::mip::build_mip_suite;
use iconic_mip::{solve_mip, MipSettings};

fn main() {
    let suite = build_mip_suite();
    let max_time: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);

    for (name, prob, _known_opt) in suite {
        if !name.starts_with("qkp_n40") {
            continue;
        }
        let settings = MipSettings::<f64> {
            max_nodes: 10_000,
            max_time,
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let sol = solve_mip(&prob, &settings);
        let ms = t0.elapsed().as_secs_f64();
        println!(
            "{}: status={:?} obj={:.6} bound={:.6} nodes={} pivots={} time={:.3}s root_heur_spend={:.3}s",
            name, sol.status, sol.obj_val, sol.best_bound, sol.nodes, sol.simplex_iters, ms,
            sol.heur_root_spend
        );
        for e in &sol.heur_events {
            println!(
                "  heur {:>24} verdict={:?} ms={:.1}",
                e.name, e.verdict, e.ms
            );
        }
    }
}
