//! Diagnostic: attribute node-count changes from the heuristic spend meter,
//! the gap-triggered improvement mode, and the Driebeek pseudo-cost seed.
//!
//! The suite runs each instance with the default `MipSettings`; this binary
//! re-solves one instance under settings variants so the components can be
//! measured separately:
//!   - variant `plain`: meter off + improvement mode off (the pre-meter
//!     scheduling; only the Driebeek seed differs from the old code),
//!   - variant `meter`: meter on, improvement mode off,
//!   - variant `improve`: meter on, improvement mode on (the new defaults),
//!   - variant `nopolish`: improvement mode on but polish disabled is not a
//!     setting -- the polish gate is the improvement mode itself, so this
//!     variant is not run; the two settings above bracket it.
//!
//! Usage: diag_heur_meter [name-filter] [max_nodes] [max_time]

use iconic_bench::mip::build_mip_suite;
use iconic_mip::{solve_mip, MipSettings};

fn main() {
    let filter = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "mdk_hard_n60".to_string());
    let max_nodes: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let max_time: f64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);

    let suite = build_mip_suite();
    println!(
        "{:<30} {:<9} {:>8} {:>9} {:>10} {:>12} {:>10}",
        "instance", "variant", "nodes", "iters", "time_ms", "status", "obj"
    );
    for (name, prob, _known_opt) in &suite {
        if !name.contains(&filter) {
            continue;
        }
        let variants: [(&str, MipSettings<f64>); 3] = [
            ("plain", {
                MipSettings::<f64> {
                    max_nodes,
                    max_time,
                    heuristic_time_frac: 0.0,
                    improvement_gap_threshold: 0.0,
                    ..Default::default()
                }
            }),
            ("meter", {
                MipSettings::<f64> {
                    max_nodes,
                    max_time,
                    improvement_gap_threshold: 0.0,
                    ..Default::default()
                }
            }),
            ("default", {
                MipSettings::<f64> {
                    max_nodes,
                    max_time,
                    ..Default::default()
                }
            }),
        ];
        for (label, settings) in variants {
            let t0 = std::time::Instant::now();
            let sol = solve_mip(prob, &settings);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "{:<30} {:<9} {:>8} {:>9} {:>10.1} {:>12?} {:>10.6}",
                name, label, sol.nodes, sol.simplex_iters, ms, sol.status, sol.obj_val
            );
        }
    }
}
