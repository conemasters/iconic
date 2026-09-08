#![allow(clippy::field_reassign_with_default)]
//! Isolate which component turns the j4_m3 job-shop instance into a trivially
//! zero-makespan problem: MIP presolve, the heuristics, or their combination.

use iconic_bench::mip::gen_job_shop;
use iconic_mip::{solve_mip, MipSettings};

fn main() {
    let prob = gen_job_shop(4, 3, 190004);
    for &(pre, heur) in &[(false, false), (true, false), (false, true), (true, true)] {
        let mut s = MipSettings::<f64>::default();
        s.max_nodes = 5000;
        s.max_time = 60.0;
        s.mip_presolve = pre;
        s.heuristics = heur;
        let sol = solve_mip(&prob, &s);
        println!(
            "presolve={pre:<5} heuristics={heur:<5} -> status={:?} obj={:.6} bound={:.6} nodes={}",
            sol.status, sol.obj_val, sol.best_bound, sol.nodes
        );
    }
}
