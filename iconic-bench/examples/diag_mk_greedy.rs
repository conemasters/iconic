//! Probe: current multi_knapsack_greedy value on multiknap_i25_b4 (optimum -764.8918500604)
use iconic_bench::mip::gen_multiple_knapsack;
use iconic_mip::{check_feasibility, solve_mip, MipSettings};

fn main() {
    let prob = gen_multiple_knapsack(25, 4, 130025);
    let (x, obj) = iconic_mip::bounds::multi_knapsack_greedy_heuristic(&prob).unwrap();
    println!("greedy obj: {:.10} feasible: {}", -obj, check_feasibility(&x, &prob));
    let s = MipSettings::<f64> {
        max_nodes: 0,
        max_time: 30.0,
        cut_rounds: 0,
        tree_cuts: false,
        heuristics: false,
        ..Default::default()
    };
    let sol = solve_mip(&prob, &s);
    println!("root LP bound: {:.10}", sol.best_bound);
}
