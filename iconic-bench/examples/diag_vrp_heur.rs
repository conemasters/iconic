//! Does the CVRP construction heuristic fire, and what does it return?
//!
//! The root-heuristic log only shows what ran, so a heuristic that declines the instance
//! is indistinguishable from one that was never reached (the root phase budget skips the
//! tail once it is spent). Calling it directly separates those.

use iconic_bench::mip::gen_cvrp;
use iconic_mip::{bounds, MipSettings};

fn main() {
    for &(cust, k) in &[(8usize, 2usize), (10, 3), (15, 4)] {
        let prob = gen_cvrp(cust, k, 85000 + cust as u64 + k as u64);
        let settings = MipSettings::<f64>::default();
        let name = format!("cvrp_n{cust}_k{k}");
        match bounds::vrp_nearest_neighbor_routes(&prob, &settings) {
            Some((x, obj)) => {
                let recomputed: f64 = (0..prob.q.len()).map(|j| prob.q[j] * x[j]).sum();
                println!("{name:14} -> obj {obj:.6}  (q.x = {recomputed:.6})");
            }
            None => println!("{name:14} -> None (declined)"),
        }
    }
}
