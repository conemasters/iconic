//! Temporary probe: time the socp n600_c30_d15 solve's per-iteration split.
//! Removed once the split is known (the probe exists to find the 694ms/iter).

use iconic_bench::kkt_residual;
use iconic_bench::random_socp;
use iconic_core::Settings;
use iconic_ipm::conic::solve_cone_qp;

fn main() {
    let settings = Settings::<f64>::default();
    // Mimic the suite's ordering: smaller SOCPs first, then n600.
    for &(n, c, d) in &[
        (10, 2, 4),
        (25, 4, 5),
        (50, 6, 6),
        (100, 8, 8),
        (200, 12, 10),
        (400, 20, 12),
    ] {
        let (prob, cones) = random_socp(n, c, d, 11);
        let sol = solve_cone_qp(&prob, &cones, &settings);
        println!(
            "socp n{n}_c{c}_d{d}: {:?} {} iters kkt={:.1e}",
            sol.status,
            sol.iters,
            kkt_residual(&prob, &sol)
        );
    }
    let (prob, cones) = random_socp(600, 30, 15, 11);
    let t0 = std::time::Instant::now();
    let sol = solve_cone_qp(&prob, &cones, &settings);
    println!(
        "socp n600_c30_d15: status={:?} iters={} total={:.3}s kkt={:.2e}",
        sol.status,
        sol.iters,
        t0.elapsed().as_secs_f64(),
        kkt_residual(&prob, &sol)
    );
}
