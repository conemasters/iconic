//! Probe the "shape1" SOCP (n=12, 5 SOC cones dim 4): sweep seeds,
//! reporting status/iters/worst. Run: cargo run --release -p iconic-bench --example diag_socp_shape1
use iconic_bench::random_socp;
use iconic_core::Settings;
use iconic_ipm::conic::solve_cone_qp;

fn main() {
    let settings = Settings::<f64>::default();
    let mut worst = (0usize, 0.0f64, 0u64, String::new());
    let mut worst_status = (0usize, String::new());
    for seed in 0..48u64 {
        let (prob, cones) = random_socp(12, 5, 4, seed);
        let sol = solve_cone_qp(&prob, &cones, &settings);
        let st = format!("{:?}", sol.status);
        println!(
            "seed={:2} status={} iters={:2} obj={:.8}",
            seed, st, sol.iters, sol.obj_val
        );
        if sol.iters > worst.0 || (st == "NumericalError" || st == "MaxIterations") {
            worst = (sol.iters, sol.obj_val, seed, st.clone());
        }
        if st == "NumericalError" || st == "MaxIterations" {
            worst_status = (sol.iters, st.clone());
        }
    }
    println!(
        "worst: iters={} status={} seed={} obj={}",
        worst.0, worst.3, worst.2, worst.1
    );
    println!("worst_status: {:?}", worst_status);
    // Nearby shapes for context.
    for &(n, c, d) in &[
        (12, 4, 4),
        (12, 6, 4),
        (10, 5, 4),
        (16, 5, 4),
        (12, 5, 3),
        (12, 5, 5),
    ] {
        let mut w = (0usize, 0u64, String::new());
        for seed in 0..48u64 {
            let (prob, cones) = random_socp(n, c, d, seed);
            let sol = solve_cone_qp(&prob, &cones, &settings);
            if sol.iters > w.0 {
                w = (sol.iters, seed, format!("{:?}", sol.status));
            }
        }
        println!(
            "shape n={} c={} d={}: worst {}/48 seeds -> {} iters {} (seed {})",
            n, c, d, 24, w.0, w.2, w.1
        );
    }
}
