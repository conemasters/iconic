//! Trace the transport LP family (P = 0.1·I, sparse — the augmented sparse-KKT
//! path). Run: cargo run --release -p iconic-bench --example diag_lp_transport -- 10 12
use iconic_bench::lp_transport;
use iconic_core::Settings;
use iconic_presolve::solve_presolved;

fn main() {
    let sup: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let dem: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let prob = lp_transport(sup, dem, 61);
    let sol = solve_presolved(&prob, &Settings::default());
    println!(
        "sparse_kkt(default): status={:?} iters={} obj={:.10}",
        sol.status, sol.iters, sol.obj_val
    );
    let s = Settings {
        sparse_kkt: false,
        ..Default::default()
    };
    let sol2 = solve_presolved(&prob, &s);
    println!(
        "dense:              status={:?} iters={} obj={:.10}",
        sol2.status, sol2.iters, sol2.obj_val
    );
    let s3 = Settings {
        presolve: false,
        ..Default::default()
    };
    let sol3 = solve_presolved(&prob, &s3);
    println!(
        "no presolve:        status={:?} iters={} obj={:.10}",
        sol3.status, sol3.iters, sol3.obj_val
    );
}
