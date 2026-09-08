//! Trace qp_random through the QP path (exact suite instance — the dense-P
//! arm where the mu-squared regularization must stay at full strength).
//! Run: ICONIC_TRACE_LP=1 cargo run --release -p iconic-bench --example diag_qp_random -- 100 80
use iconic_bench::random_qp;
use iconic_core::Settings;
use iconic_presolve::solve_presolved;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let m: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);
    let seed: u64 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let prob = random_qp(n, m, seed);
    let sol = solve_presolved(&prob, &Settings::default());
    println!(
        "status={:?} iters={} obj={:.10}",
        sol.status, sol.iters, sol.obj_val
    );
}
