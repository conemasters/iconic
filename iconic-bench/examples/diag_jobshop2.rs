#![allow(clippy::field_reassign_with_default)]
//! Which root heuristic produces the zero-makespan incumbent on j4_m3, and is
//! the point it returns actually feasible for the original problem?

use iconic_bench::mip::gen_job_shop;
use iconic_mip::{check_feasibility, heuristics, MipSettings};

fn main() {
    let prob = gen_job_shop(4, 3, 190004);
    let settings = MipSettings::<f64>::default();

    let report = |name: &str, r: Option<(Vec<f64>, f64)>| match r {
        None => println!("{name:<18} -> none"),
        Some((x, obj)) => {
            let feas = check_feasibility(&x, &prob);
            println!("{name:<18} -> obj={obj:.6}  check_feasibility={feas}");
            if !feas {
                // Report the worst violated row so the failure is concrete.
                let n = x.len();
                let mut worst = (0usize, 0.0f64);
                let mut row = 0usize;
                for cone in &prob.cones {
                    let is_zero = matches!(cone, iconic_core::Cone::Zero(_));
                    for _ in 0..cone.dim() {
                        if row >= prob.b.len() {
                            break;
                        }
                        let ax: f64 = (0..n).map(|j| prob.a.get(row, j) * x[j]).sum();
                        let s = prob.b[row] - ax;
                        let v = if is_zero { s.abs() } else { (-s).max(0.0) };
                        if v > worst.1 {
                            worst = (row, v);
                        }
                        row += 1;
                    }
                }
                println!(
                    "{:<18}    worst row {} violated by {:.6e}",
                    "", worst.0, worst.1
                );
            }
        }
    };

    report(
        "feasibility_pump",
        heuristics::feasibility_pump(&prob, &settings, 5),
    );
    report(
        "relaxation_free_search",
        heuristics::relaxation_free_search(&prob, &settings),
    );
    report(
        "rounding_dive",
        heuristics::rounding_dive(&prob, &settings, prob.q.len()),
    );
}
