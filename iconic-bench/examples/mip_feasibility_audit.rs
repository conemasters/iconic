use iconic_bench::mip::build_mip_suite;
use iconic_core::Cone;
use iconic_mip::{solve_mip, MipSettings, MipStatus};

fn main() {
    let problems = build_mip_suite();
    let mut flagged = 0usize;
    let mut checked = 0usize;

    for (name, prob, _) in &problems {
        let settings = MipSettings::<f64> {
            max_nodes: 10000,
            max_time: 30.0,
            mip_presolve: true,
            heuristics: true,
            ..Default::default()
        };

        let sol = solve_mip(prob, &settings);
        if !matches!(sol.status, MipStatus::Optimal | MipStatus::Feasible) {
            continue;
        }
        checked += 1;
        let x = &sol.x;

        // Bound + integrality violations
        let mut max_bound_viol = 0.0f64;
        for (j, &xj) in x.iter().enumerate() {
            if xj < prob.lb[j] - 1e-6 {
                max_bound_viol = max_bound_viol.max(prob.lb[j] - xj);
            }
            if xj > prob.ub[j] + 1e-6 {
                max_bound_viol = max_bound_viol.max(xj - prob.ub[j]);
            }
        }
        let mut max_int_viol = 0.0f64;
        for (j, &xj) in x.iter().enumerate() {
            if prob.var_types[j].is_integer() {
                max_int_viol = max_int_viol.max((xj - xj.round()).abs());
            }
        }

        // Row violations, dispatching Zero (equality) vs NonNegative (<=) cones correctly
        let mut max_row_viol = 0.0f64;
        let mut row = 0usize;
        for cone in &prob.cones {
            let d = cone.dim();
            let is_zero = matches!(cone, Cone::Zero(_));
            for _ in 0..d {
                let ax: f64 = x
                    .iter()
                    .enumerate()
                    .map(|(j, &xj)| prob.a.get(row, j) * xj)
                    .sum();
                let viol = if is_zero {
                    (ax - prob.b[row]).abs()
                } else {
                    (ax - prob.b[row]).max(0.0)
                };
                if viol > max_row_viol {
                    max_row_viol = viol;
                }
                row += 1;
            }
        }

        let bad = max_bound_viol > 1e-5 || max_int_viol > 1e-5 || max_row_viol > 1e-5;
        if bad {
            flagged += 1;
            println!(
                "FLAGGED {name}: status={:?} max_bound_viol={max_bound_viol:.3e} max_int_viol={max_int_viol:.3e} max_row_viol={max_row_viol:.3e}",
                sol.status
            );
        }
    }
    println!("\nAudit complete: {checked} cases claimed Optimal/Feasible, {flagged} flagged as false positives.");
}
