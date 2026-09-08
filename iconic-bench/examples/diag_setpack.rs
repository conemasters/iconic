// Index-based loops read more clearly than iterator adapters here.
#![allow(clippy::needless_range_loop)]
use iconic_bench::mip::gen_set_packing;
use iconic_core::Scalar;
use iconic_mip::{solve_mip, MipProblem, MipSettings};

fn greedy_mis<T: Scalar + PartialOrd>(prob: &MipProblem<T>) -> Vec<T> {
    let n = prob.q.len();
    let one = T::one();
    let zero = T::zero();
    let eps = T::from_f64(1e-8).unwrap();

    // Sort variables by weight (descending) — heavier is better in max independent set
    let mut indices: Vec<usize> = (0..n).collect();
    // Objective is min -w·x, so higher -q[j] means higher weight
    indices.sort_by(|&a, &b| {
        prob.q[b]
            .partial_cmp(&prob.q[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = vec![false; n];
    let mut solution = vec![zero; n];

    for &j in &indices {
        // Check if j conflicts with any already-selected variable
        let mut conflicts = false;
        for k in 0..n {
            if selected[k] {
                // Check every row for a conflict
                for i in 0..prob.b.len() {
                    let a_ij = prob.a.get(i, j);
                    let a_ik = prob.a.get(i, k);
                    if a_ij > eps && a_ik > eps && a_ij + a_ik > prob.b[i] + eps {
                        conflicts = true;
                        break;
                    }
                }
                if conflicts {
                    break;
                }
            }
        }
        if !conflicts {
            selected[j] = true;
            solution[j] = one;
        }
    }
    solution
}

fn main() {
    // Generate the exact instance
    let prob = gen_set_packing(80, 0.15, 12222 + 80);

    println!("=== setpack_hard_n80_d15 ===");
    println!(
        "n={}, m={}, n_bin={}",
        prob.q.len(),
        prob.b.len(),
        prob.var_types.iter().filter(|v| v.is_integer()).count()
    );

    // Compute greedy MIS incumbent
    let greedy_x = greedy_mis(&prob);
    let greedy_obj: f64 = greedy_x
        .iter()
        .enumerate()
        .map(|(j, &xj)| prob.q[j] * xj)
        .sum();
    let greedy_count = greedy_x.iter().filter(|&&xj| xj > 0.5).count();
    println!(
        "Greedy MIS: obj={:.6}, selected {} / {} vars",
        greedy_obj,
        greedy_count,
        prob.q.len()
    );

    // Verify feasibility
    let max_row_viol: f64 = (0..prob.b.len())
        .map(|i| {
            let ax: f64 = (0..prob.q.len())
                .map(|j| prob.a.get(i, j) * greedy_x[j])
                .sum();
            (ax - prob.b[i]).max(0.0)
        })
        .fold(0.0, f64::max);
    println!("Greedy max row violation: {:.3e}", max_row_viol);

    // Run with default settings (30s, 10k nodes)
    let settings = MipSettings::<f64> {
        max_nodes: 10000,
        max_time: 30.0,
        mip_presolve: true,
        heuristics: true,
        ..Default::default()
    };
    println!("\n=== Running MIP solver ===");
    let sol = solve_mip(&prob, &settings);
    println!("\n=== Result ===");
    println!(
        "status={:?}, obj={:.6}, best_bound={:.6}, gap={:.6}, nodes={}, time={:.3}s",
        sol.status, sol.obj_val, sol.best_bound, sol.rel_gap, sol.nodes, 0.0
    ); // time not in Solution
}
