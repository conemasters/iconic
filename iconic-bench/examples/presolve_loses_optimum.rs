//! Which presolve pass discards a known-optimal point?
//!
//! A pass is allowed to move the optimum to a different point (symmetry breaking keeps one
//! representative per orbit) but never to change its *value*. When the value drops, some
//! pass removed every optimal solution, and the search afterwards reports a proof of the
//! wrong answer. Running the passes one at a time against a point whose objective is known
//! names the culprit directly, which is otherwise invisible: the solve simply returns a
//! worse number and calls it Optimal.
//!
//! The instance is bin packing with a conflict graph, carrying the symmetry-breaking
//! fixings `x_{i,k} = 0` for `k < B-1-i`. Those are sound -- the point checked below
//! satisfies them -- and they are what makes the bug reachable, because they leave many
//! variables fixed at `ub = 0`.

use iconic_bench::mip::gen_bin_packing_conflict;
use iconic_core::Cone;
use iconic_mip::{
    aggregate_parallel_rows, cancel_mip_nonzeros, fbbt_presolve, mip_presolve,
    mip_redundancy_detect, probing_presolve, substitute_implied_fixed, MipProblem,
};

/// Is `x` feasible for `p`, to the suite's own constraint tolerance?
fn feasible(p: &MipProblem<f64>, x: &[f64]) -> Result<(), String> {
    let (n, m) = (p.q.len(), p.b.len());
    if x.len() != n {
        return Err(format!("dimension changed to n={n}"));
    }
    for (j, &xj) in x.iter().enumerate() {
        if xj < p.lb[j] - 1e-6 || xj > p.ub[j] + 1e-6 {
            return Err(format!("var {j} = {xj} outside [{}, {}]", p.lb[j], p.ub[j]));
        }
    }
    let mut r = 0usize;
    for cone in &p.cones {
        let d = cone.dim();
        let eq = matches!(cone, Cone::Zero(_));
        for i in r..(r + d).min(m) {
            let act: f64 = (0..n).map(|j| p.a.get(i, j) * x[j]).sum();
            let bad = if eq {
                (act - p.b[i]).abs() > 1e-6
            } else {
                act > p.b[i] + 1e-6
            };
            if bad {
                return Err(format!(
                    "row {i} ({}) act={act:.6} b={:.6}",
                    if eq { "eq" } else { "leq" },
                    p.b[i]
                ));
            }
        }
        r += d;
    }
    Ok(())
}

fn main() {
    let (n_items, n_bins) = (8usize, 8usize);
    let mut p = gen_bin_packing_conflict(n_items, 0.3, 210008);

    // Symmetry fixings: item `i` may not use a bin below `B-1-i`.
    for i in 0..n_items {
        for k in 0..n_bins {
            if k + i + 1 < n_bins {
                p.ub[i * n_bins + k] = 0.0;
            }
        }
    }

    // A known optimum, relabelled onto the high bins so it respects the fixings.
    let groups: [&[usize]; 3] = [&[0, 1, 3, 4], &[2], &[5, 6, 7]];
    let labels = [7usize, 6, 5];
    let mut x = vec![0.0f64; p.q.len()];
    for (g, &lab) in groups.iter().zip(labels.iter()) {
        for &item in *g {
            x[item * n_bins + lab] = 1.0;
        }
        x[n_bins * n_bins + lab] = 1.0;
    }
    let obj: f64 = (0..p.q.len()).map(|j| p.q[j] * x[j]).sum();
    println!("reference point objective = {obj}  (true optimum 3)");
    match feasible(&p, &x) {
        Ok(()) => println!("feasible after the fixings themselves: yes"),
        Err(e) => {
            println!("ALREADY infeasible before any pass: {e}");
            return;
        }
    }

    type Pass = (&'static str, fn(&mut MipProblem<f64>));
    let passes: Vec<Pass> = vec![
        ("aggregate_parallel_rows", |p| aggregate_parallel_rows(p)),
        ("substitute_implied_fixed", |p| substitute_implied_fixed(p)),
        ("mip_presolve", |p| {
            mip_presolve(p);
        }),
        ("fbbt_presolve", |p| {
            fbbt_presolve(p);
        }),
        ("probing_presolve", |p| {
            probing_presolve(p);
        }),
        ("cancel_mip_nonzeros", |p| {
            cancel_mip_nonzeros(p);
        }),
        ("mip_redundancy_detect", |p| {
            mip_redundancy_detect(p);
        }),
    ];

    // Cumulative, in pipeline order: report the first pass after which the point dies.
    for (name, f) in &passes {
        f(&mut p);
        match feasible(&p, &x) {
            Ok(()) => println!("  after {name:<26} still feasible"),
            Err(e) => {
                println!("  after {name:<26} LOST -- {e}");
                return;
            }
        }
    }
    println!("all passes preserved the reference optimum");
}
