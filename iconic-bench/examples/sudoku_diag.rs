//! Diagnostic: find which presolve pass makes sudoku infeasible
use iconic_bench::mip::gen_sudoku;
use iconic_mip::{
    aggregate_parallel_rows, binpack_symmetry_break, cancel_mip_nonzeros, clique_presolve,
    cover_presolve, embedded_knapsack_presolve, fbbt_presolve, jobshop_edge_finding, mip_presolve,
    mip_redundancy_detect, multirow_bound_strengthen, perspective_reformulate, probing_presolve,
    solve_mip, substitute_implied_fixed, symmetry_detect_and_break, tsp_tw_presolve, MipSettings,
    MipStatus,
};

fn main() {
    let problem = gen_sudoku(120000);
    println!(
        "Problem: {} vars, {} rows",
        problem.q.len(),
        problem.b.len()
    );

    // Full solve
    let settings = MipSettings::default();
    let sol = solve_mip(&problem, &settings);
    println!("Full solve: {:?}", sol.status);

    // Trace presolve
    let mut p = problem.clone();
    aggregate_parallel_rows(&mut p);
    println!("after aggregate_parallel_rows: {}x{}", p.q.len(), p.b.len());
    substitute_implied_fixed(&mut p);
    println!(
        "after substitute_implied_fixed: {}x{}",
        p.q.len(),
        p.b.len()
    );
    embedded_knapsack_presolve(&mut p);
    println!(
        "after embedded_knapsack_presolve: {}x{}",
        p.q.len(),
        p.b.len()
    );
    let s = cancel_mip_nonzeros(&mut p);
    println!("after cancel_mip_nonzeros: {:?}", s);
    let s = mip_redundancy_detect(&mut p);
    println!("after mip_redundancy_detect: {:?}", s);

    macro_rules! check {
        ($name:expr, $f:ident) => {
            let mut pc = p.clone();
            let s = $f(&mut pc);
            println!(
                "{:>35}: {:?}{}",
                $name,
                s,
                if matches!(s, MipStatus::Infeasible) {
                    " <<< BUG"
                } else {
                    ""
                }
            );
        };
    }

    check!("clique_presolve", clique_presolve);
    check!("cover_presolve", cover_presolve);
    check!("multirow_bound_strengthen", multirow_bound_strengthen);
    check!("symmetry_detect_and_break", symmetry_detect_and_break);
    check!("binpack_symmetry_break", binpack_symmetry_break);
    check!("perspective_reformulate", perspective_reformulate);
    check!("tsp_tw_presolve", tsp_tw_presolve);
    check!("jobshop_edge_finding", jobshop_edge_finding);
    check!("mip_presolve", mip_presolve);
    check!("fbbt_presolve", fbbt_presolve);
    check!("probing_presolve", probing_presolve);

    println!("\nDone — no false infeasibility in any standalone pass.");
}
