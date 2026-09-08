//! Dump a MIP instance after each presolve pass, cumulatively.
//!
//! A pass that changes the problem's optimal *value* has discarded solutions it should
//! have kept. Comparing the optimum before and after each pass localises which one, and
//! comparing values (rather than a specific point) stays correct for passes that are
//! allowed to discard particular optima -- symmetry breaking keeps one representative of
//! each orbit, so the point moves but the value must not.
//!
//! usage: presolve_pass_bisect <instance-name>   → writes stage JSON to stdout
use iconic_bench::export::mip_to_json_pub;
use iconic_bench::mip::build_mip_suite;

fn main() {
    let target = std::env::args().nth(1).expect("instance name");
    let (_, prob, _) = build_mip_suite()
        .into_iter()
        .find(|(n, _, _)| *n == target)
        .expect("unknown instance");

    let mut p = prob.clone();
    let mut stages: Vec<String> = vec![mip_to_json_pub("stage00_original", &p)];

    macro_rules! stage {
        ($label:literal, $call:expr) => {{
            let _ = $call;
            stages.push(mip_to_json_pub($label, &p));
        }};
    }

    stage!(
        "stage01_aggregate_parallel_rows",
        iconic_mip::aggregate_parallel_rows(&mut p)
    );
    stage!(
        "stage02_substitute_implied_fixed",
        iconic_mip::substitute_implied_fixed(&mut p)
    );
    stage!(
        "stage03_embedded_knapsack",
        iconic_mip::embedded_knapsack_presolve(&mut p)
    );
    stage!(
        "stage04_cancel_nonzeros",
        iconic_mip::cancel_mip_nonzeros(&mut p)
    );
    stage!(
        "stage05_redundancy_detect",
        iconic_mip::mip_redundancy_detect(&mut p)
    );
    stage!("stage06_clique", iconic_mip::clique_presolve(&mut p));
    stage!(
        "stage07_binconf_clique",
        iconic_mip::binconf_clique_strengthen(&mut p)
    );
    stage!("stage08_cover", iconic_mip::cover_presolve(&mut p));
    stage!(
        "stage09_setpart_singleton",
        iconic_mip::setpart_singleton_substitution(&mut p)
    );
    stage!(
        "stage10_multirow_bound",
        iconic_mip::multirow_bound_strengthen(&mut p)
    );
    stage!(
        "stage11_symmetry",
        iconic_mip::symmetry_detect_and_break(&mut p)
    );
    stage!(
        "stage12_binpack_symmetry",
        iconic_mip::binpack_symmetry_break(&mut p)
    );
    stage!(
        "stage13_perspective",
        iconic_mip::perspective_reformulate(&mut p)
    );
    stage!("stage14_tsp_tw", iconic_mip::tsp_tw_presolve(&mut p));
    stage!(
        "stage15_jobshop_edge_finding",
        iconic_mip::jobshop_edge_finding(&mut p)
    );
    stage!("stage16_mip_presolve", iconic_mip::mip_presolve(&mut p));
    stage!("stage17_fbbt", iconic_mip::fbbt_presolve(&mut p));
    stage!("stage18_probing", iconic_mip::probing_presolve(&mut p));

    // `solve_mip` does not run these once: it iterates mip_presolve / fbbt / probing /
    // cancel / redundancy five times to a fixed point. A pass can be individually
    // value-preserving and still lose the optimum once compounded with the others.
    for pass in 0..5 {
        let _ = iconic_mip::mip_presolve(&mut p);
        stages.push(mip_to_json_pub(&format!("loop{pass}a_mip_presolve"), &p));
        let _ = iconic_mip::fbbt_presolve(&mut p);
        stages.push(mip_to_json_pub(&format!("loop{pass}b_fbbt"), &p));
        let _ = iconic_mip::probing_presolve(&mut p);
        stages.push(mip_to_json_pub(&format!("loop{pass}c_probing"), &p));
        let _ = iconic_mip::cancel_mip_nonzeros(&mut p);
        stages.push(mip_to_json_pub(&format!("loop{pass}d_cancel"), &p));
        let _ = iconic_mip::mip_redundancy_detect(&mut p);
        stages.push(mip_to_json_pub(&format!("loop{pass}e_redundancy"), &p));
    }

    println!("[{}]", stages.join(","));
}
