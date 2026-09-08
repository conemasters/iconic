//! Does symmetry breaking fire on the suite's symmetric instances, and by how much?
//!
//! `symmetry_detect_and_break` and `binpack_symmetry_break` both run in presolve, but a
//! detector that silently matches nothing is indistinguishable from one that is working.
//! This reports the rows each adds per instance.
use iconic_bench::mip::build_mip_suite;
use iconic_mip::{binpack_symmetry_break, symmetry_detect_and_break};

fn main() {
    let filters: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "{:<24}{:>8}{:>10}{:>10}{:>10}",
        "instance", "rows", "+colrow", "+binpack", "fixed"
    );
    for (name, prob, _) in build_mip_suite() {
        if !filters.is_empty() && !filters.iter().any(|f| name.contains(f.as_str())) {
            continue;
        }
        let m0 = prob.b.len();
        let mut p = prob.clone();
        let _ = symmetry_detect_and_break(&mut p);
        let m1 = p.b.len();
        let _ = binpack_symmetry_break(&mut p);
        let m2 = p.b.len();
        let fixed =
            p.ub.iter()
                .zip(prob.ub.iter())
                .filter(|(a, b)| **a == 0.0 && **b != 0.0)
                .count();
        println!("{name:<24}{m0:>8}{:>10}{:>10}{fixed:>10}", m1 - m0, m2 - m1);
    }
}
