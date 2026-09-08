//! Dump the exact problem instances `suite::build_suite` and `mip::build_mip_suite`
//! define, as JSON, so `scripts/bench_all_solvers.py` can drive every installed CVXPY
//! solver over the *same* instances instead of re-implementing its own generators. Rust
//! stays the single source of truth for problem data; Python only consumes it.
//!
//! Matrices are dense row-major `f64` arrays (via `DenseMatrix::data`, already row-major),
//! so `numpy.array(...).reshape(nrows, ncols)` on the Python side reconstructs them exactly.

use crate::mip::build_mip_suite;
use crate::suite::{build_suite, Spec};
use iconic_core::Cone;
use iconic_ipm::QpProblem;
use iconic_mip::{MipProblem, VarType};

fn f64_array_json(xs: &[f64]) -> String {
    let mut s = String::from("[");
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}

fn str_array_json(xs: &[&str]) -> String {
    let mut s = String::from("[");
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        s.push_str(x);
        s.push('"');
    }
    s.push(']');
    s
}

fn qp_to_json(category: &str, name: &str, prob: &QpProblem<f64>) -> String {
    format!(
        "{{\"category\":\"{category}\",\"name\":\"{name}\",\"n\":{},\"m_eq\":{},\"m_in\":{},\
         \"P\":{},\"q\":{},\"A_eq\":{},\"b_eq\":{},\"A_in\":{},\"b_in\":{}}}",
        prob.q.len(),
        prob.b_eq.len(),
        prob.b_in.len(),
        f64_array_json(&prob.p.data),
        f64_array_json(&prob.q),
        f64_array_json(&prob.a_eq.data),
        f64_array_json(&prob.b_eq),
        f64_array_json(&prob.a_in.data),
        f64_array_json(&prob.b_in),
    )
}

/// Dump every `Spec::Qp` case in `build_suite` with `n <= max_n` as one JSON array. Conic
/// (SOCP/SDP) cases are skipped — the cross-solver comparison this feeds is QP/LP-only.
/// `max_n` keeps the export usable for CVXPY solvers far slower than ICONIC (ECOS, CVXOPT):
/// `build_suite`'s largest QPs (n=800) are for ICONIC's own native suite only.
pub fn export_qp_suite(max_n: usize) -> String {
    let mut objs = Vec::new();
    for case in build_suite() {
        if let Spec::Qp(prob) = &case.spec {
            if prob.q.len() <= max_n {
                objs.push(qp_to_json(case.category, &case.name, prob));
            }
        }
    }
    format!("[{}]", objs.join(","))
}

fn cone_kind_dim(c: &Cone) -> (&'static str, usize) {
    match c {
        Cone::Zero(d) => ("zero", *d),
        Cone::NonNegative(d) => ("nonneg", *d),
        Cone::SecondOrder(d) => ("soc", *d),
        Cone::PsdTriangle(d) => ("psd", *d),
        Cone::Exponential => ("exp", 3),
        Cone::Power(_) => ("power", 3),
        Cone::GenPower(_, d) => ("genpower", *d),
    }
}

fn var_type_str(v: VarType) -> &'static str {
    match v {
        VarType::Continuous => "continuous",
        VarType::Integer => "integer",
        VarType::Binary => "binary",
    }
}

/// Serialize one MIP instance, for callers outside this module that need the same
/// schema -- e.g. dumping the problem after each presolve pass to localise which pass
/// changed its optimal value.
pub fn mip_to_json_pub(name: &str, prob: &MipProblem<f64>) -> String {
    mip_to_json(name, prob)
}

fn mip_to_json(name: &str, prob: &MipProblem<f64>) -> String {
    let cones: Vec<String> = prob
        .cones
        .iter()
        .map(|c| {
            let (kind, dim) = cone_kind_dim(c);
            format!("[\"{kind}\",{dim}]")
        })
        .collect();
    let var_types: Vec<&str> = prob.var_types.iter().map(|&v| var_type_str(v)).collect();
    format!(
        "{{\"category\":\"mip\",\"name\":\"{name}\",\"n\":{},\"m\":{},\
         \"P\":{},\"q\":{},\"A\":{},\"b\":{},\"cones\":[{}],\"var_types\":{},\"lb\":{},\"ub\":{}}}",
        prob.q.len(),
        prob.b.len(),
        f64_array_json(&prob.p.data),
        f64_array_json(&prob.q),
        f64_array_json(&prob.a.data),
        f64_array_json(&prob.b),
        cones.join(","),
        str_array_json(&var_types),
        f64_array_json(&prob.lb),
        f64_array_json(&prob.ub),
    )
}

/// Dump every `build_mip_suite` instance with `n <= max_n` as one JSON array — the same
/// instances `run --mip` times ICONIC's own branch-and-bound against, so a Python-side
/// multi-solver comparison lands on identical (category, name) keys and can be merged
/// directly with the Rust-produced ICONIC rows.
pub fn export_mip_suite(max_n: usize) -> String {
    let mut objs = Vec::new();
    for (name, prob, _) in build_mip_suite() {
        if prob.q.len() <= max_n {
            objs.push(mip_to_json(&name, &prob));
        }
    }
    format!("[{}]", objs.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qp_export_is_well_formed_and_size_capped() {
        let json = export_qp_suite(50);
        assert!(json.starts_with('[') && json.ends_with(']'));
        assert!(json.contains("\"qp_random\""));
        // Every exported case must respect the cap (n<=50 excludes e.g. "n100_m80").
        assert!(!json.contains("\"n100_m80\""));
        assert!(json.contains("\"n10_m10\""));
        // Balanced brackets: a cheap proxy for "no truncated/garbled object".
        assert_eq!(json.matches('{').count(), json.matches('}').count());
    }

    #[test]
    fn mip_export_is_well_formed_and_size_capped() {
        let json = export_mip_suite(15);
        assert!(json.starts_with('[') && json.ends_with(']'));
        assert!(json.contains("\"knapsack_n=10\""));
        assert!(!json.contains("\"knapsack_n=20\""));
        assert!(json.contains("\"cones\":[[\"nonneg\","));
        assert_eq!(json.matches('{').count(), json.matches('}').count());
    }
}
