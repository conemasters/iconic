use iconic_bench::lp_transport;
use iconic_core::Settings;
use iconic_ipm::solve_qp;
fn main() {
    let prob = lp_transport(60, 80, 61);
    let settings = Settings::<f64>::default();
    // Decisive test: c-scaled problem with the ORIGINAL-UNIT term vs identity.
    use iconic_ipm::{solve_qp_with_termination, TermScale};
    let n = prob.q.len();
    let qn = prob.q.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let pmean = (0..n)
        .map(|j| {
            (0..n)
                .map(|i| prob.p.get(i, j).abs())
                .fold(0.0f64, f64::max)
        })
        .sum::<f64>()
        / n as f64;
    let c = 1.0 / pmean.max(qn);
    let mut cscaled = prob.clone();
    for i in 0..n {
        for j in 0..n {
            cscaled.p.set(i, j, cscaled.p.get(i, j) * c);
        }
    }
    for v in cscaled.q.iter_mut() {
        *v *= c;
    }
    let term = TermScale {
        dual: vec![1.0 / c; n],
        prim_eq: vec![],
        prim_in: vec![1.0; prob.b_in.len()],
        comp: 1.0 / c,
    };
    let t = std::time::Instant::now();
    let sol = solve_qp_with_termination(&cscaled, &settings, &term);
    println!(
        "c-scaled + orig-unit term: {:?} {} iters {:.3}s",
        sol.status,
        sol.iters,
        t.elapsed().as_secs_f64()
    );
    let t0 = std::time::Instant::now();
    let sol_raw = solve_qp(&prob, &settings);
    println!(
        "raw original           : {:?} {} iters {:.3}s",
        sol_raw.status,
        sol_raw.iters,
        t0.elapsed().as_secs_f64()
    );
    let t1 = std::time::Instant::now();
    let sol_pre = iconic_presolve::solve_presolved(&prob, &settings);
    println!(
        "presolved (default)    : {:?} {} iters {:.3}s kkt={:.1e}",
        sol_pre.status,
        sol_pre.iters,
        t1.elapsed().as_secs_f64(),
        iconic_bench::kkt_residual(&prob, &sol_pre)
    );
    // Per-pass timing of the reduction chain.
    let mut cur = prob.clone();
    for (name, f) in [
        (
            "fold_negated_pairs",
            Box::new(|p: &iconic_ipm::QpProblem<f64>| {
                reductions::fold_negated_pairs(p, &reductions::SparseAIn::build(p)).map(|(r, _)| r)
            })
                as Box<
                    dyn Fn(
                        &iconic_ipm::QpProblem<f64>,
                    )
                        -> Result<iconic_ipm::QpProblem<f64>, iconic_core::Status>,
                >,
        ),
        (
            "eliminate_fixed_vars",
            Box::new(|p: &iconic_ipm::QpProblem<f64>| {
                reductions::eliminate_fixed_vars(p).map(|(r, _)| r)
            }),
        ),
        (
            "eliminate_doubleton_eqs",
            Box::new(|p: &iconic_ipm::QpProblem<f64>| {
                reductions::eliminate_doubleton_eqs(p, settings.fill_budget).map(|(r, _)| r)
            }),
        ),
        (
            "eliminate_empty_cols",
            Box::new(|p: &iconic_ipm::QpProblem<f64>| {
                reductions::eliminate_empty_cols(p, &reductions::SparseAIn::build(p))
                    .map(|(r, _)| r)
            }),
        ),
        (
            "reduce_rows",
            Box::new(|p: &iconic_ipm::QpProblem<f64>| {
                reductions::reduce_rows(p, &reductions::SparseAIn::build(p)).map(|(r, _)| r)
            }),
        ),
        (
            "remove_redundant_ineqs",
            Box::new(|p: &iconic_ipm::QpProblem<f64>| {
                reductions::remove_redundant_ineqs(p, &reductions::SparseAIn::build(p))
                    .map(|(r, _)| r)
            }),
        ),
        (
            "remove_dependent_eq_rows",
            Box::new(|p: &iconic_ipm::QpProblem<f64>| {
                reductions::remove_dependent_eq_rows(p).map(|(r, _)| r)
            }),
        ),
    ] {
        let t = std::time::Instant::now();
        match f(&cur) {
            Ok(r) => {
                cur = r;
                println!("  pass {name:28}: {:.3}s", t.elapsed().as_secs_f64());
            }
            Err(_) => println!("  pass {name}: Err"),
        }
    }

    // Time one SparseAIn build + one fold directly.
    let t_b = std::time::Instant::now();
    let sp1 = reductions::SparseAIn::build(&prob);
    println!("SparseAIn build: {:.3}s", t_b.elapsed().as_secs_f64());
    let t_f = std::time::Instant::now();
    let _ = reductions::fold_negated_pairs(&prob, &sp1).unwrap();
    println!("fold (sparse views): {:.3}s", t_f.elapsed().as_secs_f64());

    // Replicate the reduction chain pass by pass, solving raw after each.
    use iconic_presolve::reductions;
    let mut cur = prob.clone();
    let mut label = String::new();
    for (name, res) in [
        (
            "fold_negated_pairs",
            reductions::fold_negated_pairs(
                &cur,
                &iconic_presolve::reductions::SparseAIn::build(&cur),
            )
            .map(|(r, _)| r),
        ),
        (
            "eliminate_fixed_vars",
            reductions::eliminate_fixed_vars(&cur).map(|(r, _)| r),
        ),
        (
            "eliminate_doubleton_eqs",
            reductions::eliminate_doubleton_eqs(&cur, settings.fill_budget).map(|(r, _)| r),
        ),
        (
            "eliminate_empty_cols",
            reductions::eliminate_empty_cols(
                &cur,
                &iconic_presolve::reductions::SparseAIn::build(&cur),
            )
            .map(|(r, _)| r),
        ),
        (
            "reduce_rows",
            reductions::reduce_rows(&cur, &iconic_presolve::reductions::SparseAIn::build(&cur))
                .map(|(r, _)| r),
        ),
        (
            "remove_redundant_ineqs",
            reductions::remove_redundant_ineqs(
                &cur,
                &iconic_presolve::reductions::SparseAIn::build(&cur),
            )
            .map(|(r, _)| r),
        ),
        (
            "remove_dependent_eq_rows",
            reductions::remove_dependent_eq_rows(&cur).map(|(r, _)| r),
        ),
    ] {
        match res {
            Ok(r) => {
                cur = r;
                label = name.to_string();
            }
            Err(st) => {
                println!("{name}: Err({st:?})");
                break;
            }
        }
    }
    println!("chain ({label}): n={} mi={}", cur.q.len(), cur.b_in.len());
    let t = std::time::Instant::now();
    let sol = solve_qp(&cur, &settings);
    println!(
        "chain-reduced raw: {:?} {} iters {:.3}s",
        sol.status,
        sol.iters,
        t.elapsed().as_secs_f64()
    );
}
