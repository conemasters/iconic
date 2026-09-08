use iconic_core::Settings;
use iconic_ipm::QpProblem;
use iconic_linalg::DenseMatrix;
use std::time::Instant;

fn rng_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn make_problem(n: usize, mi: usize, seed: u64) -> QpProblem<f64> {
    let mut state = seed;
    let mut l = vec![0.0f64; n * n];
    for v in l.iter_mut() {
        *v = rng_next(&mut state) * 0.1;
    }
    let mut p = DenseMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += l[i * n + k] * l[j * n + k];
            }
            p.set(i, j, acc);
        }
        p.set(i, i, p.get(i, i) + 0.01);
    }
    let q: Vec<f64> = (0..n).map(|_| rng_next(&mut state)).collect();
    let mut a_in = DenseMatrix::<f64>::zeros(mi, n);
    for i in 0..mi {
        for j in 0..n {
            a_in.set(i, j, rng_next(&mut state));
        }
    }
    let b_in: Vec<f64> = (0..mi)
        .map(|_| rng_next(&mut state).abs() * 5.0 + 1.0)
        .collect();
    QpProblem {
        p,
        q,
        a_eq: DenseMatrix::<f64>::zeros(0, n),
        b_eq: vec![],
        a_in,
        b_in,
        a_eq_csr: None,
        a_in_csr: None,
    }
}

fn main() {
    let n = 200usize;
    let mi = 150usize;
    let prob = make_problem(n, mi, 42 + n as u64);
    let settings = Settings::<f64>::default();
    println!(
        "presolve={} sparse_kkt={}",
        settings.presolve, settings.sparse_kkt
    );
    let t0 = Instant::now();
    let sol = iconic_presolve::solve_presolved(&prob, &settings);
    println!(
        "solve_presolved: {:.2}ms iters={} status={:?} obj={:.4}",
        t0.elapsed().as_secs_f64() * 1000.0,
        sol.iters,
        sol.status,
        sol.obj_val
    );
}
