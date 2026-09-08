//! Regression test: `dpotrf_ul(.., b'U')` must honor the triangle on every
//! BLAS backend. The faer fallback historically ignored `uplo` and factored
//! the row-major UPPER triangle of the condensed Hessian — for a QP whose
//! gram fills only row-major lower that silently dropped the Aᵀ(Z/S)A
//! contribution, producing a valid-looking PD factor of the wrong matrix
//! (`SolvedInaccurate`, wrong objective) whenever system BLAS was absent.
use iconic_core::Settings;
use iconic_ipm::{QpProblem, solve_qp};
use iconic_linalg::dense::DenseMatrix;

/// Dense QP: n ≥ 48 with general (multi-nonzeros) inequality rows, so the
/// condensed path assembles H via dsyrk (row-major lower gram) and factors
/// through `dpotrf_ul(n, .., b'U')`.
fn build(n: usize, m_in: usize, seed: u64) -> QpProblem<f64> {
    let mut s = seed;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / ((1u64 << 31) as f64) * 2.0 - 1.0
    };
    let mut p = DenseMatrix::zeros(n, n);
    for i in 0..n { for j in i..n { let v = rnd() * 0.05; p.set(i, j, v); p.set(j, i, v); } }
    for i in 0..n { p.set(i, i, p.get(i, i) + 2.0); }
    let q: Vec<f64> = (0..n).map(|_| rnd()).collect();
    let x0: Vec<f64> = (0..n).map(|_| 0.5 * rnd()).collect();
    let big = 10.0;
    let rows = m_in + 2 * n;
    let mut a_in = DenseMatrix::zeros(rows, n);
    let mut b_in = vec![0.0; rows];
    for r in 0..m_in { for j in 0..n { a_in.set(r, j, rnd()); } }
    let ax0 = a_in.matvec(&x0);
    for r in 0..m_in { b_in[r] = ax0[r] + 0.75; }
    for j in 0..n {
        a_in.set(m_in + j, j, 1.0);
        b_in[m_in + j] = big;
        a_in.set(m_in + n + j, j, -1.0);
        b_in[m_in + n + j] = big;
    }
    QpProblem::inequality_only(p, q, a_in, b_in)
}

#[test]
fn blas_off_matches_blas_on_dense_condensed_qp() {
    let prob = build(100, 60, 42);

    let (on_status, on_iters, on_obj) = {
        let sol = solve_qp(&prob, &Settings::<f64>::default());
        (format!("{:?}", sol.status), sol.iters, sol.obj_val)
    };
    // The override is thread-local: run the disabled solve in isolation.
    let (off_status, off_iters, off_obj) = std::thread::spawn(move || {
        iconic_linalg::blas::set_blas_enabled(false);
        let sol = solve_qp(&prob, &Settings::<f64>::default());
        (format!("{:?}", sol.status), sol.iters, sol.obj_val)
    })
    .join()
    .expect("worker");

    assert_eq!(on_status, "Solved", "BLAS-on baseline should solve");
    assert_eq!(
        off_status, on_status,
        "BLAS-off status diverged ({off_status} iters={off_iters} obj={off_obj} vs {on_status} iters={on_iters} obj={on_obj})"
    );
    assert!(
        (on_obj - off_obj).abs() < 1e-6,
        "BLAS-off objective diverged: {off_obj} vs {on_obj}"
    );
}
