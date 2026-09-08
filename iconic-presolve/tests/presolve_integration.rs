use iconic_core::{Settings, Status};
use iconic_ipm::QpProblem;
use iconic_linalg::DenseMatrix;
use iconic_presolve::solve_presolved;

#[test]
fn regression_free_var_substitution() {
    // 2 vars, 2 eqs, no inequalities — free-vars should fire
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
        q: vec![0.0, 0.0],
        a_eq: DenseMatrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, -1.0]),
        b_eq: vec![5.0, 1.0],
        a_in: DenseMatrix::zeros(0, 2),
        b_in: vec![],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let sol = solve_presolved(&prob, &Settings::<f64>::default());
    assert_eq!(sol.status, Status::Solved);
    assert!((sol.x[0] - 1.0).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 2.0).abs() < 1e-5, "x1={}", sol.x[1]);
    // x0 + 2x1 = 5, 3x0 - x1 = 1 → x0=1, x1=2
}

#[test]
fn regression_row_merging() {
    // 3 vars, 2 eqs sharing x0 in exactly 2 rows
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0., 0., 0., 1.0, 0., 0., 0., 1.0]),
        q: vec![0.0; 3],
        a_eq: DenseMatrix::from_row_major(2, 3, vec![1.0, 1.0, 1.0, 1.0, -1.0, 2.0]),
        b_eq: vec![4.0, 0.0],
        a_in: DenseMatrix::zeros(0, 3),
        b_in: vec![],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let sol = solve_presolved(&prob, &Settings::<f64>::default());
    assert_eq!(sol.status, Status::Solved);
}

#[test]
fn regression_standard_qp_unchanged() {
    // Standard QP regression: ensure presolve doesn't break existing solves
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(2, 2, vec![3.0, 0.0, 0.0, 2.0]),
        q: vec![-2.0, -6.0],
        a_eq: DenseMatrix::zeros(0, 2),
        b_eq: vec![],
        a_in: DenseMatrix::from_row_major(3, 2, vec![1.0, 2.0, 1.0, 0.0, 0.0, 1.0]),
        b_in: vec![4.0, 3.0, 2.0],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let sol = solve_presolved(&prob, &Settings::<f64>::default());
    assert_eq!(sol.status, Status::Solved);
}

#[test]
fn regression_lp_via_presolve() {
    // LP (P=0) via presolve: min -x0-x1 s.t. x0+x1<=1, x0>=0, x1>=0
    // Optimum: x=[1,0] or [0,1], obj=-1
    let prob = QpProblem {
        p: DenseMatrix::zeros(2, 2),
        q: vec![-1.0, -1.0],
        a_eq: DenseMatrix::zeros(0, 2),
        b_eq: vec![],
        a_in: DenseMatrix::from_row_major(3, 2, vec![1.0, 1.0, -1.0, 0.0, 0.0, -1.0]),
        b_in: vec![1.0, 0.0, 0.0],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let sol = solve_presolved(&prob, &Settings::<f64>::default());
    assert_eq!(sol.status, Status::Solved);
    assert!(sol.obj_val < -0.9, "obj={}", sol.obj_val);
}
