use iconic_core::Settings;
use iconic_ipm::{solve_qp, QpProblem};
use iconic_linalg::DenseMatrix;
use iconic_presolve::solve_presolved;

#[test]
fn solve_presolved_end_to_end_free_var_then_subst_none() {
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(
            6,
            6,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ),
        q: vec![-1.0, -2.0, 0.5, -0.5, 0.1, 0.2],
        a_eq: DenseMatrix::from_row_major(
            2,
            6,
            vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ),
        b_eq: vec![4.0, 5.0],
        a_in: DenseMatrix::zeros(0, 6),
        b_in: vec![],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let direct = solve_qp(&prob, &Settings::<f64>::default());
    println!(
        "direct: status={:?} x={:?} obj={}",
        direct.status, direct.x, direct.obj_val
    );

    let via_presolve = solve_presolved(&prob, &Settings::<f64>::default());
    println!(
        "presolved: status={:?} x={:?} obj={}",
        via_presolve.status, via_presolve.x, via_presolve.obj_val
    );

    assert_eq!(via_presolve.x.len(), 6, "restored x must have original n=6");
    for j in 0..6 {
        assert!(
            (via_presolve.x[j] - direct.x[j]).abs() < 1e-6,
            "x{j}: presolved={} direct={}",
            via_presolve.x[j],
            direct.x[j]
        );
    }
}
