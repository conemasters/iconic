use iconic_core::{Settings, Status};
use iconic_ipm::{solve_qp, QpProblem};
use iconic_linalg::DenseMatrix;
use iconic_presolve::solve_presolved;

/// Regression for the `merge_equality_rows` objective-fold bug: the
/// candidacy gate `col_is_diag_p` only forbids P cross-coupling between the
/// merge candidate and *other* variables — it does not by itself guarantee
/// the candidate's own `P[j,j]`/`q[j]` are zero. A version of
/// `merge_equality_rows` that let such a column through without folding (or
/// excluding) its diagonal quadratic/linear cost would silently return a
/// feasible-but-suboptimal point.
///
/// This drives the bug through the real, unmodified `solve_presolved` entry
/// point (not an isolated call to `merge_equality_rows`), so it also proves
/// the pipeline gate (`merge_equality_rows` only runs once `n >= 5`, see
/// `lib.rs`) is actually exercised, unlike `regression_row_merging` in
/// `presolve_integration.rs` (n=3, never reaches that gate).
///
/// Construction: n=30, P = I (so every variable, including any merge
/// candidate, carries a real quadratic cost), two 15-nonzero equality rows
/// sharing x0 exclusively. The row width (14 other nonzeros) exceeds
/// `eliminate_free_vars`'s `max_fill=12` gate, so free-variable substitution
/// skips x0 despite it appearing in exactly 2 rows, letting x0 survive as a
/// `merge_equality_rows` candidate.
#[test]
fn merge_equality_rows_bug_does_not_corrupt_full_pipeline_solve() {
    let n = 30;
    let mut p = DenseMatrix::zeros(n, n);
    for i in 0..n {
        p.set(i, i, 1.0);
    }
    let q = vec![0.0; n];
    let mut a_eq = DenseMatrix::zeros(2, n);
    for j in 0..15 {
        a_eq.set(0, j, 1.0 + j as f64 * 0.01);
    }
    a_eq.set(1, 0, 2.0);
    for (idx, j) in (15..29).enumerate() {
        a_eq.set(1, j, 0.5 + idx as f64 * 0.02);
    }
    let b_eq = vec![40.0, 30.0];
    let prob = QpProblem {
        p,
        q,
        a_eq,
        b_eq,
        a_in: DenseMatrix::zeros(0, n),
        b_in: vec![],
        a_eq_csr: None,
        a_in_csr: None,
    };

    let settings = Settings::<f64>::default();
    let direct = solve_qp(&prob, &settings);
    assert_eq!(direct.status, Status::Solved);

    let presolved = solve_presolved(&prob, &settings);
    assert_eq!(presolved.status, Status::Solved);

    // The buggy variant (an unguarded merge that drops P_jj/q_j on the
    // merged variable instead of folding or excluding it) produced an
    // objective error of ~82 and primal errors up to ~2.6 on this instance;
    // a correct implementation (whether by folding the substitution into
    // P/q, or by excluding P_jj != 0 columns from candidacy) matches the
    // direct solve tightly.
    assert!(
        (presolved.obj_val - direct.obj_val).abs() < 1e-6,
        "obj_val: presolved={} direct={} diff={:.3e}",
        presolved.obj_val,
        direct.obj_val,
        (presolved.obj_val - direct.obj_val).abs()
    );
    for k in 0..n {
        assert!(
            (presolved.x[k] - direct.x[k]).abs() < 1e-6,
            "x{k}: presolved={} direct={} diff={:.3e}",
            presolved.x[k],
            direct.x[k],
            (presolved.x[k] - direct.x[k]).abs()
        );
    }
}
