use iconic_core::Settings;
use iconic_ipm::{solve_qp, QpProblem};
use iconic_linalg::DenseMatrix;
use iconic_presolve::reductions::{merge_equality_rows, restore_merged_rows};

// merge_equality_rows eliminates a variable x_j that appears in exactly two
// equality rows (and nowhere else), replacing both rows with one merged row
// that cancels x_j. col_is_diag_p only checks OFF-diagonal P entries are
// zero for column j; it does not by itself guarantee P[j,j] == 0. Unlike
// eliminate_doubleton_eqs / eliminate_free_vars (which fold the eliminated
// variable's quadratic contribution into the surviving columns via a TxPxT
// congruence), merge_equality_rows's row-merge construction has no general
// way to fold a nonzero diagonal P[j,j] into the aggregate row it produces,
// so the candidacy gate additionally excludes columns with P[j,j] != 0 (see
// the `pjj.abs() > ...` check in merge_equality_rows) -- the merge simply
// does not fire on such columns, so `restore_merged_rows` is never asked to
// recover one, and no wrong optimum can be produced.
//
// 3 vars, 2 eq rows sharing x1 (col 1) exclusively, P diagonal with P11=1.0
// (a genuine QP term on the merged variable, not just an LP/free-var case).
#[test]
fn merge_equality_rows_skips_nonzero_diagonal_p_on_merged_var() {
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        q: vec![0.0, 0.0, 0.0],
        a_eq: DenseMatrix::from_row_major(
            2,
            3,
            vec![
                1.0, 1.0, 0.0, //  x0 + x1      = 3
                0.0, 2.0, 1.0, //       2x1 + x2 = 8
            ],
        ),
        b_eq: vec![3.0, 8.0],
        a_in: DenseMatrix::zeros(0, 3),
        b_in: vec![],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let direct = solve_qp(&prob, &Settings::<f64>::default());

    let (reduced, red) = merge_equality_rows(&prob, 8).expect("merge should not error");
    assert!(
        red.merges.is_empty(),
        "column with nonzero diagonal P (x1) must not be merged out"
    );
    let reduced_sol = solve_qp(&reduced, &Settings::<f64>::default());
    let restored = restore_merged_rows(&prob, &red, &reduced_sol);

    // Original-space equality feasibility must hold regardless.
    for r in 0..2 {
        let mut lhs = 0.0;
        for k in 0..3 {
            lhs += prob.a_eq.get(r, k) * restored.x[k];
        }
        assert!(
            (lhs - prob.b_eq[r]).abs() < 1e-6,
            "row {r} infeasible: {lhs} vs {}",
            prob.b_eq[r]
        );
    }

    // The recovered primal must match a direct solve of the original QP.
    for k in 0..3 {
        assert!(
            (restored.x[k] - direct.x[k]).abs() < 1e-6,
            "x{k}: merged={} direct={}",
            restored.x[k],
            direct.x[k]
        );
    }
}
