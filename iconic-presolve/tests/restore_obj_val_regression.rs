// Regression tests: every `restore_*` postsolve function for a variable-eliminating
// reduction pass (`eliminate_fixed_vars`, `eliminate_doubleton_eqs`,
// `eliminate_free_vars`, `merge_equality_rows`) must report the objective value of
// the RESTORED, original-dimension solution — not a verbatim copy of the smaller
// reduced problem's own `obj_val`. When a variable is eliminated via a
// constant/affine substitution, that substitution contributes a term to the true
// original-space objective (e.g. `½·P_jj·β² + q_j·β`) that is absent from the
// reduced problem's own objective. Copying `reduced.obj_val` verbatim silently
// drops that contribution.
//
// These pass-level `restore_*` functions are called directly here (bypassing
// `solve_presolved`, which happens to recompute `obj_val` from scratch at
// original dimensions right before returning and so does not exercise this bug).

use iconic_core::{Settings, Status};
use iconic_ipm::{solve_qp, QpProblem};
use iconic_linalg::DenseMatrix;
use iconic_presolve::reductions::{
    eliminate_doubleton_eqs, eliminate_fixed_vars, eliminate_free_vars, merge_equality_rows,
    restore_doubleton_eqs, restore_fixed_vars, restore_free_vars, restore_merged_rows,
};

/// `restore_fixed_vars`: `min ½x0² + ½x1² + 3x0  s.t.  x0 = 7`.
/// True optimum: x = [7, 0], obj = ½·49 + 3·7 = 45.5.
/// The reduced 1-D problem (`min ½x1²` after fixing x0) has its OWN optimum at
/// x1=0 with reduced-space obj_val = 0 — verbatim-copying that would silently
/// report 0.0 instead of 45.5.
#[test]
fn restore_fixed_vars_reports_original_space_objective() {
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
        q: vec![3.0, 0.0],
        a_eq: DenseMatrix::from_row_major(1, 2, vec![1.0, 0.0]),
        b_eq: vec![7.0],
        a_in: DenseMatrix::zeros(0, 2),
        b_in: vec![],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let (reduced, fix) = eliminate_fixed_vars(&prob).unwrap();
    assert_eq!(reduced.q.len(), 1, "x0 should be eliminated");

    let reduced_sol = solve_qp(&reduced, &Settings::<f64>::default());
    // Sanity: the reduced problem's own objective is nowhere near the true
    // original-space objective, so this test actually distinguishes the bug.
    assert!(
        (reduced_sol.obj_val - 45.5).abs() > 1.0,
        "reduced obj_val unexpectedly close to 45.5: {}",
        reduced_sol.obj_val
    );

    let restored = restore_fixed_vars(&prob, &fix, &reduced_sol);
    assert_eq!(restored.status, Status::Solved);
    assert!((restored.x[0] - 7.0).abs() < 1e-7, "x0={}", restored.x[0]);
    assert!((restored.x[1] - 0.0).abs() < 1e-7, "x1={}", restored.x[1]);
    assert!(
        (restored.obj_val - 45.5).abs() < 1e-6,
        "obj_val={} expected 45.5",
        restored.obj_val
    );
}

/// `restore_doubleton_eqs`: `min ½x0² + ½x1²  s.t.  x0 + x1 = 4`.
/// True optimum by symmetry: x = [2, 2], obj = ½·4 + ½·4 = 4.0.
/// The reduced 1-D problem's own obj_val is unrelated to 4.0 (verbatim-copying
/// was observed to even flip sign on this shape).
#[test]
fn restore_doubleton_eqs_reports_original_space_objective() {
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
        q: vec![0.0, 0.0],
        a_eq: DenseMatrix::from_row_major(1, 2, vec![1.0, 1.0]),
        b_eq: vec![4.0],
        a_in: DenseMatrix::zeros(0, 2),
        b_in: vec![],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let direct = solve_qp(&prob, &Settings::<f64>::default());
    assert!(
        (direct.obj_val - 4.0).abs() < 1e-6,
        "sanity: direct solve obj"
    );

    let (reduced, red) = eliminate_doubleton_eqs(&prob, 10.0).unwrap();
    assert_eq!(reduced.q.len(), 1, "one variable eliminated");
    let reduced_sol = solve_qp(&reduced, &Settings::<f64>::default());

    let restored = restore_doubleton_eqs(&prob, &red, &reduced_sol);
    assert_eq!(restored.status, Status::Solved);
    assert!((restored.x[0] - 2.0).abs() < 1e-6, "x0={}", restored.x[0]);
    assert!((restored.x[1] - 2.0).abs() < 1e-6, "x1={}", restored.x[1]);
    assert!(
        (restored.obj_val - 4.0).abs() < 1e-6,
        "obj_val={} expected 4.0 (direct solve: {})",
        restored.obj_val,
        direct.obj_val
    );
}

/// `restore_free_vars`: `min ½x0² + ½x1² + 5x0  s.t.  x0 + x1 = 4`.
/// x0 has diagonal-only P, no inequality involvement, and appears in exactly one
/// equality row, so it is the variable `eliminate_free_vars` substitutes out.
/// True optimum (closed form / verified against a direct solve of the original
/// problem): x0 = -0.5, x1 = 4.5, obj = 7.75.
#[test]
fn restore_free_vars_reports_original_space_objective() {
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
        q: vec![5.0, 0.0],
        a_eq: DenseMatrix::from_row_major(1, 2, vec![1.0, 1.0]),
        b_eq: vec![4.0],
        a_in: DenseMatrix::zeros(0, 2),
        b_in: vec![],
        a_eq_csr: None,
        a_in_csr: None,
    };
    let direct = solve_qp(&prob, &Settings::<f64>::default());
    assert!(
        (direct.obj_val - 7.75).abs() < 1e-6,
        "sanity: direct solve obj = {}",
        direct.obj_val
    );

    let (reduced, red) = eliminate_free_vars(&prob, 8, 8).unwrap();
    assert_eq!(
        red.elims.len(),
        1,
        "expected exactly one free-var elimination"
    );
    assert_eq!(reduced.q.len(), 1, "one variable eliminated");

    let reduced_sol = solve_qp(&reduced, &Settings::<f64>::default());
    let restored = restore_free_vars(&prob, &red, &reduced_sol);
    assert_eq!(restored.status, Status::Solved);
    assert!(
        (restored.x[0] - (-0.5)).abs() < 1e-6,
        "x0={}",
        restored.x[0]
    );
    assert!((restored.x[1] - 4.5).abs() < 1e-6, "x1={}", restored.x[1]);
    assert!(
        (restored.obj_val - 7.75).abs() < 1e-6,
        "obj_val={} expected 7.75 (direct solve: {})",
        restored.obj_val,
        direct.obj_val
    );
}

/// `restore_merged_rows`: a variable shared by exactly two equality rows (and
/// nowhere else) is merged out by cross-multiplying the two rows.
///
/// `min ½x0² + ½x2²  s.t.  x0 + x1 = 3, 2x1 + x2 = 8` (x1 has zero P entry and
/// zero q, so it is eligible for merging: no off-diagonal P coupling AND no
/// diagonal P term of its own). True optimum verified against a direct solve of
/// the original 3-variable problem.
#[test]
fn restore_merged_rows_reports_original_space_objective() {
    let prob = QpProblem {
        p: DenseMatrix::from_row_major(3, 3, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
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
    assert!(!red.merges.is_empty(), "expected the merge to fire on x1");
    let reduced_sol = solve_qp(&reduced, &Settings::<f64>::default());
    let restored = restore_merged_rows(&prob, &red, &reduced_sol);

    assert_eq!(restored.status, Status::Solved);
    for k in 0..3 {
        assert!(
            (restored.x[k] - direct.x[k]).abs() < 1e-6,
            "x{k}: merged={} direct={}",
            restored.x[k],
            direct.x[k]
        );
    }
    assert!(
        (restored.obj_val - direct.obj_val).abs() < 1e-6,
        "obj_val={} expected {} (direct solve)",
        restored.obj_val,
        direct.obj_val
    );
}
