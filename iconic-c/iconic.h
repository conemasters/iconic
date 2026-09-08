/* ICONIC — Native C API
 *
 * Usage:
 *   #include "iconic.h"
 *   // Link with -liconic_c
 *   // Sparse (CSC) call:
 *   int ret = iconic_solve(n, m, p_indptr, p_indices, p_data, q,
 *                         a_indptr, a_indices, a_data, b,
 *                         cone_specs, n_cones, presolve,
 *                         x_out, s_out, z_out, &obj_val, &iters, &status);
 *   // Dense (row-major) call:
 *   int ret = iconic_solve_dense(n, m, p_dense, q, a_dense, b,
 *                               cone_specs, n_cones, presolve,
 *                               x_out, s_out, z_out, &obj_val, &iters, &status);
 */

#ifndef ICONIC_H
#define ICONIC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes returned in *status_out. */
#define ICONIC_SOLVED              0
#define ICONIC_SOLVED_INACCURATE    1
#define ICONIC_PRIMAL_INFEASIBLE    2
#define ICONIC_DUAL_INFEASIBLE      3
#define ICONIC_MAX_ITERATIONS       4
#define ICONIC_TIME_LIMIT           5
#define ICONIC_NUMERICAL_ERROR      6
#define ICONIC_UNSOLVED            -1

/**
 * Solve the cone program with CSC (compressed sparse column) matrices.
 *
 *   minimize    ½ xᵀ P x + qᵀ x
 *   subject to  A x + s = b,  s ∈ K
 *
 * where K is a product of cones given by `cone_specs`.
 *
 * Sparse matrix format (CSC):
 *   - indptr: length ncols+1, column start positions
 *   - indices: length nnz, row indices
 *   - data:    length nnz, values
 *
 * P may be NULL for a zero Hessian.  q and b are dense arrays.
 *
 * `cone_specs` is an array of n_cones null-terminated strings of the form
 * "kind:dim" where kind ∈ {zero, nonneg, soc, psd, exp}.
 *
 * Output buffers (caller-allocated):
 *   x_out[n], s_out[m], z_out[m]
 *   obj_val_out[1], iters_out[1], status_out[1]
 *
 * Returns 0 on success, -1 on error.
 */
int iconic_solve(
    int n, int m,
    const int *p_indptr, const int *p_indices, const double *p_data,
    const double *q,
    const int *a_indptr, const int *a_indices, const double *a_data,
    const double *b,
    const char **cone_specs, int n_cones, int presolve,
    double *x_out, double *s_out, double *z_out,
    double *obj_val_out, int *iters_out, int *status_out
);

/**
 * Solve the cone program with dense row-major matrices (same mathematical
 * program as iconic_solve, but P and A are passed as dense f64 arrays).
 *
 *   P: n×n row-major (or NULL for zero).
 *   A: m×n row-major.
 *
 * Output buffers are the same as iconic_solve.
 */
int iconic_solve_dense(
    int n, int m,
    const double *p_dense, const double *q,
    const double *a_dense, const double *b,
    const char **cone_specs, int n_cones, int presolve,
    double *x_out, double *s_out, double *z_out,
    double *obj_val_out, int *iters_out, int *status_out
);

/**
 * Versioned settings struct for the *_with_settings entry points.
 *
 * Sentinel values mean "use the default": 0 for max_iters (the C-layer
 * default is 800) and the eps_* fields, -1 for the int32 switches, 0 for
 * blas_threads.  `version` must be 1; a caller compiled against a newer
 * header is rejected with -1 rather than silently mis-parsed.
 */
typedef struct {
    int32_t version;      /* must be 1 */
    int32_t max_iters;    /* 0 → default (800) */
    double  eps_abs;      /* 0.0 → default (1e-8) */
    double  eps_rel;      /* 0.0 → default (1e-8) */
    double  eps_gap;      /* 0.0 → default (1e-8) */
    int32_t presolve;     /* -1 → default (true) */
    int32_t hsd;          /* -1 → default (true) */
    int32_t sparse_kkt;   /* -1 → default (auto) */
    int32_t blas_threads; /* 0 → default (auto) */
} IconicSettings;

/**
 * Solve a cone program with CSC matrices and explicit settings.
 *
 * Identical to iconic_solve plus one trailing argument: `settings` is a
 * pointer to a IconicSettings struct, or NULL for the legacy behavior
 * (max_iters 800, presolve from the `presolve` argument).  With a struct
 * present, every non-sentinel field governs.
 */
int iconic_solve_with_settings(
    int n, int m,
    const int *p_indptr, const int *p_indices, const double *p_data,
    const double *q,
    const int *a_indptr, const int *a_indices, const double *a_data,
    const double *b,
    const char **cone_specs, int n_cones, int presolve,
    double *x_out, double *s_out, double *z_out,
    double *obj_val_out, int *iters_out, int *status_out,
    const IconicSettings *settings
);

/**
 * Solve a cone program with dense row-major matrices and explicit settings.
 * Identical to iconic_solve_dense plus the trailing `settings` argument
 * (NULL for the legacy behavior).
 */
int iconic_solve_dense_with_settings(
    int n, int m,
    const double *p_dense, const double *q,
    const double *a_dense, const double *b,
    const char **cone_specs, int n_cones, int presolve,
    double *x_out, double *s_out, double *z_out,
    double *obj_val_out, int *iters_out, int *status_out,
    const IconicSettings *settings
);

/**
 * Return the ICONIC version string.  Do NOT free.
 */
const char *iconic_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ICONIC_H */
