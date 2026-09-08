//! Native C ABI for the ICONIC convex optimizer.
//!
//! Exposes `extern "C"` functions so CVXPY (or any C-compatible language) can
//! call ICONIC directly through ctypes, avoiding the PyO3 conversion layer.
//! All inputs are raw pointers; outputs go into caller-allocated buffers.
//!
//! The main entry points are [`iconic_solve`] / [`iconic_solve_dense`], plus the
//! settings-carrying variants [`iconic_solve_with_settings`] /
//! [`iconic_solve_dense_with_settings`] which take a trailing `IconicSettings`
//! struct (`NULL` = the legacy behavior of the plain entries).

use iconic_api::{solve as api_solve, ConeProgram, SolveError};
use iconic_core::{Cone, Settings, Status};
use iconic_linalg::DenseMatrix;
use iconic_mip::{solve_mip, MipProblem, MipSettings, MipStatus, VarType};
use std::ffi::CStr;
use std::os::raw::c_char;

/// What the MIP entry points parse out of their matrix/cone arguments.
type MipInputs = (usize, usize, DenseMatrix<f64>, DenseMatrix<f64>, Vec<Cone>);

/// Settings struct passed to the `*_with_settings` entry points.
///
/// Layout must match `iconic.h`'s `IconicSettings` and the backend's ctypes
/// mirror. Sentinel values mean "use the default": `0` for `max_iters` (the
/// C-layer default is 800, the pre-settings behavior) and the `eps_*` fields,
/// `-1` for the int32 switches (`presolve`/`hsd`/`sparse_kkt`), `0` for
/// `blas_threads`. A struct `version` other than 1 is rejected (-1) rather
/// than silently mis-parsed — a caller compiled against a newer header must
/// not get old behavior under a new layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IconicSettings {
    /// Struct version — must be 1.
    pub version: i32,
    /// Max IPM iterations (`0` → 800).
    pub max_iters: i32,
    /// Absolute residual/gap tolerance (`0.0` → 1e-8).
    pub eps_abs: f64,
    /// Relative residual tolerance (`0.0` → 1e-8).
    pub eps_rel: f64,
    /// Duality-gap tolerance (`0.0` → 1e-8).
    pub eps_gap: f64,
    /// Presolve on/off (`-1` → default true).
    pub presolve: i32,
    /// HSD embedding on/off (`-1` → default true).
    pub hsd: i32,
    /// Force the sparse KKT path (`-1` → default auto).
    pub sparse_kkt: i32,
    /// Platform-BLAS worker-thread cap (`0` → auto).
    pub blas_threads: i32,
}

/// Build solver settings from an optional C struct.
///
/// `None` reproduces the legacy entry-point behavior exactly: the `presolve`
/// argument plus `max_iters: 800` on top of `Settings::default()`. With a
/// struct present, every field the struct carries governs (sentinels fall
/// back to the documented defaults). Returns `Err` for an unsupported struct
/// version or a negative `max_iters`.
unsafe fn build_settings(presolve: i32, cs: Option<&IconicSettings>) -> Result<Settings<f64>, ()> {
    Ok(match cs {
        None => Settings {
            presolve: presolve != 0,
            max_iters: 800,
            ..Settings::default()
        },
        Some(cs) => {
            if cs.version != 1 || cs.max_iters < 0 {
                return Err(());
            }
            Settings {
                presolve: if cs.presolve < 0 {
                    true
                } else {
                    cs.presolve != 0
                },
                max_iters: if cs.max_iters == 0 {
                    800
                } else {
                    cs.max_iters as usize
                },
                eps_abs: if cs.eps_abs == 0.0 { 1e-8 } else { cs.eps_abs },
                eps_rel: if cs.eps_rel == 0.0 { 1e-8 } else { cs.eps_rel },
                eps_gap: if cs.eps_gap == 0.0 { 1e-8 } else { cs.eps_gap },
                sparse_kkt: if cs.sparse_kkt < 0 {
                    false
                } else {
                    cs.sparse_kkt != 0
                },
                blas_threads: if cs.blas_threads <= 0 {
                    None
                } else {
                    Some(cs.blas_threads as usize)
                },
                hsd: if cs.hsd < 0 { true } else { cs.hsd != 0 },
                ..Settings::default()
            }
        }
    })
}

/// Solve a parsed program and write the output buffers. Shared by all four
/// cone-program entry points (CSC and dense, with and without settings).
#[allow(clippy::too_many_arguments)]
unsafe fn run_solve(
    prog: &ConeProgram<f64>,
    settings: &Settings<f64>,
    x_out: *mut f64,
    s_out: *mut f64,
    z_out: *mut f64,
    obj_val_out: *mut f64,
    iters_out: *mut i32,
    status_out: *mut i32,
) -> Result<(), SolveError> {
    match api_solve(prog, settings) {
        Ok(sol) => {
            let n = prog.q.len();
            let m = prog.b.len();
            unsafe { write_out(x_out, &sol.x, n) };
            unsafe { write_out(s_out, &sol.s, m) };
            unsafe { write_out(z_out, &sol.z, m) };
            if !obj_val_out.is_null() {
                unsafe { *obj_val_out = sol.obj_val };
            }
            if !iters_out.is_null() {
                unsafe { *iters_out = sol.iters as i32 };
            }
            if !status_out.is_null() {
                unsafe { *status_out = status_to_code(sol.status) };
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Status codes returned to C callers.
const STATUS_SOLVED: i32 = 0;
const STATUS_SOLVED_INACCURATE: i32 = 1;
const STATUS_PRIMAL_INFEASIBLE: i32 = 2;
const STATUS_DUAL_INFEASIBLE: i32 = 3;
const STATUS_MAX_ITERATIONS: i32 = 4;
const STATUS_TIME_LIMIT: i32 = 5;
const STATUS_NUMERICAL_ERROR: i32 = 6;
const STATUS_UNSOLVED: i32 = -1;

fn status_to_code(s: Status) -> i32 {
    match s {
        Status::Solved => STATUS_SOLVED,
        Status::SolvedInaccurate => STATUS_SOLVED_INACCURATE,
        Status::PrimalInfeasible => STATUS_PRIMAL_INFEASIBLE,
        Status::DualInfeasible => STATUS_DUAL_INFEASIBLE,
        Status::MaxIterations => STATUS_MAX_ITERATIONS,
        Status::TimeLimit => STATUS_TIME_LIMIT,
        Status::NumericalError => STATUS_NUMERICAL_ERROR,
        Status::Unsolved => STATUS_UNSOLVED,
    }
}

/// Read a dense vector from a C pointer + length.
unsafe fn read_vec(ptr: *const f64, len: usize) -> Vec<f64> {
    if ptr.is_null() || len == 0 {
        return vec![];
    }
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

/// Write a vector into a caller-allocated output buffer.
unsafe fn write_out(out: *mut f64, v: &[f64], len: usize) {
    if out.is_null() || len == 0 {
        return;
    }
    let n = v.len().min(len);
    unsafe {
        std::ptr::copy_nonoverlapping(v.as_ptr(), out, n);
    }
}

/// Build a `DenseMatrix` from CSC triplets, returning a dense row-major buffer.
/// The caller passes `indptr` (length ncols+1), `indices` (length nnz),
/// `data` (length nnz). The result is `nrows × ncols` in row-major order.
unsafe fn dense_from_csc(
    indptr: *const i32,
    indices: *const i32,
    data: *const f64,
    nrows: usize,
    ncols: usize,
) -> DenseMatrix<f64> {
    if indptr.is_null() || indices.is_null() || data.is_null() || nrows == 0 || ncols == 0 {
        return DenseMatrix::zeros(nrows, ncols);
    }
    let indptr = unsafe { std::slice::from_raw_parts(indptr, ncols + 1) };
    let indices = unsafe { std::slice::from_raw_parts(indices, indptr[ncols] as usize) };
    let nzval = unsafe { std::slice::from_raw_parts(data, indptr[ncols] as usize) };
    let mut buf = vec![0.0f64; nrows * ncols];
    for j in 0..ncols {
        for p in indptr[j] as usize..indptr[j + 1] as usize {
            let row = indices[p] as usize;
            if row < nrows {
                buf[row * ncols + j] = nzval[p];
            }
        }
    }
    DenseMatrix::from_row_major(nrows, ncols, buf)
}

/// Parse cones and build `(P, A)` from CSC triplets (`NULL` indptr → zero
/// matrix), the shared front half of the MIP entry points.
#[allow(clippy::too_many_arguments)]
unsafe fn mip_inputs_from_csc(
    n: i32,
    m: i32,
    p_indptr: *const i32,
    p_indices: *const i32,
    p_data: *const f64,
    a_indptr: *const i32,
    a_indices: *const i32,
    a_data: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
) -> Option<MipInputs> {
    if n <= 0 || m < 0 {
        return None;
    }
    let nu = n as usize;
    let mu = m as usize;
    let cones = unsafe { parse_cones(cone_specs, n_cones) }.ok()?;
    let p_mat = if p_indptr.is_null() {
        DenseMatrix::zeros(nu, nu)
    } else {
        unsafe { dense_from_csc(p_indptr, p_indices, p_data, nu, nu) }
    };
    let a_mat = if a_indptr.is_null() || mu == 0 {
        DenseMatrix::zeros(mu, nu)
    } else {
        unsafe { dense_from_csc(a_indptr, a_indices, a_data, mu, nu) }
    };
    Some((nu, mu, p_mat, a_mat, cones))
}

/// The dense-row-major twin of [`mip_inputs_from_csc`].
#[allow(clippy::too_many_arguments)]
unsafe fn mip_inputs_from_dense(
    n: i32,
    m: i32,
    p_dense: *const f64,
    a_dense: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
) -> Option<MipInputs> {
    if n <= 0 || m < 0 {
        return None;
    }
    let nu = n as usize;
    let mu = m as usize;
    let cones = unsafe { parse_cones(cone_specs, n_cones) }.ok()?;
    let p_mat = if p_dense.is_null() {
        DenseMatrix::zeros(nu, nu)
    } else {
        DenseMatrix::from_row_major(
            nu,
            nu,
            unsafe { std::slice::from_raw_parts(p_dense, nu * nu) }.to_vec(),
        )
    };
    let a_mat = if a_dense.is_null() || mu == 0 {
        DenseMatrix::zeros(mu, nu)
    } else {
        DenseMatrix::from_row_major(
            mu,
            nu,
            unsafe { std::slice::from_raw_parts(a_dense, mu * nu) }.to_vec(),
        )
    };
    Some((nu, mu, p_mat, a_mat, cones))
}

/// Parse cone specifications from a null-terminated string array.
/// Each entry is `"kind:dim"` (e.g. `"zero:5"`, `"nonneg:10"`, `"soc:3"`,
/// `"psd:6"`, `"exp:3"`). Returns a `Vec<Cone>` or an error string.
unsafe fn parse_cones(cone_specs: *const *const c_char, n_cones: i32) -> Result<Vec<Cone>, String> {
    if n_cones <= 0 {
        return Ok(Vec::new());
    }
    if cone_specs.is_null() {
        return Err("null cone_specs with n_cones > 0".into());
    }
    let mut cones = Vec::with_capacity(n_cones as usize);
    for i in 0..n_cones as usize {
        let spec_ptr = unsafe { *cone_specs.add(i) };
        if spec_ptr.is_null() {
            return Err("null cone spec".into());
        }
        let spec = unsafe { CStr::from_ptr(spec_ptr) }
            .to_str()
            .map_err(|_| "invalid UTF-8 in cone spec")?;
        let (kind, dim_str) = spec
            .split_once(':')
            .ok_or_else(|| format!("invalid cone spec: {spec}"))?;
        let cone = match kind {
            // Power cone, spec "power:<alpha>" (the backend encodes CVXPY's
            // PowCone3D as "power:3:<alpha>" — dimension is always 3).
            "power" => {
                let alpha: f64 = dim_str
                    .split_once(':')
                    .and_then(|(_, a)| a.parse().ok())
                    .ok_or_else(|| format!("invalid power alpha: {spec}"))?;
                Cone::Power(alpha)
            }
            _ => {
                let dim: usize = dim_str
                    .parse()
                    .map_err(|_| format!("invalid cone dim: {spec}"))?;
                match kind {
                    "zero" => Cone::Zero(dim),
                    "nonneg" => Cone::NonNegative(dim),
                    "soc" => Cone::SecondOrder(dim),
                    "psd" => Cone::PsdTriangle(dim),
                    "exp" => Cone::Exponential,
                    _ => return Err(format!("unknown cone kind: {kind}")),
                }
            }
        };
        cones.push(cone);
    }
    Ok(cones)
}

/// Solve a cone program.
///
/// All matrices are passed in **CSC** (compressed sparse column) format:
/// - `p_indptr`, `p_indices`, `p_data`: the `P` matrix (`n×n`, upper triangle only,
///   or pass `NULL` for zero `P`).
/// - `q`: dense objective vector (length `n`).
/// - `a_indptr`, `a_indices`, `a_data`: the `A` constraint matrix (`m×n`).
/// - `b`: dense RHS vector (length `m`).
/// - `cone_specs`: null-terminated string array of `"kind:dim"` specs (e.g. `"nonneg:5"`).
/// - `presolve`: 0 = no presolve, 1 = presolve enabled.
///
/// Output buffers (caller must allocate):
/// - `x_out`: length `n` (primal solution).
/// - `s_out`: length `m` (slacks).
/// - `z_out`: length `m` (duals).
/// - `obj_val_out`: pointer to a single `f64` (objective value).
/// - `iters_out`: pointer to a single `i32` (iteration count).
/// - `status_out`: pointer to a single `i32` (status code, see above).
///
/// Returns 0 on success, -1 on error (invalid arguments / unsupported cone).
///
/// Equivalent to [`iconic_solve_with_settings`] with `settings = NULL` (max_iters
/// 800, presolve from the `presolve` argument, all other settings default).
///
/// # Safety
///
/// All pointer arguments must be valid or NULL (as indicated). Output buffers
/// must be large enough for the expected result size.
#[no_mangle]
pub unsafe extern "C" fn iconic_solve(
    n: i32,
    m: i32,
    p_indptr: *const i32,
    p_indices: *const i32,
    p_data: *const f64,
    q: *const f64,
    a_indptr: *const i32,
    a_indices: *const i32,
    a_data: *const f64,
    b: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
    presolve: i32,
    x_out: *mut f64,
    s_out: *mut f64,
    z_out: *mut f64,
    obj_val_out: *mut f64,
    iters_out: *mut i32,
    status_out: *mut i32,
) -> i32 {
    unsafe {
        solve_csc(
            n,
            m,
            p_indptr,
            p_indices,
            p_data,
            q,
            a_indptr,
            a_indices,
            a_data,
            b,
            cone_specs,
            n_cones,
            presolve,
            x_out,
            s_out,
            z_out,
            obj_val_out,
            iters_out,
            status_out,
            None,
        )
    }
}

/// Solve a cone program (CSC input) with explicit settings.
///
/// Identical signature to [`iconic_solve`] plus one trailing argument:
/// - `settings`: pointer to a [`IconicSettings`] struct, or `NULL` for the
///   legacy behavior (max_iters 800, presolve from the `presolve` argument).
///   With a struct present, every non-sentinel field governs (see
///   [`IconicSettings`] for the sentinel values). A struct `version` other
///   than 1 is rejected with -1.
///
/// # Safety
///
/// All pointer arguments must be valid or NULL (as indicated).
#[no_mangle]
pub unsafe extern "C" fn iconic_solve_with_settings(
    n: i32,
    m: i32,
    p_indptr: *const i32,
    p_indices: *const i32,
    p_data: *const f64,
    q: *const f64,
    a_indptr: *const i32,
    a_indices: *const i32,
    a_data: *const f64,
    b: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
    presolve: i32,
    x_out: *mut f64,
    s_out: *mut f64,
    z_out: *mut f64,
    obj_val_out: *mut f64,
    iters_out: *mut i32,
    status_out: *mut i32,
    settings: *const IconicSettings,
) -> i32 {
    unsafe {
        solve_csc(
            n,
            m,
            p_indptr,
            p_indices,
            p_data,
            q,
            a_indptr,
            a_indices,
            a_data,
            b,
            cone_specs,
            n_cones,
            presolve,
            x_out,
            s_out,
            z_out,
            obj_val_out,
            iters_out,
            status_out,
            if settings.is_null() {
                None
            } else {
                Some(&*settings)
            },
        )
    }
}

/// Shared CSC implementation behind [`iconic_solve`] / [`iconic_solve_with_settings`].
#[allow(clippy::too_many_arguments)]
unsafe fn solve_csc(
    n: i32,
    m: i32,
    p_indptr: *const i32,
    p_indices: *const i32,
    p_data: *const f64,
    q: *const f64,
    a_indptr: *const i32,
    a_indices: *const i32,
    a_data: *const f64,
    b: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
    presolve: i32,
    x_out: *mut f64,
    s_out: *mut f64,
    z_out: *mut f64,
    obj_val_out: *mut f64,
    iters_out: *mut i32,
    status_out: *mut i32,
    settings: Option<&IconicSettings>,
) -> i32 {
    if n <= 0 || m < 0 {
        return -1;
    }
    let n = n as usize;
    let m = m as usize;

    // Parse cones.
    let cones = match unsafe { parse_cones(cone_specs, n_cones) } {
        Ok(c) => c,
        Err(_) => return -1,
    };

    // Build P matrix from CSC.
    let p_mat = if p_indptr.is_null() {
        DenseMatrix::zeros(n, n)
    } else {
        unsafe { dense_from_csc(p_indptr, p_indices, p_data, n, n) }
    };

    // Build A matrix from CSC.
    let a_mat = if a_indptr.is_null() || m == 0 {
        DenseMatrix::zeros(m, n)
    } else {
        unsafe { dense_from_csc(a_indptr, a_indices, a_data, m, n) }
    };

    let q_vec = unsafe { read_vec(q, n) };
    let b_vec = unsafe { read_vec(b, m) };

    let prog = ConeProgram {
        p: p_mat,
        q: q_vec,
        a: a_mat,
        a_csc: None,
        b: b_vec,
        cones,
    };

    let settings = match unsafe { build_settings(presolve, settings) } {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match unsafe {
        run_solve(
            &prog,
            &settings,
            x_out,
            s_out,
            z_out,
            obj_val_out,
            iters_out,
            status_out,
        )
    } {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Solve a cone program with **dense** row-major matrices.
///
/// - `p_dense`: `n×n` row-major f64 (or `NULL` for zero `P`).
/// - `q`: dense objective vector (length `n`).
/// - `a_dense`: `m×n` row-major f64.
/// - `b`: dense RHS vector (length `m`).
/// - `cone_specs`: null-terminated string array of `"kind:dim"` specs.
/// - `presolve`: 0 = no presolve, 1 = presolve enabled.
///
/// Output buffers (caller-allocated):
/// - `x_out[n]`, `s_out[m]`, `z_out[m]`, `obj_val_out[1]`, `iters_out[1]`, `status_out[1]`.
///
/// Returns 0 on success, -1 on error.
///
/// Equivalent to [`iconic_solve_dense_with_settings`] with `settings = NULL`
/// (max_iters 800, presolve from the `presolve` argument).
///
/// # Safety
///
/// All pointer arguments must be valid or NULL (as indicated). Output buffers
/// must be large enough for the expected result size.
#[no_mangle]
pub unsafe extern "C" fn iconic_solve_dense(
    n: i32,
    m: i32,
    p_dense: *const f64,
    q: *const f64,
    a_dense: *const f64,
    b: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
    presolve: i32,
    x_out: *mut f64,
    s_out: *mut f64,
    z_out: *mut f64,
    obj_val_out: *mut f64,
    iters_out: *mut i32,
    status_out: *mut i32,
) -> i32 {
    unsafe {
        solve_dense_impl(
            n,
            m,
            p_dense,
            q,
            a_dense,
            b,
            cone_specs,
            n_cones,
            presolve,
            x_out,
            s_out,
            z_out,
            obj_val_out,
            iters_out,
            status_out,
            None,
        )
    }
}

/// Solve a cone program (dense row-major input) with explicit settings.
///
/// Identical signature to [`iconic_solve_dense`] plus one trailing argument:
/// - `settings`: pointer to a [`IconicSettings`] struct, or `NULL` for the
///   legacy behavior. A struct `version` other than 1 is rejected with -1.
///
/// # Safety
///
/// All pointer arguments must be valid or NULL (as indicated).
#[no_mangle]
pub unsafe extern "C" fn iconic_solve_dense_with_settings(
    n: i32,
    m: i32,
    p_dense: *const f64,
    q: *const f64,
    a_dense: *const f64,
    b: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
    presolve: i32,
    x_out: *mut f64,
    s_out: *mut f64,
    z_out: *mut f64,
    obj_val_out: *mut f64,
    iters_out: *mut i32,
    status_out: *mut i32,
    settings: *const IconicSettings,
) -> i32 {
    unsafe {
        solve_dense_impl(
            n,
            m,
            p_dense,
            q,
            a_dense,
            b,
            cone_specs,
            n_cones,
            presolve,
            x_out,
            s_out,
            z_out,
            obj_val_out,
            iters_out,
            status_out,
            if settings.is_null() {
                None
            } else {
                Some(&*settings)
            },
        )
    }
}

/// Shared dense implementation behind [`iconic_solve_dense`] /
/// [`iconic_solve_dense_with_settings`].
#[allow(clippy::too_many_arguments)]
unsafe fn solve_dense_impl(
    n: i32,
    m: i32,
    p_dense: *const f64,
    q: *const f64,
    a_dense: *const f64,
    b: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
    presolve: i32,
    x_out: *mut f64,
    s_out: *mut f64,
    z_out: *mut f64,
    obj_val_out: *mut f64,
    iters_out: *mut i32,
    status_out: *mut i32,
    settings: Option<&IconicSettings>,
) -> i32 {
    if n <= 0 || m < 0 {
        return -1;
    }
    let n = n as usize;
    let m = m as usize;

    let cones = match unsafe { parse_cones(cone_specs, n_cones) } {
        Ok(c) => c,
        Err(_) => return -1,
    };

    let p_mat = if p_dense.is_null() {
        DenseMatrix::zeros(n, n)
    } else {
        DenseMatrix::from_row_major(
            n,
            n,
            unsafe { std::slice::from_raw_parts(p_dense, n * n) }.to_vec(),
        )
    };

    let a_mat = if a_dense.is_null() || m == 0 {
        DenseMatrix::zeros(m, n)
    } else {
        DenseMatrix::from_row_major(
            m,
            n,
            unsafe { std::slice::from_raw_parts(a_dense, m * n) }.to_vec(),
        )
    };

    let q_vec = unsafe { read_vec(q, n) };
    let b_vec = unsafe { read_vec(b, m) };

    let prog = ConeProgram {
        p: p_mat,
        q: q_vec,
        a: a_mat,
        a_csc: None,
        b: b_vec,
        cones,
    };

    let settings = match unsafe { build_settings(presolve, settings) } {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match unsafe {
        run_solve(
            &prog,
            &settings,
            x_out,
            s_out,
            z_out,
            obj_val_out,
            iters_out,
            status_out,
        )
    } {
        Ok(()) => 0,
        Err(e) => {
            if std::env::var_os("ICONIC_C_DEBUG").is_some() {
                eprintln!(
                    "[iconic-c] solve error: {e:?} (n={n} m={m} cones={:?})",
                    prog.cones
                );
            }
            -1
        }
    }
}

// ── MIP status codes ─────────────────────────────────────────────────────
const MIP_STATUS_OPTIMAL: i32 = 10;
const MIP_STATUS_FEASIBLE: i32 = 11;
const MIP_STATUS_INFEASIBLE: i32 = 12;
const MIP_STATUS_UNBOUNDED: i32 = 13;
const MIP_STATUS_NODE_LIMIT: i32 = 14;
const MIP_STATUS_TIME_LIMIT: i32 = 15;
const MIP_STATUS_NO_SOLUTION: i32 = 16;
const MIP_STATUS_NUMERICAL_ERROR: i32 = 17;

fn mip_status_to_code(s: MipStatus) -> i32 {
    match s {
        MipStatus::Optimal => MIP_STATUS_OPTIMAL,
        MipStatus::Feasible => MIP_STATUS_FEASIBLE,
        MipStatus::Infeasible => MIP_STATUS_INFEASIBLE,
        MipStatus::Unbounded => MIP_STATUS_UNBOUNDED,
        MipStatus::NodeLimit => MIP_STATUS_NODE_LIMIT,
        MipStatus::TimeLimit => MIP_STATUS_TIME_LIMIT,
        MipStatus::NoSolution => MIP_STATUS_NO_SOLUTION,
        MipStatus::NumericalError => MIP_STATUS_NUMERICAL_ERROR,
    }
}

/// Shared MIP implementation behind [`iconic_solve_mip`] /
/// [`iconic_solve_mip_csc`]: parses the variable types / bounds / gap settings,
/// runs the branch-and-bound search, and writes the solution into the
/// caller's buffers.
#[allow(clippy::too_many_arguments)]
unsafe fn solve_mip_impl(
    n: usize,
    m: usize,
    p_mat: DenseMatrix<f64>,
    a_mat: DenseMatrix<f64>,
    cones: Vec<Cone>,
    q: *const f64,
    b: *const f64,
    var_types: *const i32,
    lb: *const f64,
    ub: *const f64,
    max_nodes: i32,
    max_time: f64,
    abs_gap: f64,
    rel_gap: f64,
    x_out: *mut f64,
    obj_val_out: *mut f64,
    best_bound_out: *mut f64,
    gap_out: *mut f64,
    nodes_out: *mut i32,
    simplex_iters_out: *mut i32,
    status_out: *mut i32,
) -> i32 {
    let q_vec = unsafe { read_vec(q, n) };
    let b_vec = unsafe { read_vec(b, m) };

    let vt_vec: Vec<VarType> = if var_types.is_null() {
        vec![VarType::Continuous; n]
    } else {
        let raw = unsafe { std::slice::from_raw_parts(var_types, n) };
        raw.iter()
            .map(|&v| match v {
                1 => VarType::Integer,
                2 => VarType::Binary,
                _ => VarType::Continuous,
            })
            .collect()
    };

    // NULL lb defaults to -∞ (unbounded below), consistent with NULL ub→+∞.
    let lb_vec = if lb.is_null() {
        vec![-f64::INFINITY; n]
    } else {
        unsafe { read_vec(lb, n) }
    };
    let ub_vec = if ub.is_null() {
        vec![f64::INFINITY; n]
    } else {
        unsafe { read_vec(ub, n) }
    };

    let problem = MipProblem {
        p: p_mat,
        q: q_vec,
        a: a_mat,
        b: b_vec,
        cones,
        var_types: vt_vec,
        lb: lb_vec,
        ub: ub_vec,
        warm_start: None,
    };

    let mut settings = MipSettings::<f64>::default();
    if max_nodes > 0 {
        settings.max_nodes = max_nodes as usize;
    }
    if max_time > 0.0 {
        settings.max_time = max_time;
    }
    if abs_gap > 0.0 {
        settings.abs_gap = abs_gap;
    }
    if rel_gap > 0.0 {
        settings.rel_gap = rel_gap;
    }
    settings.mip_presolve = true;
    settings.heuristics = true;

    let sol = solve_mip(&problem, &settings);

    unsafe { write_out(x_out, &sol.x, n) };
    if !obj_val_out.is_null() {
        unsafe { *obj_val_out = sol.obj_val };
    }
    if !best_bound_out.is_null() {
        unsafe { *best_bound_out = sol.best_bound };
    }
    if !gap_out.is_null() {
        unsafe { *gap_out = sol.rel_gap };
    }
    if !nodes_out.is_null() {
        unsafe { *nodes_out = sol.nodes as i32 };
    }
    if !simplex_iters_out.is_null() {
        unsafe { *simplex_iters_out = sol.simplex_iters as i32 };
    }
    if !status_out.is_null() {
        unsafe { *status_out = mip_status_to_code(sol.status) };
    }
    0
}

/// Solve a mixed-integer program.
///
/// Same dense input format as [`iconic_solve_dense`], plus:
/// - `var_types`: length `n`, 0=continuous, 1=integer, 2=binary.
/// - `lb`: variable lower bounds (length `n`).
/// - `ub`: variable upper bounds (length `n`).
/// - `max_nodes`: max B&B nodes (0 = default 1,000,000).
/// - `max_time`: max wall-clock seconds (0.0 = default 3600).
/// - `abs_gap`: absolute MIP gap tolerance (0.0 = default 1e-6).
/// - `rel_gap`: relative MIP gap tolerance (0.0 = default 1e-4).
///
/// Output buffers (caller must allocate) — same as [`iconic_solve_dense`], plus:
/// - `best_bound_out`: pointer to a single `f64` (best dual bound).
/// - `gap_out`: pointer to a single `f64` (relative gap).
/// - `nodes_out`: pointer to a single `i32` (nodes explored).
/// - `simplex_iters_out`: pointer to a single `i32` (simplex iterations).
///
/// # Safety
///
/// All pointer arguments must be valid or NULL (as indicated).
#[no_mangle]
pub unsafe extern "C" fn iconic_solve_mip(
    n: i32,
    m: i32,
    p_dense: *const f64,
    q: *const f64,
    a_dense: *const f64,
    b: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
    var_types: *const i32,
    lb: *const f64,
    ub: *const f64,
    max_nodes: i32,
    max_time: f64,
    abs_gap: f64,
    rel_gap: f64,
    x_out: *mut f64,
    obj_val_out: *mut f64,
    best_bound_out: *mut f64,
    gap_out: *mut f64,
    nodes_out: *mut i32,
    simplex_iters_out: *mut i32,
    status_out: *mut i32,
) -> i32 {
    let Some((n, m, p_mat, a_mat, cones)) =
        (unsafe { mip_inputs_from_dense(n, m, p_dense, a_dense, cone_specs, n_cones) })
    else {
        return -1;
    };

    unsafe {
        solve_mip_impl(
            n,
            m,
            p_mat,
            a_mat,
            cones,
            q,
            b,
            var_types,
            lb,
            ub,
            max_nodes,
            max_time,
            abs_gap,
            rel_gap,
            x_out,
            obj_val_out,
            best_bound_out,
            gap_out,
            nodes_out,
            simplex_iters_out,
            status_out,
        )
    }
}

/// Solve a mixed-integer program with **sparse CSC** input.
///
/// Same as [`iconic_solve_mip`] but accepts CSC format for P and A. The matrices
/// are converted to dense internally (the MIP engine operates on
/// `DenseMatrix`), so this entry point exists purely to let callers hand over
/// data they already have in CSC form.
///
/// - `p_indptr`, `p_indices`, `p_data`: P matrix CSC (upper triangle, or NULL).
/// - `a_indptr`, `a_indices`, `a_data`: A matrix CSC.
/// - Other arguments identical to [`iconic_solve_mip`].
///
/// # Safety
/// All pointer arguments must be valid or NULL.
#[no_mangle]
pub unsafe extern "C" fn iconic_solve_mip_csc(
    n: i32,
    m: i32,
    p_indptr: *const i32,
    p_indices: *const i32,
    p_data: *const f64,
    q: *const f64,
    a_indptr: *const i32,
    a_indices: *const i32,
    a_data: *const f64,
    b: *const f64,
    cone_specs: *const *const c_char,
    n_cones: i32,
    var_types: *const i32,
    lb: *const f64,
    ub: *const f64,
    max_nodes: i32,
    max_time: f64,
    abs_gap: f64,
    rel_gap: f64,
    x_out: *mut f64,
    obj_val_out: *mut f64,
    best_bound_out: *mut f64,
    gap_out: *mut f64,
    nodes_out: *mut i32,
    simplex_iters_out: *mut i32,
    status_out: *mut i32,
) -> i32 {
    let Some((nu, mu, p_mat, a_mat, cones)) = (unsafe {
        mip_inputs_from_csc(
            n, m, p_indptr, p_indices, p_data, a_indptr, a_indices, a_data, cone_specs, n_cones,
        )
    }) else {
        return -1;
    };

    unsafe {
        solve_mip_impl(
            nu,
            mu,
            p_mat,
            a_mat,
            cones,
            q,
            b,
            var_types,
            lb,
            ub,
            max_nodes,
            max_time,
            abs_gap,
            rel_gap,
            x_out,
            obj_val_out,
            best_bound_out,
            gap_out,
            nodes_out,
            simplex_iters_out,
            status_out,
        )
    }
}

/// Return the ICONIC version string.
/// The caller must NOT free the returned pointer.
#[no_mangle]
pub extern "C" fn iconic_version() -> *const c_char {
    concat!("ICONIC ", env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_not_null() {
        let v = iconic_version();
        assert!(!v.is_null());
        let s = unsafe { CStr::from_ptr(v) }.to_str().unwrap();
        assert!(s.contains("ICONIC"));
    }

    #[test]
    fn status_mapping_is_complete() {
        for s in [
            Status::Solved,
            Status::SolvedInaccurate,
            Status::PrimalInfeasible,
            Status::DualInfeasible,
            Status::MaxIterations,
            Status::TimeLimit,
            Status::NumericalError,
            Status::Unsolved,
        ] {
            let c = status_to_code(s);
            assert!(c != -2, "unmapped status {s:?}");
        }
    }
}

#[cfg(test)]
mod parse_probe {
    use super::*;
    #[test]
    fn parse_power_spec() {
        let specs: Vec<Vec<u8>> = vec![b"power:3:0.5\0".to_vec()];
        let ptrs: Vec<*const i8> = specs.iter().map(|s| s.as_ptr() as *const i8).collect();
        let cones = unsafe { parse_cones(ptrs.as_ptr(), 1) };
        println!("parsed: {:?}", cones);
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;
    use std::ffi::CString;

    /// min ½‖y−c‖² s.t. y ∈ SOC(3) with c = (1,2,2) — the Euclidean projection
    /// onto the second-order cone. A genuine IPM solve: converges in 7 iters at
    /// the tight tolerance, 4 at 1e-4, and the presolve chain has no analytic
    /// path for a single SOC (unlike small doubleton QPs, which the presolve
    /// solves at iters=0). Dense row-major: P = I₃, q = −c, A = −I₃, b = 0,
    /// cones [soc:3].
    ///
    type FixedSocp = (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<CString>,
        Vec<*const c_char>,
    );

    /// Returns the data plus the cone-spec `CString`s (kept alive so the
    /// pointers stay valid) and the pointer list.
    fn fixed_socp() -> FixedSocp {
        let p = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let q = vec![-1.0, -2.0, -2.0];
        let a = vec![-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
        let b = vec![0.0, 0.0, 0.0];
        let specs: Vec<CString> = vec![CString::new("soc:3").unwrap()];
        let ptrs: Vec<*const c_char> = specs.iter().map(|s| s.as_ptr()).collect();
        (p, q, a, b, specs, ptrs)
    }

    struct CResult {
        x: Vec<f64>,
        s: Vec<f64>,
        z: Vec<f64>,
        obj: f64,
        iters: i32,
        status: i32,
        ret: i32,
    }

    /// Call `iconic_solve_dense` (settings = None) or
    /// `iconic_solve_dense_with_settings` (settings = Some).
    unsafe fn solve_dense_test(
        p: &[f64],
        q: &[f64],
        a: &[f64],
        b: &[f64],
        cone_ptrs: &[*const c_char],
        presolve: i32,
        settings: Option<&IconicSettings>,
    ) -> CResult {
        let n = q.len();
        let m = b.len();
        let mut x = vec![0.0; n];
        let mut s = vec![0.0; m];
        let mut z = vec![0.0; m];
        let mut obj = 0.0;
        let mut iters = 0i32;
        let mut status = 0i32;
        let ret = match settings {
            None => unsafe {
                iconic_solve_dense(
                    n as i32,
                    m as i32,
                    p.as_ptr(),
                    q.as_ptr(),
                    a.as_ptr(),
                    b.as_ptr(),
                    cone_ptrs.as_ptr(),
                    cone_ptrs.len() as i32,
                    presolve,
                    x.as_mut_ptr(),
                    s.as_mut_ptr(),
                    z.as_mut_ptr(),
                    &mut obj,
                    &mut iters,
                    &mut status,
                )
            },
            Some(cs) => unsafe {
                iconic_solve_dense_with_settings(
                    n as i32,
                    m as i32,
                    p.as_ptr(),
                    q.as_ptr(),
                    a.as_ptr(),
                    b.as_ptr(),
                    cone_ptrs.as_ptr(),
                    cone_ptrs.len() as i32,
                    presolve,
                    x.as_mut_ptr(),
                    s.as_mut_ptr(),
                    z.as_mut_ptr(),
                    &mut obj,
                    &mut iters,
                    &mut status,
                    cs,
                )
            },
        };
        CResult {
            x,
            s,
            z,
            obj,
            iters,
            status,
            ret,
        }
    }

    /// Max KKT residual of the fixed SOCP, mirroring iconic-bench's
    /// `cone_kkt_residual` (stationarity + primal + normalized complementarity).
    fn kkt_res(r: &CResult) -> f64 {
        // rd = Px + q + Aᵀz = x + q − z (A = −I)
        let q = [-1.0, -2.0, -2.0];
        let rd =
            r.x.iter()
                .zip(&q)
                .zip(&r.z)
                .map(|((&xi, &qi), &zi)| (xi + qi - zi).abs())
                .fold(0.0f64, f64::max);
        // rh = A x + s − b = s − x
        let mut rh = 0.0f64;
        for i in 0..3 {
            rh = rh.max((r.s[i] - r.x[i]).abs());
        }
        let comp = (r.s[0] * r.z[0] + r.s[1] * r.z[1] + r.s[2] * r.z[2]).abs() / 3.0;
        rd.max(rh).max(comp)
    }

    fn default_settings_struct() -> IconicSettings {
        IconicSettings {
            version: 1,
            max_iters: 0,
            eps_abs: 0.0,
            eps_rel: 0.0,
            eps_gap: 0.0,
            presolve: -1,
            hsd: -1,
            sparse_kkt: -1,
            blas_threads: 0,
        }
    }

    #[test]
    fn null_settings_is_bit_identical_to_legacy_entry() {
        // The `*_with_settings` entry with a NULL settings pointer must behave
        // exactly like the legacy entry (presolve arg + max_iters 800).
        let (p, q, a, b, _specs, cones) = fixed_socp();
        let r_legacy = unsafe { solve_dense_test(&p, &q, &a, &b, &cones, 1, None) };
        let r_null = unsafe {
            solve_dense_test(&p, &q, &a, &b, &cones, 1, Some(&default_settings_struct()))
        };
        assert_eq!(r_legacy.ret, 0);
        assert_eq!(r_null.ret, 0);
        assert_eq!(r_legacy.status, r_null.status);
        assert_eq!(r_legacy.iters, r_null.iters);
        assert_eq!(r_legacy.obj.to_bits(), r_null.obj.to_bits());
        for (u, v) in r_legacy.x.iter().zip(r_null.x.iter()) {
            assert_eq!(u.to_bits(), v.to_bits(), "x differs");
        }
        for (u, v) in r_legacy.z.iter().zip(r_null.z.iter()) {
            assert_eq!(u.to_bits(), v.to_bits(), "z differs");
        }
    }

    #[test]
    fn settings_struct_with_defaults_matches_legacy() {
        // A fully-sentinel struct (version 1) must also reproduce legacy
        // behavior — the sentinels are defined to mean exactly that. The
        // legacy `presolve` argument is passed as 0 while the struct's
        // sentinel resolves to default true, proving the struct governs.
        let (p, q, a, b, _specs, cones) = fixed_socp();
        let r_legacy = unsafe { solve_dense_test(&p, &q, &a, &b, &cones, 1, None) };
        let r_sentinel = unsafe {
            solve_dense_test(&p, &q, &a, &b, &cones, 0, Some(&default_settings_struct()))
        };
        assert_eq!(r_legacy.status, r_sentinel.status);
        assert_eq!(r_legacy.iters, r_sentinel.iters);
        assert_eq!(r_legacy.obj.to_bits(), r_sentinel.obj.to_bits());
    }

    #[test]
    fn max_iters_cap_binds() {
        // A tiny max_iters must stop the solve at the cap. The engine reports
        // the 0-based iteration index, so a capped run reports max_iters − 1
        // iterations and grades below Solved (the best-iterate grading at
        // relaxed tolerance — SolvedInaccurate on this small instance).
        let (p, q, a, b, _specs, cones) = fixed_socp();
        for max_iters in [1i32, 5] {
            let mut cs = default_settings_struct();
            cs.max_iters = max_iters;
            let r = unsafe { solve_dense_test(&p, &q, &a, &b, &cones, 1, Some(&cs)) };
            assert_eq!(r.ret, 0);
            assert_eq!(
                r.iters,
                max_iters - 1,
                "capped run must stop at the cap (0-based): iters={} cap={}",
                r.iters,
                max_iters
            );
            assert_ne!(
                r.status, STATUS_SOLVED,
                "status={} at cap={}",
                r.status, max_iters
            );
        }
        // With a comfortable budget the same instance solves cleanly.
        let r_full = unsafe {
            solve_dense_test(&p, &q, &a, &b, &cones, 1, Some(&default_settings_struct()))
        };
        assert_eq!(r_full.status, STATUS_SOLVED, "status={}", r_full.status);
        assert!(r_full.iters > 5, "full solve iters={}", r_full.iters);
    }

    #[test]
    fn loose_eps_fewer_iters_larger_kkt() {
        // eps = 1e-4 must terminate earlier than eps = 1e-8: strictly fewer
        // iterations and a strictly larger final KKT residual.
        let (p, q, a, b, _specs, cones) = fixed_socp();
        let mut cs_loose = default_settings_struct();
        cs_loose.eps_abs = 1e-4;
        cs_loose.eps_rel = 1e-4;
        cs_loose.eps_gap = 1e-4;
        let mut cs_tight = default_settings_struct();
        cs_tight.eps_abs = 1e-8;
        cs_tight.eps_rel = 1e-8;
        cs_tight.eps_gap = 1e-8;
        let r_loose = unsafe { solve_dense_test(&p, &q, &a, &b, &cones, 1, Some(&cs_loose)) };
        let r_tight = unsafe { solve_dense_test(&p, &q, &a, &b, &cones, 1, Some(&cs_tight)) };
        assert_eq!(
            r_loose.status, STATUS_SOLVED,
            "loose status={}",
            r_loose.status
        );
        assert_eq!(
            r_tight.status, STATUS_SOLVED,
            "tight status={}",
            r_tight.status
        );
        assert!(
            r_loose.iters < r_tight.iters,
            "loose iters {} !< tight iters {}",
            r_loose.iters,
            r_tight.iters
        );
        assert!(
            kkt_res(&r_loose) > kkt_res(&r_tight),
            "loose kkt {} !> tight kkt {}",
            kkt_res(&r_loose),
            kkt_res(&r_tight)
        );
        // The tight solve actually converges to the known projection optimum.
        // c = (1,2,2): ‖v‖ = 2√2, t = 1 < ‖v‖ so y* = (scale·‖v‖, scale·v)
        // with scale = (‖v‖ + t)/(2‖v‖). The program is min ½‖y‖² − cᵀy =
        // ½‖y−c‖² − ½‖c‖², so obj* = ½‖y*−c‖² − ½‖c‖².
        let nv = 2.0 * std::f64::consts::SQRT_2;
        let scale = (nv + 1.0) / (2.0 * nv);
        let y_star = [scale * nv, scale * 2.0, scale * 2.0];
        let obj_star = 0.5
            * ((y_star[0] - 1.0).powi(2) + (y_star[1] - 2.0).powi(2) + (y_star[2] - 2.0).powi(2))
            - 0.5 * (1.0 + 4.0 + 4.0);
        assert!(
            (r_tight.obj - obj_star).abs() < 1e-6,
            "obj={} vs {obj_star}",
            r_tight.obj
        );
    }

    #[test]
    fn version_mismatch_is_rejected() {
        let (p, q, a, b, _specs, cones) = fixed_socp();
        let mut cs = default_settings_struct();
        cs.version = 2;
        let r = unsafe { solve_dense_test(&p, &q, &a, &b, &cones, 1, Some(&cs)) };
        assert_eq!(
            r.ret, -1,
            "a caller compiled against a newer header must be rejected"
        );
    }
}
