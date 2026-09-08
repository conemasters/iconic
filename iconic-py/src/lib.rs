//! `iconic` — Python bindings for the ICONIC convex optimizer (the native half of the
//! CVXPY backend).
//!
//! Exposes [`solve`] over the canonical conic standard form
//! `min ½xᵀPx + qᵀx s.t. Ax + s = b, s ∈ K`, with the cone product given as a list
//! of `(kind, dim)` pairs (`"zero"`, `"nonneg"`, `"soc"`, `"psd"`, `"exp"`) in
//! canonical order. Matrices are passed as SciPy **CSC** triples (numpy `int64`
//! index arrays + `f64` data), so no `.todense()` is ever materialized in Python.

use iconic_api::{solve as api_solve, ConeProgram, SolveError};
use iconic_core::{Cone, Settings};
use iconic_linalg::{CscMatrix, DenseMatrix};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The solution returned to Python.
#[pyclass]
struct Solution {
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    x: Vec<f64>,
    #[pyo3(get)]
    obj_val: f64,
}

/// Build a row-major `DenseMatrix` from a SciPy **CSC** matrix (column pointers, row
/// indices, values), scattering the nonzeros into a dense buffer. CVXPY hands the backend
/// sparse matrices; taking them as CSC avoids `.todense()` building (and copying) a full
/// dense array in Python — for a typical sparse A that array is enormous (a 9000-nonzero A
/// can be 27M dense entries), so the CSC path is far cheaper in both time and peak memory.
fn dense_from_csc(
    indptr: &[i64],
    indices: &[i64],
    data: &[f64],
    nrows: usize,
    ncols: usize,
) -> DenseMatrix<f64> {
    let mut buf = vec![0.0f64; nrows * ncols];
    for j in 0..ncols {
        for p in indptr[j] as usize..indptr[j + 1] as usize {
            buf[indices[p] as usize * ncols + j] = data[p];
        }
    }
    DenseMatrix::from_row_major(nrows, ncols, buf)
}

fn build_cones(specs: &[(String, usize)]) -> PyResult<Vec<Cone>> {
    specs
        .iter()
        .map(|(kind, d)| match kind.as_str() {
            "zero" => Ok(Cone::Zero(*d)),
            "nonneg" => Ok(Cone::NonNegative(*d)),
            "soc" => Ok(Cone::SecondOrder(*d)),
            // `d` is the svec length k(k+1)/2 of a k×k PSD matrix.
            "psd" => Ok(Cone::PsdTriangle(*d)),
            // The 3-D exponential cone (`d` ignored / expected 3).
            "exp" => Ok(Cone::Exponential),
            other => Err(PyValueError::new_err(format!("unsupported cone '{other}'"))),
        })
        .collect()
}

/// Solve a cone program. `P` (`n×n`) and `A` (`m×n`) are given as SciPy **CSC** triples
/// (`indptr`, `indices`, `data`, int64 index arrays); `q`/`b` are 1-D float arrays; `cones`
/// is a list of `(kind, dim)` pairs summing to `m`. Taking the matrices sparse avoids a
/// Python `.todense()` that would materialize a full (often >99.9%-zero) dense array.
#[pyfunction]
#[pyo3(signature = (n, m, p_indptr, p_indices, p_data, q, a_indptr, a_indices, a_data, b, cones, presolve=true))]
#[allow(clippy::too_many_arguments)]
fn solve(
    n: usize,
    m: usize,
    p_indptr: PyReadonlyArray1<i64>,
    p_indices: PyReadonlyArray1<i64>,
    p_data: PyReadonlyArray1<f64>,
    q: PyReadonlyArray1<f64>,
    a_indptr: PyReadonlyArray1<i64>,
    a_indices: PyReadonlyArray1<i64>,
    a_data: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    cones: Vec<(String, usize)>,
    presolve: bool,
) -> PyResult<Solution> {
    let a_csc = {
        let ip = a_indptr.as_slice()?;
        let ix = a_indices.as_slice()?;
        let da = a_data.as_slice()?;
        let colptr: Vec<usize> = ip.iter().map(|&v| v as usize).collect();
        let rowval: Vec<usize> = ix.iter().map(|&v| v as usize).collect();
        let nzval: Vec<f64> = da.to_vec();
        if colptr.len() == n + 1 && colptr[n] == rowval.len() {
            Some(CscMatrix {
                m,
                n,
                colptr,
                rowval,
                nzval,
            })
        } else {
            None
        }
    };
    let prog = ConeProgram {
        p: dense_from_csc(
            p_indptr.as_slice()?,
            p_indices.as_slice()?,
            p_data.as_slice()?,
            n,
            n,
        ),
        q: q.as_slice()?.to_vec(),
        a: dense_from_csc(
            a_indptr.as_slice()?,
            a_indices.as_slice()?,
            a_data.as_slice()?,
            m,
            n,
        ),
        b: b.as_slice()?.to_vec(),
        a_csc,
        cones: build_cones(&cones)?,
    };
    let settings = Settings::<f64> {
        presolve,
        ..Settings::default()
    };
    match api_solve(&prog, &settings) {
        Ok(sol) => Ok(Solution {
            status: format!("{:?}", sol.status),
            x: sol.x,
            obj_val: sol.obj_val,
        }),
        Err(SolveError::UnsupportedCone) => Err(PyValueError::new_err("unsupported cone")),
        Err(SolveError::ConeOrder) => {
            Err(PyValueError::new_err("Zero cones must precede the rest"))
        }
        Err(SolveError::DimensionMismatch) => Err(PyValueError::new_err("dimension mismatch")),
    }
}

/// The `iconic._iconic` Python module (native extension).
#[pymodule]
fn _iconic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Solution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
