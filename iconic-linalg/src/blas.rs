//! Platform BLAS/LAPACK dispatch — the single entry point for all dense linear algebra.
//!
//! Dispatch strategy: Apple Accelerate on macOS, system BLAS (OpenBLAS/MKL/BLIS/
//! Netlib) on Linux, pure-Rust `faer` everywhere as fallback.
//!
//! # Layout convention
//!
//! All public functions accept **row-major** slices (ICONIC's native layout) and use
//! the CBLAS interface with `CblasRowMajor` ordering — no row↔column translation needed.
//! On platforms without CBLAS (very rare), we fall back to faer.

#![allow(unexpected_cfgs)] // `blas_system` cfg emitted by build.rs, not a Cargo feature

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Once;

// ---------------------------------------------------------------------------
// Runtime BLAS gate
// ---------------------------------------------------------------------------
// Platform BLAS (Accelerate/OpenBLAS/MKL) is available at compile time on macOS
// and on Linux when build.rs detects a system BLAS. However, it is gated to
// activate only for the Woodbury path (low-rank P) by default — the dense
// condensed QP path and the conic path use faer's SIMD kernels, which are
// faster for the per-iteration small/medium dense factorizations in those paths.
// The `use-blas` feature flag enables it unconditionally.
// ---------------------------------------------------------------------------
static BLAS_ENABLED: AtomicBool = AtomicBool::new(cfg!(feature = "use-blas"));

// Per-thread override: a thread can opt ITSELF into a different dispatch
// choice without changing what every other thread's solves see. The
// me==0 escalation regression test relies on this: it needs its solve to
// run with platform BLAS off (the dsyrk summation-order difference is what
// triggers the first-attempt Cholesky failure it guards), but a process-wide
// toggle raced every other test running in parallel — a concurrent solve
// (measured: the illcond n80 regression test) dispatched differently
// mid-solve and produced a different, garbage result (kkt >= 1000 vs the
// deterministic 1.80). A thread opting itself out affects only that
// thread's own wrapper calls.
thread_local! {
    static BLAS_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// OpenBLAS's own internal worker-thread pool has no equivalent to
/// `faer_dense::set_parallelism_seq`'s size-based sequential/parallel switch —
/// left uncapped, it defaults to one thread per core. Profiling (`perf record`
/// on a repeated dense-QP-solve workload sweeping n=400..2000 on a 24-core
/// machine) found >90% of *cycles* going to `sched_yield` inside idle OpenBLAS
/// worker threads rather than arithmetic, because ICONIC's per-iteration KKT-scale
/// gemm/dsyrk/dgemv/dpotrf calls are individually too small to amortize the
/// wake/dispatch/sync cost across many threads. Measured wall-clock: capping to
/// 4 threads was 1.16x-2.5x faster than the 24-thread default at every tested
/// size, with no size where more threads won.
///
/// The automatic default (`None`) uses the measured optimum (4), and
/// `Some(n)` overrides — a Threads-style parameter (default automatic
/// "may choose fewer"; an explicit cap overrides). The large LAPACK factors
/// (`dsytrf`) scale with this cap.
#[cfg(not(target_os = "macos"))]
static OPENBLAS_THREADS_INIT: Once = Once::new();

#[cfg(not(target_os = "macos"))]
static BLAS_THREAD_CAP: AtomicUsize = AtomicUsize::new(4);

/// Set the OpenBLAS worker-thread cap. `None` restores the automatic default
/// (the measured optimum, 4) — a Threads-style parameter.
#[cfg(not(target_os = "macos"))]
pub fn set_blas_threads(cap: Option<usize>) {
    let n = cap.unwrap_or(4).clamp(
        1,
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4),
    );
    BLAS_THREAD_CAP.store(n, Ordering::Relaxed);
    if OPENBLAS_THREADS_INIT.is_completed() {
        unsafe {
            ffi::openblas_set_num_threads(n as i32);
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn ensure_openblas_thread_cap() {
    OPENBLAS_THREADS_INIT.call_once(|| unsafe {
        ffi::openblas_set_num_threads(BLAS_THREAD_CAP.load(Ordering::Relaxed) as i32);
    });
}

/// Enable or disable platform BLAS dispatch at runtime, for THIS thread only.
///
/// When `true`, functions dispatch to platform BLAS (Accelerate/OpenBLAS/MKL)
/// where available. When `false` (the default without the `use-blas` feature),
/// faer's SIMD kernels are used on all paths. The override is per-thread and
/// falls back to the process-wide setting (`BLAS_ENABLED`) when unset, so a
/// test can opt its own worker thread into a different dispatch without
/// racing the solves of every other thread in the process. Threads spawned
/// afterwards start from the process default (no inheritance) — call this
/// inside the thread that needs the override.
pub fn set_blas_enabled(enabled: bool) {
    BLAS_OVERRIDE.with(|o| o.set(Some(enabled)));
}

/// Whether platform BLAS dispatch is currently active.
///
/// Every BLAS wrapper (`gemm`, `dsyrk`, `dpotrf`, ...) checks this before
/// dispatching, so it is the one choke point guaranteed to run before any
/// actual OpenBLAS call — including when BLAS is enabled via the `use-blas`
/// Cargo feature default rather than an explicit `set_blas_enabled(true)`
/// call. Piggyback the one-time OpenBLAS thread-cap init here rather than in
/// `set_blas_enabled`, which the feature-default path never calls.
pub fn blas_enabled() -> bool {
    let enabled = BLAS_OVERRIDE
        .with(|o| o.get())
        .unwrap_or_else(|| BLAS_ENABLED.load(Ordering::Relaxed));
    #[cfg(not(target_os = "macos"))]
    if enabled {
        ensure_openblas_thread_cap();
    }
    enabled
}
// ---------------------------------------------------------------------------
// CBLAS enum constants (from cblas.h)
// ---------------------------------------------------------------------------
const CBLAS_ROW_MAJOR: u32 = 101;
const CBLAS_NO_TRANS: u32 = 111;
const CBLAS_TRANS: u32 = 112;
const CBLAS_LOWER: u32 = 122;

// ---------------------------------------------------------------------------
// Platform BLAS FFI declarations (CBLAS interface)
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "macos", blas_system))]
mod ffi {
    #[cfg(target_os = "macos")]
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        pub fn cblas_dgemm(
            order: u32,
            transa: u32,
            transb: u32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );

        pub fn cblas_dsyrk(
            order: u32,
            uplo: u32,
            trans: u32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );

        pub fn cblas_dgemv(
            order: u32,
            trans: u32,
            m: i32,
            n: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            x: *const f64,
            incx: i32,
            beta: f64,
            y: *mut f64,
            incy: i32,
        );

        pub fn dpotrf_(
            uplo: *const u8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            info: *mut i32,
        );

        pub fn dpotrs_(
            uplo: *const u8,
            n: *const i32,
            nrhs: *const i32,
            a: *const f64,
            lda: *const i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32,
        );

        pub fn dsytrf_(
            uplo: *const u8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
        );

        pub fn dsytrs_(
            uplo: *const u8,
            n: *const i32,
            nrhs: *const i32,
            a: *const f64,
            lda: *const i32,
            ipiv: *const i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32,
        );

        pub fn dgetrf_(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            info: *mut i32,
        );

        pub fn dgetrs_(
            trans: *const u8,
            n: *const i32,
            nrhs: *const i32,
            a: *const f64,
            lda: *const i32,
            ipiv: *const i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32,
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[link(name = "openblas")]
    extern "C" {
        /// Caps OpenBLAS's own worker-thread pool — see `ensure_openblas_thread_cap`.
        pub fn openblas_set_num_threads(num_threads: i32);

        pub fn cblas_dgemv(
            order: u32,
            trans: u32,
            m: i32,
            n: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            x: *const f64,
            incx: i32,
            beta: f64,
            y: *mut f64,
            incy: i32,
        );

        pub fn cblas_dgemm(
            order: u32,
            transa: u32,
            transb: u32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );

        pub fn cblas_dsyrk(
            order: u32,
            uplo: u32,
            trans: u32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );

        pub fn dpotrf_(
            uplo: *const u8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            info: *mut i32,
        );

        pub fn dpotrs_(
            uplo: *const u8,
            n: *const i32,
            nrhs: *const i32,
            a: *const f64,
            lda: *const i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32,
        );

        pub fn dsytrf_(
            uplo: *const u8,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
        );

        pub fn dsytrs_(
            uplo: *const u8,
            n: *const i32,
            nrhs: *const i32,
            a: *const f64,
            lda: *const i32,
            ipiv: *const i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32,
        );

        pub fn dgetrf_(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            info: *mut i32,
        );

        pub fn dgetrs_(
            trans: *const u8,
            n: *const i32,
            nrhs: *const i32,
            a: *const f64,
            lda: *const i32,
            ipiv: *const i32,
            b: *mut f64,
            ldb: *const i32,
            info: *mut i32,
        );
    }
}

// ---------------------------------------------------------------------------
// Safe wrappers (platform BLAS path — CBLAS row-major)
// ---------------------------------------------------------------------------

#[cfg(any(target_os = "macos", blas_system))]
mod platform {
    use super::ffi;

    #[inline]
    pub fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        c: &mut [f64],
        ldc: usize,
        alpha: f64,
        beta: f64,
        trans_a: bool,
        trans_b: bool,
    ) {
        let ta = if trans_a {
            super::CBLAS_TRANS
        } else {
            super::CBLAS_NO_TRANS
        };
        let tb = if trans_b {
            super::CBLAS_TRANS
        } else {
            super::CBLAS_NO_TRANS
        };
        unsafe {
            ffi::cblas_dgemm(
                super::CBLAS_ROW_MAJOR,
                ta,
                tb,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }

    /// y = α·A·x + β·y  where A is m×n row-major.
    #[inline]
    pub fn gemv(
        m: usize,
        n: usize,
        a: &[f64],
        lda: usize,
        x: &[f64],
        y: &mut [f64],
        alpha: f64,
        beta: f64,
        trans: bool,
    ) {
        let tr = if trans {
            super::CBLAS_TRANS
        } else {
            super::CBLAS_NO_TRANS
        };
        unsafe {
            ffi::cblas_dgemv(
                super::CBLAS_ROW_MAJOR,
                tr,
                m as i32,
                n as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                x.as_ptr(),
                1,
                beta,
                y.as_mut_ptr(),
                1,
            );
        }
    }

    /// C = α·Aᵀ·A + β·C  where A is g×n row-major, C is n×n symmetric.
    /// Only the lower triangle of C is written (CBLAS Lower).
    #[inline]
    pub fn dsyrk(
        n: usize,
        g: usize,
        a: &[f64],
        lda: usize,
        c: &mut [f64],
        ldc: usize,
        alpha: f64,
        beta: f64,
    ) {
        unsafe {
            ffi::cblas_dsyrk(
                super::CBLAS_ROW_MAJOR,
                super::CBLAS_LOWER,    // fill lower triangle (row-major)
                super::CBLAS_NO_TRANS, // A is n×g (Bᵀ), C = A·Aᵀ = Bᵀ·B
                n as i32,
                g as i32, // N=n, K=g
                alpha,
                a.as_ptr(),
                lda as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }

    // ===================================================================
    // LAPACK: dpotrf / dpotrs — Cholesky factor & solve
    // These use Fortran column-major ABI — caller packs to column-major.
    // ===================================================================

    pub fn dpotrf_ul(n: usize, a: &mut [f64], uplo: u8) -> bool {
        let n32 = n as i32;
        let lda = n32;
        let mut info: i32 = 0;
        unsafe {
            ffi::dpotrf_(&uplo, &n32, a.as_mut_ptr(), &lda, &mut info);
        }
        info == 0
    }

    #[allow(dead_code)] // live on macOS/system-BLAS builds only; Linux routes to faer
    pub fn dpotrs(n: usize, a: &[f64], b: &mut [f64], _nrhs: usize) {
        dpotrs_ul(n, a, b, 1, b'L')
    }

    pub fn dpotrs_ul(n: usize, a: &[f64], b: &mut [f64], _nrhs: usize, uplo: u8) {
        let n32 = n as i32;
        let nrhs: i32 = 1;
        let lda = n32;
        let ldb = n32;
        let mut info: i32 = 0;
        unsafe {
            ffi::dpotrs_(
                &uplo,
                &n32,
                &nrhs,
                a.as_ptr(),
                &lda,
                b.as_mut_ptr(),
                &ldb,
                &mut info,
            );
        }
    }

    /// LAPACK `dsytrf` — the symmetric-indefinite (Bunch–Kaufman) factorization
    /// `A = U D Uᵀ` with 2×2 pivoting, for quasidefinite KKT systems. OpenBLAS
    /// multithreads this; faer 0.24 has no parallel LDLᵀ/LBLT, so the conic
    /// engine's big quasidefinite factors are sequential today. Returns the
    /// pivot permutation, or `None` when the factorization fails.
    pub fn dsytrf(n: usize, a: &mut [f64]) -> Option<Vec<i32>> {
        let n32 = n as i32;
        let lda = n32;
        let uplo = b'L';
        let mut ipiv = vec![0i32; n];
        let mut info: i32 = 0;
        // Workspace query, then allocate.
        let lwork: i32 = -1;
        let mut work_tmp = [0.0f64; 1];
        unsafe {
            ffi::dsytrf_(
                &uplo,
                &n32,
                a.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                work_tmp.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return None;
        }
        let lwork = work_tmp[0] as i32;
        let mut work = vec![0.0f64; lwork.max(1) as usize];
        unsafe {
            ffi::dsytrf_(
                &uplo,
                &n32,
                a.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        if info != 0 {
            return None;
        }
        Some(ipiv)
    }
    #[allow(unused_mut)]
    pub fn dsytrs(n: usize, a: &[f64], ipiv: &[i32], b: &mut [f64]) {
        let n32 = n as i32;
        let nrhs: i32 = 1;
        let lda = n32;
        let ldb = n32;
        let uplo = b'L';
        let mut info: i32 = 0;
        unsafe {
            ffi::dsytrs_(
                &uplo,
                &n32,
                &nrhs,
                a.as_ptr(),
                &lda,
                ipiv.as_ptr(),
                b.as_mut_ptr(),
                &ldb,
                &mut info,
            );
        }
    }

    /// LAPACK `dgetrf` — LU with partial pivoting, `A = P L U` (row-major
    /// `m×n` input factored in place; `U` in the pivot rows, the unit-lower
    /// `L` multipliers below the diagonal, row swaps recorded in `ipiv`
    /// (1-based)). This is exactly the convention the simplex's dense basis
    /// factor uses, so the hand-rolled factor can be swapped for LAPACK
    /// where BLAS is available. Returns `None` on failure (`info != 0`).
    pub fn getrf(m: usize, n: usize, a: &mut [f64]) -> Option<Vec<i32>> {
        // LAPACK requires lda >= max(1, m); a zero-row factor is empty.
        if m == 0 || n == 0 {
            return Some(Vec::new());
        }
        let m32 = m as i32;
        let n32 = n as i32;
        let lda = m32;
        let mut ipiv = vec![0i32; m.min(n)];
        let mut info: i32 = 0;
        unsafe {
            ffi::dgetrf_(
                &m32,
                &n32,
                a.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return None;
        }
        Some(ipiv)
    }

    /// LAPACK `dgetrs` — solve `A x = b` (`trans = b'N'`) or `Aᵀ x = b`
    /// (`trans = b'T'`) against a [`getrf`] factorization, one right-hand
    /// side in place.
    pub fn getrs(n: usize, a: &[f64], ipiv: &[i32], b: &mut [f64], trans: bool) {
        if n == 0 {
            return;
        }
        let n32 = n as i32;
        let nrhs: i32 = 1;
        let lda = n32;
        let ldb = n32;
        let tr = if trans { b'T' } else { b'N' };
        let mut info: i32 = 0;
        unsafe {
            ffi::dgetrs_(
                &tr,
                &n32,
                &nrhs,
                a.as_ptr(),
                &lda,
                ipiv.as_ptr(),
                b.as_mut_ptr(),
                &ldb,
                &mut info,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatch wrappers — the public API
// ---------------------------------------------------------------------------
// On platforms with BLAS, check blas_enabled() to choose between platform BLAS
// and faer at runtime. On platforms without BLAS, always use faer.
// ---------------------------------------------------------------------------

#[inline]
#[allow(unused_variables)]
/// Factor a symmetric-indefinite matrix via the platform BLAS (LAPACK
/// `dsytrf`, the 2×2-pivoted Bunch–Kaufman) when enabled — OpenBLAS
/// multithreads it, which faer 0.24's LDLᵀ/LBLT cannot. Returns `None` when
/// BLAS is unavailable or the factor fails (callers fall back to their own
/// factorization). The matrix is consumed in place (the factor owns it).
pub fn dsytrf(n: usize, a: &mut [f64]) -> Option<Vec<i32>> {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        return platform::dsytrf(n, a);
    }
    None
}

/// Solve `A x = b` against a [`dsytrf`] factorization (LAPACK `dsytrs`).
/// Only meaningful after a successful `dsytrf` of the same matrix.
pub fn dsytrs(n: usize, a: &[f64], ipiv: &[i32], b: &mut [f64]) {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        platform::dsytrs(n, a, ipiv, b);
    }
}

/// LAPACK `dgetrf` — LU with partial pivoting (row-major `m×n`, factored in
/// place, 1-based pivot permutation returned). Returns `None` when BLAS is
/// unavailable or the factorization fails.
///
/// Runs under a temporary single-thread OpenBLAS cap: OpenBLAS's *parallel*
/// `dgetrf` recursion overflows the stack on repeated small factorizations
/// (reproduced: 5+ consecutive 105×105 `dgetrf` calls from a worker thread
/// with the default 4-thread pool — the recursive panel partitioning
/// re-enters the parallel region without bound). The sequential path is
/// unaffected; the thread cap is restored afterwards. The dgetrs solves are
/// not recursive and stay multi-threaded.
pub fn getrf(m: usize, n: usize, a: &mut [f64]) -> Option<Vec<i32>> {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        #[cfg(not(target_os = "macos"))]
        {
            let prev = BLAS_THREAD_CAP.load(Ordering::Relaxed);
            set_blas_threads(Some(1));
            let r = platform::getrf(m, n, a);
            set_blas_threads(Some(prev));
            return r;
        }
        #[cfg(target_os = "macos")]
        {
            return platform::getrf(m, n, a);
        }
    }
    None
}

/// Solve against a [`getrf`] factorization (`trans = true` solves `Aᵀ x = b`).
/// Only meaningful after a successful `getrf` of the same matrix.
pub fn getrs(n: usize, a: &[f64], ipiv: &[i32], b: &mut [f64], trans: bool) {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        platform::getrs(n, a, ipiv, b, trans);
    }
}

pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
    alpha: f64,
    beta: f64,
    trans_a: bool,
    trans_b: bool,
) {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        return platform::gemm(
            m, n, k, a, lda, b, ldb, c, ldc, alpha, beta, trans_a, trans_b,
        );
    }
    faer_backend::gemm(
        m, n, k, a, lda, b, ldb, c, ldc, alpha, beta, trans_a, trans_b,
    );
}

/// y = α·A·x + β·y  where A is m×n row-major.
/// On macOS, dispatches to Accelerate's cblas_dgemv (SIMD + AMX).
/// Falls back to a simple row-major loop on platforms without BLAS.
#[inline]
#[allow(unused_variables)]
pub fn gemv(
    m: usize,
    n: usize,
    a: &[f64],
    lda: usize,
    x: &[f64],
    y: &mut [f64],
    alpha: f64,
    beta: f64,
    trans: bool,
) {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        return platform::gemv(m, n, a, lda, x, y, alpha, beta, trans);
    }
    // Pure-Rust fallback
    if trans {
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..m {
                s += a[i * lda + j] * x[i];
            }
            y[j] = alpha * s + beta * y[j];
        }
    } else {
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..n {
                s += a[i * lda + j] * x[j];
            }
            y[i] = alpha * s + beta * y[i];
        }
    }
}

/// Shorthand: y = A·x (no scaling, fresh output), lda=n
pub fn gemv_short(m: usize, n: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; m];
    gemv(m, n, a, n, x, &mut y, 1.0, 0.0, false);
    y
}

/// BLAS-accelerated dense matvec: y = A·x. Returns None if BLAS disabled.
/// Generic over T: zero-copy when T=f64, converts otherwise.
#[inline]
pub fn dense_matvec<T>(m: usize, n: usize, a: &[T], x: &[T]) -> Option<Vec<T>>
where
    T: num_traits::Float + num_traits::ToPrimitive + num_traits::FromPrimitive + 'static,
{
    if !blas_enabled() {
        return None;
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let a64 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let x64 = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let y64 = gemv_short(m, n, a64, x64);
        Some(unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(y64) })
    } else {
        let af: Vec<f64> = a.iter().map(|&v| v.to_f64().expect("finite scalar")).collect();
        let xf: Vec<f64> = x.iter().map(|&v| v.to_f64().expect("finite scalar")).collect();
        let yf = gemv_short(m, n, &af, &xf);
        Some(yf.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect())
    }
}

/// BLAS-accelerated dense matvec transpose: y = Aᵀ·x. Generic over T.
#[inline]
pub fn dense_matvec_t<T>(m: usize, n: usize, a: &[T], x: &[T]) -> Option<Vec<T>>
where
    T: num_traits::Float + num_traits::ToPrimitive + num_traits::FromPrimitive + 'static,
{
    if !blas_enabled() {
        return None;
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        let a64 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let x64 = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let mut y64 = vec![0.0f64; n];
        gemv(m, n, a64, n, x64, &mut y64, 1.0, 0.0, true);
        Some(unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(y64) })
    } else {
        let af: Vec<f64> = a.iter().map(|&v| v.to_f64().expect("finite scalar")).collect();
        let xf: Vec<f64> = x.iter().map(|&v| v.to_f64().expect("finite scalar")).collect();
        let mut yf = vec![0.0f64; n];
        gemv(m, n, &af, n, &xf, &mut yf, 1.0, 0.0, true);
        Some(yf.iter().map(|&v| T::from_f64(v).expect("scalar literal")).collect())
    }
}

// Keep old names for backward compatibility.
#[inline]
pub fn dense_matvec_f64(m: usize, n: usize, a: &[f64], x: &[f64]) -> Option<Vec<f64>> {
    dense_matvec::<f64>(m, n, a, x)
}
#[inline]
pub fn dense_matvec_t_f64(m: usize, n: usize, a: &[f64], x: &[f64]) -> Option<Vec<f64>> {
    dense_matvec_t::<f64>(m, n, a, x)
}

#[inline]
#[allow(unused_variables)]
pub fn dsyrk(
    n: usize,
    g: usize,
    a: &[f64],
    lda: usize,
    c: &mut [f64],
    ldc: usize,
    alpha: f64,
    beta: f64,
) {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        return platform::dsyrk(n, g, a, lda, c, ldc, alpha, beta);
    }
    faer_backend::dsyrk(n, g, a, lda, c, ldc, alpha, beta);
}

#[inline]
#[allow(unused_variables)]
pub fn dpotrf(n: usize, a: &mut [f64]) -> bool {
    dpotrf_ul(n, a, b'L')
}

/// `dpotrf` with an explicit triangle: the row-major data's lower triangle
/// corresponds to LAPACK's `'U'` (and vice versa), so a caller whose matrix
/// has only one triangle populated (e.g. the condensed H, whose gram fills
/// the lower) must pass `'U'` — the default `'L'` would factor the
/// unpopulated upper half.
pub fn dpotrf_ul(n: usize, a: &mut [f64], uplo: u8) -> bool {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        return platform::dpotrf_ul(n, a, uplo);
    }
    faer_backend::dpotrf_ul(n, a, uplo)
}

#[inline]
#[allow(unused_variables)]
pub fn dpotrs(n: usize, a: &[f64], b: &mut [f64], _nrhs: usize) {
    dpotrs_ul(n, a, b, 1, b'L')
}

/// `dpotrs` with an explicit triangle — must match the `dpotrf_ul` call's.
#[inline]
#[allow(unused_variables)]
pub fn dpotrs_ul(n: usize, a: &[f64], b: &mut [f64], _nrhs: usize, uplo: u8) {
    #[cfg(any(target_os = "macos", blas_system))]
    if blas_enabled() {
        return platform::dpotrs_ul(n, a, b, 1, uplo);
    }
    faer_backend::dpotrs_ul(n, a, b, 1, uplo)
}

// ---------------------------------------------------------------------------
// faer fallback (all other platforms)
// ---------------------------------------------------------------------------

// Faer backend — always compiled (runtime dispatch chooses between
// this and the platform BLAS backend when BLAS is available).
mod faer_backend {
    use faer::linalg::matmul::matmul;
    use faer::mat::{MatMut, MatRef};
    use faer::{Accum, Mat, Side};

    pub fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        _lda: usize,
        b: &[f64],
        _ldb: usize,
        c: &mut [f64],
        ldc: usize,
        alpha: f64,
        beta: f64,
        trans_a: bool,
        trans_b: bool,
    ) {
        let (ar, ac) = if trans_a { (k, m) } else { (m, k) };
        let a_view = MatRef::from_row_major_slice(a, ar, ac);
        let (br, bc) = if trans_b { (n, k) } else { (k, n) };
        let b_view = MatRef::from_row_major_slice(b, br, bc);

        let a_t = if trans_a { a_view.transpose() } else { a_view };
        let b_t = if trans_b { b_view.transpose() } else { b_view };

        if beta == 0.0 {
            let c_view = MatMut::from_row_major_slice_mut(c, m, n);
            matmul(c_view, Accum::Replace, a_t, b_t, alpha, faer::Par::Seq);
        } else {
            for i in 0..m {
                for j in 0..n {
                    c[i * ldc + j] *= beta;
                }
            }
            let c_view = MatMut::from_row_major_slice_mut(c, m, n);
            matmul(c_view, Accum::Add, a_t, b_t, alpha, faer::Par::Seq);
        }
    }

    pub fn dsyrk(
        n: usize,
        g: usize,
        a: &[f64],
        _lda: usize,
        c: &mut [f64],
        ldc: usize,
        alpha: f64,
        beta: f64,
    ) {
        // C = α·A·Aᵀ + β·C where A is n×g row-major (Bᵀ).
        // Match BLAS convention (NO_TRANS): fill only the lower triangle.
        let a_view = MatRef::from_row_major_slice(a, n, g);
        if beta != 1.0 {
            // Scale the *lower* triangle only — the upper one is zeroed below, and
            // this loop is the caller's only "clear C" when beta=0. A buffer reused
            // across calls (the IPM's dsyrk_gram) accumulates instead of replacing
            // if the wrong half is scaled; silent on freshly-zeroed test buffers.
            for i in 0..n {
                for j in 0..=i {
                    c[i * ldc + j] *= beta;
                }
            }
        }
        let c_view = MatMut::from_row_major_slice_mut(c, n, n);
        matmul(
            c_view,
            Accum::Add,
            a_view,
            a_view.transpose(),
            alpha,
            faer::Par::Seq,
        );
        // faer matmul fills the full matrix; zero the strict upper triangle
        // to match the BLAS dsyrk convention (callers mirror when needed).
        for i in 0..n {
            for j in (i + 1)..n {
                c[i * ldc + j] = 0.0;
            }
        }
    }

    /// Triangle conventions mirror `platform::dpotrf_ul`: `'U'` means the
    /// row-major LOWER triangle is populated (the condensed H's gram fills
    /// row-major lower), `'L'` means column-major lower. The factor is
    /// written back into the same triangle that was read.
    pub fn dpotrf_ul(n: usize, a: &mut [f64], uplo: u8) -> bool {
        let row_major_lower = uplo == b'U';
        // Read the populated triangle symmetrically into faer's matrix.
        let m = if row_major_lower {
            Mat::from_fn(
                n,
                n,
                |i, j| if i >= j { a[i * n + j] } else { a[j * n + i] },
            )
        } else {
            Mat::from_fn(
                n,
                n,
                |i, j| if i >= j { a[i + j * n] } else { a[j + i * n] },
            )
        };
        match m.llt(Side::Lower) {
            Ok(llt) => {
                let l = llt.L();
                if row_major_lower {
                    for i in 0..n {
                        for j in 0..=i {
                            a[i * n + j] = l[(i, j)];
                        }
                    }
                } else {
                    for j in 0..n {
                        for i in j..n {
                            a[i + j * n] = l[(i, j)];
                        }
                    }
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Must be called with the same `uplo` as the matching `dpotrf_ul` call.
    pub fn dpotrs_ul(n: usize, a: &[f64], b: &mut [f64], _nrhs: usize, uplo: u8) {
        // Solve A x = L Lᵀ x = b: forward substitution (L y = b), then back
        // substitution (Lᵀ x = y). The index expression follows the packed
        // triangle: row-major lower (`a[i*n+j]`) or column-major lower
        // (`a[i + j*n]`).
        let at = |i: usize, j: usize| -> f64 {
            if uplo == b'U' { a[i * n + j] } else { a[i + j * n] }
        };
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= at(i, k) * y[k];
            }
            y[i] = s / at(i, i);
        }
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= at(k, i) * x[k]; // Lᵀ[i,k] = L[k,i]
            }
            x[i] = s / at(i, i);
        }
        b[..n].copy_from_slice(&x[..n]);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_small() {
        // C = A * B: A 3×2, B 2×4 → C 3×4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 12];
        gemm(3, 4, 2, &a, 2, &b, 4, &mut c, 4, 1.0, 0.0, false, false);
        let expected = [
            11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0,
        ];
        for i in 0..12 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-12,
                "gemm mismatch at {i}: {} vs {}",
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn gemm_rectangular() {
        // C = A * B: A 2×5, B 5×3 → C 2×3 (non-square test)
        let a: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect(); // 2×5
        let b: Vec<f64> = (0..15).map(|i| (i + 1) as f64).collect(); // 5×3
        let mut c = vec![0.0; 6];
        gemm(2, 3, 5, &a, 5, &b, 3, &mut c, 3, 1.0, 0.0, false, false);
        // Reference: compute by hand
        // A = [[1,2,3,4,5],[6,7,8,9,10]]
        // B = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
        // C[0,0] = 1+8+21+40+65 = 135
        // C[0,1] = 2+10+24+44+70 = 150
        // C[0,2] = 3+12+27+48+75 = 165
        // C[1,0] = 6+28+56+90+130 = 310
        // C[1,1] = 12+35+64+99+140 = 350
        // C[1,2] = 18+42+72+108+150 = 390
        let expected = [135.0, 150.0, 165.0, 310.0, 350.0, 390.0];
        for i in 0..6 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-10,
                "gemm_rect mismatch at {i}: {} vs {}",
                c[i],
                expected[i]
            );
        }
    }

    #[test]
    fn syrk_ata() {
        // dsyrk computes C = B·Bᵀ where B is n×g.  To get C = Aᵀ·A (A is g×n),
        // pass B = Aᵀ (n×g).  A original is 4×3 → B = Aᵀ is 3×4.
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]; // 4×3
        let n = 3;
        let g = 4;
        let mut bt = vec![0.0; n * g]; // B = Aᵀ, 3×4 row-major
        for i in 0..g {
            for j in 0..n {
                bt[j * g + i] = a[i * n + j];
            }
        }
        let mut c = vec![0.0; 9];
        dsyrk(n, g, &bt, g, &mut c, n, 1.0, 0.0); // B is n×g, lda=g, C is n×n
                                                  // C[0,0]=166, C[1,0]=188, C[2,0]=210, C[1,1]=214, C[2,1]=240, C[2,2]=270
        let lower = vec![166.0, 0.0, 0.0, 188.0, 214.0, 0.0, 210.0, 240.0, 270.0];
        for i in 0..n {
            for j in 0..n {
                if i >= j {
                    let idx = i * n + j;
                    assert!(
                        (c[idx] - lower[idx]).abs() < 1e-10,
                        "syrk mismatch at ({i},{j}): {} vs {}",
                        c[idx],
                        lower[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn syrk_wide() {
        // dsyrk computes C = B·Bᵀ where B is n×g.  A is 2×5, so B = Aᵀ is 5×2.
        let a: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect(); // 2×5
        let n = 5;
        let g = 2;
        let mut bt = vec![0.0; n * g]; // B = Aᵀ, 5×2 row-major
        for i in 0..g {
            for j in 0..n {
                bt[j * g + i] = a[i * n + j];
            }
        }
        let mut c = vec![0.0; 25];
        dsyrk(n, g, &bt, g, &mut c, n, 1.0, 0.0); // B is 5×2, lda=2
                                                  // C[0,0] = 1+36 = 37, C[0,1] = 2+42 = 44
        assert!((c[0] - 37.0).abs() < 1e-10, "C[0,0]");
        assert!(c[1].abs() < 1e-10, "C[0,1] should be 0");
        assert!((c[5] - 44.0).abs() < 1e-10, "C[1,0]");
        assert!((c[6] - 53.0).abs() < 1e-10, "C[1,1]");
    }

    #[test]
    fn dpotrf_solves() {
        // SPD 3×3, column-major lower triangle.
        let mut a = vec![4.0, 2.0, 1.0, 0.0, 6.0, 3.0, 0.0, 0.0, 5.0];
        let n = 3;
        assert!(dpotrf(n, &mut a));
        let mut b = vec![1.0, 2.0, 3.0];
        dpotrs(n, &a, &mut b, 1);
        // A*x should = [1,2,3]
        let ax = [
            4.0 * b[0] + 2.0 * b[1] + 1.0 * b[2],
            2.0 * b[0] + 6.0 * b[1] + 3.0 * b[2],
            1.0 * b[0] + 3.0 * b[1] + 5.0 * b[2],
        ];
        for i in 0..3 {
            let expected = [1.0, 2.0, 3.0][i];
            assert!(
                (ax[i] - expected).abs() < 1e-12,
                "dpotrs residual at {i}: {} vs {}",
                ax[i],
                expected
            );
        }
    }

    #[test]
    fn dpotrf_non_pd() {
        if !blas_enabled() {
            return;
        }
        let mut a = vec![-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(!dpotrf(3, &mut a));
    }
}
