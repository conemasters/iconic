//! Direct A/B: OpenBLAS dsytrf vs faer LDLᵀ on a 1280-dim quasidefinite KKT.
use iconic_linalg::blas;
use iconic_linalg::faer_dense::FaerLdlt;
use iconic_linalg::DenseMatrix;

fn main() {
    let dim = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1280);
    let _ = dim;
    // Quasidefinite KKT: [[I + rho, A^T],[A, -D]] with a dense random A.
    let dim2 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1280);
    let n = (dim2 * 2) / 3;
    let m = dim2 - n;
    let mut st = 42u64;
    let mut nxt = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
    };
    let mut a = DenseMatrix::zeros(m, n);
    for r in 0..m {
        for j in 0..n {
            a.set(r, j, nxt());
        }
    }
    let mut kkt = DenseMatrix::zeros(dim, dim);
    for i in 0..n {
        kkt.set(i, i, 1.0 + 1e-6);
    }
    for r in 0..m {
        for j in 0..n {
            let v = a.get(r, j);
            kkt.set(n + r, j, v);
            kkt.set(j, n + r, v);
        }
        kkt.set(n + r, n + r, -(1.0 + 1e-6));
    }

    // faer LDLᵀ (sequential).
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let f = FaerLdlt::factor(&kkt).unwrap();
        std::hint::black_box(&f);
    }
    let faer_ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;

    // OpenBLAS dsytrf at several thread caps: the large factors' sweet spot
    // vs the small ops' (the reference practice: automatic = measured
    // optimum, explicit override available).
    blas::set_blas_enabled(true);
    for cap in [1usize, 4, 8, 16, 24] {
        iconic_linalg::blas::set_blas_threads(Some(cap));
        let t = std::time::Instant::now();
        let mut last = None;
        for _ in 0..10 {
            let mut data = (*kkt.data).clone();
            let ipiv = blas::dsytrf(dim, &mut data).unwrap();
            last = Some(ipiv);
        }
        std::hint::black_box(&last);
        let ms = t.elapsed().as_secs_f64() * 1e3 / 10.0;
        println!("dsytrf threads={cap:2}: {ms:.1}ms");
    }
    println!("faer LDLᵀ: {faer_ms:.1}ms");

    // Cholesky A/B: faer's par_llt vs OpenBLAS dpotrf, on the (PD) negation
    // of the KKT's (2,2) block (the condensed path's shape).
    let mut neg = (*kkt.data).clone();
    for v in neg.iter_mut() {
        *v = -*v;
    }
    // Make it PD: shift the diagonal.
    let dimf = dim2;
    for i in 0..dimf {
        neg[i * dimf + i] += 100.0;
    }
    let t = std::time::Instant::now();
    for _ in 0..10 {
        let f = iconic_linalg::faer_dense::FaerLlt::factor_from(&DenseMatrix::from_row_major(
            dimf,
            dimf,
            neg.clone(),
        ))
        .unwrap();
        std::hint::black_box(&f);
    }
    let faer_llt = t.elapsed().as_secs_f64() * 1e3 / 10.0;
    let t = std::time::Instant::now();
    for _ in 0..10 {
        let mut d = neg.clone();
        let ok = blas::dpotrf(dimf, &mut d);
        std::hint::black_box(&ok);
    }
    let blas_llt = t.elapsed().as_secs_f64() * 1e3 / 10.0;
    println!(
        "Cholesky: faer par_llt {faer_llt:.1}ms  OpenBLAS dpotrf {blas_llt:.1}ms  speedup {:.2}x",
        faer_llt / blas_llt
    );

    // Correctness: H x = b via the BLAS dpotrf/dpotrs.
    let mut hh = neg.clone();
    let ok = blas::dpotrf(dimf, &mut hh);
    let mut b = vec![1.0f64; dimf];
    blas::dpotrs(dimf, &hh, &mut b, 1);
    // Check H x = b: compute max residual via the original matrix.
    let mut resid = 0.0f64;
    for i in 0..dimf {
        let mut acc = 0.0f64;
        for j in 0..dimf {
            acc += neg[i * dimf + j] * b[j];
        }
        let r = (acc - 1.0).abs();
        if r > resid {
            resid = r;
        }
    }
    println!("dpotrf correctness: ok={ok} max|Hx - b| = {resid:.2e}");

    // dsyrk gram A/B: faer vs OpenBLAS on the condensed gram (m x n dense).
    let (n_, m_) = (dim2 * 2 / 3, dim2 - dim2 * 2 / 3);
    let mut a_flat = vec![0.0f64; m_ * n_];
    let mut st2 = 99u64;
    for v in a_flat.iter_mut() {
        st2 = st2
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = (st2 >> 11) as f64 / ((1u64 << 53) as f64) * 2.0 - 1.0;
    }
    let mut c_faer = vec![0.0f64; m_ * m_];
    blas::set_blas_enabled(false);
    let t = std::time::Instant::now();
    for _ in 0..10 {
        blas::dsyrk(m_, n_, &a_flat, n_, &mut c_faer, m_, 1.0, 0.0);
    }
    let faer_syrk = t.elapsed().as_secs_f64() * 1e3 / 10.0;
    let mut c_blas = vec![0.0f64; m_ * m_];
    blas::set_blas_enabled(true);
    let t = std::time::Instant::now();
    for _ in 0..10 {
        blas::dsyrk(m_, n_, &a_flat, n_, &mut c_blas, m_, 1.0, 0.0);
    }
    let blas_syrk = t.elapsed().as_secs_f64() * 1e3 / 10.0;
    println!(
        "dsyrk gram: faer {faer_syrk:.1}ms  OpenBLAS {blas_syrk:.1}ms  speedup {:.2}x",
        faer_syrk / blas_syrk
    );
}
