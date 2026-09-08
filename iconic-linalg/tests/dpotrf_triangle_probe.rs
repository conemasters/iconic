// dpotrf_ul/dpotrs_ul must honor the triangle argument on every backend:
// 'U' = row-major lower populated, 'L' = column-major lower populated.
use iconic_linalg::blas;

fn check_factor(tag: &str) {
    let n = 4usize;
    // SPD matrix defined by its row-major lower triangle; upper holds sentinel garbage.
    let mut a = vec![0.0f64; n * n];
    let lower = [
        [ 4.0, 0.0, 0.0, 0.0],
        [ 2.0, 5.0, 0.0, 0.0],
        [ 1.0, 0.5, 6.0, 0.0],
        [ 0.5, 1.0, 2.0, 7.0],
    ];
    let expect = [
        [4.0,2.0,1.0,0.5],[2.0,5.0,0.5,1.0],[1.0,0.5,6.0,2.0],[0.5,1.0,2.0,7.0],
    ];
    for i in 0..n { for j in 0..=i { a[i*n+j] = lower[i][j]; } }
    for i in 0..n { for j in (i+1)..n { a[i*n+j] = 99.0; } }
    let ok = blas::dpotrf_ul(n, &mut a, b'U');
    assert!(ok, "[{tag}] factor failed");
    let l = |i: usize, j: usize| if i >= j { a[i*n+j] } else { 0.0 };
    for (i, row) in expect.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            let mut s = 0.0;
            for k in 0..n { s += l(i,k)*l(j,k); }
            assert!((s-e).abs() < 1e-12, "[{tag}] ({i},{j}): {s} vs {e}");
        }
    }

    // Solve with the factor still packed row-major lower.
    let rhs = [1.0, -2.0, 3.0, 0.5];
    let mut x = rhs;
    blas::dpotrs_ul(n, &a, &mut x, 1, b'U');
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n { s += expect[i][j] * x[j]; }
        assert!((s - rhs[i]).abs() < 1e-10, "[{tag}] solve row {i}");
    }
}

fn check_default_colmajor(tag: &str) {
    // Default 'L': column-major lower packed (fully-populated symmetric source).
    let n = 3usize;
    let sym = [[6.0, 2.0, 1.0], [2.0, 5.0, 0.5], [1.0, 0.5, 4.0]];
    let mut a = vec![0.0f64; n * n];
    for j in 0..n { for i in j..n { a[i + j*n] = sym[i][j]; } } // col-major lower
    assert!(blas::dpotrf_ul(n, &mut a, b'L'), "[{tag}] factor failed");
    let l = |i: usize, j: usize| -> f64 { if i >= j { a[i + j*n] } else { 0.0 } };
    for (i, row) in sym.iter().enumerate() {
        for (j, &e) in row.iter().enumerate() {
            let mut s = 0.0;
            for k in 0..n { s += l(i,k)*l(j,k); }
            assert!((s-e).abs() < 1e-12, "[{tag}] ({i},{j}): {s} vs {e}");
        }
    }
}

#[test]
fn platform_backend_row_major_lower() {
    check_factor("platform");
    check_default_colmajor("platform");
}

#[test]
fn faer_fallback_row_major_lower() {
    iconic_linalg::blas::set_blas_enabled(false);
    check_factor("faer");
    check_default_colmajor("faer");
}
