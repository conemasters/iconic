//! Profile the conic's per-iteration phases (env ICONIC_CONIC_PROF=1).
use iconic_core::Settings;
use iconic_ipm::conic::{solve_cone_qp, Cone};

fn main() {
    let settings = Settings::<f64>::default();
    // The CVXPY-style boxed SOCP: [nonneg n, soc 36] with the 86-row A.
    let n = 50usize;
    let mut st = 0u64;
    let mut nxt = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 11) as f64 / ((1u64 << 53) as f64)
    };
    let mut a = iconic_linalg::DenseMatrix::zeros(36 + 2 * n, n);
    for r in 0..36 {
        for j in 0..n {
            a.set(r, j, 2.0 * nxt() - 1.0);
        }
    }
    for j in 0..n {
        a.set(36 + j, j, 1.0);
        a.set(36 + n + j, j, -1.0);
    }
    let q: Vec<f64> = (0..n).map(|_| 2.0 * nxt() - 1.0).collect();
    let b = vec![1.0; 36 + 2 * n];
    let prob = iconic_ipm::QpProblem {
        p: iconic_linalg::DenseMatrix::zeros(n, n),
        q,
        a_eq: iconic_linalg::DenseMatrix::zeros(0, n),
        b_eq: vec![],
        a_in: a,
        b_in: b,
        a_eq_csr: None,
        a_in_csr: None,
    };
    let cones = vec![
        iconic_ipm::conic::Cone::Soc(36),
        iconic_ipm::conic::Cone::NonNeg(2 * n),
    ];
    let t = std::time::Instant::now();
    let sol = solve_cone_qp(&prob, &cones, &settings);
    eprintln!(
        "[diag] native boxed SOCP: wall {:.3}ms iters={}",
        t.elapsed().as_secs_f64() * 1e3,
        sol.iters
    );
    // The api's solve (the dense ABI's entry): the same shape via the ConeProgram.
    use iconic_api::ConeProgram;
    let me = 0usize;
    let mi = 36 + 2 * n;
    let mut ap = iconic_linalg::DenseMatrix::zeros(me + mi, n);
    for r in 0..mi {
        for j in 0..n {
            ap.set(me + r, j, prob.a_in.get(r, j));
        }
    }
    let bv = prob.b_in.clone();
    let mut cs = vec![iconic_core::Cone::Zero(me)];
    cs.push(iconic_core::Cone::SecondOrder(36));
    cs.push(iconic_core::Cone::NonNegative(2 * n));
    let prog = ConeProgram {
        p: prob.p.clone(),
        q: prob.q.clone(),
        a: ap,
        a_csc: None,
        b: bv,
        cones: cs,
    };
    let mut api_s = settings.clone();
    api_s.max_iters = 800;
    let t = std::time::Instant::now();
    let sol2 = iconic_api::solve(&prog, &api_s).unwrap();
    eprintln!(
        "[diag] api boxed SOCP: wall {:.3}ms iters={} status={:?}",
        t.elapsed().as_secs_f64() * 1e3,
        sol2.iters,
        sol2.status
    );
    // The C ABI's exact settings: max_iters 800 + presolve flag.
    let mut c_s = settings.clone();
    c_s.max_iters = 800;
    c_s.presolve = true;
    let t = std::time::Instant::now();
    let sol3 = iconic_api::solve(&prog, &c_s).unwrap();
    eprintln!(
        "[diag] c-settings boxed SOCP: wall {:.3}ms iters={} status={:?}",
        t.elapsed().as_secs_f64() * 1e3,
        sol3.iters,
        sol3.status
    );
    // The REAL CVXPY shape: n=51, A 102x51, [nonneg 51, soc 51].
    let n51 = 51usize;
    let mut st51 = 7u64;
    let mut nx51 = move || {
        st51 = st51
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st51 >> 11) as f64 / ((1u64 << 53) as f64)
    };
    let mut a51 = iconic_linalg::DenseMatrix::zeros(102, n51);
    for r in 0..51 {
        for j in 0..n51 {
            a51.set(r, j, 2.0 * nx51() - 1.0);
        }
    }
    for j in 0..n51 {
        a51.set(51 + j, j, 1.0);
    }
    let q51: Vec<f64> = (0..n51).map(|_| 2.0 * nx51() - 1.0).collect();
    let b51 = vec![1.0; 102];
    let prob51 = iconic_ipm::QpProblem {
        p: iconic_linalg::DenseMatrix::zeros(n51, n51),
        q: q51,
        a_eq: iconic_linalg::DenseMatrix::zeros(0, n51),
        b_eq: vec![],
        a_in: a51,
        b_in: b51,
        a_eq_csr: None,
        a_in_csr: None,
    };
    let cones51 = vec![
        iconic_ipm::conic::Cone::NonNeg(51),
        iconic_ipm::conic::Cone::Soc(51),
    ];
    let t = std::time::Instant::now();
    let sol51 = solve_cone_qp(&prob51, &cones51, &settings);
    eprintln!(
        "[diag] real-shape native: wall {:.3}ms iters={} status={:?}",
        t.elapsed().as_secs_f64() * 1e3,
        sol51.iters,
        sol51.status
    );
    iconic_linalg::blas::set_blas_enabled(false);
    let t = std::time::Instant::now();
    let sol52 = solve_cone_qp(&prob51, &cones51, &settings);
    eprintln!(
        "[diag] real-shape native (no BLAS): wall {:.3}ms iters={} status={:?}",
        t.elapsed().as_secs_f64() * 1e3,
        sol52.iters,
        sol52.status
    );
    // The api's solve on the real shape (the C ABI's entry).
    let me51 = 0usize;
    let mi51 = 102usize;
    let mut ap51 = iconic_linalg::DenseMatrix::zeros(me51 + mi51, n51);
    for r in 0..mi51 {
        for j in 0..n51 {
            ap51.set(me51 + r, j, prob51.a_in.get(r, j));
        }
    }
    let mut cs51 = vec![iconic_core::Cone::Zero(me51)];
    cs51.push(iconic_core::Cone::NonNegative(51));
    cs51.push(iconic_core::Cone::SecondOrder(51));
    let prog51 = iconic_api::ConeProgram {
        p: prob51.p.clone(),
        q: prob51.q.clone(),
        a: ap51,
        a_csc: None,
        b: prob51.b_in.clone(),
        cones: cs51,
    };
    let t = std::time::Instant::now();
    let sol53 = iconic_api::solve(&prog51, &settings).unwrap();
    eprintln!(
        "[diag] real-shape api: wall {:.3}ms iters={} status={:?}",
        t.elapsed().as_secs_f64() * 1e3,
        sol53.iters,
        sol53.status
    );
    // The C ABI's exact settings.
    let mut c51 = settings.clone();
    c51.max_iters = 800;
    c51.presolve = true;
    let t = std::time::Instant::now();
    let sol54 = iconic_api::solve(&prog51, &c51).unwrap();
    eprintln!(
        "[diag] real-shape api C-settings: wall {:.3}ms iters={} status={:?}",
        t.elapsed().as_secs_f64() * 1e3,
        sol54.iters,
        sol54.status
    );
    // The api's solve on the EXACT real data (the C ABI's full path).
    if std::path::Path::new("/tmp/real_A.bin").exists() {
        let a_raw = std::fs::read("/tmp/real_A.bin").unwrap();
        let b_raw = std::fs::read("/tmp/real_b.bin").unwrap();
        let q_raw = std::fs::read("/tmp/real_q.bin").unwrap();
        let rd = |buf: &[u8]| -> Vec<f64> {
            buf.chunks(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let a_real = rd(&a_raw);
        let b_real = rd(&b_raw);
        let q_real = rd(&q_raw);
        let nr = b_real.len();
        let nc = q_real.len();
        let mut ar = iconic_linalg::DenseMatrix::zeros(nr, nc);
        for r in 0..nr {
            for j in 0..nc {
                ar.set(r, j, a_real[r * nc + j]);
            }
        }
        let prog_real = iconic_api::ConeProgram {
            p: iconic_linalg::DenseMatrix::zeros(nc, nc),
            q: q_real,
            a: ar,
            a_csc: None,
            b: b_real,
            cones: vec![
                iconic_core::Cone::NonNegative(51),
                iconic_core::Cone::SecondOrder(51),
            ],
        };
        let mut cr = settings.clone();
        cr.max_iters = 800;
        let t = std::time::Instant::now();
        let sol_real_api = iconic_api::solve(&prog_real, &cr).unwrap();
        eprintln!(
            "[diag] EXACT real api: wall {:.3}ms iters={} status={:?}",
            t.elapsed().as_secs_f64() * 1e3,
            sol_real_api.iters,
            sol_real_api.status
        );
    }
    // The EXACT real data with BLAS off (the wheel's configuration).
    if std::path::Path::new("/tmp/real_A.bin").exists() {
        iconic_linalg::blas::set_blas_enabled(false);
        let a_raw = std::fs::read("/tmp/real_A.bin").unwrap();
        let b_raw = std::fs::read("/tmp/real_b.bin").unwrap();
        let q_raw = std::fs::read("/tmp/real_q.bin").unwrap();
        let rd = |buf: &[u8]| -> Vec<f64> {
            buf.chunks(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let a_real = rd(&a_raw);
        let b_real = rd(&b_raw);
        let q_real = rd(&q_raw);
        let nr = b_real.len();
        let nc = q_real.len();
        let mut ar = iconic_linalg::DenseMatrix::zeros(nr, nc);
        for r in 0..nr {
            for j in 0..nc {
                ar.set(r, j, a_real[r * nc + j]);
            }
        }
        let prob_real = iconic_ipm::QpProblem {
            p: iconic_linalg::DenseMatrix::zeros(nc, nc),
            q: q_real,
            a_eq: iconic_linalg::DenseMatrix::zeros(0, nc),
            b_eq: vec![],
            a_in: ar,
            b_in: b_real,
            a_eq_csr: None,
            a_in_csr: None,
        };
        let cones_real = vec![
            iconic_ipm::conic::Cone::NonNeg(51),
            iconic_ipm::conic::Cone::Soc(51),
        ];
        let t = std::time::Instant::now();
        let sol_real_nb = solve_cone_qp(&prob_real, &cones_real, &settings);
        eprintln!(
            "[diag] EXACT real no-BLAS: wall {:.3}ms iters={} status={:?}",
            t.elapsed().as_secs_f64() * 1e3,
            sol_real_nb.iters,
            sol_real_nb.status
        );
    }
    // The EXACT real CVXPY data (raw f64 dumps).
    if std::path::Path::new("/tmp/real_A.bin").exists() {
        let a_raw = std::fs::read("/tmp/real_A.bin").unwrap();
        let b_raw = std::fs::read("/tmp/real_b.bin").unwrap();
        let q_raw = std::fs::read("/tmp/real_q.bin").unwrap();
        let rd = |buf: &[u8]| -> Vec<f64> {
            buf.chunks(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let a_real = rd(&a_raw);
        let b_real = rd(&b_raw);
        let q_real = rd(&q_raw);
        let nr = b_real.len();
        let nc = q_real.len();
        let mut ar = iconic_linalg::DenseMatrix::zeros(nr, nc);
        for r in 0..nr {
            for j in 0..nc {
                ar.set(r, j, a_real[r * nc + j]);
            }
        }
        let prob_real = iconic_ipm::QpProblem {
            p: iconic_linalg::DenseMatrix::zeros(nc, nc),
            q: q_real,
            a_eq: iconic_linalg::DenseMatrix::zeros(0, nc),
            b_eq: vec![],
            a_in: ar,
            b_in: b_real,
            a_eq_csr: None,
            a_in_csr: None,
        };
        let cones_real = vec![
            iconic_ipm::conic::Cone::NonNeg(51),
            iconic_ipm::conic::Cone::Soc(51),
        ];
        let t = std::time::Instant::now();
        let sol_real = solve_cone_qp(&prob_real, &cones_real, &settings);
        eprintln!(
            "[diag] EXACT real native: wall {:.3}ms iters={} status={:?}",
            t.elapsed().as_secs_f64() * 1e3,
            sol_real.iters,
            sol_real.status
        );
        let mut s_real = settings.clone();
        s_real.max_iters = 800;
        let t = std::time::Instant::now();
        let sol_real2 = solve_cone_qp(&prob_real, &cones_real, &s_real);
        eprintln!(
            "[diag] EXACT real native 800: wall {:.3}ms iters={} status={:?}",
            t.elapsed().as_secs_f64() * 1e3,
            sol_real2.iters,
            sol_real2.status
        );
    }
    println!("---");
    let (prob2, cones2) = {
        // Random SDP: min tr(CX) s.t. X in PSD, small.
        let k = 4usize;
        let mut rng = 5u64;
        let mut nxt = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / ((1u64 << 53) as f64)
        };
        let n = k * (k + 1) / 2;
        let mut p = iconic_linalg::DenseMatrix::zeros(n, n);
        for i in 0..n {
            p.set(i, i, 1.0);
        }
        let q: Vec<f64> = (0..n).map(|_| 2.0 * nxt() - 1.0).collect();
        let a = iconic_linalg::DenseMatrix::zeros(n, n);
        let b = vec![0.0; n];
        (
            iconic_ipm::QpProblem {
                p,
                q,
                a_eq: iconic_linalg::DenseMatrix::zeros(0, n),
                b_eq: vec![],
                a_in: a,
                b_in: b,
                a_eq_csr: None,
                a_in_csr: None,
            },
            vec![Cone::Psd(k)],
        )
    };
    let _ = solve_cone_qp(&prob2, &cones2, &settings);
}
