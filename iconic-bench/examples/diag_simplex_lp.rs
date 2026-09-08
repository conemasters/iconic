//! Replay a node LP that the dual simplex reported infeasible, and check that verdict
//! independently.
//!
//! Reads the dump written by `ICONIC_DUMP_FAILED_LP` (path in argv[1]) and reports:
//!   - what `DualSolver::cold_solve` says,
//!   - the structure of the LP (how many columns are fixed, i.e. `l == u`),
//!   - whether the LP is *actually* feasible, via the IPM through iconic-api.

use iconic_simplex::{DualSolver, Status as DsStatus};

/// `(c, a, b, l, u, m, n)` — objective, row-major matrix, rhs, bounds, and dimensions.
type Lp = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
);

fn parse(path: &str) -> Lp {
    let txt = std::fs::read_to_string(path).expect("read dump");
    let mut m = 0usize;
    let mut n = 0usize;
    let (mut c, mut b, mut l, mut u, mut a) = (vec![], vec![], vec![], vec![], vec![]);
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        let key = it.next().unwrap_or("");
        let vals: Vec<f64> = it.filter_map(|t| t.parse().ok()).collect();
        match key {
            "m" => m = vals[0] as usize,
            "n" => n = vals[0] as usize,
            "c" => c = vals,
            "b" => b = vals,
            "l" => l = vals,
            "u" => u = vals,
            "a" => a = vals,
            _ => {}
        }
    }
    (c, a, b, l, u, m, n)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: diag_simplex_lp <dump>");
    let (c, a, b, l, u, m, n) = parse(&path);
    println!(
        "LP: m={m} n={n}  (a has {} entries, expected {})",
        a.len(),
        m * n
    );

    let fixed: Vec<usize> = (0..n).filter(|&j| (u[j] - l[j]).abs() < 1e-12).collect();
    let free_cols = (0..n).filter(|&j| l[j] < -1e19 && u[j] > 1e19).count();
    println!(
        "columns: {} fixed (l == u), {} free, {} ordinary",
        fixed.len(),
        free_cols,
        n - fixed.len() - free_cols
    );
    println!("fixed column indices: {:?}", &fixed[..fixed.len().min(20)]);

    let mut s: DualSolver<f64> = DualSolver::new(&c, &a, &b, &l, &u, m, n);
    let sol = s.cold_solve();
    println!(
        "dual simplex: status={:?} iters={} obj={:.6}",
        sol.status, sol.iters, sol.obj
    );
    let nf: Vec<usize> = (0..sol.x.len())
        .filter(|&j| !sol.x[j].is_finite())
        .collect();
    println!(
        "  non-finite x entries: {} {:?}",
        nf.len(),
        &nf[..nf.len().min(8)]
    );
    let huge_x: Vec<usize> = (0..n.min(sol.x.len()))
        .filter(|&j| sol.x[j].is_finite() && sol.x[j].abs() > 1e15)
        .collect();
    println!(
        "  |x_j| > 1e15 at: {} {:?}",
        huge_x.len(),
        &huge_x[..huge_x.len().min(8)]
    );
    if nf.is_empty() {
        let mut worst = 0.0f64;
        for i in 0..m {
            let ax: f64 = (0..n).map(|j| a[i * n + j] * sol.x[j]).sum();
            worst = worst.max((ax - b[i]).abs());
        }
        println!("  max |Ax-b| = {worst:.3e}");
    }

    // Independent verdict: is a feasible point findable at all? Solve the same LP as a
    // cone program (equalities Ax = b plus the box) with the IPM.
    let mut ap = iconic_linalg::DenseMatrix::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            ap.set(i, j, a[i * n + j]);
        }
    }
    let mut lb_rows = 0usize;
    for j in 0..n {
        if l[j] > -1e19 {
            lb_rows += 1;
        }
        if u[j] < 1e19 {
            lb_rows += 1;
        }
    }
    let mut big = iconic_linalg::DenseMatrix::<f64>::zeros(m + lb_rows, n);
    let mut bb = vec![0.0; m + lb_rows];
    for i in 0..m {
        for j in 0..n {
            big.set(i, j, ap.get(i, j));
        }
        bb[i] = b[i];
    }
    let mut r = m;
    for j in 0..n {
        if l[j] > -1e19 {
            big.set(r, j, -1.0);
            bb[r] = -l[j];
            r += 1;
        }
        if u[j] < 1e19 {
            big.set(r, j, 1.0);
            bb[r] = u[j];
            r += 1;
        }
    }
    let prog = iconic_api::ConeProgram {
        p: iconic_linalg::DenseMatrix::zeros(n, n),
        q: c.clone(),
        a: big,
        a_csc: None,
        b: bb,
        cones: vec![
            iconic_core::Cone::Zero(m),
            iconic_core::Cone::NonNegative(lb_rows),
        ],
    };
    match iconic_api::solve(&prog, &iconic_core::Settings::<f64>::default()) {
        Ok(s) => println!("IPM verdict: status={:?} obj={:.6}", s.status, s.obj_val),
        Err(e) => println!("IPM verdict: error {e:?}"),
    }

    if matches!(sol.status, DsStatus::Infeasible) {
        println!("\n>>> simplex says Infeasible; compare against the IPM verdict above.");
    }
}
