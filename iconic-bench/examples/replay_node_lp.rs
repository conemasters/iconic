//! Re-solve a node LP dumped by `ICONIC_DUMP_NODE_LPS`, reporting what the dual simplex
//! says from a cold start.
//!
//! The dump records the value the solver used as that node's bound. If a cold solve of the
//! same LP disagrees, the bound came from the warm-start path rather than from the LP.
use iconic_simplex::{CscCols, DualSolver, Status as DsStatus};

fn main() {
    let path = std::env::args().nth(1).expect("dump file");
    let txt = std::fs::read_to_string(&path).expect("read");
    let (mut m, mut n) = (0usize, 0usize);
    let mut iconic_obj = f64::NAN;
    let (mut c, mut b, mut l, mut u) = (vec![], vec![], vec![], vec![]);
    let mut trips: Vec<(usize, usize, f64)> = Vec::new();
    let mut in_trips = false;
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        let key = it.next().unwrap_or("");
        if in_trips {
            let v: Vec<&str> = std::iter::once(key).chain(it).collect();
            if v.len() == 3 {
                trips.push((
                    v[0].parse().unwrap(),
                    v[1].parse().unwrap(),
                    v[2].parse().unwrap(),
                ));
            }
            continue;
        }
        let vals: Vec<f64> = it.filter_map(|t| t.parse().ok()).collect();
        match key {
            "m" => m = vals[0] as usize,
            "n" => n = vals[0] as usize,
            "iconic_obj" => iconic_obj = vals[0],
            "c" => c = vals,
            "b" => b = vals,
            "l" => l = vals,
            "u" => u = vals,
            "triplets" => in_trips = true,
            _ => {}
        }
    }
    // Rebuild CSC from the triplets (already grouped by column, rows ascending).
    let mut col_start = vec![0usize; n + 1];
    for &(_, j, _) in &trips {
        col_start[j + 1] += 1;
    }
    for j in 0..n {
        col_start[j + 1] += col_start[j];
    }
    let mut row_idx = vec![0usize; trips.len()];
    let mut val = vec![0.0f64; trips.len()];
    let mut fill = col_start[..n].to_vec();
    for &(i, j, v) in &trips {
        row_idx[fill[j]] = i;
        val[fill[j]] = v;
        fill[j] += 1;
    }

    let a = CscCols {
        col_start,
        row_idx,
        val,
    };
    let mut s: DualSolver<f64> = DualSolver::from_csc(c.clone(), a, b, l, u, m, n);
    let sol = s.cold_solve();
    let obj: f64 = c.iter().zip(&sol.x).map(|(cj, xj)| cj * xj).sum();
    println!("m={m} n={n}  dump_obj={iconic_obj:.10}  cold_solve: status={:?} reported={:.10} c.x={:.10} iters={}",
        sol.status, sol.obj, obj, sol.iters);
    if matches!(sol.status, DsStatus::Optimal)
        && (sol.obj - iconic_obj).abs() > 1e-6 * iconic_obj.abs().max(1.0)
    {
        println!("  DISAGREEMENT: cold solve and the value used as the node bound differ");
    }
}
