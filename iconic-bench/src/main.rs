//! Structured benchmark runner.
//!
//! Usage:
//!   cargo run -p iconic-bench --release                      # run + print report
//!   cargo run -p iconic-bench --release -- run --out a.jsonl # run, also save JSONL
//!   cargo run -p iconic-bench --release -- run --reps 10 --out a.jsonl
//!   cargo run -p iconic-bench --release -- run --mip --out a.jsonl   # + MIP suite
//!   cargo run -p iconic-bench --release -- mip --nodes 50000 --time 30
//!   cargo run -p iconic-bench --release -- compare a.jsonl b.jsonl   # diff two runs
//!   cargo run -p iconic-bench --release -- export-qp --out qp.json [--max-n 200]
//!   cargo run -p iconic-bench --release -- export-mip --out mip.json [--max-n 150]
//!
//! The suite spans real problem families (QP, LP, LASSO, NNLS, portfolio, SVM, Huber,
//! MPC, SOCP, SDP) and deliberately degenerate / ill-conditioned ones, each at several
//! sizes. `run` prints a grouped report plus per-category shifted-geometric-mean
//! summaries; with `--out` it also writes machine-readable JSONL. `compare` diffs two
//! JSONL runs and flags status downgrades, accuracy loss, and iteration/time
//! regressions. `export-qp`/`export-mip` dump the exact QP/LP or MIP problem instances
//! (dense arrays, JSON) so `scripts/bench_all_solvers.py` can drive every installed
//! CVXPY solver over the *same* instances without redefining them in Python.
//!
//! Unknown flags are rejected (usage + exit 2), never silently ignored: before, `run
//! --only foo` or a typo'd flag started the full suite anyway, and `mip 50000 30`
//! parsed *both* positionals as `--nodes`, the second overwriting the first.

use iconic_bench::{export, mip, suite};

fn usage(cmd: &str) -> String {
    match cmd {
        "export-qp" | "export-mip" => format!("usage: {cmd} --out <file.json> [--max-n N]"),
        "mip" => "usage: mip [--out FILE] [--nodes N] [--time S] [--only FILTER]".to_string(),
        "run" => {
            "usage: run [--reps N] [--out FILE] [--mip] [--mip-nodes N] [--mip-time S]".to_string()
        }
        "compare" => "usage: compare <baseline.jsonl> <current.jsonl>".to_string(),
        _ => "usage: iconic-bench <run|mip|compare|export-qp|export-mip> [flags]\n\
              (bare `iconic-bench` runs the full suite)"
            .to_string(),
    }
}

// ── Flag parsing (separated from main so the strictness is unit-testable) ──

struct RunArgs {
    reps: usize,
    out: Option<String>,
    run_mip: bool,
    mip_nodes: usize,
    mip_time: f64,
}

/// Parse `run`'s flags. Any unknown flag is an error: silently ignoring one used to
/// start the full suite on a typo'd or misplaced flag (`--only`, `--help`, ...).
fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut a = RunArgs {
        reps: 5,
        out: None,
        run_mip: false,
        mip_nodes: 10000,
        mip_time: 30.0,
    };
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--reps" => {
                a.reps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
                i += 2;
            }
            "--out" => {
                a.out = args.get(i + 1).cloned();
                i += 2;
            }
            "--mip" => {
                a.run_mip = true;
                i += 1;
            }
            "--mip-nodes" => {
                a.mip_nodes = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10000);
                i += 2;
            }
            "--mip-time" => {
                a.mip_time = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(30.0);
                i += 2;
            }
            other => return Err(format!("unknown flag for 'run': {other}")),
        }
    }
    Ok(a)
}

struct MipArgs {
    out: Option<String>,
    max_nodes: usize,
    max_time: f64,
    only: Option<String>,
}

/// Parse `mip`'s flags. Positionals are rejected: `mip 50000 30` used to parse both
/// as `max_nodes` (the second overwriting the first), silently capping ICONIC at 30
/// nodes, and any other stray token started the whole suite.
fn parse_mip_args(args: &[String]) -> Result<MipArgs, String> {
    let mut a = MipArgs {
        out: None,
        max_nodes: 10000,
        max_time: 30.0,
        only: None,
    };
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                a.out = args.get(i + 1).cloned();
                i += 2;
            }
            "--nodes" => {
                a.max_nodes = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10000);
                i += 2;
            }
            "--time" => {
                a.max_time = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(30.0);
                i += 2;
            }
            "--only" => {
                a.only = args.get(i + 1).cloned();
                i += 2;
            }
            other => return Err(format!("unknown flag for 'mip': {other}")),
        }
    }
    Ok(a)
}

struct ExportArgs {
    out: Option<String>,
    max_n: usize,
}

fn parse_export_args(cmd: &str, args: &[String]) -> Result<ExportArgs, String> {
    let mut a = ExportArgs {
        out: None,
        max_n: if cmd == "export-qp" { 200 } else { 150 },
    };
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                a.out = args.get(i + 1).cloned();
                i += 2;
            }
            "--max-n" => {
                a.max_n = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(a.max_n);
                i += 2;
            }
            other => return Err(format!("unknown flag for '{cmd}': {other}")),
        }
    }
    Ok(a)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    // With no arguments at all `cmd` falls back to "run", but there is then no element to
    // slice past: indexing `args[1..]` panics on an empty vec, so bare `iconic-bench` --
    // the most natural thing to type, and what the usage line advertises -- aborted with
    // "range start index 1 out of range" instead of running the default suite.
    let rest: &[String] = args.get(1..).unwrap_or(&[]);

    match cmd {
        "export-qp" | "export-mip" => {
            let a = match parse_export_args(cmd, rest) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("{e}");
                    eprintln!("{}", usage(cmd));
                    std::process::exit(2);
                }
            };
            let Some(path) = a.out else {
                eprintln!("{}", usage(cmd));
                std::process::exit(2);
            };
            let json = if cmd == "export-qp" {
                export::export_qp_suite(a.max_n)
            } else {
                export::export_mip_suite(a.max_n)
            };
            std::fs::write(&path, &json).expect("write export json");
            let n_instances = json.matches("\"category\"").count();
            println!("wrote {n_instances} instances (n <= {}) to {path}", a.max_n);
        }
        "compare" => {
            if args.len() < 3 {
                eprintln!("{}", usage("compare"));
                std::process::exit(2);
            }
            let baseline = suite::load_jsonl(&args[1]).expect("read baseline");
            let current = suite::load_jsonl(&args[2]).expect("read current");
            let regressions = suite::compare(&baseline, &current);
            std::process::exit(if regressions == 0 { 0 } else { 1 });
        }
        "mip" => {
            let a = match parse_mip_args(rest) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("{e}");
                    eprintln!("{}", usage("mip"));
                    std::process::exit(2);
                }
            };
            eprintln!(
                "Running MIP benchmark suite (max_nodes={}, max_time={}s)...",
                a.max_nodes, a.max_time
            );
            let records =
                mip::run_mip_suite_records_filtered(a.max_nodes, a.max_time, a.only.as_deref());
            suite::print_report(&records);
            if let Some(path) = a.out {
                suite::save_jsonl(&path, &records).expect("write jsonl");
                println!("\nwrote {} records to {}", records.len(), path);
            }
        }
        "run" => {
            let a = match parse_run_args(rest) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("{e}");
                    eprintln!("{}", usage("run"));
                    std::process::exit(2);
                }
            };
            let mut records = suite::run_suite(a.reps);
            if a.run_mip {
                let mip_records = mip::run_mip_suite_records(a.mip_nodes, a.mip_time);
                records.extend(mip_records);
            }
            suite::print_report(&records);
            if let Some(path) = a.out {
                suite::save_jsonl(&path, &records).expect("write jsonl");
                println!("\nwrote {} records to {}", records.len(), path);
            }
        }
        _ => {
            // Unknown subcommand. Previously anything unrecognized fell through to the
            // default run path, so `iconic-bench frobnicate` (or a typo) silently started
            // the full suite.
            eprintln!("unknown command: {cmd}");
            eprintln!("{}", usage(cmd));
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn run_rejects_unknown_flags() {
        for bad in [
            &["--only", "x"][..],
            &["--help"][..],
            &["--bogus"][..],
            &["--mip-tim"],
        ] {
            assert!(parse_run_args(&sv(bad)).is_err(), "must reject {bad:?}");
        }
    }

    #[test]
    fn run_accepts_documented_flags_and_defaults() {
        let a = parse_run_args(&sv(&[])).unwrap();
        assert_eq!((a.reps, a.mip_nodes, a.mip_time), (5, 10000, 30.0));
        let a = parse_run_args(&sv(&[
            "--reps",
            "3",
            "--mip",
            "--mip-nodes",
            "500",
            "--mip-time",
            "7.5",
        ]))
        .unwrap();
        assert!(a.run_mip);
        assert_eq!(a.reps, 3);
        assert_eq!(a.mip_nodes, 500);
        assert_eq!(a.mip_time, 7.5);
    }

    /// The positional collision: `mip 50000 30` used to parse both positionals as
    /// max_nodes, the second overwriting the first and silently capping ICONIC at 30
    /// nodes. Positionals must now be rejected outright.
    #[test]
    fn mip_rejects_positionals() {
        assert!(
            parse_mip_args(&sv(&["50000", "30"])).is_err(),
            "positionals must be rejected"
        );
        assert!(parse_mip_args(&sv(&["50000"])).is_err());
        assert!(parse_mip_args(&sv(&["--bogus"])).is_err());
    }

    #[test]
    fn mip_accepts_documented_flags() {
        let a =
            parse_mip_args(&sv(&["--nodes", "50000", "--time", "30", "--only", "knap"])).unwrap();
        assert_eq!(a.max_nodes, 50000);
        assert_eq!(a.max_time, 30.0);
        assert_eq!(a.only.as_deref(), Some("knap"));
        let a = parse_mip_args(&sv(&[])).unwrap();
        assert_eq!((a.max_nodes, a.max_time), (10000, 30.0));
    }

    #[test]
    fn export_rejects_unknown_flags() {
        assert!(parse_export_args("export-mip", &sv(&["--wat"])).is_err());
        let a = parse_export_args("export-qp", &sv(&["--out", "x.json", "--max-n", "50"])).unwrap();
        assert_eq!(a.max_n, 50);
        assert_eq!(a.out.as_deref(), Some("x.json"));
    }
}
