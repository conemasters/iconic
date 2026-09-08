# Contributing to ICONIC

Thanks for your interest in contributing to ICONIC! This guide covers how to get
started, what we're looking for, and how the contribution process works.

## What We're Building

ICONIC (Interior-Point Conic Optimization with Certificates) is a high-performance convex
optimization and mixed-integer programming solver in pure Rust. Our goal is to be
the fastest open-source solver across QP, SOCP, SDP, exponential cone, and MIP
problems — competitive with (and often faster than) the best available
open-source alternatives.

## We Especially Value Benchmarks

**Real-life problems are the most valuable contribution you can make.** ICONIC is
built to solve problems that matter, and we can only claim to be fast and correct
if we measure against problems people actually care about.

### What makes a great benchmark contribution

- **Real-world origin.** Problems from your own work — portfolio optimization,
  control, machine learning, supply chain, scheduling, power systems, finance,
  engineering design — are gold. Synthetic generators are fine for coverage, but
  real instances reveal structure that generators miss.

- **Diverse problem classes.** We benchmark across all supported cones and MIP.
  Contributions that exercise the exponential cone (entropy, log-sum-exp, geometric
  programming), SOCP (robust least-squares, portfolio risk), SDP (matrix
  completion, Lyapunov inequalities, MAXCUT relaxations), and MIP (scheduling,
  facility location, routing) are especially valuable.

- **Known optimal values.** A problem with a verified optimum (from an exact solver,
  closed form, or published result) is much more useful than one without. We can
  validate correctness, not just time.

- **Edge cases.** Ill-conditioned, degenerate, rank-deficient, or near-infeasible
  problems stress the solver in ways well-conditioned ones don't. If you have a
  problem that makes your current solver struggle, we want it.

### How to submit a benchmark

1. Add a problem generator or a static instance file to `iconic-bench/src/` (for
   generators) or `iconic-bench/data/` (for static instances, with provenance).

2. Register it in `iconic-bench/src/suite.rs` so it's included in the standard
   `cargo run --release --bin iconic-bench` run.

3. Run the suite before and after: `cargo run --release --bin iconic-bench -- run
   --reps 5 --out before.jsonl && cargo run --release --bin iconic-bench -- run
   --reps 5 --out after.jsonl && cargo run --release --bin iconic-bench -- compare
   before.jsonl after.jsonl`

4. Include the problem's provenance — where it comes from, what the known optimum
   is, and any license information if it's from a published set.

## Code Contributions

### Setup

```sh
git clone <repo-url> && cd iconic
cargo build --release
cargo test --release
```

### Before submitting

- `cargo test --release` must be green.
- `cargo clippy --all-targets` must be clean.
- Run the benchmark suite and check `cargo run --release --bin iconic-bench -- compare
  before.jsonl after.jsonl` — no status regressions, no significant iteration/time
  regressions.

### Style

- No `unwrap()` in production code — use `expect()` with a message, or proper
  error propagation.
- Generic over `Scalar: num_traits::Float` where possible; validate f64 first.
- Commits describe the engineering directly.

### Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the solver design, linear algebra
stack, and presolve pipeline.

## License

ICONIC is licensed under the Apache License, Version 2.0 (see
[`LICENSE`](./LICENSE)). By contributing, you agree that your contributions
will be licensed under the same terms as the project.
