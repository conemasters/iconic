# ICONIC — Architecture

ICONIC uses a regularized interior-point method behind one canonical standard form.

## `iconic-ipm` — Proximal Interior-Point Method

A primal-dual infeasible IPM with proximal regularization (Proximal Method of
Multipliers). The proximal terms make the reduced KKT system **quasidefinite** —
factorizable without LICQ, handling rank-deficient and degenerate problems
robustly. Pure LPs (`P = 0`) instead route through a homogeneous self-dual
embedding, which detects infeasibility/unboundedness natively via `τ → 0`.

- **Mehrotra predictor–corrector** with Gondzio multiple centrality correctors
  (pushes complementarity into the central band, reusing the factorization —
  transformational on badly-scaled problems: 199→10 iterations).
- **Cone-aware Nesterov–Todd scaling** for symmetric cones (SOC, PSD).
- **Separate nonsymmetric engine** for the exponential cone (self-concordant barrier,
  infeasible interior start, hybrid primal-dual step).
- **Adaptive centering** (switches between aggressive affine and safe centering based
  on step quality).
- **Multi-level regularization escalation** — boosts proximal regularization and
  re-factors on factorization failure.

## Linear Algebra

| Component | Approach | Why |
|---|---|---|
| Sparse storage | Own `CscMatrix<T>` | Tight control over index maps, in-place updates |
| Fill-reducing ordering | AMD (`amd` crate) | ~12× sparse-vs-dense on banded problems (n=400) |
| Sparse LDLᵀ | Own quasidefinite, no-pivot, dynamic regularization | Control over pivot clamping and escalation |
| Dense BLAS/LAPACK | `faer` (pure Rust) | Bunch–Kaufman LBLT, Cholesky, SIMD eig, gemm |
| QP condensed KKT | `P + ρI + AᵀD⁻¹A` → Cholesky (LLT) when PD | No pivoting overhead; 26–35% faster on large equality-free QPs |
| SDP Kronecker solve | `eig(W)` → decoupled `(λᵢλⱼ + c)` system | O(k³) instead of O(k⁶) for `X⪰0` SDPs |
| Symbolic reuse | Pattern fixed → factorize numerically only | O(nnz) update per iteration, no re-ordering |

## Presolve Pipeline

1. **Validate & compact** infinite bounds
2. **Empty row / column elimination**
3. **Linearly-dependent equality removal** (row-echelon test)
4. **Dominated / parallel inequality compaction** (canonicalize by signed max entry)
5. **Duplicate row removal**
6. **Fixed-variable elimination** (singleton substitution)
7. **Doubleton equality substitution** (`xᵢ = β + γxⱼ`)
8. **Ruiz equilibration + cost scaling** (matrix-free, convergence-aware)
9. **Postsolve** — reconstruct original-space solution, duals, slacks in LIFO order

All residuals, objectives, and tolerances are in **original unscaled units**.
