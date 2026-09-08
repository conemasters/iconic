# ICONIC build & test — one command to rebuild everything that matters.
#
# The key gotcha this guards against: `maturin develop` rebuilds iconic-py (the
# PyO3 .abi3.so) but NOT iconic-c (the C ABI libiconic_c.so). When iconic-api,
# iconic-presolve, iconic-ipm, or iconic-linalg change, iconic-c must be rebuilt
# AND its .so copied to iconic-py/python/iconic/ — otherwise the CVXPY backend
# silently runs stale code.
#
# Usage:
#   make build       — release build of the whole workspace (fast check)
#   make release     — release build + copy libiconic_c.so + maturin develop
#   make install     — install the built wheel into the current Python
#   make test        — cargo test --release --workspace
#   make bench       — run the full fair benchmark + charts
#   make bench-fast  — quick benchmark subset
#   make all         — test + release + install + bench

CARGO := cargo
PYTHON := python3
ICONIC_PY := iconic-py
LIB_SRC := target/release/libiconic_c.so
LIB_DST := $(ICONIC_PY)/python/iconic/libiconic_c.so

# Strip absolute build paths (home dir, cargo registry, toolchain) from every
# compiled artifact, so locally-built binaries never embed the builder's
# environment. $(HOME) is expanded at build time — no literal path is stored here.
export RUSTFLAGS := --remap-path-prefix=$(HOME)=/build

.PHONY: build release test install bench bench-fast all clean

build:
	$(CARGO) build --release --workspace

release:
	$(CARGO) build --release -p iconic-c -p iconic-py
	cp $(LIB_SRC) $(LIB_DST)
	@echo "  → libiconic_c.so copied to $(LIB_DST)"
	cd $(ICONIC_PY) && $(PYTHON) -m maturin develop --release

install:
	$(PYTHON) -m pip install --force-reinstall --no-deps target/wheels/iconic-*.whl

test:
	$(CARGO) test --release --workspace

bench:
	$(CARGO) run --release -p iconic-bench -- export-qp --out /tmp/qp.json --max-n 200
	$(PYTHON) scripts/bench_all_solvers.py --qp-json /tmp/qp.json --iconic-jsonl /tmp/iconic.jsonl --out /tmp/results.jsonl
	$(PYTHON) scripts/plot_benchmarks.py /tmp/results.jsonl -o scripts/charts/

bench-fast:
	$(CARGO) run --release -p iconic-bench -- export-qp --out /tmp/qp.json --max-n 100
	$(PYTHON) scripts/bench_all_solvers.py --qp-json /tmp/qp.json --iconic-jsonl /tmp/iconic.jsonl --out /tmp/results.jsonl
	$(PYTHON) scripts/plot_benchmarks.py /tmp/results.jsonl -o scripts/charts/

all: test release install bench
	@echo "=== ICONIC build + test + install + bench complete ==="

clean:
	$(CARGO) clean
	rm -f /tmp/qp.json /tmp/results.jsonl
