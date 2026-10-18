# Project priorities

The user's primary goal is a performance-oriented Rust tokenizer and document
preparation engine, while learning Rust. Treat runtime efficiency and bounded
resource use as core requirements, alongside correctness.

- Measure performance in release builds. Read PERFORMANCE.md and use the
  benchmark runner before and after changes to hot paths. Keep the workload,
  artifact, toolchain, and machine comparable; report limits of synthetic data.
- Prefer algorithmic improvements, fewer allocations/copies, reused model state,
  bounded CPU work, and buffered I/O. Keep CPU work off async runtime threads.
- Preserve verified token budgets, original source byte ranges, Unicode scalar
  boundaries, deterministic special-token behavior, ordering, and cancellation.
- Keep general no-fit fallback correct for nonmonotonic token counts; do not
  silently assume that longer text always produces more tokens.
- Add caches, more concurrency, native CPU flags, or unsafe code only when
  profiling and benchmarks support the change. Explain important Rust ownership
  and concurrency choices briefly so the user can learn from the implementation.
- Validate changes with cargo fmt, cargo clippy --all-targets -- -D warnings,
  and the relevant correctness/integration tests.
