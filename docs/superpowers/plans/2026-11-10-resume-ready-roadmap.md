# fast-tokenizer: resume-ready roadmap

Written 2026-11-10. This is the spec the per-phase plans implement. Each phase gets
its own detailed plan (`docs/superpowers/plans/2026-11-10-phase<N>-*.md`) written
when the phase starts, so it is based on the code as it is then.

## Goal

Turn fast-tokenizer from a private, working engine into a public project a
recruiter can verify in five minutes: `pip install`, a headline benchmark against
the tokenizers people actually use (OpenAI tiktoken, Hugging Face tokenizers),
reproducible numbers, and a write-up that explains the engineering.

## What the project is (for the README and the resume)

Accurate framing, to be used verbatim in spirit:

- **From scratch:** the byte-level BPE core (`src/bytelevel/`): its own
  `tokenizer.json` parser, byte alphabet, added-token matcher, hand-written
  equivalents of the GPT-2 and Qwen2 pre-tokenizer regexes over Unicode class
  tables, heap-ordered merge loop, bounded word cache and hasher. It never calls
  the Hugging Face library at request time.
- **Original engineering around it:** verified token budgets, offsets-guided
  chunking with boundary preference, borrowed-slice serialization, bounded CPU
  pool with cancellation, HTTP API, ordered streaming JSONL pipeline.
- **Hugging Face `tokenizers` is used as:** the correctness oracle (differential
  tests, 55 M-input soak, `--core verify`) and the fallback for model families the
  core does not serve yet (WordPiece/BERT, Unigram/XLM-R).
- Not "a tokenizer from scratch for every model", not "a wrapper". Say: *a
  from-scratch BPE engine for GPT-style models, proven bit-identical to the
  reference implementations, with the reference as fallback.*

## Phases

| # | Phase | Deliverable | Done when |
|---|---|---|---|
| 0 | Save pending work | Soak harness, `auto` default, tests and docs committed on a branch | Branch pushed, PR open, CI green |
| 1 | Multi-thread scaling | Cache stops taking the write lock once full; parallel-scaling bench workloads | Paired before/after report shows cache-full throughput scaling past 2 threads; quick soak clean |
| 2 | Python package | PyO3 + maturin bindings: `Tokenizer.from_file`, `count`, `encode`, `encode_batch` (releases the GIL, parallel), `decode`, `truncate`, `chunk`; abi3 wheels for Linux, macOS, Windows built in CI | `pip install` from a built wheel works on all three; Python tests pass; IDs equal the Rust engine |
| 3 | GPT-4 / Llama 3 family | Pre-tokenizer for the cl100k / Llama 3 regex family; load tiktoken `.tiktoken` encodings (cl100k_base, o200k_base) directly | Bit-identical to tiktoken on corpus + soak for cl100k_base and o200k_base; Llama 3 artifact matches the HF backend |
| 4 | Head-to-head benchmark | Reproducible Python benchmark vs tiktoken and HF tokenizers: single text, batch, 1/2/4/8 threads, cl100k/o200k/GPT-2; JSON results + charts | Script runs from a clean checkout; numbers published with machine, versions and method; includes cases where we lose |
| 5 | Memory bound | Measured peak RSS for the default server bound (20 slots x 1 MiB) and for batch Python use | Documented bound in README and PERFORMANCE.md, with the command that reproduces it |
| 6 | Public launch | Public repo, README rewrite (what/why/numbers/how), architecture diagram, write-up on clairoai.tech, PyPI release via trusted publishing, crates.io optional | A stranger can `pip install`, reproduce one benchmark, and read why it is fast |

Order matters: Phase 3 comes before Phase 4 because tiktoken's own encodings
(cl100k_base, o200k_base) are the comparison people care about; a GPT-2-only
benchmark is a weaker headline.

## Global constraints (every phase)

- Correctness first: any change under `src/bytelevel/` reruns the differential
  tests and at least `cargo run --release --example core_soak -- --quick`; a full
  soak before Phase 6.
- Performance claims only from paired before/after runs (the method in
  PERFORMANCE.md), with unchanged controls and both sides' raw reports committed.
- Never claim a speed-up the numbers do not show; report losses too.
- Keep `cargo fmt --all -- --check`, `cargo clippy --all-targets --locked -- -D warnings`
  and `cargo test --all-targets --locked` green on every commit.
- Work on branches with PRs into `main`; no direct commits to `main`.
- Commit messages carry no AI co-author or attribution lines.
- Model files stay out of git (fetched with `models/fetch.py`).
- Benchmarks run on the development laptop (quiet machine), not the shared VPS.

## Open decisions for the user

1. **Public name** for the PyPI/crates package (`fast-tokenizer` is likely taken
   on PyPI). Decide before Phase 2 publishes anything; the code can use a
   placeholder.
2. **License** for the public repo (MIT or Apache-2.0 recommended; the reference
   models keep their own licenses and are not redistributed).
3. Whether the backend fallback stays a default dependency or becomes an optional
   feature in the Python package (smaller wheel vs. covering BERT/XLM-R).
