# Fast Tokenizer — context handoff for Claude

Prepared on 2026-10-18. This file summarizes the project and the work completed
in the preceding Codex session. Read [AGENTS.md](AGENTS.md),
[README.md](README.md), and [PERFORMANCE.md](PERFORMANCE.md) alongside this file.
Source code and current Git status take precedence over this snapshot.

## User intent and working priorities

The user is learning Rust by building a fast tokenization and document
preparation engine. They explicitly emphasized:

> “we have to make it as performance oriented as possible as efficient hence
> why im learning rust and implementing this in RUST”

Performance and bounded resource use are core requirements. Explain important
ownership, allocation, and concurrency decisions briefly so the user can learn.
Prefer measured algorithmic improvements and fewer allocations/copies over
speculative complexity. Preserve correctness when optimizing. The user prefers
continued execution over unnecessary confirmation questions.

The latest request was to save this Markdown handoff for Claude. It did not
request additional implementation beyond creating the handoff.

## Project identity and environment

- Workspace: `C:\Users\HP\Desktop\fast-tokenizer`.
- Public repository: <https://github.com/zainulwahaj/tokforge> (renamed from
  `fast-tokenizer` and made public on 2026-11-10; GitHub redirects the old URL).
- Branch: `main`; origin tracks the repository above.
- Public name **tokforge** (decided 2026-11-10; PyPI and crates.io names were
  free). Rust package `tokforge`, version `0.1.0`, edition 2021,
  `publish = false` until release; library crate `tokforge`; executable
  `tokforge`. License `MIT OR Apache-2.0` (`LICENSE-MIT`, `LICENSE-APACHE`).
  The local folder keeps its old name.
- Development machine: Windows x86_64, PowerShell, Intel Core i7-7820HQ
  (4 physical cores / 8 logical processors).
- Last verified toolchain: rustc 1.99.1; `rust-toolchain.toml` selects stable
  with rustfmt and Clippy. `Cargo.lock` is committed.
- Rust tools are installed in `C:\Users\HP\.cargo\bin`. A fresh terminal may
  be needed for PATH; invoking `cargo.exe` there directly also works.
- Git, GitHub CLI, Visual C++ build tools, and the Windows SDK are installed.

The project uses Hugging Face's Rust `tokenizers` crate as its tokenizer
backend. It is building an efficient Rust engine and interfaces around that
backend; it does not yet implement BPE/Unigram algorithms from scratch, train
models, or perform model inference. No production tokenizer artifact has been
selected or bundled. The only public deployment is the demo at
<https://tokforge.clairoai.tech> (see "Public demo" below).

## Completed milestones and Git history

| Commit | Completed work |
| --- | --- |
| `f7be9db` | Initial Rust scaffold and Windows setup script |
| `4fb4795` | Shared engine, CLI commands, correctness fixtures/tests |
| `7ef0ecb` | Bounded HTTP API, ordered streaming JSONL pipeline, performance baseline |
| `b832a70` | Optimized prefix search and serialization/I/O, paired benchmark reports, performance guidance |

Before creating this handoff, `HEAD` was `b832a70`, the working tree was clean,
and `HEAD` matched `origin/main`. Both implementation milestones from the last
session were committed and pushed. This handoff is a subsequent documentation
addition; inspect Git status for its current commit state.

Initial Rust installation/network restrictions were resolved and setup is
complete. Do not repeat installation or repository creation as the next task.
GitHub authentication was working in the preceding session, but should be
checked if a later network operation requires it.

### Launch work (2026-11-10)

The project is being prepared for a public release as **tokforge**, following
`docs/superpowers/plans/2026-11-10-resume-ready-roadmap.md`. Work lands as a
stack of PRs; merge them in order (each retargets to `main` after the one
below it merges):

| PR | Branch | Content |
| --- | --- | --- |
| #1 | merged | Differential soak, `auto` default, golden check ignores the Python patch version (fixed CI) |
| #2 | `cache-write-lock` | 16-shard single-hash word cache: 4.6x on 8 threads (PERFORMANCE.md "Fifth set") |
| #3 | `tokforge` | Rename to tokforge, MIT OR Apache-2.0, `Engine::encode_ids`, Python package (`python/`, PyO3 0.30 abi3), wheel CI |
| #4 | `gpt4-tokenizers` | cl100k_base and o200k_base in the core, case-folded contractions resolved, both in the reference suite |
| #5 | `launch` | Head-to-head and memory benchmarks (PERFORMANCE.md "Sixth set"), public README |

Releases: `.github/workflows/release.yml` builds abi3 wheels (Linux x86_64 and
aarch64, macOS arm64 and x86_64, Windows x64) and an sdist on packaging PRs;
pushing a `v*` tag that matches `python/Cargo.toml` publishes them to PyPI via
trusted publishing (repository environment `pypi`, deployable only from `v*`
tags; PyPI trusted publisher set up by the user) and creates a GitHub release.
**v0.1.0 was published on 2026-11-10** (<https://pypi.org/project/tokforge/>,
five wheels plus the sdist). A clean `pip install tokforge` gave IDs
identical to tiktoken on gpt2, cl100k_base and o200k_base. For the next
release, bump `version` in both `Cargo.toml` and `python/Cargo.toml`, merge,
then push the matching tag. GitHub retired the `macos-14` runner image, so use
`macos-latest`.
Still open: the write-up on clairoai.tech, and reducing the core's
load-time memory peak on large vocabularies. Llama 3 shares cl100k's pattern
but keeps running on the backend because of its `TemplateProcessing`
post-processor. Commits in this repo need the per-command safe.directory
override (see "Local environment details"); never add AI attribution lines.

## Repository map

| File | Role |
| --- | --- |
| `src/lib.rs` | Public modules, engine type re-exports, version constant |
| `src/engine.rs` | Tokenizer loading, artifact identity, encoding/counting/decoding, verified source prefixes/chunks |
| `src/work.rs` | Shared operation enum, typed response serialization, cancellation, bounded Rayon CPU pool |
| `src/api.rs` | Axum routes, typed JSON input, admission before body buffering, request deadlines/errors |
| `src/batch.rs` | Bounded JSONL reader/producer and ordered result writer with backpressure |
| `src/main.rs` | Clap commands, file/stdin/stdout handling, server startup, Ctrl+C, runtime lifecycle |
| `src/bytelevel/mod.rs`, `pretokenize.rs`, `unicode_tables.rs` | Custom byte-level BPE core: artifact gating, byte alphabet, heap-ordered merges, bounded word cache, hand-written GPT-2/Qwen2 pre-tokenizers, regex-derived Unicode class tables |
| `build.rs` | Captures the locked `tokenizers` version into `work::BACKEND` for every response |
| `examples/unicode_tables.rs` | Regenerates the Unicode class tables from the live Oniguruma regex |
| `examples/core_soak.rs` | Release differential soak: millions of corpus-derived, swept and generated inputs through the core, `Engine` in `Verify` and `Backend` modes, and the backend oracle per artifact and option combination; prints mismatches as Rust literals and shrinks them |
| `tests/bytelevel.rs` | Differential suite: core vs backend on corpus, random text, synthetic and real artifacts; engine core selection |
| `.github/workflows/ci.yml` | fmt/clippy/tests on Linux and Windows, reference job with fetched models, release binaries |
| `tests/engine.rs` | Token correctness, offsets, special tokens, Literal-mode special-piece suppression, Unicode, budgets, nonmonotonic counts |
| `tests/api.rs` | Endpoints, invalid requests, body limits, overload, shutdown, real loopback slow-body timeout |
| `tests/batch.rs` | Ordering, record recovery, open streams, bounded read-ahead, cancellation, broken output |
| `tests/cli.rs` | CLI operations, JSONL exit behavior, file input/output, overwrite protection |
| `tests/support/mod.rs` | Integration-test pool construction |
| `tests/fixtures/wordpiece.json` | Tiny hand-authored synthetic WordPiece artifact, not a production model |
| `tests/reference.rs` | Production-artifact suite: manifest hashes, oracle golden IDs, round trips, budget invariants; skips without models |
| `tests/fixtures/corpus/` | 15 committed reference texts (each under 8 KiB) covering scripts, whitespace, controls, emoji, special spellings |
| `tests/fixtures/golden/<artifact>/` | Committed oracle IDs per corpus file with oracle attribution and backend cross-check |
| `tests/fixtures/golden/generate.py`, `KNOWN_DIFFERENCES.json` | Golden generator (tiktoken, sentencepiece, slow transformers) and documented engine-vs-oracle divergences |
| `benches/throughput.rs` | Dependency-free release benchmark runner; `--tokenizer`/`--corpus` target a real artifact |
| `benchmarks/before.json`, `benchmarks/after.json` | Summaries of the first paired runs (`7ef0ecb` vs `b832a70`) |
| `benchmarks/guided-*.json` | Summaries of the second pair and its repeat (`b832a70` vs offsets-guided cuts) |
| `benchmarks/runs/` | Raw reports: `before-*`/`after-*` for the first pair, `guided-*` and `guided-recheck-*` for the second |
| `scripts/setup.ps1` | Rust setup/checks and optional private GitHub repository creation |
| `models/README.md` | Artifact directory layout, manifest fields, how to fetch and verify |
| `models/MANIFEST.json`, `models/fetch.py` | Pinned reference artifacts (repo, revision, SHA-256, license, types) and the stdlib-only fetch/verify script |
| `AGENTS.md` | Durable performance and correctness priorities |
| `PERFORMANCE.md` | Benchmark method, results, optimization rationale, remaining profiling work |

Main dependencies are `tokenizers` 0.22 (locked to 0.22.2, Oniguruma enabled),
Axum 0.8, Tokio 1, Rayon 1, Clap 4, Serde/serde_json, SHA-256, anyhow, and
thiserror. See `Cargo.toml` and `Cargo.lock` for the complete dependency set.

## Available commands

Use release builds for meaningful runtime measurements:

```powershell
cargo build --release --locked
cargo run --release -- info
cargo run --release -- count --tokenizer tests/fixtures/wordpiece.json --text "hello world!"
cargo run --release -- encode --tokenizer tests/fixtures/wordpiece.json --text "hello world!"
cargo run --release -- decode --tokenizer tests/fixtures/wordpiece.json --text '[4,5,6]'
cargo run --release -- truncate --tokenizer tests/fixtures/wordpiece.json --text "hello world!" --max-tokens 2
cargo run --release -- chunk --tokenizer tests/fixtures/wordpiece.json --text "hello world!" --max-tokens 2
```

The sample count is 3 and the sample IDs are `[4,5,6]`. Replace the fixture path
with `models/tokenizer.json` for a real model. Model JSON artifacts under
`models/` are ignored by Git; record their model source and revision separately.
There is no automatic artifact download or built-in default production model.

`encode --ids-only` returns IDs and count only. `--core auto|backend|verify`
selects the tokenization core (`auto` uses the custom byte-level BPE core for
qualifying artifacts, `verify` double-encodes with the backend and fails on
disagreement). **The default is `auto`** (both `CoreMode::default()` and the
CLI): the user's decision on 2026-10-18 was to keep `verify` as the default
until the differential soak passed clean on GPT-2 and Qwen2.5, and it did that
day ("Differential soak" in PERFORMANCE.md; item 6 under "Next useful work").
Verify costs about one backend encode per request on qualifying artifacts and
nothing on artifacts the core does not serve. `serve`
adds `GET /metrics` and `POST /v1/batch`; every response carries
`schema_version`, `engine_version`, `backend`, and `artifact.core`.
Benchmark a real artifact with
`cargo bench --bench throughput -- --tokenizer models/gpt2.tokenizer.json --corpus <file> --http-load 8`.
Run the differential soak of the custom core against the backend with
`cargo run --release --example core_soak -- --output soak.json` (about 80
minutes for 55 million inputs on the development machine; `--quick` is a
three-minute smoke run of the same shape; `--seed`, `--random`, `--threads`,
`--artifact gpt2` and `--self-test` narrow or exercise it). It skips artifacts
the core does not support and exits nonzero on any mismatch.

Single-operation commands accept `--text`, `--file`, or stdin when both are
omitted. `--text` and `--file` conflict. Decode input is a JSON array of unsigned
32-bit IDs. File/stdin whitespace is preserved; PowerShell pipelines can add
newlines or change encoding, so files are preferable for exact source bytes.
Success data goes to stdout as JSON; errors go to stderr with a failing exit.

## Engine correctness contract

- The engine hashes the exact original `tokenizer.json` bytes with SHA-256 and
  reports the hash, the vocabulary size, and `suppressed_special_pieces` in
  every response and in `info`. The hash is not an upstream model name.
- Saved padding and truncation are disabled. Counts cover the entire input.
  Positive BPE dropout is rejected so counts and budgets remain deterministic.
- Special-token spellings inside text are literal by default. Separately,
  `--special-tokens recognize` enables registered special-token matching, and
  `--add-special-tokens` enables post-processor tokens such as BOS/EOS.
  Post-processor tokens count toward the budget.
- Literal mode guarantees that text never produces a registered special-token
  ID, on every model family. Besides turning added-token matching off,
  `Engine::from_bytes` makes each special spelling that the model lists as an
  ordinary piece unreachable at load time (`suppress_special_pieces` in
  `src/engine.rs`): Unigram pieces are rescored to `-inf` with the original
  `min_score` restored so the unknown penalty is unchanged (the converted XLM-R
  vocabulary scores `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>` at 0.0, which the
  Viterbi search otherwise prefers); WordPiece/WordLevel entries leave the
  lookup map; BPE drops the merges whose result is the spelling and, under
  `ignore_merges`, the entry too. The ID-to-token table, decoding, the hash and
  `vocab_size` are untouched. `artifact.suppressed_special_pieces` reports the
  count (Recognize mode: 0; GPT-2 and Qwen2.5: 0, their byte-level regex splits
  the spelling and no merge forms it; BERT: 4, `[UNK]` exempt; XLM-R: 5). The
  unknown token's own spelling stays reachable in WordPiece, WordLevel and BPE
  because the unknown ID is looked up by it. Loading fails with
  `Error::UnsuppressibleSpecialToken` when a special spelling is a single
  scalar or, for Unigram, has a scalar without a piece of its own; Recognize
  mode still loads. Cost is load-time only: one targeted re-read of the model
  section (`unk_id` for Unigram; the merge list for BPE only when both halves of
  some split of a spelling are entries) and one model rebuild. Measured on
  2026-10-18 (PERFORMANCE.md, third pair): fixture workloads unchanged within
  run-to-run ranges; load adds about 0.6 s on XLM-R's 250k-piece Unigram
  vocabulary and about 20 ms on BERT, nothing on GPT-2 or Qwen2.5.
- Decode rejects unknown IDs. Special tokens are preserved unless
  `--skip-special-tokens` is set. Exact round trips depend on the model's
  normalizer/decoder; they are not promised for arbitrary artifacts.
- Encoding offsets are UTF-8 byte ranges. Synthetic tokens may have `(0,0)`;
  byte-level token offsets may overlap or lie within a Unicode scalar. They
  must not be used blindly as Rust string slicing boundaries.
- Truncate returns an original source prefix. Chunk returns nonoverlapping,
  contiguous source slices that reconstruct every byte, including whitespace.
  Ranges use inclusive `start_byte` and exclusive `end_byte`.
- Each returned slice is independently encoded to verify its token budget.
  Slice boundaries preserve Unicode scalars and never separate CR from LF.
- Chunk cuts prefer, in order, a blank line, a line break, a sentence end
  followed by whitespace, then a pre-tokenizer word boundary, taking the
  strongest level that still fills at least half the budget
  (`MIN_FILL_PERCENT` in `src/engine.rs`); otherwise the fullest subword or
  scalar boundary. Truncate prefers a word boundary under the same rule.
  Grapheme clusters may still split at the subword level.
- Token counts are **not monotonic** with increasing prefix length. The search
  does not promise the longest fitting prefix or maximally full chunks.
  Never remove the general fallback by assuming monotonic counts.
- Zero budgets are allowed. Truncate can return empty text; chunk errors if it
  cannot make progress. Empty input produces zero chunks. Truncate fails if
  even the empty prefix's added special tokens exceed the budget.

## Worker architecture and cancellation

`CpuPool` owns a dedicated Rayon pool and an `Arc<Engine>`. Default CLI settings
are 4 workers and 16 extra admitted slots. The total capacity is
`workers + queue_capacity`; worker count must be positive and zero queue capacity
is supported. Admission uses a semaphore. Work is submitted using `spawn_fifo`,
with a Tokio oneshot carrying the result back to async code.

CPU operations and result serialization run on the Rayon pool, rather than
blocking Tokio's I/O threads. The model is loaded once per server/dataset and
shared without cloning the full tokenizer per request. Worker panics are caught
and become failures. A CPU permit is released before sending the completed
result, avoiding a slot-reuse race in the ordered JSONL pipeline.

`Cancellation` uses an atomic flag plus a watch notification for async waiters.
Drop guards cancel abandoned work. Workers check request cancellation, global
shutdown, and whether the result receiver has closed. Chunk/prefix loops check
between tokenizer calls. A single call inside the tokenizer dependency cannot
be interrupted midway, so a timed-out request retains its CPU slot until that
call returns.

The CLI explicitly shuts down its Tokio runtime in the background after async
commands finish. This avoids hanging on Tokio stdin's uninterruptible blocking
read after JSONL cancellation; preserve that rationale if refactoring startup.

## HTTP interface

```powershell
cargo run --release -- serve --tokenizer tests/fixtures/wordpiece.json --workers 4 --queue-capacity 16
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/count -ContentType 'application/json' -Body '{"text":"hello world!"}'
```

- Default bind: `127.0.0.1:8080`; override with `--bind IP:PORT`.
- `GET /health` and `GET /v1/info` expose health and artifact/options/capacity.
- `POST /v1/count` and `/v1/encode`: `{"text":"..."}`.
- `POST /v1/decode`: `{"ids":[4,5,6],"skip_special_tokens":false}`.
- `POST /v1/truncate` and `/v1/chunk`: `{"text":"...","max_tokens":128}`.
- `POST /v1/process`: a tagged operation object in the JSONL format below.
- The model and special-token options are fixed at startup. Unknown body fields
  are rejected. Requests deserialize directly into typed inputs, without an
  intermediate JSON tree or a second deserialization pass.
- CPU admission happens before JSON body buffering. Full capacity produces
  `503` with `Retry-After: 1`. Completed responses are transferred independently
  of CPU admission; this is not a bound on all network response buffering.
- Default body limit: 1 MiB (`--max-body-bytes`), with `413` for excess.
- Default deadline: 30 seconds (`--timeout-seconds`), covering body reading,
  queueing, CPU execution, and response construction. It excludes subsequent
  network transfer. Deadline responses use `408`.
- Malformed JSON: `400`; unsupported content type: `415`; invalid schema,
  token IDs, or operation budgets: `422`. Errors contain `error.code/message`.
- Ctrl+C initiates graceful HTTP shutdown and cancels CPU work.
- This is a local HTTP service without authentication or TLS. Put it behind a
  proxy before exposing it, as the public demo does.

### Public demo

<https://tokforge.clairoai.tech> runs on the user's Contabo VPS (`ssh contabo`,
Ubuntu 24.04, Caddy with one file per site in `/etc/caddy/sites/`). Files are
in `demo/`: `index.html` is the page; `deploy/tokforge@.service` is a systemd
template run once per encoding (gpt2 on 18101, cl100k_base on 18102,
o200k_base on 18103, loopback only, 2 workers + 6 slots, 128 KiB bodies, 5 s
deadline, `DynamicUser`, `MemoryMax=256M`, sandboxed); `deploy/tokforge.caddy`
serves the page and exposes only `POST /api/<encoding>/v1/{count,encode,chunk}`;
`deploy/install.sh` verifies the model files against `models/MANIFEST.json`,
installs to `/opt/tokforge`, and rolls back the Caddy site if it does not
validate. To update: copy the CI `tokforge-linux-x86_64` binary, the three model
files, `MANIFEST.json` and the `demo/` files into one directory on the VPS and
run `install.sh` as root. Measured on 2026-11-10: 40 concurrent 58 KB encodes
gave 34 responses and 6 `503`s (admission working); memory peaked at 37, 72 and
145 MiB for gpt2, cl100k_base and o200k_base. Caddy sorts directives by its own
order, so the API must stay inside `handle /api/*`; a sibling `route` loses to
the catch-all `handle` and every call 404s.

## JSONL interface

Input is one tagged operation per physical line:

```jsonl
{"op":"count","text":"hello world!"}
{"op":"encode","text":"hello café 你好!"}
{"op":"decode","ids":[4,5,6],"skip_special_tokens":false}
{"op":"truncate","text":"hello world!","max_tokens":2}
{"op":"chunk","text":"hello\r\nworld!","max_tokens":2}
```

```powershell
cargo run --release -- jsonl --tokenizer tests/fixtures/wordpiece.json --file records.jsonl --output results.jsonl --workers 4 --queue-capacity 16
```

- Omit `--file` for stdin; omit `--output` for stdout.
- Output files use `create_new`: existing output files and input/output aliasing
  are rejected rather than overwritten.
- Success shape: `{"line":1,"result":{...operation response...}}`.
  Failure shape: `{"line":2,"error":{"code":"...","message":"..."}}`.
- There is one producer and one ordered consumer, not a Tokio task per record.
  Jobs can execute concurrently; output remains in physical input-line order.
- A separate record-window permit is held through writing the result. This
  bounds queued/running/completed records to pool capacity, with a bounded
  64 KiB CLI output buffer in addition. Slow output stops further reading.
- Ready writes are batched. Flush before waiting on more input or a worker,
  after 64 records, and at EOF, so open stdin still receives prompt results.
- Default maximum line size is 1 MiB (`--max-line-bytes`), excluding LF; CR in
  CRLF counts toward the limit. Oversized lines are drained without retaining
  the whole line and yield `line_too_large`.
- Blank lines, invalid UTF-8, malformed JSON, unknown fields, and engine failures
  yield per-line errors; later records continue. A final newline is optional.
- Per-record failures produce a nonzero final exit after processing the stream.
  I/O errors, broken output, or Ctrl+C stop processing and may leave partial
  output. A summary goes to stderr.
- Input records and result expansion are bounded by workload/window settings,
  not by total dataset size. Outputs can be much larger than inputs; memory
  peaks have not yet been profiled on production artifacts.

## Performance work already completed

Two generations of prefix search preceded the current one. The first counted
the entire remaining suffix per chunk. The second probed local prefixes with
exponential growth and binary refinement, which cost a logarithmic number of
full re-encodes per chunk and cut at arbitrary scalar boundaries. The current
engine is offsets-guided: `Engine::fit` encodes one bounded window of the
remaining text with byte offsets and word indices, `guide` turns every token
boundary into a `Cut` with a predicted token count and a boundary `Level`
classified from the source text, `choose` picks the strongest level that fills
at least half the budget, and `verify` re-encodes the chosen slice, shrinking
the candidate set by the measured overshoot on overflow. The window is sized by
the previous chunk's verified density (or the whole-text count for truncate),
grows when everything in it fits, and bounds memory per chunk rather than per
document. Truncate first checks whether the complete input fits. The exhaustive
fallback (`exhaustive`) remains complete for nonmonotonic counts: whole
remainder, then token boundaries, then scalar boundaries, longest first; it can
still be quadratic on pathological input.

Chunk and truncate responses serialize borrowed slices of the request text
(`Slices` in `src/work.rs`), so the document is never copied a second time;
the public `Engine::chunk`/`truncate` API still returns owned `Chunk`s built
from `Span`s. Decode validates IDs against a bitset built once at load instead
of a per-ID string lookup.

CLI, HTTP, and JSONL share typed response serialization in `src/work.rs`.
Encoding responses serialize borrowed slices from `tokenizers::Encoding`
directly into byte buffers, avoiding an owned result copy and JSON value tree.
The public `Engine::encode` API still returns owned vectors/strings; its ownership
contract was preserved. Counting uses `encode_fast`, avoiding source offsets.
Complete successful JSONL envelopes are serialized on CPU workers, then buffered
for output instead of assembling and flushing every row on the async thread.

Release configuration uses thin LTO and one codegen unit. Native CPU flags,
unsafe code, and SIMD have not been introduced; the custom core's bounded word
cache and caller-owned output buffers are the only cache and buffer reuse.

A custom byte-level BPE core (`src/bytelevel`) serves encode and count for
artifacts it fully understands (GPT-2 and Qwen2.5 among the reference set):
hand-written equivalents of the two pre-tokenizer regexes over Unicode class
tables captured from the Oniguruma build in Cargo.lock, raw-byte merging in the
backend's heap order (lowest rank, then leftmost), a bounded 10,000-word cache,
FxHash maps, and caller-owned output buffers; token strings are borrowed from
the vocabulary table at serialization time. It is enabled per artifact only
when every component is recognized, Literal mode suppressed nothing, vocabulary
sizes agree, and a load-time self-check matches the backend; per request it
declines inputs it cannot reproduce exactly (NFC would change the text, or a
case-folded contraction), which the backend then serves. `tests/bytelevel.rs`
proves bit-identical IDs, tokens, offsets, word indices and masks against the
backend on the corpus, thousands of random texts, synthetic artifacts, and the
real GPT-2 and Qwen2.5 files, and through the public `Engine` API including
chunk cuts. The exhaustive no-fit fallback is bounded to the guide window plus
the nearest 1,024 scalar boundaries (a documented contract revision). The
runner reports p50/p95/p99, allocations per iteration, per-stage backend
timings, direct core-vs-backend workloads, and a loopback HTTP load test.

### Recorded benchmark results

Run with `cargo bench --bench throughput`; optionally append
`-- --output benchmarks/local.json`. Debug benchmark binaries return without
running measurements, including when invoked by `cargo test --all-targets`.

The runner warms each operation and measures for at least 300 ms per workload.
Baseline `7ef0ecb` and optimized binaries were compiled first, then alternated
for three runs without concurrent compilation. Reported values are medians of
the three runs' mean timings, not p50/p95/p99 per-request latencies.

| Workload | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Count | 11.69 | 11.78 | 0.99x |
| Public owned-buffer encode | 13.35 | 13.91 | 0.96x |
| Truncate to 128 tokens | 25.32 | 13.44 | 1.88x |
| Chunk at 128 tokens | 537.12 | 83.22 | 6.45x |
| Encode 2,000 JSONL records | 30.02 | 17.36 | 1.73x |

Count and public encode were unchanged controls; do not claim they improved.
The first four workloads use 33,800 bytes of repeated synthetic WordPiece text.
JSONL uses 2,000 encode records / 102,000 input bytes, four workers, sixteen
extra slots, and a sink writer; pool startup and model loading are included.

These results establish improvements on the included fixture only. They do not
establish production tokenizer performance, HTTP/network throughput, disk I/O
speed, peak memory, allocation counts, or tail latency. Initial isolated runs
were noisy, which is why the paired runs and unchanged controls were retained.
See the committed raw reports rather than relying on conversational numbers.

A second pair (2026-10-18, baseline `b832a70` rebuilt from a detached worktree
into a separate target directory, same method) measured the offsets-guided
cuts. Summaries: `benchmarks/guided-before.json`, `benchmarks/guided-after.json`,
plus an immediate repeat in `benchmarks/guided-recheck-*.json` because the JSONL
control moved first one way and then the other:

| Workload | Before (ms) | After (ms) | Speedup | Repeat |
| --- | ---: | ---: | ---: | ---: |
| Count | 12.16 | 11.61 | 1.05x | 0.99x |
| Public owned-buffer encode | 12.90 | 13.36 | 0.97x | 0.96x |
| Truncate to 128 tokens | 13.07 | 11.28 | 1.16x | 1.12x |
| Chunk at 128 tokens | 82.83 | 33.06 | 2.51x | 2.28x |
| Encode 2,000 JSONL records | 12.31 | 15.09 | 0.82x | 1.11x |

Chunk is the real gain (2.3x to 2.5x); truncate is about 1.1x. Count, encode,
and JSONL are unchanged controls with overlapping per-run ranges; do not claim
they moved. The boundary-quality change is verified by tests, not timings.

A fourth set (2026-10-18, "Fourth set" in PERFORMANCE.md) measured the custom
byte-level BPE core against the backend on the real GPT-2 and Qwen2.5 artifacts
with the 33.8 KB NFC-normalized reference corpus, on a quiet machine: GPT-2
count 1.28 ms (core) vs 24.33 ms (backend) with 18 vs 108,378 allocations per
run, direct encode 1.67 vs 24.21 ms; Qwen2.5 count 2.36 vs 24.63 ms, encode
2.65 vs 24.80 ms. End-to-end loopback HTTP (`POST /v1/count`, 34 KB body, 8
keep-alive clients, 4 workers + 16 slots): GPT-2 about 1,200 to 1,280 requests/s
with p99 near 8 ms and no rejections; Qwen2.5 566 requests/s, p99 28.7 ms. The
public `Engine::encode` (owned `Encoded`, one `String` per token) is 4.7 ms on
GPT-2; HTTP/CLI/JSONL serialize borrowed tokens and never build it. The JSONL
workload with a real artifact still includes a model load per iteration and is
not a pipeline throughput figure.

## Validation state

Last implementation validation passed:

```powershell
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
```

There are 66 tests: 16 library unit tests (engine cut classification and
cancellation, work-module response fields and batches, byte-level core
pre-tokenizer alternatives, alphabet and added-token matching, the word-cache
shard table), 5 API tests, 6 pipeline tests, 7 CLI tests, 16 engine
integration tests (one pins `auto` as the default core mode), 9 differential
core tests in `tests/bytelevel.rs` (two pin the word cache: identical output
with a full cache, and the 10,000-word bound under concurrent misses)
(the seventh is the soak's hard-case catalogue: added tokens whole, partial,
nested, mis-cased and glued, whitespace and CR/LF mixes, digit runs,
contractions in every case, combining marks and NFC singletons, ZWJ emoji,
half-width forms, unassigned and control scalars, a strided sweep of every
scalar in five contexts, scalar pairs and class-boundary strings; the whole
file takes about two minutes in debug mode with the real artifacts present,
most of it in the engine-selection test's chunking), and the 7 reference
tests described below. Coverage includes core selection and bit-identity
through the public `Engine` API (IDs, tokens, offsets, masks, chunk cuts)
against the backend-only engine, `--core backend|verify`, IDs-only encode, batch records with per-item errors
and nesting/size limits, `/metrics` counters, `schema_version`/`backend`
fields, exact IDs/offsets, special-token options, Literal-mode
suppression of model-internal special pieces (inline Unigram, WordPiece and BPE
tokenizers, refused single-scalar spellings, unchanged unknown handling,
pre-tokenizer split claims for the byte-level regex and BERT), padding/truncation
overrides, byte-level Unicode round trips, malformed inputs, source
reconstruction, tiny/zero budgets, nonmonotonic prefix counts, paragraph/line/
sentence/word boundary preference, CRLF preservation, half-fill fallback,
decode ID validation, cancellation, HTTP capacity/limits/deadlines, ordered
results, bounded read-ahead, open stdin, and output file preservation.

The reference artifact suite (`cargo test --test reference --locked -- --show-output`)
adds 7 tests: three run everywhere (manifest provenance, corpus validity, golden
and corpus hash consistency) and one per artifact that prints `SKIPPED` and
passes when its gitignored model file is absent. With all four models present
it checks 180 golden cases and reports 28 of them as `KNOWN DIFFERENCE`
(documented backend divergences, see "Next useful work"); it takes about a
minute in debug mode on the development machine. Regenerate goldens with
`.setup/oracle-venv/Scripts/python.exe -I tests/fixtures/golden/generate.py`
(tiktoken, sentencepiece, transformers 4.x, tokenizers installed there) and
check for drift with `--check`.

The benchmark debug-build guard was also verified using
`cargo test --bench throughput --locked`, and the release runner was exercised
with `--tokenizer models/gpt2.tokenizer.json --corpus tests/fixtures/corpus/mixed_multilingual.txt`
and with the XLM-R artifact. This handoff is documentation; the implementation
checks above describe the sessions that made the changes, not a fresh test run
caused by reading this file.

## Next useful work

Roadmap agreed with the user on 2026-10-18. The Hugging Face core stays as the
correctness oracle and fallback; any custom fast path is enabled only behind
load-time component gating plus differential tests against that oracle.

1. Real artifacts and golden corpora: DONE on 2026-10-18 (uncommitted in the
   working tree at the time of writing). `models/MANIFEST.json` pins
   `openai-community/gpt2`, `Qwen/Qwen2.5-0.5B`, `google-bert/bert-base-uncased`
   and `FacebookAI/xlm-roberta-base` by revision commit, SHA-256, size, license
   and component types; `python -I models/fetch.py` downloads and verifies them
   (model JSON and `models/oracle/` stay out of Git). `tests/fixtures/corpus/`
   holds 15 committed files (ASCII, code, numbers, combining marks, Hangul,
   full-width/ligatures, Arabic/Devanagari, CJK, emoji, whitespace, control
   bytes, long words, special-token spellings, mixed multilingual, empty).
   `tests/fixtures/golden/<artifact>/` holds 60 committed golden files (180
   cases) produced by `tests/fixtures/golden/generate.py` with independent
   oracles: tiktoken 0.14.0 for GPT-2, sentencepiece 0.2.2 plus the slow
   transformers 4.57.6 XLMRobertaTokenizer for XLM-R, the slow transformers
   4.57.6 BertTokenizer (Google BasicTokenizer+WordpieceTokenizer) for BERT, and
   the transformers fast Qwen2 tokenizer for Qwen2.5 (same backend family as the
   engine, so a cross-version check only). Every case also records whether the
   Python `tokenizers` 0.22.2 bindings agreed with the oracle. `tests/reference.rs`
   verifies manifest hashes and vocab sizes, exact ID equality in Literal and
   Recognize modes with and without post-processor tokens, decode round trips
   (exact for GPT-2, NFC-only for Qwen2.5), and chunk/truncate invariants at
   budgets 1, 7, 64 and 512; it prints `SKIPPED` and passes when a model file is
   absent (`FAST_TOKENIZER_REQUIRE_MODELS=1` makes that a failure). The budget-1
   invariant runs on the first 128 bytes of each file unless
   `FAST_TOKENIZER_EXHAUSTIVE=1`, because budget 1 drives every multi-token
   scalar into the quadratic no-fit fallback (a full-file debug run exceeded ten
   minutes). Result: GPT-2 matches tiktoken on all 30 cases and Qwen2.5 matches
   on all 30; BERT differs on 8 cases and XLM-R on 20. All 28 differing cases
   equal the Python `tokenizers` bindings' output, i.e. they are divergences of
   the Hugging Face Rust backend from the original implementations, recorded
   verbatim in `tests/fixtures/golden/KNOWN_DIFFERENCES.json` (goldens were not
   edited): BERT keeps unassigned code points as `[UNK]` where Google's code
   strips them as controls; the XLM-R `Precompiled` normalizer drops the acute in
   decomposed Vietnamese, never composes conjoining jamo or half-width kana
   voicing marks; `WhitespaceSplit` treats U+0085 as whitespace. A ninth entry
   (the converted Unigram vocab lists `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`
   as score-0 pieces, so Literal mode emitted their IDs for the plain spellings,
   2 cases) was closed the same day by the load-time suppression described in
   the correctness contract; `tests/reference.rs` now also asserts each
   artifact's suppression count and that every special ID still decodes while
   Literal text never yields it. No chunk/truncate/round-trip invariant failed on any
   artifact. `benches/throughput.rs` accepts `--tokenizer` and `--corpus`;
   default reports are unchanged, real-artifact reports add `tokenizer`,
   `vocab_size` and `corpus`, and the JSONL workload then includes a model load
   per iteration.
2. Offsets-guided chunking with boundary preference: DONE on 2026-10-18 (see
   "Performance work already completed" and the second benchmark pair). Also
   landed: borrowed chunk/truncate serialization, decode ID bitset.
3. Profiling instrumentation: DONE on 2026-10-18. `benches/throughput.rs` has a
   counting global allocator (allocations and bytes per iteration), per-iteration
   p50/p95/p99, per-stage backend timings (`stage_normalizer`,
   `stage_pre_tokenizer`, `stage_model`), direct `backend_*`/`core_*` workloads,
   and `--http-load N --requests M` for end-to-end loopback latency of
   `POST /v1/count`. Still open: measuring peak memory for the default bound of
   20 admitted slots × 1 MiB bodies × roughly 10× encode output expansion.
4. Custom byte-level BPE core: DONE on 2026-10-18 for the GPT-2 and Qwen2 regex
   families (`src/bytelevel`, details under "Performance work already
   completed"; numbers in PERFORMANCE.md). Differences from the original plan:
   Llama 3's regex is not implemented yet; the cache is shared and bounded at
   10,000 words, as 16 lock-sharded insert-only tables rather than
   thread-local; `--core verify`
   double-encodes every request rather than a sample; `ignore_merges`
   whole-word lookup exists but no real artifact exercises it yet.
5. Product completion: DONE on 2026-10-18 except the memory bound and Python
   bindings. Identical IDs to the oracles except the 30 documented
   backend-vs-original cases; throughput and tail latency on real artifacts in
   PERFORMANCE.md; release binaries built by `.github/workflows/ci.yml`;
   `schema_version` 1 plus `engine_version`, `backend` and `artifact.core` in
   every response; `ids_only`/`--ids-only` encode; `POST /v1/batch` (256
   operations under one slot, per-item errors); `GET /metrics` (Prometheus
   text). A documented memory bound and Python bindings remain open.

6. Differential soak of the custom core: DONE on 2026-10-18/19.
   `examples/core_soak.rs` compared the core, `Engine` in `CoreMode::Verify`
   and the backend oracle on 55,259,448 inputs (6,907,431 per artifact and
   option combination: corpus files and lines, sliding windows and random
   slices, class-boundary strings, every Unicode scalar in five contexts,
   59,536 scalar pairs and 1,000,000 seeded generator inputs biased toward the
   hard cases), seed 539365400, 8 threads, 80 minutes: zero mismatches in IDs,
   offsets, masks, word ids, tokens or counts, and zero illegitimate declines.
   On that evidence the default core mode returned to `auto` the same day
   ("Differential soak" in PERFORMANCE.md). `tests/bytelevel.rs` gained a
   hard-case catalogue test of the same shape. Rerun the soak after any change
   under `src/bytelevel/` or a `tokenizers` upgrade. The soak's throughput
   probe found the word cache capping multi-threaded throughput; fixed on
   2026-11-10 with a 16-shard, single-hash cache (PERFORMANCE.md, "Fifth
   set"): 4.6x on 8 threads with cache hits, 3.9x with a full cache,
   single-thread hits 1.48x, single-thread `core_count` control unchanged.

Still open from before: tune worker/window/batch sizes from measurements rather
than assumptions, and investigate the pathological no-fit fallback without
weakening correctness for arbitrary normalizers or nonmonotonic token counts.
Newly open from the reference suite: decide whether the remaining backend
divergences in `KNOWN_DIFFERENCES.json` are acceptable or need an engine-side
fix (a sentencepiece-faithful normalizer for Unigram artifacts and a BERT
clean-text step that strips category-Cn code points; the Literal-mode
control-piece case is closed); regenerate goldens with
`.setup/oracle-venv` (Python 3.12, transformers pinned below 5 because 5.x
removed the slow tokenizers) whenever the corpus or a pinned revision changes.

Do not describe the HTTP API or JSONL pipeline as unimplemented; both are done.
Do not restart from the scaffold or replace verified optimizations with the old
full-suffix search. Further performance work should retain comparable baseline
reports and run the relevant correctness tests.

## Local environment details for continuation

Git can report dubious ownership because this workspace was initially created
under a sandbox account. The preceding session used a per-command exception:

```powershell
git -c safe.directory=C:/Users/HP/Desktop/fast-tokenizer status --short
```

Use that exact repository-scoped override if needed; no global wildcard trust
setting was added. Inspect current permissions instead of assuming the original
sandbox/network restrictions still apply. `.setup/`, `target/`, model JSON files,
and `data/` are ignored. A temporary baseline worktree was removed after the
paired measurements; baseline/optimized benchmark executables may remain in
ignored `.setup/` as local artifacts, not portable project requirements.

Keep this handoff and the other project docs current when the implementation,
measured performance, model artifacts, or next steps change.
