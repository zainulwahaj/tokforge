# Fast Tokenizer

Rust tokenization and document preparation project.

## Current status

The shared engine and CLI support `info`, `count`, `encode`, `decode`, `truncate`,
and `chunk`, using a local Hugging Face `tokenizer.json` artifact. `serve` exposes
the same operations over HTTP, and `jsonl` processes ordered streaming records.
All results are JSON. The release benchmark runner and measured optimization
results are described in [PERFORMANCE.md](PERFORMANCE.md). For GPT-2 and
Qwen2-style artifacts, encode and count run on a custom byte-level BPE core that
is bit-identical to the Hugging Face backend and verified differentially; other
artifacts use the backend. Correctness against
four pinned production artifacts (GPT-2, Qwen2.5, BERT, XLM-R) is checked by
`tests/reference.rs` using committed golden IDs from independent oracles; see
[models/README.md](models/README.md) for fetching the artifacts and
`tests/fixtures/golden/KNOWN_DIFFERENCES.json` for the documented divergences
between the Hugging Face backend and the original implementations.

## Try the CLI

The tiny WordPiece fixture is included for testing; use your model's exported
tokenizer for real counts.

```powershell
cargo run --release -- info
cargo run --release -- count --tokenizer tests/fixtures/wordpiece.json --text "hello world!"
cargo run --release -- encode --tokenizer tests/fixtures/wordpiece.json --text "hello world!"
cargo run --release -- decode --tokenizer tests/fixtures/wordpiece.json --text '[4,5,6]'
cargo run --release -- truncate --tokenizer tests/fixtures/wordpiece.json --text "hello world!" --max-tokens 2
cargo run --release -- chunk --tokenizer tests/fixtures/wordpiece.json --text "hello world!" --max-tokens 2
```

For your own tokenizer, replace the fixture path with `models/tokenizer.json`.
Use `--file document.txt` for UTF-8 files or omit both `--text` and `--file` to
read stdin. Decode accepts a JSON array of unsigned token IDs through the same
input options. Whitespace and newlines in file/stdin input are preserved;
shell pipelines may add a newline or change encoding, so prefer files when
exact source bytes matter. Invalid input exits unsuccessfully with an error on
stderr.

`encode --ids-only` returns only IDs and the count. `--core backend` forces the
Hugging Face backend for an artifact the custom core would otherwise serve, and
`--core verify` runs both on every request and fails any request where they
disagree. `info --tokenizer <path>` reports `artifact.core`, the response schema
version, and the locked backend crate version.

## HTTP API

Start a server (substitute your production tokenizer when ready):

```powershell
cargo run --release -- serve --tokenizer tests/fixtures/wordpiece.json --workers 4 --queue-capacity 16
```

The default address is `127.0.0.1:8080`; change it with `--bind IP:PORT`.
The tokenizer and its special-token options are fixed at startup. The server
provides a local HTTP interface without authentication or TLS.

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/v1/info
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/count -ContentType 'application/json' -Body '{"text":"hello world!"}'
```

| Endpoint | JSON request |
| --- | --- |
| `GET /health` | None |
| `GET /metrics` | None; Prometheus text: requests per operation, job outcomes, busy rejections, in-flight jobs, capacity, uptime |
| `GET /v1/info` | None; reports artifact (with the active core), options, schema and backend versions, capacity, in-flight jobs |
| `POST /v1/count` | `{"text":"hello world!"}` |
| `POST /v1/encode` | `{"text":"hello world!"}`, or add `"ids_only":true` for IDs and count only |
| `POST /v1/decode` | `{"ids":[4,5,6],"skip_special_tokens":false}` |
| `POST /v1/truncate` | `{"text":"hello world!","max_tokens":2}` |
| `POST /v1/chunk` | `{"text":"hello world!","max_tokens":2}` |
| `POST /v1/batch` | `{"operations":[{"op":"count","text":"a"},{"op":"encode","text":"b","ids_only":true}]}`: up to 256 operations run in order under one admission slot; each item carries its result or its own `error` |
| `POST /v1/process` | Any operation record from the JSONL format below |

Successful operation responses have the same result fields as the corresponding
CLI command, preceded by `schema_version` (1; bumped only for incompatible
changes), `engine_version`, `backend` (the locked `tokenizers` crate version),
`artifact` (hash, vocabulary size, suppressed special pieces, active core), and
the effective `options`. Unknown request fields are rejected. Operation errors use
`{"error":{"code":"tokenizer_error","message":"..."}}`.

CPU work runs on dedicated Rayon threads, including result serialization.
At most `workers + queue-capacity` requests can be admitted through JSON body
reading and CPU work. Admission happens before buffering a request body; excess
requests receive `503` with `Retry-After: 1`. Completed HTTP responses are handed
to the network independently of CPU admission.

`--max-body-bytes` defaults to 1048576; larger JSON bodies receive `413`.
`--timeout-seconds` defaults to 30 and covers request body reading, queueing,
tokenization, and response construction (not network transfer of the completed
response). Deadlines return `408`, malformed JSON returns `400`, unsupported
content types return `415`, and invalid operations or token IDs return `422`.
Ctrl+C stops accepting connections and cancels pending CPU work.

## Streaming JSONL

Each physical input line is one operation object. Available operations:

```jsonl
{"op":"count","text":"hello world!"}
{"op":"encode","text":"hello café 你好!"}
{"op":"decode","ids":[4,5,6],"skip_special_tokens":false}
{"op":"truncate","text":"hello world!","max_tokens":2}
{"op":"chunk","text":"hello\r\nworld!","max_tokens":2}
{"op":"encode","text":"hello","ids_only":true}
{"op":"batch","operations":[{"op":"count","text":"a"},{"op":"decode","ids":[4]}]}
```

Run against a file or leave out `--file` to read stdin:

```powershell
cargo run --release -- jsonl --tokenizer tests/fixtures/wordpiece.json --file records.jsonl --output results.jsonl --workers 4 --queue-capacity 16
```

Omit `--output` to write compact JSONL to stdout. An output file must be new;
existing files, including the input itself, are rejected. Logs and a final
record/error count go to stderr. A successful record produces
`{"line":1,"result":{"artifact":{...},"options":{...},"count":3}}`.
A failed record produces `{"line":2,"error":{"code":"invalid_request","message":"..."}}`.

Records run concurrently but results are written in input order. Ready records
are batched, with a flush before waiting on input or CPU work, after 64 records,
and at EOF. A result is emitted promptly even while stdin remains open.
The pipeline keeps at most `workers + queue-capacity` records queued, executing,
or waiting to be written, plus the CLI's bounded 64 KiB output buffer. Slow input
and slow output apply backpressure. Memory scales with that window and record
size, rather than total dataset size; results can be larger than their inputs.

`--max-line-bytes` defaults to 1048576, excluding LF (a CR in CRLF counts toward
the limit). Oversized lines are drained without retaining the entire line and
produce `line_too_large`. Blank lines, invalid UTF-8, unknown fields, malformed
JSON, and operation errors produce per-line errors; subsequent records continue.
The final line may omit its newline. Any failed record makes the command exit
unsuccessfully after finishing the stream. I/O errors, broken pipes, and Ctrl+C
stop processing; any output already written is retained and can be partial.

Cancellation is cooperative: queued jobs skip tokenization, and chunking and
prefix search check cancellation between tokenizer calls. An individual call
into the tokenizer library cannot be interrupted midway. The request retains
its CPU slot until that call returns, even if the HTTP deadline has expired.

## Tokenization contract

- Every model-backed response includes the SHA-256 of the original artifact
  bytes, its vocabulary size, and `suppressed_special_pieces` (see below).
  Encoding responses also report the effective special-token options. Record
  the source model and revision alongside your artifact; the hash does not
  identify its upstream model name.
- Saved padding and truncation are disabled so the entire input is counted.
  BPE dropout is rejected to keep results reproducible.
- Special-token spellings in text are treated literally by default. Use
  `--special-tokens recognize` to enable the artifact's registered special-token
  matching. Separately, `--add-special-tokens` enables post-processing such as
  BOS/EOS insertion. These extra tokens count toward the budget.
- Literal mode guarantees that text never yields a registered special-token ID.
  Added-token matching is off, and at load time the engine also makes every
  special spelling that the model lists as an ordinary piece unreachable:
  Unigram pieces are rescored to negative infinity (the unknown penalty is kept
  as it was), WordPiece and WordLevel entries leave the lookup table, and BPE
  loses the merges that would form the spelling (plus the entry itself when the
  model uses `ignore_merges`). IDs, decoding and the artifact hash are
  unchanged; `artifact.suppressed_special_pieces` reports how many spellings
  were touched so results stay reproducible (0 in Recognize mode; 0 for GPT-2
  and Qwen2.5, whose special tokens no merge can form; 4 for BERT; 5 for XLM-R).
  Two limits are explicit. The unknown token's own spelling stays reachable in
  WordPiece, WordLevel and BPE because those models find the unknown ID through
  it, so literal `[UNK]` can still count as unknown text. Loading fails in
  Literal mode when a special spelling is a single scalar or, for Unigram,
  contains a scalar without a piece of its own, because no suppression exists
  there that leaves other text unchanged; use Recognize mode for such artifacts.
- Encode and count are served by the custom byte-level BPE core when the artifact
  qualifies: plain BPE merges over the GPT-2 byte alphabet, the GPT-2 or Qwen2
  pre-tokenizer regex, no normalizer or NFC, added tokens without strip or
  single-word rules, no token-adding post-processor, and nothing suppressed by
  Literal mode. The core is enabled only after a load-time self-check against
  the backend and is bit-identical to it on every input it accepts (IDs, tokens,
  offsets, word indices, mask); inputs it cannot reproduce exactly (text that
  NFC would change, or a case-folded contraction) are served by the backend per
  request. `artifact.core` names the active core; `--core backend` forces the
  backend and `--core verify` runs both on every request and fails it if they
  disagree. `tests/bytelevel.rs` is the differential proof.
- Decode preserves special tokens unless `--skip-special-tokens` is set and
  rejects unknown IDs. Decoding follows the artifact's normalization/decoder;
  arbitrary models do not guarantee an exact text round trip.
- Encode offsets are UTF-8 byte ranges from the tokenizer. Synthetic tokens may
  have `[0, 0]` offsets, and byte-level token offsets may overlap or cover only
  part of a Unicode scalar; do not slice Rust strings blindly at those offsets.
- Truncate returns an original source prefix. Chunk returns contiguous,
  nonoverlapping slices covering the entire source, including whitespace.
  Each output is independently re-encoded and verified against `--max-tokens`.
  Source ranges use an inclusive `start_byte` and exclusive `end_byte`.
  Cuts preserve Unicode scalars and never separate CR from LF. Chunk cuts
  prefer, in order, a blank line, a line break, a sentence end followed by
  whitespace, and a pre-tokenizer word boundary, taking the strongest boundary
  that still fills at least half the budget; otherwise they take the fullest
  subword or scalar boundary. Truncate prefers a word boundary under the same
  rule. Grapheme clusters may still split at the subword level.
- Cuts are guided by one encoding of a bounded window of the remaining text:
  token byte offsets and word indices predict where the budget runs out, and
  the verified density of the previous chunk sizes the next window. Every
  candidate is verified by encoding that slice on its own; a measured overshoot
  shrinks the candidate set before retrying. Truncate first checks whether the
  entire input fits. Token counts can decrease as text grows, so neither
  operation promises the longest possible prefix or fullest possible chunks.
  A rare exhaustive fallback tries the whole remainder, every token boundary in
  the guide window, then the 1,024 scalar boundaries nearest the slice start,
  longest first. Work per cut is bounded by the window plus that limit rather
  than by the document; a prefix that would only fit beyond the limit is
  reported as no fit (`BudgetTooSmall`).
- Zero budgets are accepted: truncate may return empty text, while chunk
  fails if it cannot make progress. Empty source yields zero chunks. If the
  empty prefix's added special tokens exceed the budget, truncate fails.

## Windows prerequisites

- Git and GitHub CLI (`gh`)
- Visual Studio C++ build tools and Windows SDK
- Stable Rust, Cargo, rustfmt, and Clippy (installed using rustup)
- Network access to Rust distribution servers, crates.io, and GitHub

On this machine, Rust and its formatting/lint tools are now installed alongside
Git, GitHub CLI, Visual C++ tools, and the Windows SDK. GitHub access requires an
authenticated `gh` account.

## Complete setup

Open PowerShell in this project folder in an environment with network access:

```powershell
.\scripts\setup.ps1
```

The script downloads the official Rust installer and verifies its published
SHA-256 checksum, installs the stable toolchain plus rustfmt and Clippy, fetches
dependencies, checks the build, and creates and pushes a **private** GitHub
repository using the active `gh` account. If `gh` requires authentication, it
stops with the login command; run that command and rerun setup. It does not
replace an existing remote or force-push.

To install and verify prerequisites without creating the GitHub repository:

```powershell
.\scripts\setup.ps1 -SkipGitHub
```

After Rust is installed, open a fresh terminal if Cargo is not yet on PATH.

```powershell
cargo run -- info
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Use `C:\Users\HP\Desktop\fast-tokenizer` as the workspace folder for this
project. Changing a command's working directory does not change the app's
selected workspace or its permission scope.

## Next implementation steps

1. Keep publishing real-artifact throughput and tail latency from the runner
   (`--tokenizer`, `--corpus`, `--http-load`) on a quiet machine; see PERFORMANCE.md.
2. Decide whether the remaining backend divergences in `tests/fixtures/golden/KNOWN_DIFFERENCES.json`
   (Precompiled-normalizer composition, U+0085 as whitespace, category-Cn stripping)
   need engine-side fixes; the Literal-mode control-piece leak on XLM-R is closed by
   load-time suppression.
3. Extend the custom core to more byte-level families (Llama 3 regex, `ignore_merges`
   artifacts) under the same differential gate, and tune worker/window/batch sizes
   from measurements.

The current tests cover known IDs and byte offsets, saved padding/truncation
overrides, special tokens, Literal-mode suppression of model-internal special
pieces (Unigram, WordPiece, BPE) with unknown-handling controls, malformed
input, Unicode byte-level round trips,
source reconstruction, nonmonotonic token counts, and CLI file/stdin handling.
API and pipeline tests also cover HTTP limits, overload, slow-body deadlines,
ordered output, bounded read-ahead, record recovery, cancellation, and output
file preservation. The reference suite (`cargo test --test reference -- --show-output`)
compares the engine with tiktoken, sentencepiece, and the slow transformers
tokenizers on 15 corpus files per artifact in both special-token modes, checks
decode round trips where the model allows them, and verifies chunk/truncate
invariants at budgets 1, 7, 64, and 512; it prints `SKIPPED` and passes when the
gitignored model files are absent.
The tokenizer integration uses the [tokenizers Rust API](https://docs.rs/tokenizers/0.22.2/tokenizers/tokenizer/struct.TokenizerImpl.html).
HTTP limits follow [Axum's body-limit API](https://docs.rs/axum/0.8.9/axum/extract/struct.DefaultBodyLimit.html);
CPU work uses [Rayon thread pools](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html).
