# Fast Tokenizer

Rust tokenization and document preparation project.

## Current status

The shared engine and CLI support `info`, `count`, `encode`, `decode`, `truncate`,
and `chunk`, using a local Hugging Face `tokenizer.json` artifact. `serve` exposes
the same operations over HTTP, and `jsonl` processes ordered streaming records.
All results are JSON. Large-document optimization and performance benchmarks
are next.

## Try the CLI

The tiny WordPiece fixture is included for testing; use your model's exported
tokenizer for real counts.

```powershell
cargo run -- info
cargo run -- count --tokenizer tests/fixtures/wordpiece.json --text "hello world!"
cargo run -- encode --tokenizer tests/fixtures/wordpiece.json --text "hello world!"
cargo run -- decode --tokenizer tests/fixtures/wordpiece.json --text '[4,5,6]'
cargo run -- truncate --tokenizer tests/fixtures/wordpiece.json --text "hello world!" --max-tokens 2
cargo run -- chunk --tokenizer tests/fixtures/wordpiece.json --text "hello world!" --max-tokens 2
```

For your own tokenizer, replace the fixture path with `models/tokenizer.json`.
Use `--file document.txt` for UTF-8 files or omit both `--text` and `--file` to
read stdin. Decode accepts a JSON array of unsigned token IDs through the same
input options. Whitespace and newlines in file/stdin input are preserved;
shell pipelines may add a newline or change encoding, so prefer files when
exact source bytes matter. Invalid input exits unsuccessfully with an error on
stderr.

## HTTP API

Start a server (substitute your production tokenizer when ready):

```powershell
cargo run -- serve --tokenizer tests/fixtures/wordpiece.json --workers 4 --queue-capacity 16
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
| `GET /v1/info` | None; reports artifact, options, and total CPU capacity |
| `POST /v1/count` | `{"text":"hello world!"}` |
| `POST /v1/encode` | `{"text":"hello world!"}` |
| `POST /v1/decode` | `{"ids":[4,5,6],"skip_special_tokens":false}` |
| `POST /v1/truncate` | `{"text":"hello world!","max_tokens":2}` |
| `POST /v1/chunk` | `{"text":"hello world!","max_tokens":2}` |
| `POST /v1/process` | Any operation record from the JSONL format below |

Successful operation responses have the same result fields as the corresponding
CLI command, with artifact identity and effective options. Unknown request
fields are rejected. Operation errors use
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
```

Run against a file or leave out `--file` to read stdin:

```powershell
cargo run -- jsonl --tokenizer tests/fixtures/wordpiece.json --file records.jsonl --output results.jsonl --workers 4 --queue-capacity 16
```

Omit `--output` to write compact JSONL to stdout. An output file must be new;
existing files, including the input itself, are rejected. Logs and a final
record/error count go to stderr. A successful record produces
`{"line":1,"result":{"artifact":{...},"options":{...},"count":3}}`.
A failed record produces `{"line":2,"error":{"code":"invalid_request","message":"..."}}`.

Records run concurrently but results are written in input order, flushed after
each record, even while stdin remains open. The pipeline keeps at most
`workers + queue-capacity` records in progress through output flush. Slow input
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
  bytes and its vocabulary size. Encoding responses also report the effective
  special-token options. Record the source model and revision alongside your
  artifact; the hash does not identify its upstream model name.
- Saved padding and truncation are disabled so the entire input is counted.
  BPE dropout is rejected to keep results reproducible.
- Special-token spellings in text are treated literally by default. Use
  `--special-tokens recognize` to enable the artifact's registered special-token
  matching. Separately, `--add-special-tokens` enables post-processing such as
  BOS/EOS insertion. These extra tokens count toward the budget. Literal mode
  disables special matching; the underlying model can still emit a special ID
  if that spelling exists in its normal vocabulary.
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
  Cuts preserve Unicode scalars, but may split words or grapheme clusters.
- Prefix search uses binary probing with an exhaustive fallback when no
  nonempty candidate fits. Token counts can decrease as text grows, so this
  does not promise the longest possible prefix or fullest possible chunks.
  This correctness-first implementation repeatedly encodes prefixes and is
  not yet optimized for large documents; fallback can be quadratic.
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

1. Optimize prefix search and add reproducible throughput/latency benchmarks.
2. Expand fixtures to real model artifacts and cross-implementation comparisons.

The current tests cover known IDs and byte offsets, saved padding/truncation
overrides, special tokens, malformed input, Unicode byte-level round trips,
source reconstruction, nonmonotonic token counts, and CLI file/stdin handling.
API and pipeline tests also cover HTTP limits, overload, slow-body deadlines,
ordered output, bounded read-ahead, record recovery, cancellation, and output
file preservation.
The tokenizer integration uses the [tokenizers Rust API](https://docs.rs/tokenizers/0.22.2/tokenizers/tokenizer/struct.TokenizerImpl.html).
HTTP limits follow [Axum's body-limit API](https://docs.rs/axum/0.8.9/axum/extract/struct.DefaultBodyLimit.html);
CPU work uses [Rayon thread pools](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html).
