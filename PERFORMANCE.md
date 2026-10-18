# Performance

Performance decisions are measured in optimized Rust builds. The release profile
uses thin LTO and one codegen unit. Run `cargo build --release` and use the binary
in `target/release` when measuring runtime; debug builds are for development.

## Reproduce the benchmark

```powershell
cargo bench --bench throughput
cargo bench --bench throughput -- --output benchmarks/local.json
```

The dependency-free runner warms each operation, then measures repeated runs for
at least 300 ms. It emits input bytes, iterations, elapsed seconds, mean time,
and MiB/s. The synthetic WordPiece artifact and input are fixed in the repository,
and the report records the artifact SHA-256, operating system, and CPU identifier.
The ordinary count/encode/truncate/chunk corpus contains 33,800 UTF-8 bytes.
The JSONL test contains 2,000 encode records (102,000 input bytes), four CPU
workers, and sixteen extra slots. It includes pool startup and model loading
per dataset run, and writes to an in-memory sink to isolate processing from disk.

Recorded on 2026-10-18 using Windows x86_64, Intel Core i7-7820HQ (4 cores,
8 logical processors), and rustc 1.99.1. The baseline is commit `7ef0ecb`;
summary reports are [before.json](benchmarks/before.json) and
[after.json](benchmarks/after.json). After compiling both binaries, three runs
of each were alternated without concurrent compilation. The table reports the
median of each workload's three mean timings; the raw runs are in
[`benchmarks/runs`](benchmarks/runs). These short synthetic runs show direction;
repeat with production model artifacts and representative data before drawing
conclusions about production throughput. They do not measure HTTP/network
overhead, disk throughput, allocator counts, peak memory, or tail latency.

| Workload | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Count | 11.69 | 11.78 | 0.99× |
| Encode (public owned-buffer API) | 13.35 | 13.91 | 0.96× |
| Truncate to 128 tokens | 25.32 | 13.44 | 1.88× |
| Chunk at 128 tokens | 537.12 | 83.22 | 6.45× |
| Encode 2,000 JSONL records | 30.02 | 17.36 | 1.73× |

Count and the public owned-buffer encode operation were unchanged and serve as
controls. Their small variations are not reported as improvements. The CLI,
HTTP, and JSONL encode response paths use the new borrowed-buffer serializer.

## Design choices

- Load the tokenizer once for a server or dataset, then share it using `Arc`.
  No model reparsing or full tokenizer cloning occurs per record.
- Use dedicated Rayon CPU workers and bounded admission. Tokio handles I/O;
  CPU work and response serialization stay on the CPU pool. The pipeline uses
  one producer and one ordered consumer rather than a Tokio task per record.
- Deserialize HTTP bodies directly into typed requests and serialize typed
  responses directly into byte buffers. Encoding responses borrow the tokenizer's
  ID, token, mask, and offset buffers while serializing; they avoid an intermediate
  owned encoding copy and `serde_json::Value` tree. CLI, HTTP, and JSONL share
  this response path.
- Count with `encode_fast`, which avoids calculating source offsets.
- Find chunk prefixes using local exponential probes, binary refinement, and
  the previous chunk's byte length as a density estimate. Budget-filling probes
  return immediately. This removes the old repeated full-suffix encoding and
  character-index allocation from the common path.
- Serialize complete JSONL success records on the CPU pool. Batch ready writes
  through a 64 KiB CLI buffer, flushing before idle waits or after 64 records,
  so throughput does not require a flush syscall per record and open streams
  still receive prompt output.
- Keep record size and in-flight work bounded so a larger dataset does not
  require loading the full input or retaining all outputs.

## Remaining work

The general tokenizer contract permits nonmonotonic token counts. When no local
prefix fits, exhaustive fallback remains necessary to avoid incorrectly reporting
that no progress is possible; pathological no-fit inputs can still be quadratic.
Source slices remain independently re-encoded and verified. Unicode scalar
boundaries, full source reconstruction, and cancellation are tested after changes.

Next measurements should use real BPE/Unigram artifacts, longer documents,
normalization-heavy text, and multilingual corpora. Profile allocation counts,
memory peaks, and p50/p95/p99 request latency under concurrent load. Compare
worker/window settings and batch sizes before adding caches, buffer pools,
SIMD, or unsafe code. Keep output correctness checks alongside performance runs.
