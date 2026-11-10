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

## Second pair: offsets-guided cuts

Recorded on 2026-10-18 on the same machine and toolchain. The baseline is
commit `b832a70`, the optimized binary of the first pair, rebuilt from a
detached worktree into a separate target directory so the baseline build never
saw the new source. Summary reports are
[guided-before.json](benchmarks/guided-before.json) and
[guided-after.json](benchmarks/guided-after.json); raw runs are
`benchmarks/runs/guided-*.json`. Same method: three alternating runs of each
binary, no concurrent compilation, median of the three mean timings.

| Workload | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Count | 12.16 | 11.61 | 1.05× |
| Encode (public owned-buffer API) | 12.90 | 13.36 | 0.97× |
| Truncate to 128 tokens | 13.07 | 11.28 | 1.16× |
| Chunk at 128 tokens | 82.83 | 33.06 | 2.51× |
| Encode 2,000 JSONL records | 12.31 | 15.09 | 0.82× |

The JSONL control moved against the change although none of its code changed,
so the whole pair was repeated immediately
([guided-recheck-before.json](benchmarks/guided-recheck-before.json),
[guided-recheck-after.json](benchmarks/guided-recheck-after.json)):

| Workload | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Count | 11.42 | 11.49 | 0.99× |
| Encode (public owned-buffer API) | 12.56 | 13.13 | 0.96× |
| Truncate to 128 tokens | 13.10 | 11.66 | 1.12× |
| Chunk at 128 tokens | 76.85 | 33.77 | 2.28× |
| Encode 2,000 JSONL records | 14.53 | 13.08 | 1.11× |

Read the two sets together: chunking is 2.3× to 2.5× faster and truncate about
1.1× faster. Count, encode, and JSONL are unchanged controls whose per-run
ranges overlap across binaries; JSONL spans roughly 12 to 15 ms in every set
because it includes pool and runtime startup. The chunk gain matches the cost
model: about one and a half guide passes plus one verification pass over the
document instead of a logarithmic number of full re-encodes per chunk. The same
change stops chunks from ending mid-word; that quality effect is covered by the
engine tests, not by this benchmark. These numbers remain fixture-only.

## Third pair: Literal-mode special-piece suppression (load-time only)

Recorded on 2026-10-18 on the same machine and toolchain. The change makes
every registered special-token spelling unreachable inside the model when
`--special-tokens literal` is in effect (see the tokenization contract in
README.md); it rebuilds the model once at load and touches no per-request path.
Both binaries were compiled from one isolated copy of the working tree with the
committed bench source, the baseline with the change reversed, then three
alternating runs of each without concurrent compilation by this session (another
session was compiling on the machine, so ranges are wider than in the pairs
above). Medians of the three mean timings, with the per-run range:

| Workload | Before (ms) | After (ms) | Before range | After range | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Count | 12.11 | 11.89 | 11.95–12.83 | 11.39–12.15 | 1.02× |
| Encode (public owned-buffer API) | 12.97 | 12.99 | 12.30–15.47 | 12.71–13.53 | 1.00× |
| Truncate to 128 tokens | 11.28 | 11.26 | 11.21–13.41 | 11.20–11.97 | 1.00× |
| Chunk at 128 tokens | 32.02 | 32.02 | 31.78–36.01 | 30.76–36.60 | 1.00× |
| Encode 2,000 JSONL records | 12.76 | 13.03 | 11.59–15.55 | 10.77–14.06 | 0.98× |

All five workloads are unchanged controls; every range overlaps. The fixture
is a WordPiece artifact whose three non-`[UNK]` special entries are removed
from a ten-entry map, so even the JSONL workload's per-iteration model load is
unaffected. The raw reports for this pair were not added to `benchmarks/`.

The cost is paid once per process, when the artifact loads. In-process
`Engine::from_bytes` timings (release, seven runs each, sorted, same noisy
conditions) on the pinned reference artifacts:

| Artifact | Recognize (ms) | Literal (ms) | What Literal rebuilds |
| --- | ---: | ---: | --- |
| xlm-roberta-base (Unigram, 250,002 pieces) | 1077–2002 | 1682–3774 | re-reads the 9 MB file for `unk_id`, rescores 5 pieces, rebuilds the trie |
| bert-base-uncased (WordPiece, 30,522 entries) | 55–65 | 75–97 | rebuilds the lookup map without 4 entries |
| gpt2 (byte-level BPE) | 146–168 | 145–162 | nothing: no merge forms `<|endoftext|>` |
| qwen2.5-0.5b (byte-level BPE) | 441–604 | 422–459 | nothing: specials are added tokens outside the model vocabulary |

Roughly +0.6 s on the minimums for XLM-R and +20 ms for BERT. The Unigram
rebuild is the only notable cost: the backend keeps `unk_id` private and its
scores immutable, so the engine has to scan the document once more and build a
second trie before dropping the first. Servers and dataset runs load once, so
this does not affect throughput; one-shot CLI calls on a large Unigram
artifact pay it every time. Exposing those fields upstream, or caching a
rebuilt model, would remove it.

## Fourth set: custom byte-level BPE core and real artifacts

Recorded on 2026-10-18 on the same machine and toolchain, release profile, with
no concurrent compilation. Input is the 15 reference corpus files concatenated
and NFC-normalized (33,813 bytes; `tests/fixtures/corpus/`), so the custom core
accepts every byte of it. Each workload runs for at least 300 ms after a warm-up;
the runner now reports per-iteration p50/p95/p99 and heap allocations per
iteration from a counting global allocator. `backend_*` call the Hugging Face
tokenizer directly, `core_*` call the custom core directly, and the unprefixed
rows go through `Engine`, which routes to the core for these two artifacts.

| GPT-2 (50,257 vocab), 33.8 KB | Mean (ms) | p99 (ms) | Allocations / iteration |
| --- | ---: | ---: | ---: |
| `backend_count` (tokenizers 0.22.2 `encode_fast`) | 24.33 | 25.04 | 108,378 |
| `core_count` | 1.28 | 1.35 | 18 |
| `backend_encode` (ids, tokens, offsets) | 24.21 | 25.89 | 108,378 |
| `core_encode` (ids, offsets, word ids; tokens borrowed) | 1.67 | 1.80 | 18 |
| `count` via `Engine` | 1.29 | 1.47 | 18 |
| `encode` via `Engine` (public owned `Encoded`, one `String` per token) | 4.69 | 5.00 | 18,523 |
| `truncate_128` via `Engine` | 1.43 | 1.91 | 53 |
| `chunk_128` via `Engine` | 6.19 | 6.64 | 6,449 |
| `stage_pre_tokenizer` (backend regex alone) | 16.90 | 17.26 | 70,162 |
| `stage_model` (backend merges alone) | 5.53 | 6.80 | 38,079 |

| Qwen2.5-0.5B (151,665 vocab, NFC), 33.8 KB | Mean (ms) | p99 (ms) | Allocations / iteration |
| --- | ---: | ---: | ---: |
| `backend_count` | 24.63 | 27.28 | 117,570 |
| `core_count` | 2.36 | 2.91 | 22 |
| `backend_encode` | 24.80 | 29.65 | 117,570 |
| `core_encode` | 2.65 | 2.82 | 22 |
| `count` via `Engine` | 2.45 | 3.55 | 22 |
| `encode` via `Engine` (owned `Encoded`) | 4.25 | 5.20 | 14,008 |
| `truncate_128` via `Engine` | 2.67 | 4.62 | 55 |
| `chunk_128` via `Engine` | 7.00 | 10.98 | 5,106 |
| `stage_normalizer` (backend NFC alone) | 1.52 | 1.68 | 40 |
| `stage_pre_tokenizer` | 18.52 | 22.31 | 82,653 |
| `stage_model` | 6.06 | 7.83 | 34,863 |

Counting is 19x faster than the backend on GPT-2 and 10x on Qwen2.5, with 18 to
22 allocations per 34 KB input instead of over 100,000. Direct encoding is 14x
and 9x faster. The stage rows explain where the backend spends its time: the
regex pre-tokenizer with its normalized-string alignment bookkeeping costs three
times the merges themselves, and both allocate per word and per token; the core
pre-tokenizes in place over byte ranges and merges raw bytes with a bounded
word cache. The public `Engine::encode` is slower than `core_encode` only because
it still returns an owned `Encoded` with one `String` per token; the HTTP, CLI
and JSONL paths serialize borrowed token strings and never build it. The Qwen
core pays an NFC quick-check per request (`is_nfc_quick`, about 0.1 ms here) and
declines inputs that NFC would change, which the backend then serves.

End-to-end HTTP latency (`--http-load 8 --requests 4000`, loopback TCP, keep-alive
clients, 4 workers plus 16 queued slots, the same 33.8 KB body on `POST /v1/count`):

| Artifact | Requests/s | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) | non-200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 | 1,279 | 6.25 | 7.31 | 7.66 | 9.09 | 0 |
| Qwen2.5-0.5B | 566 | 13.05 | 23.96 | 28.74 | 49.88 | 0 |

Each request parses a 34 KB JSON body before the 1.3 ms (GPT-2) or 2.4 ms (Qwen)
of tokenization, and eight clients share four workers, so latency is roughly two
service times plus parsing. A repeat of the GPT-2 run gave 1,193 requests/s with
p99 8.37 ms, so treat the figures as a range, not a point.

Overload (`--http-load 32 --requests 4000`, GPT-2, 32 clients against the default
capacity of 20 admitted jobs): 2,170 of 4,000 requests were rejected with `503`
and `Retry-After: 1` before their bodies were read, each in well under a
millisecond (p50 of all responses 0.84 ms); the accepted requests kept a p99 of
26.7 ms and a maximum of 32.4 ms, and the server never queued beyond its
capacity. Because a rejected request's body is never read, the server closes
that connection after the response, which the client sees as a reset; the load
client counts those as failed requests and reconnects after a 5 ms backoff. This
is the intended shape: bounded memory and latency for admitted work, explicit
shedding for the rest. Qwen's wider tail comes from its larger merge table
and vocabulary thrashing the cache on this diverse corpus. These are single-machine
loopback figures on a 4-core laptop and exclude network transfer.

The default fixture run (synthetic WordPiece, backend only) stayed within the
range of the earlier pairs: count 13.6 ms, encode 14.3 ms, truncate 11.0 ms,
chunk 33.9 ms, JSONL 15.3 ms. The WordPiece fixture is never served by the core,
so it is a control for the engine changes around it (shared `Encoding` type,
header fields, bounded fallback).

The JSONL workload with a real artifact (316 ms for GPT-2, 1,019 ms for Qwen)
is dominated by loading the artifact per iteration, as before; it is not a
throughput figure for the pipeline.

## Differential soak: custom core against the backend

Run on 2026-10-18/19 on the development machine (release profile, 8 threads,
`tokenizers` 0.22.2) with `examples/core_soak.rs`:

```powershell
cargo run --release --example core_soak -- --output soak.json
```

The soak loads every artifact in `models/MANIFEST.json` that the core accepts
(GPT-2 and Qwen2.5-0.5B; BERT and XLM-R are skipped as unsupported) and, for
each of the four option combinations (`SpecialTokens::Literal` or `Recognize`,
`add_special_tokens` off or on), runs every input through four things: the
core directly (`ByteLevelBpe`, so declines are counted and justified), an
`Engine` in `CoreMode::Verify` (the production path, which double-encodes and
fails on disagreement), an `Engine` in `CoreMode::Backend` (cross-checked
against the oracle on the corpus inputs) and a backend `Tokenizer` configured
exactly as the engine configures it (the oracle). Wherever the core served the
input, IDs, byte offsets, special-token mask, word ids, token strings and the
count-only path had to equal the oracle's `encode`, and the oracle's
`encode_fast` count had to equal its `encode` length. A decline is only
accepted when NFC really changes the text or an apostrophe is followed by a
non-ASCII scalar; anything else counts as an illegitimate decline.

Inputs per artifact and combination, 6,907,431 in all, indexed so that any one
of them is reproducible from the seed (539365400, `0x20261018`) and its index:

| Category | Inputs | What |
| --- | ---: | --- |
| corpus | 633 | the 15 corpus files whole, each line, and each line with its newline |
| window | 145,108 | sliding windows of 1 to 1,024 scalars over every file, stride a third of the size |
| slice | 100,000 | random slices at scalar boundaries, 1 to 8,192 scalars, skewed short |
| boundary | 41,834 | both orders of every adjacent pair of regex-class runs (1,610 runs) in 12 contexts |
| scalar | 5,560,320 | every Unicode scalar (1,112,064) alone and as `'c`, ` c`, `c\n`, `ac1` |
| pair | 59,536 | every ordered pair from 244 hard scalars (whitespace, controls, marks, ZWJ, singletons, byte-alphabet characters, unassigned values) |
| random | 1,000,000 | seeded generator biased toward whitespace runs, CR/LF mixes, digit runs, mixed-case contractions, many scripts, combining marks, ZWJ emoji, half-width forms, unassigned and control code points, byte-alphabet characters, and the artifact's added tokens whole, partial, nested, mis-cased and glued; 1 to 1,500 pieces, occasionally mutated |

Results (every row is the full 6,907,431 inputs; GPT-2 rows carry 439.0 MB of
text, Qwen2.5 rows 446.8 MB because its generator pieces include 22 added
tokens):

| Artifact | Special tokens | `add_special_tokens` | Wall (s) | Inputs/s | Served by core | Declined (NFC / contraction) | Mismatches |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 | literal | off | 644.9 | 10,711 | 100% | 0 / 0 | 0 |
| GPT-2 | literal | on | 609.7 | 11,329 | 100% | 0 / 0 | 0 |
| GPT-2 | recognize | off | 579.0 | 11,931 | 100% | 0 / 0 | 0 |
| GPT-2 | recognize | on | 719.1 | 9,606 | 100% | 0 / 0 | 0 |
| Qwen2.5-0.5B | literal | off | 543.5 | 12,709 | 78.03% | 341,822 / 1,175,443 | 0 |
| Qwen2.5-0.5B | literal | on | 537.7 | 12,846 | 78.03% | 341,822 / 1,175,443 | 0 |
| Qwen2.5-0.5B | recognize | off | 566.3 | 12,197 | 78.01% | 341,822 / 1,176,953 | 0 |
| Qwen2.5-0.5B | recognize | on | 564.5 | 12,235 | 78.01% | 341,822 / 1,176,953 | 0 |

Total: 55,259,448 inputs in 4,786 s (80 minutes), zero mismatches of any kind,
zero illegitimate declines, and the `Verify` engine never raised
`CoreMismatch`. The soak was not cut short; the full report is
`soak-full.json` from that run (kept outside the repository; the table above
is its summary). A `--quick` smoke run of the same shape (1,596,632 inputs,
187 s) was also clean. The 22% of inputs that the Qwen2.5 core declined are
dominated by the sweep's `'c` context: every non-ASCII scalar after an
apostrophe (1,111,936 inputs) is a declared fallback because the regex
engine's case folding for `(?i:'s|'t|...)` is not reproduced; the remaining
63,507 contraction declines and the 341,822 NFC declines come from the
boundary, pair and random categories, and all were justified. GPT-2 (no
normalizer, case-sensitive contractions) declined nothing. The largest single
input produced 19,032 tokens on GPT-2.

The harness itself was checked with `--self-test --quick`, which feeds the
core the opposite special-token mode from the oracle: 45,414 of the 1,596,632
inputs then mismatched (every input carrying a special spelling, in every
combination), each was printed as a Rust string literal with both ID sequences
and the first differing index, and the shrinker reduced the first corpus
mismatch (the 1,728-byte `special_tokens.txt`) to `"<|endoftext|>"` on GPT-2
and to `"<|endoftext|>"` and `"<|im_end|>"` on Qwen2.5, the shortest spellings
that still differ. The engines are not swapped in that mode, so the engine
path stays clean, and the self-test exits successfully only when mismatches
were produced.

Throughput of the soak itself is bounded by the backend, which runs three to
five times per input (encode, `encode_fast`, and again inside the `Verify`
engine); with the engine path off, one thread compares about 4,300 random
inputs per second, two threads 6,100, four 10,300 and eight 7,800, so the
default of eight threads is past the knee on this 4-core machine for the long
random inputs.

A separate probe (`core.count` on one 140 KB text of 40,000 distinct short
words, so the bounded word cache is full and every lookup misses, against a
1,000-word text that fits) showed why: with cache hits the core scales from
6.5 MB/s on one thread to about 17.5 MB/s on four and eight threads, but with
the cache full it stops at about 7.5 MB/s from two threads upward (3.9 MB/s
on one), because `ByteLevelBpe::word_ids` takes the cache's exclusive write
lock on every miss even when the cache can no longer accept entries; the
backend `Tokenizer` scales no better on the same text (0.4 MB/s per thread,
0.8 to 2.2 MB/s in total on four and eight threads, consistent with allocator
contention from its per-word allocations). These are single runs on a shared
machine, not paired benchmarks. The fix, and the read-lock contention it
turned up, are measured in "Fifth set" below.

Recommendation: `auto` is the default core mode again. The gating self-check
plus this soak are the evidence; `--core verify` remains available for anyone
who wants the production-time check at the cost of one backend encode per
request. Rerun the soak (same seed, or a new one) after any change under
`src/bytelevel/`, after a `tokenizers` upgrade, or before enabling the core
for a new artifact family; `tests/bytelevel.rs` carries a hard-case catalogue
of the same shape that runs in seconds. `--self-test` swaps the core's
special-token mode so mismatches appear, which checks the reporting and the
shrinker rather than the core.

## Fifth set: word-cache scaling (sharded, single-hash cache)

Measured 2026-11-10 on the development laptop (i7-7820HQ, 4 cores / 8
threads). Baseline `7e6b7a1` (one `RwLock<HashMap>` cache, scaling workloads
added) against candidate `ca4b32b`, each built into its own target directory,
alternated five times with `--tokenizer models/gpt2.tokenizer.json --corpus
tests/fixtures/corpus/mixed_multilingual.txt`. Medians of `mib_per_second`;
raw reports in `benchmarks/runs/scaling-{before,after}-{1..5}.json`,
summaries in `benchmarks/scaling-{before,after}.json`.

`core_count_{hit,full}_{N}t` run one shared `ByteLevelBpe` on N threads over
letter-only words: 140,002 bytes cycling through 1,000 distinct words that all
fit in the cache (`hit`), or 181,722 bytes of 40,000 distinct words that
overflow it (`full`). Throughput is the total over all threads. Each timed
iteration spawns its threads, which understates absolute multi-thread numbers
by a few percent on Windows; both sides share the harness, so the comparison
holds.

| Workload | Before (MiB/s) | After (MiB/s) | Change | Per-run ranges |
| --- | ---: | ---: | ---: | --- |
| hit, 1 thread | 32.7 | 48.5 | 1.48x | 32.2–33.7 / 46.1–49.4 |
| hit, 2 threads | 28.1 | 62.9 | 2.23x | 27.2–32.0 / 57.3–63.8 |
| hit, 4 threads | 24.9 | 91.2 | 3.66x | 24.3–28.2 / 80.2–97.1 |
| hit, 8 threads | 23.7 | 109.4 | 4.62x | 21.9–28.5 / 93.9–131.2 |
| full, 1 thread | 12.6 | 12.7 | 1.00x | 9.9–12.7 / 8.4–13.2 |
| full, 2 threads | 19.0 | 23.4 | 1.23x | 10.2–19.6 / 11.4–24.2 |
| full, 4 threads | 14.4 | 25.6 | 1.79x | 13.5–14.9 / 23.4–39.3 |
| full, 8 threads | 9.9 | 38.8 | 3.91x | 9.0–10.7 / 21.0–57.4 |
| `core_count` (control, 1 thread) | 53.2 | 54.3 | 1.02x | 33.6–53.7 / 41.8–57.4 |
| `core_encode` (control) | 34.4 | 37.5 | 1.09x | 23.0–34.9 / 27.0–39.0 |

What changed and why:

- With one lock, cache *hits* got slower as threads were added (32.7 MiB/s on
  one thread, 23.7 on eight): every lookup does two atomic operations on the
  same lock word, so the threads fight over one cache line. The write lock on
  misses, which the soak probe pointed at, was only part of it.
- The cache is now 16 shards, chosen by the top four bits of the word's
  FxHash, each behind its own `RwLock` aligned to 128 bytes. A miss takes its
  shard's write lock only if the shard had room under the read lock, and
  re-checks under the write lock. The total bound is unchanged: 16 × 625 =
  10,000 words.
- A first version kept a `HashMap` per shard and cost 11% on single-thread
  `core_count` (56.1 → 49.9 MiB/s, non-overlapping ranges), because each word
  was hashed twice. Each shard is now an insert-only open-addressing table of
  1,024 slots that stores the hash, so one FxHash picks both shard and slot.
  Single-thread hits also got 1.48x faster than the baseline, which hashed
  once too; the gain comes from the simpler table (a direct probe on a stored
  hash instead of `HashMap`'s generic lookup), and which part of it matters
  most was not measured.
- The slot comes from hash bits 40..50, not the low bits. For short words such
  as `" the1"`, FxHash's low bits depend only on the first input bytes;
  indexing by them gave probe runs of 208 slots in a full shard, which showed
  up as a 4x slowdown on `full_1t` in an intermediate run.
  `shard_table_spreads_real_words_across_slots` pins this.

The `full` workloads stay well below `hit` because most words miss and go
through the merge loop; that cost is per word, not contention. The 8-thread
`full` range is wide (21–57 MiB/s) on this laptop, which runs at 4 physical
cores; the median is the claim.

Costs of the new cache, measured or derived:

- Each shard stops at 625 words even while others have room, so a working set
  near the bound is cached slightly less than before: in simulation the first
  shard filled after 9,350 to 9,820 distinct words. Words beyond that are
  merged on every occurrence. This changes speed only, never output.
- The tables are allocated at load: 16 × 1,024 slots × 40 bytes = 640 KiB per
  engine, where the old map started empty. A full old cache used about the
  same (hashbrown at 10,000 entries is roughly 530 KiB).

Verification: the differential suite (`tests/bytelevel.rs`, now with
full-cache and concurrent-bound tests) and the quick soak (1,596,632 inputs,
0 mismatches). The full 55 M-input soak was not rerun for this change: lookups
compare the whole word after the hash, so a cache hit can only return that
word's own merge output, and the full-cache differential test passes. It is
rerun before the public release.

## Sixth set: head-to-head against tiktoken and Hugging Face tokenizers

Measured 2026-11-10 on the development laptop with
`python python/benchmarks/compare.py --repeat 9` (tokforge 0.1.0, tiktoken
0.14.0, tokenizers 0.22.2, Python 3.14). Raw results:
`benchmarks/python-compare.json`. All three libraries load the same tokenizer
(tiktoken by encoding name, the others from the same `tokenizer.json`), and the
script stops before timing unless every library returns ids identical to
tiktoken's, for the whole dataset and for the batch. Datasets come from the
repository: its Markdown docs (`english`), its Rust sources (`code`) and the
hard-case corpus (`multilingual`). `single` encodes the dataset as one string on
one thread; `batch` cuts it into ~4 KB documents on line boundaries and uses all
cores (tiktoken `num_threads=8`; tokforge and tokenizers on their rayon pools).

| Encoding | Dataset | Workload | tokforge | tiktoken | tokenizers | vs tiktoken |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| gpt2 | english | single | 34.5 | 7.8 | 1.2 | 4.44x |
| gpt2 | english | batch | 54.0 | 15.1 | 4.3 | 3.57x |
| gpt2 | code | single | 22.0 | 5.8 | 0.9 | 3.79x |
| gpt2 | code | batch | 45.3 | 15.0 | 4.1 | 3.02x |
| gpt2 | multilingual | single | 14.5 | 6.3 | 1.2 | 2.29x |
| gpt2 | multilingual | batch | 24.6 | 8.5 | 3.8 | 2.88x |
| cl100k_base | english | single | 36.0 | 7.9 | 1.2 | 4.56x |
| cl100k_base | english | batch | 59.1 | 14.4 | 4.6 | 4.09x |
| cl100k_base | code | single | 25.5 | 5.5 | 1.0 | 4.62x |
| cl100k_base | code | batch | 54.4 | 14.4 | 4.6 | 3.78x |
| cl100k_base | multilingual | single | 11.6 | 5.4 | 1.1 | 2.16x |
| cl100k_base | multilingual | batch | 19.0 | 8.2 | 3.5 | 2.30x |
| o200k_base | english | single | 31.8 | 12.0 | 1.2 | 2.65x |
| o200k_base | english | batch | 53.8 | 17.6 | 4.8 | 3.06x |
| o200k_base | code | single | 22.1 | 9.1 | 1.0 | 2.42x |
| o200k_base | code | batch | 50.6 | 20.5 | 4.6 | 2.47x |
| o200k_base | multilingual | single | 10.9 | 7.7 | 1.2 | 1.41x |
| o200k_base | multilingual | batch | 20.4 | 8.8 | 3.6 | 2.32x |

MB/s of UTF-8 input, medians of 9 runs after a warm-up. Single runs per
configuration on a laptop, not paired; a trial run with 3 repeats landed within
a few percent of every figure. The multilingual set is the deliberately hostile
corpus, which is why it is the slowest for everyone; o200k on it is the closest
result.

Two notes on what the numbers do not show. The batch figures include building
Python lists of ids, which runs on one thread after the parallel part, so the
batch speed-up over single is smaller than the core's own scaling ("Fifth
set"). Before the contraction change below, cl100k and o200k on the
`multilingual` set ran at Hugging Face backend speed, because one `l'été` in the
corpus sent the whole request to the backend.

Case-folded contractions. The case-insensitive contraction alternative
`(?i:'s|'t|'re|'ve|'m|'ll|'d)` used to make the core decline any request with an
apostrophe followed by a non-ASCII scalar, since the regex engine's case folding
was not modelled. `examples/unicode_tables.rs` now asks Oniguruma which non-ASCII
scalars the pattern can match in any letter position; over all of Unicode the
answer is U+017F `ſ`, which folds to `s`. The core treats it as `s`, every other
non-ASCII scalar as no match, and still declines for any scalar the generated
`CONTRACTION_FOLDS` table lists that the code does not model. Hard-case declines
on cl100k went from 5,300 to 0; the quick soak afterwards (all four byte-level
artifacts, 3,193,264 inputs) had no mismatch and the core served every input
for GPT-2, cl100k and o200k.

Memory, from `python python/benchmarks/memory.py` (`benchmarks/memory.json`),
each library in its own process:

| Encoding | Library | After load (MiB) | Peak while loading | Encoding a batch adds |
| --- | --- | ---: | ---: | ---: |
| gpt2 | tokforge | 35.2 | 49.6 | 0.0 |
| gpt2 | tiktoken | 44.3 | 52.4 | 0.0 |
| cl100k_base | tokforge | 48.5 | 81.0 | 0.0 |
| cl100k_base | tiktoken | 58.8 | 60.0 | 1.4 |
| o200k_base | tokforge | 78.1 | 145.7 | 0.0 |
| o200k_base | tiktoken | 89.1 | 92.4 | 0.7 |

tokforge keeps the Hugging Face tokenizer loaded as its fallback and still holds
less than tiktoken after loading. The load-time peak is higher on the large
vocabularies because the core parses its own copy of `tokenizer.json` (every
merge as a string) next to the backend's; parsing the file once would remove
most of it. The HTTP server (GPT-2, 4 workers and 16 queue slots, 20 concurrent
`POST /v1/encode` with 1,017,001-byte bodies and 8.2 MB responses each) went
from 20.6 MiB idle to a 99.7 MiB peak and answered all 20 with 200 in 0.44 s.

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
- Make Literal mode's guarantee a load-time property, not a per-request check:
  special spellings the model could emit are suppressed inside the model once
  (Unigram scores, WordPiece entries, BPE merges), so encoding runs the
  unchanged backend fast path and the vocabulary map is built once for both
  the identity fields and the decode bitset.
- Find chunk and truncate cuts from one guide encoding of a bounded window of
  the remaining text. Token byte offsets and word indices predict how many
  tokens end at each candidate cut; the source text classifies each cut as a
  paragraph, line, sentence, word, or subword boundary; the strongest level that
  fills at least half the budget is verified by encoding that slice on its own.
  On overflow the candidate set shrinks by the measured overshoot and retries.
  The window is half again the predicted slice, sized from the previous chunk's
  verified density (or the whole-text count for truncate), and doubles when
  everything in it fits, so memory is bounded per chunk, not per document. This
  replaces a logarithmic number of full re-encodes per chunk with roughly one
  and a half guide passes plus one verification pass over the document, and
  stops cutting words in half.
- Chunk and truncate responses serialize borrowed slices of the request text,
  so a chunk response never holds a second copy of the document. Decode checks
  IDs against a bitset built once at load rather than a per-ID string lookup.
- Serialize complete JSONL success records on the CPU pool. Batch ready writes
  through a 64 KiB CLI buffer, flushing before idle waits or after 64 records,
  so throughput does not require a flush syscall per record and open streams
  still receive prompt output.
- Keep record size and in-flight work bounded so a larger dataset does not
  require loading the full input or retaining all outputs.

## Remaining work

The general tokenizer contract permits nonmonotonic token counts. When no guided
cut verifies, the exhaustive fallback tries the whole remainder, every token
boundary in the guide window, then the 1,024 scalar boundaries nearest the slice
start, longest first. Work per cut is bounded by the window plus that limit; a
prefix that would only fit beyond it is reported as no fit, which is the
documented trade for bounded latency. Source slices remain independently
re-encoded and verified.
Unicode scalar boundaries, CRLF preservation, boundary preference, full source
reconstruction, and cancellation are tested after changes.

Still to measure: peak memory under the default admission bound, longer
documents, and the remaining backend-only artifacts (BERT, XLM-R) under load.
Compare worker/window settings and batch sizes from measurements before adding
SIMD or unsafe code. The custom core could extend to the Llama 3 regex family
and to `ignore_merges` artifacts under the same differential gate. Keep output
correctness checks (`tests/bytelevel.rs`, `tests/reference.rs`) alongside
performance runs. `--core auto` is the default again since the differential
soak passed clean ("Differential soak" above); `--core verify` adds one backend
encode per request on top of the core for anyone who wants the production-time
check.
