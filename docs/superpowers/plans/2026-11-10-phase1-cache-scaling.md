# Phase 1: Multi-thread scaling of the byte-level core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the write-lock bottleneck that caps the core at ~2x one thread once its word cache is full, and prove it with paired measurements.

**Architecture:** `ByteLevelBpe::word_ids` (`src/bytelevel/mod.rs:714`) takes the cache's exclusive write lock on every miss, even when the cache is full and nothing will be inserted. Record "there is room" while the read lock is held and take the write lock only then (re-checking capacity under it). Add bench workloads that drive 1/2/4/8 threads over a cache-friendly text and a cache-overflowing text, then run the project's paired before/after method.

**Tech Stack:** Rust (std `RwLock`, `std::thread::scope`), existing harness-free bench runner `benches/throughput.rs`, `examples/core_soak.rs`.

**Spec:** `docs/superpowers/plans/2026-11-10-resume-ready-roadmap.md` (Phase 1 row and Global constraints).

## Global Constraints

- Any change under `src/bytelevel/` reruns `tests/bytelevel.rs` and `cargo run --release --example core_soak -- --quick` (exit 0, zero mismatches).
- Performance claims only from paired runs: baseline binary built from the commit before the change into a separate target dir, alternated three times, medians reported, raw reports committed.
- Cache stays bounded at `CACHE_CAPACITY` (10,000) entries; words of `CACHE_MAX_WORD_BYTES` (256) or more are never cached. Both unchanged.
- fmt, clippy `-D warnings` and `cargo test --all-targets --locked` green on every commit. No AI attribution in commit messages.

## Review Focus

1. **Cache exactly at capacity during concurrent misses** — two threads both see `len == CAPACITY - 1` under the read lock; the re-check under the write lock must keep `len <= CAPACITY`. Pinned by Task 2's `cache_never_exceeds_capacity_under_concurrent_misses`.
2. **Full cache must not change output** — words after the 10,000th are merged, not looked up; IDs must still equal the backend. Pinned by Task 2's `full_cache_keeps_core_identical_to_backend`.
3. **Poisoned lock** — existing `unwrap_or_else(|e| e.into_inner())` behaviour must be kept on both locks (no new `unwrap()` on a lock).
4. **Words of exactly 255 vs 256 bytes** — the cacheable boundary is unchanged (`<`, not `<=`). Covered by keeping the existing condition; reviewer checks the diff.
5. **Bench misreporting** — multi-thread workloads must report aggregate bytes (`text.len() * threads`), and the "full" workloads must run on a core whose cache was filled first. Pinned by Task 1's report-shape check.

---

### Task 1: Parallel-scaling bench workloads

**Files:**
- Modify: `benches/throughput.rs` (new `scaling_workloads`, called next to `core_workloads` at line ~339)

**Interfaces:**
- Consumes: `measure(name: &str, bytes: usize, run: impl FnMut()) -> Value` (`benches/throughput.rs:171`), `ByteLevelBpe::from_json`, `ByteLevelBpe::count(&self, text: &str, recognize: bool)`.
- Produces: report entries named `core_count_hit_{1,2,4,8}t` and `core_count_full_{1,2,4,8}t`, each with `input_bytes = text.len() * threads`.

- [ ] **Step 1: Add the two synthetic texts**

`fn scaling_texts() -> (String, String)`: both about 140 KB of `" " + word` pieces where `word` is the base-26 lowercase spelling of `i` (letters only, so the GPT-2/Qwen2 pre-tokenizers keep each piece as one word).
- `hit`: words for `i in 0..1_000`, repeated until ≥ 140,000 bytes.
- `full`: words for `i in 0..40_000` once (distinct, overflows the 10,000-entry cache).

- [ ] **Step 2: Add `fn scaling_workloads(json: &[u8]) -> Vec<Value>`**

Returns empty if `ByteLevelBpe::from_json` fails. Builds **two** cores: one for the hit workloads, one for the full workloads; before timing the full workloads, call `count(full, false)` once so its cache is full. For each `threads in [1, 2, 4, 8]`, `measure(name, text.len() * threads, || std::thread::scope(...))` where each spawned thread runs `black_box(core.count(black_box(text), false).unwrap())`.

- [ ] **Step 3: Call it** after `results.extend(core_workloads(&json, &text));` inside the `if args.tokenizer.is_some()` block.

- [ ] **Step 4: Verify the report shape**

Run: `cargo bench --bench throughput -- --tokenizer models/gpt2.tokenizer.json --corpus tests/fixtures/corpus/mixed_multilingual.txt --output target/scaling-check.json`
Expected: the JSON contains 8 entries `core_count_hit_1t` … `core_count_full_8t`; for each, `input_bytes` equals the 1t entry's `input_bytes` times the thread count.

- [ ] **Step 5: Verify the debug guard still holds**

Run: `cargo test --bench throughput --locked`
Expected: PASS (debug bench binaries return without measuring).

- [ ] **Step 6: Commit**

```bash
git add benches/throughput.rs
git commit -m "bench: add 1-8 thread scaling workloads for the core's word cache"
```

### Task 2: Skip the write lock once the cache is full

**Files:**
- Modify: `src/bytelevel/mod.rs:714-732` (`word_ids`), plus a hidden accessor near `vocab_size` (`:530`)
- Test: `tests/bytelevel.rs` (two new tests using `synthetic_bpe` and `assert_core_matches`)

**Interfaces:**
- Consumes: `synthetic_bpe(pre_tokenizer: PreTokenizerWrapper, nfc: bool) -> Vec<u8>` and `assert_core_matches(json: &[u8], label: &str, texts: &[String]) -> usize` (`tests/bytelevel.rs:340`, `:274`).
- Produces: `#[doc(hidden)] pub fn cached_words(&self) -> usize` on `ByteLevelBpe` (current cache entry count; for tests and benches only) and `pub const CACHE_CAPACITY: usize = 10_000` (make the existing constant public, value unchanged).

- [ ] **Step 1: Write the failing tests** in `tests/bytelevel.rs`

```rust
#[test]
fn full_cache_keeps_core_identical_to_backend() {
    let json = synthetic_bpe(PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, true)), false);
    // 12,000 distinct letter-only words, in chunks, so the cache fills mid-run
    let texts: Vec<String> = (0..12).map(|c| (c * 1000..(c + 1) * 1000).map(|i| format!(" {}", base26(i))).collect()).collect();
    assert_core_matches(&json, "full-cache", &texts);
    let core = ByteLevelBpe::from_json(&json).unwrap();
    for t in &texts { core.count(t, false).unwrap(); }
    assert_eq!(core.cached_words(), CACHE_CAPACITY);
}

#[test]
fn cache_never_exceeds_capacity_under_concurrent_misses() {
    let json = synthetic_bpe(PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, true)), false);
    let core = ByteLevelBpe::from_json(&json).unwrap();
    std::thread::scope(|s| for t in 0..8 {
        let core = &core;
        s.spawn(move || { let text: String = (t * 5000..(t + 1) * 5000).map(|i| format!(" {}", base26(i))).collect(); core.count(&text, false).unwrap(); });
    });
    assert_eq!(core.cached_words(), CACHE_CAPACITY);
}
```
(`base26` is a small test helper: lowercase base-26 spelling of `i`.)

- [ ] **Step 2: Run them to see them fail**

Run: `cargo test --test bytelevel --locked full_cache cache_never`
Expected: compile error — `cached_words` / `CACHE_CAPACITY` not public.

- [ ] **Step 3: Add `cached_words` and make `CACHE_CAPACITY` public; run again**

Expected: both tests PASS on the *old* locking (they pin behaviour, not speed). This proves the fix in Step 4 keeps behaviour.

- [ ] **Step 4: Change `word_ids`**

Under the read lock: on a hit, copy and return (unchanged); on a miss, set `let room = cache.len() < CACHE_CAPACITY;` before the lock drops. After `self.merge(word, scratch)`, take the write lock only `if cacheable && room`, and insert only if `cache.len() < CACHE_CAPACITY` is still true under the write lock. Keep `unwrap_or_else(|e| e.into_inner())` on both locks.

- [ ] **Step 5: Run the core's tests**

Run: `cargo test --test bytelevel --locked`
Expected: all PASS (includes the hard-case catalogue and, with models present, the real GPT-2/Qwen2.5 checks).

- [ ] **Step 6: Run the quick soak**

Run: `cargo run --release --example core_soak -- --quick --output target/soak-quick.json`
Expected: exit 0, zero mismatches, zero illegitimate declines.

- [ ] **Step 7: Commit**

```bash
git add src/bytelevel/mod.rs tests/bytelevel.rs
git commit -m "core: skip the cache write lock once the word cache is full"
```

### Task 3: Paired measurement and docs

**Files:**
- Create: `benchmarks/scaling-before.json`, `benchmarks/scaling-after.json`, `benchmarks/runs/scaling-{before,after}-{1,2,3}.json`
- Modify: `PERFORMANCE.md` (new "Fifth set: cache scaling" section), `CLAUDE.md` (item 6 "Newly open" → done, with numbers)

**Interfaces:**
- Consumes: Task 1 workloads; the Task 1 commit as baseline, the Task 2 commit as candidate.

- [ ] **Step 1: Build both binaries**

Baseline: `git worktree add ../ft-baseline <Task-1 commit>` then `cargo bench --bench throughput --no-run` there with `CARGO_TARGET_DIR=.setup/target-baseline`. Candidate: same from the Task 2 commit into `.setup/target-candidate`. Remove the worktree afterwards.

- [ ] **Step 2: Alternate three runs each** on a quiet laptop (no compiles running), `--tokenizer models/gpt2.tokenizer.json --corpus tests/fixtures/corpus/mixed_multilingual.txt`, writing `benchmarks/runs/scaling-{before,after}-{1,2,3}.json`.

- [ ] **Step 3: Summarise** medians of `mib_per_second` per workload into `benchmarks/scaling-{before,after}.json`.

Expected outcome to report honestly either way: `core_count_full_{2,4,8}t` rises above the baseline's ~2x-of-one-thread ceiling; `core_count_hit_*` and the existing single-thread `core_count` are unchanged controls within run-to-run range.

- [ ] **Step 4: Document** in `PERFORMANCE.md` (table: workload, before, after, speed-up, per-run range) and update `CLAUDE.md`. If the full-cache case did not improve, say so and keep the change only if the controls did not regress.

- [ ] **Step 5: Final checks and commit**

Run: `cargo fmt --all -- --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --all-targets --locked`
Expected: all pass.

```bash
git add benchmarks/ PERFORMANCE.md CLAUDE.md
git commit -m "perf: paired measurement of cache-full multi-thread scaling"
```
