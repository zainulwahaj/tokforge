# Phase 2: tokforge rename and Python package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project to tokforge and ship a `pip install tokforge` package whose IDs equal the Rust engine's, with GIL-free parallel batch encoding and wheels for Linux, macOS and Windows built in CI.

**Architecture:** The root crate becomes `tokforge` (lib `tokforge`, binary `tokforge`). A new workspace member `python/` is a PyO3 `cdylib` (abi3, Python ≥ 3.9) built by maturin; it wraps `tokforge::Engine` and releases the GIL for every call that tokenizes, using rayon for batches. `Engine` gains a public `encode_ids` that returns ids from the borrowed fast path, so Python never builds per-token strings.

**Tech Stack:** Rust, PyO3 0.30 (`abi3-py39`), maturin 1.15/1.16, rayon, pytest; `tiktoken` only as a test oracle.

**Spec:** `docs/superpowers/plans/2026-11-10-resume-ready-roadmap.md` (Phase 2 row, Global constraints, decisions: name tokforge, MIT OR Apache-2.0, backend fallback included).

## Global Constraints

- Package and crate name `tokforge`; license `MIT OR Apache-2.0` with `LICENSE-MIT` and `LICENSE-APACHE` at the root.
- The backend fallback stays a normal dependency (any `tokenizer.json` loads).
- No behaviour change to the engine beyond the additive `encode_ids`; the existing 61+ tests stay green.
- fmt, clippy `-D warnings`, `cargo test --all-targets --locked` green on every commit; Python tests green where Python changes.
- No AI attribution anywhere; commit messages per `~/.claude/CLAUDE.md`.

## Review Focus

1. **Non-UTF-8 / surrogate strings from Python** (`"\ud800"`): must raise `ValueError`, not panic or produce garbage. Test in Task 3.
2. **Unknown ids in `decode`**: raise `ValueError` with the engine's message (the engine rejects unknown ids). Test in Task 3.
3. **Empty inputs**: `encode("") == []`, `count("") == 0`, `chunk("", n) == []`, `encode_batch([]) == []`. Test in Task 3.
4. **Batch order**: `encode_batch` returns results in input order regardless of thread scheduling. Test in Task 3 (compare to sequential `encode`).
5. **Engine panic**: a Rust panic must surface as a Python exception, not abort the interpreter (PyO3 converts panics to `PanicException`; no `catch_unwind` needed, but no `unwrap` on user input either).

---

### Task 1: Rename to tokforge and add licenses

**Files:** `Cargo.toml` (package name, `[lib] name`, `[[bin]] name`, `license`, `description`, `repository`), every `use fast_tokenizer::` in `tests/`, `benches/`, `examples/`, `src/main.rs`; `README.md`, `CLAUDE.md`, `AGENTS.md`, `PERFORMANCE.md`, `models/README.md`, `.github/workflows/ci.yml` (binary/artifact names); create `LICENSE-MIT`, `LICENSE-APACHE`.

- [ ] Rename; `tokenizer` binary becomes `tokforge`. Historical benchmark reports keep their recorded names.
- [ ] Verify: `cargo test --all-targets --locked` green; `cargo run --release -- info` prints `tokforge`; `grep -rn fast_tokenizer src tests benches examples` empty.
- [ ] Commit `Rename the project to tokforge and add MIT/Apache-2.0 licenses`.

### Task 2: `Engine::encode_ids`

**Files:** `src/engine.rs`, `tests/engine.rs`.

**Interfaces — Produces:** `pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>, Error>` — ids of `encode_raw(text)`, identical to `encode(text)?.ids`.

- [ ] RED: `encode_ids_matches_encode` in `tests/engine.rs` — for the WordPiece fixture and (when present) GPT-2, over the corpus texts and `""`, `encode_ids(t) == encode(t).ids`. Fails to compile.
- [ ] GREEN: implement via `encode_raw(text).map(|e| e.ids().to_vec())`.
- [ ] Commit `Add Engine::encode_ids for callers that only need ids`.

### Task 3: PyO3 bindings

**Files:** root `Cargo.toml` (`[workspace] members = [".", "python"]`), create `python/Cargo.toml` (crate `tokforge-python`, `crate-type = ["cdylib"]`, `pyo3 = { version = "0.30", features = ["abi3-py39", "extension-module"] }`, `rayon`, `tokforge = { path = ".." }`), `python/pyproject.toml` (maturin backend, `name = "tokforge"`, `requires-python = ">=3.9"`, license, readme), `python/src/lib.rs`, `python/tokforge.pyi` (type stubs), `python/tests/test_tokforge.py`, `python/README.md`.

**Interfaces — Produces (Python):**

```python
class Tokenizer:
    @staticmethod
    def from_file(path: str, *, special_tokens: str = "literal", add_special_tokens: bool = False, core: str = "auto") -> "Tokenizer": ...
    @staticmethod
    def from_str(json: str, *, special_tokens: str = "literal", add_special_tokens: bool = False, core: str = "auto") -> "Tokenizer": ...
    def count(self, text: str) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...   # parallel, GIL released
    def count_batch(self, texts: list[str]) -> list[int]: ...          # parallel, GIL released
    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str: ...
    def truncate(self, text: str, max_tokens: int) -> str: ...
    def chunk(self, text: str, max_tokens: int) -> list[str]: ...
    vocab_size: int; sha256: str; core: str   # read-only properties
__version__: str
```

Errors: invalid option strings and engine errors raise `ValueError`; missing file raises `OSError`.

- [ ] RED: `python/tests/test_tokforge.py` — WordPiece fixture: `count("hello world!") == 3`, `encode(...) == [4, 5, 6]`, `decode([4,5,6])` round trip, the five Review Focus cases; GPT-2 (skip when `models/gpt2.tokenizer.json` is absent): `encode` equals `tiktoken.get_encoding("gpt2").encode(t, disallowed_special=())` on every corpus file, `encode_batch(corpus) == [encode(t) for t in corpus]`, `core == "bytelevel-bpe"` (whatever `artifact.core` reports). Run in a venv with `maturin develop`: fails (module missing).
- [ ] GREEN: implement `python/src/lib.rs`; tokenizing methods wrap work in `py.detach(...)` (PyO3 0.30 name for releasing the GIL; fall back to `allow_threads` if the API differs) and batches use `rayon` `par_iter` preserving order.
- [ ] Verify: `maturin develop --release` then `pytest python/tests -q` green; `cargo clippy --workspace --all-targets --locked -- -D warnings` clean.
- [ ] Commit `Add Python bindings (tokforge package)`.

### Task 4: Wheels in CI

**Files:** `.github/workflows/ci.yml` (new job `python`): matrix ubuntu/windows/macos, `PyO3/maturin-action` builds an abi3 wheel, installs it, runs `pytest python/tests` (models absent → GPT-2 tests skip; fixture tests run), uploads wheels as artifacts.

- [ ] Verify on the PR: the three `python` jobs green and three wheels attached.
- [ ] Commit `Build and test abi3 wheels on Linux, macOS and Windows`.
