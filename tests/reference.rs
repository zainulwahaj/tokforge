//! Reference correctness against production tokenizer artifacts and independent oracles.
//!
//! Artifacts live under `models/` and are gitignored; `python -I models/fetch.py` downloads them at
//! the pinned revisions recorded in `models/MANIFEST.json`. Golden IDs under
//! `tests/fixtures/golden/<artifact>/` are committed and name the oracle that produced them.
//!
//! When an artifact file is absent, the artifact-specific test prints `SKIPPED: ...` and passes so
//! `cargo test` works on a fresh clone (run with `-- --show-output` to see the notice). Set
//! `FAST_TOKENIZER_REQUIRE_MODELS=1` to turn a skip into a failure, e.g. on a machine that has the
//! models. Golden files are never edited to fit the engine: a documented divergence is listed in
//! `tests/fixtures/golden/KNOWN_DIFFERENCES.json` and reported as `KNOWN DIFFERENCE`.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokforge::{Chunk, EncodeOptions, Engine, Error, SpecialTokens};

/// Every artifact the manifest must list. Each has its own `#[test]` below so failures are isolated.
const ARTIFACTS: [&str; 4] = [
    "gpt2",
    "qwen2.5-0.5b",
    "bert-base-uncased",
    "xlm-roberta-base",
];
const BUDGETS: [usize; 4] = [1, 7, 64, 512];
const MAX_CORPUS_BYTES: u64 = 8192;
/// Budget 1 makes every multi-token scalar (emoji, CJK, `▁`-prefixed pieces) hit the engine's
/// documented exhaustive no-fit fallback, which re-counts every prefix of the remaining suffix and is
/// quadratic. Full 8 KiB files take many minutes per artifact in debug builds, so the routine suite
/// checks budget 1 on this many leading bytes of each file; `FAST_TOKENIZER_EXHAUSTIVE=1` uses the
/// whole file (run it in release mode). Budgets of 7 and above never trigger the fallback here
/// because a single scalar costs at most four byte-level tokens or two Unigram pieces.
const BUDGET_ONE_PREFIX_BYTES: usize = 128;

/// The text a budget invariant runs on: the whole corpus file except for the bounded budget-1 case.
fn budget_input(text: &str, budget: usize) -> &str {
    if budget > 1
        || text.len() <= BUDGET_ONE_PREFIX_BYTES
        || std::env::var_os("FAST_TOKENIZER_EXHAUSTIVE").is_some()
    {
        return text;
    }
    let mut end = BUDGET_ONE_PREFIX_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[derive(Deserialize)]
struct Manifest {
    artifacts: Vec<ManifestEntry>,
}

#[derive(Deserialize)]
struct ManifestEntry {
    name: String,
    file: String,
    sha256: Option<String>,
    vocab_size: Option<usize>,
    roundtrip: Roundtrip,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
enum Roundtrip {
    /// `decode(encode(text)) == text` for every valid UTF-8 input.
    Exact,
    /// Exact only when the input is already NFC-normalized (the artifact has an NFC normalizer).
    NfcOnly,
    /// Lossy normalization; no exact round trip is promised.
    None,
}

#[derive(Deserialize)]
struct Golden {
    artifact: String,
    artifact_sha256: String,
    corpus_file: String,
    corpus_sha256: String,
    text_is_nfc: bool,
    cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    special_tokens: Mode,
    add_special_tokens: bool,
    oracle: String,
    ids: Vec<u32>,
    /// Whether the Python `tokenizers` bindings (same backend family as the engine) agreed with
    /// the oracle; when they did not, their IDs are recorded so a divergence can be attributed.
    #[serde(default)]
    cross_check_matches: Option<bool>,
    #[serde(default)]
    cross_check_ids: Option<Vec<u32>>,
}

impl GoldenCase {
    /// Attribute a divergence: upstream backend behaviour, or something specific to the engine.
    fn attribution(&self, engine_ids: &[u32]) -> &'static str {
        match (&self.cross_check_ids, self.cross_check_matches) {
            (Some(ids), _) if ids == engine_ids => {
                "\n      attribution: engine output equals the Python `tokenizers` bindings; the divergence is between the HF Rust backend and the independent oracle, not engine logic"
            }
            (None, Some(true)) => {
                "\n      attribution: the Python `tokenizers` bindings agreed with the oracle, so this divergence is specific to the engine"
            }
            _ => "",
        }
    }
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Literal,
    Recognize,
}

impl Mode {
    fn special_tokens(self) -> SpecialTokens {
        match self {
            Mode::Literal => SpecialTokens::Literal,
            Mode::Recognize => SpecialTokens::Recognize,
        }
    }
}

#[derive(Deserialize)]
struct KnownDifferences {
    entries: Vec<KnownDifference>,
}

#[derive(Deserialize)]
struct KnownDifference {
    artifact: String,
    corpus_file: String,
    special_tokens: Option<Mode>,
    add_special_tokens: Option<bool>,
    reason: String,
}

impl KnownDifference {
    fn matches(&self, artifact: &str, corpus_file: &str, mode: Mode, add_special: bool) -> bool {
        self.artifact == artifact
            && self.corpus_file == corpus_file
            && self.special_tokens.is_none_or(|m| m == mode)
            && self.add_special_tokens.is_none_or(|a| a == add_special)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn manifest() -> Manifest {
    read_json(&root().join("models/MANIFEST.json"))
}

fn known_differences() -> Vec<KnownDifference> {
    read_json::<KnownDifferences>(&root().join("tests/fixtures/golden/KNOWN_DIFFERENCES.json"))
        .entries
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Corpus files in a fixed order so goldens and tests agree on coverage.
fn corpus_files() -> Vec<PathBuf> {
    let dir = root().join("tests/fixtures/corpus");
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "txt"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no corpus files under {}", dir.display());
    files
}

fn file_name(path: &Path) -> &str {
    path.file_name().unwrap().to_str().unwrap()
}

fn golden_path(artifact: &str, corpus: &Path) -> PathBuf {
    root()
        .join("tests/fixtures/golden")
        .join(artifact)
        .join(format!(
            "{}.json",
            corpus.file_stem().unwrap().to_str().unwrap()
        ))
}

/// Registered special tokens per artifact and how many of them Literal mode has to make
/// unreachable inside the model. Byte-level BPE lists `<|endoftext|>` without any merge producing
/// it and Qwen's specials are added tokens outside the model vocabulary, so nothing is rebuilt
/// there; WordPiece drops its four non-`[UNK]` entries; the converted Unigram vocabulary scores all
/// five control pieces, so all five are suppressed.
fn special_tokens(name: &str) -> (usize, &'static [(u32, &'static str)]) {
    match name {
        "gpt2" => (0, &[(50256, "<|endoftext|>")]),
        "qwen2.5-0.5b" => (
            0,
            &[
                (151643, "<|endoftext|>"),
                (151644, "<|im_start|>"),
                (151645, "<|im_end|>"),
            ],
        ),
        "bert-base-uncased" => (
            4,
            &[
                (0, "[PAD]"),
                (100, "[UNK]"),
                (101, "[CLS]"),
                (102, "[SEP]"),
                (103, "[MASK]"),
            ],
        ),
        "xlm-roberta-base" => (
            5,
            &[
                (0, "<s>"),
                (1, "<pad>"),
                (2, "</s>"),
                (3, "<unk>"),
                (250001, "<mask>"),
            ],
        ),
        other => panic!("no special-token expectations for {other}"),
    }
}

fn skip(reason: String) {
    if std::env::var_os("FAST_TOKENIZER_REQUIRE_MODELS").is_some() {
        panic!("{reason} (FAST_TOKENIZER_REQUIRE_MODELS is set)");
    }
    eprintln!("SKIPPED: {reason}");
}

/// One engine per (special-token mode, add_special_tokens) pair. Options are fixed at load time, so
/// the artifact is parsed once per combination the goldens or invariants actually need.
struct Engines<'a> {
    bytes: &'a [u8],
    loaded: HashMap<(Mode, bool), Engine>,
}

impl Engines<'_> {
    fn get(&mut self, mode: Mode, add_special_tokens: bool) -> &Engine {
        self.loaded
            .entry((mode, add_special_tokens))
            .or_insert_with(|| {
                Engine::from_bytes(
                    self.bytes,
                    EncodeOptions {
                        add_special_tokens,
                        special_tokens: mode.special_tokens(),
                        ..Default::default()
                    },
                )
                .expect("artifact loads")
            })
    }
}

fn spellings(engine: &Engine, ids: &[u32]) -> Vec<String> {
    ids.iter()
        .map(|&id| {
            engine
                .decode(&[id], false)
                .unwrap_or_else(|_| "<unknown id>".into())
        })
        .collect()
}

fn describe_mismatch(
    engine: &Engine,
    ours: &[u32],
    our_tokens: &[String],
    oracle: &[u32],
) -> String {
    let first = ours
        .iter()
        .zip(oracle)
        .position(|(a, b)| a != b)
        .unwrap_or(ours.len().min(oracle.len()));
    let lo = first.saturating_sub(4);
    let window = |len: usize| lo..(first + 8).min(len);
    format!(
        "engine {} ids, oracle {} ids, first divergence at index {first}\n      engine ids   {:?}\n      engine toks  {:?}\n      oracle ids   {:?}\n      oracle spell {:?}",
        ours.len(),
        oracle.len(),
        &ours[window(ours.len())],
        &our_tokens[window(our_tokens.len())],
        &oracle[window(oracle.len())],
        spellings(engine, &oracle[window(oracle.len())]),
    )
}

fn first_char(text: &str, offset: usize) -> &str {
    let len = text[offset..]
        .chars()
        .next()
        .expect("offset inside text")
        .len_utf8();
    &text[offset..offset + len]
}

/// Chunk/truncate contract for one text and budget. `exact` enables decode round trips of slices.
fn check_budget(
    engine: &Engine,
    text: &str,
    budget: usize,
    exact: bool,
    label: &str,
    failures: &mut Vec<String>,
) {
    let mut fail = |message: String| failures.push(format!("{label} budget {budget}: {message}"));
    let full_count = engine.count(text).unwrap();
    let empty_count = engine.count("").unwrap();

    match engine.truncate(text, budget) {
        Ok(prefix) => {
            if budget < empty_count {
                fail(format!(
                    "truncate succeeded although the empty prefix needs {empty_count} tokens"
                ));
            }
            check_slice(engine, text, &prefix, budget, exact, "truncate", &mut fail);
            if prefix.start_byte != 0 {
                fail(format!("truncate start_byte {} != 0", prefix.start_byte));
            }
            if full_count <= budget && prefix.end_byte != text.len() {
                fail(format!(
                    "whole text has {full_count} tokens but truncate returned {} of {} bytes",
                    prefix.end_byte,
                    text.len()
                ));
            }
        }
        Err(Error::EmptyBudget) if budget < empty_count => {}
        Err(error) => fail(format!("truncate failed: {error}")),
    }

    match engine.chunk(text, budget) {
        Ok(chunks) => {
            if text.is_empty() {
                if !chunks.is_empty() {
                    fail(format!("empty text produced {} chunks", chunks.len()));
                }
                return;
            }
            if chunks.is_empty() {
                fail("non-empty text produced zero chunks".into());
                return;
            }
            let mut previous = 0;
            for chunk in &chunks {
                if chunk.start_byte != previous {
                    fail(format!(
                        "chunk at {} is not contiguous with previous end {previous}",
                        chunk.start_byte
                    ));
                }
                if chunk.end_byte <= chunk.start_byte {
                    fail(format!("empty chunk at byte {}", chunk.start_byte));
                }
                check_slice(engine, text, chunk, budget, exact, "chunk", &mut fail);
                previous = chunk.end_byte;
            }
            if previous != text.len() {
                fail(format!(
                    "chunks end at byte {previous}, text has {} bytes",
                    text.len()
                ));
            }
            let rebuilt: String = chunks.iter().map(|c| c.text.as_str()).collect();
            if rebuilt != text {
                fail("concatenated chunk texts do not reconstruct the source".into());
            }
        }
        // The contract allows failure only when no nonempty prefix at `offset` fits. That implies
        // the single scalar at `offset` alone exceeds the budget; verify that necessary condition.
        Err(Error::BudgetTooSmall { offset }) => {
            if !text.is_char_boundary(offset) || offset >= text.len() {
                fail(format!(
                    "BudgetTooSmall offset {offset} is not a scalar boundary inside text"
                ));
            } else {
                let single = engine.count(first_char(text, offset)).unwrap();
                if single <= budget {
                    fail(format!(
                        "BudgetTooSmall at byte {offset} but the scalar {:?} there counts {single} <= {budget}",
                        first_char(text, offset)
                    ));
                }
            }
        }
        Err(Error::EmptyBudget) if budget < empty_count => {}
        Err(error) => fail(format!("chunk failed: {error}")),
    }
}

fn check_slice(
    engine: &Engine,
    text: &str,
    slice: &Chunk,
    budget: usize,
    exact: bool,
    what: &str,
    fail: &mut impl FnMut(String),
) {
    let (start, end) = (slice.start_byte, slice.end_byte);
    if !text.is_char_boundary(start)
        || !text.is_char_boundary(end)
        || start > end
        || end > text.len()
    {
        fail(format!(
            "{what} range {start}..{end} is not a scalar-aligned range of the text"
        ));
        return;
    }
    if slice.text != text[start..end] {
        fail(format!(
            "{what} text differs from source bytes {start}..{end}"
        ));
    }
    if slice.count > budget {
        fail(format!(
            "{what} {start}..{end} count {} exceeds budget",
            slice.count
        ));
    }
    // Independent of the count path: the full encode pipeline must agree with the reported count.
    let encoded = engine.encode(&slice.text).unwrap();
    if encoded.ids.len() != slice.count || encoded.count != slice.count {
        fail(format!(
            "{what} {start}..{end} reports count {} but encode yields {} ids",
            slice.count,
            encoded.ids.len()
        ));
    }
    if encoded.ids.len() != engine.count(&slice.text).unwrap() {
        fail(format!(
            "{what} {start}..{end}: count() and encode() disagree"
        ));
    }
    if exact {
        match engine.decode(&encoded.ids, false) {
            Ok(decoded) if decoded == slice.text => {}
            Ok(decoded) => fail(format!(
                "{what} {start}..{end} does not round trip: {decoded:?} != {:?}",
                slice.text
            )),
            Err(error) => fail(format!("{what} {start}..{end} decode failed: {error}")),
        }
    }
}

fn check_artifact(name: &str) {
    let manifest = manifest();
    let entry = manifest
        .artifacts
        .iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("models/MANIFEST.json has no entry named {name}"));
    let path = root().join("models").join(&entry.file);
    if !path.is_file() {
        skip(format!(
            "{name}: {} is not present; run `python -I models/fetch.py --artifact {name}` to download it at the pinned revision",
            path.display()
        ));
        return;
    }
    let expected_sha = entry.sha256.as_deref().unwrap_or_else(|| {
        panic!(
            "manifest entry {name} has no sha256; run `python -I models/fetch.py --write-manifest`"
        )
    });
    let bytes = fs::read(&path).unwrap();
    assert_eq!(
        sha256_hex(&bytes),
        expected_sha,
        "{name}: {} does not match the manifest sha256",
        path.display()
    );
    let known = known_differences();
    let mut engines = Engines {
        bytes: &bytes,
        loaded: HashMap::new(),
    };
    let mut failures = Vec::new();
    let mut known_hits = Vec::new();
    let mut cases_checked = 0usize;

    let probe = engines.get(Mode::Literal, false);
    assert_eq!(
        probe.artifact().sha256,
        expected_sha,
        "{name}: engine hash differs"
    );
    assert_eq!(
        Some(probe.artifact().vocab_size),
        entry.vocab_size,
        "{name}: vocab_size differs from manifest"
    );
    let (suppressed, specials) = special_tokens(name);
    assert_eq!(
        probe.artifact().suppressed_special_pieces,
        suppressed,
        "{name}: Literal-mode suppression count"
    );
    assert_eq!(
        engines
            .get(Mode::Recognize, false)
            .artifact()
            .suppressed_special_pieces,
        0,
        "{name}: Recognize mode must not suppress"
    );
    // Literal text never yields a special ID; Recognize yields exactly it; and suppression changes
    // segmentation only, so every special ID still decodes to its spelling in both modes.
    for &(id, spelling) in specials {
        for mode in [Mode::Literal, Mode::Recognize] {
            let engine = engines.get(mode, false);
            assert_eq!(
                engine.decode(&[id], false).unwrap(),
                spelling,
                "{name}: decode of special id {id} in {mode:?} mode"
            );
            let ids = engine.encode(spelling).unwrap().ids;
            match mode {
                Mode::Literal => assert!(
                    !ids.contains(&id),
                    "{name}: Literal mode emitted special id {id} for {spelling:?}: {ids:?}"
                ),
                Mode::Recognize => assert_eq!(
                    ids,
                    [id],
                    "{name}: Recognize mode did not emit {id} alone for {spelling:?}"
                ),
            }
        }
    }

    for corpus in corpus_files() {
        let corpus_name = file_name(&corpus);
        let text = String::from_utf8(fs::read(&corpus).unwrap())
            .unwrap_or_else(|e| panic!("{corpus_name} is not UTF-8: {e}"));
        let golden_file = golden_path(name, &corpus);
        if !golden_file.is_file() {
            failures.push(format!(
                "{name}/{corpus_name}: golden file {} is missing; run `python -I tests/fixtures/golden/generate.py --artifact {name}`",
                golden_file.display()
            ));
            continue;
        }
        let golden: Golden = read_json(&golden_file);
        assert_eq!(
            golden.artifact,
            name,
            "{}: wrong artifact",
            golden_file.display()
        );
        assert_eq!(
            golden.corpus_file,
            corpus_name,
            "{}: wrong corpus file",
            golden_file.display()
        );
        if golden.artifact_sha256 != expected_sha {
            failures.push(format!(
                "{name}/{corpus_name}: golden was generated from artifact sha256 {} but the manifest pins {expected_sha}",
                golden.artifact_sha256
            ));
            continue;
        }
        if golden.corpus_sha256 != sha256_hex(text.as_bytes()) {
            failures.push(format!(
                "{name}/{corpus_name}: corpus file changed since the golden was generated; regenerate"
            ));
            continue;
        }
        let exact = match entry.roundtrip {
            Roundtrip::Exact => true,
            Roundtrip::NfcOnly => golden.text_is_nfc,
            Roundtrip::None => false,
        };

        for case in &golden.cases {
            cases_checked += 1;
            let label = format!(
                "{name}/{corpus_name} [{:?}, add_special_tokens={}] oracle={}",
                case.special_tokens, case.add_special_tokens, case.oracle
            );
            let engine = engines.get(case.special_tokens, case.add_special_tokens);
            let encoded = engine.encode(&text).unwrap();
            if encoded.count != encoded.ids.len()
                || engine.count(&text).unwrap() != encoded.ids.len()
            {
                failures.push(format!(
                    "{label}: encode reports {} ids, count {} and count() {}",
                    encoded.ids.len(),
                    encoded.count,
                    engine.count(&text).unwrap()
                ));
            }
            let known_entry = known.iter().find(|k| {
                k.matches(
                    name,
                    corpus_name,
                    case.special_tokens,
                    case.add_special_tokens,
                )
            });
            match (encoded.ids == case.ids, known_entry) {
                (true, None) => {}
                (true, Some(k)) => failures.push(format!(
                    "{label}: STALE KNOWN DIFFERENCE, engine now matches the oracle; remove the entry: {}",
                    k.reason
                )),
                (false, Some(k)) => {
                    let detail = describe_mismatch(engine, &encoded.ids, &encoded.tokens, &case.ids);
                    eprintln!(
                        "KNOWN DIFFERENCE {label}\n    reason: {}\n    {detail}{}",
                        k.reason,
                        case.attribution(&encoded.ids)
                    );
                    known_hits.push(label.clone());
                }
                (false, None) => failures.push(format!(
                    "MISMATCH {label}\n    {}{}",
                    describe_mismatch(engine, &encoded.ids, &encoded.tokens, &case.ids),
                    case.attribution(&encoded.ids)
                )),
            }
            if exact && !case.add_special_tokens {
                for (who, ids) in [("engine", &encoded.ids), ("oracle", &case.ids)] {
                    match engine.decode(ids, false) {
                        Ok(decoded) if decoded == text => {}
                        Ok(_) => failures
                            .push(format!("{label}: decode of {who} ids does not round trip")),
                        Err(error) => {
                            failures.push(format!("{label}: decode of {who} ids failed: {error}"))
                        }
                    }
                }
            }
        }

        // Budget invariants: both special-token modes without post-processor tokens at every budget,
        // plus post-processor tokens where the artifact adds any (budgets below the overhead must fail).
        for mode in [Mode::Literal, Mode::Recognize] {
            for budget in BUDGETS {
                let engine = engines.get(mode, false);
                let label = format!("{name}/{corpus_name} [{mode:?}, add_special_tokens=false]");
                let input = budget_input(&text, budget);
                check_budget(engine, input, budget, exact, &label, &mut failures);
            }
        }
        let with_special = engines.get(Mode::Literal, true);
        if with_special.count("").unwrap() > 0 {
            for budget in BUDGETS {
                let label = format!("{name}/{corpus_name} [Literal, add_special_tokens=true]");
                let input = budget_input(&text, budget);
                check_budget(with_special, input, budget, false, &label, &mut failures);
            }
        }
    }

    eprintln!(
        "{name}: {cases_checked} golden cases checked, {} known differences, {} failures",
        known_hits.len(),
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{name}: {} failure(s)\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

#[test]
fn manifest_lists_every_reference_artifact_with_pinned_provenance() {
    let manifest = manifest();
    for name in ARTIFACTS {
        let entry = manifest
            .artifacts
            .iter()
            .find(|a| a.name == name)
            .unwrap_or_else(|| panic!("manifest lacks {name}"));
        assert!(
            entry.sha256.as_deref().is_some_and(|s| s.len() == 64),
            "{name}: sha256 not recorded"
        );
        assert!(
            entry.vocab_size.is_some(),
            "{name}: vocab_size not recorded"
        );
        assert!(
            entry.file.ends_with(".json"),
            "{name}: artifact file must be gitignored JSON"
        );
    }
    let mut names: Vec<&str> = manifest.artifacts.iter().map(|a| a.name.as_str()).collect();
    names.sort_unstable();
    let mut expected = ARTIFACTS.to_vec();
    expected.sort_unstable();
    assert_eq!(
        names, expected,
        "every manifest artifact needs a test function here"
    );
}

#[test]
fn corpus_files_are_small_valid_utf8_and_cover_special_spellings() {
    let files = corpus_files();
    let mut all = String::new();
    for path in &files {
        let bytes = fs::read(path).unwrap();
        assert!(
            bytes.len() as u64 <= MAX_CORPUS_BYTES,
            "{} is {} bytes; keep corpus files under 8 KiB",
            path.display(),
            bytes.len()
        );
        let text = String::from_utf8(bytes)
            .unwrap_or_else(|e| panic!("{} is not valid UTF-8: {e}", path.display()));
        all.push_str(&text);
    }
    for spelling in [
        "<|endoftext|>",
        "<|im_start|>",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "<s>",
        "</s>",
        "<mask>",
    ] {
        assert!(
            all.contains(spelling),
            "corpus lacks the special spelling {spelling}"
        );
    }
    for needle in [
        "\r\n", "\t", "\u{a0}", "\u{301}", "\u{fb01}", "\u{200d}", "\0",
    ] {
        assert!(all.contains(needle), "corpus lacks {needle:?}");
    }
}

/// Runs without model files: every committed golden must describe the pinned artifact and the
/// current corpus bytes, so a stale golden is caught on any machine.
#[test]
fn goldens_match_manifest_hashes_and_corpus_bytes() {
    let manifest = manifest();
    let known = known_differences();
    let files = corpus_files();
    let mut missing = Vec::new();
    for entry in &manifest.artifacts {
        for corpus in &files {
            let path = golden_path(&entry.name, corpus);
            if !path.is_file() {
                missing.push(path.display().to_string());
                continue;
            }
            let golden: Golden = read_json(&path);
            assert_eq!(golden.artifact, entry.name, "{}", path.display());
            assert_eq!(golden.corpus_file, file_name(corpus), "{}", path.display());
            assert_eq!(
                Some(golden.artifact_sha256.as_str()),
                entry.sha256.as_deref(),
                "{}: generated from a different artifact than the manifest pins",
                path.display()
            );
            assert_eq!(
                golden.corpus_sha256,
                sha256_hex(&fs::read(corpus).unwrap()),
                "{}: corpus file changed after the golden was generated",
                path.display()
            );
            assert!(!golden.cases.is_empty(), "{}: no cases", path.display());
            for case in &golden.cases {
                assert!(
                    !case.oracle.is_empty(),
                    "{}: case without oracle attribution",
                    path.display()
                );
            }
        }
    }
    assert!(
        missing.is_empty(),
        "missing golden files:\n{}",
        missing.join("\n")
    );
    for k in &known {
        assert!(
            manifest.artifacts.iter().any(|a| a.name == k.artifact),
            "KNOWN_DIFFERENCES.json names unknown artifact {}",
            k.artifact
        );
        assert!(
            files.iter().any(|f| file_name(f) == k.corpus_file),
            "KNOWN_DIFFERENCES.json names unknown corpus file {}",
            k.corpus_file
        );
        assert!(!k.reason.is_empty(), "known difference without a reason");
    }
}

#[test]
fn gpt2_matches_tiktoken_goldens_and_budget_invariants() {
    check_artifact("gpt2");
}

#[test]
fn qwen2_5_matches_transformers_goldens_and_budget_invariants() {
    check_artifact("qwen2.5-0.5b");
}

#[test]
fn bert_base_uncased_matches_slow_bert_goldens_and_budget_invariants() {
    check_artifact("bert-base-uncased");
}

#[test]
fn xlm_roberta_base_matches_sentencepiece_goldens_and_budget_invariants() {
    check_artifact("xlm-roberta-base");
}
