//! Differential tests: the custom byte-level BPE core must be bit-identical to the Hugging Face
//! backend on every input it accepts. Synthetic artifacts run everywhere; the real GPT-2 and
//! Qwen2.5 artifacts run when `models/` has them and print `SKIPPED` otherwise.

use std::{fs, path::PathBuf};

use fast_tokenizer::bytelevel::{
    classify, pretokenize_split, ByteLevelBpe, Class, Fallback, Output, Pattern, QWEN2_PATTERN,
};
use tokenizers::{
    models::bpe::BPE,
    normalizers::NormalizerWrapper,
    pre_tokenizers::{byte_level::ByteLevel, split::Split, PreTokenizerWrapper},
    tokenizer::{
        normalizer::SplitDelimiterBehavior, Normalizer, OffsetReferential, OffsetType,
        PreTokenizedString, PreTokenizer,
    },
    AddedToken, NormalizedString, Tokenizer,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn corpus_texts() -> Vec<(String, String)> {
    let dir = root().join("tests/fixtures/corpus");
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let name = p.file_name().unwrap().to_str().unwrap().to_owned();
            (name, fs::read_to_string(&p).unwrap())
        })
        .collect()
}

/// Small deterministic generator so failures reproduce exactly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// Text that stresses every regex alternative: contractions in both cases, digit runs,
/// punctuation glued to words, every kind of whitespace, CR/LF mixes, multi-byte scripts,
/// emoji and special-token spellings.
fn random_text(rng: &mut Rng, specials: &[&str]) -> String {
    const PIECES: &[&str] = &[
        "hello",
        "World",
        "it",
        "don",
        "they",
        "I",
        "x",
        "ABC",
        "naïve",
        "café",
        "日本語",
        "한국어",
        "مرحبا",
        "привет",
        "Ωmega",
        "ﬁ",
        "ſ",
        "😀",
        "👍🏽",
        "🇵🇰",
        "\u{301}",
        "\u{200d}",
        "'s",
        "'S",
        "'t",
        "'re",
        "'RE",
        "'ve",
        "'m",
        "'ll",
        "'d",
        "'x",
        "'",
        "''",
        "0",
        "7",
        "42",
        "1234567",
        "٣",
        "½",
        ".",
        ",",
        "!",
        "?",
        "...",
        "---",
        "#",
        "@",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "/",
        "\\",
        "|",
        "~",
        "`",
        "\"",
        ";",
        ":",
        "=",
        "+",
        "-",
        "_",
        " ",
        "  ",
        "   ",
        "\t",
        "\n",
        "\r",
        "\r\n",
        "\n\n",
        " \n",
        "\n ",
        "\u{a0}",
        "\u{3000}",
        "\u{85}",
        "\u{2028}",
        "\u{200b}",
        "\u{feff}",
        "\0",
        "\x07",
        "\x7f",
        "\u{fffd}",
        "\u{e000}",
        "\u{10ffff}",
        "Ġ",
        "ĠĠ",
        "\u{100}",
    ];
    let len = 1 + rng.below(40);
    let mut text = String::new();
    for _ in 0..len {
        if !specials.is_empty() && rng.below(12) == 0 {
            text.push_str(specials[rng.below(specials.len())]);
        } else if rng.below(9) == 0 {
            let code = rng.below(0x11_0000) as u32;
            if let Some(c) = char::from_u32(code) {
                text.push(c);
            }
        } else {
            text.push_str(rng.pick(PIECES));
        }
    }
    text
}

fn backend_splits(pre: &dyn PreTokenizer, text: &str) -> Vec<(usize, usize)> {
    let mut pretokenized = PreTokenizedString::from(text);
    pre.pre_tokenize(&mut pretokenized).unwrap();
    pretokenized
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(_, offsets, _)| offsets)
        .collect()
}

fn core_splits(pattern: Pattern, text: &str) -> Option<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    pretokenize_split(pattern, text, |s, e| out.push((s, e)))
        .ok()
        .map(|()| out)
}

fn qwen2_backend_pre_tokenizer() -> PreTokenizerWrapper {
    PreTokenizerWrapper::Sequence(tokenizers::pre_tokenizers::sequence::Sequence::new(vec![
        PreTokenizerWrapper::Split(
            Split::new(
                tokenizers::pre_tokenizers::split::SplitPattern::Regex(QWEN2_PATTERN.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]))
}

/// The regex engine's own view of a scalar, from how many splits it makes around it.
fn backend_class(pre: &ByteLevel, c: char) -> Class {
    let splits = |text: &str| backend_splits(pre, text).len();
    if splits(&format!("a{c}a")) == 1 {
        Class::Letter
    } else if splits(&format!("1{c}1")) == 1 {
        Class::Number
    } else if splits(&format!("!{c}!")) == 1 {
        Class::Other
    } else {
        Class::Whitespace
    }
}

#[test]
fn unicode_tables_match_the_regex_engine_on_a_dense_sample() {
    let pre = ByteLevel::new(false, false, true);
    let mut checked = 0;
    let mut code = 0u32;
    while code <= char::MAX as u32 {
        if let Some(c) = char::from_u32(code) {
            assert_eq!(classify(c), backend_class(&pre, c), "U+{code:04X}");
            checked += 1;
        }
        // Dense below U+3000 (Latin through CJK punctuation), then a stride that still lands on
        // every block boundary region; the generator covered every scalar once.
        code += if code < 0x3000 { 1 } else { 61 };
    }
    assert!(checked > 30_000, "{checked}");
}

#[test]
fn pretokenizer_matches_backend_on_corpus_and_random_text() {
    let gpt2 = ByteLevel::new(false, false, true);
    let qwen2 = qwen2_backend_pre_tokenizer();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut texts: Vec<String> = corpus_texts().into_iter().map(|(_, t)| t).collect();
    for _ in 0..3000 {
        texts.push(random_text(&mut rng, &[]));
    }
    let mut folded = 0;
    for text in &texts {
        assert_eq!(
            core_splits(Pattern::Gpt2, text).unwrap(),
            backend_splits(&gpt2, text),
            "GPT-2 pattern on {text:?}"
        );
        match core_splits(Pattern::Qwen2, text) {
            Some(splits) => assert_eq!(
                splits,
                backend_splits(&qwen2, text),
                "Qwen2 pattern on {text:?}"
            ),
            None => folded += 1,
        }
    }
    // Folded contractions are a declared fallback, not silent divergence.
    assert!(folded < texts.len() / 4, "too many fallbacks: {folded}");
}

/// Compare the core with a backend `Tokenizer` loaded from the same JSON, in both special-token
/// modes, on every text; return how many inputs the core declined.
fn assert_core_matches(json: &[u8], label: &str, texts: &[String]) -> usize {
    let core = ByteLevelBpe::from_json(json).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mut backend = Tokenizer::from_bytes(json).unwrap();
    backend.with_truncation(None).unwrap();
    backend.with_padding(None);
    assert_eq!(
        core.vocab_size(),
        backend.get_vocab_size(true),
        "{label}: vocab size"
    );
    let mut out = Output::default();
    let mut declined = 0;
    for recognize in [false, true] {
        backend.set_encode_special_tokens(!recognize);
        for text in texts {
            let expected = backend.encode(text.as_str(), false).unwrap();
            match core.encode(text, recognize, &mut out) {
                Ok(()) => {}
                Err(Fallback::Normalization) => {
                    // Declining is only legitimate when NFC really changes the text.
                    let mut normalized = NormalizedString::from(text.as_str());
                    NormalizerWrapper::NFC(tokenizers::normalizers::unicode::NFC)
                        .normalize(&mut normalized)
                        .unwrap();
                    assert_ne!(
                        normalized.get(),
                        text,
                        "{label}: declined NFC-stable {text:?}"
                    );
                    declined += 1;
                    continue;
                }
                Err(Fallback::Pretokenize(_)) => {
                    assert!(
                        text.contains('\''),
                        "{label}: declined {text:?} without an apostrophe"
                    );
                    declined += 1;
                    continue;
                }
            }
            let context = format!("{label} recognize={recognize} text={text:?}");
            assert_eq!(out.ids, expected.get_ids(), "ids {context}");
            assert_eq!(out.offsets, expected.get_offsets(), "offsets {context}");
            assert_eq!(
                out.special_mask,
                expected.get_special_tokens_mask(),
                "mask {context}"
            );
            let words: Vec<Option<u32>> = out.words.iter().map(|&w| Some(w)).collect();
            assert_eq!(words, expected.get_word_ids(), "words {context}");
            let tokens: Vec<&str> = out.ids.iter().map(|&id| core.token(id).unwrap()).collect();
            let expected_tokens: Vec<&str> =
                expected.get_tokens().iter().map(String::as_str).collect();
            assert_eq!(tokens, expected_tokens, "tokens {context}");
            assert_eq!(
                core.count(text, recognize).unwrap(),
                expected.len(),
                "count {context}"
            );
        }
    }
    declined
}

/// A byte-level BPE with real merges and two special tokens, built in memory.
fn synthetic_bpe(pre_tokenizer: PreTokenizerWrapper, nfc: bool) -> Vec<u8> {
    let alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    let mut vocab: tokenizers::models::bpe::Vocab = alphabet
        .iter()
        .enumerate()
        .map(|(id, c)| (c.to_string(), id as u32))
        .collect();
    let mut merges = Vec::new();
    let mut add = |vocab: &mut tokenizers::models::bpe::Vocab, a: &str, b: &str| {
        let merged = format!("{a}{b}");
        let id = vocab.len() as u32;
        vocab.entry(merged).or_insert(id);
        merges.push((a.to_owned(), b.to_owned()));
    };
    // Ġ is the space byte. Merges include chains and a competing pair to exercise ranks.
    for (a, b) in [
        ("h", "e"),
        ("l", "l"),
        ("he", "ll"),
        ("hell", "o"),
        ("Ġ", "w"),
        ("o", "r"),
        ("Ġw", "or"),
        ("Ġwor", "l"),
        ("Ġworl", "d"),
        ("a", "a"),
        ("aa", "aa"),
        ("Ġ", "Ġ"),
        ("ĠĠ", "ĠĠ"),
        ("1", "2"),
        ("Ċ", "Ċ"),
        ("ã", "ģ"),
        ("ãģ", "¯"),
        ("'", "s"),
        ("Ġ", "a"),
    ] {
        add(&mut vocab, a, b);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
    if nfc {
        tokenizer.with_normalizer(Some(NormalizerWrapper::NFC(
            tokenizers::normalizers::unicode::NFC,
        )));
    }
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|endoftext|>", true).normalized(!nfc),
        AddedToken::from("<|im_start|>", true).normalized(!nfc),
    ]);
    tokenizer.to_string(false).unwrap().into_bytes()
}

#[test]
fn synthetic_artifacts_match_backend_in_both_modes() {
    let specials = ["<|endoftext|>", "<|im_start|>", "<|endof", "text|>"];
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut texts: Vec<String> = corpus_texts().into_iter().map(|(_, t)| t).collect();
    texts.extend(
        [
            "",
            " ",
            "hello world",
            "hello<|endoftext|>world",
            "aaaaaaa",
            "  \n\n x",
        ]
        .map(String::from),
    );
    for _ in 0..2000 {
        texts.push(random_text(&mut rng, &specials));
    }
    let gpt2 = synthetic_bpe(
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, true)),
        false,
    );
    assert_eq!(assert_core_matches(&gpt2, "synthetic gpt2", &texts), 0);
    let qwen2 = synthetic_bpe(qwen2_backend_pre_tokenizer(), true);
    let declined = assert_core_matches(&qwen2, "synthetic qwen2+nfc", &texts);
    assert!(
        declined > 0,
        "the corpus has non-NFC text, so some inputs must be declined"
    );
}

#[test]
fn unsupported_artifacts_are_rejected_with_a_reason() {
    let error = ByteLevelBpe::from_json(include_bytes!("fixtures/wordpiece.json")).unwrap_err();
    assert!(!error.0.is_empty());
    for (file, needle) in [
        ("bert-base-uncased.tokenizer.json", "not recognized"),
        ("xlm-roberta-base.tokenizer.json", "not recognized"),
    ] {
        let path = root().join("models").join(file);
        if !path.is_file() {
            eprintln!("SKIPPED: {} not present", path.display());
            continue;
        }
        let error = ByteLevelBpe::from_json(&fs::read(&path).unwrap()).unwrap_err();
        assert!(
            error.0.contains(needle) || error.0.contains("not BPE"),
            "{file}: {error}"
        );
    }
}

#[test]
fn real_artifacts_match_backend_on_corpus_and_random_text() {
    let mut rng = Rng(0x6a09_e667_f3bc_c909);
    let mut ran = 0;
    for (file, specials) in [
        ("gpt2.tokenizer.json", vec!["<|endoftext|>"]),
        (
            "qwen2.5-0.5b.tokenizer.json",
            vec![
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|fim_prefix|>",
                "<|vision_start|>",
            ],
        ),
    ] {
        let path = root().join("models").join(file);
        if !path.is_file() {
            eprintln!(
                "SKIPPED: {} not present; run `python -I models/fetch.py`",
                path.display()
            );
            continue;
        }
        let json = fs::read(&path).unwrap();
        let mut texts: Vec<String> = corpus_texts().into_iter().map(|(_, t)| t).collect();
        for _ in 0..1500 {
            texts.push(random_text(&mut rng, &specials));
        }
        let declined = assert_core_matches(&json, file, &texts);
        eprintln!(
            "{file}: {} texts compared, {declined} declined to the backend",
            texts.len()
        );
        ran += 1;
    }
    if ran == 0 {
        eprintln!("SKIPPED: no supported real artifacts present");
    }
}

/// The engine selects the core per artifact and mode, and its public API is identical whichever
/// core serves a request.
#[test]
fn engine_selects_the_core_and_stays_identical_to_the_backend_engine() {
    use fast_tokenizer::{CoreMode, EncodeOptions, Engine, SpecialTokens, BACKEND_CORE};

    let wordpiece = Engine::from_bytes(
        include_bytes!("fixtures/wordpiece.json"),
        EncodeOptions::default(),
    )
    .unwrap();
    assert_eq!(wordpiece.artifact().core, BACKEND_CORE);

    let mut rng = Rng(0x3c6e_f372_fe94_f82b);
    let mut texts: Vec<String> = corpus_texts().into_iter().map(|(_, t)| t).collect();
    for _ in 0..300 {
        texts.push(random_text(&mut rng, &["<|endoftext|>", "<|im_start|>"]));
    }
    let mut artifacts = vec![(
        "synthetic gpt2".to_owned(),
        synthetic_bpe(
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, true)),
            false,
        ),
    )];
    for file in ["gpt2.tokenizer.json", "qwen2.5-0.5b.tokenizer.json"] {
        let path = root().join("models").join(file);
        if path.is_file() {
            artifacts.push((file.to_owned(), fs::read(&path).unwrap()));
        } else {
            eprintln!("SKIPPED: {} not present", path.display());
        }
    }
    for (label, json) in &artifacts {
        for special_tokens in [SpecialTokens::Literal, SpecialTokens::Recognize] {
            let options = |core| EncodeOptions {
                add_special_tokens: false,
                special_tokens,
                core,
            };
            let fast = Engine::from_bytes(json, options(CoreMode::Auto)).unwrap();
            let backend = Engine::from_bytes(json, options(CoreMode::Backend)).unwrap();
            let verify = Engine::from_bytes(json, options(CoreMode::Verify)).unwrap();
            assert!(
                fast.artifact().core.starts_with("bytelevel ("),
                "{label}: {}",
                fast.artifact().core
            );
            assert_eq!(backend.artifact().core, BACKEND_CORE, "{label}");
            assert_eq!(fast.artifact().sha256, backend.artifact().sha256);
            for text in &texts {
                let a = fast.encode(text).unwrap();
                let b = backend.encode(text).unwrap();
                assert_eq!(a.ids, b.ids, "{label} {special_tokens:?} ids {text:?}");
                assert_eq!(
                    a.tokens, b.tokens,
                    "{label} {special_tokens:?} tokens {text:?}"
                );
                assert_eq!(
                    a.offsets, b.offsets,
                    "{label} {special_tokens:?} offsets {text:?}"
                );
                assert_eq!(a.special_tokens_mask, b.special_tokens_mask, "{label}");
                assert_eq!(fast.count(text).unwrap(), b.count, "{label} count {text:?}");
                // Verify mode re-encodes with the backend and must agree on every input,
                // including the ones the core declines (those are served by the backend).
                assert_eq!(
                    verify.encode(text).unwrap().ids,
                    b.ids,
                    "{label} verify {text:?}"
                );
                assert_eq!(verify.count(text).unwrap(), b.count);
                for budget in [7, 64] {
                    let fast_chunks = fast.chunk(text, budget).unwrap();
                    let backend_chunks = backend.chunk(text, budget).unwrap();
                    assert_eq!(
                        fast_chunks
                            .iter()
                            .map(|c| (c.start_byte, c.end_byte, c.count))
                            .collect::<Vec<_>>(),
                        backend_chunks
                            .iter()
                            .map(|c| (c.start_byte, c.end_byte, c.count))
                            .collect::<Vec<_>>(),
                        "{label} chunk budget {budget} {text:?}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The differential soak's hard-case catalogue, in test form.
// ---------------------------------------------------------------------------------------------

/// The `n`-th Unicode scalar value, surrogates skipped.
fn nth_scalar(n: u32) -> char {
    char::from_u32(if n < 0xD800 { n } else { n + 0x800 }).unwrap()
}

const SCALAR_COUNT: u32 = 0x11_0000 - 0x800;

/// Added-token spellings from the artifact JSON, in file order.
fn added_token_spellings(json: &[u8]) -> Vec<String> {
    let value: serde_json::Value = serde_json::from_slice(json).unwrap();
    value["added_tokens"]
        .as_array()
        .map(|tokens| {
            tokens
                .iter()
                .filter_map(|t| t["content"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// Inputs shaped like the categories the release soak (`examples/core_soak.rs`) covers, at a
/// size that runs in seconds: the artifact's added tokens whole, doubled, partial, mis-cased and
/// glued to every neighbour class; whitespace and CR/LF mixes; digit runs in several scripts;
/// contractions in every case including folded ones; combining marks and NFC singletons; ZWJ
/// emoji; half-width forms; unassigned, control and byte-alphabet scalars; a strided sweep of
/// every scalar in five contexts; a sample of scalar pairs; and strings crossing regex class
/// boundaries.
fn soak_hard_cases(added: &[String]) -> Vec<String> {
    let mut texts: Vec<String> = Vec::new();
    for token in added {
        let chars: Vec<char> = token.chars().collect();
        let n = chars.len();
        let take = |range: std::ops::Range<usize>| chars[range].iter().collect::<String>();
        texts.push(token.clone());
        texts.push(format!("{token}{token}"));
        texts.push(format!("{token}{token}{token}"));
        texts.push(take(0..n - 1));
        texts.push(take(1..n));
        texts.push(take(0..n / 2));
        texts.push(take(n / 2..n));
        texts.push(format!("{} {}", take(0..n / 2), take(n / 2..n)));
        texts.push(format!("{}{}", take(0..n - 1), take(0..n)));
        texts.push(format!("{}{}", take(0..n), take(1..n)));
        texts.push(token.to_uppercase());
        for (before, after) in [
            ("a", ""),
            ("", "a"),
            ("1", ""),
            ("", "1"),
            (" ", ""),
            ("", " "),
            ("  ", "  "),
            ("\n", ""),
            ("", "\n"),
            ("\r\n", "\r\n"),
            ("", "."),
            ("'", "s"),
            ("don", "'t"),
            ("\u{e9}", "\u{301}"),
            ("<|", "|>"),
            ("\u{1f600}", "\u{1f600}"),
            ("\u{3000}", "\u{85}"),
        ] {
            texts.push(format!("{before}{token}{after}"));
        }
    }
    for (i, a) in added.iter().enumerate() {
        for b in added.iter().skip(i + 1).take(2) {
            texts.push(format!("{a}{b}"));
            texts.push(format!("{b}{a}"));
            let cut: String = a.chars().take(a.chars().count() - 2).collect();
            texts.push(format!("{cut}{b}"));
        }
    }
    const CATALOGUE: &[&str] = &[
        // Whitespace runs, CR/LF mixes, Unicode spaces.
        " ",
        "  ",
        "   ",
        "\t",
        "\t\t",
        " \t ",
        "\n",
        "\n\n",
        "\r\n",
        "\r\n\r\n",
        "\r",
        "\r\r",
        "\r\r\n",
        "\n\r",
        "\r \n",
        " \n",
        "\n ",
        " \n ",
        "  \n  ",
        " \r",
        "\r ",
        "\t\n",
        "\n\t",
        "\u{b}",
        "\u{c}",
        "\u{85}",
        "\u{a0}",
        "\u{a0}\u{a0}",
        " \u{a0} ",
        "\u{a0}\n",
        "\u{1680}",
        "\u{2000}",
        "\u{2007}",
        "\u{200a}",
        "\u{2028}",
        "\u{2029}",
        "\u{202f}",
        "\u{205f}",
        "\u{3000}",
        "\r\u{85}\n",
        "\u{2028}\n",
        " \u{3000}\r\n",
        "\u{2029}\r",
        "a\rb",
        "a\r\nb",
        "a\n\rb",
        "a \r\n b",
        "x   ",
        "   x",
        "x \n",
        "x\n ",
        "! \r\n\r\n",
        ".\r",
        " \r \n ",
        "\u{3000}\r\u{a0}\n\u{2029}x",
        "\u{85}\u{2028} x",
        "\u{a0}\u{a0}a",
        "\u{2028}abc",
        "\tabc",
        " abc",
        "\u{1f600}abc",
        "a\u{a0}b",
        // Digit runs.
        "0",
        "42",
        "1234",
        "1234567",
        "0000000000",
        "9876543210123",
        "3.14159",
        "1,000,000.50",
        "2026-10-18",
        "19:04:00",
        "-5",
        "+3",
        "1e10",
        "0x1F",
        "\u{663}",
        "\u{661}\u{662}\u{663}",
        "\u{967}\u{968}\u{969}",
        "\u{ff11}\u{ff12}\u{ff13}",
        "\u{1d7d8}\u{1d7d9}",
        "\u{b2}\u{b3}\u{b9}",
        "\u{bd}\u{bc}\u{be}",
        "\u{2153}",
        "\u{216b}",
        "\u{2170}",
        "\u{2460}\u{2461}",
        "\u{3007}",
        "12\u{663}4",
        "1\u{b2}",
        "\u{1d7d8}9",
        " 123",
        " \u{663}",
        "a1b2c3",
        "1a2b3c",
        "1 2 3",
        "1\n2",
        // Contractions in every case, folded candidates, apostrophe oddities.
        "don't",
        "DON'T",
        "Don'T",
        "doN'T",
        "'tis",
        "'TIS",
        "it's",
        "IT'S",
        "It'S",
        "'s",
        "'S",
        "'t",
        "'T",
        "'re",
        "'RE",
        "'Re",
        "'rE",
        "'ve",
        "'VE",
        "'m",
        "'M",
        "'ll",
        "'LL",
        "'Ll",
        "'lL",
        "'d",
        "'D",
        "'x",
        "'r",
        "'v",
        "'l",
        "'rr",
        "'le",
        "'",
        "''",
        "'''",
        "''s",
        "'s'",
        "'s's",
        "'ren",
        "'ven",
        "'lls",
        "'dd",
        "'tt",
        "y'all",
        "rock'n'roll",
        "'em",
        "O'Neil",
        "we're",
        "WE'RE",
        "I'm",
        "you'll",
        "YOU'LL",
        "she'd",
        "they've",
        "can't",
        "'90s",
        "'1",
        "' s",
        "'\ts",
        "'\ns",
        "'\u{17f}",
        "'\u{df}",
        "'\u{ff33}",
        "'\u{ff53}",
        "'\u{2bc}s",
        "\u{2019}s",
        "\u{2018}s",
        "'\u{212a}",
        "'\u{130}",
        "'\u{131}",
        "'\u{fb06}",
        "'\u{1e9e}",
        "'\u{e9}",
        "'r\u{e9}",
        "'l\u{13f}",
        "'v\u{e9}",
        "a's",
        "a'S",
        "a'sb",
        "a'Sb",
        "x'll'll",
        "'s'S'\u{17f}",
        // Letters from many scripts, including marks and jamo.
        "na\u{ef}ve",
        "caf\u{e9}",
        "\u{3a9}mega",
        "stra\u{df}e",
        "\u{fb01}ne",
        "\u{17f}",
        "\u{130}stanbul",
        "\u{1c5}",
        "\u{3b1}\u{3b2}\u{3b3}",
        "\u{41f}\u{440}\u{438}",
        "\u{5e9}\u{5dc}\u{5d5}\u{5dd}",
        "\u{645}\u{631}\u{62d}\u{628}\u{627}",
        "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947}",
        "\u{e2a}\u{e27}\u{e31}\u{e2a}\u{e14}\u{e35}",
        "\u{d55c}\u{ad6d}\u{c5b4}",
        "\u{1100}\u{1161}",
        "\u{1100}\u{11a8}",
        "\u{ac00}\u{11a8}",
        "\u{3131}\u{314f}",
        "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}",
        "\u{ff76}\u{ff80}\u{ff76}\u{ff85}",
        "\u{ff76}\u{ff9e}",
        "\u{ff8a}\u{ff9f}",
        "\u{65e5}\u{672c}\u{8a9e}",
        "\u{20000}\u{20001}",
        "\u{ff28}\u{ff45}\u{ff4c}",
        "\u{13a0}\u{13a1}",
        "\u{1781}\u{17d2}\u{1798}",
        "\u{10400}\u{10401}",
        "\u{1d400}\u{1d401}",
        "\u{210c}",
        "\u{2115}",
        "\u{2b0}",
        "\u{24d0}",
        "\u{aa}",
        "\u{b5}",
        "\u{1e9b}",
        "\u{a731}",
        "\u{5d0}\u{5b8}",
        "\u{627}\u{64e}",
        "\u{915}\u{94d}\u{937}",
        // Combining marks and NFC singletons (declined by the NFC artifacts, served by GPT-2).
        "e\u{301}",
        "a\u{308}\u{301}",
        "\u{301}",
        "\u{300}\u{300}",
        "x\u{300}\u{300}",
        "A\u{30a}",
        "\u{c5}",
        "\u{212b}",
        "\u{2126}",
        "\u{344}",
        "\u{f73}",
        "\u{958}",
        "\u{915}\u{93c}",
        "\u{1e9b}\u{323}",
        "\u{1ea1}\u{302}",
        "\u{1ec7}",
        "\u{1100}\u{1161}\u{11a8}",
        "\u{ac01}",
        "\u{e33}",
        "\u{627}\u{653}",
        "\u{622}",
        "\u{64e}\u{651}",
        "\u{651}\u{64e}",
        "\u{34f}",
        "\u{1ab0}",
        "\u{20d0}",
        "\u{20e3}",
        "\u{fe20}",
        "\u{1d165}",
        "\u{e0100}",
        "\u{fe0f}",
        "o\u{308}",
        "\u{f6}",
        "\u{6c}\u{323}\u{304}",
        "\u{1e38}",
        "\u{3b1}\u{301}",
        "\u{391}\u{345}",
        "\u{1fbc}",
        "\u{9c7}\u{9be}",
        "\u{9cb}",
        // Emoji and sequences.
        "\u{1f600}",
        "\u{1f44d}\u{1f3fd}",
        "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}",
        "\u{1f3f3}\u{fe0f}\u{200d}\u{1f308}",
        "\u{1f1f5}\u{1f1f0}",
        "1\u{fe0f}\u{20e3}",
        "#\u{fe0f}\u{20e3}",
        "\u{a9}\u{fe0f}",
        "\u{263a}",
        "\u{2764}\u{fe0f}\u{200d}\u{1f525}",
        "\u{1f1e6}",
        "\u{200d}",
        "\u{e0067}",
        "\u{1f3f4}\u{e0067}\u{e0062}\u{e0065}\u{e006e}\u{e0067}\u{e007f}",
        "\u{1f600}abc1",
        // Half-width and full-width forms.
        "\u{ff71}\u{ff72}\u{ff73}",
        "\u{ff21}\u{ff11}",
        "\u{ff01}\u{ff1f}",
        "\u{ffdc}",
        "\u{ff61}",
        "\u{ff62}\u{ff63}",
        "\u{ff70}",
        "\u{ffe0}",
        "\u{ff10}",
        "\u{3000}\u{ff21}",
        // Unassigned, private, noncharacter, control and format scalars.
        "\u{378}",
        "\u{380}",
        "\u{2fe0}",
        "\u{e0080}",
        "\u{10fffe}",
        "\u{10ffff}",
        "\u{fdd0}",
        "\u{fffe}",
        "\u{ffff}",
        "\u{40000}",
        "\u{e000}",
        "\u{f0000}",
        "\0",
        "\u{1}",
        "\u{7}",
        "\u{1b}",
        "\u{1c}",
        "\u{1f}",
        "\u{7f}",
        "\u{80}",
        "\u{9f}",
        "\u{ad}",
        "\u{200b}",
        "\u{200c}",
        "\u{200e}",
        "\u{202e}",
        "\u{2060}",
        "\u{2066}",
        "\u{feff}",
        "\u{61c}",
        "\u{180e}",
        "\u{e0001}",
        "\u{e007f}",
        "\0\0",
        "\u{1b}[0m",
        "a\0b",
        "a\u{200b}b",
        // Byte-alphabet characters as input text.
        "\u{120}",
        "\u{120}\u{120}",
        "\u{10a}",
        "\u{100}",
        "\u{143}",
        "\u{c3}\u{a9}",
        "\u{120}a",
        "a\u{120}",
        "\u{c4}\u{a0}",
        "\u{e2}\u{80}\u{99}",
        // Punctuation, code and markup.
        "...",
        "---",
        "!!!",
        "?!",
        "<>",
        "</",
        "/>",
        "<|",
        "|>",
        "<|>",
        "\u{2014}",
        "\u{2026}",
        "\u{ab}\u{bb}",
        "\u{201c}\u{201d}",
        "https://example.com/a?b=c&d=e",
        "user@example.com",
        "fn main() { let x = 1; }",
        "x += 1;",
        "a->b",
        "\\n",
        "%d",
        "#!/bin/sh",
        "{\"key\": [1, 2]}",
        "<div class=\"a\">",
        "--flag=value",
        "C:\\Users\\x",
        "1+1=2",
        "$100",
        "100%",
        "#1",
        "@user",
        " !!",
        "!! ",
        " ! !",
    ];
    texts.extend(CATALOGUE.iter().map(|s| (*s).to_owned()));
    // Long pre-tokens past the word-cache limit, and a long run of a special spelling.
    for piece in ["a", " ", "9", "!", "\u{e9}", "\u{1f600}", "\n", "\u{120}"] {
        texts.push(piece.repeat(300));
    }
    texts.push("<|endoftext|>".repeat(30));
    // Every scalar at a stride, in five contexts.
    let mut n = 0;
    while n < SCALAR_COUNT {
        let c = nth_scalar(n);
        for (before, after) in [("", ""), ("'", ""), (" ", ""), ("", "\n"), ("a", "1")] {
            texts.push(format!("{before}{c}{after}"));
        }
        n += 509;
    }
    // Pairs from a sample of hard scalars.
    const SAMPLE: &[char] = &[
        ' ',
        '\t',
        '\n',
        '\r',
        '\u{b}',
        '\0',
        '\'',
        '"',
        '0',
        '9',
        'a',
        'Z',
        's',
        'S',
        'd',
        'l',
        '!',
        '<',
        '|',
        '>',
        '\u{85}',
        '\u{a0}',
        '\u{ad}',
        '\u{b2}',
        '\u{b5}',
        '\u{c5}',
        '\u{e9}',
        '\u{2028}',
        '\u{3000}',
        '\u{200b}',
        '\u{200d}',
        '\u{feff}',
        '\u{300}',
        '\u{301}',
        '\u{20e3}',
        '\u{fe0f}',
        '\u{3b1}',
        '\u{5d0}',
        '\u{639}',
        '\u{915}',
        '\u{94d}',
        '\u{e01}',
        '\u{ac00}',
        '\u{1100}',
        '\u{1161}',
        '\u{ff71}',
        '\u{ff9e}',
        '\u{4e2d}',
        '\u{ff21}',
        '\u{1d7d8}',
        '\u{216b}',
        '\u{17f}',
        '\u{df}',
        '\u{130}',
        '\u{663}',
        '\u{bd}',
        '\u{20ac}',
        '\u{2019}',
        '\u{1f600}',
        '\u{1f3fd}',
        '\u{1f1f5}',
        '\u{e000}',
        '\u{378}',
        '\u{fffe}',
        '\u{10ffff}',
        '\u{212b}',
        '\u{958}',
        '\u{120}',
        '\u{10a}',
        '\u{c3}',
        '\u{fffd}',
    ];
    for &a in SAMPLE {
        for &b in SAMPLE {
            texts.push(format!("{a}{b}"));
        }
    }
    // Strings crossing every fourth boundary between runs of equal regex class.
    let mut previous: Option<(char, Class)> = None;
    let mut boundaries = 0u32;
    for n in 0..SCALAR_COUNT {
        let c = nth_scalar(n);
        let class = classify(c);
        if let Some((p, pclass)) = previous {
            if pclass != class {
                boundaries += 1;
                if boundaries.is_multiple_of(4) {
                    texts.push(format!("{p}{c}"));
                    texts.push(format!("{c}{p}"));
                    texts.push(format!(" {p}{c}a"));
                    texts.push(format!("1{c}{p}\n"));
                    texts.push(format!("'{p}{c}'s"));
                }
            }
        }
        previous = Some((c, class));
    }
    texts
}

/// The soak's categories at test scale: run on the synthetic artifacts everywhere and on the
/// real GPT-2 and Qwen2.5 files when present. A core change or a backend upgrade that breaks
/// any category fails here within seconds, before a full `examples/core_soak.rs` run.
#[test]
fn soak_hard_cases_match_backend_on_synthetic_and_real_artifacts() {
    let mut artifacts = vec![
        (
            "synthetic gpt2".to_owned(),
            synthetic_bpe(
                PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, true)),
                false,
            ),
            false,
        ),
        (
            "synthetic qwen2+nfc".to_owned(),
            synthetic_bpe(qwen2_backend_pre_tokenizer(), true),
            true,
        ),
    ];
    for (file, nfc) in [
        ("gpt2.tokenizer.json", false),
        ("qwen2.5-0.5b.tokenizer.json", true),
    ] {
        let path = root().join("models").join(file);
        if path.is_file() {
            artifacts.push((file.to_owned(), fs::read(&path).unwrap(), nfc));
        } else {
            eprintln!("SKIPPED: {} not present", path.display());
        }
    }
    for (label, json, nfc) in &artifacts {
        let texts = soak_hard_cases(&added_token_spellings(json));
        let declined = assert_core_matches(json, label, &texts);
        if *nfc {
            assert!(declined > 0, "{label}: NFC singletons must be declined");
        } else {
            assert_eq!(
                declined, 0,
                "{label}: a GPT-2 style artifact declines nothing"
            );
        }
        eprintln!(
            "{label}: {} hard cases compared, {declined} declined",
            texts.len()
        );
    }
}
