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
