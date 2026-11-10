use tokenizers::{
    decoders::byte_level::ByteLevel as ByteDecoder, models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel, Tokenizer,
};
use tokforge::{EncodeOptions, Engine, Error, SpecialTokens};

const FIXTURE: &[u8] = include_bytes!("fixtures/wordpiece.json");

fn engine(add_special_tokens: bool) -> Engine {
    Engine::from_bytes(
        FIXTURE,
        EncodeOptions {
            add_special_tokens,
            ..Default::default()
        },
    )
    .unwrap()
}

fn byte_engine(dropout: Option<f32>) -> Result<Engine, Error> {
    let vocab: tokenizers::models::bpe::Vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(id, ch)| (ch.to_string(), id as u32))
        .collect();
    let mut model = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap();
    model.dropout = dropout;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
    tokenizer.with_decoder(Some(ByteDecoder::default()));
    Engine::from_bytes(
        tokenizer.to_string(false).unwrap().as_bytes(),
        EncodeOptions::default(),
    )
}

#[test]
fn known_ids_offsets_and_unlimited_counts() {
    let engine = engine(false);
    let encoding = engine.encode("hello café 你好!").unwrap();
    assert_eq!(encoding.ids, [4, 7, 8, 6]);
    assert_eq!(encoding.offsets, [(0, 5), (6, 11), (12, 18), (18, 19)]);
    assert_eq!(
        engine.count("hello world hello world hello world").unwrap(),
        6
    );
    assert_eq!(engine.count("hello").unwrap(), 1);
    assert_eq!(engine.count("").unwrap(), 0);
    assert_eq!(engine.decode(&[4, 5, 6], false).unwrap(), "hello world!");
}

#[test]
fn special_input_and_post_processor_options_are_independent() {
    let literal = engine(false);
    let recognize = Engine::from_bytes(
        FIXTURE,
        EncodeOptions {
            special_tokens: SpecialTokens::Recognize,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!literal.encode("[CLS]").unwrap().ids.contains(&1));
    assert_eq!(recognize.encode("[CLS]").unwrap().ids, [1]);
    let with_special = engine(true);
    assert_eq!(with_special.encode("hello").unwrap().ids, [1, 4, 2]);
    assert_eq!(with_special.count("").unwrap(), 2);
    assert_eq!(with_special.decode(&[1, 4, 2], true).unwrap(), "hello");
    assert_eq!(
        with_special.decode(&[1, 4, 2], false).unwrap(),
        "[CLS] hello [SEP]"
    );
}

#[test]
fn identity_hashes_original_bytes_and_options_do_not_change_it() {
    let first = engine(false);
    assert_eq!(first.artifact().sha256.len(), 64);
    assert_eq!(first.artifact().sha256, engine(true).artifact().sha256);
    let mut changed = FIXTURE.to_vec();
    changed.push(b'\n');
    let second = Engine::from_bytes(&changed, EncodeOptions::default()).unwrap();
    assert_ne!(first.artifact().sha256, second.artifact().sha256);
    assert_eq!(first.artifact().vocab_size, 10);
}

#[test]
fn byte_level_unicode_roundtrip_and_source_partitions() {
    let engine = byte_engine(None).unwrap();
    for text in [
        "",
        "hello\r\nworld  ",
        "café 你好 🦀",
        "a\u{301}👩‍💻\0",
        "مرحبا",
    ] {
        let encoded = engine.encode(text).unwrap();
        assert_eq!(encoded.count, engine.count(text).unwrap());
        assert_eq!(engine.decode(&encoded.ids, false).unwrap(), text);
        for budget in [4, 7, 16, 100] {
            let chunks = engine.chunk(text, budget).unwrap();
            assert_eq!(
                chunks.iter().map(|c| c.text.as_str()).collect::<String>(),
                text
            );
            let mut previous = 0;
            for chunk in chunks {
                assert_eq!(chunk.start_byte, previous);
                assert_eq!(&text[chunk.start_byte..chunk.end_byte], chunk.text);
                assert!(chunk.end_byte > chunk.start_byte);
                assert_eq!(chunk.count, engine.count(&chunk.text).unwrap());
                assert!(chunk.count <= budget);
                previous = chunk.end_byte;
            }
            assert_eq!(previous, text.len());
            let prefix = engine.truncate(text, budget).unwrap();
            assert_eq!(&text[..prefix.end_byte], prefix.text);
            assert!(prefix.count <= budget);
            assert_eq!(prefix.count, engine.count(&prefix.text).unwrap());
        }
    }
}

#[test]
fn budgets_handle_empty_text_zero_and_special_overhead() {
    let bytes = byte_engine(None).unwrap();
    assert_eq!(bytes.truncate("🦀", 3).unwrap().text, "");
    assert!(matches!(
        bytes.chunk("🦀", 3),
        Err(Error::BudgetTooSmall { offset: 0 })
    ));
    assert_eq!(bytes.truncate("hello", 0).unwrap().text, "");
    assert!(bytes.chunk("hello", 0).is_err());
    assert!(bytes.chunk("", 0).unwrap().is_empty());
    assert!(matches!(
        engine(true).truncate("hello", 1),
        Err(Error::EmptyBudget)
    ));
    assert_eq!(engine(true).truncate("hello", 2).unwrap().count, 2);
    for chunk in engine(true).chunk("hello world hello", 3).unwrap() {
        assert!(chunk.count <= 3);
        assert_eq!(chunk.count, engine(true).count(&chunk.text).unwrap());
    }
}

#[test]
fn rejects_bad_artifacts_unknown_ids_and_dropout() {
    assert!(Engine::from_bytes(b"{}", EncodeOptions::default()).is_err());
    assert!(matches!(
        engine(false).decode(&[99], false),
        Err(Error::UnknownId(99))
    ));
    assert!(matches!(byte_engine(Some(0.5)), Err(Error::Dropout)));
    assert!(byte_engine(Some(0.0)).is_ok());
}

#[test]
fn nonmonotonic_counts_do_not_cause_false_no_prefix_errors() {
    // "abcdef" merges down to one token. Shorter prefixes have several tokens;
    // binary probing of this ten-character source misses the fitting prefix.
    let model = BPE::builder()
        .vocab_and_merges(
            [
                ("a", 0),
                ("b", 1),
                ("c", 2),
                ("d", 3),
                ("e", 4),
                ("f", 5),
                ("x", 6),
                ("ef", 7),
                ("def", 8),
                ("cdef", 9),
                ("bcdef", 10),
                ("abcdef", 11),
                ("aabcdef", 12),
            ]
            .into_iter()
            .map(|(s, id)| (s.to_owned(), id))
            .collect::<tokenizers::models::bpe::Vocab>(),
            [
                ("e", "f"),
                ("d", "ef"),
                ("c", "def"),
                ("b", "cdef"),
                ("a", "bcdef"),
                ("a", "abcdef"),
            ]
            .into_iter()
            .map(|(a, b)| (a.to_owned(), b.to_owned()))
            .collect(),
        )
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    // Expand the initial scalar so even the one-character prefix exceeds budget.
    tokenizer.with_normalizer(Some(
        tokenizers::normalizers::replace::Replace::new("a", "aa").unwrap(),
    ));
    let engine = Engine::from_bytes(
        tokenizer.to_string(false).unwrap().as_bytes(),
        EncodeOptions::default(),
    )
    .unwrap();
    assert_eq!(engine.count("abcdef").unwrap(), 1);
    assert!(engine.count("abcde").unwrap() > 1);
    let prefix = engine.truncate("abcdefxxxx", 1).unwrap();
    assert_eq!(prefix.text, "abcdef");
    assert_eq!(prefix.count, 1);
}

#[test]
fn chunks_prefer_paragraph_line_sentence_and_word_boundaries() {
    // One token per byte, so budgets translate directly into byte lengths.
    let engine = byte_engine(None).unwrap();
    let texts = |text: &str, budget: usize| -> Vec<String> {
        engine
            .chunk(text, budget)
            .unwrap()
            .into_iter()
            .map(|chunk| chunk.text)
            .collect()
    };
    assert_eq!(
        texts("aaaa\n\nbbbb\n\ncccc", 8),
        ["aaaa\n\n", "bbbb\n\n", "cccc"]
    );
    assert_eq!(texts("a\r\nb", 2), ["a", "\r\n", "b"]);
    for chunk in texts("Hi there. Bye now. Ok.", 12) {
        assert!(chunk.trim_end().ends_with('.'), "{chunk:?}");
    }
    let words = texts("hello world foo bar baz", 8);
    assert!(words.len() >= 3, "{words:?}");
    for pair in words.windows(2) {
        assert!(
            pair[0].ends_with(' ') || pair[1].starts_with(' '),
            "{pair:?}"
        );
    }
    // A long word offers nothing better than byte boundaries and still fills the budget.
    let lengths: Vec<usize> = texts(&"a".repeat(20), 8).iter().map(String::len).collect();
    assert_eq!(lengths, [8, 8, 4]);
    // A paragraph boundary that would leave the chunk under half full yields to a word boundary.
    let heading = texts(&format!("Hi\n\n{}", "word ".repeat(10)), 16);
    assert!(
        heading[0].len() > 4 && heading[0].ends_with(' '),
        "{heading:?}"
    );
}

#[test]
fn truncate_prefers_whole_words_but_fills_long_words() {
    let engine = byte_engine(None).unwrap();
    let prefix = engine.truncate("hello world foo", 8).unwrap();
    assert_eq!(prefix.text.trim_end(), "hello");
    assert!(prefix.count <= 8);
    assert_eq!(engine.truncate(&"a".repeat(20), 8).unwrap().text.len(), 8);
    let whole = engine.truncate("hello world foo", 100).unwrap();
    assert_eq!((whole.text.as_str(), whole.count), ("hello world foo", 15));
}

#[test]
fn decode_validates_ids_without_lookups_for_every_id() {
    let engine = engine(false);
    assert!(engine
        .decode(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], false)
        .is_ok());
    for bad in [10, 63, 64, 65, 1_000_000, u32::MAX] {
        assert!(matches!(engine.decode(&[4, bad], false), Err(Error::UnknownId(id)) if id == bad));
    }
}

/// Literal mode must never let the model itself emit a registered special ID.
/// Appended as its own module: the tests above belong to another session.
mod literal_special_pieces {
    use tokenizers::{
        models::{bpe::BPE, unigram::Unigram, wordpiece::WordPiece},
        pre_tokenizers::{
            bert::BertPreTokenizer,
            byte_level::ByteLevel,
            metaspace::{Metaspace, PrependScheme},
            whitespace::WhitespaceSplit,
        },
        AddedToken, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer,
    };
    use tokforge::{EncodeOptions, Engine, Error, SpecialTokens};

    fn load(tokenizer: &Tokenizer, special_tokens: SpecialTokens) -> Result<Engine, Error> {
        Engine::from_bytes(
            tokenizer.to_string(false).unwrap().as_bytes(),
            EncodeOptions {
                special_tokens,
                ..Default::default()
            },
        )
    }

    fn both(tokenizer: &Tokenizer) -> (Engine, Engine) {
        (
            load(tokenizer, SpecialTokens::Literal).unwrap(),
            load(tokenizer, SpecialTokens::Recognize).unwrap(),
        )
    }

    fn register(tokenizer: &mut Tokenizer, spellings: &[&str]) {
        let tokens: Vec<AddedToken> = spellings
            .iter()
            .map(|spelling| AddedToken::from(*spelling, true))
            .collect();
        tokenizer.add_special_tokens(&tokens);
    }

    fn refused(tokenizer: &Tokenizer, spelling: &str) {
        assert!(
            matches!(
                load(tokenizer, SpecialTokens::Literal),
                Err(Error::UnsuppressibleSpecialToken(token)) if token == spelling
            ),
            "{spelling:?} should be refused in Literal mode"
        );
        assert!(load(tokenizer, SpecialTokens::Recognize).is_ok());
    }

    /// Index is the ID. `b` has no piece, so it is always an unknown; `x` + `a`
    /// beats `xa` by 0.1 only while the unknown penalty is finite (both paths
    /// carry one unknown, so a finite penalty cancels and -inf ties).
    const UNIGRAM: &[(&str, f64)] = &[
        ("<unk>", 0.0),
        ("<s>", 0.0),
        ("\u{2581}", -1.0),
        ("<", -2.0),
        ("s", -2.0),
        (">", -2.0),
        ("a", -2.0),
        ("\u{2581}a", -1.5),
        ("\u{2581}ab", -30.0),
        ("x", -3.0),
        ("u", -2.0),
        ("n", -2.0),
        ("k", -2.0),
        ("xa", -5.1),
    ];

    fn unigram(pieces: &[(&str, f64)]) -> Tokenizer {
        let vocab = pieces
            .iter()
            .map(|(piece, score)| ((*piece).to_owned(), *score))
            .collect();
        let mut tokenizer = Tokenizer::new(Unigram::from(vocab, Some(0), false).unwrap());
        tokenizer.with_pre_tokenizer(Some(Metaspace::new(
            '\u{2581}',
            PrependScheme::Always,
            true,
        )));
        tokenizer.with_decoder(Some(Metaspace::new(
            '\u{2581}',
            PrependScheme::Always,
            true,
        )));
        tokenizer
    }

    #[test]
    fn unigram_literal_never_emits_a_control_piece() {
        let mut tokenizer = unigram(UNIGRAM);
        register(&mut tokenizer, &["<s>", "<unk>"]);
        let (literal, recognize) = both(&tokenizer);
        assert_eq!(literal.artifact().suppressed_special_pieces, 2);
        assert_eq!(recognize.artifact().suppressed_special_pieces, 0);
        assert_eq!(literal.artifact().sha256, recognize.artifact().sha256);
        assert_eq!(literal.artifact().vocab_size, 14);
        // Without suppression the score-0 piece `<s>` wins outright (-1 vs -7).
        let encoded = literal.encode("<s>").unwrap();
        assert_eq!(encoded.ids, [2, 3, 4, 5], "{:?}", encoded.tokens);
        assert_eq!(recognize.encode("<s>").unwrap().ids, [1]);
        // The unknown piece's spelling is suppressed too; its scalars are pieces.
        assert_eq!(literal.encode("<unk>").unwrap().ids, [2, 3, 10, 11, 12, 5]);
        assert_eq!(recognize.encode("<unk>").unwrap().ids, [0]);
        // Unknown handling is untouched: the penalty still derives from the
        // lowest finite score. A piece always beats an unknown run (`▁ab` at
        // -30 still wins), and among paths with one unknown each, `x` + `a`
        // beats `xa`; a -inf penalty would tie them and pick `xa` instead.
        for text in ["ab", "b", "a", "bxa", "xs", "s > a", "<< >>", "u n k"] {
            let ids = literal.encode(text).unwrap().ids;
            assert_eq!(ids, recognize.encode(text).unwrap().ids, "{text:?}");
            assert_eq!(literal.count(text).unwrap(), ids.len());
        }
        assert_eq!(literal.encode("ab").unwrap().ids, [8]);
        assert_eq!(literal.encode("bxa").unwrap().ids, [2, 0, 9, 6]);
        // Suppression changes segmentation only; every ID still decodes.
        for engine in [&literal, &recognize] {
            assert_eq!(engine.decode(&[1], false).unwrap(), "<s>");
            assert_eq!(engine.decode(&[1, 0], false).unwrap(), "<s><unk>");
            assert_eq!(engine.decode(&[1, 7], true).unwrap(), "a");
        }
    }

    #[test]
    fn unigram_refuses_spellings_it_cannot_make_unreachable() {
        // A single-scalar piece disables the unknown fallback at its position.
        let mut tokenizer = unigram(UNIGRAM);
        register(&mut tokenizer, &["x"]);
        refused(&tokenizer, "x");
        // `b` has no piece of its own, so an unknown run could spell `<b>`.
        let mut pieces = UNIGRAM.to_vec();
        pieces.push(("<b>", 0.0));
        let mut tokenizer = unigram(&pieces);
        register(&mut tokenizer, &["<b>"]);
        refused(&tokenizer, "<b>");
        // A special token outside the model vocabulary needs no suppression.
        let mut tokenizer = unigram(UNIGRAM);
        register(&mut tokenizer, &["<pad>"]);
        let (literal, recognize) = both(&tokenizer);
        assert_eq!(literal.artifact().suppressed_special_pieces, 0);
        assert_eq!(recognize.encode("<pad>").unwrap().ids, [14]);
        assert!(!literal.encode("<pad>").unwrap().ids.contains(&14));
    }

    const WORDPIECE: &[&str] = &[
        "[UNK]", "[CLS]", "[SEP]", "[", "CLS", "]", "##CLS", "##]", "hello",
    ];

    fn wordpiece(vocab: &[&str]) -> Tokenizer {
        let vocab: tokenizers::models::bpe::Vocab = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| ((*token).to_owned(), id as u32))
            .collect();
        let model = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        // Splits on whitespace only, so `[CLS]` reaches the model whole.
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer
    }

    #[test]
    fn wordpiece_without_punctuation_splitting_drops_special_entries() {
        let mut tokenizer = wordpiece(WORDPIECE);
        register(&mut tokenizer, &["[UNK]", "[CLS]", "[SEP]"]);
        let (literal, recognize) = both(&tokenizer);
        assert_eq!(literal.artifact().suppressed_special_pieces, 2);
        assert_eq!(literal.artifact().vocab_size, 9);
        assert_eq!(literal.encode("[CLS]").unwrap().ids, [3, 6, 7]);
        assert_eq!(recognize.encode("[CLS]").unwrap().ids, [1]);
        // No subword path remains for `[SEP]`, so it is unknown like any other
        // unknown word. The unknown token's own spelling is the one exemption:
        // the model finds the unknown ID through it.
        assert_eq!(literal.encode("[SEP]").unwrap().ids, [0]);
        assert_eq!(literal.encode("[UNK]").unwrap().ids, [0]);
        assert_eq!(recognize.encode("[SEP] hello").unwrap().ids, [2, 8]);
        assert_eq!(literal.encode("hello").unwrap().ids, [8]);
        for engine in [&literal, &recognize] {
            assert_eq!(
                engine.decode(&[1, 2, 8], false).unwrap(),
                "[CLS] [SEP] hello"
            );
        }
        let mut tokenizer = wordpiece(WORDPIECE);
        register(&mut tokenizer, &["]"]);
        refused(&tokenizer, "]");
        // The committed fixture lists [CLS], [SEP] and [PAD] as entries.
        assert_eq!(super::engine(false).artifact().suppressed_special_pieces, 3);
        let recognize = Engine::from_bytes(
            super::FIXTURE,
            EncodeOptions {
                special_tokens: SpecialTokens::Recognize,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(recognize.artifact().suppressed_special_pieces, 0);
    }

    const BPE_VOCAB: &[&str] = &["<", "s", ">", "<s", "<s>", "a", "b", "ab", "<s>a"];
    const BPE_MERGES: &[(&str, &str)] = &[("<", "s"), ("<s", ">"), ("a", "b"), ("<s>", "a")];

    fn bpe(vocab: &[&str], merges: &[(&str, &str)], ignore_merges: bool) -> Tokenizer {
        let vocab: tokenizers::models::bpe::Vocab = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| ((*token).to_owned(), id as u32))
            .collect();
        let merges = merges
            .iter()
            .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
            .collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(ignore_merges)
            .build()
            .unwrap();
        // No pre-tokenizer: the whole text is one word.
        Tokenizer::new(model)
    }

    #[test]
    fn bpe_drops_the_merges_that_form_a_special_spelling() {
        for ignore_merges in [false, true] {
            let mut tokenizer = bpe(BPE_VOCAB, BPE_MERGES, ignore_merges);
            register(&mut tokenizer, &["<s>"]);
            let (literal, recognize) = both(&tokenizer);
            assert_eq!(literal.artifact().suppressed_special_pieces, 1);
            // `<` + `s` still merges; `<s` + `>` no longer does.
            assert_eq!(literal.encode("<s>").unwrap().ids, [3, 2]);
            assert_eq!(recognize.encode("<s>").unwrap().ids, [4]);
            assert_eq!(recognize.encode("<s>a").unwrap().ids, [4, 5]);
            // With `ignore_merges` the whole word `<s>a` is an ordinary entry.
            let expected: &[u32] = if ignore_merges { &[8] } else { &[3, 2, 5] };
            assert_eq!(literal.encode("<s>a").unwrap().ids, expected);
            for text in ["ab", "a", "b", "s", "<<", ">a"] {
                let ids = literal.encode(text).unwrap().ids;
                assert_eq!(ids, recognize.encode(text).unwrap().ids, "{text:?}");
            }
            for engine in [&literal, &recognize] {
                assert_eq!(engine.decode(&[4, 8], false).unwrap(), "<s> <s>a");
            }
        }
        // The entry exists but no merge produces it (GPT-2's `<|endoftext|>`
        // has this shape): nothing is rebuilt and nothing can emit it.
        let mut tokenizer = bpe(&["<", "s", ">", "<s>"], &[], false);
        register(&mut tokenizer, &["<s>"]);
        let (literal, recognize) = both(&tokenizer);
        assert_eq!(literal.artifact().suppressed_special_pieces, 0);
        assert_eq!(literal.encode("<s>").unwrap().ids, [0, 1, 2]);
        assert_eq!(recognize.encode("<s>").unwrap().ids, [3]);
        let mut tokenizer = bpe(BPE_VOCAB, BPE_MERGES, false);
        register(&mut tokenizer, &["s"]);
        refused(&tokenizer, "s");
    }

    /// The byte-level regex and BERT pre-tokenizers split these spellings before
    /// the model sees them, which is why the real GPT-2 and BERT artifacts need
    /// no rebuild (tests/reference.rs checks the loaded artifacts). Metaspace
    /// and whitespace splitting deliver them whole, so those families do.
    #[test]
    fn regex_and_bert_pre_tokenizers_split_special_spellings() {
        fn splits(pre: &impl PreTokenizer, text: &str) -> Vec<String> {
            let mut pretokenized = PreTokenizedString::from(text);
            pre.pre_tokenize(&mut pretokenized).unwrap();
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(split, _, _)| split.to_owned())
                .collect()
        }
        assert_eq!(
            splits(&ByteLevel::new(false, false, true), "<|endoftext|>"),
            ["<|", "endoftext", "|>"]
        );
        assert_eq!(splits(&BertPreTokenizer, "[CLS]"), ["[", "CLS", "]"]);
        assert_eq!(
            splits(
                &Metaspace::new('\u{2581}', PrependScheme::Always, true),
                "<s>"
            ),
            ["\u{2581}<s>"]
        );
        assert_eq!(splits(&WhitespaceSplit, "[CLS]"), ["[CLS]"]);
    }
}

#[test]
fn auto_is_the_default_core_mode_since_the_soak_passed() {
    // User decision of 2026-10-18: double-encode against the backend by default
    // until the differential soak was clean, then flip this to Auto. The soak
    // (examples/core_soak.rs) passed the same day; PERFORMANCE.md records it.
    use tokforge::CoreMode;
    assert_eq!(EncodeOptions::default().core, CoreMode::Auto);
    assert_eq!(engine(false).options().core, CoreMode::Auto);
}

/// `encode_ids` takes the borrowed fast path; it must give exactly the ids `encode` gives,
/// for every core and option combination the engine can be loaded with.
#[test]
fn encode_ids_matches_encode() {
    let corpus = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/corpus");
    let mut texts: Vec<String> = std::fs::read_dir(&corpus)
        .unwrap()
        .map(|entry| std::fs::read_to_string(entry.unwrap().path()).unwrap())
        .collect();
    texts.extend(["", " ", "hello world!", "hello [SEP] world"].map(String::from));

    let gpt2 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models/gpt2.tokenizer.json");
    let mut engines = vec![engine(false), engine(true)];
    if gpt2.exists() {
        for special_tokens in [SpecialTokens::Literal, SpecialTokens::Recognize] {
            engines.push(
                Engine::from_file(
                    &gpt2,
                    EncodeOptions {
                        special_tokens,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        }
    }
    for engine in &engines {
        for text in &texts {
            assert_eq!(
                engine.encode_ids(text).unwrap(),
                engine.encode(text).unwrap().ids
            );
        }
    }
}
