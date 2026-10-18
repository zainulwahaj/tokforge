use fast_tokenizer::{EncodeOptions, Engine, Error, SpecialTokens};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteDecoder, models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel, Tokenizer,
};

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
