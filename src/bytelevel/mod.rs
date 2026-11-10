//! A byte-level BPE fast path for artifacts whose every component is understood: GPT-2 or Qwen2
//! regex pre-tokenization, optional NFC normalization, plain BPE merges over the GPT-2 byte
//! alphabet, and added tokens without strip or single-word rules. Output is bit-identical to the
//! Hugging Face backend on the same artifact (`tests/bytelevel.rs` enforces that differentially);
//! anything the core cannot reproduce with certainty is reported as [`Fallback`] so the caller
//! uses the backend for that input.
//!
//! Why it is faster: text is pre-tokenized in place and tokenized as raw bytes, so there is no
//! normalized-string copy with per-character alignments, no conversion of every word into the
//! `Ġ`-alphabet string form, no per-token `String`, and the merge map uses a small fast hash.
//! Counting allocates nothing per token; encoding writes into caller-owned buffers.

mod pretokenize;
mod unicode_tables;

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    hash::{BuildHasherDefault, Hasher},
    sync::RwLock,
};

use serde::Deserialize;
use unicode_normalization_alignments::{is_nfc, is_nfc_quick, IsNormalized};

pub use pretokenize::{
    classify, split as pretokenize_split, Class, Pattern, Unsupported as PretokenizeUnsupported,
    QWEN2_PATTERN,
};

/// Words at least this long are not cached (matches the backend's policy).
const CACHE_MAX_WORD_BYTES: usize = 256;
/// Cached words; insertion stops when full, so memory is bounded per engine.
pub const CACHE_CAPACITY: usize = 10_000;

/// The FxHash recipe: a few multiplies per word, which beats SipHash for `(u32, u32)` merge
/// keys and short byte strings. Hash quality is irrelevant here because keys are not adversarial
/// in a way that matters (a flood of unique words only disables the bounded cache).
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.add(u64::from_le_bytes(*chunk));
        }
        if !rest.is_empty() {
            let mut buffer = [0u8; 8];
            buffer[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buffer));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// The word cache is split by word hash into independent shards. With one lock, every lookup on
/// every thread hit the same lock word, and hits got slower as threads were added.
const CACHE_SHARD_BITS: u32 = 4;
const CACHE_SHARDS: usize = 1 << CACHE_SHARD_BITS;
const SHARD_CAPACITY: usize = CACHE_CAPACITY / CACHE_SHARDS;
const _: () = assert!(CACHE_CAPACITY.is_multiple_of(CACHE_SHARDS));

/// Slots per shard: a power of two, so at `SHARD_CAPACITY` entries the table is at most ~61%
/// full and linear probes stay short.
const SHARD_SLOTS: usize = 1024;
const _: () = assert!(SHARD_SLOTS.is_power_of_two() && SHARD_CAPACITY * 10 <= SHARD_SLOTS * 7);
/// Where the slot bits start in the word hash; they must not overlap the shard bits on top.
const SLOT_SHIFT: u32 = 40;
const _: () = assert!(SLOT_SHIFT + SHARD_SLOTS.trailing_zeros() <= 64 - CACHE_SHARD_BITS);

struct ShardEntry {
    hash: u64,
    word: Box<[u8]>,
    ids: Box<[u32]>,
}

/// Insert-only open-addressing table keyed by a hash the caller already computed, so each word
/// is hashed once: the high bits pick the shard, the low bits the slot. Entries are never
/// removed, which keeps linear probing correct without tombstones.
struct ShardTable {
    slots: Box<[Option<ShardEntry>]>,
    len: usize,
}

impl ShardTable {
    fn new() -> Self {
        Self {
            slots: (0..SHARD_SLOTS).map(|_| None).collect(),
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Home slot from hash bits 40..50: FxHash's multiply mixes upward, so its low bits depend
    /// only on the first few input bytes (every " the…" word would share a home), while the
    /// top 4 bits are taken by the shard index.
    fn home(hash: u64) -> usize {
        (hash >> SLOT_SHIFT) as usize & (SHARD_SLOTS - 1)
    }

    /// Longest distance from an entry's home slot to where it sits (probe length on a hit).
    #[cfg(test)]
    fn max_displacement(&self) -> usize {
        (0..SHARD_SLOTS)
            .filter_map(|slot| {
                let entry = self.slots[slot].as_ref()?;
                Some((slot + SHARD_SLOTS - Self::home(entry.hash)) & (SHARD_SLOTS - 1))
            })
            .max()
            .unwrap_or(0)
    }

    fn get(&self, hash: u64, word: &[u8]) -> Option<&[u32]> {
        let mut slot = Self::home(hash);
        loop {
            match &self.slots[slot] {
                None => return None,
                Some(entry) if entry.hash == hash && *entry.word == *word => {
                    return Some(&entry.ids);
                }
                Some(_) => slot = (slot + 1) & (SHARD_SLOTS - 1),
            }
        }
    }

    /// Adds `word` unless it is already present or the shard is at `SHARD_CAPACITY`.
    fn insert(&mut self, hash: u64, word: &[u8], ids: &[u32]) {
        if self.len >= SHARD_CAPACITY {
            return;
        }
        let mut slot = Self::home(hash);
        loop {
            match &self.slots[slot] {
                None => break,
                Some(entry) if entry.hash == hash && *entry.word == *word => return,
                Some(_) => slot = (slot + 1) & (SHARD_SLOTS - 1),
            }
        }
        self.slots[slot] = Some(ShardEntry {
            hash,
            word: word.into(),
            ids: ids.into(),
        });
        self.len += 1;
    }
}

/// One shard per 128 bytes so two shards' locks never share a cache line.
#[repr(align(128))]
struct CacheShard(RwLock<ShardTable>);

/// Word bytes to their ids, bounded by `CACHE_CAPACITY` in total (`SHARD_CAPACITY` per shard).
struct WordCache {
    shards: Box<[CacheShard]>,
}

impl WordCache {
    fn new() -> Self {
        Self {
            shards: (0..CACHE_SHARDS)
                .map(|_| CacheShard(RwLock::new(ShardTable::new())))
                .collect(),
        }
    }

    /// The word's FxHash and the shard that owns it (picked by the high bits, which the
    /// multiply mixes best; the table uses the low bits).
    fn locate(&self, word: &[u8]) -> (u64, &RwLock<ShardTable>) {
        let mut hasher = FxHasher::default();
        hasher.write(word);
        let hash = hasher.finish();
        (
            hash,
            &self.shards[(hash >> (64 - CACHE_SHARD_BITS)) as usize].0,
        )
    }

    fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.0.read().unwrap_or_else(|e| e.into_inner()).len())
            .sum()
    }
}

/// The artifact has a component this core does not reproduce; load it with the backend only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn unsupported<T>(reason: impl Into<String>) -> Result<T, Unsupported> {
    Err(Unsupported(reason.into()))
}

/// This particular input needs the backend even though the artifact is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// NFC normalization would change the text, so offsets need the backend's alignments.
    Normalization,
    Pretokenize(PretokenizeUnsupported),
}

#[derive(Deserialize)]
struct ArtifactJson {
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    post_processor: Option<serde_json::Value>,
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    model: ModelJson,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default = "default_true")]
    normalized: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
struct ModelJson {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    dropout: Option<f32>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
    vocab: serde_json::Value,
    merges: Vec<MergeJson>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MergeJson {
    Text(String),
    Pair(String, String),
}

/// GPT-2's reversible byte-to-character alphabet. Printable Latin-1 bytes stand for themselves;
/// the 68 others are mapped onto U+0100 upward, in byte order.
fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let mut forward = ['\0'; 256];
    let mut inverse = HashMap::with_capacity(256);
    let mut next = 256u32;
    for byte in 0..=255u8 {
        let printable = (b'!'..=b'~').contains(&byte)
            || (0xA1..=0xAC).contains(&byte)
            || (0xAE..=0xFF).contains(&byte);
        let c = if printable {
            char::from(byte)
        } else {
            let c = char::from_u32(next).unwrap();
            next += 1;
            c
        };
        forward[byte as usize] = c;
        inverse.insert(c, byte);
    }
    (forward, inverse)
}

fn token_bytes(token: &str, inverse: &HashMap<char, u8>) -> Option<Vec<u8>> {
    token.chars().map(|c| inverse.get(&c).copied()).collect()
}

struct AddedToken {
    bytes: Box<[u8]>,
    id: u32,
    /// Skipped in literal mode (the backend's `encode_special_tokens`); a non-special added token
    /// is matched in both modes.
    special: bool,
}

/// Added tokens matched in one pass. The backend runs a leftmost-longest Aho-Corasick automaton
/// over all of them and then drops special matches in literal mode, so the dropped span rejoins
/// the surrounding text; this reproduces that order with a first-byte filter instead of an
/// automaton, which is plenty for the handful of added tokens these artifacts carry.
struct AddedSet {
    /// Longest first, so the first match at a position is the leftmost-longest one.
    tokens: Vec<AddedToken>,
    first_bytes: [bool; 256],
}

impl Default for AddedSet {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            first_bytes: [false; 256],
        }
    }
}

impl AddedSet {
    fn push(&mut self, token: AddedToken) {
        self.first_bytes[token.bytes[0] as usize] = true;
        self.tokens.push(token);
        self.tokens
            .sort_by(|a, b| b.bytes.len().cmp(&a.bytes.len()).then(a.id.cmp(&b.id)));
    }

    /// Leftmost-longest match starting in `from..to` and ending by `to`.
    fn next_match(&self, bytes: &[u8], from: usize, to: usize) -> Option<(usize, &AddedToken)> {
        if self.tokens.is_empty() {
            return None;
        }
        let window = &bytes[..to];
        for (index, &byte) in window.iter().enumerate().skip(from) {
            if !self.first_bytes[byte as usize] {
                continue;
            }
            for token in &self.tokens {
                if window[index..].starts_with(&token.bytes) {
                    return Some((index, token));
                }
            }
        }
        None
    }
}

/// Caller-owned encode buffers, reused across calls so steady-state encoding allocates only
/// when a text has more tokens than any earlier one.
#[derive(Default, Debug)]
pub struct Output {
    pub ids: Vec<u32>,
    /// Source byte ranges with the backend's semantics: a token covering part of a multi-byte
    /// scalar reports that whole scalar, so ranges may overlap or repeat.
    pub offsets: Vec<(usize, usize)>,
    /// Pre-token (word) index of each token; added tokens are words of their own.
    pub words: Vec<u32>,
    /// Backend semantics: 1 only for tokens a post-processor adds. Supported artifacts add none,
    /// so this is all zeros, including for recognized special spellings.
    pub special_mask: Vec<u32>,
}

impl Output {
    pub fn clear(&mut self) {
        self.ids.clear();
        self.offsets.clear();
        self.words.clear();
        self.special_mask.clear();
    }

    fn push(&mut self, id: u32, offsets: (usize, usize), word: u32) {
        self.ids.push(id);
        self.offsets.push(offsets);
        self.words.push(word);
        self.special_mask.push(0);
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    prev: i32,
    next: i32,
    /// Byte length; zero marks a symbol merged into its left neighbour.
    len: u32,
}

#[derive(PartialEq, Eq)]
struct Merge {
    rank: u32,
    pos: u32,
    new_id: u32,
}

// Lowest rank first, then leftmost position: the same order the backend's heap pops in.
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-call scratch space for the merge loop.
#[derive(Default)]
struct Scratch {
    symbols: Vec<Symbol>,
    heap: BinaryHeap<Merge>,
    ids: Vec<u32>,
}

pub struct ByteLevelBpe {
    pattern: Pattern,
    nfc: bool,
    ignore_merges: bool,
    /// Token bytes to id, for `ignore_merges` whole-word lookups and merge construction.
    vocab: FxMap<Box<[u8]>, u32>,
    /// Backend-identical token strings, indexed by id (empty for unused ids).
    tokens: Vec<Box<str>>,
    /// Byte length of each token, indexed by id.
    lens: Vec<u32>,
    byte_ids: [u32; 256],
    /// (left id << 32 | right id) to (rank, merged id).
    merges: FxMap<u64, (u32, u32)>,
    /// Added tokens matched on the raw text (`normalized: false`), then those matched after
    /// normalization (`normalized: true`); with identity normalization both see the same bytes.
    raw_added: AddedSet,
    normalized_added: AddedSet,
    added_count: usize,
    cache: WordCache,
    vocab_size: usize,
}

impl std::fmt::Debug for ByteLevelBpe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteLevelBpe")
            .field("pattern", &self.pattern)
            .field("nfc", &self.nfc)
            .field("vocab_size", &self.vocab_size)
            .field("added_tokens", &self.added_count)
            .finish()
    }
}

#[inline]
fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

impl ByteLevelBpe {
    /// Build the fast path from the exact `tokenizer.json` bytes, or explain why this artifact
    /// stays on the backend.
    pub fn from_json(bytes: &[u8]) -> Result<Self, Unsupported> {
        let artifact: ArtifactJson = match serde_json::from_slice(bytes) {
            Ok(artifact) => artifact,
            Err(error) => return unsupported(format!("artifact shape not recognized: {error}")),
        };
        let model = &artifact.model;
        if let Some(kind) = &model.kind {
            if kind != "BPE" {
                return unsupported(format!("model type {kind} is not BPE"));
            }
        }
        if model.dropout.is_some_and(|d| d > 0.0) {
            return unsupported("BPE dropout is enabled");
        }
        if model
            .continuing_subword_prefix
            .as_deref()
            .is_some_and(|p| !p.is_empty())
            || model
                .end_of_word_suffix
                .as_deref()
                .is_some_and(|s| !s.is_empty())
        {
            return unsupported("BPE uses a subword prefix or suffix");
        }
        if model.byte_fallback {
            return unsupported("BPE byte_fallback is enabled");
        }
        let nfc = match &artifact.normalizer {
            None | Some(serde_json::Value::Null) => false,
            Some(value) if value.get("type").and_then(|t| t.as_str()) == Some("NFC") => true,
            Some(value) => {
                return unsupported(format!(
                    "normalizer {} is not supported",
                    value.get("type").and_then(|t| t.as_str()).unwrap_or("?")
                ))
            }
        };
        let pattern = detect_pattern(artifact.pre_tokenizer.as_ref())?;
        match &artifact.post_processor {
            None | Some(serde_json::Value::Null) => {}
            Some(value)
                if value.get("type").and_then(|t| t.as_str()) == Some("ByteLevel")
                    && value.get("trim_offsets").and_then(|t| t.as_bool()) != Some(true) => {}
            Some(value) => {
                return unsupported(format!(
                    "post-processor {} is not supported",
                    value.get("type").and_then(|t| t.as_str()).unwrap_or("?")
                ))
            }
        }
        let vocab_json: HashMap<String, u32> = match serde_json::from_value(model.vocab.clone()) {
            Ok(vocab) => vocab,
            Err(_) => return unsupported("vocabulary is not a token-to-id map"),
        };

        let (_, inverse) = byte_alphabet();
        let mut raw_added = AddedSet::default();
        let mut normalized_added = AddedSet::default();
        for token in &artifact.added_tokens {
            if token.lstrip || token.rstrip || token.single_word {
                return unsupported(format!(
                    "added token {:?} uses lstrip, rstrip or single_word",
                    token.content
                ));
            }
            if token.content.is_empty() {
                return unsupported("empty added token");
            }
            let added = AddedToken {
                bytes: token.content.as_bytes().into(),
                id: token.id,
                special: token.special,
            };
            if token.normalized {
                normalized_added.push(added);
            } else {
                raw_added.push(added);
            }
        }

        let max_id = vocab_json
            .values()
            .chain(artifact.added_tokens.iter().map(|t| &t.id))
            .max()
            .copied()
            .map_or(0, |id| id as usize + 1);
        let mut tokens: Vec<Box<str>> = vec![Box::from(""); max_id];
        let mut lens = vec![0u32; max_id];
        let mut vocab: FxMap<Box<[u8]>, u32> = FxMap::default();
        vocab.reserve(vocab_json.len());
        let mut extra_added = 0;
        for token in &artifact.added_tokens {
            if !vocab_json.contains_key(&token.content) {
                extra_added += 1;
                tokens[token.id as usize] = token.content.as_str().into();
                lens[token.id as usize] = token.content.len() as u32;
            }
        }
        for (token, &id) in &vocab_json {
            let Some(bytes) = token_bytes(token, &inverse) else {
                if artifact.added_tokens.iter().any(|t| &t.content == token) {
                    tokens[id as usize] = token.as_str().into();
                    lens[id as usize] = token.len() as u32;
                    continue;
                }
                return unsupported(format!(
                    "vocabulary token {token:?} is outside the byte-level alphabet"
                ));
            };
            tokens[id as usize] = token.as_str().into();
            lens[id as usize] = bytes.len() as u32;
            vocab.insert(bytes.into_boxed_slice(), id);
        }
        let mut byte_ids = [0u32; 256];
        for byte in 0..=255u8 {
            match vocab.get(&[byte][..]) {
                Some(&id) => byte_ids[byte as usize] = id,
                None => return unsupported(format!("byte {byte:#04x} has no vocabulary token")),
            }
        }

        let mut merges: FxMap<u64, (u32, u32)> = FxMap::default();
        merges.reserve(model.merges.len());
        for (rank, merge) in model.merges.iter().enumerate() {
            let (left, right) = match merge {
                MergeJson::Pair(a, b) => (a.as_str(), b.as_str()),
                MergeJson::Text(text) => match text.split_once(' ') {
                    Some(pair) => pair,
                    None => return unsupported(format!("merge {text:?} is not a pair")),
                },
            };
            let (Some(left_bytes), Some(right_bytes)) =
                (token_bytes(left, &inverse), token_bytes(right, &inverse))
            else {
                return unsupported(format!("merge {left:?} {right:?} is outside the alphabet"));
            };
            let mut merged = left_bytes.clone();
            merged.extend_from_slice(&right_bytes);
            let (Some(&left_id), Some(&right_id), Some(&new_id)) = (
                vocab.get(&left_bytes[..]),
                vocab.get(&right_bytes[..]),
                vocab.get(&merged[..]),
            ) else {
                return unsupported(format!("merge {left:?} {right:?} names unknown tokens"));
            };
            merges
                .entry(pair_key(left_id, right_id))
                .or_insert((rank as u32, new_id));
        }

        Ok(Self {
            pattern,
            nfc,
            ignore_merges: model.ignore_merges,
            vocab,
            tokens,
            lens,
            byte_ids,
            merges,
            raw_added,
            normalized_added,
            added_count: artifact.added_tokens.len(),
            cache: WordCache::new(),
            vocab_size: vocab_json.len() + extra_added,
        })
    }

    /// Number of words in the word cache (for tests and benchmarks).
    #[doc(hidden)]
    pub fn cached_words(&self) -> usize {
        self.cache.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn pattern(&self) -> Pattern {
        self.pattern
    }

    pub fn describe(&self) -> String {
        format!(
            "byte-level BPE, {} pre-tokenizer, {}, {} added tokens",
            match self.pattern {
                Pattern::Gpt2 => "GPT-2",
                Pattern::Qwen2 => "Qwen2",
            },
            if self.nfc {
                "NFC normalizer"
            } else {
                "no normalizer"
            },
            self.added_count
        )
    }

    /// The backend's token string for an id, or `None` for an id outside the vocabulary.
    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens
            .get(id as usize)
            .map(|t| &**t)
            .filter(|t| !t.is_empty())
    }

    /// Whether the normalizer would leave `text` unchanged, so offsets are plain byte ranges.
    fn normalization_is_identity(&self, text: &str) -> bool {
        if !self.nfc || text.is_ascii() {
            return true;
        }
        match is_nfc_quick(text.chars()) {
            IsNormalized::Yes => true,
            IsNormalized::No => false,
            IsNormalized::Maybe => is_nfc(text),
        }
    }

    /// Token count of `text`; special spellings are recognized only when `recognize` is set.
    pub fn count(&self, text: &str, recognize: bool) -> Result<usize, Fallback> {
        if !self.normalization_is_identity(text) {
            return Err(Fallback::Normalization);
        }
        let mut scratch = Scratch::default();
        let mut total = 0;
        self.walk(text, recognize, &mut scratch, |segment| match segment {
            Segment::Added(..) => total += 1,
            Segment::Word(ids, _) => total += ids.len(),
        })?;
        Ok(total)
    }

    /// Encode `text` into `out` (cleared first).
    pub fn encode(&self, text: &str, recognize: bool, out: &mut Output) -> Result<(), Fallback> {
        out.clear();
        if !self.normalization_is_identity(text) {
            return Err(Fallback::Normalization);
        }
        let mut scratch = Scratch::default();
        let mut word = 0u32;
        self.walk(text, recognize, &mut scratch, |segment| {
            match segment {
                Segment::Added(id, start, end) => out.push(id, (start, end), word),
                Segment::Word(ids, start) => {
                    let mut byte = start;
                    for &id in ids {
                        let len = self.lens[id as usize] as usize;
                        let range = (floor_char(text, byte), ceil_char(text, byte + len));
                        out.push(id, range, word);
                        byte += len;
                    }
                }
            }
            word += 1;
        })?;
        Ok(())
    }

    /// Drive added-token matching and pre-tokenization, handing each segment to `visit`.
    fn walk(
        &self,
        text: &str,
        recognize: bool,
        scratch: &mut Scratch,
        mut visit: impl FnMut(Segment<'_>),
    ) -> Result<(), Fallback> {
        // Pass 1: raw-text added tokens over the whole text; pass 2: normalized added tokens
        // inside each remaining piece; then the regex pre-tokenizer inside what is left.
        let visit: &mut dyn FnMut(Segment<'_>) = &mut visit;
        self.split_added(
            &self.raw_added,
            text,
            0,
            text.len(),
            recognize,
            &mut |piece| match piece {
                Piece::Added(id, start, end) => {
                    visit(Segment::Added(id, start, end));
                    Ok(())
                }
                Piece::Text(from, to) => self.split_added(
                    &self.normalized_added,
                    text,
                    from,
                    to,
                    recognize,
                    &mut |piece| match piece {
                        Piece::Added(id, start, end) => {
                            visit(Segment::Added(id, start, end));
                            Ok(())
                        }
                        Piece::Text(from, to) => self.words(text, from, to, scratch, visit),
                    },
                ),
            },
        )
    }

    /// Report added tokens of `set` inside `text[from..to]` and the text between them, in order.
    fn split_added(
        &self,
        set: &AddedSet,
        text: &str,
        from: usize,
        to: usize,
        recognize: bool,
        on: &mut dyn FnMut(Piece) -> Result<(), Fallback>,
    ) -> Result<(), Fallback> {
        let bytes = text.as_bytes();
        let mut piece_start = from;
        let mut cursor = from;
        while let Some((start, token)) = set.next_match(bytes, cursor, to) {
            let end = start + token.bytes.len();
            cursor = end;
            if token.special && !recognize {
                // Literal mode: the spelling stays ordinary text (the backend skips the match
                // but its automaton has already consumed the span, so scanning resumes after it).
                continue;
            }
            if piece_start < start {
                on(Piece::Text(piece_start, start))?;
            }
            on(Piece::Added(token.id, start, end))?;
            piece_start = end;
        }
        if piece_start < to {
            on(Piece::Text(piece_start, to))?;
        }
        Ok(())
    }

    /// Pre-tokenize `text[from..to]` and emit each word's ids.
    fn words(
        &self,
        text: &str,
        from: usize,
        to: usize,
        scratch: &mut Scratch,
        visit: &mut dyn FnMut(Segment<'_>),
    ) -> Result<(), Fallback> {
        let segment = &text[from..to];
        let mut failed = None;
        let result = pretokenize::split(self.pattern, segment, |s, e| {
            if failed.is_some() {
                return;
            }
            match self.word_ids(&segment.as_bytes()[s..e], scratch) {
                Ok(()) => visit(Segment::Word(&scratch.ids, from + s)),
                Err(fallback) => failed = Some(fallback),
            }
        });
        if let Err(reason) = result {
            return Err(Fallback::Pretokenize(reason));
        }
        failed.map_or(Ok(()), Err)
    }

    /// Leave the ids of `word` in `scratch.ids`, from the cache when possible.
    fn word_ids(&self, word: &[u8], scratch: &mut Scratch) -> Result<(), Fallback> {
        scratch.ids.clear();
        let located = (word.len() < CACHE_MAX_WORD_BYTES).then(|| self.cache.locate(word));
        // Whether the shard had room when we looked; a full shard never takes the write lock.
        let mut room = false;
        if let Some((hash, shard)) = located {
            let cache = shard.read().unwrap_or_else(|e| e.into_inner());
            if let Some(ids) = cache.get(hash, word) {
                scratch.ids.extend_from_slice(ids);
                return Ok(());
            }
            room = cache.len() < SHARD_CAPACITY;
        }
        self.merge(word, scratch);
        if let (true, Some((hash, shard))) = (room, located) {
            // insert re-checks capacity: another thread may have filled the shard meanwhile.
            shard
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(hash, word, &scratch.ids);
        }
        Ok(())
    }

    /// Byte-level BPE over one pre-token, with the backend's merge order: lowest rank first,
    /// leftmost on ties, stale heap entries skipped.
    fn merge(&self, word: &[u8], scratch: &mut Scratch) {
        let Scratch { symbols, heap, ids } = scratch;
        symbols.clear();
        heap.clear();
        if self.ignore_merges {
            if let Some(&id) = self.vocab.get(word) {
                ids.push(id);
                return;
            }
        }
        let n = word.len();
        symbols.extend(word.iter().enumerate().map(|(i, &b)| Symbol {
            id: self.byte_ids[b as usize],
            prev: i as i32 - 1,
            next: if i + 1 < n { i as i32 + 1 } else { -1 },
            len: 1,
        }));
        for i in 0..n.saturating_sub(1) {
            if let Some(&(rank, new_id)) =
                self.merges.get(&pair_key(symbols[i].id, symbols[i + 1].id))
            {
                heap.push(Merge {
                    rank,
                    pos: i as u32,
                    new_id,
                });
            }
        }
        while let Some(top) = heap.pop() {
            let pos = top.pos as usize;
            let left = symbols[pos];
            if left.len == 0 || left.next < 0 {
                continue;
            }
            let right_pos = left.next as usize;
            let right = symbols[right_pos];
            // Skip entries made stale by an earlier merge of either side.
            match self.merges.get(&pair_key(left.id, right.id)) {
                Some(&(_, new_id)) if new_id == top.new_id => {}
                _ => continue,
            }
            symbols[pos].id = top.new_id;
            symbols[pos].len += right.len;
            symbols[pos].next = right.next;
            symbols[right_pos].len = 0;
            if right.next >= 0 {
                symbols[right.next as usize].prev = pos as i32;
            }
            let current = symbols[pos];
            if current.prev >= 0 {
                let prev = symbols[current.prev as usize];
                if let Some(&(rank, new_id)) = self.merges.get(&pair_key(prev.id, current.id)) {
                    heap.push(Merge {
                        rank,
                        pos: current.prev as u32,
                        new_id,
                    });
                }
            }
            if current.next >= 0 {
                let next = symbols[current.next as usize];
                if let Some(&(rank, new_id)) = self.merges.get(&pair_key(current.id, next.id)) {
                    heap.push(Merge {
                        rank,
                        pos: pos as u32,
                        new_id,
                    });
                }
            }
        }
        ids.extend(symbols.iter().filter(|s| s.len > 0).map(|s| s.id));
    }
}

enum Segment<'a> {
    /// A matched added token: id, start byte, end byte.
    Added(u32, usize, usize),
    /// A pre-token's ids and its start byte in the text.
    Word(&'a [u32], usize),
}

/// One added-token pass splits a byte range into matched tokens and the text between them.
enum Piece {
    Added(u32, usize, usize),
    Text(usize, usize),
}

fn floor_char(text: &str, mut index: usize) -> usize {
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_char(text: &str, mut index: usize) -> usize {
    while !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

fn detect_pattern(pre_tokenizer: Option<&serde_json::Value>) -> Result<Pattern, Unsupported> {
    let Some(value) = pre_tokenizer else {
        return unsupported("no pre-tokenizer");
    };
    let kind = value.get("type").and_then(|t| t.as_str()).unwrap_or("?");
    let is_plain_byte_level = |v: &serde_json::Value, use_regex: bool| {
        v.get("type").and_then(|t| t.as_str()) == Some("ByteLevel")
            && v.get("add_prefix_space").and_then(|t| t.as_bool()) != Some(true)
            && v.get("use_regex").and_then(|t| t.as_bool()).unwrap_or(true) == use_regex
    };
    if is_plain_byte_level(value, true) {
        return Ok(Pattern::Gpt2);
    }
    if kind == "Sequence" {
        if let Some([split, byte_level]) = value
            .get("pretokenizers")
            .and_then(|p| p.as_array())
            .map(|p| p.as_slice())
        {
            let regex = split
                .get("pattern")
                .and_then(|p| p.get("Regex"))
                .and_then(|r| r.as_str());
            if split.get("type").and_then(|t| t.as_str()) == Some("Split")
                && regex == Some(QWEN2_PATTERN)
                && split.get("behavior").and_then(|b| b.as_str()) == Some("Isolated")
                && split.get("invert").and_then(|i| i.as_bool()) != Some(true)
                && is_plain_byte_level(byte_level, false)
            {
                return Ok(Pattern::Qwen2);
            }
        }
    }
    unsupported(format!(
        "pre-tokenizer {kind} is not a supported byte-level configuration"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_table_finds_words_whose_hashes_collide() {
        let mut table = ShardTable::new();
        // Same home slot (all slot bits zero), different full hashes; then an exact collision.
        assert_eq!(
            ShardTable::home(0x10),
            ShardTable::home(0x10 + SHARD_SLOTS as u64)
        );
        table.insert(0x10, b"alpha", &[1]);
        table.insert(0x10 + SHARD_SLOTS as u64, b"beta", &[2, 3]);
        table.insert(0x10, b"gamma", &[4]);
        assert_eq!(table.get(0x10, b"alpha"), Some(&[1u32][..]));
        assert_eq!(
            table.get(0x10 + SHARD_SLOTS as u64, b"beta"),
            Some(&[2u32, 3][..])
        );
        assert_eq!(table.get(0x10, b"gamma"), Some(&[4u32][..]));
        assert_eq!(table.get(0x10, b"delta"), None);
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn shard_table_spreads_real_words_across_slots() {
        // Short space-prefixed words, as the pre-tokenizer produces them. FxHash's low bits
        // depend only on the first input bytes here, so indexing by them piles words up.
        let mut table = ShardTable::new();
        let mut index = 0usize;
        while table.len() < SHARD_CAPACITY {
            let word = format!(
                " {}",
                ["the", "and", "for", "with", "that", "from"][index % 6]
            ) + &index.to_string();
            let mut hasher = FxHasher::default();
            hasher.write(word.as_bytes());
            table.insert(hasher.finish(), word.as_bytes(), &[0]);
            index += 1;
        }
        let worst = table.max_displacement();
        assert!(worst <= 32, "longest probe run is {worst} slots");
    }

    #[test]
    fn shard_table_ignores_duplicates_and_stops_at_capacity() {
        let mut table = ShardTable::new();
        table.insert(7, b"same", &[1]);
        table.insert(7, b"same", &[9]);
        assert_eq!(table.get(7, b"same"), Some(&[1u32][..]));
        for i in 0..2 * SHARD_CAPACITY as u64 {
            table.insert(i * 31, &i.to_le_bytes(), &[0]);
        }
        assert_eq!(table.len(), SHARD_CAPACITY);
    }

    #[test]
    fn alphabet_is_a_bijection_matching_the_gpt2_table() {
        let (forward, inverse) = byte_alphabet();
        assert_eq!(forward[b' ' as usize], '\u{120}'); // Ġ
        assert_eq!(forward[b'\n' as usize], '\u{10a}'); // Ċ
        assert_eq!(forward[b'a' as usize], 'a');
        assert_eq!(forward[0xff], '\u{ff}');
        assert_eq!(inverse.len(), 256);
        for byte in 0..=255u8 {
            assert_eq!(inverse[&forward[byte as usize]], byte);
        }
    }

    #[test]
    fn rejects_artifacts_with_unsupported_components() {
        let wordpiece = include_bytes!("../../tests/fixtures/wordpiece.json");
        let error = ByteLevelBpe::from_json(wordpiece).err().unwrap();
        assert!(
            error.0.contains("not BPE") || error.0.contains("not recognized"),
            "{error}"
        );
    }

    #[test]
    fn added_token_matching_is_leftmost_longest_and_mode_aware() {
        let mut set = AddedSet::default();
        set.push(AddedToken {
            bytes: b"<|end|>"[..].into(),
            id: 1,
            special: true,
        });
        set.push(AddedToken {
            bytes: b"<|end|><|end|>"[..].into(),
            id: 2,
            special: false,
        });
        let text = b"x<|end|><|end|>y<|end|>";
        let (start, token) = set.next_match(text, 0, text.len()).unwrap();
        assert_eq!((start, token.id), (1, 2));
        let (start, token) = set.next_match(text, 15, text.len()).unwrap();
        assert_eq!((start, token.id), (16, 1));
        assert!(set.next_match(text, 17, text.len()).is_none());
    }
}
