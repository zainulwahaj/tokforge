use std::{collections::HashSet, fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bytelevel::{self, ByteLevelBpe};
use tokenizers::{
    models::{
        bpe::BPE, unigram::Unigram, wordlevel::WordLevel, wordpiece::WordPiece, ModelWrapper,
    },
    Model, Tokenizer,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read tokenizer: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("BPE dropout must be disabled for reproducible counts and token budgets")]
    Dropout,
    #[error("unknown token ID: {0}")]
    UnknownId(u32),
    #[error("the custom core and the backend disagreed on this input ({0}); run with `--core backend` and report the artifact")]
    CoreMismatch(&'static str),
    #[error(
        "special token {0:?} cannot be made unreachable for literal text (it is a single scalar, or a scalar of it has no piece of its own); use `--special-tokens recognize`"
    )]
    UnsuppressibleSpecialToken(String),
    #[error("no nonempty UTF-8 prefix fits the token budget at source byte {offset}")]
    BudgetTooSmall { offset: usize },
    #[error("the empty input's special tokens exceed the token budget")]
    EmptyBudget,
    #[error("operation cancelled")]
    Cancelled,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialTokens {
    /// Treat special-token spellings in input as ordinary text.
    #[default]
    Literal,
    /// Recognize the artifact's registered special tokens in input.
    Recognize,
}

/// Which tokenization core serves requests for an artifact the custom core supports.
///
/// `Auto` is the default: the differential soak (`examples/core_soak.rs`; PERFORMANCE.md,
/// "Differential soak") passed clean on GPT-2 and Qwen2.5 on 2026-10-18, which was the
/// user's condition for trusting the custom core without a backend re-encode per request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMode {
    /// The custom byte-level BPE core when the artifact qualifies, otherwise the backend.
    #[default]
    Auto,
    /// Always the Hugging Face backend.
    Backend,
    /// Run both on every request and fail the request if they disagree (a differential check
    /// against the backend at production time; costs about one backend encode per request).
    Verify,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct EncodeOptions {
    pub add_special_tokens: bool,
    pub special_tokens: SpecialTokens,
    pub core: CoreMode,
}

#[derive(Debug, Serialize)]
pub struct Artifact {
    /// Hash of the original tokenizer.json bytes, before runtime overrides.
    pub sha256: String,
    pub vocab_size: usize,
    /// Registered special-token spellings that the model itself could have
    /// emitted as one piece for ordinary text and that Literal mode made
    /// unreachable at load time. Recognize mode changes nothing and reports 0.
    /// IDs and decoding are untouched; the hash still names the original bytes.
    pub suppressed_special_pieces: usize,
    /// The core serving encode and count for this artifact: `backend` (the Hugging Face
    /// tokenizers crate) or `bytelevel (...)`, the custom core whose every output is
    /// bit-identical to the backend on the inputs it accepts; it defers to the backend
    /// per request for inputs it declines.
    pub core: String,
}

#[derive(Debug, Serialize)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
    /// UTF-8 byte ranges in the input; synthetic tokens may have (0, 0).
    /// Byte-level tokens may cover only part of a Unicode scalar.
    pub offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
    pub count: usize,
}

/// A verified source slice: inclusive start byte, exclusive end byte, and the
/// token count of that slice encoded on its own. It borrows nothing, so callers
/// decide whether to copy the text or serialize a slice of the original.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Span {
    pub start_byte: usize,
    pub end_byte: usize,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct Chunk {
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub count: usize,
}

impl Chunk {
    fn from_span(text: &str, span: Span) -> Self {
        Self {
            text: text[span.start_byte..span.end_byte].to_owned(),
            start_byte: span.start_byte,
            end_byte: span.end_byte,
            count: span.count,
        }
    }
}

/// What the source text looks like at a cut between two tokens, weakest first.
/// A stronger boundary is also every weaker one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    /// Inside a word: a subword or byte boundary.
    Token,
    /// Between pre-tokenizer words, or next to whitespace.
    Word,
    /// After a sentence terminator followed by whitespace.
    Sentence,
    /// After a line break.
    Line,
    /// After a blank line, or at the end of the text.
    Paragraph,
}

const LEVELS: usize = 5;

/// Accept a weaker boundary when the strongest available one would leave the
/// budget less than this full. Whole paragraphs are preferred, but not at the
/// cost of emitting a tiny chunk in front of a long paragraph.
const MIN_FILL_PERCENT: usize = 50;
/// First guide window, in bytes per budgeted token, before any density has been
/// measured. It only sizes the first encode; cuts never depend on it.
const INITIAL_WINDOW_BYTES_PER_TOKEN: usize = 6;
/// Guide windows cover half again the predicted slice so the limit falls inside.
const WINDOW_SLACK: (usize, usize) = (3, 2);
const MIN_WINDOW_BYTES: usize = 64;
/// Scalar boundaries the exhaustive fallback tries after the guided cuts failed, nearest the
/// slice start. A no-fit situation is resolved by a short prefix when it is resolvable at all
/// (tiny budgets), and the unbounded scan was quadratic in the remaining document.
const EXHAUSTIVE_SCALAR_LIMIT: usize = 1024;

/// A candidate end for a slice. `tokens` is how many content tokens of the guide
/// encoding end at or before `end`; the budget check compares against it.
#[derive(Clone, Copy, Debug)]
struct Cut {
    end: usize,
    tokens: usize,
    level: Level,
}

/// Bitset of IDs the artifact can decode. One bit per ID replaces a per-ID
/// string lookup, so validating a large ID array allocates nothing.
struct IdSet(Vec<u64>);

impl IdSet {
    fn contains(&self, id: u32) -> bool {
        self.0
            .get((id / 64) as usize)
            .is_some_and(|word| (word >> (id % 64)) & 1 == 1)
    }
}

/// Name reported in `Artifact::core` when the backend serves every request.
pub const BACKEND_CORE: &str = "backend";

/// Text encoded by both cores at load time. The core is enabled only if the two agree here; it
/// is not a proof (the differential suite is), but it catches a backend upgrade that changed
/// behaviour before any request is served.
const SELF_CHECK: &str = "Hello world! It's 2026-10-18T19:04:00Z.  Tabs\tand\r\nCRLF, caf\u{e9} na\u{ef}ve \u{65e5}\u{672c}\u{8a9e} \u{d55c}\u{ad6d}\u{c5b4} \u{645}\u{631}\u{62d}\u{628}\u{627} \u{1f600}\u{1f44d}\u{1f3fd} <|endoftext|> <|im_start|>x<|im_end|> 1234567 don't 'tis   trailing   ";

/// One encoding from whichever core produced it. Accessors hide the difference so the chunk
/// guide and the serializers work on either without copying.
pub enum Encoding<'e> {
    Backend(tokenizers::Encoding),
    Core {
        output: bytelevel::Output,
        core: &'e ByteLevelBpe,
    },
}

impl Encoding<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::Backend(encoding) => encoding.len(),
            Self::Core { output, .. } => output.ids.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ids(&self) -> &[u32] {
        match self {
            Self::Backend(encoding) => encoding.get_ids(),
            Self::Core { output, .. } => &output.ids,
        }
    }

    pub fn offsets(&self) -> &[(usize, usize)] {
        match self {
            Self::Backend(encoding) => encoding.get_offsets(),
            Self::Core { output, .. } => &output.offsets,
        }
    }

    pub fn special_tokens_mask(&self) -> &[u32] {
        match self {
            Self::Backend(encoding) => encoding.get_special_tokens_mask(),
            Self::Core { output, .. } => &output.special_mask,
        }
    }

    pub fn word_id(&self, index: usize) -> Option<u32> {
        match self {
            Self::Backend(encoding) => encoding.get_word_ids().get(index).copied().flatten(),
            Self::Core { output, .. } => output.words.get(index).copied(),
        }
    }

    /// Token strings in the backend's spelling, borrowed: from the encoding for the backend,
    /// from the core's vocabulary table for the core (no per-token allocation either way).
    pub fn tokens(&self) -> impl Iterator<Item = &str> + '_ {
        let (backend, core): (
            Option<&tokenizers::Encoding>,
            Option<(&bytelevel::Output, &ByteLevelBpe)>,
        ) = match self {
            Self::Backend(encoding) => (Some(encoding), None),
            Self::Core { output, core } => (None, Some((output, core))),
        };
        backend
            .into_iter()
            .flat_map(|encoding| encoding.get_tokens().iter().map(String::as_str))
            .chain(core.into_iter().flat_map(|(output, core)| {
                output
                    .ids
                    .iter()
                    .map(move |&id| core.token(id).unwrap_or(""))
            }))
    }
}

pub struct Engine {
    tokenizer: Tokenizer,
    artifact: Artifact,
    options: EncodeOptions,
    ids: IdSet,
    core: Option<ByteLevelBpe>,
}

impl Engine {
    pub fn from_file(path: impl AsRef<Path>, options: EncodeOptions) -> Result<Self, Error> {
        Self::from_bytes(&fs::read(path)?, options)
    }

    pub fn from_bytes(bytes: &[u8], options: EncodeOptions) -> Result<Self, Error> {
        let mut tokenizer = Tokenizer::from_bytes(bytes).map_err(tokenizer_error)?;
        if let tokenizers::models::ModelWrapper::BPE(model) = tokenizer.get_model() {
            if model.dropout.is_some_and(|dropout| dropout > 0.0) {
                return Err(Error::Dropout);
            }
        }
        // Counts must cover the entire input, independently of training settings.
        tokenizer.with_truncation(None).map_err(tokenizer_error)?;
        tokenizer.with_padding(None);
        let literal = matches!(options.special_tokens, SpecialTokens::Literal);
        tokenizer.set_encode_special_tokens(literal);
        // The vocabulary map (model plus added tokens) is built once, read for
        // its size and its ID bits, and dropped. It describes the artifact as
        // loaded: the Literal-mode rebuild below never changes an ID.
        let vocab = tokenizer.get_vocab(true);
        let bits = vocab.values().map(|&id| id as usize + 1).max().unwrap_or(0);
        let mut words = vec![0u64; bits.div_ceil(64)];
        for &id in vocab.values() {
            words[id as usize / 64] |= 1 << (id % 64);
        }
        let suppressed_special_pieces = if literal {
            suppress_special_pieces(&mut tokenizer, bytes)?
        } else {
            0
        };
        let core = select_core(
            bytes,
            &tokenizer,
            options,
            suppressed_special_pieces,
            vocab.len(),
        );
        let artifact = Artifact {
            sha256: format!("{:x}", Sha256::digest(bytes)),
            vocab_size: vocab.len(),
            suppressed_special_pieces,
            core: core.as_ref().map_or_else(
                || BACKEND_CORE.to_owned(),
                |core| format!("bytelevel ({})", core.describe()),
            ),
        };
        Ok(Self {
            tokenizer,
            artifact,
            options,
            ids: IdSet(words),
            core,
        })
    }

    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    pub fn options(&self) -> EncodeOptions {
        self.options
    }

    pub fn encode(&self, text: &str) -> Result<Encoded, Error> {
        let encoding = self.encode_raw(text)?;
        Ok(Encoded {
            ids: encoding.ids().to_vec(),
            tokens: encoding.tokens().map(str::to_owned).collect(),
            offsets: encoding.offsets().to_vec(),
            special_tokens_mask: encoding.special_tokens_mask().to_vec(),
            count: encoding.len(),
        })
    }

    /// Whether added special spellings in text are matched (Recognize) or left literal.
    fn recognize(&self) -> bool {
        matches!(self.options.special_tokens, SpecialTokens::Recognize)
    }

    fn backend_encode(&self, text: &str) -> Result<tokenizers::Encoding, Error> {
        self.tokenizer
            .encode(text, self.options.add_special_tokens)
            .map_err(tokenizer_error)
    }

    fn backend_count(&self, text: &str) -> Result<usize, Error> {
        self.tokenizer
            .encode_fast(text, self.options.add_special_tokens)
            .map(|encoding| encoding.len())
            .map_err(tokenizer_error)
    }

    /// Encode with the active core. The custom core writes into fresh buffers and declines
    /// inputs it cannot reproduce exactly, which fall through to the backend.
    pub(crate) fn encode_raw(&self, text: &str) -> Result<Encoding<'_>, Error> {
        if let Some(core) = &self.core {
            let mut output = bytelevel::Output::default();
            if core.encode(text, self.recognize(), &mut output).is_ok() {
                if self.options.core == CoreMode::Verify {
                    let backend = self.backend_encode(text)?;
                    let same = output.ids == backend.get_ids()
                        && output.offsets == backend.get_offsets()
                        && output.special_mask == backend.get_special_tokens_mask();
                    if !same {
                        return Err(Error::CoreMismatch("encode"));
                    }
                }
                return Ok(Encoding::Core { output, core });
            }
        }
        self.backend_encode(text).map(Encoding::Backend)
    }

    /// Token ids of `text`, the same as `encode(text)?.ids` without building offsets, masks
    /// or token strings.
    pub fn encode_ids(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.encode_raw(text)
            .map(|encoding| encoding.ids().to_vec())
    }

    pub fn count(&self, text: &str) -> Result<usize, Error> {
        if let Some(core) = &self.core {
            if let Ok(count) = core.count(text, self.recognize()) {
                if self.options.core == CoreMode::Verify && self.backend_count(text)? != count {
                    return Err(Error::CoreMismatch("count"));
                }
                return Ok(count);
            }
        }
        self.backend_count(text)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        if let Some(&id) = ids.iter().find(|&&id| !self.ids.contains(id)) {
            return Err(Error::UnknownId(id));
        }
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(tokenizer_error)
    }

    /// Return a verified source prefix, cut at a word boundary when one fills at
    /// least half the budget. It need not be the longest fitting prefix: token
    /// counts are not monotonic as text grows.
    pub fn truncate(&self, text: &str, max_tokens: usize) -> Result<Chunk, Error> {
        let span = self.truncate_span_checked(text, max_tokens, &|| Ok(()))?;
        Ok(Chunk::from_span(text, span))
    }

    pub(crate) fn truncate_span_checked(
        &self,
        text: &str,
        max_tokens: usize,
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<Span, Error> {
        check()?;
        let count = self.count(text)?;
        if count <= max_tokens {
            return Ok(Span {
                start_byte: 0,
                end_byte: text.len(),
                count,
            });
        }
        let overhead = self.overhead(max_tokens)?;
        // The whole text was just counted, so its density predicts the cut.
        let content = count.saturating_sub(overhead).max(1);
        let predicted = text.len().saturating_mul(max_tokens - overhead) / content;
        let (end, count) =
            self.fit(text, 0, max_tokens, overhead, predicted, Level::Word, check)?;
        Ok(Span {
            start_byte: 0,
            end_byte: end,
            count,
        })
    }

    /// Partition all source bytes into nonoverlapping, independently encoded
    /// chunks. Cuts prefer paragraph, line, sentence, then word boundaries, and
    /// fall back to subword or scalar boundaries only when nothing else fits.
    pub fn chunk(&self, text: &str, max_tokens: usize) -> Result<Vec<Chunk>, Error> {
        let spans = self.chunk_spans_checked(text, max_tokens, &|| Ok(()))?;
        Ok(spans
            .into_iter()
            .map(|span| Chunk::from_span(text, span))
            .collect())
    }

    pub(crate) fn chunk_spans_checked(
        &self,
        text: &str,
        max_tokens: usize,
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<Vec<Span>, Error> {
        check()?;
        let mut spans = Vec::new();
        if text.is_empty() {
            return Ok(spans);
        }
        let overhead = self.overhead(max_tokens)?;
        let budget = max_tokens - overhead;
        let mut predicted = budget.saturating_mul(INITIAL_WINDOW_BYTES_PER_TOKEN);
        let mut start = 0;
        while start < text.len() {
            check()?;
            let (end, count) = self.fit(
                text,
                start,
                max_tokens,
                overhead,
                predicted,
                Level::Paragraph,
                check,
            )?;
            if end == start {
                return Err(Error::BudgetTooSmall { offset: start });
            }
            spans.push(Span {
                start_byte: start,
                end_byte: end,
                count,
            });
            // The verified chunk's own density predicts the next window.
            predicted =
                (end - start).saturating_mul(budget) / count.saturating_sub(overhead).max(1);
            start = end;
        }
        Ok(spans)
    }

    /// Tokens the post-processor adds to every slice, which count toward budgets.
    fn overhead(&self, max_tokens: usize) -> Result<usize, Error> {
        let overhead = self.count("")?;
        if overhead > max_tokens {
            return Err(Error::EmptyBudget);
        }
        Ok(overhead)
    }

    /// Find a verified end for the slice starting at `start`. One guide encoding
    /// of a bounded window predicts where the budget runs out; every returned
    /// slice is still re-encoded on its own before it is accepted.
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        text: &str,
        start: usize,
        max_tokens: usize,
        overhead: usize,
        predicted_bytes: usize,
        max_level: Level,
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<(usize, usize), Error> {
        let budget = max_tokens - overhead;
        let remaining = text.len() - start;
        let mut window = (predicted_bytes.saturating_mul(WINDOW_SLACK.0) / WINDOW_SLACK.1)
            .max(MIN_WINDOW_BYTES)
            .min(remaining);
        loop {
            check()?;
            let end = char_boundary_at_or_after(text, start + window);
            let cuts = self.guide(text, start, end)?;
            let fitting = cuts.partition_point(|cut| cut.tokens <= budget);
            if fitting == cuts.len() && end < text.len() {
                // Everything in the window fits, so the limit lies beyond it.
                window = window.saturating_mul(2).min(remaining);
                continue;
            }
            if let Some(found) = self.verify(
                text,
                start,
                max_tokens,
                budget,
                &cuts[..fitting],
                max_level,
                check,
            )? {
                return Ok(found);
            }
            return self.exhaustive(text, start, max_tokens, &cuts, check);
        }
    }

    /// Encode `text[start..end]` once and list every cut between its tokens.
    /// Only the token offsets, word indices, and special mask are read; the
    /// encoding itself is dropped before this returns.
    fn guide(&self, text: &str, start: usize, end: usize) -> Result<Vec<Cut>, Error> {
        let encoding = self.encode_raw(&text[start..end])?;
        let offsets = encoding.offsets();
        let special = encoding.special_tokens_mask();
        let len = end - start;
        let mut cuts = Vec::with_capacity(offsets.len() + 1);
        let mut tokens = 0;
        let mut reach = 0;
        let mut previous: Option<Option<u32>> = None;
        for (index, &(token_start, token_end)) in offsets.iter().enumerate() {
            if special.get(index) == Some(&1) {
                // Post-processor tokens have no source position; they are overhead.
                continue;
            }
            let word = encoding.word_id(index);
            if let Some(previous) = previous {
                // The byte-level pre-tokenizer trims spaces from offsets, so a
                // boundary may be a gap. Both ends of the gap are candidates.
                let gap = token_start.min(len) > reach;
                push_cut(
                    text,
                    start,
                    &mut cuts,
                    reach,
                    tokens,
                    gap || previous != word,
                );
                if gap {
                    push_cut(text, start, &mut cuts, token_start.min(len), tokens, true);
                }
            }
            reach = reach.max(token_end.min(len));
            tokens += 1;
            previous = Some(word);
        }
        push_cut(text, start, &mut cuts, reach, tokens, true);
        push_cut(text, start, &mut cuts, len, tokens, true);
        Ok(cuts)
    }

    /// Verify the preferred cut; on overflow, retry with cuts predicted shorter
    /// by at least the measured overshoot. Each retry strictly shrinks the set.
    #[allow(clippy::too_many_arguments)]
    fn verify(
        &self,
        text: &str,
        start: usize,
        max_tokens: usize,
        budget: usize,
        mut cuts: &[Cut],
        max_level: Level,
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<Option<(usize, usize)>, Error> {
        while !cuts.is_empty() {
            check()?;
            let chosen = choose(cuts, budget, max_level, text.len());
            let cut = cuts[chosen];
            let count = self.count(&text[start..cut.end])?;
            if count <= max_tokens {
                return Ok(Some((cut.end, count)));
            }
            let limit = cut.tokens.saturating_sub(count - max_tokens);
            let keep = cuts[..chosen].partition_point(|cut| cut.tokens <= limit);
            cuts = &cuts[..keep];
        }
        Ok(None)
    }

    /// Rare fallback. Counts need not be monotonic, so a longer slice may fit
    /// although every guided cut failed. Try the whole remainder, then every
    /// token boundary, then the nearest `EXHAUSTIVE_SCALAR_LIMIT` scalar
    /// boundaries, longest first. Work is bounded by the guide window plus that
    /// limit rather than by the document; a prefix that would only fit beyond
    /// the limit is reported as no fit. Cancellation is checked between calls.
    fn exhaustive(
        &self,
        text: &str,
        start: usize,
        max_tokens: usize,
        cuts: &[Cut],
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<(usize, usize), Error> {
        check()?;
        let rest = &text[start..];
        let count = self.count(rest)?;
        if count <= max_tokens {
            return Ok((text.len(), count));
        }
        for cut in cuts.iter().rev().filter(|cut| cut.end < text.len()) {
            check()?;
            let count = self.count(&text[start..cut.end])?;
            if count <= max_tokens {
                return Ok((cut.end, count));
            }
        }
        let ends: Vec<usize> = rest
            .char_indices()
            .skip(1)
            .take(EXHAUSTIVE_SCALAR_LIMIT)
            .map(|(offset, _)| start + offset)
            .collect();
        for end in ends.into_iter().rev() {
            if cuts.binary_search_by_key(&end, |cut| cut.end).is_ok() {
                continue;
            }
            check()?;
            let count = self.count(&text[start..end])?;
            if count <= max_tokens {
                return Ok((end, count));
            }
        }
        let empty = self.count("")?;
        if empty > max_tokens {
            return Err(Error::EmptyBudget);
        }
        Ok((start, empty))
    }
}

/// Record a cut at `start + relative`, keeping `cuts` sorted and deduplicated.
fn push_cut(
    text: &str,
    start: usize,
    cuts: &mut Vec<Cut>,
    relative: usize,
    tokens: usize,
    word_boundary: bool,
) {
    let end = start + relative;
    if relative == 0 || end > text.len() || !text.is_char_boundary(end) {
        return;
    }
    if let Some(last) = cuts.last_mut() {
        if last.end > end {
            return;
        }
        if last.end == end {
            // Byte-level tokens inside one scalar share offsets; the later count
            // includes all of them.
            last.tokens = last.tokens.max(tokens);
            if word_boundary {
                last.level = last.level.max(classify(text, end, true));
            }
            return;
        }
    }
    cuts.push(Cut {
        end,
        tokens,
        level: classify(text, end, word_boundary),
    });
}

/// Choose among cuts that are predicted to fit: the strongest boundary level at
/// or below `max_level` that fills at least half the budget, otherwise the
/// fullest cut. Equal fills prefer the earlier byte, so separating whitespace
/// leads the following slice rather than trailing this one.
fn choose(cuts: &[Cut], budget: usize, max_level: Level, text_len: usize) -> usize {
    let mut best = [None::<usize>; LEVELS];
    for (index, cut) in cuts.iter().enumerate().rev() {
        // A cut at level N is a candidate for every level up to N.
        for slot in best.iter_mut().take(cut.level as usize + 1) {
            match *slot {
                Some(current) if cuts[current].tokens != cut.tokens => {}
                Some(current) if cuts[current].end <= cut.end => {}
                _ => *slot = Some(index),
            }
        }
    }
    let minimum = (budget * MIN_FILL_PERCENT).div_ceil(100);
    for level in (Level::Word as usize..=max_level as usize).rev() {
        if let Some(index) = best[level] {
            if cuts[index].tokens >= minimum || cuts[index].end == text_len {
                return index;
            }
        }
    }
    best[Level::Token as usize].expect("every cut registers at the token level")
}

fn classify(text: &str, end: usize, word_boundary: bool) -> Level {
    if end >= text.len() {
        return Level::Paragraph;
    }
    let (before, after) = text.split_at(end);
    // Never separate a carriage return from its line feed.
    if before.ends_with('\r') && after.starts_with('\n') {
        return Level::Token;
    }
    let trimmed = before.trim_end();
    let whitespace = &before[trimmed.len()..];
    match line_breaks(whitespace) {
        0 => {}
        1 => return Level::Line,
        _ => return Level::Paragraph,
    }
    let spaced = !whitespace.is_empty() || after.starts_with(char::is_whitespace);
    if spaced && ends_sentence(trimmed) {
        return Level::Sentence;
    }
    if spaced || word_boundary {
        Level::Word
    } else {
        Level::Token
    }
}

/// Count line breaks, treating CRLF as one.
fn line_breaks(whitespace: &str) -> usize {
    let mut breaks = 0;
    let mut chars = whitespace.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\n' => breaks += 1,
            '\r' => {
                breaks += 1;
                chars.next_if_eq(&'\n');
            }
            _ => {}
        }
    }
    breaks
}

fn ends_sentence(trimmed: &str) -> bool {
    let body = trimmed.trim_end_matches(|c: char| {
        matches!(
            c,
            '"' | '\''
                | ')'
                | ']'
                | '}'
                | '\u{201D}'
                | '\u{2019}'
                | '\u{00BB}'
                | '\u{300D}'
                | '\u{300F}'
        )
    });
    body.ends_with(|c: char| {
        matches!(
            c,
            '.' | '!'
                | '?'
                | '\u{3002}'
                | '\u{FF01}'
                | '\u{FF1F}'
                | '\u{2026}'
                | '\u{061F}'
                | '\u{0964}'
        )
    })
}

fn char_boundary_at_or_after(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

/// Make every registered special-token spelling unreachable for the model
/// itself, so Literal text never produces a special ID. Disabling added-token
/// matching (`set_encode_special_tokens`) closes one path; the other is the
/// model's own vocabulary, which may list the spelling as an ordinary piece
/// (converted sentencepiece models keep `<s>` as a score-0 Unigram piece). The
/// rebuilt model keeps every ID, so decoding and identity are unchanged, and
/// the work happens once per load, never per request. Returns the number of
/// spellings suppressed.
fn suppress_special_pieces(tokenizer: &mut Tokenizer, bytes: &[u8]) -> Result<usize, Error> {
    let model = tokenizer.get_model();
    // Only spellings the model knows as a piece can be emitted by it. Owned
    // copies (a handful of short strings) because the tokenizer is mutated
    // below while these are still in use; sorted so errors are deterministic.
    let mut spellings: Vec<String> = tokenizer
        .get_added_vocabulary()
        .get_added_tokens_decoder()
        .values()
        .filter(|token| token.special && model.token_to_id(&token.content).is_some())
        .map(|token| token.content.clone())
        .collect();
    spellings.sort_unstable();
    spellings.dedup();
    if spellings.is_empty() {
        return Ok(0);
    }
    let rebuilt: Option<(ModelWrapper, usize)> = match model {
        ModelWrapper::Unigram(model) => {
            suppress_unigram(model, &spellings, bytes)?.map(|(m, n)| (m.into(), n))
        }
        ModelWrapper::WordPiece(model) => {
            suppress_wordpiece(model, &spellings)?.map(|(m, n)| (m.into(), n))
        }
        ModelWrapper::WordLevel(model) => {
            suppress_wordlevel(model, &spellings)?.map(|(m, n)| (m.into(), n))
        }
        ModelWrapper::BPE(model) => {
            suppress_bpe(model, &spellings, bytes)?.map(|(m, n)| (m.into(), n))
        }
    };
    match rebuilt {
        Some((model, count)) => {
            // The tokenizer owns its model: this drops the original trie/maps
            // and installs the rebuilt one. Both exist briefly during the load.
            tokenizer.with_model(model);
            Ok(count)
        }
        None => Ok(0),
    }
}

/// Only the `model` section is read again, and only the named fields; serde
/// skips the rest of the document (the whole vocabulary) without allocating.
#[derive(Deserialize)]
struct Document<M> {
    model: M,
}

#[derive(Deserialize)]
struct UnigramFields {
    #[serde(default)]
    unk_id: Option<usize>,
}

#[derive(Deserialize)]
struct BpeFields {
    #[serde(default)]
    merges: MergeList,
}

/// Both merge encodings the backend accepts: `[a, b]` pairs or legacy `"a b"`.
#[derive(Deserialize)]
#[serde(untagged)]
enum MergeList {
    Pairs(Vec<(String, String)>),
    Legacy(Vec<String>),
}

impl Default for MergeList {
    fn default() -> Self {
        Self::Pairs(Vec::new())
    }
}

fn model_section<M: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<M, Error> {
    serde_json::from_slice::<Document<M>>(bytes)
        .map(|document| document.model)
        .map_err(|error| Error::Tokenizer(format!("could not re-read the model section: {error}")))
}

fn single_scalar(spelling: &str) -> bool {
    spelling.chars().nth(1).is_none()
}

/// Unigram: the Viterbi search keeps the best-scoring path, so a piece scored
/// `-inf` is never chosen while a finite path exists. One exists for every
/// suppressed spelling because each of its scalars is required to be a piece
/// of its own (checked below). The vocabulary keeps its order, so IDs are
/// stable. `Unigram::from` derives the unknown penalty from the lowest score;
/// it is restored afterwards so unknown handling is byte-for-byte identical.
fn suppress_unigram(
    model: &Unigram,
    spellings: &[String],
    bytes: &[u8],
) -> Result<Option<(Unigram, usize)>, Error> {
    for spelling in spellings {
        // A single-scalar piece disables the unknown fallback at its position,
        // so at -inf it could still be the only path. A scalar without a piece
        // of its own is covered by the unknown fallback, and upstream maps a
        // run of unknown scalars back to an ID by its text, which could spell
        // the special token. Refuse both rather than guess.
        let mut buffer = [0u8; 4];
        let every_scalar_is_a_piece = spelling
            .chars()
            .all(|c| model.token_to_id(c.encode_utf8(&mut buffer)).is_some());
        if single_scalar(spelling) || !every_scalar_is_a_piece {
            return Err(Error::UnsuppressibleSpecialToken(spelling.clone()));
        }
    }
    // `unk_id` is private upstream; it is the one field read back from the bytes.
    let unk_id = model_section::<UnigramFields>(bytes)?.unk_id;
    let suppressed: HashSet<&str> = spellings.iter().map(String::as_str).collect();
    // One owned copy of the vocabulary, moved into the new model (no second copy).
    let vocab: Vec<(String, f64)> = model
        .iter()
        .map(|(piece, score)| {
            let score = if suppressed.contains(piece.as_str()) {
                f64::NEG_INFINITY
            } else {
                *score
            };
            (piece.clone(), score)
        })
        .collect();
    let mut rebuilt =
        Unigram::from(vocab, unk_id, model.byte_fallback()).map_err(tokenizer_error)?;
    rebuilt.min_score = model.min_score;
    Ok(Some((rebuilt, spellings.len())))
}

/// WordPiece looks subwords up by string, so the spelling leaves the lookup
/// table; its ID still decodes through the added vocabulary. The unknown
/// token's own spelling stays: the model finds the unknown ID through it.
fn suppress_wordpiece(
    model: &WordPiece,
    spellings: &[String],
) -> Result<Option<(WordPiece, usize)>, Error> {
    let remove = removable(spellings, &model.unk_token, true)?;
    if remove.is_empty() {
        return Ok(None);
    }
    let vocab = retain_vocab(model, &remove);
    let rebuilt = WordPiece::builder()
        .vocab(vocab)
        .unk_token(model.unk_token.clone())
        .continuing_subword_prefix(model.continuing_subword_prefix.clone())
        .max_input_chars_per_word(model.max_input_chars_per_word)
        .build()
        .map_err(tokenizer_error)?;
    Ok(Some((rebuilt, remove.len())))
}

/// WordLevel looks whole words up, so a removed spelling becomes the unknown
/// token, exactly like any other word the model does not know.
fn suppress_wordlevel(
    model: &WordLevel,
    spellings: &[String],
) -> Result<Option<(WordLevel, usize)>, Error> {
    let remove = removable(spellings, &model.unk_token, false)?;
    if remove.is_empty() {
        return Ok(None);
    }
    let vocab = retain_vocab(model, &remove);
    let rebuilt = WordLevel::builder()
        .vocab(vocab)
        .unk_token(model.unk_token.clone())
        .build()
        .map_err(tokenizer_error)?;
    Ok(Some((rebuilt, remove.len())))
}

/// Spellings whose vocabulary entry can go. Removing a single-scalar entry from
/// a subword model would turn every word starting with that scalar unknown, so
/// that is refused where subwords are looked up.
fn removable<'a>(
    spellings: &'a [String],
    unk_token: &str,
    subwords: bool,
) -> Result<HashSet<&'a str>, Error> {
    let mut remove = HashSet::new();
    for spelling in spellings {
        if spelling == unk_token {
            continue;
        }
        if subwords && single_scalar(spelling) {
            return Err(Error::UnsuppressibleSpecialToken(spelling.clone()));
        }
        remove.insert(spelling.as_str());
    }
    Ok(remove)
}

/// The model's token map without `remove`. The backend exposes its shared map
/// type only through the `bpe` module; every builder accepts it.
fn retain_vocab(model: &impl Model, remove: &HashSet<&str>) -> tokenizers::models::bpe::Vocab {
    model
        .get_vocab()
        .into_iter()
        .filter(|(token, _)| !remove.contains(token.as_str()))
        .collect()
}

/// BPE forms a multi-scalar token only through a merge whose result is that
/// token, or, with `ignore_merges`, by looking the whole word up. Those merges
/// are dropped and, for `ignore_merges`, the vocabulary entry too; ranks keep
/// their relative order. The unknown token's entry stays because unknown
/// scalars are mapped through it. Byte-level artifacts such as GPT-2 list
/// `<|endoftext|>` without any merge producing it, so nothing is rebuilt there.
fn suppress_bpe(
    model: &BPE,
    spellings: &[String],
    bytes: &[u8],
) -> Result<Option<(BPE, usize)>, Error> {
    let prefix = model.continuing_subword_prefix.as_deref().unwrap_or("");
    let mut remove_entries: HashSet<&str> = HashSet::new();
    let mut candidates: HashSet<&str> = HashSet::new();
    for spelling in spellings {
        if single_scalar(spelling) {
            // Scalars are looked up directly; the entry cannot go.
            return Err(Error::UnsuppressibleSpecialToken(spelling.clone()));
        }
        if model.ignore_merges && model.unk_token.as_deref() != Some(spelling) {
            remove_entries.insert(spelling);
        }
        // A merge can only produce the spelling if both halves are entries
        // (the second with the continuing prefix). This cheap check avoids
        // re-reading the merge list for artifacts whose special tokens were
        // appended to the vocabulary without merges.
        let mergeable = spelling.char_indices().skip(1).any(|(split, _)| {
            model.token_to_id(&spelling[..split]).is_some()
                && model
                    .token_to_id(&format!("{prefix}{}", &spelling[split..]))
                    .is_some()
        });
        if mergeable {
            candidates.insert(spelling);
        }
    }
    if remove_entries.is_empty() && candidates.is_empty() {
        return Ok(None);
    }
    let merges = match model_section::<BpeFields>(bytes)?.merges {
        MergeList::Pairs(pairs) => pairs,
        MergeList::Legacy(lines) => lines
            .into_iter()
            .map(|line| {
                line.split_once(' ')
                    .map(|(a, b)| (a.to_owned(), b.to_owned()))
                    .ok_or_else(|| Error::Tokenizer(format!("malformed merge {line:?}")))
            })
            .collect::<Result<Vec<_>, Error>>()?,
    };
    let produces = |a: &str, b: &str, spelling: &str| {
        let tail = b.get(prefix.len()..).unwrap_or_default();
        a.len() + tail.len() == spelling.len()
            && spelling.starts_with(a)
            && spelling.ends_with(tail)
    };
    let mut suppressed: HashSet<&str> = remove_entries.clone();
    let merges: Vec<(String, String)> = merges
        .into_iter()
        .filter(|(a, b)| {
            if let Some(&spelling) = candidates.iter().find(|s| produces(a, b, s)) {
                suppressed.insert(spelling);
                return false;
            }
            // Merges using a removed entry as a part could never fire again
            // and the builder requires both parts to exist.
            !(remove_entries.contains(a.as_str()) || remove_entries.contains(b.as_str()))
        })
        .collect();
    if suppressed.is_empty() {
        return Ok(None);
    }
    let vocab = retain_vocab(model, &remove_entries);
    let mut builder = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .fuse_unk(model.fuse_unk)
        .byte_fallback(model.byte_fallback)
        .ignore_merges(model.ignore_merges);
    if let Some(dropout) = model.dropout {
        builder = builder.dropout(dropout);
    }
    if let Some(unk) = &model.unk_token {
        builder = builder.unk_token(unk.clone());
    }
    if let Some(prefix) = &model.continuing_subword_prefix {
        builder = builder.continuing_subword_prefix(prefix.clone());
    }
    if let Some(suffix) = &model.end_of_word_suffix {
        builder = builder.end_of_word_suffix(suffix.clone());
    }
    let rebuilt = builder.build().map_err(tokenizer_error)?;
    Ok(Some((rebuilt, suppressed.len())))
}

fn tokenizer_error(error: tokenizers::Error) -> Error {
    Error::Tokenizer(error.to_string())
}

/// Decide whether the custom core may serve this artifact: the mode allows it, the core
/// understands every component, Literal mode rebuilt nothing (the core reads the original
/// merges), the vocabularies agree in size, and the self-check text encodes identically.
fn select_core(
    bytes: &[u8],
    tokenizer: &Tokenizer,
    options: EncodeOptions,
    suppressed_special_pieces: usize,
    vocab_size: usize,
) -> Option<ByteLevelBpe> {
    if options.core == CoreMode::Backend || suppressed_special_pieces > 0 {
        return None;
    }
    let core = ByteLevelBpe::from_json(bytes).ok()?;
    if core.vocab_size() != vocab_size {
        return None;
    }
    let recognize = matches!(options.special_tokens, SpecialTokens::Recognize);
    let mut output = bytelevel::Output::default();
    if core.encode(SELF_CHECK, recognize, &mut output).is_err() {
        return None;
    }
    let backend = tokenizer
        .encode(SELF_CHECK, options.add_special_tokens)
        .ok()?;
    let same = output.ids == backend.get_ids()
        && output.offsets == backend.get_offsets()
        && output.special_mask == backend.get_special_tokens_mask()
        && core.count(SELF_CHECK, recognize).ok() == Some(backend.len());
    same.then_some(core)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn prefix_search_and_chunking_honor_cooperative_cancellation() {
        let engine = Engine::from_bytes(
            include_bytes!("../tests/fixtures/wordpiece.json"),
            EncodeOptions::default(),
        )
        .unwrap();
        let text = "hello world! ".repeat(100);
        for chunk in [false, true] {
            let checks = AtomicUsize::new(0);
            let check = || {
                if checks.fetch_add(1, Ordering::Relaxed) >= 2 {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            };
            let result = if chunk {
                engine.chunk_spans_checked(&text, 0, &check).map(|_| ())
            } else {
                engine.truncate_span_checked(&text, 0, &check).map(|_| ())
            };
            assert!(matches!(result, Err(Error::Cancelled)));
        }
    }

    #[test]
    fn cuts_are_classified_from_the_source_text() {
        assert_eq!(classify("a\n\nb", 3, true), Level::Paragraph);
        assert_eq!(classify("a\r\n\r\nb", 5, true), Level::Paragraph);
        assert_eq!(classify("a\nb", 2, true), Level::Line);
        assert_eq!(classify("a\n  b", 4, true), Level::Line);
        assert_eq!(classify("a\r\nb", 2, true), Level::Token);
        assert_eq!(classify("Hi. Bye", 3, true), Level::Sentence);
        assert_eq!(classify("Hi. Bye", 4, true), Level::Sentence);
        assert_eq!(classify("Hi.\" Bye", 4, true), Level::Sentence);
        assert_eq!(classify("a b", 1, false), Level::Word);
        assert_eq!(classify("ab", 1, true), Level::Word);
        assert_eq!(classify("ab", 1, false), Level::Token);
        assert_eq!(classify("a.b", 2, false), Level::Token);
        assert_eq!(classify("x", 1, false), Level::Paragraph);
    }

    #[test]
    fn choice_prefers_strong_boundaries_that_fill_half_the_budget() {
        let cut = |end, tokens, level| Cut { end, tokens, level };
        let cuts = [
            cut(2, 2, Level::Paragraph),
            cut(5, 5, Level::Word),
            cut(6, 5, Level::Word),
            cut(8, 8, Level::Token),
        ];
        // The paragraph cut fills a quarter; the word cut wins and takes the earlier byte.
        assert_eq!(choose(&cuts, 8, Level::Paragraph, 100), 1);
        assert_eq!(choose(&cuts, 4, Level::Paragraph, 100), 0);
        assert_eq!(choose(&cuts, 8, Level::Token, 100), 3);
        // The end of the text is always acceptable.
        assert_eq!(choose(&cuts[..1], 100, Level::Paragraph, 2), 0);
    }
}
