use std::{fs, path::Path};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

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

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct EncodeOptions {
    pub add_special_tokens: bool,
    pub special_tokens: SpecialTokens,
}

#[derive(Debug, Serialize)]
pub struct Artifact {
    /// Hash of the original tokenizer.json bytes, before runtime overrides.
    pub sha256: String,
    pub vocab_size: usize,
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

#[derive(Debug, Serialize)]
pub struct Chunk {
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub count: usize,
}

pub struct Engine {
    tokenizer: Tokenizer,
    artifact: Artifact,
    options: EncodeOptions,
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
        tokenizer
            .set_encode_special_tokens(matches!(options.special_tokens, SpecialTokens::Literal));
        let artifact = Artifact {
            sha256: format!("{:x}", Sha256::digest(bytes)),
            vocab_size: tokenizer.get_vocab_size(true),
        };
        Ok(Self {
            tokenizer,
            artifact,
            options,
        })
    }

    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    pub fn options(&self) -> EncodeOptions {
        self.options
    }

    pub fn encode(&self, text: &str) -> Result<Encoded, Error> {
        let encoding = self
            .tokenizer
            .encode(text, self.options.add_special_tokens)
            .map_err(tokenizer_error)?;
        Ok(Encoded {
            ids: encoding.get_ids().to_vec(),
            tokens: encoding.get_tokens().to_vec(),
            offsets: encoding.get_offsets().to_vec(),
            special_tokens_mask: encoding.get_special_tokens_mask().to_vec(),
            count: encoding.len(),
        })
    }

    pub fn count(&self, text: &str) -> Result<usize, Error> {
        self.tokenizer
            .encode_fast(text, self.options.add_special_tokens)
            .map(|encoding| encoding.len())
            .map_err(tokenizer_error)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        for &id in ids {
            if self.tokenizer.id_to_token(id).is_none() {
                return Err(Error::UnknownId(id));
            }
        }
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(tokenizer_error)
    }

    /// Return a verified source prefix. It need not be the longest fitting prefix:
    /// token counts are not monotonic as text grows.
    pub fn truncate(&self, text: &str, max_tokens: usize) -> Result<Chunk, Error> {
        self.truncate_checked(text, max_tokens, &|| Ok(()))
    }

    pub(crate) fn truncate_checked(
        &self,
        text: &str,
        max_tokens: usize,
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<Chunk, Error> {
        let (end, count) = self.fitting_prefix(text, max_tokens, check)?;
        Ok(Chunk {
            text: text[..end].to_owned(),
            start_byte: 0,
            end_byte: end,
            count,
        })
    }

    /// Partition all source bytes into nonoverlapping, independently encoded chunks.
    /// Boundaries are Unicode scalar boundaries, not necessarily grapheme boundaries.
    pub fn chunk(&self, text: &str, max_tokens: usize) -> Result<Vec<Chunk>, Error> {
        self.chunk_checked(text, max_tokens, &|| Ok(()))
    }

    pub(crate) fn chunk_checked(
        &self,
        text: &str,
        max_tokens: usize,
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<Vec<Chunk>, Error> {
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < text.len() {
            check()?;
            let (length, count) = self.fitting_prefix(&text[start..], max_tokens, check)?;
            if length == 0 {
                return Err(Error::BudgetTooSmall { offset: start });
            }
            let end = start + length;
            chunks.push(Chunk {
                text: text[start..end].to_owned(),
                start_byte: start,
                end_byte: end,
                count,
            });
            start = end;
        }
        Ok(chunks)
    }

    fn fitting_prefix(
        &self,
        text: &str,
        budget: usize,
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<(usize, usize), Error> {
        check()?;
        let full_count = self.count(text)?;
        if full_count <= budget {
            return Ok((text.len(), full_count));
        }
        let boundaries: Vec<usize> = text
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(text.len()))
            .collect();
        let mut low = 1;
        let mut high = boundaries.len() - 1;
        let mut best = None;
        // Every accepted candidate is encoded on its own; monotonic counts are
        // not required for safety. Binary probing need not find a maximal prefix.
        while low < high {
            check()?;
            let middle = low + (high - low) / 2;
            let end = boundaries[middle];
            let count = self.count(&text[..end])?;
            if count <= budget {
                best = Some((end, count));
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        if let Some(prefix) = best {
            return Ok(prefix);
        }
        // Merges/normalization can make a longer prefix fit when a shorter one
        // does not. Fallback prevents incorrectly declaring no progress possible.
        for &end in boundaries.iter().skip(1).rev().skip(1) {
            check()?;
            let count = self.count(&text[..end])?;
            if count <= budget {
                return Ok((end, count));
            }
        }
        let empty_count = self.count("")?;
        if empty_count > budget {
            return Err(Error::EmptyBudget);
        }
        Ok((0, empty_count))
    }
}

fn tokenizer_error(error: tokenizers::Error) -> Error {
    Error::Tokenizer(error.to_string())
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
                engine.chunk_checked(&text, 1, &check).map(|_| ())
            } else {
                engine.truncate_checked(&text, 1, &check).map(|_| ())
            };
            assert!(matches!(result, Err(Error::Cancelled)));
        }
    }
}
