//! Hand-written equivalents of the two byte-level BPE pre-tokenizer regexes used by the
//! supported artifacts. Each alternative of the original pattern is tried in order at the
//! current position, exactly as the backtracking engine does, so the splits are identical.
//!
//! GPT-2 (`ByteLevel` pre-tokenizer):
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!
//! Qwen2 (`Split` with the pattern below, then `ByteLevel` without its own regex):
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! Scalar classes come from `unicode_tables.rs`, captured from the regex engine itself.

use super::unicode_tables::{LETTERS, NUMBERS, WHITESPACE};

/// The exact regex text the Qwen2 family stores in its `Split` pre-tokenizer.
pub const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Gpt2,
    Qwen2,
}

fn in_ranges(table: &[(u32, u32)], c: char) -> bool {
    let code = c as u32;
    table
        .binary_search_by(|&(start, end)| {
            if code < start {
                std::cmp::Ordering::Greater
            } else if code > end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

#[inline]
pub(crate) fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_alphabetic()
    } else {
        in_ranges(LETTERS, c)
    }
}

#[inline]
pub(crate) fn is_number(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_digit()
    } else {
        in_ranges(NUMBERS, c)
    }
}

/// `\s` as the regex engine defines it (the Unicode White_Space property).
#[inline]
pub(crate) fn is_space(c: char) -> bool {
    if c.is_ascii() {
        matches!(c, ' ' | '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r')
    } else {
        in_ranges(WHITESPACE, c)
    }
}

/// `[^\s\p{L}\p{N}]`
#[inline]
fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

/// How the pre-tokenizer regex classifies a scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Letter,
    Number,
    Whitespace,
    Other,
}

pub fn classify(c: char) -> Class {
    if is_letter(c) {
        Class::Letter
    } else if is_number(c) {
        Class::Number
    } else if is_space(c) {
        Class::Whitespace
    } else {
        Class::Other
    }
}

fn is_newline(c: char) -> bool {
    matches!(c, '\r' | '\n')
}

/// Length in bytes of the run of chars satisfying `pred` starting at `start`.
fn run(text: &str, start: usize, pred: impl Fn(char) -> bool) -> usize {
    let mut end = start;
    for c in text[start..].chars() {
        if !pred(c) {
            break;
        }
        end += c.len_utf8();
    }
    end - start
}

/// Why a text cannot be pre-tokenized with certainty; the caller falls back to the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    /// A case-insensitive contraction next to a non-ASCII scalar: the regex engine's Unicode
    /// case folding (for example `ſ` or ligatures) is not reproduced here.
    FoldedContraction,
}

/// Call `emit(start, end)` for each pre-token byte range, in order. Every byte of `text`
/// belongs to exactly one range.
pub fn split(
    pattern: Pattern,
    text: &str,
    mut emit: impl FnMut(usize, usize),
) -> Result<(), Unsupported> {
    let mut pos = 0;
    while pos < text.len() {
        let len = match pattern {
            Pattern::Gpt2 => gpt2_match(text, pos),
            Pattern::Qwen2 => qwen2_match(text, pos)?,
        };
        debug_assert!(len > 0 && text.is_char_boundary(pos + len));
        emit(pos, pos + len);
        pos += len;
    }
    Ok(())
}

/// Case-sensitive `'s|'t|'re|'ve|'m|'ll|'d`.
fn contraction(rest: &[u8]) -> usize {
    if rest.first() != Some(&b'\'') {
        return 0;
    }
    match rest.get(1) {
        Some(b's' | b't' | b'm' | b'd') => 2,
        Some(b'r') if rest.get(2) == Some(&b'e') => 3,
        Some(b'v') if rest.get(2) == Some(&b'e') => 3,
        Some(b'l') if rest.get(2) == Some(&b'l') => 3,
        _ => 0,
    }
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)` over ASCII; a non-ASCII scalar after the apostrophe is
/// reported so the caller can defer to the regex engine's case folding rules.
fn contraction_ci(text: &str, pos: usize) -> Result<usize, Unsupported> {
    let rest = &text.as_bytes()[pos..];
    if rest.first() != Some(&b'\'') {
        return Ok(0);
    }
    let lower = |index: usize| rest.get(index).map(u8::to_ascii_lowercase);
    if rest.get(1).is_some_and(|b| !b.is_ascii()) {
        return Err(Unsupported::FoldedContraction);
    }
    Ok(match lower(1) {
        Some(b's' | b't' | b'm' | b'd') => 2,
        Some(b'r' | b'v' | b'l') => {
            if rest.get(2).is_some_and(|b| !b.is_ascii()) {
                return Err(Unsupported::FoldedContraction);
            }
            match (lower(1), lower(2)) {
                (Some(b'r'), Some(b'e')) | (Some(b'v'), Some(b'e')) | (Some(b'l'), Some(b'l')) => 3,
                _ => 0,
            }
        }
        _ => 0,
    })
}

/// `\s+(?!\S)` then `\s+`: a whitespace run keeps its last scalar for the following token
/// unless the run reaches the end of the text or is a single scalar.
fn whitespace_tail(text: &str, pos: usize) -> usize {
    let len = run(text, pos, is_space);
    debug_assert!(len > 0);
    if pos + len == text.len() {
        return len;
    }
    let last = text[pos..pos + len].chars().next_back().unwrap().len_utf8();
    if len > last {
        len - last
    } else {
        len
    }
}

fn gpt2_match(text: &str, pos: usize) -> usize {
    let rest = &text[pos..];
    let bytes = rest.as_bytes();
    let apostrophe = contraction(bytes);
    if apostrophe > 0 {
        return apostrophe;
    }
    let space = usize::from(bytes[0] == b' ');
    // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`: the optional space counts only when the
    // run after it is non-empty; otherwise the alternative is retried without it (which fails
    // for a space, since a space is none of these classes) and the next alternative runs.
    for class in [is_letter, is_number, is_other] {
        let len = run(text, pos + space, class);
        if len > 0 {
            return space + len;
        }
    }
    whitespace_tail(text, pos)
}

fn qwen2_match(text: &str, pos: usize) -> Result<usize, Unsupported> {
    let rest = &text[pos..];
    let first = rest.chars().next().unwrap();
    let apostrophe = contraction_ci(text, pos)?;
    if apostrophe > 0 {
        return Ok(apostrophe);
    }
    // `[^\r\n\p{L}\p{N}]?\p{L}+`
    let prefix = if !is_newline(first) && !is_letter(first) && !is_number(first) {
        first.len_utf8()
    } else {
        0
    };
    let letters = run(text, pos + prefix, is_letter);
    if letters > 0 {
        return Ok(prefix + letters);
    }
    // `\p{N}`: one scalar.
    if is_number(first) {
        return Ok(first.len_utf8());
    }
    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    let space = usize::from(first == ' ');
    let others = run(text, pos + space, is_other);
    if others > 0 {
        let newlines = run(text, pos + space + others, is_newline);
        return Ok(space + others + newlines);
    }
    // `\s*[\r\n]+`: the whitespace run up to and including its last CR or LF.
    let spaces = run(text, pos, is_space);
    debug_assert!(spaces > 0, "every scalar belongs to some alternative");
    if let Some(last_newline) = rest[..spaces].rfind(['\r', '\n']) {
        return Ok(last_newline + 1);
    }
    Ok(whitespace_tail(text, pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(pattern: Pattern, text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        split(pattern, text, |s, e| out.push(&text[s..e])).unwrap();
        out
    }

    #[test]
    fn gpt2_alternatives_follow_regex_order() {
        assert_eq!(
            pieces(Pattern::Gpt2, "Hello world's  test 123,  \n\nx"),
            ["Hello", " world", "'s", " ", " test", " 123", ",", "  \n", "\n", "x"]
        );
        assert_eq!(pieces(Pattern::Gpt2, "   "), ["   "]);
        assert_eq!(pieces(Pattern::Gpt2, " a"), [" a"]);
        assert_eq!(
            pieces(Pattern::Gpt2, "a \u{a0}b"),
            ["a", " ", "\u{a0}", "b"]
        );
        assert_eq!(pieces(Pattern::Gpt2, "'sand 'S"), ["'s", "and", " '", "S"]);
        assert_eq!(pieces(Pattern::Gpt2, "😀😀 x"), ["😀😀", " x"]);
    }

    #[test]
    fn qwen2_alternatives_follow_regex_order() {
        assert_eq!(
            pieces(Pattern::Qwen2, "Hello world's 123\n\n  x\tab,\r\n"),
            ["Hello", " world", "'s", " ", "1", "2", "3", "\n\n", " ", " x", "\tab", ",\r\n"]
        );
        assert_eq!(pieces(Pattern::Qwen2, "'S 'Re"), ["'S", " '", "Re"]);
        assert_eq!(
            pieces(Pattern::Qwen2, "a  \n \n  b"),
            ["a", "  \n \n", " ", " b"]
        );
        assert_eq!(pieces(Pattern::Qwen2, "x   "), ["x", "   "]);
        assert_eq!(
            split(Pattern::Qwen2, "it'ſ", |_, _| {}),
            Err(Unsupported::FoldedContraction)
        );
    }

    #[test]
    fn classes_match_std_for_ascii_and_cover_common_scripts() {
        assert!(is_letter('é') && is_letter('漢') && is_letter('ع'));
        assert!(!is_letter('1') && !is_letter(' ') && !is_letter('😀'));
        assert!(is_number('٣') && is_number('½') && is_number('Ⅻ'));
        assert!(is_space('\u{3000}') && is_space('\u{85}') && is_space('\u{2028}'));
        assert!(!is_space('\u{200b}'));
    }
}
