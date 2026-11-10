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
//! cl100k_base (GPT-4, GPT-3.5-turbo) and Llama 3: the Qwen2 pattern with `\p{N}{1,3}` in place
//! of `\p{N}`, so digits are grouped up to three at a time.
//!
//! Scalar classes come from `unicode_tables.rs`, captured from the regex engine itself.

use super::unicode_tables::{
    CONTRACTION_FOLDS, LETTERS, NUMBERS, O200K_LOWER, O200K_UPPER, WHITESPACE,
};

/// The exact regex text the Qwen2 family stores in its `Split` pre-tokenizer.
pub const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The regex text cl100k_base conversions (and Llama 3) store in their `Split` pre-tokenizer.
pub const CL100K_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The regex text o200k_base (GPT-4o) conversions store in their `Split` pre-tokenizer.
pub const O200K_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Gpt2,
    Qwen2,
    Cl100k,
    O200k,
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

/// o200k `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`: letters that are not lowercase, plus marks.
#[inline]
fn is_upperish(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_uppercase()
    } else {
        in_ranges(O200K_UPPER, c)
    }
}

/// o200k `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`: letters that are not uppercase or titlecase, plus marks.
#[inline]
fn is_lowerish(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_lowercase()
    } else {
        in_ranges(O200K_LOWER, c)
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
            Pattern::Qwen2 => qwen2_match(text, pos, 1)?,
            Pattern::Cl100k => qwen2_match(text, pos, 3)?,
            Pattern::O200k => o200k_match(text, pos)?,
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

/// The one non-ASCII scalar the regex engine's case folding maps onto a contraction letter:
/// U+017F LATIN SMALL LETTER LONG S folds to `s`.
const LONG_S: char = '\u{17f}';

/// A non-ASCII scalar in a contraction letter position: `Ok(true)` if it is the letter `s` by
/// case folding, `Ok(false)` if it cannot be any contraction letter, and an error for a scalar
/// that folds onto a contraction letter in a way this code does not model (`CONTRACTION_FOLDS`
/// is generated from the engine, so a future engine could add one).
fn folds_to_s(c: char) -> Result<bool, Unsupported> {
    if c == LONG_S {
        Ok(true)
    } else if in_ranges(CONTRACTION_FOLDS, c) {
        Err(Unsupported::FoldedContraction)
    } else {
        Ok(false)
    }
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`, including the engine's case folding of non-ASCII scalars.
fn contraction_ci(text: &str, pos: usize) -> Result<usize, Unsupported> {
    let rest = &text[pos..];
    let mut chars = rest.chars();
    if chars.next() != Some('\'') {
        return Ok(0);
    }
    let Some(first) = chars.next() else {
        return Ok(0);
    };
    if !first.is_ascii() {
        return Ok(if folds_to_s(first)? {
            1 + first.len_utf8()
        } else {
            0
        });
    }
    let second = chars.next();
    if let Some(c) = second.filter(|c| !c.is_ascii()) {
        // No contraction has `s` second, so a folding scalar there ends the match too; one that
        // folds some other way is still reported.
        folds_to_s(c)?;
    }
    let second = second
        .filter(char::is_ascii)
        .map(|c| c.to_ascii_lowercase());
    Ok(match (first.to_ascii_lowercase(), second) {
        ('s' | 't' | 'm' | 'd', _) => 2,
        ('r' | 'v', Some('e')) | ('l', Some('l')) => 3,
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

/// Qwen2's pattern, whose digit alternative is `\p{N}` (`max_digits` 1), and cl100k's, the same
/// pattern with `\p{N}{1,3}` (`max_digits` 3).
fn qwen2_match(text: &str, pos: usize, max_digits: usize) -> Result<usize, Unsupported> {
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
    // `\p{N}` or `\p{N}{1,3}`: up to `max_digits` scalars.
    if is_number(first) {
        return Ok(rest
            .chars()
            .take(max_digits)
            .take_while(|&c| is_number(c))
            .map(char::len_utf8)
            .sum());
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

/// `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+` from `start`: the greedy star
/// gives back one scalar at a time until a lower-ish scalar can start the plus, which then takes
/// its whole run. Returns the end, or `None` when no lower-ish scalar is reachable.
fn upper_star_lower_plus(text: &str, start: usize) -> Option<usize> {
    let mut at = start + run(text, start, is_upperish);
    loop {
        if text[at..].chars().next().is_some_and(is_lowerish) {
            return Some(at + run(text, at, is_lowerish));
        }
        if at == start {
            return None;
        }
        at -= text[start..at].chars().next_back().unwrap().len_utf8();
    }
}

/// The two word alternatives of o200k, each `[^\r\n\p{L}\p{N}]?<letters>(?i:'s|...)?`. The
/// optional prefix is tried taken first, then not, within each alternative, and alternative 1
/// is exhausted before alternative 2, as in the backtracking engine.
fn o200k_word(text: &str, pos: usize) -> Result<Option<usize>, Unsupported> {
    let first = text[pos..].chars().next().unwrap();
    let with_prefix = !is_newline(first) && !is_letter(first) && !is_number(first);
    let prefixed = pos + first.len_utf8();
    let starts = if with_prefix {
        [Some(prefixed), Some(pos)]
    } else {
        [Some(pos), None]
    };
    // `[upper-ish]*[lower-ish]+`
    for start in starts.into_iter().flatten() {
        if let Some(end) = upper_star_lower_plus(text, start) {
            return Ok(Some(end + contraction_ci(text, end)?));
        }
    }
    // `[upper-ish]+[lower-ish]*`
    for start in starts.into_iter().flatten() {
        let upper = run(text, start, is_upperish);
        if upper > 0 {
            let end = start + upper + run(text, start + upper, is_lowerish);
            return Ok(Some(end + contraction_ci(text, end)?));
        }
    }
    Ok(None)
}

fn o200k_match(text: &str, pos: usize) -> Result<usize, Unsupported> {
    if let Some(end) = o200k_word(text, pos)? {
        return Ok(end - pos);
    }
    let rest = &text[pos..];
    let first = rest.chars().next().unwrap();
    // `\p{N}{1,3}`
    if is_number(first) {
        return Ok(rest
            .chars()
            .take(3)
            .take_while(|&c| is_number(c))
            .map(char::len_utf8)
            .sum());
    }
    // ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
    let space = usize::from(first == ' ');
    let others = run(text, pos + space, is_other);
    if others > 0 {
        let tail = run(text, pos + space + others, |c| {
            matches!(c, '\r' | '\n' | '/')
        });
        return Ok(space + others + tail);
    }
    // `\s*[\r\n]+`, then `\s+(?!\S)|\s+`
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
        // `ſ` case-folds to `s`, so it completes a contraction; other non-ASCII letters never do.
        // Expected pieces from Python's `regex` module.
        assert_eq!(pieces(Pattern::Qwen2, "it'ſ"), ["it", "'ſ"]);
        assert_eq!(pieces(Pattern::Qwen2, "l'été"), ["l", "'été"]);
        assert_eq!(pieces(Pattern::Qwen2, "x'ré"), ["x", "'ré"]);
        assert_eq!(
            pieces(Pattern::Cl100k, "dell'università"),
            ["dell", "'università"]
        );
    }

    #[test]
    fn cl100k_groups_digits_in_threes_and_otherwise_follows_qwen2() {
        assert_eq!(
            pieces(Pattern::Cl100k, "Hello world's 1234567\n\n  x"),
            ["Hello", " world", "'s", " ", "123", "456", "7", "\n\n", " ", " x"]
        );
        assert_eq!(pieces(Pattern::Cl100k, "12"), ["12"]);
        assert_eq!(pieces(Pattern::Cl100k, "a1234b"), ["a", "123", "4", "b"]);
        // Non-ASCII digits are \p{N} too, and a group never crosses a non-digit.
        assert_eq!(pieces(Pattern::Cl100k, "٣٤٥٦ 1½"), ["٣٤٥", "٦", " ", "1½"]);
    }

    /// Expected pieces from Python's `regex` module on the o200k_base pattern.
    #[test]
    fn o200k_splits_on_case_and_attaches_contractions() {
        let cases: &[(&str, &[&str])] = &[
            (
                "HelloWorld's 12345 x",
                &["Hello", "World's", " ", "123", "45", " x"],
            ),
            ("ABC def", &["ABC", " def"]),
            ("it's DON'T", &["it's", " DON'T"]),
            ("  \n\n x\t/", &["  \n\n", " x", "\t", "/"]),
            ("a/b/\n", &["a", "/b", "/\n"]),
            ("中文 Ελλάδα", &["中文", " Ελλάδα"]),
            ("x\u{301}y \u{301}a", &["x\u{301}y", " \u{301}a"]),
            ("\u{1c5}ungla", &["\u{1c5}ungla"]),
            ("'s", &["'s"]),
            ("1/2", &["1", "/", "2"]),
            ("\u{2b0}\u{1d43}", &["\u{2b0}\u{1d43}"]),
            ("\u{301}", &["\u{301}"]),
            ("!\u{301}x", &["!\u{301}x"]),
            ("ABCdef GHi", &["ABCdef", " GHi"]),
            ("\u{2160}\u{2161} x", &["\u{2160}\u{2161}", " x"]),
        ];
        for (text, expected) in cases {
            assert_eq!(&pieces(Pattern::O200k, text), expected, "{text:?}");
        }
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
