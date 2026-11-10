//! Differential soak: the custom byte-level BPE core against the Hugging Face backend on
//! millions of inputs per artifact and option combination, in release mode.
//!
//!     cargo run --release --example core_soak
//!     cargo run --release --example core_soak -- --random 1000000 --seed 20261018 --output soak.json
//!     cargo run --release --example core_soak -- --quick            # a smoke run of the same shape
//!
//! For every artifact in `models/MANIFEST.json` that the core accepts, and for every combination
//! of `SpecialTokens::{Literal, Recognize}` and `add_special_tokens` on/off, each input is run
//! through four things: the core directly (`ByteLevelBpe`, so declines are visible), an `Engine`
//! in `CoreMode::Verify` (the production path, which double-encodes and fails on disagreement),
//! an `Engine` in `CoreMode::Backend` (cross-checked against the oracle on the corpus-derived
//! inputs) and a backend `Tokenizer` configured exactly as the engine configures it (the oracle,
//! which also yields word ids and `encode_fast` counts). IDs, byte offsets, the special-token
//! mask, word ids, token strings and counts must all be equal wherever the core serves the input.
//!
//! Inputs, in index order so any one of them is reproducible from the seed and its index:
//! every corpus file whole and each of its lines; sliding windows and random slices of the files
//! at scalar boundaries; strings crossing every regex class boundary; every Unicode scalar alone
//! and in four contexts; every pair from a sample set of hard scalars; and a seeded generator
//! biased toward whitespace runs, CR/LF mixes, digit runs, mixed-case contractions, many scripts,
//! combining marks, ZWJ emoji, half-width forms, unassigned and control code points, byte-alphabet
//! characters, and the artifact's own added-token spellings whole, partial, nested and glued.
//!
//! The first mismatches are printed as Rust string literals with both ID sequences and the first
//! differing index; after each combination every distinct mismatch is shrunk by deleting scalars
//! while the same kind of mismatch persists, and the minimal input is printed the same way.
use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokenizers::{Encoding, Tokenizer};
use tokforge::{
    bytelevel::{classify, ByteLevelBpe, Class, Fallback, Output, PretokenizeUnsupported},
    CoreMode, EncodeOptions, Engine, Error, SpecialTokens, BACKEND_CORE,
};
use unicode_normalization_alignments::UnicodeNormalization;

// ---------------------------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------------------------

struct Args {
    /// Inputs from the seeded generator per artifact and combination.
    random: u64,
    seed: u64,
    threads: usize,
    /// Only artifacts whose manifest name contains one of these.
    artifacts: Vec<String>,
    output: Option<PathBuf>,
    /// Mismatches kept (and shrunk) per combination; the rest are only counted.
    max_mismatches: usize,
    /// Run the `Verify` engine on every N-th input (1 = every input).
    engine_stride: u64,
    /// Take every N-th scalar in the full-range sweeps (1 = every scalar).
    scalar_stride: u64,
    /// Random corpus slices per artifact.
    slices: u64,
    /// Window sizes are divided by this to thin the sliding-window set (1 = all).
    window_thinning: u64,
    /// Feed the core the opposite special-token mode so that mismatches appear on inputs with
    /// special spellings: exercises the reporting and the shrinker.
    self_test: bool,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = Args {
        random: 1_000_000,
        seed: 0x2026_1018,
        threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        artifacts: Vec::new(),
        output: None,
        max_mismatches: 40,
        engine_stride: 1,
        scalar_stride: 1,
        slices: 100_000,
        window_thinning: 1,
        self_test: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = |name: &str| -> anyhow::Result<String> {
            iter.next()
                .ok_or_else(|| anyhow::anyhow!("{name} needs a value"))
        };
        match flag.as_str() {
            "--random" => args.random = value("--random")?.parse()?,
            "--seed" => args.seed = value("--seed")?.parse()?,
            "--threads" => args.threads = value("--threads")?.parse()?,
            "--artifact" => args.artifacts.push(value("--artifact")?),
            "--output" => args.output = Some(PathBuf::from(value("--output")?)),
            "--max-mismatches" => args.max_mismatches = value("--max-mismatches")?.parse()?,
            "--engine-stride" => args.engine_stride = value("--engine-stride")?.parse()?,
            "--scalar-stride" => args.scalar_stride = value("--scalar-stride")?.parse()?,
            "--slices" => args.slices = value("--slices")?.parse()?,
            "--window-thinning" => args.window_thinning = value("--window-thinning")?.parse()?,
            "--self-test" => args.self_test = true,
            "--quick" => {
                args.random = 20_000;
                args.scalar_stride = 97;
                args.slices = 2_000;
                args.window_thinning = 8;
            }
            other => anyhow::bail!(
                "unknown argument {other}; expected --random, --seed, --threads, --artifact, \
                 --output, --max-mismatches, --engine-stride, --scalar-stride, --slices, \
                 --window-thinning or --quick"
            ),
        }
    }
    anyhow::ensure!(args.threads > 0 && args.engine_stride > 0 && args.scalar_stride > 0);
    Ok(args)
}

// ---------------------------------------------------------------------------------------------
// Deterministic randomness
// ---------------------------------------------------------------------------------------------

/// SplitMix64: every input is generated from `Rng::for_input(seed, index)`, so a mismatch is
/// reproducible from the seed and the index alone, whatever the thread count.
#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn for_input(seed: u64, index: u64) -> Self {
        let mut rng = Rng(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        rng.next();
        rng
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        self.next() % n
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len() as u64) as usize]
    }

    fn pick_owned<'a>(&mut self, items: &'a [String]) -> &'a str {
        &items[self.below(items.len() as u64) as usize]
    }

    fn scalar(&mut self) -> char {
        loop {
            if let Some(c) = char::from_u32(self.below(0x11_0000) as u32) {
                return c;
            }
        }
    }

    fn scalar_in(&mut self, start: u32, end: u32) -> Option<char> {
        char::from_u32(start + self.below(u64::from(end - start + 1)) as u32)
    }
}

/// The `n`-th Unicode scalar value (surrogates skipped), for exhaustive sweeps.
fn nth_scalar(n: u64) -> char {
    let code = if n < 0xD800 { n } else { n + 0x800 };
    char::from_u32(code as u32).expect("surrogates are skipped")
}

const SCALAR_COUNT: u64 = 0x11_0000 - 0x800;

// ---------------------------------------------------------------------------------------------
// Piece tables for the generator
// ---------------------------------------------------------------------------------------------

const WHITESPACE: &[&str] = &[
    " ",
    "  ",
    "   ",
    "    ",
    "\t",
    "\t\t",
    " \t",
    "\t ",
    " \t ",
    "\n",
    "\n\n",
    "\n\n\n",
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
    "\u{b}\u{c}",
    "\u{85}",
    "\u{a0}",
    "\u{a0}\u{a0}",
    " \u{a0} ",
    "\u{a0}\n",
    "\u{1680}",
    "\u{2000}",
    "\u{2003}",
    "\u{2007}",
    "\u{200a}",
    "\u{2028}",
    "\u{2029}",
    "\u{202f}",
    "\u{205f}",
    "\u{3000}",
    "\u{3000}\u{3000}",
    "\r\u{85}\n",
    "\u{2028}\n",
    " \u{3000}\r\n",
    "\u{2029}\r",
];

const DIGITS: &[&str] = &[
    "0",
    "7",
    "42",
    "100",
    "1234",
    "12345",
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
    "1/2",
    "\u{663}",
    "\u{661}\u{662}\u{663}",
    "\u{6f1}\u{6f2}",
    "\u{967}\u{968}\u{969}",
    "\u{9e7}\u{9e8}",
    "\u{e51}\u{e52}",
    "\u{ff11}\u{ff12}\u{ff13}",
    "\u{1d7d8}\u{1d7d9}",
    "\u{b2}\u{b3}\u{b9}",
    "\u{bd}\u{bc}\u{be}",
    "\u{2153}",
    "\u{216b}",
    "\u{2170}",
    "\u{2460}\u{2461}",
    "\u{2469}",
    "\u{3007}",
    "\u{3021}",
    "\u{1369}",
    "12\u{663}4",
    "1\u{b2}",
    "\u{1d7d8}9",
    "1\u{2044}2",
];

const CONTRACTIONS: &[&str] = &[
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
    "'vv",
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
    "'sa",
    "'ta",
    "y'all",
    "rock'n'roll",
    "'em",
    "O'Neil",
    "we're",
    "WE'RE",
    "I'm",
    "I'M",
    "you'll",
    "YOU'LL",
    "she'd",
    "SHE'D",
    "they've",
    "THEY'VE",
    "can't",
    "won't",
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
];

const WORDS: &[&str] = &[
    "hello",
    "World",
    "it",
    "the",
    "they",
    "I",
    "x",
    "ABC",
    "MiXeD",
    "Supercalifragilisticexpialidocious",
    "na\u{ef}ve",
    "caf\u{e9}",
    "\u{3a9}mega",
    "stra\u{df}e",
    "\u{fb01}ne",
    "\u{17f}",
    "\u{130}stanbul",
    "\u{1c5}",
    "\u{3b1}\u{3b2}\u{3b3}",
    "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}",
    "\u{561}\u{575}\u{562}",
    "\u{5e9}\u{5dc}\u{5d5}\u{5dd}",
    "\u{645}\u{631}\u{62d}\u{628}\u{627}",
    "\u{710}\u{72b}\u{720}\u{72c}",
    "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947}",
    "\u{995}\u{9c7}\u{9ae}\u{9a8}",
    "\u{e2a}\u{e27}\u{e31}\u{e2a}\u{e14}\u{e35}",
    "\u{f56}\u{f7c}\u{f51}",
    "\u{10d2}\u{10d0}\u{10db}\u{10d0}",
    "\u{d55c}\u{ad6d}\u{c5b4}",
    "\u{1100}\u{1161}",
    "\u{1100}\u{11a8}",
    "\u{ac00}\u{11a8}",
    "\u{3131}\u{314f}",
    "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}",
    "\u{30ab}\u{30bf}\u{30ab}\u{30ca}",
    "\u{ff76}\u{ff80}\u{ff76}\u{ff85}",
    "\u{ff76}\u{ff9e}",
    "\u{ff8a}\u{ff9f}",
    "\u{65e5}\u{672c}\u{8a9e}",
    "\u{4e2d}\u{6587}",
    "\u{20000}\u{20001}",
    "\u{ff28}\u{ff45}\u{ff4c}\u{ff4c}\u{ff4f}",
    "\u{13a0}\u{13a1}",
    "\u{1230}\u{1235}",
    "\u{1820}\u{1821}",
    "\u{1781}\u{17d2}\u{1798}",
    "\u{1019}\u{103c}\u{1014}\u{103a}",
    "\u{d9c}\u{dd2}",
    "\u{ba4}\u{bae}\u{bbf}\u{bb4}\u{bcd}",
    "\u{c24}\u{c46}\u{c32}\u{c41}",
    "\u{c95}\u{ca8}\u{ccd}\u{ca8}",
    "\u{d2e}\u{d32}\u{d2f}\u{d3e}",
    "\u{e9e}\u{eb2}\u{ea7}",
    "\u{a000}\u{a001}",
    "\u{a500}\u{a501}",
    "\u{10400}\u{10401}",
    "\u{10330}\u{10331}",
    "\u{10300}",
    "\u{1e900}\u{1e901}",
    "\u{1d400}\u{1d401}",
    "\u{1d538}",
    "\u{210c}",
    "\u{2115}",
    "\u{2135}",
    "\u{2b0}",
    "\u{1d43}",
    "\u{24d0}",
    "\u{aa}",
    "\u{ba}",
    "\u{b5}",
    "\u{3c0}",
    "\u{1e9b}",
    "\u{a731}",
    "\u{1f1}",
    "\u{1f3}",
    "\u{fb00}",
    "\u{fb06}",
    "\u{fffd}",
    "\u{1d7cb}",
    "\u{2c00}",
    "\u{10900}",
    "\u{16a0}",
    "\u{1b000}",
    "\u{5d0}\u{5b8}",
    "\u{627}\u{64e}",
    "\u{915}\u{94d}\u{937}",
    "\u{e01}\u{e49}",
];

const COMBINING: &[&str] = &[
    "e\u{301}",
    "a\u{308}\u{301}",
    "\u{301}",
    "\u{300}\u{300}",
    "x\u{300}\u{300}",
    "\u{41}\u{30a}",
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
    "\u{5d0}\u{5b7}\u{5bc}",
    "\u{627}\u{653}",
    "\u{622}",
    "\u{64e}\u{651}",
    "\u{651}\u{64e}",
    "\u{34f}",
    "\u{35c}",
    "\u{362}",
    "\u{1ab0}",
    "\u{20d0}",
    "\u{20e3}",
    "\u{fe20}",
    "\u{1d165}",
    "\u{1d167}",
    "\u{e0100}",
    "\u{fe00}",
    "\u{fe0f}",
    "o\u{308}",
    "\u{f6}",
    "\u{6c}\u{323}\u{304}",
    "\u{1e38}",
    "\u{1e36}\u{304}",
    "\u{3b1}\u{301}",
    "\u{3ac}",
    "\u{1f71}",
    "\u{391}\u{345}",
    "\u{1fbc}",
    "\u{b7}\u{300}",
    "\u{302}\u{301}\u{300}",
    "\u{9c7}\u{9be}",
    "\u{9cb}",
];

const EMOJI: &[&str] = &[
    "\u{1f600}",
    "\u{1f44d}",
    "\u{1f44d}\u{1f3fd}",
    "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}\u{200d}\u{1f466}",
    "\u{1f3f3}\u{fe0f}\u{200d}\u{1f308}",
    "\u{1f1f5}\u{1f1f0}",
    "\u{1f1fa}\u{1f1f8}\u{1f1ec}\u{1f1e7}",
    "1\u{fe0f}\u{20e3}",
    "#\u{fe0f}\u{20e3}",
    "*\u{20e3}",
    "\u{a9}\u{fe0f}",
    "\u{263a}\u{fe0f}",
    "\u{263a}",
    "\u{2764}\u{fe0f}\u{200d}\u{1f525}",
    "\u{1f9d1}\u{1f3ff}\u{200d}\u{1f680}",
    "\u{1fae0}",
    "\u{1f972}",
    "\u{1f1e6}",
    "\u{fe0f}",
    "\u{200d}",
    "\u{200d}\u{200d}",
    "\u{e0067}",
    "\u{1f3f4}\u{e0067}\u{e0062}\u{e0065}\u{e006e}\u{e0067}\u{e007f}",
    "\u{1f600}\u{1f600}\u{1f600}",
    "\u{2600}",
    "\u{2b50}",
    "\u{1f1ff}",
    "\u{1f3fb}",
    "\u{1f46a}",
    "\u{1f469}\u{200d}\u{2695}\u{fe0f}",
];

const HALFWIDTH: &[&str] = &[
    "\u{ff76}\u{ff9e}",
    "\u{ff8a}\u{ff9f}",
    "\u{ff71}\u{ff72}\u{ff73}",
    "\u{ff21}\u{ff11}",
    "\u{ff01}\u{ff1f}",
    "\u{ffdc}",
    "\u{ff61}",
    "\u{ff62}\u{ff63}",
    "\u{ff64}",
    "\u{ff65}",
    "\u{ff70}",
    "\u{ffa0}",
    "\u{ffe0}",
    "\u{ffe8}",
    "\u{ff10}",
    "\u{ff5e}",
    "\u{3000}\u{ff21}",
];

const UNASSIGNED: &[&str] = &[
    "\u{378}",
    "\u{380}",
    "\u{38b}",
    "\u{530}",
    "\u{58c}",
    "\u{2fe0}",
    "\u{e0080}",
    "\u{10fffe}",
    "\u{10ffff}",
    "\u{1fffe}",
    "\u{fdd0}",
    "\u{fdef}",
    "\u{fffe}",
    "\u{ffff}",
    "\u{40000}",
    "\u{dffff}",
    "\u{e000}",
    "\u{f8ff}",
    "\u{f0000}",
    "\u{10fffd}",
    "\u{2065}",
    "\u{fff0}",
    "\u{1ffff}",
    "\u{323b0}",
];

const CONTROL: &[&str] = &[
    "\0",
    "\u{1}",
    "\u{7}",
    "\u{8}",
    "\u{e}",
    "\u{1b}",
    "\u{1c}",
    "\u{1d}",
    "\u{1e}",
    "\u{1f}",
    "\u{7f}",
    "\u{80}",
    "\u{84}",
    "\u{85}",
    "\u{86}",
    "\u{9f}",
    "\u{ad}",
    "\u{200b}",
    "\u{200c}",
    "\u{200e}",
    "\u{200f}",
    "\u{202a}",
    "\u{202e}",
    "\u{2060}",
    "\u{2066}",
    "\u{2069}",
    "\u{feff}",
    "\u{61c}",
    "\u{180e}",
    "\u{fff9}",
    "\u{fffb}",
    "\u{e0001}",
    "\u{e0020}",
    "\u{e007f}",
    "\u{110bd}",
    "\u{1bca0}",
    "\0\0",
    "\u{1b}[0m",
];

const PUNCTUATION: &[&str] = &[
    ".",
    ",",
    "!",
    "?",
    "...",
    "---",
    "!!!",
    "?!",
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
    "\u{2018}\u{2019}",
    "\u{2022}",
    "\u{2192}",
    "\u{20ac}",
    "\u{a3}",
    "\u{a5}",
    "\u{a9}",
    "\u{ae}",
    "\u{2122}",
    "\u{b0}",
    "\u{b1}",
    "\u{d7}",
    "\u{f7}",
    "\u{221a}",
    "\u{221e}",
    "\u{2260}",
    "\u{2264}",
    "\u{2211}",
    "\u{bf}",
    "\u{a1}",
    "\u{a7}",
    "\u{b6}",
    "\u{b7}",
    "\u{b8}",
    "\u{ac}",
    "\u{af}",
    "\u{a8}",
    "\u{b4}",
    "\u{2500}\u{2502}",
    "\u{2588}",
    "\u{3001}",
    "\u{3002}",
    "\u{300c}\u{300d}",
    "\u{60c}",
    "\u{61f}",
    "\u{964}",
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
];

const ALPHABET_CHARS: &[&str] = &[
    "\u{120}",
    "\u{120}\u{120}",
    "\u{10a}",
    "\u{109}",
    "\u{100}",
    "\u{101}",
    "\u{143}",
    "\u{c3}",
    "\u{c2}",
    "\u{c3}\u{a9}",
    "\u{120}a",
    "a\u{120}",
    "\u{120}\u{10a}",
    "\u{121}",
    "\u{c4}\u{a0}",
    "\u{e2}\u{80}\u{99}",
];

/// Unicode blocks the random-run generator draws scalars from (inclusive ranges).
const BLOCKS: &[(u32, u32)] = &[
    (0x0020, 0x007e),
    (0x0000, 0x00ff),
    (0x00c0, 0x024f),
    (0x0250, 0x02ff),
    (0x0300, 0x036f),
    (0x0370, 0x03ff),
    (0x0400, 0x04ff),
    (0x0530, 0x058f),
    (0x0590, 0x05ff),
    (0x0600, 0x06ff),
    (0x0700, 0x074f),
    (0x0900, 0x097f),
    (0x0980, 0x09ff),
    (0x0b80, 0x0bff),
    (0x0e00, 0x0e7f),
    (0x0f00, 0x0fff),
    (0x10a0, 0x10ff),
    (0x1100, 0x11ff),
    (0x1200, 0x137f),
    (0x13a0, 0x13ff),
    (0x1780, 0x17ff),
    (0x1800, 0x18af),
    (0x1d00, 0x1d7f),
    (0x1e00, 0x1eff),
    (0x1f00, 0x1fff),
    (0x2000, 0x206f),
    (0x2070, 0x209f),
    (0x20d0, 0x20ff),
    (0x2100, 0x214f),
    (0x2150, 0x218f),
    (0x2190, 0x22ff),
    (0x2460, 0x24ff),
    (0x2c00, 0x2c5f),
    (0x3000, 0x303f),
    (0x3040, 0x309f),
    (0x30a0, 0x30ff),
    (0x3130, 0x318f),
    (0x4e00, 0x4eff),
    (0xa960, 0xa97f),
    (0xac00, 0xacff),
    (0xd7b0, 0xd7ff),
    (0xe000, 0xe0ff),
    (0xfb00, 0xfb4f),
    (0xfe00, 0xfe0f),
    (0xfe20, 0xfe2f),
    (0xff00, 0xffef),
    (0xfff0, 0xffff),
    (0x10400, 0x1044f),
    (0x1d100, 0x1d1ff),
    (0x1d400, 0x1d7ff),
    (0x1f1e6, 0x1f1ff),
    (0x1f300, 0x1f5ff),
    (0x1f900, 0x1f9ff),
    (0x20000, 0x200ff),
    (0x2fe00, 0x2fe7f),
    (0xe0000, 0xe007f),
    (0xe0100, 0xe01ef),
    (0xf0000, 0xf00ff),
    (0x10ff00, 0x10ffff),
];

/// Scalars whose pairwise combinations are tried exhaustively: one or two representatives of
/// every class the regexes distinguish, every kind of invisible or combining scalar, the
/// byte-alphabet characters, singletons that NFC rewrites, and unassigned or private values.
const PAIR_SAMPLE: &[char] = &[
    ' ',
    '\t',
    '\n',
    '\r',
    '\u{b}',
    '\u{c}',
    '\0',
    '\u{1f}',
    '\u{7f}',
    '\'',
    '"',
    '`',
    '0',
    '1',
    '9',
    'a',
    'z',
    'A',
    'Z',
    's',
    't',
    'S',
    'T',
    'd',
    'm',
    'r',
    'e',
    'v',
    'l',
    'L',
    '!',
    '#',
    '$',
    '%',
    '&',
    '(',
    ')',
    '*',
    '+',
    ',',
    '-',
    '.',
    '/',
    ':',
    ';',
    '<',
    '=',
    '>',
    '?',
    '@',
    '[',
    '\\',
    ']',
    '^',
    '_',
    '{',
    '|',
    '}',
    '~',
    '\u{80}',
    '\u{85}',
    '\u{a0}',
    '\u{ad}',
    '\u{b2}',
    '\u{b5}',
    '\u{bc}',
    '\u{c5}',
    '\u{e9}',
    '\u{f7}',
    '\u{ff}',
    '\u{1680}',
    '\u{2000}',
    '\u{2007}',
    '\u{200a}',
    '\u{2028}',
    '\u{2029}',
    '\u{202f}',
    '\u{205f}',
    '\u{3000}',
    '\u{200b}',
    '\u{200c}',
    '\u{200d}',
    '\u{200e}',
    '\u{202e}',
    '\u{2060}',
    '\u{2066}',
    '\u{feff}',
    '\u{61c}',
    '\u{180e}',
    '\u{e0001}',
    '\u{e0020}',
    '\u{e007f}',
    '\u{300}',
    '\u{301}',
    '\u{308}',
    '\u{323}',
    '\u{34f}',
    '\u{591}',
    '\u{5b0}',
    '\u{64b}',
    '\u{670}',
    '\u{93c}',
    '\u{94d}',
    '\u{e31}',
    '\u{e33}',
    '\u{1ab0}',
    '\u{20d0}',
    '\u{20e3}',
    '\u{fe0e}',
    '\u{fe0f}',
    '\u{fe20}',
    '\u{1d165}',
    '\u{1d167}',
    '\u{3b1}',
    '\u{3a9}',
    '\u{44f}',
    '\u{561}',
    '\u{5d0}',
    '\u{639}',
    '\u{710}',
    '\u{780}',
    '\u{915}',
    '\u{995}',
    '\u{e01}',
    '\u{f40}',
    '\u{10d0}',
    '\u{ac00}',
    '\u{ac01}',
    '\u{1100}',
    '\u{1161}',
    '\u{11a8}',
    '\u{3131}',
    '\u{3042}',
    '\u{30a2}',
    '\u{ff71}',
    '\u{ff9e}',
    '\u{ff9f}',
    '\u{4e2d}',
    '\u{20000}',
    '\u{ff21}',
    '\u{ff11}',
    '\u{ff01}',
    '\u{13a0}',
    '\u{1200}',
    '\u{1820}',
    '\u{1780}',
    '\u{10400}',
    '\u{1d400}',
    '\u{1d7d8}',
    '\u{210c}',
    '\u{2115}',
    '\u{2135}',
    '\u{216b}',
    '\u{2170}',
    '\u{aa}',
    '\u{ba}',
    '\u{2b0}',
    '\u{1c5}',
    '\u{17f}',
    '\u{df}',
    '\u{130}',
    '\u{131}',
    '\u{fb01}',
    '\u{a731}',
    '\u{663}',
    '\u{969}',
    '\u{ff10}',
    '\u{2460}',
    '\u{bd}',
    '\u{2153}',
    '\u{20ac}',
    '\u{a3}',
    '\u{a9}',
    '\u{ae}',
    '\u{2122}',
    '\u{b0}',
    '\u{b1}',
    '\u{d7}',
    '\u{2192}',
    '\u{221e}',
    '\u{2022}',
    '\u{2026}',
    '\u{2014}',
    '\u{201c}',
    '\u{201d}',
    '\u{2018}',
    '\u{2019}',
    '\u{ab}',
    '\u{bb}',
    '\u{bf}',
    '\u{a1}',
    '\u{a7}',
    '\u{b6}',
    '\u{ac}',
    '\u{af}',
    '\u{a8}',
    '\u{b4}',
    '\u{b8}',
    '\u{b7}',
    '\u{1f600}',
    '\u{1f44d}',
    '\u{1f3fd}',
    '\u{1f1f5}',
    '\u{1f1f0}',
    '\u{2764}',
    '\u{263a}',
    '\u{1fae0}',
    '\u{1f9d1}',
    '\u{1f680}',
    '\u{1f3f3}',
    '\u{1f308}',
    '\u{1f3f4}',
    '\u{e000}',
    '\u{f8ff}',
    '\u{f0000}',
    '\u{10fffd}',
    '\u{378}',
    '\u{380}',
    '\u{2fe0}',
    '\u{e0080}',
    '\u{fdd0}',
    '\u{fffe}',
    '\u{ffff}',
    '\u{1fffe}',
    '\u{10ffff}',
    '\u{40000}',
    '\u{dffff}',
    '\u{212b}',
    '\u{2126}',
    '\u{340}',
    '\u{958}',
    '\u{1e9b}',
    '\u{f73}',
    '\u{120}',
    '\u{10a}',
    '\u{100}',
    '\u{105}',
    '\u{c3}',
    '\u{c2}',
    '\u{fffd}',
    '\u{1fef}',
    '\u{2000}',
    '\u{3007}',
    '\u{3021}',
];

/// Contexts applied to every scalar `c` in the exhaustive sweeps.
const SCALAR_CONTEXTS: &[(&str, &str)] = &[("", ""), ("'", ""), (" ", ""), ("", "\n"), ("a", "1")];

// ---------------------------------------------------------------------------------------------
// Input space
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Category {
    Corpus,
    Window,
    Slice,
    Boundary,
    Scalar,
    Pair,
    Random,
}

impl Category {
    const ALL: [Category; 7] = [
        Category::Corpus,
        Category::Window,
        Category::Slice,
        Category::Boundary,
        Category::Scalar,
        Category::Pair,
        Category::Random,
    ];

    fn name(self) -> &'static str {
        match self {
            Category::Corpus => "corpus",
            Category::Window => "window",
            Category::Slice => "slice",
            Category::Boundary => "boundary",
            Category::Scalar => "scalar",
            Category::Pair => "pair",
            Category::Random => "random",
        }
    }
}

/// Pieces derived from the artifact's own added tokens: whole, doubled, partial, mis-cased,
/// and glued to letters, digits, punctuation, whitespace and each other.
fn special_pieces(added: &[String]) -> Vec<String> {
    let mut pieces = Vec::new();
    for token in added {
        let chars: Vec<char> = token.chars().collect();
        let n = chars.len();
        let take = |range: std::ops::Range<usize>| chars[range].iter().collect::<String>();
        pieces.push(token.clone());
        pieces.push(format!("{token}{token}"));
        pieces.push(format!("{token}{token}{token}"));
        if n > 1 {
            pieces.push(take(0..n - 1));
            pieces.push(take(1..n));
            pieces.push(take(0..n / 2));
            pieces.push(take(n / 2..n));
            pieces.push(format!("{} {}", take(0..n / 2), take(n / 2..n)));
            pieces.push(format!("{}{}", take(0..n - 1), take(0..n)));
            pieces.push(format!("{}{}", take(0..n), take(1..n)));
        }
        let upper = token.to_uppercase();
        if upper != *token {
            pieces.push(upper);
        }
        let lower = token.to_lowercase();
        if lower != *token {
            pieces.push(lower);
        }
        for (before, after) in [
            ("a", ""),
            ("", "a"),
            ("1", ""),
            ("", "1"),
            (" ", ""),
            ("", " "),
            ("\n", ""),
            ("", "\n"),
            ("\r\n", "\r\n"),
            ("", "."),
            ("'", "s"),
            ("\u{e9}", "\u{301}"),
            ("  ", "  "),
            ("<|", "|>"),
            ("\u{1f600}", "\u{1f600}"),
        ] {
            pieces.push(format!("{before}{token}{after}"));
        }
    }
    for (i, a) in added.iter().enumerate() {
        for b in added.iter().skip(i + 1).take(3) {
            pieces.push(format!("{a}{b}"));
            pieces.push(format!("{b}{a}"));
            let cut: String = a
                .chars()
                .take(a.chars().count().saturating_sub(2))
                .collect();
            pieces.push(format!("{cut}{b}"));
        }
    }
    if pieces.is_empty() {
        pieces.push("<|endoftext|>".to_owned());
    }
    pieces
}

struct Generator {
    specials: Vec<String>,
    corpus_lines: Vec<String>,
}

impl Generator {
    fn piece(&self, rng: &mut Rng, out: &mut String) {
        match rng.below(100) {
            0..=11 => out.push_str(rng.pick(WHITESPACE)),
            12..=14 => {
                let n = 1 + rng.below(24) as usize;
                let c = rng.pick(&[" ", "\t", "\n", "\r\n", "\u{a0}", "\u{3000}"]);
                for _ in 0..n {
                    out.push_str(c);
                }
            }
            15..=22 => out.push_str(rng.pick(DIGITS)),
            23..=24 => {
                let n = 1 + rng.below(16) as usize;
                for _ in 0..n {
                    out.push(char::from(b'0' + rng.below(10) as u8));
                }
            }
            25..=34 => out.push_str(rng.pick(CONTRACTIONS)),
            35..=48 => out.push_str(rng.pick(WORDS)),
            49..=52 => {
                let (start, end) = BLOCKS[rng.below(BLOCKS.len() as u64) as usize];
                let n = 1 + rng.below(8) as usize;
                for _ in 0..n {
                    if let Some(c) = rng.scalar_in(start, end) {
                        out.push(c);
                    }
                }
            }
            53..=58 => out.push_str(rng.pick(COMBINING)),
            59..=65 => out.push_str(rng.pick(EMOJI)),
            66..=67 => out.push_str(rng.pick(HALFWIDTH)),
            68..=69 => out.push_str(rng.pick(UNASSIGNED)),
            70..=72 => out.push_str(rng.pick(CONTROL)),
            73..=80 => out.push_str(rng.pick_owned(&self.specials)),
            81..=90 => out.push_str(rng.pick(PUNCTUATION)),
            91..=92 => out.push_str(rng.pick(ALPHABET_CHARS)),
            93..=95 => out.push(rng.scalar()),
            96 => {
                if let Some(c) = rng.scalar_in(0, 0xffff) {
                    out.push(c);
                }
            }
            97 => out.push(char::from(rng.below(128) as u8)),
            98 => {
                let n = 100 + rng.below(300) as usize;
                let c = rng.pick(&[
                    "a",
                    " ",
                    "9",
                    "!",
                    "\u{e9}",
                    "\u{1f600}",
                    "\n",
                    "\t",
                    "\u{120}",
                    "A",
                    "'",
                ]);
                for _ in 0..n {
                    out.push_str(c);
                }
            }
            _ => {
                if !self.corpus_lines.is_empty() {
                    out.push_str(rng.pick_owned(&self.corpus_lines));
                }
            }
        }
    }

    fn generate(&self, rng: &mut Rng, out: &mut String) {
        out.clear();
        let pieces = match rng.below(1000) {
            0..=549 => 1 + rng.below(5),
            550..=849 => 6 + rng.below(25),
            850..=984 => 31 + rng.below(170),
            _ => 201 + rng.below(1300),
        };
        for _ in 0..pieces {
            self.piece(rng, out);
        }
        // Occasionally mutate: delete a scalar, duplicate a slice, or insert an apostrophe.
        if rng.chance(10) && !out.is_empty() {
            let boundaries: Vec<usize> = out.char_indices().map(|(i, _)| i).collect();
            let at = boundaries[rng.below(boundaries.len() as u64) as usize];
            match rng.below(3) {
                0 => {
                    let len = out[at..].chars().next().map_or(0, char::len_utf8);
                    out.replace_range(at..at + len, "");
                }
                1 => {
                    let end = boundaries[rng.below(boundaries.len() as u64) as usize];
                    let (from, to) = if at <= end { (at, end) } else { (end, at) };
                    let copy = out[from..to].to_owned();
                    out.insert_str(to, &copy);
                }
                _ => out.insert(at, '\''),
            }
        }
    }
}

struct Inputs {
    corpus_names: Vec<String>,
    corpus: Vec<String>,
    /// Whole files and lines.
    fixed: Vec<(Category, String)>,
    /// (file, start byte, end byte).
    windows: Vec<(u16, u32, u32)>,
    slices: Vec<(u16, u32, u32)>,
    boundary: Vec<String>,
    scalar_stride: u64,
    random: u64,
    seed: u64,
    generator: Generator,
}

impl Inputs {
    fn build(root: &Path, args: &Args, added: &[String]) -> anyhow::Result<Self> {
        let dir = root.join("tests/fixtures/corpus");
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()?;
        files.retain(|p| p.extension().is_some_and(|e| e == "txt"));
        files.sort();
        let mut corpus_names = Vec::new();
        let mut corpus = Vec::new();
        let mut fixed = Vec::new();
        let mut corpus_lines = Vec::new();
        for path in &files {
            let text = fs::read_to_string(path)?;
            corpus_names.push(path.file_name().unwrap().to_string_lossy().into_owned());
            fixed.push((Category::Corpus, text.clone()));
            for line in text.lines() {
                fixed.push((Category::Corpus, line.to_owned()));
                if !line.is_empty() {
                    corpus_lines.push(line.to_owned());
                }
            }
            for line in text.split_inclusive('\n') {
                fixed.push((Category::Corpus, line.to_owned()));
            }
            corpus.push(text);
        }
        anyhow::ensure!(
            !corpus.is_empty(),
            "no corpus files under {}",
            dir.display()
        );

        // Sliding windows at scalar boundaries, sizes in scalars, stride a third of the size.
        let sizes: &[usize] = &[
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024,
        ];
        let mut windows = Vec::new();
        for (file, text) in corpus.iter().enumerate() {
            let boundaries: Vec<usize> = text
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(text.len()))
                .collect();
            let chars = boundaries.len() - 1;
            for &size in sizes {
                let stride = (size / 3).max(1) * args.window_thinning as usize;
                let mut start = 0;
                while start < chars {
                    let end = (start + size).min(chars);
                    windows.push((
                        file as u16,
                        boundaries[start] as u32,
                        boundaries[end] as u32,
                    ));
                    start += stride;
                }
            }
        }

        // Random slices, lengths skewed short but reaching whole files.
        let mut slices = Vec::new();
        let mut rng = Rng::for_input(args.seed, u64::MAX);
        let nonempty: Vec<usize> = (0..corpus.len())
            .filter(|&i| !corpus[i].is_empty())
            .collect();
        for _ in 0..args.slices {
            let file = nonempty[rng.below(nonempty.len() as u64) as usize];
            let text = &corpus[file];
            let boundaries: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
            let chars = boundaries.len();
            let len = (match rng.below(100) {
                0..=59 => 1 + rng.below(32),
                60..=89 => 33 + rng.below(224),
                90..=98 => 257 + rng.below(1792),
                _ => 2049 + rng.below(6144),
            }) as usize;
            let start = rng.below(chars as u64) as usize;
            let end = (start + len).min(chars);
            let end_byte = if end == chars {
                text.len()
            } else {
                boundaries[end]
            };
            slices.push((file as u16, boundaries[start] as u32, end_byte as u32));
        }

        // Strings crossing every boundary between runs of equal regex class.
        let mut boundary = Vec::new();
        let mut previous: Option<(char, Class)> = None;
        let mut runs = 0usize;
        for n in 0..SCALAR_COUNT {
            let c = nth_scalar(n);
            let class = classify(c);
            if let Some((p, pclass)) = previous {
                if pclass != class {
                    runs += 1;
                    for (before, between, after) in [
                        ("", "", ""),
                        ("", "", "a"),
                        (" ", "", ""),
                        ("", " ", ""),
                        ("1", "", "1"),
                        ("x", "", "x"),
                        ("", "\n", ""),
                        ("'", "", ""),
                        ("", "'s", ""),
                        ("\r", "", "\n"),
                        ("", "", "'"),
                        ("<|", "", "|>"),
                    ] {
                        boundary.push(format!("{before}{p}{between}{c}{after}"));
                        boundary.push(format!("{before}{c}{between}{p}{after}"));
                    }
                    boundary.push(format!("'{c}"));
                    boundary.push(format!("'{p}"));
                }
            }
            previous = Some((c, class));
        }
        eprintln!(
            "  {} class runs, {} boundary strings, {} windows, {} slices, {} fixed",
            runs + 1,
            boundary.len(),
            windows.len(),
            slices.len(),
            fixed.len()
        );
        Ok(Self {
            corpus_names,
            corpus,
            fixed,
            windows,
            slices,
            boundary,
            scalar_stride: args.scalar_stride,
            random: args.random,
            seed: args.seed,
            generator: Generator {
                specials: special_pieces(added),
                corpus_lines,
            },
        })
    }

    fn scalars(&self) -> u64 {
        SCALAR_COUNT.div_ceil(self.scalar_stride)
    }

    fn pairs(&self) -> u64 {
        let n = PAIR_SAMPLE.len() as u64;
        n * n
    }

    fn count(&self, category: Category) -> u64 {
        match category {
            Category::Corpus => self.fixed.len() as u64,
            Category::Window => self.windows.len() as u64,
            Category::Slice => self.slices.len() as u64,
            Category::Boundary => self.boundary.len() as u64,
            Category::Scalar => self.scalars() * SCALAR_CONTEXTS.len() as u64,
            Category::Pair => self.pairs(),
            Category::Random => self.random,
        }
    }

    fn total(&self) -> u64 {
        Category::ALL.iter().map(|&c| self.count(c)).sum()
    }

    /// Fill `out` with input `index` and return its category.
    fn get(&self, mut index: u64, out: &mut String) -> Category {
        out.clear();
        for category in Category::ALL {
            let count = self.count(category);
            if index >= count {
                index -= count;
                continue;
            }
            match category {
                Category::Corpus => out.push_str(&self.fixed[index as usize].1),
                Category::Window | Category::Slice => {
                    let (file, start, end) = if category == Category::Window {
                        self.windows[index as usize]
                    } else {
                        self.slices[index as usize]
                    };
                    out.push_str(&self.corpus[file as usize][start as usize..end as usize]);
                }
                Category::Boundary => out.push_str(&self.boundary[index as usize]),
                Category::Scalar => {
                    let contexts = SCALAR_CONTEXTS.len() as u64;
                    let (before, after) = SCALAR_CONTEXTS[(index % contexts) as usize];
                    let c = nth_scalar((index / contexts) * self.scalar_stride);
                    out.push_str(before);
                    out.push(c);
                    out.push_str(after);
                }
                Category::Pair => {
                    let n = PAIR_SAMPLE.len() as u64;
                    out.push(PAIR_SAMPLE[(index / n) as usize]);
                    out.push(PAIR_SAMPLE[(index % n) as usize]);
                }
                Category::Random => {
                    let mut rng = Rng::for_input(self.seed, index);
                    self.generator.generate(&mut rng, out);
                }
            }
            return category;
        }
        unreachable!("index past the input space")
    }
}

// ---------------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Combo {
    recognize: bool,
    add_special: bool,
}

impl Combo {
    const ALL: [Combo; 4] = [
        Combo {
            recognize: false,
            add_special: false,
        },
        Combo {
            recognize: false,
            add_special: true,
        },
        Combo {
            recognize: true,
            add_special: false,
        },
        Combo {
            recognize: true,
            add_special: true,
        },
    ];

    fn label(self) -> String {
        format!(
            "{}/add_special={}",
            if self.recognize {
                "recognize"
            } else {
                "literal"
            },
            self.add_special
        )
    }

    fn options(self, core: CoreMode) -> EncodeOptions {
        EncodeOptions {
            add_special_tokens: self.add_special,
            special_tokens: if self.recognize {
                SpecialTokens::Recognize
            } else {
                SpecialTokens::Literal
            },
            core,
        }
    }
}

#[derive(Default, Clone)]
struct Stats {
    inputs: u64,
    bytes: u64,
    by_category: [u64; 7],
    served_by_core: u64,
    declined_normalization: u64,
    declined_contraction: u64,
    illegitimate_declines: u64,
    engine_checked: u64,
    backend_engine_checked: u64,
    mismatches: u64,
    mismatch_kinds: Vec<(&'static str, u64)>,
    max_tokens: usize,
}

impl Stats {
    fn merge(&mut self, other: &Stats) {
        self.inputs += other.inputs;
        self.bytes += other.bytes;
        for (a, b) in self.by_category.iter_mut().zip(&other.by_category) {
            *a += b;
        }
        self.served_by_core += other.served_by_core;
        self.declined_normalization += other.declined_normalization;
        self.declined_contraction += other.declined_contraction;
        self.illegitimate_declines += other.illegitimate_declines;
        self.engine_checked += other.engine_checked;
        self.backend_engine_checked += other.backend_engine_checked;
        self.mismatches += other.mismatches;
        for &(kind, n) in &other.mismatch_kinds {
            self.bump(kind, n);
        }
        self.max_tokens = self.max_tokens.max(other.max_tokens);
    }

    fn bump(&mut self, kind: &'static str, n: u64) {
        match self.mismatch_kinds.iter_mut().find(|(k, _)| *k == kind) {
            Some(entry) => entry.1 += n,
            None => self.mismatch_kinds.push((kind, n)),
        }
    }
}

#[derive(Clone)]
struct Mismatch {
    kind: &'static str,
    category: Category,
    index: u64,
    text: String,
    core_ids: Vec<u32>,
    backend_ids: Vec<u32>,
    first_diff: Option<usize>,
    detail: String,
    minimal: Option<Box<Mismatch>>,
}

/// One disagreement between the core and the oracle.
struct Diff {
    kind: &'static str,
    detail: String,
    core_ids: Vec<u32>,
    backend_ids: Vec<u32>,
    first_diff: Option<usize>,
}

struct Harness<'a> {
    combo: Combo,
    core: &'a ByteLevelBpe,
    oracle: Tokenizer,
    verify: Engine,
    backend: Engine,
    /// Self-test: the core runs in the opposite special-token mode from the oracle.
    inject: bool,
}

fn first_difference<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    if a == b {
        None
    } else {
        Some(
            a.iter()
                .zip(b)
                .position(|(x, y)| x != y)
                .unwrap_or(a.len().min(b.len())),
        )
    }
}

fn excerpt<T: std::fmt::Debug>(items: &[T], around: usize) -> String {
    let start = around.saturating_sub(4);
    let end = (around + 5).min(items.len());
    let body: Vec<String> = items[start..end].iter().map(|i| format!("{i:?}")).collect();
    format!(
        "[{}{}{}] (len {})",
        if start > 0 { ".., " } else { "" },
        body.join(", "),
        if end < items.len() { ", .." } else { "" },
        items.len()
    )
}

/// Whether NFC really changes `text` (the only legitimate reason for a normalization decline).
fn nfc_changes(text: &str) -> bool {
    !text.nfc().map(|(c, _)| c).eq(text.chars())
}

/// Whether an apostrophe is followed by a non-ASCII scalar somewhere (the only legitimate reason
/// for a contraction decline; the core may also decline a non-ASCII third byte after `'r`, `'v`
/// or `'l`).
fn has_folding_candidate(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.iter().enumerate().any(|(i, &b)| {
        b == b'\''
            && (bytes.get(i + 1).is_some_and(|b| !b.is_ascii())
                || (matches!(
                    bytes.get(i + 1),
                    Some(b'r' | b'R' | b'v' | b'V' | b'l' | b'L')
                ) && bytes.get(i + 2).is_some_and(|b| !b.is_ascii())))
    })
}

impl Harness<'_> {
    /// Compare the core with the oracle on `text`; `None` when the core declined. Fills
    /// `stats` and returns the first mismatch found.
    fn oracle_encode(&self, text: &str) -> Encoding {
        self.oracle
            .encode(text, self.combo.add_special)
            .expect("backend encode")
    }

    fn compare_core(
        &self,
        text: &str,
        expected: &Encoding,
        out: &mut Output,
        stats: Option<&mut Stats>,
    ) -> Result<Option<Diff>, Fallback> {
        let expected_ids = expected.get_ids();
        match self
            .core
            .encode(text, self.combo.recognize ^ self.inject, out)
        {
            Ok(()) => {}
            Err(reason) => {
                if let Some(stats) = stats {
                    match reason {
                        Fallback::Normalization => {
                            stats.declined_normalization += 1;
                            if !nfc_changes(text) {
                                stats.illegitimate_declines += 1;
                            }
                        }
                        Fallback::Pretokenize(PretokenizeUnsupported::FoldedContraction) => {
                            stats.declined_contraction += 1;
                            if !has_folding_candidate(text) {
                                stats.illegitimate_declines += 1;
                            }
                        }
                    }
                }
                return Err(reason);
            }
        }
        if let Some(stats) = stats {
            stats.served_by_core += 1;
            stats.max_tokens = stats.max_tokens.max(expected.len());
        }
        let detail = |kind: &'static str, detail: String, first_diff: Option<usize>| {
            Ok(Some(Diff {
                kind,
                detail,
                core_ids: out.ids.clone(),
                backend_ids: expected_ids.to_vec(),
                first_diff,
            }))
        };
        if let Some(i) = first_difference(&out.ids, expected_ids) {
            return detail(
                "ids",
                format!(
                    "core {} vs backend {}",
                    excerpt(&out.ids, i),
                    excerpt(expected_ids, i)
                ),
                Some(i),
            );
        }
        if let Some(i) = first_difference(&out.offsets, expected.get_offsets()) {
            return detail(
                "offsets",
                format!(
                    "core {} vs backend {}",
                    excerpt(&out.offsets, i),
                    excerpt(expected.get_offsets(), i)
                ),
                Some(i),
            );
        }
        if let Some(i) = first_difference(&out.special_mask, expected.get_special_tokens_mask()) {
            return detail(
                "mask",
                format!(
                    "core {} vs backend {}",
                    excerpt(&out.special_mask, i),
                    excerpt(expected.get_special_tokens_mask(), i)
                ),
                Some(i),
            );
        }
        let words: Vec<Option<u32>> = out.words.iter().map(|&w| Some(w)).collect();
        if let Some(i) = first_difference(&words, expected.get_word_ids()) {
            return detail(
                "words",
                format!(
                    "core {} vs backend {}",
                    excerpt(&words, i),
                    excerpt(expected.get_word_ids(), i)
                ),
                Some(i),
            );
        }
        let tokens: Vec<&str> = out
            .ids
            .iter()
            .map(|&id| self.core.token(id).unwrap_or("<missing>"))
            .collect();
        let expected_tokens: Vec<&str> = expected.get_tokens().iter().map(String::as_str).collect();
        if let Some(i) = first_difference(&tokens, &expected_tokens) {
            return detail(
                "tokens",
                format!(
                    "core {} vs backend {}",
                    excerpt(&tokens, i),
                    excerpt(&expected_tokens, i)
                ),
                Some(i),
            );
        }
        let count = self
            .core
            .count(text, self.combo.recognize ^ self.inject)
            .expect("count declines exactly when encode declines");
        if count != expected.len() {
            return detail(
                "count",
                format!("core count {count} vs backend {}", expected.len()),
                None,
            );
        }
        let fast = self
            .oracle
            .encode_fast(text, self.combo.add_special)
            .expect("backend encode_fast")
            .len();
        if fast != expected.len() {
            return detail(
                "backend_fast_count",
                format!("backend encode_fast {fast} vs encode {}", expected.len()),
                None,
            );
        }
        Ok(None)
    }

    /// The production path: the `Verify` engine must neither fail nor differ from the oracle.
    fn compare_engine(
        &self,
        text: &str,
        expected: &Encoding,
    ) -> Option<(&'static str, String, Vec<u32>, Option<usize>)> {
        let encoded = match self.verify.encode(text) {
            Ok(encoded) => encoded,
            Err(Error::CoreMismatch(what)) => {
                return Some((
                    "engine_verify_mismatch",
                    format!("Engine(Verify) reported a core mismatch on {what}"),
                    Vec::new(),
                    None,
                ))
            }
            Err(error) => {
                return Some((
                    "engine_error",
                    format!("Engine(Verify) failed: {error}"),
                    Vec::new(),
                    None,
                ))
            }
        };
        if let Some(i) = first_difference(&encoded.ids, expected.get_ids()) {
            return Some((
                "engine_ids",
                format!(
                    "engine {} vs backend {}",
                    excerpt(&encoded.ids, i),
                    excerpt(expected.get_ids(), i)
                ),
                encoded.ids.clone(),
                Some(i),
            ));
        }
        if let Some(i) = first_difference(&encoded.offsets, expected.get_offsets()) {
            return Some((
                "engine_offsets",
                format!(
                    "engine {} vs backend {}",
                    excerpt(&encoded.offsets, i),
                    excerpt(expected.get_offsets(), i)
                ),
                encoded.ids.clone(),
                Some(i),
            ));
        }
        if let Some(i) = first_difference(
            &encoded.special_tokens_mask,
            expected.get_special_tokens_mask(),
        ) {
            return Some((
                "engine_mask",
                format!(
                    "engine {} vs backend {}",
                    excerpt(&encoded.special_tokens_mask, i),
                    excerpt(expected.get_special_tokens_mask(), i)
                ),
                encoded.ids.clone(),
                Some(i),
            ));
        }
        if let Some(i) = first_difference(&encoded.tokens, expected.get_tokens()) {
            return Some((
                "engine_tokens",
                format!(
                    "engine {} vs backend {}",
                    excerpt(&encoded.tokens, i),
                    excerpt(expected.get_tokens(), i)
                ),
                encoded.ids.clone(),
                Some(i),
            ));
        }
        if encoded.count != expected.len() {
            return Some((
                "engine_encode_count",
                format!(
                    "engine count {} vs backend {}",
                    encoded.count,
                    expected.len()
                ),
                encoded.ids.clone(),
                None,
            ));
        }
        match self.verify.count(text) {
            Ok(count) if count == expected.len() => None,
            Ok(count) => Some((
                "engine_count",
                format!("Engine(Verify).count {count} vs backend {}", expected.len()),
                encoded.ids,
                None,
            )),
            Err(error) => Some((
                "engine_count_error",
                format!("Engine(Verify).count failed: {error}"),
                encoded.ids,
                None,
            )),
        }
    }

    /// Everything for one input: core vs oracle, then the engines.
    fn check(
        &self,
        text: &str,
        category: Category,
        index: u64,
        out: &mut Output,
        stats: &mut Stats,
        engine: bool,
    ) -> Option<Mismatch> {
        stats.inputs += 1;
        stats.bytes += text.len() as u64;
        stats.by_category[category as usize] += 1;
        let make = |kind, detail, core_ids, backend_ids, first_diff| Mismatch {
            kind,
            category,
            index,
            text: text.to_owned(),
            core_ids,
            backend_ids,
            first_diff,
            detail,
            minimal: None,
        };
        let expected = self.oracle_encode(text);
        if let Ok(Some(diff)) = self.compare_core(text, &expected, out, Some(stats)) {
            return Some(make(
                diff.kind,
                diff.detail,
                diff.core_ids,
                diff.backend_ids,
                diff.first_diff,
            ));
        }
        if engine {
            stats.engine_checked += 1;
            if let Some((kind, detail, ids, first)) = self.compare_engine(text, &expected) {
                return Some(make(kind, detail, ids, expected.get_ids().to_vec(), first));
            }
            if category == Category::Corpus {
                stats.backend_engine_checked += 1;
                match self.backend.encode(text) {
                    Ok(encoded)
                        if encoded.ids == expected.get_ids()
                            && encoded.offsets == expected.get_offsets()
                            && self.backend.count(text).ok() == Some(expected.len()) => {}
                    Ok(encoded) => {
                        return Some(make(
                            "backend_engine",
                            "Engine(Backend) differs from the oracle tokenizer".to_owned(),
                            encoded.ids,
                            expected.get_ids().to_vec(),
                            None,
                        ))
                    }
                    Err(error) => {
                        return Some(make(
                            "backend_engine_error",
                            format!("Engine(Backend) failed: {error}"),
                            Vec::new(),
                            expected.get_ids().to_vec(),
                            None,
                        ))
                    }
                }
            }
        }
        None
    }

    /// The kind of core-vs-oracle mismatch on `text`, if any (used while shrinking).
    fn core_mismatch_kind(&self, text: &str, out: &mut Output) -> Option<&'static str> {
        match self.compare_core(text, &self.oracle_encode(text), out, None) {
            Ok(Some(diff)) => Some(diff.kind),
            _ => None,
        }
    }

    /// Delete scalars while the same kind of mismatch persists (ddmin-style chunks first, then
    /// single scalars), bounded by a number of probes.
    fn shrink(&self, mismatch: &Mismatch) -> Mismatch {
        let mut out = Output::default();
        let mut chars: Vec<char> = mismatch.text.chars().collect();
        let kind = mismatch.kind;
        let mut budget = 200_000usize;
        let mut size = (chars.len() / 2).max(1);
        while budget > 0 && !chars.is_empty() {
            let mut progressed = false;
            let mut start = 0;
            while start < chars.len() && budget > 0 {
                let end = (start + size).min(chars.len());
                let candidate: Vec<char> = chars[..start]
                    .iter()
                    .chain(&chars[end..])
                    .copied()
                    .collect();
                let text: String = candidate.iter().collect();
                budget -= 1;
                if self.core_mismatch_kind(&text, &mut out) == Some(kind) {
                    chars = candidate;
                    progressed = true;
                } else {
                    start = end;
                }
            }
            if !progressed {
                if size == 1 {
                    break;
                }
                size = (size / 2).max(1);
            }
        }
        let text: String = chars.iter().collect();
        let (detail, core_ids, backend_ids, first_diff) =
            match self.compare_core(&text, &self.oracle_encode(&text), &mut out, None) {
                Ok(Some(diff)) => (
                    diff.detail,
                    diff.core_ids,
                    diff.backend_ids,
                    diff.first_diff,
                ),
                _ => (
                    "shrunk input no longer mismatches (nondeterministic?)".to_owned(),
                    Vec::new(),
                    Vec::new(),
                    None,
                ),
            };
        Mismatch {
            kind,
            category: mismatch.category,
            index: mismatch.index,
            text,
            core_ids,
            backend_ids,
            first_diff,
            detail,
            minimal: None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------------------------

/// A Rust string literal for `text`: printable ASCII as is, everything else escaped, so the
/// exact input can be pasted into a test.
fn literal(text: &str) -> String {
    let mut s = String::with_capacity(text.len() + 2);
    s.push('"');
    for c in text.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            ' '..='~' => s.push(c),
            _ => s.push_str(&format!("\\u{{{:x}}}", c as u32)),
        }
    }
    s.push('"');
    s
}

fn ids_text(ids: &[u32]) -> String {
    if ids.len() <= 256 {
        format!("{ids:?}")
    } else {
        format!("{:?} .. ({} ids)", &ids[..256], ids.len())
    }
}

fn print_mismatch(mismatch: &Mismatch, heading: &str) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(
        out,
        "{heading}: kind={} category={} index={} bytes={}",
        mismatch.kind,
        mismatch.category.name(),
        mismatch.index,
        mismatch.text.len()
    );
    let _ = writeln!(out, "  input: {}", literal(&mismatch.text));
    let _ = writeln!(out, "  core ids:    {}", ids_text(&mismatch.core_ids));
    let _ = writeln!(out, "  backend ids: {}", ids_text(&mismatch.backend_ids));
    let _ = writeln!(
        out,
        "  first differing index: {}",
        mismatch
            .first_diff
            .map_or("n/a".to_owned(), |i| i.to_string())
    );
    let _ = writeln!(out, "  detail: {}", mismatch.detail);
}

fn mismatch_json(m: &Mismatch) -> Value {
    json!({
        "kind": m.kind,
        "category": m.category.name(),
        "index": m.index,
        "input": literal(&m.text),
        "input_bytes": m.text.len(),
        "core_ids": m.core_ids,
        "backend_ids": m.backend_ids,
        "first_differing_index": m.first_diff,
        "detail": m.detail,
        "minimal": m.minimal.as_ref().map(|min| json!({
            "input": literal(&min.text),
            "input_bytes": min.text.len(),
            "core_ids": min.core_ids,
            "backend_ids": min.backend_ids,
            "first_differing_index": min.first_diff,
            "detail": min.detail,
        })),
    })
}

fn format_count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------------------------

struct ManifestArtifact {
    name: String,
    file: String,
    sha256: String,
}

fn manifest(root: &Path) -> anyhow::Result<Vec<ManifestArtifact>> {
    let value: Value = serde_json::from_slice(&fs::read(root.join("models/MANIFEST.json"))?)?;
    let artifacts = value["artifacts"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("manifest has no artifacts"))?;
    Ok(artifacts
        .iter()
        .map(|a| ManifestArtifact {
            name: a["name"].as_str().unwrap_or("?").to_owned(),
            file: a["file"].as_str().unwrap_or("?").to_owned(),
            sha256: a["sha256"].as_str().unwrap_or("").to_owned(),
        })
        .collect())
}

fn added_tokens(json: &[u8]) -> Vec<String> {
    let value: Value = serde_json::from_slice(json).unwrap_or(Value::Null);
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

fn run_combo(
    args: &Args,
    inputs: &Inputs,
    harness: &Harness<'_>,
    label: &str,
) -> (Stats, Vec<Mismatch>, Duration) {
    let total = inputs.total();
    let next = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let bytes = AtomicU64::new(0);
    let core_served = AtomicU64::new(0);
    let mismatch_count = AtomicU64::new(0);
    let kept: Mutex<Vec<Mismatch>> = Mutex::new(Vec::new());
    let started = Instant::now();
    const CHUNK: u64 = 256;

    let (stats, elapsed) = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..args.threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut stats = Stats::default();
                    let mut out = Output::default();
                    let mut text = String::new();
                    loop {
                        let start = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if start >= total {
                            break;
                        }
                        let end = (start + CHUNK).min(total);
                        let mut chunk_bytes = 0u64;
                        let mut chunk_served = 0u64;
                        for index in start..end {
                            let category = inputs.get(index, &mut text);
                            let before_served = stats.served_by_core;
                            let engine = index.is_multiple_of(args.engine_stride);
                            if let Some(mismatch) =
                                harness.check(&text, category, index, &mut out, &mut stats, engine)
                            {
                                stats.mismatches += 1;
                                stats.bump(mismatch.kind, 1);
                                let n = mismatch_count.fetch_add(1, Ordering::Relaxed);
                                if (n as usize) < args.max_mismatches {
                                    print_mismatch(
                                        &mismatch,
                                        &format!("MISMATCH #{} {label}", n + 1),
                                    );
                                    kept.lock().unwrap().push(mismatch);
                                }
                            }
                            chunk_bytes += text.len() as u64;
                            chunk_served += stats.served_by_core - before_served;
                        }
                        done.fetch_add(end - start, Ordering::Relaxed);
                        bytes.fetch_add(chunk_bytes, Ordering::Relaxed);
                        core_served.fetch_add(chunk_served, Ordering::Relaxed);
                    }
                    stats
                })
            })
            .collect();
        let mut last_report = Instant::now();
        loop {
            std::thread::sleep(Duration::from_millis(200));
            let finished = done.load(Ordering::Relaxed);
            if finished >= total || workers.iter().all(|w| w.is_finished()) {
                break;
            }
            if last_report.elapsed() >= Duration::from_secs(20) {
                last_report = Instant::now();
                let secs = started.elapsed().as_secs_f64();
                let served = core_served.load(Ordering::Relaxed);
                eprintln!(
                    "  [{label}] {} / {} inputs ({:.0}%), {:.0} inputs/s, {:.1} MB/s, core served {:.1}%, mismatches {}, {:.0}s",
                    format_count(finished),
                    format_count(total),
                    100.0 * finished as f64 / total as f64,
                    finished as f64 / secs,
                    bytes.load(Ordering::Relaxed) as f64 / secs / 1e6,
                    100.0 * served as f64 / finished.max(1) as f64,
                    mismatch_count.load(Ordering::Relaxed),
                    secs
                );
            }
        }
        let mut stats = Stats::default();
        for worker in workers {
            stats.merge(&worker.join().expect("worker panicked"));
        }
        (stats, started.elapsed())
    });
    let mismatches = kept.into_inner().unwrap();
    (stats, mismatches, elapsed)
}

fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if cfg!(debug_assertions) {
        eprintln!("warning: debug build; use `cargo run --release --example core_soak`");
    }
    eprintln!(
        "core_soak: seed {} ({:#x}), {} threads, {} random inputs per combination, engine stride {}, scalar stride {}",
        args.seed, args.seed, args.threads, format_count(args.random), args.engine_stride, args.scalar_stride
    );
    let mut report = json!({
        "schema_version": 1,
        "seed": args.seed,
        "threads": args.threads,
        "random_inputs": args.random,
        "engine_stride": args.engine_stride,
        "scalar_stride": args.scalar_stride,
        "backend": tokforge::work::BACKEND,
        "engine_version": tokforge::VERSION,
        "artifacts": [],
    });
    let mut total_mismatches = 0u64;
    let mut total_inputs = 0u64;
    let overall = Instant::now();

    for entry in manifest(&root)? {
        if !args.artifacts.is_empty() && !args.artifacts.iter().any(|a| entry.name.contains(a)) {
            continue;
        }
        let path = root.join("models").join(&entry.file);
        let Ok(json) = fs::read(&path) else {
            eprintln!(
                "SKIPPED {}: {} not present; run `python -I models/fetch.py`",
                entry.name,
                path.display()
            );
            continue;
        };
        let sha256 = format!("{:x}", Sha256::digest(&json));
        if !entry.sha256.is_empty() && sha256 != entry.sha256 {
            anyhow::bail!(
                "{}: SHA-256 {sha256} does not match the manifest ({})",
                entry.name,
                entry.sha256
            );
        }
        let core = match ByteLevelBpe::from_json(&json) {
            Ok(core) => core,
            Err(reason) => {
                eprintln!(
                    "SKIPPED {}: core does not support it ({reason})",
                    entry.name
                );
                continue;
            }
        };
        eprintln!(
            "\n=== {} ({}; {}, sha256 {}) ===",
            entry.name,
            entry.file,
            core.describe(),
            &sha256[..12]
        );
        let added = added_tokens(&json);
        let inputs = Inputs::build(&root, &args, &added)?;
        let per_combo = inputs.total();
        eprintln!(
            "  {} inputs per combination: {}",
            format_count(per_combo),
            Category::ALL
                .iter()
                .map(|&c| format!("{} {}", c.name(), format_count(inputs.count(c))))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut artifact_report = json!({
            "name": entry.name,
            "file": entry.file,
            "sha256": sha256,
            "core": core.describe(),
            "vocab_size": core.vocab_size(),
            "added_tokens": added,
            "corpus_files": inputs.corpus_names.clone(),
            "inputs_per_combination": per_combo,
            "inputs_by_category": Category::ALL.iter().map(|&c| json!({"category": c.name(), "inputs": inputs.count(c)})).collect::<Vec<_>>(),
            "combinations": [],
        });

        for combo in Combo::ALL {
            let label = format!(
                "{} {}{}",
                entry.name,
                combo.label(),
                if args.self_test { " [self-test]" } else { "" }
            );
            let mut oracle = Tokenizer::from_bytes(&json).map_err(|e| anyhow::anyhow!("{e}"))?;
            oracle
                .with_truncation(None)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            oracle.with_padding(None);
            oracle.set_encode_special_tokens(!combo.recognize);
            let verify = Engine::from_bytes(&json, combo.options(CoreMode::Verify))?;
            let backend = Engine::from_bytes(&json, combo.options(CoreMode::Backend))?;
            anyhow::ensure!(
                verify.artifact().core.starts_with("bytelevel ("),
                "{label}: the engine did not select the core: {}",
                verify.artifact().core
            );
            anyhow::ensure!(backend.artifact().core == BACKEND_CORE);
            anyhow::ensure!(
                core.vocab_size() == oracle.get_vocab_size(true),
                "{label}: vocab size {} vs backend {}",
                core.vocab_size(),
                oracle.get_vocab_size(true)
            );
            let harness = Harness {
                combo,
                core: &core,
                oracle,
                verify,
                backend,
                inject: args.self_test,
            };
            eprintln!(
                "  --- {label}: engine core {:?}",
                harness.verify.artifact().core
            );
            let (stats, mut mismatches, elapsed) = run_combo(&args, &inputs, &harness, &label);
            total_inputs += stats.inputs;
            total_mismatches += stats.mismatches;
            let secs = elapsed.as_secs_f64();
            eprintln!(
                "  {label}: {} inputs, {} bytes, {:.1}s, {:.0} inputs/s, {:.1} MB/s; core served {} ({:.2}%), declined NFC {} / contraction {} (illegitimate {}); engine checked {} (backend engine {}); mismatches {}{}",
                format_count(stats.inputs),
                format_count(stats.bytes),
                secs,
                stats.inputs as f64 / secs,
                stats.bytes as f64 / secs / 1e6,
                format_count(stats.served_by_core),
                100.0 * stats.served_by_core as f64 / stats.inputs.max(1) as f64,
                format_count(stats.declined_normalization),
                format_count(stats.declined_contraction),
                stats.illegitimate_declines,
                format_count(stats.engine_checked),
                format_count(stats.backend_engine_checked),
                stats.mismatches,
                if stats.mismatch_kinds.is_empty() {
                    String::new()
                } else {
                    format!(" ({:?})", stats.mismatch_kinds)
                }
            );

            // Shrink each distinct mismatch (by kind and text) to a minimal input.
            let mut seen = HashSet::new();
            let mut distinct_minimal = HashSet::new();
            for mismatch in &mut mismatches {
                if !seen.insert((mismatch.kind, mismatch.text.clone())) {
                    continue;
                }
                if mismatch.kind.starts_with("engine") || mismatch.kind.starts_with("backend") {
                    continue;
                }
                let minimal = harness.shrink(mismatch);
                if distinct_minimal.insert((minimal.kind, minimal.text.clone())) {
                    print_mismatch(&minimal, &format!("MINIMAL {label}"));
                }
                mismatch.minimal = Some(Box::new(minimal));
            }
            let combo_report = json!({
                "special_tokens": if combo.recognize { "recognize" } else { "literal" },
                "add_special_tokens": combo.add_special,
                "engine_core": harness.verify.artifact().core,
                "inputs": stats.inputs,
                "bytes": stats.bytes,
                "wall_seconds": secs,
                "inputs_per_second": stats.inputs as f64 / secs,
                "served_by_core": stats.served_by_core,
                "declined": {
                    "normalization": stats.declined_normalization,
                    "folded_contraction": stats.declined_contraction,
                    "illegitimate": stats.illegitimate_declines,
                },
                "engine_checked": stats.engine_checked,
                "backend_engine_checked": stats.backend_engine_checked,
                "max_tokens_in_one_input": stats.max_tokens,
                "mismatches": stats.mismatches,
                "mismatch_kinds": stats.mismatch_kinds.iter().map(|(k, n)| json!({"kind": k, "count": n})).collect::<Vec<_>>(),
                "mismatch_examples": mismatches.iter().map(mismatch_json).collect::<Vec<_>>(),
                "distinct_minimal_inputs": distinct_minimal.len(),
            });
            artifact_report["combinations"]
                .as_array_mut()
                .unwrap()
                .push(combo_report);
            if let Some(path) = &args.output {
                // Written after every combination so an interrupted run still leaves a record.
                let mut partial = report.clone();
                partial["artifacts"]
                    .as_array_mut()
                    .unwrap()
                    .push(artifact_report.clone());
                partial["partial"] = json!(true);
                fs::write(path, serde_json::to_string_pretty(&partial)?)?;
            }
        }
        report["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(artifact_report);
    }

    report["total_inputs"] = json!(total_inputs);
    report["total_mismatches"] = json!(total_mismatches);
    report["wall_seconds"] = json!(overall.elapsed().as_secs_f64());
    if let Some(path) = &args.output {
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
        eprintln!("report written to {}", path.display());
    }
    eprintln!(
        "\ncore_soak: {} inputs compared in {:.0}s, {} mismatches",
        format_count(total_inputs),
        overall.elapsed().as_secs_f64(),
        total_mismatches
    );
    if total_inputs == 0 {
        eprintln!("no supported artifact was present; nothing was compared");
    }
    if args.self_test {
        eprintln!("self-test: the injected mode swap must produce mismatches and minimal inputs");
        if total_mismatches == 0 {
            std::process::exit(1);
        }
    } else if total_mismatches > 0 {
        std::process::exit(1);
    }
    Ok(())
}
