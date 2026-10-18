use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use fast_tokenizer::{EncodeOptions, Engine, SpecialTokens};
use serde_json::{json, Value};

#[derive(Parser)]
#[command(name = "tokenizer", version, about = "Rust tokenization engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report version and, optionally, tokenizer identity.
    Info {
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },
    /// Count tokens in UTF-8 text.
    Count(TextCommand),
    /// Encode text into IDs, tokens, and source byte offsets.
    Encode(TextCommand),
    /// Decode a JSON array of unsigned token IDs.
    Decode {
        #[arg(long)]
        tokenizer: PathBuf,
        #[command(flatten)]
        input: Input,
        /// Remove registered special tokens from the decoded output.
        #[arg(long)]
        skip_special_tokens: bool,
    },
    /// Keep a source prefix that fits the token budget.
    Truncate(BudgetCommand),
    /// Split all source text into chunks with verified token budgets.
    Chunk(BudgetCommand),
}

#[derive(Args)]
struct TextCommand {
    #[command(flatten)]
    model: Model,
    #[command(flatten)]
    input: Input,
}

#[derive(Args)]
struct BudgetCommand {
    #[command(flatten)]
    text: TextCommand,
    #[arg(long)]
    max_tokens: usize,
}

#[derive(Clone, Copy, ValueEnum)]
enum SpecialPolicy {
    Literal,
    Recognize,
}

#[derive(Args)]
struct Model {
    /// Local Hugging Face tokenizer.json artifact.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Include the artifact's post-processor tokens, such as BOS/EOS.
    #[arg(long)]
    add_special_tokens: bool,
    /// How special-token spellings inside input text are handled.
    #[arg(long, value_enum, default_value = "literal")]
    special_tokens: SpecialPolicy,
}

impl Model {
    fn load(&self) -> Result<Engine> {
        let options = EncodeOptions {
            add_special_tokens: self.add_special_tokens,
            special_tokens: match self.special_tokens {
                SpecialPolicy::Literal => SpecialTokens::Literal,
                SpecialPolicy::Recognize => SpecialTokens::Recognize,
            },
        };
        Engine::from_file(&self.tokenizer, options)
            .with_context(|| format!("loading {}", self.tokenizer.display()))
    }
}

#[derive(Args)]
struct Input {
    /// Text (or a JSON ID array for decode). Omit to read stdin.
    #[arg(long, conflicts_with = "file", allow_hyphen_values = true)]
    text: Option<String>,
    /// Read a UTF-8 file, preserving its whitespace and newlines.
    #[arg(long)]
    file: Option<PathBuf>,
}

impl Input {
    fn read(&self) -> Result<String> {
        if let Some(text) = &self.text {
            return Ok(text.clone());
        }
        if let Some(path) = &self.file {
            return fs::read_to_string(path).with_context(|| format!("reading {}", path.display()));
        }
        let mut text = String::new();
        io::stdin()
            .read_to_string(&mut text)
            .context("reading UTF-8 stdin")?;
        Ok(text)
    }
}

fn run(cli: Cli) -> Result<Value> {
    match cli.command {
        Command::Info { tokenizer } => {
            let mut info = json!({
                "name": "fast-tokenizer", "version": fast_tokenizer::VERSION,
                "status": "cli", "commands": ["info", "count", "encode", "decode", "truncate", "chunk"],
            });
            if let Some(path) = tokenizer {
                let engine = Engine::from_file(&path, EncodeOptions::default())
                    .with_context(|| format!("loading {}", path.display()))?;
                info["artifact"] = json!(engine.artifact());
                info["options"] = json!(engine.options());
            }
            Ok(info)
        }
        Command::Count(args) => {
            let engine = args.model.load()?;
            Ok(
                json!({"artifact": engine.artifact(), "options": engine.options(),
                "count": engine.count(&args.input.read()?)?}),
            )
        }
        Command::Encode(args) => {
            let engine = args.model.load()?;
            Ok(
                json!({"artifact": engine.artifact(), "options": engine.options(),
                "encoding": engine.encode(&args.input.read()?)?}),
            )
        }
        Command::Decode {
            tokenizer,
            input,
            skip_special_tokens,
        } => {
            let engine = Engine::from_file(&tokenizer, EncodeOptions::default())
                .with_context(|| format!("loading {}", tokenizer.display()))?;
            let ids: Vec<u32> = serde_json::from_str(&input.read()?)
                .context("decode input must be a JSON array of unsigned 32-bit token IDs")?;
            Ok(
                json!({"artifact": engine.artifact(), "skip_special_tokens": skip_special_tokens,
                "text": engine.decode(&ids, skip_special_tokens)?}),
            )
        }
        Command::Truncate(args) => {
            let engine = args.text.model.load()?;
            let text = args.text.input.read()?;
            let result = engine.truncate(&text, args.max_tokens)?;
            Ok(
                json!({"artifact": engine.artifact(), "options": engine.options(),
                "max_tokens": args.max_tokens, "truncated": result.end_byte < text.len(),
                "result": result}),
            )
        }
        Command::Chunk(args) => {
            let engine = args.text.model.load()?;
            let text = args.text.input.read()?;
            Ok(
                json!({"artifact": engine.artifact(), "options": engine.options(),
                "max_tokens": args.max_tokens, "chunks": engine.chunk(&text, args.max_tokens)?}),
            )
        }
    }
}

fn main() -> Result<()> {
    let output = run(Cli::parse())?;
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer_pretty(&mut stdout, &output)?;
    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}
