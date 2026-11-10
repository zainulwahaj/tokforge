use std::{
    fs,
    io::{self, Read, Write},
    net::SocketAddr,
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::json;
use tokforge::{
    api::{self, ApiConfig},
    batch::{self, BatchConfig},
    work::{execute, Cancellation, CpuPool, Operation},
};
use tokforge::{CoreMode, EncodeOptions, Engine, SpecialTokens};

#[derive(Parser)]
#[command(
    name = "tokforge",
    version,
    about = "Fast, verified BPE tokenization engine"
)]
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
    Encode(EncodeCommand),
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
    /// Serve the HTTP API using a bounded CPU pool.
    Serve(ServeCommand),
    /// Process operation records as ordered JSONL from a file or stdin.
    Jsonl(JsonlCommand),
}

#[derive(Args)]
struct PoolArgs {
    #[command(flatten)]
    model: Model,
    /// Number of dedicated CPU worker threads.
    #[arg(long, default_value_t = 4, value_parser = positive)]
    workers: usize,
    /// Additional admitted jobs beyond the worker count (zero is allowed).
    #[arg(long, default_value_t = 16)]
    queue_capacity: usize,
}

impl PoolArgs {
    fn load(&self, cancel: Cancellation) -> Result<CpuPool> {
        CpuPool::new(
            self.model.load()?,
            self.workers,
            self.queue_capacity,
            cancel,
        )
    }
}

#[derive(Args)]
struct ServeCommand {
    #[command(flatten)]
    pool: PoolArgs,
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    #[arg(long, default_value_t = 1_048_576, value_parser = positive)]
    max_body_bytes: usize,
    #[arg(long, default_value_t = 30, value_parser = positive)]
    timeout_seconds: usize,
}

#[derive(Args)]
struct JsonlCommand {
    #[command(flatten)]
    pool: PoolArgs,
    /// Read JSONL from a UTF-8 file; omit for stdin.
    #[arg(long)]
    file: Option<PathBuf>,
    /// Create a new output file; omit for stdout. Existing files are rejected.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Maximum bytes in one record, excluding the newline.
    #[arg(long, default_value_t = 1_048_576, value_parser = positive)]
    max_line_bytes: usize,
}

fn positive(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| "expected a positive integer".to_owned())
}

#[derive(Args)]
struct TextCommand {
    #[command(flatten)]
    model: Model,
    #[command(flatten)]
    input: Input,
}

#[derive(Args)]
struct EncodeCommand {
    #[command(flatten)]
    text: TextCommand,
    /// Return only IDs and the count: no token strings, offsets, or mask.
    #[arg(long)]
    ids_only: bool,
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

#[derive(Clone, Copy, ValueEnum)]
enum CorePolicy {
    /// Custom byte-level BPE core when the artifact qualifies, otherwise the backend.
    Auto,
    /// Always the Hugging Face backend.
    Backend,
    /// Run both cores on every request and fail requests where they disagree.
    Verify,
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
    /// Which tokenization core serves encode and count: `auto` (the default) serves
    /// qualifying artifacts from the custom core, `verify` double-encodes with the backend.
    #[arg(long, value_enum, default_value = "auto")]
    core: CorePolicy,
}

impl Model {
    fn load(&self) -> Result<Engine> {
        let options = EncodeOptions {
            add_special_tokens: self.add_special_tokens,
            special_tokens: match self.special_tokens {
                SpecialPolicy::Literal => SpecialTokens::Literal,
                SpecialPolicy::Recognize => SpecialTokens::Recognize,
            },
            core: match self.core {
                CorePolicy::Auto => CoreMode::Auto,
                CorePolicy::Backend => CoreMode::Backend,
                CorePolicy::Verify => CoreMode::Verify,
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

fn run(cli: Cli) -> Result<Vec<u8>> {
    match cli.command {
        Command::Info { tokenizer } => {
            let mut info = json!({
                "name": "tokforge", "version": tokforge::VERSION,
                "schema_version": tokforge::work::SCHEMA_VERSION,
                "backend": tokforge::work::BACKEND,
                "status": "ready", "commands": ["info", "count", "encode", "decode", "truncate", "chunk", "serve", "jsonl"],
            });
            if let Some(path) = tokenizer {
                let engine = Engine::from_file(&path, EncodeOptions::default())
                    .with_context(|| format!("loading {}", path.display()))?;
                info["artifact"] = json!(engine.artifact());
                info["options"] = json!(engine.options());
            }
            Ok(serde_json::to_vec_pretty(&info)?)
        }
        Command::Count(args) => {
            let engine = args.model.load()?;
            Ok(execute(
                &engine,
                Operation::Count {
                    text: args.input.read()?,
                },
            )?)
        }
        Command::Encode(args) => {
            let engine = args.text.model.load()?;
            Ok(execute(
                &engine,
                Operation::Encode {
                    text: args.text.input.read()?,
                    ids_only: args.ids_only,
                },
            )?)
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
            Ok(execute(
                &engine,
                Operation::Decode {
                    ids,
                    skip_special_tokens,
                },
            )?)
        }
        Command::Truncate(args) => {
            let engine = args.text.model.load()?;
            Ok(execute(
                &engine,
                Operation::Truncate {
                    text: args.text.input.read()?,
                    max_tokens: args.max_tokens,
                },
            )?)
        }
        Command::Chunk(args) => {
            let engine = args.text.model.load()?;
            Ok(execute(
                &engine,
                Operation::Chunk {
                    text: args.text.input.read()?,
                    max_tokens: args.max_tokens,
                },
            )?)
        }
        Command::Serve(_) | Command::Jsonl(_) => {
            unreachable!("async commands are dispatched by main")
        }
    }
}

async fn serve(args: ServeCommand) -> Result<()> {
    let cancel = Cancellation::default();
    let _guard = cancel.guard();
    let pool = args.pool.load(cancel.clone())?;
    let config = ApiConfig {
        max_body_bytes: args.max_body_bytes,
        request_timeout: Duration::from_secs(args.timeout_seconds as u64),
    };
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    eprintln!("Listening on http://{}", listener.local_addr()?);
    axum::serve(listener, api::router(pool, config))
        .with_graceful_shutdown(async move {
            if let Err(error) = tokio::signal::ctrl_c().await {
                eprintln!("Could not listen for Ctrl+C: {error}");
            }
            cancel.cancel();
        })
        .await?;
    Ok(())
}

async fn jsonl(args: JsonlCommand) -> Result<()> {
    let cancel = Cancellation::default();
    let _guard = cancel.guard();
    let pool = args.pool.load(cancel.clone())?;
    let reader: Box<dyn tokio::io::AsyncBufRead + Unpin + Send> = if let Some(path) = args.file {
        Box::new(tokio::io::BufReader::new(
            tokio::fs::File::open(&path)
                .await
                .with_context(|| format!("opening {}", path.display()))?,
        ))
    } else {
        Box::new(tokio::io::BufReader::new(tokio::io::stdin()))
    };
    let writer: Box<dyn tokio::io::AsyncWrite + Unpin + Send> = if let Some(path) = args.output {
        Box::new(tokio::io::BufWriter::with_capacity(
            64 * 1024,
            tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .await
                .with_context(|| format!("creating new output {}", path.display()))?,
        ))
    } else {
        Box::new(tokio::io::BufWriter::with_capacity(
            64 * 1024,
            tokio::io::stdout(),
        ))
    };
    let signal_cancel = cancel.clone();
    let signal = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_cancel.cancel();
        }
    });
    let result = batch::process(
        reader,
        writer,
        pool,
        BatchConfig {
            max_line_bytes: args.max_line_bytes,
        },
        cancel,
    )
    .await;
    signal.abort();
    let summary = result?;
    eprintln!(
        "Processed {} records; {} errors",
        summary.records, summary.errors
    );
    anyhow::ensure!(
        summary.errors == 0,
        "JSONL contains failed records; see output for details"
    );
    Ok(())
}

fn run_async(future: impl std::future::Future<Output = Result<()>>) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(future);
    // Tokio stdin uses a blocking read that cannot be interrupted. Do not wait
    // for that read when a cancelled JSONL pipeline exits.
    runtime.shutdown_background();
    result
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let output = match cli.command {
        Command::Serve(args) => return run_async(serve(args)),
        Command::Jsonl(args) => return run_async(jsonl(args)),
        command => run(Cli { command })?,
    };
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    stdout.write_all(&output)?;
    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}
