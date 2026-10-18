use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "tokenizer", version, about = "Rust tokenization engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Report scaffold status and verify the executable can run.
    Info,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info => {
            let info = serde_json::json!({
                "name": "fast-tokenizer",
                "version": fast_tokenizer::VERSION,
                "status": "scaffold",
            });
            println!("{}", serde_json::to_string_pretty(&info)?);
        }
    }
    Ok(())
}
