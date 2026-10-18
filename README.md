# Fast Tokenizer

Rust tokenization and document preparation project.

## Current status

Repository scaffold only. The CLI currently supports `tokenizer info` to verify
the build environment. Counting, encoding, decoding, batching, the HTTP API,
and token-aware chunking are the next implementation steps.

## Windows prerequisites

- Git and GitHub CLI (`gh`)
- Visual Studio C++ build tools and Windows SDK
- Stable Rust, Cargo, rustfmt, and Clippy (installed using rustup)
- Network access to Rust distribution servers, crates.io, and GitHub

On this machine, Git, GitHub CLI, Visual C++ tools, and the Windows SDK were
detected. Rust installation and GitHub repository creation could not be completed
from the restricted workspace because outbound connections were denied.

## Complete setup

Open PowerShell in this project folder in an environment with network access:

```powershell
.\scripts\setup.ps1
```

The script downloads the official Rust installer and verifies its published
SHA-256 checksum, installs the stable toolchain plus rustfmt and Clippy, fetches
dependencies, checks the build, and creates and pushes a **private** GitHub
repository using the active `gh` account. If `gh` requires authentication, it
stops with the login command; run that command and rerun setup. It does not
replace an existing remote or force-push.

To install and verify prerequisites without creating the GitHub repository:

```powershell
.\scripts\setup.ps1 -SkipGitHub
```

After Rust is installed, open a fresh terminal if Cargo is not yet on PATH.

```powershell
cargo run -- info
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

Use `C:\Users\HP\Desktop\fast-tokenizer` as the workspace folder for this
project. Changing a command's working directory does not change the app's
selected workspace or its permission scope.

## Intended implementation

1. Shared tokenizer engine with explicit special-token behavior and artifact identity.
2. CLI count, encode, decode, truncate, and chunk commands.
3. HTTP endpoints backed by a bounded CPU pool.
4. Streaming JSONL processing with bounded queues and cancellation.
5. Unicode-safe chunks with verified token budgets and original source offsets.
6. Correctness fixtures and reproducible throughput/latency benchmarks.
