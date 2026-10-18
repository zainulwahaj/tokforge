//! Expose the locked `tokenizers` crate version to the binary so every response and the info
//! endpoint can name the exact backend. Read from Cargo.lock at build time; no network, no
//! dependency, and a lock bump changes the reported version automatically.
use std::{env, fs, path::Path};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lock = Path::new(&manifest_dir).join("Cargo.lock");
    println!("cargo:rerun-if-changed=Cargo.lock");
    let version = fs::read_to_string(&lock)
        .ok()
        .and_then(|text| locked_version(&text, "tokenizers"))
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=TOKENIZERS_VERSION={version}");
}

/// The `version` line of the `[[package]]` block whose `name` is `name`.
fn locked_version(lock: &str, name: &str) -> Option<String> {
    let mut in_block = false;
    for line in lock.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            in_block = false;
        } else if line == format!("name = \"{name}\"") {
            in_block = true;
        } else if in_block {
            if let Some(version) = line.strip_prefix("version = ") {
                return Some(version.trim_matches('"').to_owned());
            }
        }
    }
    None
}
