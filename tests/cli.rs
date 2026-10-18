use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

use serde_json::Value;

fn run(command: &str, args: &[&str], stdin: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tokenizer"))
        .args([
            command,
            "--tokenizer",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/wordpiece.json"),
        ])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = child.stdin.take().unwrap();
    if let Some(input) = stdin {
        pipe.write_all(input.as_bytes()).unwrap();
    }
    drop(pipe);
    child.wait_with_output().unwrap()
}

fn json(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn count_encode_decode_and_info() {
    assert_eq!(
        json(run("count", &["--text", "hello world!"], None))["count"],
        3
    );
    let encoded = json(run("encode", &[], Some("hello café 你好!")));
    assert_eq!(encoded["encoding"]["ids"], serde_json::json!([4, 7, 8, 6]));
    assert_eq!(
        json(run("decode", &[], Some("[4,5,6]")))["text"],
        "hello world!"
    );
    assert_eq!(json(run("info", &[], None))["artifact"]["vocab_size"], 10);
}

#[test]
fn file_input_and_budget_commands_preserve_source() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let source = "hello\r\nworld!  café";
    file.write_all(source.as_bytes()).unwrap();
    let result = json(run(
        "chunk",
        &["--file", file.path().to_str().unwrap(), "--max-tokens", "2"],
        None,
    ));
    let chunks = result["chunks"].as_array().unwrap();
    let reconstructed: String = chunks.iter().map(|c| c["text"].as_str().unwrap()).collect();
    assert_eq!(reconstructed, source);
    assert!(chunks.iter().all(|c| c["count"].as_u64().unwrap() <= 2));
    let prefix = json(run(
        "truncate",
        &["--text", "hello world", "--max-tokens", "1"],
        None,
    ));
    assert_eq!(prefix["truncated"], true);
    assert_eq!(prefix["result"]["count"], 1);
    assert!("hello world".starts_with(prefix["result"]["text"].as_str().unwrap()));
}

#[test]
fn invalid_inputs_fail_without_success_json() {
    for (command, args) in [
        ("decode", vec!["--text", "[999]"]),
        ("decode", vec!["--text", "[-1]"]),
        ("decode", vec!["--text", "not json"]),
        ("count", vec!["--text", "hello", "--file", "missing.txt"]),
        ("count", vec!["--file", "missing.txt"]),
        ("chunk", vec!["--text", "hello", "--max-tokens", "0"]),
        (
            "truncate",
            vec![
                "--text",
                "hello",
                "--max-tokens",
                "1",
                "--add-special-tokens",
            ],
        ),
    ] {
        let output = run(command, &args, None);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}
