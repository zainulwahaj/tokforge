use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

use serde_json::Value;

fn run(command: &str, args: &[&str], stdin: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tokforge"))
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
fn jsonl_streams_results_and_reports_partial_record_failures() {
    let output = run(
        "jsonl",
        &["--workers", "2", "--queue-capacity", "1"],
        Some("{\"op\":\"count\",\"text\":\"hello world!\"}\n{\"op\":\"decode\",\"ids\":[4,5,6]}\n"),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["result"]["count"], 3);
    assert_eq!(rows[1]["result"]["text"], "hello world!");
    let output = run(
        "jsonl",
        &[],
        Some("bad json\n{\"op\":\"count\",\"text\":\"hello\"}"),
    );
    assert!(!output.status.success());
    let rows: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows[0]["error"]["code"], "invalid_request");
    assert_eq!(rows[1]["result"]["count"], 1);
    assert!(String::from_utf8_lossy(&output.stderr).contains("2 records; 1 errors"));
}

#[test]
fn jsonl_file_output_never_overwrites_existing_files_or_input() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("input.jsonl");
    let destination = directory.path().join("output.jsonl");
    let input = "{\"op\":\"count\",\"text\":\"hello\"}\n";
    std::fs::write(&source, input).unwrap();
    let output = run(
        "jsonl",
        &[
            "--file",
            source.to_str().unwrap(),
            "--output",
            destination.to_str().unwrap(),
        ],
        None,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let saved = std::fs::read(&destination).unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&saved).unwrap()["result"]["count"],
        1
    );
    for existing in [&source, &destination] {
        assert!(!run(
            "jsonl",
            &[
                "--file",
                source.to_str().unwrap(),
                "--output",
                existing.to_str().unwrap()
            ],
            None
        )
        .status
        .success());
    }
    assert_eq!(std::fs::read_to_string(&source).unwrap(), input);
    assert_eq!(std::fs::read(&destination).unwrap(), saved);
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

#[test]
fn ids_only_encode_and_info_name_the_schema_and_backend() {
    let lean = json(run(
        "encode",
        &["--ids-only", "--text", "hello world!"],
        None,
    ));
    assert_eq!(lean["encoding"]["ids"], serde_json::json!([4, 5, 6]));
    assert_eq!(lean["encoding"]["count"], 3);
    assert!(lean["encoding"].get("tokens").is_none());
    assert_eq!(lean["schema_version"], 1);
    assert!(lean["backend"].as_str().unwrap().starts_with("tokenizers "));
    let full = json(run("encode", &["--text", "hello world!"], None));
    assert_eq!(full["encoding"]["tokens"].as_array().unwrap().len(), 3);
    let info = json(run("info", &[], None));
    assert_eq!(info["schema_version"], 1);
    assert_eq!(info["artifact"]["vocab_size"], 10);
    assert!(info["backend"].as_str().unwrap().starts_with("tokenizers "));
}

#[test]
fn core_flag_is_reported_and_backend_can_be_forced() {
    let auto = json(run("info", &[], None));
    // The synthetic WordPiece fixture is never served by the custom core.
    assert_eq!(auto["artifact"]["core"], "backend");
    // Auto is the default since the differential soak passed clean.
    assert_eq!(auto["options"]["core"], "auto");
    let forced = json(run(
        "count",
        &["--core", "backend", "--text", "hello world"],
        None,
    ));
    assert_eq!(forced["count"], 2);
    assert_eq!(forced["options"]["core"], "backend");
    assert_eq!(forced["artifact"]["core"], "backend");
    let verify = json(run(
        "count",
        &["--core", "verify", "--text", "hello world"],
        None,
    ));
    assert_eq!(verify["count"], 2);
    assert_eq!(verify["options"]["core"], "verify");
}
