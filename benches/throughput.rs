//! Dependency-free benchmark runner; use identical hardware/toolchain for comparisons.
use std::{
    hint::black_box,
    io::Cursor,
    sync::Arc,
    time::{Duration, Instant},
};

use fast_tokenizer::{
    batch::{self, BatchConfig},
    work::{Cancellation, CpuPool},
    EncodeOptions, Engine,
};
use serde_json::{json, Value};

fn engine() -> Engine {
    Engine::from_bytes(
        include_bytes!("../tests/fixtures/wordpiece.json"),
        EncodeOptions::default(),
    )
    .unwrap()
}

fn measure(name: &str, bytes: usize, mut run: impl FnMut()) -> Value {
    run(); // Warm tokenizer caches and thread pools before timing.
    let start = Instant::now();
    let mut iterations = 0;
    loop {
        run();
        iterations += 1;
        if start.elapsed() >= Duration::from_millis(300) {
            break;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    json!({"name": name, "input_bytes": bytes, "iterations": iterations,
        "seconds": seconds, "ms_per_iteration": seconds * 1000.0 / iterations as f64,
        "mib_per_second": bytes as f64 * iterations as f64 / seconds / 1_048_576.0})
}

fn main() -> anyhow::Result<()> {
    // Cargo test --all-targets also launches harness-free bench binaries. Keep
    // debug builds out of performance reports and out of routine test timing.
    if cfg!(debug_assertions) || std::env::args().any(|arg| arg == "--test") {
        return Ok(());
    }
    let engine = engine();
    let text = "hello café 你好 world! ".repeat(1300); // 33,800 bytes, deterministic fixture vocabulary.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let records: Arc<[u8]> = ("{\"op\":\"encode\",\"text\":\"hello café 你好 world!\"}\n"
        .repeat(2000))
    .into_bytes()
    .into();
    let mut results = vec![
        measure("count", text.len(), || {
            black_box(engine.count(black_box(&text)).unwrap());
        }),
        measure("encode", text.len(), || {
            black_box(engine.encode(black_box(&text)).unwrap());
        }),
        measure("truncate_128", text.len(), || {
            black_box(engine.truncate(black_box(&text), 128).unwrap());
        }),
        measure("chunk_128", text.len(), || {
            black_box(engine.chunk(black_box(&text), 128).unwrap());
        }),
    ];
    results.push(measure("jsonl_encode_2000", records.len(), || {
        // A fresh run needs a fresh cancellation scope; pool startup is included.
        let cancel = Cancellation::default();
        let pool = CpuPool::new(self::engine(), 4, 16, cancel.clone()).unwrap();
        let summary = runtime
            .block_on(batch::process(
                Cursor::new(records.clone()),
                tokio::io::sink(),
                pool,
                BatchConfig::default(),
                cancel,
            ))
            .unwrap();
        assert_eq!(summary.records, 2000);
        assert_eq!(summary.errors, 0);
        black_box(summary);
    }));
    let output = json!({"fixture_sha256": engine.artifact().sha256,
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "cpu": std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_default(),
        "profile": "release", "results": results});
    let bytes = serde_json::to_vec_pretty(&output)?;
    let args: Vec<String> = std::env::args().collect();
    if let Some(index) = args.iter().position(|arg| arg == "--output") {
        std::fs::write(
            args.get(index + 1)
                .ok_or_else(|| anyhow::anyhow!("--output needs a path"))?,
            &bytes,
        )?;
    }
    println!("{}", String::from_utf8(bytes)?);
    Ok(())
}
