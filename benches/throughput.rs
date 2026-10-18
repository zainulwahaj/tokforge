//! Dependency-free benchmark runner; use identical hardware/toolchain for comparisons.
//! Without flags it measures the committed synthetic fixture and the five original workloads.
//! `--tokenizer <tokenizer.json>` and `--corpus <text file>` target a real artifact and text and
//! add stage timings plus, when the artifact is supported, custom-core workloads. Every result
//! carries per-iteration p50/p95/p99 and heap allocation counts; `--output <path>` writes JSON.
//! `--http-load <concurrency>` also serves the API on a loopback port and measures end-to-end
//! request latency for `POST /v1/count` from that many keep-alive clients (`--requests` total).
use std::{
    alloc::{GlobalAlloc, Layout, System},
    future::IntoFuture,
    hint::black_box,
    io::Cursor,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use fast_tokenizer::{
    api::{router, ApiConfig},
    batch::{self, BatchConfig},
    bytelevel::{ByteLevelBpe, Output},
    work::{Cancellation, CpuPool},
    EncodeOptions, Engine,
};
use serde_json::{json, Value};
use tokenizers::{
    tokenizer::{
        Model, NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString,
        PreTokenizer,
    },
    Tokenizer,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Counts every heap allocation so a workload's allocations per iteration can be reported.
/// Counting costs two relaxed atomic increments per allocation, which is negligible next to the
/// allocation itself and identical for every workload being compared.
struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

struct Args {
    output: Option<String>,
    /// Path to a `tokenizer.json` artifact, for example one listed in `models/MANIFEST.json`.
    tokenizer: Option<String>,
    /// UTF-8 text file used as the count/encode/truncate/chunk input and, line by line, as the
    /// JSONL record texts.
    corpus: Option<String>,
    /// Concurrent keep-alive HTTP clients for the loopback load test; absent means no load test.
    http_load: Option<String>,
    /// Total requests for the load test.
    requests: Option<String>,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = Args {
        output: None,
        tokenizer: None,
        corpus: None,
        http_load: None,
        requests: None,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let slot = match arg.as_str() {
            "--output" => &mut args.output,
            "--tokenizer" => &mut args.tokenizer,
            "--corpus" => &mut args.corpus,
            "--http-load" => &mut args.http_load,
            "--requests" => &mut args.requests,
            "--bench" => continue, // appended by `cargo bench`
            other => anyhow::bail!(
                "unknown argument {other}; expected --output, --tokenizer, --corpus, --http-load, --requests"
            ),
        };
        *slot = Some(
            iter.next()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))?,
        );
    }
    Ok(args)
}

fn artifact_bytes(path: Option<&str>) -> anyhow::Result<Vec<u8>> {
    Ok(match path {
        Some(path) => std::fs::read(path)?,
        None => include_bytes!("../tests/fixtures/wordpiece.json").to_vec(),
    })
}

fn engine(path: Option<&str>) -> anyhow::Result<Engine> {
    Ok(Engine::from_bytes(
        &artifact_bytes(path)?,
        EncodeOptions::default(),
    )?)
}

/// 2,000 encode records. The synthetic record stays byte-identical to earlier reports; a corpus
/// cycles through its non-empty lines so record sizes reflect the real text.
fn records(corpus: Option<&str>) -> anyhow::Result<Arc<[u8]>> {
    let Some(text) = corpus else {
        return Ok(
            ("{\"op\":\"encode\",\"text\":\"hello café 你好 world!\"}\n".repeat(2000))
                .into_bytes()
                .into(),
        );
    };
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    anyhow::ensure!(!lines.is_empty(), "corpus has no non-empty lines");
    let mut out = Vec::new();
    for line in lines.iter().cycle().take(2000) {
        serde_json::to_writer(&mut out, &json!({"op": "encode", "text": line}))?;
        out.push(b'\n');
    }
    Ok(out.into())
}

fn percentile(sorted: &[Duration], p: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index].as_secs_f64() * 1000.0
}

/// Warm once, then repeat for at least 300 ms. Reports the mean (comparable with earlier
/// reports), per-iteration percentiles, and heap allocations per iteration.
fn measure(name: &str, bytes: usize, mut run: impl FnMut()) -> Value {
    run(); // Warm tokenizer caches and thread pools before timing.
    let mut samples = Vec::with_capacity(1024);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let allocated = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let start = Instant::now();
    loop {
        let began = Instant::now();
        run();
        samples.push(began.elapsed());
        if start.elapsed() >= Duration::from_millis(300) {
            break;
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let iterations = samples.len();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations;
    let allocated = ALLOCATED_BYTES.load(Ordering::Relaxed) - allocated;
    samples.sort_unstable();
    json!({"name": name, "input_bytes": bytes, "iterations": iterations,
        "seconds": seconds, "ms_per_iteration": seconds * 1000.0 / iterations as f64,
        "mib_per_second": bytes as f64 * iterations as f64 / seconds / 1_048_576.0,
        "p50_ms": percentile(&samples, 0.50), "p95_ms": percentile(&samples, 0.95),
        "p99_ms": percentile(&samples, 0.99),
        "allocations_per_iteration": allocations as f64 / iterations as f64,
        "allocated_bytes_per_iteration": allocated as f64 / iterations as f64})
}

/// Per-stage timings of the backend pipeline on `text`: normalizer, pre-tokenizer, and model,
/// each measured on the output of the previous stage so the sum approximates one encode.
fn stage_workloads(json: &[u8], text: &str) -> anyhow::Result<Vec<Value>> {
    let tokenizer = Tokenizer::from_bytes(json).map_err(|e| anyhow::anyhow!(e))?;
    let mut results = Vec::new();
    let normalized = {
        let mut normalized = NormalizedString::from(text);
        if let Some(normalizer) = tokenizer.get_normalizer() {
            results.push(measure("stage_normalizer", text.len(), || {
                let mut fresh = NormalizedString::from(black_box(text));
                normalizer.normalize(&mut fresh).unwrap();
                black_box(&fresh);
            }));
            normalizer
                .normalize(&mut normalized)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        normalized
    };
    let pre_tokenizer = tokenizer.get_pre_tokenizer();
    let words: Vec<String> = {
        let mut pretokenized = PreTokenizedString::from(normalized.clone());
        if let Some(pre) = pre_tokenizer {
            results.push(measure("stage_pre_tokenizer", text.len(), || {
                let mut fresh = PreTokenizedString::from(black_box(normalized.clone()));
                pre.pre_tokenize(&mut fresh).unwrap();
                black_box(&fresh);
            }));
            pre.pre_tokenize(&mut pretokenized)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        pretokenized
            .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
            .into_iter()
            .map(|(piece, _, _)| piece.to_owned())
            .collect()
    };
    let model = tokenizer.get_model();
    results.push(measure("stage_model", text.len(), || {
        for word in black_box(&words) {
            black_box(model.tokenize(word).unwrap());
        }
    }));
    Ok(results)
}

/// The custom byte-level core measured directly, alongside the backend on the same artifact.
fn core_workloads(json: &[u8], text: &str) -> Vec<Value> {
    let Ok(core) = ByteLevelBpe::from_json(json) else {
        return Vec::new();
    };
    let mut backend = Tokenizer::from_bytes(json).unwrap();
    backend.with_truncation(None).unwrap();
    backend.with_padding(None);
    backend.set_encode_special_tokens(true);
    if core.count(text, false).is_err() {
        eprintln!(
            "custom core declines this corpus (needs normalization); skipping core workloads"
        );
        return Vec::new();
    }
    let mut out = Output::default();
    vec![
        measure("backend_count", text.len(), || {
            black_box(backend.encode_fast(black_box(text), false).unwrap().len());
        }),
        measure("core_count", text.len(), || {
            black_box(core.count(black_box(text), false).unwrap());
        }),
        measure("backend_encode", text.len(), || {
            black_box(backend.encode(black_box(text), false).unwrap());
        }),
        measure("core_encode", text.len(), || {
            core.encode(black_box(text), false, &mut out).unwrap();
            black_box(&out);
        }),
    ]
}

fn main() -> anyhow::Result<()> {
    // Cargo test --all-targets also launches harness-free bench binaries. Keep
    // debug builds out of performance reports and out of routine test timing.
    if cfg!(debug_assertions) || std::env::args().any(|arg| arg == "--test") {
        return Ok(());
    }
    let args = parse_args()?;
    let json = artifact_bytes(args.tokenizer.as_deref())?;
    let engine = Engine::from_bytes(&json, EncodeOptions::default())?;
    let corpus_text = match &args.corpus {
        Some(path) => Some(std::fs::read_to_string(path)?),
        None => None,
    };
    // Default input: 33,800 bytes with a deterministic fixture vocabulary.
    let text = corpus_text
        .clone()
        .unwrap_or_else(|| "hello café 你好 world! ".repeat(1300));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let records = records(corpus_text.as_deref())?;
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
        // A fresh run needs a fresh cancellation scope; pool startup and model loading are included.
        let cancel = Cancellation::default();
        let pool = CpuPool::new(
            self::engine(args.tokenizer.as_deref()).unwrap(),
            4,
            16,
            cancel.clone(),
        )
        .unwrap();
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
    if args.tokenizer.is_some() {
        results.extend(stage_workloads(&json, &text)?);
        results.extend(core_workloads(&json, &text));
    }
    if let Some(concurrency) = &args.http_load {
        let concurrency: usize = concurrency.parse()?;
        let requests: usize = args.requests.as_deref().unwrap_or("2000").parse()?;
        anyhow::ensure!(
            concurrency > 0 && requests >= concurrency,
            "bad load parameters"
        );
        results.push(http_load(&json, &text, concurrency, requests)?);
    }
    let mut output = json!({"fixture_sha256": engine.artifact().sha256,
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        "cpu": std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_default(),
        "profile": "release", "results": results});
    // Only real-artifact runs add these keys, so default reports keep their established shape.
    if args.tokenizer.is_some() || args.corpus.is_some() {
        output["tokenizer"] = json!(args
            .tokenizer
            .as_deref()
            .unwrap_or("tests/fixtures/wordpiece.json"));
        output["vocab_size"] = json!(engine.artifact().vocab_size);
        output["corpus"] = json!(args.corpus.as_deref().unwrap_or("synthetic"));
    }
    let bytes = serde_json::to_vec_pretty(&output)?;
    if let Some(path) = &args.output {
        std::fs::write(path, &bytes)?;
    }
    println!("{}", String::from_utf8(bytes)?);
    Ok(())
}

/// End-to-end latency of `POST /v1/count` over loopback TCP: a real listener, the production
/// router with default limits, a four-worker pool with sixteen extra slots, and `concurrency`
/// keep-alive HTTP/1.1 clients each sending its share of `requests`. Latency includes JSON
/// parsing, admission, queueing, CPU work and response serialization, which the in-process
/// workloads above leave out. No HTTP client crate: requests are written by hand and responses
/// read by Content-Length.
fn http_load(
    json: &[u8],
    text: &str,
    concurrency: usize,
    requests: usize,
) -> anyhow::Result<Value> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let cancel = Cancellation::default();
        let pool = CpuPool::new(
            Engine::from_bytes(json, EncodeOptions::default())?,
            4,
            16,
            cancel.clone(),
        )?;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let shutdown = cancel.clone();
        let server = tokio::spawn(
            axum::serve(listener, router(pool, ApiConfig::default()))
                .with_graceful_shutdown(async move { shutdown.cancelled().await })
                .into_future(),
        );
        let body: Arc<[u8]> = serde_json::to_vec(&json!({"text": text}))?.into();
        let head: Arc<str> = format!(
            "POST /v1/count HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into();
        // Warm up before timing.
        http_client(addr, head.clone(), body.clone(), concurrency).await?;
        let started = Instant::now();
        let mut clients = Vec::with_capacity(concurrency);
        for index in 0..concurrency {
            let share = requests / concurrency + usize::from(index < requests % concurrency);
            clients.push(tokio::spawn(http_client(addr, head.clone(), body.clone(), share)));
        }
        let mut latencies = Vec::with_capacity(requests);
        let mut non_ok = 0usize;
        for client in clients {
            let (times, failures) = client.await??;
            latencies.extend(times);
            non_ok += failures;
        }
        let seconds = started.elapsed().as_secs_f64();
        cancel.cancel();
        let _ = server.await;
        latencies.sort_unstable();
        Ok(json!({"name": format!("http_count_c{concurrency}"), "input_bytes": text.len(),
            "requests": latencies.len(), "concurrency": concurrency, "seconds": seconds,
            "requests_per_second": latencies.len() as f64 / seconds,
            "ms_per_iteration": seconds * 1000.0 / latencies.len() as f64,
            "p50_ms": percentile(&latencies, 0.50), "p95_ms": percentile(&latencies, 0.95),
            "p99_ms": percentile(&latencies, 0.99),
            "max_ms": latencies.last().map_or(0.0, |d| d.as_secs_f64() * 1000.0),
            "non_200_responses": non_ok}))
    })
}

/// Pause after a rejected or dropped request before reconnecting.
const REJECTION_BACKOFF: Duration = Duration::from_millis(5);

/// One client sending `count` requests back to back on a keep-alive connection; returns each
/// request's latency and how many responses were not `200`. The server closes a connection after
/// a `503` it issued before reading the body (an unread body cannot be skipped), so the client
/// reconnects after any non-200 response; a connection closed before any response byte counts
/// as a failed request (`non_200_responses`) and the client reconnects.
async fn http_client(
    addr: std::net::SocketAddr,
    head: Arc<str>,
    body: Arc<[u8]>,
    count: usize,
) -> anyhow::Result<(Vec<Duration>, usize)> {
    let mut stream = connect(addr).await?;
    let mut buffer = Vec::with_capacity(16 * 1024);
    let mut latencies = Vec::with_capacity(count);
    let mut non_ok = 0;
    let mut sent = 0;
    while sent < count {
        let started = Instant::now();
        match http_request(&mut stream, &head, &body, &mut buffer).await? {
            Some(status_ok) => {
                latencies.push(started.elapsed());
                non_ok += usize::from(!status_ok);
                sent += 1;
                if !status_ok {
                    // Honour the server's Retry-After in spirit: back off briefly instead of
                    // hammering the listener with reconnects, as a real client would.
                    tokio::time::sleep(REJECTION_BACKOFF).await;
                    stream = connect(addr).await?;
                }
            }
            None => {
                // Closed before any response byte: under overload the server may drop a
                // connection it has already answered with 503 on a previous request, or a
                // reset races the write. Count it as a failed request so the run always
                // finishes, then reconnect.
                latencies.push(started.elapsed());
                non_ok += 1;
                sent += 1;
                tokio::time::sleep(REJECTION_BACKOFF).await;
                stream = connect(addr).await?;
            }
        }
    }
    Ok((latencies, non_ok))
}

/// Send one request and read one response; `Ok(None)` means the connection closed before any
/// response byte, `Ok(Some(status_ok))` otherwise.
async fn http_request(
    stream: &mut tokio::net::TcpStream,
    head: &str,
    body: &[u8],
    buffer: &mut Vec<u8>,
) -> anyhow::Result<Option<bool>> {
    if stream.write_all(head.as_bytes()).await.is_err() || stream.write_all(body).await.is_err() {
        return Ok(None);
    }
    buffer.clear();
    let mut chunk = [0u8; 8192];
    let (header_end, content_length, status_ok) = loop {
        // A reset counts as a close: the server closes a 503-answered connection without
        // draining the request body, which the peer OS reports as a reset.
        let read = match stream.read(&mut chunk).await {
            Ok(read) => read,
            Err(error) if is_disconnect(&error) => 0,
            Err(error) => return Err(error.into()),
        };
        if read == 0 {
            anyhow::ensure!(
                buffer.is_empty(),
                "server closed the connection mid-response"
            );
            return Ok(None);
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = std::str::from_utf8(&buffer[..end])?;
            let status_ok = headers.starts_with("HTTP/1.1 200");
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())?
                })
                .ok_or_else(|| anyhow::anyhow!("response without Content-Length"))?;
            break (end + 4, length, status_ok);
        }
    };
    while buffer.len() < header_end + content_length {
        let read = match stream.read(&mut chunk).await {
            Ok(read) => read,
            Err(error) if is_disconnect(&error) => 0,
            Err(error) => return Err(error.into()),
        };
        if read == 0 {
            // Headers arrived but the body was cut off: a failed response, not a fatal error.
            return Ok(Some(false));
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    Ok(Some(status_ok))
}

/// Connect with retries: under overload the OS may refuse or reset new connections for a
/// moment (reset storms, ephemeral-port churn), which a real client also rides out.
async fn connect(addr: std::net::SocketAddr) -> anyhow::Result<tokio::net::TcpStream> {
    let mut attempt = 0;
    loop {
        match tokio::net::TcpStream::connect(addr).await {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                return Ok(stream);
            }
            Err(error) if attempt < 100 => {
                attempt += 1;
                let _ = error;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(error) => return Err(anyhow::anyhow!("could not reconnect: {error}")),
        }
    }
}

fn is_disconnect(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::UnexpectedEof
    )
}
