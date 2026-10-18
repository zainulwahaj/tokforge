//! Bounded CPU admission, shared operations, cooperative cancellation, and metrics.
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};
use tokio::sync::{oneshot, watch, OwnedSemaphorePermit, Semaphore};

use crate::{Artifact, EncodeOptions, Engine, Error, Span};

/// Version of the JSON response shape. Bumped only for incompatible changes; additive fields
/// keep the number.
pub const SCHEMA_VERSION: u32 = 1;
/// The locked Hugging Face backend, as `tokenizers <version>` (captured by build.rs).
pub const BACKEND: &str = concat!("tokenizers ", env!("TOKENIZERS_VERSION"));
/// Most operations one batch request may carry; the HTTP body limit bounds their total size.
pub const MAX_BATCH_OPERATIONS: usize = 256;

#[derive(Clone)]
pub struct Cancellation {
    flag: Arc<AtomicBool>,
    changed: watch::Sender<bool>,
}

impl Default for Cancellation {
    fn default() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            changed: watch::channel(false).0,
        }
    }
}

impl Cancellation {
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.changed.send_replace(true);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        let mut receiver = self.changed.subscribe();
        let _ = receiver.wait_for(|cancelled| *cancelled).await;
    }

    pub fn guard(&self) -> CancelOnDrop {
        CancelOnDrop(self.clone())
    }
}

pub struct CancelOnDrop(Cancellation);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Count {
        text: String,
    },
    Encode {
        text: String,
        /// Return only `ids` and `count`: no token strings, offsets, or mask. Most of an encode
        /// response's bytes are the token strings and offsets, so this is the lean path for
        /// callers that feed IDs to a model.
        #[serde(default)]
        ids_only: bool,
    },
    Decode {
        ids: Vec<u32>,
        #[serde(default)]
        skip_special_tokens: bool,
    },
    Truncate {
        text: String,
        max_tokens: usize,
    },
    Chunk {
        text: String,
        max_tokens: usize,
    },
    /// Several operations under one admission slot, executed in order on one worker. Each item
    /// succeeds or fails on its own; the batch fails as a whole only when cancelled.
    Batch {
        operations: Vec<Operation>,
    },
}

/// Operation kinds in `Metrics` order.
pub const OPERATION_NAMES: [&str; 6] = ["count", "encode", "decode", "truncate", "chunk", "batch"];

impl Operation {
    fn kind(&self) -> usize {
        match self {
            Self::Count { .. } => 0,
            Self::Encode { .. } => 1,
            Self::Decode { .. } => 2,
            Self::Truncate { .. } => 3,
            Self::Chunk { .. } => 4,
            Self::Batch { .. } => 5,
        }
    }

    fn execute<'e>(
        self,
        engine: &'e Engine,
        check: &impl Fn() -> Result<(), Error>,
        metrics: Option<&Metrics>,
        nested: bool,
    ) -> Result<ResultData<'e>, WorkError> {
        if let Some(metrics) = metrics {
            metrics.requests[self.kind()].fetch_add(1, Ordering::Relaxed);
        }
        check()?;
        let output = match self {
            Self::Count { text } => ResultData::Count {
                count: engine.count(&text)?,
            },
            Self::Encode { text, ids_only } => ResultData::Encode {
                encoding: engine.encode_raw(&text)?,
                ids_only,
            },
            Self::Decode {
                ids,
                skip_special_tokens,
            } => ResultData::Decode {
                text: engine.decode(&ids, skip_special_tokens)?,
                skip_special_tokens,
            },
            Self::Truncate { text, max_tokens } => {
                let span = engine.truncate_span_checked(&text, max_tokens, check)?;
                ResultData::Truncate {
                    truncated: span.end_byte < text.len(),
                    max_tokens,
                    result: Slices {
                        spans: vec![span],
                        text,
                    },
                }
            }
            Self::Chunk { text, max_tokens } => {
                let spans = engine.chunk_spans_checked(&text, max_tokens, check)?;
                ResultData::Chunk {
                    chunks: Slices { spans, text },
                    max_tokens,
                }
            }
            Self::Batch { operations } => {
                if nested {
                    return Err(WorkError::InvalidRequest(
                        "a batch cannot contain another batch".into(),
                    ));
                }
                if operations.len() > MAX_BATCH_OPERATIONS {
                    return Err(WorkError::InvalidRequest(format!(
                        "a batch may contain at most {MAX_BATCH_OPERATIONS} operations"
                    )));
                }
                let mut results = Vec::with_capacity(operations.len());
                for operation in operations {
                    match operation.execute(engine, check, metrics, true) {
                        Ok(result) => results.push(BatchItem::Ok(Box::new(result))),
                        Err(error) if error.is_cancelled() => return Err(error),
                        Err(error) => results.push(BatchItem::Err {
                            error: ErrorBody {
                                code: error.code(),
                                message: error.to_string(),
                            },
                        }),
                    }
                }
                ResultData::Batch { results }
            }
        };
        check()?;
        Ok(output)
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum ResultData<'e> {
    Count {
        count: usize,
    },
    Encode {
        #[serde(flatten, serialize_with = "serialize_encoding")]
        encoding: crate::Encoding<'e>,
        #[serde(skip)]
        ids_only: bool,
    },
    Decode {
        text: String,
        skip_special_tokens: bool,
    },
    Truncate {
        truncated: bool,
        max_tokens: usize,
        #[serde(serialize_with = "serialize_slice")]
        result: Slices,
    },
    Chunk {
        #[serde(serialize_with = "serialize_slices")]
        chunks: Slices,
        max_tokens: usize,
    },
    Batch {
        results: Vec<BatchItem<'e>>,
    },
}

#[derive(Serialize)]
#[serde(untagged)]
enum BatchItem<'e> {
    // Boxed so the error variant does not pay for the largest result's size.
    Ok(Box<ResultData<'e>>),
    Err { error: ErrorBody },
}

#[derive(Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

/// The request text, moved out of the operation, plus verified spans into it.
/// Serialization borrows each slice, so a chunk response never holds a second
/// copy of the document: the input String is the only owner of those bytes.
struct Slices {
    text: String,
    spans: Vec<Span>,
}

#[derive(Serialize)]
struct SliceRef<'a> {
    text: &'a str,
    start_byte: usize,
    end_byte: usize,
    count: usize,
}

impl Slices {
    fn slice(&self, span: Span) -> SliceRef<'_> {
        SliceRef {
            text: &self.text[span.start_byte..span.end_byte],
            start_byte: span.start_byte,
            end_byte: span.end_byte,
            count: span.count,
        }
    }
}

fn serialize_slice<S: Serializer>(slices: &Slices, serializer: S) -> Result<S::Ok, S::Error> {
    slices.slice(slices.spans[0]).serialize(serializer)
}

fn serialize_slices<S: Serializer>(slices: &Slices, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(slices.spans.iter().map(|&span| slices.slice(span)))
}

// Serialize directly from the tokenizer's owned buffers. Building an Encoded
// plus a serde_json::Value would copy every ID, token string, and byte offset.
// The encoding is written as a nested `encoding` object, as before.
fn serialize_encoding<S: Serializer>(
    encoding: &crate::Encoding<'_>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct Wrapper<'a> {
        #[serde(serialize_with = "serialize_encoding_fields")]
        encoding: &'a crate::Encoding<'a>,
    }
    Wrapper { encoding }.serialize(serializer)
}

/// Token strings streamed straight from the encoding's borrowed spellings.
struct Tokens<'a>(&'a crate::Encoding<'a>);

impl Serialize for Tokens<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.tokens())
    }
}

fn serialize_encoding_fields<S: Serializer>(
    encoding: &&crate::Encoding<'_>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut output = serializer.serialize_struct("Encoded", 5)?;
    output.serialize_field("ids", encoding.ids())?;
    output.serialize_field("tokens", &Tokens(encoding))?;
    output.serialize_field("offsets", encoding.offsets())?;
    output.serialize_field("special_tokens_mask", encoding.special_tokens_mask())?;
    output.serialize_field("count", &encoding.len())?;
    output.end()
}

/// IDs-only encode response body: `{"encoding": {"ids": [...], "count": n}}`.
struct IdsOnly<'a>(&'a crate::Encoding<'a>);

impl Serialize for IdsOnly<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Fields<'a> {
            ids: &'a [u32],
            count: usize,
        }
        #[derive(Serialize)]
        struct Wrapper<'a> {
            encoding: Fields<'a>,
        }
        Wrapper {
            encoding: Fields {
                ids: self.0.ids(),
                count: self.0.len(),
            },
        }
        .serialize(serializer)
    }
}

impl ResultData<'_> {
    /// Serialize into `bytes` after the common response prefix fields.
    fn write_into<W: std::io::Write>(&self, writer: W) -> Result<(), serde_json::Error> {
        match self {
            // The IDs-only path bypasses the generic serializer so tokens and offsets are
            // never touched.
            ResultData::Encode {
                encoding,
                ids_only: true,
            } => serde_json::to_writer(writer, &IdsOnly(encoding)),
            other => serde_json::to_writer(writer, other),
        }
    }
}

/// Fields every response carries before the operation's own fields.
#[derive(Serialize)]
struct Header<'a> {
    schema_version: u32,
    engine_version: &'static str,
    backend: &'static str,
    artifact: &'a Artifact,
    options: EncodeOptions,
}

/// Execute and serialize a single operation without constructing a JSON tree.
pub fn execute(engine: &Engine, operation: Operation) -> WorkResult {
    execute_checked(engine, operation, &|| Ok(()), None, None)
}

fn execute_checked(
    engine: &Engine,
    operation: Operation,
    check: &impl Fn() -> Result<(), Error>,
    line: Option<u64>,
    metrics: Option<&Metrics>,
) -> WorkResult {
    let result = operation.execute(engine, check, metrics, false)?;
    let header = Header {
        schema_version: SCHEMA_VERSION,
        engine_version: crate::VERSION,
        backend: BACKEND,
        artifact: engine.artifact(),
        options: engine.options(),
    };
    // Write `{"line":n,"result":{<header fields>,<result fields>}}` or `{<header>,<result>}` by
    // splicing two flat objects: the header's closing brace is replaced by a comma and the
    // result's opening brace is dropped, which keeps both halves on their fast serializers.
    let mut bytes = Vec::with_capacity(512);
    if let Some(line) = line {
        bytes.extend_from_slice(b"{\"line\":");
        bytes.extend_from_slice(line.to_string().as_bytes());
        bytes.extend_from_slice(b",\"result\":");
    }
    serde_json::to_writer(&mut bytes, &header).map_err(|_| WorkError::Failed)?;
    debug_assert_eq!(bytes.last(), Some(&b'}'));
    bytes.pop();
    bytes.push(b',');
    let body_start = bytes.len();
    result
        .write_into(&mut bytes)
        .map_err(|_| WorkError::Failed)?;
    debug_assert_eq!(bytes.get(body_start), Some(&b'{'));
    bytes.remove(body_start);
    if line.is_some() {
        bytes.extend_from_slice(b"}\n");
    }
    check()?;
    Ok(bytes)
}

#[derive(Debug, thiserror::Error)]
pub enum WorkError {
    #[error("CPU capacity is full; retry later")]
    Busy,
    #[error("operation cancelled")]
    Cancelled,
    #[error(transparent)]
    Engine(#[from] Error),
    #[error("{0}")]
    InvalidRequest(String),
    #[error("CPU worker failed")]
    Failed,
}

impl WorkError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Busy => "busy",
            Self::Cancelled | Self::Engine(Error::Cancelled) => "cancelled",
            Self::Engine(_) => "tokenizer_error",
            Self::InvalidRequest(_) => "invalid_request",
            Self::Failed => "worker_failed",
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Engine(Error::Cancelled))
    }
}

pub type WorkResult = Result<Vec<u8>, WorkError>;
pub type Ticket = oneshot::Receiver<WorkResult>;

/// Process-wide counters, cheap enough to update on every job. Exposed by `GET /metrics`.
#[derive(Debug)]
pub struct Metrics {
    started: Instant,
    requests: [AtomicU64; OPERATION_NAMES.len()],
    completed: AtomicU64,
    errors: AtomicU64,
    cancelled: AtomicU64,
    failed: AtomicU64,
    busy_rejections: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            requests: Default::default(),
            completed: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            busy_rejections: AtomicU64::new(0),
        }
    }
}

impl Metrics {
    fn record(&self, result: &WorkResult) {
        let counter = match result {
            Ok(_) => &self.completed,
            Err(error) if error.is_cancelled() => &self.cancelled,
            Err(WorkError::Failed) => &self.failed,
            Err(_) => &self.errors,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn requests(&self, operation: &str) -> u64 {
        OPERATION_NAMES
            .iter()
            .position(|name| *name == operation)
            .map_or(0, |index| self.requests[index].load(Ordering::Relaxed))
    }

    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Relaxed)
    }

    pub fn busy_rejections(&self) -> u64 {
        self.busy_rejections.load(Ordering::Relaxed)
    }

    /// Prometheus text exposition.
    pub fn render(&self, capacity: usize, in_flight: usize) -> String {
        let mut out = String::with_capacity(1024);
        out.push_str("# TYPE tokenizer_requests_total counter\n");
        for (index, name) in OPERATION_NAMES.iter().enumerate() {
            let value = self.requests[index].load(Ordering::Relaxed);
            out.push_str(&format!(
                "tokenizer_requests_total{{op=\"{name}\"}} {value}\n"
            ));
        }
        out.push_str("# TYPE tokenizer_jobs_total counter\n");
        for (outcome, counter) in [
            ("completed", &self.completed),
            ("error", &self.errors),
            ("cancelled", &self.cancelled),
            ("failed", &self.failed),
        ] {
            let value = counter.load(Ordering::Relaxed);
            out.push_str(&format!(
                "tokenizer_jobs_total{{outcome=\"{outcome}\"}} {value}\n"
            ));
        }
        out.push_str("# TYPE tokenizer_busy_rejections_total counter\n");
        out.push_str(&format!(
            "tokenizer_busy_rejections_total {}\n",
            self.busy_rejections.load(Ordering::Relaxed)
        ));
        out.push_str("# TYPE tokenizer_in_flight gauge\n");
        out.push_str(&format!("tokenizer_in_flight {in_flight}\n"));
        out.push_str("# TYPE tokenizer_capacity gauge\n");
        out.push_str(&format!("tokenizer_capacity {capacity}\n"));
        out.push_str("# TYPE tokenizer_uptime_seconds gauge\n");
        out.push_str(&format!(
            "tokenizer_uptime_seconds {:.3}\n",
            self.started.elapsed().as_secs_f64()
        ));
        out
    }
}

#[derive(Clone)]
pub struct CpuPool {
    pool: Arc<rayon::ThreadPool>,
    engine: Arc<Engine>,
    slots: Arc<Semaphore>,
    shutdown: Cancellation,
    capacity: usize,
    metrics: Arc<Metrics>,
}

impl CpuPool {
    pub fn new(
        engine: Engine,
        workers: usize,
        queue_capacity: usize,
        shutdown: Cancellation,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(workers > 0, "workers must be positive");
        let capacity = workers
            .checked_add(queue_capacity)
            .filter(|n| *n <= Semaphore::MAX_PERMITS)
            .ok_or_else(|| anyhow::anyhow!("CPU capacity is too large"))?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("tokenizer-cpu-{index}"))
            .build()?;
        Ok(Self {
            pool: Arc::new(pool),
            engine: Arc::new(engine),
            slots: Arc::new(Semaphore::new(capacity)),
            shutdown,
            capacity,
            metrics: Arc::new(Metrics::default()),
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Admitted jobs right now: queued, running, or completed but not yet collected.
    pub fn in_flight(&self) -> usize {
        self.capacity - self.slots.available_permits()
    }
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
    pub fn shutdown(&self) -> &Cancellation {
        &self.shutdown
    }
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn reserve(&self) -> Result<OwnedSemaphorePermit, WorkError> {
        if self.shutdown.is_cancelled() {
            return Err(WorkError::Cancelled);
        }
        self.slots.clone().try_acquire_owned().map_err(|_| {
            self.metrics.busy_rejections.fetch_add(1, Ordering::Relaxed);
            WorkError::Busy
        })
    }

    pub fn submit(
        &self,
        operation: Operation,
        cancel: Cancellation,
        permit: OwnedSemaphorePermit,
    ) -> Ticket {
        self.submit_inner(operation, cancel, permit, None)
    }

    pub(crate) fn submit_record(
        &self,
        operation: Operation,
        cancel: Cancellation,
        permit: OwnedSemaphorePermit,
        line: u64,
    ) -> Ticket {
        self.submit_inner(operation, cancel, permit, Some(line))
    }

    fn submit_inner(
        &self,
        operation: Operation,
        cancel: Cancellation,
        permit: OwnedSemaphorePermit,
        line: Option<u64>,
    ) -> Ticket {
        let (sender, receiver) = oneshot::channel();
        let engine = self.engine.clone();
        let shutdown = self.shutdown.clone();
        let metrics = self.metrics.clone();
        self.pool.spawn_fifo(move || {
            let _permit = permit;
            let check = || {
                if cancel.is_cancelled() || shutdown.is_cancelled() || sender.is_closed() {
                    Err(Error::Cancelled)
                } else {
                    Ok(())
                }
            };
            let result = catch_unwind(AssertUnwindSafe(|| {
                execute_checked(&engine, operation, &check, line, Some(&metrics))
            }))
            .unwrap_or(Err(WorkError::Failed));
            metrics.record(&result);
            drop(_permit);
            let _ = sender.send(result);
        });
        receiver
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn pool() -> CpuPool {
        CpuPool::new(
            Engine::from_bytes(
                include_bytes!("../tests/fixtures/wordpiece.json"),
                EncodeOptions::default(),
            )
            .unwrap(),
            1,
            1,
            Cancellation::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn queued_cancellation_releases_capacity_and_shutdown_rejects_new_work() {
        let pool = pool();
        let (entered, ready) = oneshot::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        pool.pool.spawn_fifo(move || {
            let _ = entered.send(());
            let _ = blocked.recv();
        });
        tokio::time::timeout(Duration::from_secs(3), ready)
            .await
            .unwrap()
            .unwrap();
        let cancel = Cancellation::default();
        let ticket = pool.submit(
            Operation::Count {
                text: "hello".to_owned(),
            },
            cancel.clone(),
            pool.reserve().unwrap(),
        );
        let held = pool.reserve().unwrap();
        assert!(matches!(pool.reserve(), Err(WorkError::Busy)));
        assert_eq!(pool.metrics().busy_rejections(), 1);
        cancel.cancel();
        release.send(()).unwrap();
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(3), ticket)
                .await
                .unwrap()
                .unwrap(),
            Err(WorkError::Engine(Error::Cancelled))
        ));
        assert!(pool.reserve().is_ok());
        drop(held);
        pool.shutdown.cancel();
        assert!(matches!(pool.reserve(), Err(WorkError::Cancelled)));
    }

    #[tokio::test]
    async fn cancellation_is_sticky_for_late_and_existing_waiters() {
        let cancel = Cancellation::default();
        let waiting = tokio::spawn({
            let cancel = cancel.clone();
            async move { cancel.cancelled().await }
        });
        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(3), waiting)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), cancel.cancelled())
            .await
            .unwrap();
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn responses_carry_schema_and_backend_and_ids_only_omits_tokens() {
        let engine = Engine::from_bytes(
            include_bytes!("../tests/fixtures/wordpiece.json"),
            EncodeOptions::default(),
        )
        .unwrap();
        let full: serde_json::Value = serde_json::from_slice(
            &execute(
                &engine,
                Operation::Encode {
                    text: "hello world!".into(),
                    ids_only: false,
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(full["schema_version"], SCHEMA_VERSION);
        assert_eq!(full["engine_version"], crate::VERSION);
        assert!(full["backend"].as_str().unwrap().starts_with("tokenizers "));
        assert_eq!(full["encoding"]["ids"], serde_json::json!([4, 5, 6]));
        assert!(full["encoding"]["tokens"].is_array());
        let lean: serde_json::Value = serde_json::from_slice(
            &execute(
                &engine,
                Operation::Encode {
                    text: "hello world!".into(),
                    ids_only: true,
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(lean["encoding"]["ids"], serde_json::json!([4, 5, 6]));
        assert_eq!(lean["encoding"]["count"], 3);
        assert!(lean["encoding"].get("tokens").is_none());
        assert!(lean["encoding"].get("offsets").is_none());
        assert_eq!(lean["artifact"]["vocab_size"], 10);
    }

    #[test]
    fn batches_keep_order_and_isolate_item_errors() {
        let engine = Engine::from_bytes(
            include_bytes!("../tests/fixtures/wordpiece.json"),
            EncodeOptions::default(),
        )
        .unwrap();
        let batch = Operation::Batch {
            operations: vec![
                Operation::Count {
                    text: "hello world".into(),
                },
                Operation::Decode {
                    ids: vec![99],
                    skip_special_tokens: false,
                },
                Operation::Batch { operations: vec![] },
                Operation::Truncate {
                    text: "hello world hello".into(),
                    max_tokens: 2,
                },
            ],
        };
        let value: serde_json::Value =
            serde_json::from_slice(&execute(&engine, batch).unwrap()).unwrap();
        let results = value["results"].as_array().unwrap();
        assert_eq!(results.len(), 4);
        assert_eq!(results[0]["count"], 2);
        assert_eq!(results[1]["error"]["code"], "tokenizer_error");
        assert_eq!(results[2]["error"]["code"], "invalid_request");
        assert_eq!(results[3]["result"]["count"], 2);
        let too_many = Operation::Batch {
            operations: (0..=MAX_BATCH_OPERATIONS)
                .map(|_| Operation::Count { text: "x".into() })
                .collect(),
        };
        assert!(matches!(
            execute(&engine, too_many),
            Err(WorkError::InvalidRequest(_))
        ));
    }
}
