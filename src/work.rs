//! Bounded CPU admission, shared operations, and cooperative cancellation.
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};
use tokio::sync::{oneshot, watch, OwnedSemaphorePermit, Semaphore};

use crate::{Artifact, Chunk, EncodeOptions, Engine, Error};

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
}

impl Operation {
    fn execute(
        self,
        engine: &Engine,
        check: &impl Fn() -> Result<(), Error>,
    ) -> Result<ResultData, Error> {
        check()?;
        let output = match self {
            Self::Count { text } => ResultData::Count {
                count: engine.count(&text)?,
            },
            Self::Encode { text } => ResultData::Encode {
                encoding: engine.encode_raw(&text)?,
            },
            Self::Decode {
                ids,
                skip_special_tokens,
            } => ResultData::Decode {
                text: engine.decode(&ids, skip_special_tokens)?,
                skip_special_tokens,
            },
            Self::Truncate { text, max_tokens } => {
                let prefix = engine.truncate_checked(&text, max_tokens, check)?;
                ResultData::Truncate {
                    truncated: prefix.end_byte < text.len(),
                    max_tokens,
                    result: prefix,
                }
            }
            Self::Chunk { text, max_tokens } => ResultData::Chunk {
                chunks: engine.chunk_checked(&text, max_tokens, check)?,
                max_tokens,
            },
        };
        check()?;
        Ok(output)
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum ResultData {
    Count {
        count: usize,
    },
    Encode {
        #[serde(serialize_with = "serialize_encoding")]
        encoding: tokenizers::Encoding,
    },
    Decode {
        text: String,
        skip_special_tokens: bool,
    },
    Truncate {
        truncated: bool,
        max_tokens: usize,
        result: Chunk,
    },
    Chunk {
        chunks: Vec<Chunk>,
        max_tokens: usize,
    },
}

// Serialize directly from the tokenizer's owned buffers. Building an Encoded
// plus a serde_json::Value would copy every ID, token string, and byte offset.
fn serialize_encoding<S: Serializer>(
    encoding: &tokenizers::Encoding,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut output = serializer.serialize_struct("Encoded", 5)?;
    output.serialize_field("ids", encoding.get_ids())?;
    output.serialize_field("tokens", encoding.get_tokens())?;
    output.serialize_field("offsets", encoding.get_offsets())?;
    output.serialize_field("special_tokens_mask", encoding.get_special_tokens_mask())?;
    output.serialize_field("count", &encoding.len())?;
    output.end()
}

#[derive(Serialize)]
struct Response<'a> {
    artifact: &'a Artifact,
    options: EncodeOptions,
    #[serde(flatten)]
    result: ResultData,
}

#[derive(Serialize)]
struct Record<T> {
    line: u64,
    result: T,
}

/// Execute and serialize a single operation without constructing a JSON tree.
pub fn execute(engine: &Engine, operation: Operation) -> WorkResult {
    execute_checked(engine, operation, &|| Ok(()), None)
}

fn execute_checked(
    engine: &Engine,
    operation: Operation,
    check: &impl Fn() -> Result<(), Error>,
    line: Option<u64>,
) -> WorkResult {
    let result = operation.execute(engine, check)?;
    let response = Response {
        artifact: engine.artifact(),
        options: engine.options(),
        result,
    };
    let mut bytes = Vec::with_capacity(512);
    if let Some(line) = line {
        serde_json::to_writer(
            &mut bytes,
            &Record {
                line,
                result: response,
            },
        )
        .map_err(|_| WorkError::Failed)?;
        bytes.push(b'\n');
    } else {
        serde_json::to_writer(&mut bytes, &response).map_err(|_| WorkError::Failed)?;
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
    #[error("CPU worker failed")]
    Failed,
}

impl WorkError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Busy => "busy",
            Self::Cancelled | Self::Engine(Error::Cancelled) => "cancelled",
            Self::Engine(_) => "tokenizer_error",
            Self::Failed => "worker_failed",
        }
    }
}

pub type WorkResult = Result<Vec<u8>, WorkError>;
pub type Ticket = oneshot::Receiver<WorkResult>;

#[derive(Clone)]
pub struct CpuPool {
    pool: Arc<rayon::ThreadPool>,
    engine: Arc<Engine>,
    slots: Arc<Semaphore>,
    shutdown: Cancellation,
    capacity: usize,
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
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
    pub fn shutdown(&self) -> &Cancellation {
        &self.shutdown
    }

    pub fn reserve(&self) -> Result<OwnedSemaphorePermit, WorkError> {
        if self.shutdown.is_cancelled() {
            return Err(WorkError::Cancelled);
        }
        self.slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| WorkError::Busy)
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
                execute_checked(&engine, operation, &check, line)
            }))
            .unwrap_or(Err(WorkError::Failed));
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
}
