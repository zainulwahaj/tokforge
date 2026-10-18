//! Bounded CPU admission, shared operations, and cooperative cancellation.
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{oneshot, watch, OwnedSemaphorePermit, Semaphore};

use crate::{Engine, Error};

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
    ) -> Result<Value, Error> {
        check()?;
        let mut output = json!({"artifact": engine.artifact(), "options": engine.options()});
        match self {
            Self::Count { text } => output["count"] = json!(engine.count(&text)?),
            Self::Encode { text } => output["encoding"] = json!(engine.encode(&text)?),
            Self::Decode {
                ids,
                skip_special_tokens,
            } => {
                output["text"] = json!(engine.decode(&ids, skip_special_tokens)?);
                output["skip_special_tokens"] = json!(skip_special_tokens);
            }
            Self::Truncate { text, max_tokens } => {
                let prefix = engine.truncate_checked(&text, max_tokens, check)?;
                output["truncated"] = json!(prefix.end_byte < text.len());
                output["max_tokens"] = json!(max_tokens);
                output["result"] = json!(prefix);
            }
            Self::Chunk { text, max_tokens } => {
                output["chunks"] = json!(engine.chunk_checked(&text, max_tokens, check)?);
                output["max_tokens"] = json!(max_tokens);
            }
        }
        check()?;
        Ok(output)
    }
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
                let value = operation.execute(&engine, &check)?;
                let bytes = serde_json::to_vec(&value).map_err(|_| WorkError::Failed)?;
                check()?;
                Ok(bytes)
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

    use crate::EncodeOptions;

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
