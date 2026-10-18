//! Ordered JSONL processing with a bounded window of records and CPU jobs.
use std::{io, sync::Arc};

use serde_json::json;
use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt},
    sync::{mpsc, OwnedSemaphorePermit, Semaphore},
};

use crate::work::{Cancellation, CpuPool, Operation, Ticket, WorkError};

#[derive(Clone, Copy)]
pub struct BatchConfig {
    pub max_line_bytes: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_line_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, Default)]
pub struct Summary {
    pub records: u64,
    pub errors: u64,
}

enum ResultSource {
    Worker(Ticket),
    Invalid { code: &'static str, message: String },
}

struct Pending {
    line: u64,
    source: ResultSource,
    // Held through writing to the bounded output buffer, so slow consumers
    // bound completed results as well as queued work.
    _window: OwnedSemaphorePermit,
}

/// One response per physical line, in input order. Per-record errors continue;
/// I/O failures and cancellation stop the stream and may leave partial output.
pub async fn process<R, W>(
    reader: R,
    mut writer: W,
    pool: CpuPool,
    config: BatchConfig,
    cancel: Cancellation,
) -> anyhow::Result<Summary>
where
    R: AsyncBufRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin,
{
    anyhow::ensure!(config.max_line_bytes > 0, "max line bytes must be positive");
    let _guard = cancel.guard();
    let (sender, mut receiver) = mpsc::channel(pool.capacity());
    let producer_cancel = cancel.clone();
    let producer = tokio::spawn(async move {
        tokio::select! {
            biased;
            _ = producer_cancel.cancelled() => {},
            _ = produce(reader, pool, config, sender, producer_cancel.clone()) => {},
        }
    });
    let consume = async {
        let mut summary = Summary::default();
        let mut unflushed = 0;
        loop {
            // Batch ready results, but flush before waiting on more input. This
            // preserves interactive latency without forcing a syscall per row.
            let pending = match receiver.try_recv() {
                Ok(pending) => Some(pending),
                Err(mpsc::error::TryRecvError::Empty) => {
                    if unflushed > 0 {
                        writer.flush().await?;
                        unflushed = 0;
                    }
                    receiver.recv().await
                }
                Err(mpsc::error::TryRecvError::Disconnected) => None,
            };
            let Some(pending) = pending else {
                break;
            };
            let pending = match pending {
                Ok(pending) => pending,
                Err(error) => {
                    writer.flush().await?;
                    return Err(error.into());
                }
            };
            let Pending {
                line,
                source,
                _window,
            } = pending;
            match source {
                ResultSource::Worker(mut ticket) => {
                    let result = match ticket.try_recv() {
                        Ok(result) => result,
                        Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                            if unflushed > 0 {
                                writer.flush().await?;
                                unflushed = 0;
                            }
                            ticket.await.unwrap_or(Err(WorkError::Failed))
                        }
                        Err(_) => Err(WorkError::Failed),
                    };
                    match result {
                        Ok(bytes) => writer.write_all(&bytes).await?,
                        Err(error) => {
                            summary.errors += 1;
                            write_error(&mut writer, line, error.code(), &error.to_string())
                                .await?;
                        }
                    }
                }
                ResultSource::Invalid { code, message } => {
                    summary.errors += 1;
                    write_error(&mut writer, line, code, &message).await?;
                }
            }
            summary.records += 1;
            unflushed += 1;
            if unflushed >= 64 {
                writer.flush().await?;
                unflushed = 0;
            }
        }
        writer.flush().await?;
        Ok::<_, anyhow::Error>(summary)
    };
    let result = tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(anyhow::anyhow!("JSONL processing cancelled")),
        result = consume => result,
    };
    // Cancels input reads even when output fails or the caller drops this future.
    cancel.cancel();
    producer.await?;
    result
}

async fn write_error<W: AsyncWrite + Unpin>(
    writer: &mut W,
    line: u64,
    code: &str,
    message: &str,
) -> io::Result<()> {
    let bytes =
        serde_json::to_vec(&json!({"line": line, "error": {"code": code, "message": message}}))?;
    writer.write_all(&bytes).await?;
    writer.write_all(b"\n").await
}

async fn produce<R: AsyncBufRead + Unpin>(
    mut reader: R,
    pool: CpuPool,
    config: BatchConfig,
    sender: mpsc::Sender<io::Result<Pending>>,
    cancel: Cancellation,
) {
    let window = Arc::new(Semaphore::new(pool.capacity()));
    let mut line = 0;
    loop {
        let Ok(permit) = window.clone().acquire_owned().await else {
            return;
        };
        let record = match read_line(&mut reader, config.max_line_bytes).await {
            Ok(Some(record)) => record,
            Ok(None) => return,
            Err(error) => {
                let _ = sender.send(Err(error)).await;
                return;
            }
        };
        line += 1;
        let source = match record {
            Some(bytes) => match serde_json::from_slice::<Operation>(&bytes) {
                Ok(operation) => match pool.reserve() {
                    Ok(slot) => ResultSource::Worker(pool.submit_record(
                        operation,
                        cancel.clone(),
                        slot,
                        line,
                    )),
                    Err(error) => ResultSource::Invalid {
                        code: error.code(),
                        message: error.to_string(),
                    },
                },
                Err(error) => ResultSource::Invalid {
                    code: "invalid_request",
                    message: error.to_string(),
                },
            },
            None => ResultSource::Invalid {
                code: "line_too_large",
                message: format!("record exceeds {} bytes", config.max_line_bytes),
            },
        };
        if sender
            .send(Ok(Pending {
                line,
                source,
                _window: permit,
            }))
            .await
            .is_err()
        {
            return;
        }
    }
}

// None = EOF; Some(None) = an oversized record, drained through its newline.
// Memory is bounded even for a huge input line, and fill_buf handles split UTF-8.
async fn read_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max: usize,
) -> io::Result<Option<Option<Vec<u8>>>> {
    let mut bytes = Vec::new();
    let mut oversized = false;
    let mut seen = false;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(seen.then_some(if oversized { None } else { Some(bytes) }));
        }
        seen = true;
        let end = available.iter().position(|&byte| byte == b'\n');
        let count = end.unwrap_or(available.len());
        if !oversized {
            if count > max.saturating_sub(bytes.len()) {
                oversized = true;
                bytes.clear();
            } else {
                bytes.extend_from_slice(&available[..count]);
            }
        }
        reader.consume(count + usize::from(end.is_some()));
        if end.is_some() {
            return Ok(Some(if oversized { None } else { Some(bytes) }));
        }
    }
}
