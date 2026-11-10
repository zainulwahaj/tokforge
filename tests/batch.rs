mod support;

use std::{
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use serde_json::{json, Value};
use tokforge::{
    batch::{process, BatchConfig},
    work::Cancellation,
};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader, ReadBuf};

#[tokio::test]
async fn preserves_order_for_mixed_operations_errors_and_unicode() {
    let mut input = String::new();
    for index in 0..100 {
        let record = match index % 5 {
            0 => json!({"op": "count", "text": "hello café 你好!"}),
            1 => json!({"op": "encode", "text": "hello world!"}),
            2 => json!({"op": "decode", "ids": [4,5,6]}),
            3 => json!({"op": "truncate", "text": "hello world!", "max_tokens": 1}),
            _ => json!({"op": "chunk", "text": "hello\r\nworld!", "max_tokens": 2}),
        };
        input.push_str(&record.to_string());
        input.push('\n');
    }
    input.push_str(
        "not json\n{\"op\":\"decode\",\"ids\":[999]}\n{\"op\":\"count\",\"text\":\"hello\"}",
    );
    let mut output = Vec::new();
    let summary = process(
        io::Cursor::new(input.into_bytes()),
        &mut output,
        support::pool(3, 2),
        BatchConfig::default(),
        Cancellation::default(),
    )
    .await
    .unwrap();
    assert_eq!((summary.records, summary.errors), (103, 2));
    let rows: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row["line"], index + 1);
    }
    assert_eq!(rows[0]["result"]["count"], 4);
    assert_eq!(rows[1]["result"]["encoding"]["ids"], json!([4, 5, 6]));
    assert_eq!(rows[2]["result"]["text"], "hello world!");
    assert_eq!(rows[100]["error"]["code"], "invalid_request");
    assert_eq!(rows[101]["error"]["code"], "tokenizer_error");
    assert_eq!(rows[102]["result"]["count"], 1);
}

#[tokio::test]
async fn oversized_blank_and_invalid_utf8_records_do_not_break_following_lines() {
    let mut input = vec![b'x'; 10_000];
    input.extend_from_slice(b"\n\n\xff\n{\"op\":\"count\",\"text\":\"hello\"}\r\n");
    let reader = BufReader::with_capacity(7, io::Cursor::new(input));
    let mut output = Vec::new();
    let summary = process(
        reader,
        &mut output,
        support::pool(1, 0),
        BatchConfig { max_line_bytes: 40 },
        Cancellation::default(),
    )
    .await
    .unwrap();
    assert_eq!((summary.records, summary.errors), (4, 3));
    let rows: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(rows[0]["error"]["code"], "line_too_large");
    assert_eq!(rows[3]["result"]["count"], 1);
}

#[tokio::test]
async fn emits_results_before_input_eof_and_cancels_blocked_reads() {
    let (input, mut feed) = tokio::io::duplex(1024);
    let (output, drain) = tokio::io::duplex(1024);
    let cancel = Cancellation::default();
    let task = tokio::spawn(process(
        BufReader::new(input),
        output,
        support::pool(1, 3),
        BatchConfig::default(),
        cancel.clone(),
    ));
    feed.write_all(b"{\"op\":\"count\",\"text\":\"hello\"}\n")
        .await
        .unwrap();
    let mut drain = BufReader::new(drain);
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(3), drain.read_line(&mut line))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&line).unwrap()["result"]["count"],
        1
    );
    cancel.cancel();
    assert!(tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap()
        .is_err());
}

struct CountingReader {
    input: io::Cursor<Vec<u8>>,
    read: Arc<AtomicUsize>,
}
impl AsyncRead for CountingReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let offset = self.input.position() as usize;
        let count = buf.remaining().min(self.input.get_ref().len() - offset);
        buf.put_slice(&self.input.get_ref()[offset..offset + count]);
        self.input.set_position((offset + count) as u64);
        self.read.fetch_add(count, Ordering::SeqCst);
        Poll::Ready(Ok(()))
    }
}

struct BlockedWriter {
    started: Option<tokio::sync::oneshot::Sender<()>>,
}
impl AsyncWrite for BlockedWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        _: &[u8],
    ) -> Poll<io::Result<usize>> {
        if let Some(started) = self.started.take() {
            let _ = started.send(());
        }
        Poll::Pending
    }
    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Pending
    }
    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn slow_output_bounds_input_read_ahead_and_is_cancellable() {
    let line = b"{\"op\":\"count\",\"text\":\"hello\"}\n";
    let read = Arc::new(AtomicUsize::new(0));
    let reader = BufReader::with_capacity(
        1,
        CountingReader {
            input: io::Cursor::new(line.repeat(1000)),
            read: read.clone(),
        },
    );
    let (started, ready) = tokio::sync::oneshot::channel();
    let cancel = Cancellation::default();
    let task = tokio::spawn(process(
        reader,
        BlockedWriter {
            started: Some(started),
        },
        support::pool(1, 1),
        BatchConfig::default(),
        cancel.clone(),
    ));
    tokio::time::timeout(Duration::from_secs(3), ready)
        .await
        .unwrap()
        .unwrap();
    // Give the producer an opportunity to fill its window, without a time-based assumption.
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    assert!(read.load(Ordering::SeqCst) <= line.len() * 2);
    cancel.cancel();
    assert!(tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap()
        .is_err());
}

#[tokio::test]
async fn empty_input_and_output_failures_terminate_cleanly() {
    let mut output = Vec::new();
    let summary = process(
        io::Cursor::new(Vec::new()),
        &mut output,
        support::pool(1, 0),
        BatchConfig::default(),
        Cancellation::default(),
    )
    .await
    .unwrap();
    assert_eq!(summary.records, 0);
    assert!(output.is_empty());
    let (input, mut feed) = tokio::io::duplex(256);
    let (output, drain) = tokio::io::duplex(256);
    drop(drain);
    let task = tokio::spawn(process(
        BufReader::new(input),
        output,
        support::pool(1, 0),
        BatchConfig::default(),
        Cancellation::default(),
    ));
    feed.write_all(b"{\"op\":\"count\",\"text\":\"hello\"}\n")
        .await
        .unwrap();
    assert!(tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .unwrap()
        .unwrap()
        .is_err());
}

#[tokio::test]
async fn batch_records_and_ids_only_share_the_jsonl_envelope() {
    let input = concat!(
        "{\"op\":\"batch\",\"operations\":[{\"op\":\"count\",\"text\":\"hello world\"},{\"op\":\"decode\",\"ids\":[99]}]}\n",
        "{\"op\":\"encode\",\"text\":\"hello\",\"ids_only\":true}\n",
        "{\"op\":\"batch\",\"operations\":[{\"op\":\"batch\",\"operations\":[]}]}\n",
    );
    let mut output = Vec::new();
    let summary = process(
        io::Cursor::new(input.as_bytes().to_vec()),
        &mut output,
        support::pool(2, 2),
        BatchConfig::default(),
        Cancellation::default(),
    )
    .await
    .unwrap();
    assert_eq!((summary.records, summary.errors), (3, 0));
    let rows: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows[0]["line"], 1);
    assert_eq!(rows[0]["result"]["schema_version"], 1);
    assert_eq!(rows[0]["result"]["results"][0]["count"], 2);
    assert_eq!(
        rows[0]["result"]["results"][1]["error"]["code"],
        "tokenizer_error"
    );
    assert_eq!(rows[1]["result"]["encoding"]["ids"], json!([4]));
    assert!(rows[1]["result"]["encoding"].get("offsets").is_none());
    assert_eq!(
        rows[2]["result"]["results"][0]["error"]["code"],
        "invalid_request"
    );
}
