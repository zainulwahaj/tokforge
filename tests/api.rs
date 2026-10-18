mod support;

use std::time::Duration;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use fast_tokenizer::api::{router, ApiConfig};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

async fn call(app: &Router, path: &str, payload: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn all_operations_and_metadata_use_the_shared_engine() {
    let app = router(support::pool(2, 2), ApiConfig::default());
    for path in ["/health", "/v1/info"] {
        let response = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let (status, count) = call(&app, "/v1/count", json!({"text": "hello café 你好!"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(count["count"], 4);
    assert_eq!(count["options"]["special_tokens"], "literal");
    assert_eq!(count["artifact"]["sha256"].as_str().unwrap().len(), 64);
    let (status, encoded) = call(&app, "/v1/encode", json!({"text": "hello world!"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(encoded["encoding"]["ids"], json!([4, 5, 6]));
    let (_, decoded) = call(
        &app,
        "/v1/decode",
        json!({"ids": [1,4,2], "skip_special_tokens": true}),
    )
    .await;
    assert_eq!(decoded["text"], "hello");
    let (_, truncated) = call(
        &app,
        "/v1/truncate",
        json!({"text": "hello world!", "max_tokens": 1}),
    )
    .await;
    assert_eq!(truncated["truncated"], true);
    assert!(truncated["result"]["count"].as_u64().unwrap() <= 1);
    let (_, chunked) = call(
        &app,
        "/v1/process",
        json!({"op": "chunk", "text": "hello\r\nworld! café", "max_tokens": 2}),
    )
    .await;
    let source: String = chunked["chunks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["text"].as_str().unwrap())
        .collect();
    assert_eq!(source, "hello\r\nworld! café");
}

#[tokio::test]
async fn rejects_invalid_requests_and_enforces_body_limit() {
    let app = router(
        support::pool(1, 0),
        ApiConfig {
            max_body_bytes: 128,
            ..Default::default()
        },
    );
    for (path, payload) in [
        ("/v1/count", json!({"text": "hello", "typo": true})),
        ("/v1/count", json!({"text": 3})),
        ("/v1/count", json!({"op": "count", "text": "hello"})),
        ("/v1/decode", json!({"ids": [99]})),
        ("/v1/decode", json!({"ids": [-1]})),
        ("/v1/chunk", json!({"text": "hello", "max_tokens": 0})),
        ("/v1/process", json!({"op": "unknown", "text": "hello"})),
    ] {
        let (status, output) = call(&app, path, payload).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{output}");
        assert!(output["error"]["message"].is_string());
    }
    assert_eq!(
        call(&app, "/v1/count", json!({"text": "x".repeat(200)}))
            .await
            .0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    assert_eq!(
        call(&app, "/v1/missing", json!({})).await.0,
        StatusCode::NOT_FOUND
    );
    for (content_type, body, expected) in [
        ("application/json", "{", StatusCode::BAD_REQUEST),
        ("text/plain", "{}", StatusCode::UNSUPPORTED_MEDIA_TYPE),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::post("/v1/count")
                    .header("content-type", content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
    }
}

#[tokio::test]
async fn capacity_is_checked_before_body_parsing_and_released_after_errors() {
    let pool = support::pool(1, 0);
    let app = router(pool.clone(), ApiConfig::default());
    let slot = pool.reserve().unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/count")
                .body(Body::from("bad json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()["retry-after"], "1");
    drop(slot);
    assert_eq!(
        call(&app, "/v1/count", json!({"text": "hello"})).await.0,
        StatusCode::OK
    );
    assert_eq!(
        call(&app, "/v1/count", json!({"text": false})).await.0,
        StatusCode::UNPROCESSABLE_ENTITY
    );
    assert!(pool.reserve().is_ok());
    pool.shutdown().cancel();
    assert_eq!(
        call(&app, "/v1/count", json!({"text": "hello"})).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn slow_request_bodies_time_out_and_release_admission() {
    let pool = support::pool(1, 0);
    let app = router(
        pool.clone(),
        ApiConfig {
            request_timeout: Duration::from_millis(50),
            ..Default::default()
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
    socket.write_all(b"POST /v1/count HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{").await.unwrap();
    let mut output = String::new();
    tokio::time::timeout(Duration::from_secs(3), socket.read_to_string(&mut output))
        .await
        .unwrap()
        .unwrap();
    assert!(output.starts_with("HTTP/1.1 408"), "{output}");
    assert!(pool.reserve().is_ok());
    server.abort();
}
