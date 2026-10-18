//! HTTP interface. Admission is checked before buffering request JSON.
use std::time::Duration;

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, FromRequestParts, Path, State},
    http::{request::Parts, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::sync::OwnedSemaphorePermit;

use crate::work::{Cancellation, CpuPool, Operation, WorkError};

#[derive(Clone, Copy)]
pub struct ApiConfig {
    pub max_body_bytes: usize,
    pub request_timeout: Duration,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            max_body_bytes: 1024 * 1024,
            request_timeout: Duration::from_secs(30),
        }
    }
}

pub fn router(pool: CpuPool, config: ApiConfig) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/info", get(info))
        .route("/v1/{operation}", post(process))
        .fallback(|| async { failure(StatusCode::NOT_FOUND, "not_found", "unknown endpoint") })
        .layer(DefaultBodyLimit::max(config.max_body_bytes))
        .layer(middleware::from_fn(
            move |request, next: middleware::Next| async move {
                match tokio::time::timeout(config.request_timeout, next.run(request)).await {
                    Ok(response) => response,
                    Err(_) => failure(
                        StatusCode::REQUEST_TIMEOUT,
                        "timeout",
                        "request deadline exceeded",
                    ),
                }
            },
        ))
        .with_state(pool)
}

async fn health(State(pool): State<CpuPool>) -> Response {
    if pool.shutdown().is_cancelled() {
        failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "cancelled",
            "server is shutting down",
        )
    } else {
        Json(json!({"status": "ok"})).into_response()
    }
}

async fn info(State(pool): State<CpuPool>) -> Json<Value> {
    Json(json!({"name": "fast-tokenizer", "version": crate::VERSION,
        "artifact": pool.engine().artifact(), "options": pool.engine().options(),
        "capacity": pool.capacity()}))
}

struct Admission(OwnedSemaphorePermit);

impl FromRequestParts<CpuPool> for Admission {
    type Rejection = Response;

    async fn from_request_parts(_: &mut Parts, pool: &CpuPool) -> Result<Self, Self::Rejection> {
        pool.reserve().map(Self).map_err(work_failure)
    }
}

async fn process(
    State(pool): State<CpuPool>,
    Path(name): Path<String>,
    Admission(permit): Admission,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    let mut payload = match payload {
        Ok(Json(value)) => value,
        Err(error) => return failure(error.status(), "invalid_json", &error.body_text()),
    };
    if name != "process" {
        if !["count", "encode", "decode", "truncate", "chunk"].contains(&name.as_str()) {
            return failure(StatusCode::NOT_FOUND, "not_found", "unknown operation");
        }
        let Some(object) = payload.as_object_mut() else {
            return failure(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
                "expected a JSON object",
            );
        };
        if object.contains_key("op") {
            return failure(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
                "use op only with /v1/process",
            );
        }
        object.insert("op".to_owned(), json!(name));
    }
    let operation: Operation = match serde_json::from_value(payload) {
        Ok(operation) => operation,
        Err(error) => {
            return failure(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request",
                &error.to_string(),
            )
        }
    };
    let cancel = Cancellation::default();
    let _guard = cancel.guard();
    let ticket = pool.submit(operation, cancel, permit);
    match ticket.await {
        Ok(Ok(bytes)) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response(),
        Ok(Err(error)) => work_failure(error),
        Err(_) => work_failure(WorkError::Failed),
    }
}

fn work_failure(error: WorkError) -> Response {
    let status = match &error {
        WorkError::Busy | WorkError::Cancelled | WorkError::Engine(crate::Error::Cancelled) => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        WorkError::Engine(_) => StatusCode::UNPROCESSABLE_ENTITY,
        WorkError::Failed => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let mut response = failure(status, error.code(), &error.to_string());
    if matches!(error, WorkError::Busy) {
        response.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("1"),
        );
    }
    response
}

fn failure(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error": {"code": code, "message": message}})),
    )
        .into_response()
}
