//! Built-in live telemetry dashboard.
//!
//! - `GET /` — self-contained HTML page (`dashboard.html`, compiled in via `include_str!`)
//! - `GET /events` — Server-Sent Events stream: one `req` event per inference,
//!   plus `lagged` notices if a slow subscriber falls behind the broadcast buffer
//! - `GET /history` — last 50 requests, used by the page to backfill on load

use axum::{
    extract::State,
    response::{sse::{Event, KeepAlive, Sse}, Html},
    Json,
};
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::{
    wrappers::{errors::BroadcastStreamRecvError, BroadcastStream},
    Stream, StreamExt,
};

use crate::AppState;

// -----------------------------------------------------------------------------
// Telemetry payload
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub id: String,
    /// Unix epoch milliseconds
    pub ts: u64,
    pub latency_ms: f64,
    pub questions: u32,
    /// premise-hypothesis pairs sent in the forward pass
    pub pairs: usize,
    pub tokens: u64,
    pub state_preview: String,
    pub answers: serde_json::Value,
    // rolling totals after this request (for dashboard tiles)
    pub total_requests: u64,
    pub total_tokens: u64,
    pub avg_latency_ms: f64,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub total: u64,
    pub total_tokens: u64,
    pub total_latency_ms: f64,
}

// -----------------------------------------------------------------------------
// Handlers
// -----------------------------------------------------------------------------
pub async fn dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

pub async fn history(State(app): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let entries = app.history.read().unwrap().clone();
    Json(serde_json::json!({
        "server_start_ms": app.start_ts,
        "entries": entries,
    }))
}

/// Subscribe to the inference broadcast and forward entries as SSE events.
pub async fn sse(
    State(app): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>> + Send + 'static> {
    let rx = app.tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|item| match item {
        Ok(e) => Some(Ok(Event::default().event("req").data(
            serde_json::to_string(&e).unwrap_or_else(|_| "{}".to_string()),
        ))),
        Err(BroadcastStreamRecvError::Lagged(n)) => Some(Ok(
            Event::default().event("lagged").data(n.to_string()),
        )),
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub const DASHBOARD_HTML: &str = include_str!("dashboard.html");
