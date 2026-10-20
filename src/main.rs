mod dashboard;

use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use dashboard::{LogEntry, Stats};
#[cfg(feature = "dml")]
use ort::ep::DirectML;
use ort::session::Session;
use ort::value::Tensor;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, Mutex, Semaphore};
use uuid::Uuid;

// Entailment class index used when the model's config.json does not declare
// a recognizable one (see `load_model_meta`).
const ENTAILMENT_IDX: usize = 0;

/// Fallback position limit when the model declares none.
const DEFAULT_MAX_POSITIONS: usize = 8194;

/// Default character limit for `state` (JEV_MAX_STATE_CHARS). A 2 000-character
/// premise encodes to roughly 600-700 tokens.
const DEFAULT_MAX_STATE_CHARS: usize = 2000;

/// Default token limit per encoded pair (JEV_MAX_SEQ_TOKENS).
const DEFAULT_MAX_SEQ_TOKENS: usize = 2048;

/// Default token budget per forward pass (JEV_MAX_BATCH_TOKENS) — the real
/// driver of the ONNX cost.
const DEFAULT_MAX_BATCH_TOKENS: usize = 8192;

/// Identity of this engine, reported in every response.
const ENGINE_MODEL_ID: &str = "local-jev-rust";

// -----------------------------------------------------------------------------
// TypeSafe Wire Protocol Data Structures
// -----------------------------------------------------------------------------

/// Form in which the criteria of a question were supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CriteriaForm {
    /// `{"billing": "Payment issues", ...}` — the key is the answer label.
    Object,
    /// `["Calm", "Annoyed", ...]` — the item is both label and description.
    Array,
}

/// Criteria of a `choice` or `score` question.
///
/// Deserialized by hand instead of via `serde_json::Value` for two reasons:
///   * malformed entries are collected, not rejected, so the error can be
///     reported with the question id;
///   * duplicates are detected at all. A duplicate collapses in the
///     probability map while the answer value is still computed over the full
///     list, which silently produced distributions that did not sum to 1.
#[derive(Debug, Clone, Default)]
struct Criteria {
    /// `None` when the criteria were neither an object nor an array.
    form: Option<CriteriaForm>,
    /// (label, description) pairs in request order.
    entries: Vec<(String, String)>,
    /// Labels (objects) or descriptions (arrays) that occurred more than once.
    duplicates: Vec<String>,
    /// Values that were not strings, rendered for the error message.
    invalid: Vec<String>,
}

impl Criteria {
    /// Labels in request order — the keys of the probability map.
    fn labels(&self) -> Vec<String> {
        self.entries.iter().map(|(k, _)| k.clone()).collect()
    }
}

impl<'de> Deserialize<'de> for Criteria {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct CriteriaVisitor;

        impl<'de> Visitor<'de> for CriteriaVisitor {
            type Value = Criteria;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a criteria object or an array of strings")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Criteria, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut crit = Criteria {
                    form: Some(CriteriaForm::Object),
                    ..Default::default()
                };
                while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                    let desc = match value {
                        serde_json::Value::String(s) => s,
                        other => {
                            crit.invalid.push(other.to_string());
                            continue;
                        }
                    };
                    if crit.entries.iter().any(|(k, _)| k == &key) {
                        crit.duplicates.push(key);
                        continue;
                    }
                    crit.entries.push((key, desc));
                }
                Ok(crit)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Criteria, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut crit = Criteria {
                    form: Some(CriteriaForm::Array),
                    ..Default::default()
                };
                while let Some(value) = seq.next_element::<serde_json::Value>()? {
                    let desc = match value {
                        serde_json::Value::String(s) => s,
                        other => {
                            crit.invalid.push(other.to_string());
                            continue;
                        }
                    };
                    if crit.entries.iter().any(|(_, d)| d == &desc) {
                        crit.duplicates.push(desc.clone());
                        continue;
                    }
                    crit.entries.push((desc.clone(), desc));
                }
                Ok(crit)
            }

            // Anything that is neither an object nor an array is remembered as
            // an invalid entry; `build_tasks` turns it into a precise error.
            fn visit_str<E>(self, v: &str) -> Result<Criteria, E>
            where
                E: serde::de::Error,
            {
                Ok(Criteria {
                    invalid: vec![v.to_string()],
                    ..Default::default()
                })
            }

            fn visit_unit<E>(self) -> Result<Criteria, E>
            where
                E: serde::de::Error,
            {
                Ok(Criteria {
                    invalid: vec!["null".to_string()],
                    ..Default::default()
                })
            }

            fn visit_bool<E>(self, v: bool) -> Result<Criteria, E>
            where
                E: serde::de::Error,
            {
                Ok(Criteria {
                    invalid: vec![v.to_string()],
                    ..Default::default()
                })
            }

            fn visit_i64<E>(self, v: i64) -> Result<Criteria, E>
            where
                E: serde::de::Error,
            {
                Ok(Criteria {
                    invalid: vec![v.to_string()],
                    ..Default::default()
                })
            }

            fn visit_u64<E>(self, v: u64) -> Result<Criteria, E>
            where
                E: serde::de::Error,
            {
                Ok(Criteria {
                    invalid: vec![v.to_string()],
                    ..Default::default()
                })
            }

            fn visit_f64<E>(self, v: f64) -> Result<Criteria, E>
            where
                E: serde::de::Error,
            {
                Ok(Criteria {
                    invalid: vec![v.to_string()],
                    ..Default::default()
                })
            }
        }

        deserializer.deserialize_any(CriteriaVisitor)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct QuestionSchema {
    #[serde(rename = "type")]
    q_type: String, // "noul", "choice", "score"
    instructions: Option<String>,
    criteria: Option<Criteria>,
}

#[derive(Debug, Deserialize)]
struct SystemOneRequest {
    /// Accepted for wire compatibility, never echoed back: the response always
    /// reports the engine's own identity.
    #[allow(dead_code)]
    model: Option<String>,
    state: serde_json::Value,
    questions: BTreeMap<String, QuestionSchema>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnswerValue {
    Noul {
        #[serde(rename = "type")]
        q_type: String,
        noul: f32,
    },
    Choice {
        #[serde(rename = "type")]
        q_type: String,
        choice: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Score {
        #[serde(rename = "type")]
        q_type: String,
        score: f32,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
}

#[derive(Debug, Serialize)]
struct SystemOneResponse {
    id: String,
    model: String,
    answers: BTreeMap<String, AnswerValue>,
    usage: serde_json::Value,
}

struct AppState {
    tokenizer: Tokenizer,
    /// Arc-wrapped so the guard can be owned ('static) inside the
    /// spawn_blocking closure running the forward pass.
    session: Arc<Mutex<Session>>,
    temperature: f32,
    /// Resolved ONNX output name for the classification logits.
    logits_name: String,
    /// Entailment class index resolved from the model's config.json.
    entailment_idx: usize,
    /// Character limit for `state` (JEV_MAX_STATE_CHARS).
    max_state_chars: usize,
    /// Token limit per encoded pair (JEV_MAX_SEQ_TOKENS).
    max_seq_tokens: usize,
    /// Token budget per forward pass (JEV_MAX_BATCH_TOKENS).
    max_batch_tokens: usize,
    /// Per-request inference timeout in seconds (JEV_TIMEOUT).
    inference_timeout: u64,
    /// Inference slots; excess requests get 503 (JEV_MAX_CONCURRENCY).
    inference_permits: Arc<Semaphore>,
    /// Optional API key for the inference endpoint (JEV_API_KEY).
    api_key: Option<String>,
    /// When false, no request states are retained or broadcast (JEV_TELEMETRY).
    telemetry: bool,
    tx: broadcast::Sender<LogEntry>,
    history: RwLock<Vec<LogEntry>>,
    stats: RwLock<Stats>,
    start_ts: u64,
}

// -----------------------------------------------------------------------------
// Calibration (RLCD) & Information Entropy Mathematics
// -----------------------------------------------------------------------------
fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter()
        .map(|&x| (x / sum * 10000.0).round() / 10000.0)
        .collect()
}

/// JSON error envelope for consistent, machine-readable API errors:
/// `{"error": {"code": <status>, "message": "..."}}`.
#[derive(Debug, Clone)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }

    fn service_unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg.into(),
        }
    }

    fn gateway_timeout(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: msg.into(),
        }
    }

    fn unauthorized(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg.into(),
        }
    }

    /// Map an axum extractor rejection (bad JSON body, missing content-type,
    /// body over the size limit) into the same envelope.
    ///
    /// The wire protocol answers `400` for every malformed request; only an
    /// over-sized body keeps its `413`. Axum would otherwise answer `422` for
    /// a deserialization error and `415` for a missing content type.
    fn from_rejection(rejection: JsonRejection) -> Self {
        let status = match rejection.status() {
            StatusCode::PAYLOAD_TOO_LARGE => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::BAD_REQUEST,
        };
        Self {
            status,
            message: rejection.body_text(),
        }
    }
}

impl From<ApiError> for (StatusCode, String) {
    fn from(e: ApiError) -> Self {
        (e.status, e.message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = self.status.as_u16();
        let status = self.status;
        (
            status,
            Json(serde_json::json!({
                "error": { "code": code, "message": self.message }
            })),
        )
            .into_response()
    }
}

/// Model metadata resolved from the exported model's `config.json`.
#[derive(Debug, Clone)]
struct ModelMeta {
    /// Index of the entailment class in the classification logits.
    entailment_idx: usize,
    /// Maximum number of sequence positions the model can consume.
    max_positions: usize,
    /// Size of the embedding matrix (the tokenizer must not exceed it).
    vocab_size: Option<usize>,
}

/// Labels that must never be mistaken for the entailment class: a substring
/// match on "not_entailment" would invert every probability reported.
const NEGATIVE_NLI_LABELS: &[&str] = &[
    "not_entailment",
    "not entailment",
    "non_entailment",
    "non-entailment",
    "contradiction",
    "contradictory",
    "neutral",
    "unrelated",
    "false",
    "no",
];

/// Pick the entailment class out of an `id2label` map.
///
/// Resolution order:
///   1. exact `entailment`
///   2. exact positive synonyms (`true`, `yes`, `positive`) used by binary
///      zeroshot exports
///   3. a label that contains `entailment` but is not a negated form
fn resolve_entailment_label(id2label: &BTreeMap<String, String>) -> Option<(usize, String)> {
    let parse = |k: &String| k.parse::<usize>().ok();

    for (k, v) in id2label {
        if v.trim().eq_ignore_ascii_case("entailment") {
            if let Some(i) = parse(k) {
                return Some((i, v.clone()));
            }
        }
    }

    for (k, v) in id2label {
        let label = v.trim().to_lowercase();
        if matches!(label.as_str(), "true" | "yes" | "positive") {
            if let Some(i) = parse(k) {
                return Some((i, v.clone()));
            }
        }
    }

    for (k, v) in id2label {
        let label = v.trim().to_lowercase();
        let negated = NEGATIVE_NLI_LABELS.iter().any(|n| label == *n)
            || label.contains("not_entailment")
            || label.contains("non_entailment")
            || label.starts_with("not ")
            || label.starts_with("non-")
            || label.starts_with("un");
        if label.contains("entailment") && !negated {
            if let Some(i) = parse(k) {
                return Some((i, v.clone()));
            }
        }
    }

    None
}

/// Read the model's `config.json` (next to the ONNX file) and resolve the
/// entailment class index, the position limit and the vocabulary size.
/// Falls back to safe defaults with a warning.
fn load_model_meta(model_path: &str) -> ModelMeta {
    let config_path = std::path::Path::new(model_path).with_file_name("config.json");
    let raw = std::fs::read_to_string(&config_path).ok();

    let config = raw.and_then(|s| {
        // Tolerate a UTF-8 BOM: `from_str` would fail on it and silently
        // downgrade the label resolution to the fallback.
        let s = s.strip_prefix('\u{feff}').unwrap_or(&s);
        serde_json::from_str::<serde_json::Value>(s).ok()
    });

    let config = match config {
        Some(c) => c,
        None => {
            println!(
                "[!] config.json not readable next to '{}'; defaulting entailment index to {}",
                model_path, ENTAILMENT_IDX
            );
            return ModelMeta {
                entailment_idx: ENTAILMENT_IDX,
                max_positions: DEFAULT_MAX_POSITIONS,
                vocab_size: None,
            };
        }
    };

    let max_positions = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_MAX_POSITIONS);

    let vocab_size = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .filter(|&v| v > 0);

    let id2label: BTreeMap<String, String> = config
        .get("id2label")
        .and_then(|v| v.as_object())
        .map(|map| {
            map.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();

    let entailment_idx = match resolve_entailment_label(&id2label) {
        Some((idx, label)) => {
            println!(
                "[*] Entailment label: '{}' at index {} (from id2label)",
                label, idx
            );
            idx
        }
        None => {
            println!(
                "[!] config.json has no recognizable NLI labels; defaulting entailment index to {}",
                ENTAILMENT_IDX
            );
            ENTAILMENT_IDX
        }
    };

    ModelMeta {
        entailment_idx,
        max_positions,
        vocab_size,
    }
}

/// Strip absolute filesystem paths and ONNX Runtime internals out of an error
/// message before it reaches the client.
fn sanitize_error_detail(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    let bytes: Vec<char> = msg.chars().collect();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];
        // Windows drive path: "C:\..." / "C:/..."
        let drive_start = c.is_ascii_alphabetic()
            && i + 2 < bytes.len()
            && bytes[i + 1] == ':'
            && (bytes[i + 2] == '\\' || bytes[i + 2] == '/');
        // UNC path: "\\server\share"
        let unc_start = c == '\\' && i + 1 < bytes.len() && bytes[i + 1] == '\\';

        if drive_start || unc_start {
            let mut j = i;
            while j < bytes.len() && !bytes[j].is_whitespace() {
                j += 1;
            }
            out.push_str("<path>");
            i = j;
            continue;
        }

        if c == '\n' || c == '\r' {
            out.push(' ');
            i += 1;
            continue;
        }

        out.push(c);
        i += 1;
    }

    let cleaned = out.trim();
    if cleaned.chars().count() > 240 {
        let truncated: String = cleaned.chars().take(240).collect();
        format!("{}…", truncated)
    } else {
        cleaned.to_string()
    }
}

/// Reject an over-long `state` before any tokenization or inference happens:
/// a long premise is quadratic in the model's cost and overflows the position
/// embeddings.
fn validate_state_length(state: &str, max_chars: usize) -> Result<(), ApiError> {
    if state.trim().is_empty() {
        return Err(ApiError::bad_request(
            "state must not be empty — the engine judges a state, it cannot judge nothing",
        ));
    }
    let len = state.chars().count();
    if len > max_chars {
        return Err(ApiError::bad_request(format!(
            "state is too long: {} characters, maximum is {} (JEV_MAX_STATE_CHARS)",
            len, max_chars
        )));
    }
    Ok(())
}

/// Reject a batch that exceeds the sequence or token budget of the forward
/// pass. Enforced before the ONNX run, so an oversized request is a clean 400
/// instead of an ONNX Runtime failure.
fn validate_batch_limits(
    batch_size: usize,
    seq_len: usize,
    max_seq_tokens: usize,
    max_batch_tokens: usize,
) -> Result<(), ApiError> {
    if seq_len > max_seq_tokens {
        return Err(ApiError::bad_request(format!(
            "encoded sequence is too long: {} tokens, maximum is {} (JEV_MAX_SEQ_TOKENS)",
            seq_len, max_seq_tokens
        )));
    }
    let total = batch_size.saturating_mul(seq_len);
    if total > max_batch_tokens {
        return Err(ApiError::bad_request(format!(
            "batch is too large: {} pairs x {} tokens = {} tokens, maximum is {} (JEV_MAX_BATCH_TOKENS)",
            batch_size, seq_len, total, max_batch_tokens
        )));
    }
    Ok(())
}

fn compute_confidence(probs: &[f32]) -> f32 {
    let k = probs.len() as f32;
    if k <= 1.0 {
        return 1.0;
    }
    let mut entropy = 0.0;
    for &p in probs {
        if p > 1e-12 {
            entropy -= p * p.log2();
        }
    }
    let conf = 1.0 - (entropy / k.log2());
    (conf.clamp(0.0, 1.0) * 10000.0).round() / 10000.0
}

#[derive(Debug)]
enum TaskKind {
    Noul,
    Choice { keys: Vec<String> },
    Score { levels: Vec<String> },
}

#[derive(Debug)]
struct TaskMeta {
    id: String,
    start: usize,
    end: usize,
    kind: TaskKind,
}

/// The evaluation pairs and the per-question metadata used to slice the
/// entailment logits.
type TaskPlan = (Vec<(String, String)>, Vec<TaskMeta>);

/// Flatten the wire-protocol questions into (premise, hypothesis) pairs and
/// the per-question metadata used to slice the entailment logits. Rejects
/// malformed questions with a JSON 400 error.
fn build_tasks(
    state_text: &str,
    questions: &BTreeMap<String, QuestionSchema>,
) -> Result<TaskPlan, ApiError> {
    let mut premise_hypo_pairs: Vec<(String, String)> = Vec::new();
    let mut tasks: Vec<TaskMeta> = Vec::new();

    for (q_id, q) in questions {
        let instr = q.instructions.as_deref().unwrap_or("");
        let start_idx = premise_hypo_pairs.len();

        match q.q_type.to_lowercase().as_str() {
            "noul" => {
                premise_hypo_pairs.push((state_text.to_string(), format!("{}: Yes, true.", instr)));
                premise_hypo_pairs.push((state_text.to_string(), format!("{}: No, false.", instr)));
                tasks.push(TaskMeta {
                    id: q_id.clone(),
                    start: start_idx,
                    end: premise_hypo_pairs.len(),
                    kind: TaskKind::Noul,
                });
            }
            "choice" => {
                let crit = q.criteria.as_ref().ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "question '{}': type 'choice' requires a 'criteria' object or array of strings",
                        q_id
                    ))
                })?;

                match crit.form {
                    None => {
                        return Err(ApiError::bad_request(format!(
                            "question '{}': type 'choice' requires a 'criteria' object or array of strings",
                            q_id
                        )));
                    }
                    Some(CriteriaForm::Object) => {
                        if !crit.invalid.is_empty() {
                            return Err(ApiError::bad_request(format!(
                                "question '{}': 'choice' criteria values must be strings",
                                q_id
                            )));
                        }
                    }
                    Some(CriteriaForm::Array) => {
                        if !crit.invalid.is_empty() {
                            return Err(ApiError::bad_request(format!(
                                "question '{}': 'choice' criteria array items must be strings",
                                q_id
                            )));
                        }
                    }
                }

                // A duplicate would collapse in the probability map while the
                // winner is still chosen over the full list.
                if let Some(dup) = crit.duplicates.first() {
                    return Err(ApiError::bad_request(format!(
                        "question '{}': duplicate criterion '{}'",
                        q_id, dup
                    )));
                }

                if crit.entries.is_empty() {
                    return Err(ApiError::bad_request(format!(
                        "question '{}': 'choice' criteria must not be empty",
                        q_id
                    )));
                }

                let keys = crit.labels();
                for (_, desc) in &crit.entries {
                    premise_hypo_pairs
                        .push((state_text.to_string(), format!("{}: {}", instr, desc)));
                }
                tasks.push(TaskMeta {
                    id: q_id.clone(),
                    start: start_idx,
                    end: premise_hypo_pairs.len(),
                    kind: TaskKind::Choice { keys },
                });
            }
            "score" => {
                // `score` is ordinal, so the criteria must be an ordered
                // array; missing criteria are rejected, never defaulted.
                let crit = q.criteria.as_ref().ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "question '{}': type 'score' requires a 'criteria' array of strings",
                        q_id
                    ))
                })?;

                if crit.form != Some(CriteriaForm::Array) {
                    return Err(ApiError::bad_request(format!(
                        "question '{}': type 'score' requires a 'criteria' array of strings",
                        q_id
                    )));
                }
                if !crit.invalid.is_empty() {
                    return Err(ApiError::bad_request(format!(
                        "question '{}': 'score' criteria array items must be strings",
                        q_id
                    )));
                }
                if let Some(dup) = crit.duplicates.first() {
                    return Err(ApiError::bad_request(format!(
                        "question '{}': duplicate criterion '{}'",
                        q_id, dup
                    )));
                }
                if crit.entries.is_empty() {
                    return Err(ApiError::bad_request(format!(
                        "question '{}': 'score' criteria must not be empty",
                        q_id
                    )));
                }

                let levels = crit.labels();
                for lvl in &levels {
                    premise_hypo_pairs
                        .push((state_text.to_string(), format!("{}: {}", instr, lvl)));
                }
                tasks.push(TaskMeta {
                    id: q_id.clone(),
                    start: start_idx,
                    end: premise_hypo_pairs.len(),
                    kind: TaskKind::Score { levels },
                });
            }
            _ => {
                return Err(ApiError::bad_request(format!(
                    "Unsupported question type: {}",
                    q.q_type
                )));
            }
        }
    }

    // An empty question map produces no evaluation pairs at all.
    if premise_hypo_pairs.is_empty() {
        return Err(ApiError::bad_request(
            "Request must contain at least one valid question that produces evaluation pairs",
        ));
    }

    Ok((premise_hypo_pairs, tasks))
}

// -----------------------------------------------------------------------------
// Request Handler (Single Forward Pass)
// -----------------------------------------------------------------------------

/// Compare two secrets without leaking their relationship through timing.
fn secrets_match(provided: &str, expected: &str) -> bool {
    let a = provided.as_bytes();
    let b = expected.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Guard the inference endpoint with an optional API key (`JEV_API_KEY`).
///
/// The key is accepted as `X-API-Key: <key>` or as
/// `Authorization: Bearer <key>`. Without the variable the endpoint stays
/// open, which is the default for a loopback-only deployment.
async fn require_api_key(
    State(app): State<Arc<AppState>>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<Response, ApiError> {
    if let Some(expected) = app.api_key.as_deref() {
        let provided = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .or_else(|| {
                headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.strip_prefix("Bearer "))
                    .map(|s| s.trim().to_string())
            });

        let ok = provided
            .as_deref()
            .map(|p| secrets_match(p, expected))
            .unwrap_or(false);

        if !ok {
            return Err(ApiError::unauthorized(
                "Missing or invalid API key (X-API-Key or Authorization: Bearer)",
            ));
        }
    }

    Ok(next.run(request).await)
}

async fn handle_systemone(
    State(app): State<Arc<AppState>>,
    payload: Result<Json<SystemOneRequest>, JsonRejection>,
) -> Result<impl IntoResponse, ApiError> {
    let Json(payload) = payload.map_err(ApiError::from_rejection)?;

    let start_time = Instant::now();
    let req_id = format!("so_{}", &Uuid::new_v4().to_string().replace("-", "")[..12]);

    let state_text = match &payload.state {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    validate_state_length(&state_text, app.max_state_chars)?;

    let (mut premise_hypo_pairs, tasks) = build_tasks(&state_text, &payload.questions)?;

    // Vectorized batch tokenization. CPU-bound — offloaded to the blocking
    // pool so async worker threads stay responsive for /health, SSE, etc.
    //
    // The unique input size (the premise once plus every hypothesis) is
    // counted in the same pass: reporting the padded batch sum instead
    // multiplied the premise by the number of questions and made
    // `usage.input_tokens` meaningless for the caller.
    let (encodings, unique_input_tokens) = {
        let pairs = std::mem::take(&mut premise_hypo_pairs);
        let app_ref = Arc::clone(&app);
        match tokio::task::spawn_blocking(
            move || -> Result<(Vec<tokenizers::Encoding>, u64), tokenizers::Error> {
                let token_inputs: Vec<(&str, &str)> = pairs
                    .iter()
                    .map(|(p, h)| (p.as_str(), h.as_str()))
                    .collect();
                let encodings = app_ref.tokenizer.encode_batch(token_inputs, true)?;

                let premise = pairs.first().map(|(p, _)| p.as_str()).unwrap_or("");
                let premise_tokens = app_ref.tokenizer.encode(premise, true)?.len();
                let mut hypothesis_tokens = 0usize;
                for (_, hypothesis) in &pairs {
                    hypothesis_tokens += app_ref.tokenizer.encode(hypothesis.as_str(), true)?.len();
                }
                Ok((encodings, (premise_tokens + hypothesis_tokens) as u64))
            },
        )
        .await
        {
            Ok(Ok(res)) => res,
            Ok(Err(e)) => return Err(ApiError::internal(e.to_string())),
            Err(e) => {
                return Err(ApiError::internal(format!(
                    "Tokenization task failed: {}",
                    e
                )))
            }
        }
    };

    let batch_size = encodings.len();
    let seq_len = encodings
        .first()
        .map(|e| e.get_ids().len())
        .ok_or_else(|| ApiError::internal("Tokenizer produced no encodings"))?;

    validate_batch_limits(
        batch_size,
        seq_len,
        app.max_seq_tokens,
        app.max_batch_tokens,
    )?;

    let mut flat_input_ids: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
    let mut flat_attention_mask: Vec<i64> = Vec::with_capacity(batch_size * seq_len);

    for enc in &encodings {
        for &tok in enc.get_ids() {
            flat_input_ids.push(tok as i64);
        }
        for &msk in enc.get_attention_mask() {
            flat_attention_mask.push(msk as i64);
        }
    }

    // Real (unpadded) token count across the whole batch — the amount of
    // compute the forward pass actually costs.
    let batch_tokens = flat_attention_mask.iter().filter(|&&m| m == 1).count() as u64;

    let shape = [batch_size, seq_len];
    let input_ids_tensor = Tensor::from_array((shape, flat_input_ids))
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let attention_mask_tensor = Tensor::from_array((shape, flat_attention_mask))
        .map_err(|e| ApiError::internal(e.to_string()))?;

    // Single forward pass (GPU / DirectML), bounded by a concurrency permit
    // (overload -> 503) and a per-request timeout (-> 504). Logits extraction
    // happens inside the closure because the run output borrows the guard.
    let permit = app
        .inference_permits
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::service_unavailable("Server busy: too many concurrent inferences")
        })?;

    let entail_idx = app.entailment_idx;
    let logits_name = app.logits_name.clone();
    let session_mutex = Arc::clone(&app.session);
    let timeout = Duration::from_secs(app.inference_timeout);
    let batch_size_in = batch_size;

    // The timeout covers the *whole* chain - waiting for the session lock and
    // the forward pass alike - so a queued request cannot wait indefinitely.
    //
    // The permit travels into the blocking task and is released only when the
    // ONNX run really finishes: a timed-out inference keeps occupying its
    // slot instead of letting new requests pile up behind it.
    let run_result = tokio::time::timeout(timeout, async move {
        let mut session_guard = session_mutex.lock_owned().await;
        tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
            let _permit = permit;
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ];
            let outputs = session_guard.run(inputs).map_err(|e| e.to_string())?;
            let (out_shape, logits_data) = outputs[logits_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;

            // The output must be a 2D classification result
            // [batch, num_classes]; return the owned entailment logits.
            if out_shape.len() != 2 {
                return Err(format!(
                    "Model output must be 2D [batch, num_classes], got rank {}",
                    out_shape.len()
                ));
            }
            let num_rows = out_shape[0] as usize;
            let num_classes = out_shape[1] as usize;
            if num_classes <= entail_idx {
                return Err(format!(
                    "Model has {} classes, but entailment index {} is required",
                    num_classes, entail_idx
                ));
            }
            if num_rows != batch_size_in {
                return Err(format!(
                    "Model output batch mismatch: got {} rows for a batch of {}",
                    num_rows, batch_size_in
                ));
            }

            let mut entail_logits = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                entail_logits.push(logits_data[i * num_classes + entail_idx]);
            }
            Ok(entail_logits)
        })
        .await
    })
    .await;

    let entail_logits = match run_result {
        Err(_) => {
            return Err(ApiError::gateway_timeout(format!(
                "Inference timed out after {}s (queue time for the inference slot included)",
                app.inference_timeout
            )))
        }
        Ok(Err(join_err)) => {
            return Err(ApiError::internal(format!(
                "Inference task failed: {}",
                join_err
            )))
        }
        Ok(Ok(Err(e))) => {
            // The full ONNX Runtime error goes to the log, the client gets a
            // sanitized summary.
            eprintln!("[!] ONNX inference error ({}): {}", req_id, e);
            return Err(ApiError::internal(sanitize_error_detail(&e)));
        }
        Ok(Ok(Ok(entail))) => entail,
    };

    let num_questions = tasks.len();
    let mut answers = BTreeMap::new();

    for task in tasks {
        // Sliced defensively: a panic here would take down the whole process.
        let task_logits = match entail_logits.get(task.start..task.end) {
            Some(slice) => slice,
            None => {
                return Err(ApiError::internal(format!(
                    "Model output is shorter than the expected batch (question '{}')",
                    task.id
                )))
            }
        };
        let probs = softmax(task_logits, app.temperature);

        match task.kind {
            TaskKind::Noul => {
                let noul = probs.first().copied().unwrap_or(0.0);
                answers.insert(
                    task.id,
                    AnswerValue::Noul {
                        q_type: "noul".into(),
                        noul,
                    },
                );
            }
            TaskKind::Choice { keys } => {
                let mut prob_map = BTreeMap::new();
                let mut best_idx = 0;
                let mut max_p = f32::NEG_INFINITY;
                for (i, k) in keys.iter().enumerate() {
                    let p = probs.get(i).copied().unwrap_or(0.0);
                    prob_map.insert(k.clone(), p);
                    if p > max_p {
                        max_p = p;
                        best_idx = i;
                    }
                }
                answers.insert(
                    task.id,
                    AnswerValue::Choice {
                        q_type: "choice".into(),
                        choice: keys.get(best_idx).cloned().unwrap_or_default(),
                        confidence: compute_confidence(&probs),
                        probabilities: prob_map,
                    },
                );
            }
            TaskKind::Score { levels } => {
                let mut prob_map = BTreeMap::new();
                let mut exp_score = 0.0;
                for (i, lvl) in levels.iter().enumerate() {
                    let p = probs.get(i).copied().unwrap_or(0.0);
                    prob_map.insert(lvl.clone(), p);
                    exp_score += p * ((i + 1) as f32);
                }
                answers.insert(
                    task.id,
                    AnswerValue::Score {
                        q_type: "score".into(),
                        score: (exp_score * 100.0).round() / 100.0,
                        confidence: compute_confidence(&probs),
                        probabilities: prob_map,
                    },
                );
            }
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64() * 1000.0;

    // The answers are round-tripped through their JSON text so that the
    // dashboard sees exactly the same numbers as the API response; a direct
    // `to_value` widened every f32 to f64 and exposed the binary residue
    // (0.7705 vs 0.7705000042915344).
    let answers_json = serde_json::to_string(&answers)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .unwrap_or_default();
    let tokens_total = unique_input_tokens;

    let (total_requests, total_tokens, avg_latency_ms) = {
        let mut stats = app
            .stats
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.total += 1;
        stats.total_tokens += tokens_total;
        stats.total_latency_ms += elapsed;
        (
            stats.total,
            stats.total_tokens,
            stats.total_latency_ms / stats.total as f64,
        )
    };

    let state_preview: String = {
        let mut s: String = state_text.chars().take(140).collect();
        if state_text.chars().count() > 140 {
            s.push('…');
        }
        s
    };

    let entry = LogEntry {
        id: req_id.clone(),
        ts: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        latency_ms: (elapsed * 100.0).round() / 100.0,
        questions: num_questions as u32,
        pairs: batch_size,
        tokens: tokens_total,
        batch_tokens,
        state_preview,
        answers: answers_json,
        total_requests,
        total_tokens,
        avg_latency_ms: (avg_latency_ms * 100.0).round() / 100.0,
    };

    // With telemetry disabled nothing that contains the request state is
    // retained or broadcast - only the anonymous counters are kept.
    if app.telemetry {
        {
            let mut hist = app
                .history
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            hist.insert(0, entry.clone());
            hist.truncate(50);
        }
        let _ = app.tx.send(entry);
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-typesafe-request-id",
        HeaderValue::from_str(&req_id).unwrap(),
    );
    headers.insert(
        "x-response-time-ms",
        HeaderValue::from_str(&format!("{:.2}", elapsed)).unwrap(),
    );

    // The response always reports the engine's own identity: echoing the
    // requested `model` would let a client pass off the answer as another
    // model's.
    let resp = SystemOneResponse {
        id: req_id,
        model: ENGINE_MODEL_ID.to_string(),
        answers,
        usage: serde_json::json!({
            "input_tokens": unique_input_tokens,
            "output_tokens": 0
        }),
    };

    Ok((headers, Json(resp)))
}

/// Print the usage banner. The engine is configured through environment
/// variables only.
fn print_help() {
    println!(
        "jev-rs {} — local System One decision engine",
        env!("CARGO_PKG_VERSION")
    );
    println!();
    println!("Usage: jev-rs [--help] [--version]");
    println!();
    println!("Options:");
    println!("  -h, --help       print this help");
    println!("  -V, --version    print the version");
    println!();
    println!("Configuration (environment variables):");
    println!("  HOST, PORT                       bind address and port");
    println!("  JEV_MODEL_PATH, JEV_TOKENIZER_PATH model and tokenizer files");
    println!("  JEV_TEMPERATURE                  calibration temperature (0 < t <= 10)");
    println!("  JEV_MAX_BODY_BYTES               request body limit");
    println!("  JEV_MAX_CONCURRENCY              number of inference slots");
    println!("  JEV_TIMEOUT                      per-request inference timeout (seconds)");
    println!("  JEV_MAX_STATE_CHARS              maximum length of the 'state' field");
    println!("  JEV_MAX_SEQ_TOKENS               maximum tokens per encoded pair");
    println!("  JEV_MAX_BATCH_TOKENS             maximum padded tokens per forward pass");
    println!("  JEV_API_KEY                      require this key on the inference endpoint");
    println!("  JEV_TELEMETRY=0                  do not retain or broadcast request states");
}

/// Error type that prints its message verbatim - the default `Result`
/// reporting uses `Debug`, which would escape the multi-line instructions.
struct PlainError(String);

impl std::fmt::Display for PlainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Debug for PlainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PlainError {}

/// Resolve the tokenizer path.
///
/// A custom model path implies a tokenizer next to it, so that a model and
/// its tokenizer can never come from different exports by accident:
///   1. explicit env override (`JEV_TOKENIZER_PATH`)
///   2. `tokenizer.json` next to the model
///   3. `model_onnx/tokenizer.json` relative to the working directory
///   4. the same path relative to the executable's directory
fn resolve_tokenizer_path(model_path: &std::path::Path) -> std::path::PathBuf {
    if let Ok(value) = std::env::var("JEV_TOKENIZER_PATH") {
        if !value.trim().is_empty() {
            return std::path::PathBuf::from(value);
        }
    }

    if let Some(dir) = model_path.parent() {
        if !dir.as_os_str().is_empty() {
            let candidate = dir.join("tokenizer.json");
            if candidate.exists() {
                return candidate;
            }
        }
    }

    resolve_data_path("JEV_TOKENIZER_PATH", "model_onnx/tokenizer.json")
}

/// Resolve a data file path with a portable fallback chain:
///   1. explicit env override (`JEV_MODEL_PATH` / `JEV_TOKENIZER_PATH`)
///   2. the path relative to the current working directory
///   3. the same path relative to the executable's directory
fn resolve_data_path(env_var: &str, relative: &str) -> std::path::PathBuf {
    if let Ok(value) = std::env::var(env_var) {
        if !value.trim().is_empty() {
            return std::path::PathBuf::from(value);
        }
    }

    let cwd_path = std::path::PathBuf::from(relative);
    if cwd_path.exists() {
        return cwd_path;
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let exe_relative = dir.join(relative);
            if exe_relative.exists() {
                return exe_relative;
            }
        }
    }

    cwd_path
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The flags are a convenience; the server is configured through the
    // environment.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("jev-rs {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8000".into());
    let model_path = resolve_data_path("JEV_MODEL_PATH", "model_onnx/model.onnx");
    let tokenizer_path = resolve_tokenizer_path(&model_path);
    let temperature = {
        let t = std::env::var("JEV_TEMPERATURE")
            .ok()
            .and_then(|t| t.parse::<f32>().ok())
            .unwrap_or(1.65);
        if !t.is_finite() || t <= 0.0 || t > 10.0 {
            return Err(format!(
                "Invalid JEV_TEMPERATURE = {} (must be finite and within 0 < t <= 10)",
                t
            )
            .into());
        }
        t
    };

    let max_body = std::env::var("JEV_MAX_BODY_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(128 * 1024);
    let max_concurrency = std::env::var("JEV_MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(2);
    let inference_timeout = std::env::var("JEV_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(30);
    let max_state_chars = std::env::var("JEV_MAX_STATE_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_STATE_CHARS);
    let max_seq_tokens = std::env::var("JEV_MAX_SEQ_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_SEQ_TOKENS);
    let max_batch_tokens = std::env::var("JEV_MAX_BATCH_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_BATCH_TOKENS);
    let api_key = std::env::var("JEV_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());
    let telemetry = std::env::var("JEV_TELEMETRY")
        .map(|v| {
            !matches!(
                v.trim().to_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true);

    println!("============================================================");
    println!(" ⚡ Jev-RS: High-Performance System One Decision Engine");
    println!("============================================================");
    println!("[*] Tokenizer path:  {}", tokenizer_path.display());
    println!("[*] Model ONNX path: {}", model_path.display());
    println!("[*] Calibration T:   {}", temperature);
    println!(
        "[*] Limits: body <= {} B, concurrency = {}, timeout = {} s",
        max_body, max_concurrency, inference_timeout
    );
    println!(
        "[*] Input limits: state <= {} chars, seq <= {} tokens, batch <= {} tokens",
        max_state_chars, max_seq_tokens, max_batch_tokens
    );
    println!(
        "[*] Telemetry: {}",
        if telemetry {
            "enabled"
        } else {
            "disabled (no request states retained)"
        }
    );
    println!(
        "[*] API key: {}",
        if api_key.is_some() {
            "required"
        } else {
            "not set (open endpoint)"
        }
    );

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        PlainError(format!(
            "Failed to load tokenizer from '{}': {}\n\n\
             Fix it with one of:\n\
             \x20 * place the 'model_onnx' folder next to the executable (or in the working directory)\n\
             \x20 * set JEV_TOKENIZER_PATH / JEV_MODEL_PATH to the full paths\n\
             \x20 * export the model first: python scripts/setup_model.py",
            tokenizer_path.display(),
            e
        ))
    })?;
    tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!(
        "[*] Initializing ONNX session (Threads: {}, DirectML GPU enabled)...",
        threads
    );

    let environment = ort::init().build()?;

    let model_load_error = |e: ort::Error| -> Box<dyn std::error::Error> {
        PlainError(format!(
            "Failed to load the ONNX model from '{}': {}\n\n\
             Fix it with one of:\n\
             \x20 * place the 'model_onnx' folder next to the executable (or in the working directory)\n\
             \x20 * set JEV_MODEL_PATH / JEV_TOKENIZER_PATH to the full paths\n\
             \x20 * export the model first: python scripts/setup_model.py",
            model_path.display(),
            e
        ))
        .into()
    };

    #[cfg(feature = "dml")]
    let session = {
        println!("[*] Execution provider: DirectML (GPU)");
        Session::builder(&environment)?
            .with_execution_providers([DirectML::default().build()])?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(&model_path)
            .map_err(model_load_error)?
    };
    #[cfg(not(feature = "dml"))]
    let session = {
        println!("[*] Execution provider: CPU");
        Session::builder(&environment)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(&model_path)
            .map_err(model_load_error)?
    };

    // Resolve the logits output name from the model signature: prefer
    // "logits", otherwise accept the sole output.
    let logits_name = {
        let names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if names.iter().any(|n| n == "logits") {
            "logits".to_string()
        } else if names.len() == 1 {
            names[0].clone()
        } else {
            return Err(format!(
                "Model output 'logits' not found (outputs: {:?}); model '{}' is not a valid NLI classification export. Re-run 'python scripts/setup_model.py'.",
                names,
                model_path.display()
            )
            .into());
        }
    };
    println!("[*] Logits output:    {}", logits_name);

    // Resolve the entailment class index and the position limit from the
    // model's config.json.
    let model_meta = load_model_meta(&model_path.to_string_lossy());
    let entailment_idx = model_meta.entailment_idx;

    let max_seq_tokens = max_seq_tokens.min(model_meta.max_positions);
    if max_seq_tokens != DEFAULT_MAX_SEQ_TOKENS {
        println!(
            "[*] Sequence budget capped to {} tokens (model position limit)",
            max_seq_tokens
        );
    }

    // A tokenizer that does not belong to the model emits token ids outside
    // the embedding matrix, which corrupts the logits instead of failing.
    let tokenizer_vocab = tokenizer.get_vocab_size(true);
    if let Some(model_vocab) = model_meta.vocab_size {
        if tokenizer_vocab < model_vocab {
            return Err(PlainError(format!(
                "Tokenizer vocabulary ({}) is smaller than the model's embedding matrix ({}).\n\n\
                 The tokenizer at '{}' does not belong to the model at '{}'.\n\
                 \x20 * point JEV_TOKENIZER_PATH at the tokenizer next to the model\n\
                 \x20 * or set both JEV_MODEL_PATH and JEV_TOKENIZER_PATH to the same export",
                tokenizer_vocab,
                model_vocab,
                tokenizer_path.display(),
                model_path.display()
            ))
            .into());
        }
        if tokenizer_vocab != model_vocab {
            println!(
                "[!] Tokenizer vocabulary ({}) differs from the model's ({}); continuing.",
                tokenizer_vocab, model_vocab
            );
        }
    }

    // Warm-up inference: validate the output shape and the entailment
    // indexing before serving traffic, so a bad export fails fast.
    let mut session = session;
    {
        let shape = [1usize, 8usize];
        let warmup_ids = Tensor::from_array((shape, vec![0i64; 8]))?;
        let warmup_mask = Tensor::from_array((shape, vec![0i64; 8]))?;
        let outputs = session.run(ort::inputs![
            "input_ids" => warmup_ids,
            "attention_mask" => warmup_mask,
        ])?;
        let (out_shape, _) = outputs[logits_name.as_str()].try_extract_tensor::<f32>()?;
        if out_shape.len() != 2 {
            return Err(format!(
                "Model output must be 2D [batch, num_classes], got rank {}",
                out_shape.len()
            )
            .into());
        }
        if (out_shape[1] as usize) <= entailment_idx {
            return Err(format!(
                "Model has {} classes, but entailment index {} is required; wrong NLI export. Re-run 'python scripts/setup_model.py'.",
                out_shape[1], entailment_idx
            )
            .into());
        }
        println!(
            "[+] Warm-up OK: logits shape [{}, {}]",
            out_shape[0], out_shape[1]
        );
    }

    println!("[+] Engine loaded successfully. Ready for inference.");

    let (tx, _rx) = broadcast::channel::<LogEntry>(64);
    let state = Arc::new(AppState {
        tokenizer,
        session: Arc::new(Mutex::new(session)),
        temperature,
        logits_name,
        entailment_idx,
        max_state_chars,
        max_seq_tokens,
        max_batch_tokens,
        inference_timeout,
        inference_permits: Arc::new(Semaphore::new(max_concurrency)),
        api_key,
        telemetry,
        tx,
        history: RwLock::new(Vec::new()),
        stats: RwLock::new(Stats::default()),
        start_ts: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
    });

    let app = Router::new()
        .route("/", get(dashboard::dashboard))
        .route("/dashboard", get(dashboard::dashboard))
        .route("/events", get(dashboard::sse))
        .route("/history", get(dashboard::history))
        .route("/health", get(|| async { "ready" }))
        .route("/ready", get(|| async { "ready" }))
        .route(
            "/v1/systemone",
            post(handle_systemone).route_layer(axum::middleware::from_fn_with_state(
                Arc::clone(&state),
                require_api_key,
            )),
        )
        .with_state(state)
        .layer(DefaultBodyLimit::max(max_body));

    let bind_addr = format!("{}:{}", host, port);
    let loopback = host == "localhost" || host == "::1" || host.starts_with("127.");
    let remote_allowed = std::env::var("JEV_ALLOW_REMOTE").is_ok();
    if !loopback && !remote_allowed {
        println!(
            "[!] WARNING: binding to non-loopback host '{}'; the API has no auth / rate-limiting. Set JEV_ALLOW_REMOTE=1 to acknowledge.",
            host
        );
    }
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    println!("[+] Server listening on http://{}", bind_addr);
    println!("============================================================");

    // Graceful shutdown: Ctrl-C stops accepting new requests and lets the
    // in-flight handlers finish.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            println!("[i] Ctrl-C received — shutting down gracefully...");
            let _ = shutdown_tx.send(());
        }
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await?;
    println!("[i] Shutdown complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn question(
        q_type: &str,
        instructions: Option<&str>,
        criteria: Option<serde_json::Value>,
    ) -> QuestionSchema {
        QuestionSchema {
            q_type: q_type.into(),
            instructions: instructions.map(Into::into),
            criteria: criteria.map(|v| serde_json::from_value(v).expect("valid criteria")),
        }
    }

    fn questions(entries: &[(&str, QuestionSchema)]) -> BTreeMap<String, QuestionSchema> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    // --- build_tasks: valid shapes ---------------------------------------
    #[test]
    fn noul_produces_two_pairs() {
        let qs = questions(&[("a", question("noul", Some("Urgent?"), None))]);
        let (pairs, tasks) = build_tasks("S", &qs).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(tasks.len(), 1);
        assert!(pairs[0].1.starts_with("Urgent?: Yes, true."));
    }

    #[test]
    fn choice_object_uses_key_names() {
        let qs = questions(&[(
            "cat",
            question(
                "choice",
                Some("Category"),
                Some(serde_json::json!({"billing": "Payment issues", "general": "Other"})),
            ),
        )]);
        let (pairs, tasks) = build_tasks("S", &qs).unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(
            matches!(&tasks[0].kind, TaskKind::Choice { keys } if keys == &vec!["billing".to_string(), "general".to_string()])
        );
    }

    #[test]
    fn choice_array_uses_descriptions() {
        let qs = questions(&[(
            "cat",
            question("choice", None, Some(serde_json::json!(["A", "B"]))),
        )]);
        let (_, tasks) = build_tasks("S", &qs).unwrap();
        assert!(
            matches!(&tasks[0].kind, TaskKind::Choice { keys } if keys == &vec!["A".to_string(), "B".to_string()])
        );
    }

    #[test]
    fn score_without_criteria_is_rejected() {
        // The wire protocol requires the criteria of a `score` question;
        // there is no silent three-level default.
        let qs = questions(&[("f", question("score", Some("Frustration"), None))]);
        let err = build_tasks("S", &qs).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("question 'f'"));
        assert!(err.message.contains("criteria"));
    }

    // --- build_tasks: rejection of malformed payloads ---------------------
    #[test]
    fn choice_without_criteria_is_rejected() {
        let qs = questions(&[("a", question("choice", Some("pick"), None))]);
        let err = build_tasks("S", &qs).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("question 'a'"));
    }

    #[test]
    fn choice_with_empty_criteria_is_rejected() {
        let qs = questions(&[("a", question("choice", None, Some(serde_json::json!({}))))]);
        assert_eq!(
            build_tasks("S", &qs).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn choice_with_non_string_values_is_rejected() {
        let qs = questions(&[(
            "a",
            question("choice", None, Some(serde_json::json!({"x": 5}))),
        )]);
        assert!(build_tasks("S", &qs)
            .unwrap_err()
            .message
            .contains("criteria values must be strings"));
    }

    #[test]
    fn score_with_non_string_items_is_rejected() {
        let qs = questions(&[(
            "a",
            question("score", None, Some(serde_json::json!([1, 2, 3]))),
        )]);
        assert!(build_tasks("S", &qs)
            .unwrap_err()
            .message
            .contains("array items must be strings"));
    }

    // --- duplicate criteria -------------------------------------------------
    /// A duplicated criterion collapses in the probability map while the
    /// winner is still chosen over the full list, so the returned
    /// distribution would not sum to 1.
    #[test]
    fn duplicate_choice_criteria_are_rejected() {
        let qs = questions(&[(
            "a",
            question("choice", None, Some(serde_json::json!(["x", "x"]))),
        )]);
        let err = build_tasks("S", &qs).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("duplicate criterion"),
            "got {}",
            err.message
        );
        assert!(err.message.contains("question 'a'"));
    }

    #[test]
    fn duplicate_score_levels_are_rejected() {
        let qs = questions(&[(
            "a",
            question(
                "score",
                None,
                Some(serde_json::json!(["Low", "Low", "High"])),
            ),
        )]);
        let err = build_tasks("S", &qs).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("duplicate criterion"),
            "got {}",
            err.message
        );
    }

    /// Duplicate keys inside a criteria object are invisible to
    /// `serde_json::Value` (the parser keeps the last one), so the criteria
    /// are deserialized by hand to catch them.
    #[test]
    fn duplicate_criteria_object_keys_are_rejected() {
        let raw = r#"{"a":{"type":"choice","criteria":{"k":"x","k":"y"}}}"#;
        let parsed: BTreeMap<String, QuestionSchema> = serde_json::from_str(raw).unwrap();
        let err = build_tasks("S", &parsed).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("duplicate criterion"),
            "got {}",
            err.message
        );
    }

    #[test]
    fn distinct_criteria_keep_a_consistent_distribution() {
        let qs = questions(&[(
            "a",
            question("choice", None, Some(serde_json::json!(["x", "y", "z"]))),
        )]);
        let (pairs, tasks) = build_tasks("S", &qs).unwrap();
        assert_eq!(pairs.len(), 3);
        assert!(
            matches!(&tasks[0].kind, TaskKind::Choice { keys } if keys == &vec!["x".to_string(), "y".to_string(), "z".to_string()])
        );
    }

    /// The order of the criteria decides which option wins a tie, so the
    /// request order must survive the flattening.
    ///
    /// The raw JSON is used on purpose: `serde_json::json!` builds a
    /// `Value::Object`, which sorts its keys before the criteria are ever
    /// seen.
    #[test]
    fn criteria_order_is_preserved() {
        let raw = r#"{"a":{"type":"choice","criteria":{"zebra":"Z","alpha":"A","middle":"M"}}}"#;
        let parsed: BTreeMap<String, QuestionSchema> = serde_json::from_str(raw).unwrap();
        let (_, tasks) = build_tasks("S", &parsed).unwrap();
        assert!(matches!(
            &tasks[0].kind,
            TaskKind::Choice { keys } if keys == &vec![
                "zebra".to_string(),
                "alpha".to_string(),
                "middle".to_string()
            ]
        ));
    }

    #[test]
    fn score_with_object_criteria_is_rejected() {
        let qs = questions(&[(
            "a",
            question("score", None, Some(serde_json::json!({"x": "Low"}))),
        )]);
        let err = build_tasks("S", &qs).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err
            .message
            .contains("requires a 'criteria' array of strings"));
    }

    #[test]
    fn choice_with_numeric_criteria_is_rejected() {
        let qs = questions(&[("a", question("choice", None, Some(serde_json::json!(5))))]);
        let err = build_tasks("S", &qs).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err
            .message
            .contains("requires a 'criteria' object or array of strings"));
    }

    // --- error codes --------------------------------------------------------
    #[test]
    fn rejection_status_is_400_for_malformed_payloads() {
        // A payload that does not deserialize (missing `state`) must not leak
        // axum's 422 to the client.
        let err = serde_json::from_str::<SystemOneRequest>(r#"{"questions":{}}"#).unwrap_err();
        assert!(err.to_string().contains("missing field"));
    }

    #[test]
    fn unknown_type_is_rejected() {
        let qs = questions(&[("a", question("weird", None, None))]);
        assert!(build_tasks("S", &qs)
            .unwrap_err()
            .message
            .contains("Unsupported question type"));
    }

    #[test]
    fn empty_questions_map_is_rejected() {
        let qs = questions(&[]);
        assert!(build_tasks("S", &qs)
            .unwrap_err()
            .message
            .contains("at least one valid question"));
    }

    // --- calibration math --------------------------------------------------
    #[test]
    fn softmax_two_class_known_values() {
        let probs = softmax(&[1.0, 0.0], 1.0);
        assert_eq!(probs, vec![0.7311, 0.2689]);
    }

    #[test]
    fn softmax_flattens_with_higher_temperature() {
        let flat = softmax(&[1.0, 0.0], 2.0);
        assert_eq!(flat, vec![0.6225, 0.3775]);
        assert!(flat[0] < 0.7311);
    }

    #[test]
    fn softmax_preserves_distribution_shape() {
        let probs = softmax(&[3.0, 1.0, 0.0], 1.0);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn confidence_is_certain_for_single_class() {
        assert_eq!(compute_confidence(&[1.0]), 1.0);
    }

    #[test]
    fn confidence_is_zero_for_uniform_distribution() {
        assert_eq!(compute_confidence(&[0.5, 0.5]), 0.0);
        assert_eq!(compute_confidence(&[0.25, 0.25, 0.25, 0.25]), 0.0);
    }

    #[test]
    fn confidence_is_high_for_skewed_distribution() {
        let conf = compute_confidence(&[0.99, 0.01]);
        assert!(conf > 0.9 && conf < 0.93, "got {}", conf);
    }

    // --- entailment index resolution --------------------------------------
    fn temp_dir_with_config(json: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "jev-rs-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(json) = json {
            std::fs::write(dir.join("config.json"), json).unwrap();
        }
        dir
    }

    fn entailment_idx_of(config: Option<&str>) -> usize {
        let dir = temp_dir_with_config(config);
        let model_path = dir.join("model.onnx").to_string_lossy().to_string();
        let meta = load_model_meta(&model_path);
        let _ = std::fs::remove_dir_all(&dir);
        meta.entailment_idx
    }

    #[test]
    fn entailment_index_resolved_from_id2label() {
        assert_eq!(
            entailment_idx_of(Some(
                r#"{"id2label": {"0": "contradiction", "1": "entailment"}}"#
            )),
            1
        );
    }

    #[test]
    fn entailment_index_resolved_from_true_false() {
        assert_eq!(
            entailment_idx_of(Some(r#"{"id2label": {"0": "false", "1": "true"}}"#)),
            1
        );
    }

    #[test]
    fn entailment_index_falls_back_without_config() {
        assert_eq!(entailment_idx_of(None), ENTAILMENT_IDX);
    }

    /// "not_entailment" contains "entailment" as a substring: matching it by
    /// substring inverts every probability the engine reports.
    #[test]
    fn entailment_index_ignores_not_entailment_label() {
        assert_eq!(
            entailment_idx_of(Some(
                r#"{"id2label": {"0": "not_entailment", "1": "entailment"}}"#
            )),
            1
        );
    }

    /// The classic 3-class NLI ordering must resolve to the entailment class,
    /// not to contradiction or neutral.
    #[test]
    fn entailment_index_resolved_for_three_class_nli() {
        assert_eq!(
            entailment_idx_of(Some(
                r#"{"id2label": {"0": "contradiction", "1": "neutral", "2": "entailment"}}"#
            )),
            2
        );
    }

    #[test]
    fn entailment_index_ignores_contradiction_and_neutral() {
        assert_eq!(
            entailment_idx_of(Some(
                r#"{"id2label": {"0": "contradiction", "1": "neutral"}}"#
            )),
            ENTAILMENT_IDX
        );
    }

    /// A UTF-8 BOM in config.json must not silently downgrade the label
    /// resolution to the fallback index.
    #[test]
    fn entailment_index_survives_utf8_bom() {
        let dir = temp_dir_with_config(None);
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(br#"{"id2label": {"0": "not_entailment", "1": "entailment"}}"#);
        std::fs::write(dir.join("config.json"), bytes).unwrap();
        let model_path = dir.join("model.onnx").to_string_lossy().to_string();
        let meta = load_model_meta(&model_path);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(meta.entailment_idx, 1);
    }

    #[test]
    fn model_meta_reads_position_limit() {
        let dir = temp_dir_with_config(Some(
            r#"{"id2label": {"0": "entailment"}, "max_position_embeddings": 4096}"#,
        ));
        let model_path = dir.join("model.onnx").to_string_lossy().to_string();
        let meta = load_model_meta(&model_path);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(meta.max_positions, 4096);
        assert_eq!(meta.entailment_idx, 0);
    }

    // --- input limits -------------------------------------------------------
    #[test]
    fn state_within_limit_is_accepted() {
        assert!(validate_state_length("a short ticket", 2000).is_ok());
    }

    #[test]
    fn empty_state_is_rejected() {
        let err = validate_state_length("   ", 2000).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("must not be empty"));
    }

    #[test]
    fn state_over_limit_is_rejected_with_400() {
        let err = validate_state_length(&"A".repeat(2001), 2000).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("too long"));
    }

    #[test]
    fn state_limit_counts_characters_not_bytes() {
        // 100 multi-byte characters must count as 100, not 400.
        let state = "ä".repeat(100);
        assert!(validate_state_length(&state, 100).is_ok());
    }

    #[test]
    fn batch_within_limits_is_accepted() {
        assert!(validate_batch_limits(8, 512, 2048, 8192).is_ok());
    }

    #[test]
    fn batch_over_sequence_limit_is_rejected() {
        let err = validate_batch_limits(1, 4096, 2048, 8192).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("sequence is too long"));
    }

    #[test]
    fn batch_over_token_budget_is_rejected() {
        let err = validate_batch_limits(8, 2048, 2048, 8192).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("batch is too large"));
    }

    // --- error sanitization -------------------------------------------------
    #[test]
    fn sanitize_error_strips_windows_paths() {
        let raw = "Non-zero status code returned while running Expand node. \
                   D:\\a\\ort-artifacts\\onnxruntime\\core\\providers\\dml\\src\\MLOperatorAuthorImpl.cpp(2901)";
        let clean = sanitize_error_detail(raw);
        assert!(!clean.contains("ort-artifacts"), "leaked path: {}", clean);
        assert!(!clean.contains('D'), "leaked drive letter: {}", clean);
        assert!(clean.contains("<path>"));
    }

    #[test]
    fn sanitize_error_strips_unc_paths() {
        let clean = sanitize_error_detail(r"failed reading \\server\share\model.onnx now");
        assert!(!clean.contains("server"), "leaked UNC path: {}", clean);
    }

    #[test]
    fn sanitize_error_keeps_useful_text() {
        let clean =
            sanitize_error_detail("Model output must be 2D [batch, num_classes], got rank 3");
        assert!(clean.contains("Model output must be 2D"));
        assert!(!clean.contains("<path>"));
    }

    #[test]
    fn sanitize_error_truncates_long_messages() {
        let long = "x".repeat(1000);
        let clean = sanitize_error_detail(&long);
        assert!(clean.chars().count() <= 241);
    }

    // --- API key -----------------------------------------------------------
    #[test]
    fn api_key_comparison_is_exact() {
        assert!(secrets_match("s3cret", "s3cret"));
        assert!(!secrets_match("s3cret", "s3cre"));
        assert!(!secrets_match("s3cret", "S3CRET"));
        assert!(!secrets_match("", "s3cret"));
    }
}
