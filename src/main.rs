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
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, Mutex, Semaphore};
use uuid::Uuid;

// Fallback entailment class index for BGE-M3-style NLI exports. Only used
// when the model's config.json cannot provide the real mapping (see
// `resolve_entailment_index`).
const ENTAILMENT_IDX: usize = 0;

// -----------------------------------------------------------------------------
// TypeSafe Wire Protocol Data Structures
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Deserialize)]
struct QuestionSchema {
    #[serde(rename = "type")]
    q_type: String, // "noul", "choice", "score"
    instructions: Option<String>,
    criteria: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SystemOneRequest {
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
    /// Resolved ONNX output name for the classification logits
    /// (validated at startup; usually "logits").
    logits_name: String,
    /// Entailment class index resolved from the model's config.json.
    entailment_idx: usize,
    /// Per-request inference timeout in seconds (JEV_TIMEOUT).
    inference_timeout: u64,
    /// Limits concurrent forward passes; excess requests get 503 (JEV_MAX_CONCURRENCY).
    inference_permits: Arc<Semaphore>,
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
    exps.iter().map(|&x| (x / sum * 10000.0).round() / 10000.0).collect()
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

    /// Map an axum extractor rejection (bad JSON body, missing content-type,
    /// body over the size limit) into the same envelope.
    fn from_rejection(rejection: JsonRejection) -> Self {
        Self {
            status: rejection.status(),
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

/// Determine the NLI entailment class index from the exported model's
/// config.json (`id2label`), instead of hardcoding it. Recognizes both
/// classical NLI labeling ("entailment") and binary zeroshot exports where
/// the entailment class is labeled "true". Falls back to `ENTAILMENT_IDX`
/// with a warning when the config is missing or unrecognized.
fn resolve_entailment_index(model_path: &str) -> usize {
    let fallback = || {
        println!(
            "[!] config.json has no recognizable NLI labels; defaulting entailment index to {}",
            ENTAILMENT_IDX
        );
        ENTAILMENT_IDX
    };

    let config_path = std::path::Path::new(model_path).with_file_name("config.json");
    if !config_path.exists() {
        println!(
            "[!] config.json not found next to '{}'; defaulting entailment index to {}",
            model_path, ENTAILMENT_IDX
        );
        return ENTAILMENT_IDX;
    }

    let config = match std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    {
        Some(v) => v,
        None => return fallback(),
    };

    if let Some(map) = config.get("id2label").and_then(|v| v.as_object()) {
        for (k, v) in map {
            let label = match v.as_str() {
                Some(l) => l.to_lowercase(),
                None => continue,
            };
            if label.contains("entailment") || label == "true" {
                if let Ok(idx) = k.parse::<usize>() {
                    println!(
                        "[*] Entailment label: '{}' at index {} (from id2label)",
                        label, idx
                    );
                    return idx;
                }
            }
        }
    }
    fallback()
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

/// Flatten the wire-protocol questions into the evaluation pair list
/// (premise, hypothesis) and per-question task metadata used to slice the
/// entailment logits. Rejects malformed questions with a JSON 400 error.
fn build_tasks(
    state_text: &str,
    questions: &BTreeMap<String, QuestionSchema>,
) -> Result<(Vec<(String, String)>, Vec<TaskMeta>), ApiError> {
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
                let mut keys = Vec::new();
                match &q.criteria {
                    Some(serde_json::Value::Object(map)) => {
                        for (k, v) in map {
                            let desc = v.as_str().ok_or_else(|| {
                                ApiError::bad_request(format!(
                                    "question '{}': 'choice' criteria values must be strings",
                                    q_id
                                ))
                            })?;
                            keys.push(k.clone());
                            premise_hypo_pairs
                                .push((state_text.to_string(), format!("{}: {}", instr, desc)));
                        }
                    }
                    Some(serde_json::Value::Array(arr)) => {
                        for v in arr {
                            let desc = v.as_str().ok_or_else(|| {
                                ApiError::bad_request(format!(
                                    "question '{}': 'choice' criteria array items must be strings",
                                    q_id
                                ))
                            })?;
                            keys.push(desc.to_string());
                            premise_hypo_pairs
                                .push((state_text.to_string(), format!("{}: {}", instr, desc)));
                        }
                    }
                    _ => {
                        return Err(ApiError::bad_request(format!(
                            "question '{}': type 'choice' requires a 'criteria' object or array of strings",
                            q_id
                        )));
                    }
                }
                if keys.is_empty() {
                    return Err(ApiError::bad_request(format!(
                        "question '{}': 'choice' criteria must not be empty",
                        q_id
                    )));
                }
                tasks.push(TaskMeta {
                    id: q_id.clone(),
                    start: start_idx,
                    end: premise_hypo_pairs.len(),
                    kind: TaskKind::Choice { keys },
                });
            }
            "score" => {
                let mut levels = Vec::new();
                match &q.criteria {
                    Some(serde_json::Value::Array(arr)) => {
                        for v in arr {
                            let lvl = v.as_str().ok_or_else(|| {
                                ApiError::bad_request(format!(
                                    "question '{}': 'score' criteria array items must be strings",
                                    q_id
                                ))
                            })?;
                            levels.push(lvl.to_string());
                        }
                        if levels.is_empty() {
                            return Err(ApiError::bad_request(format!(
                                "question '{}': 'score' criteria must not be empty",
                                q_id
                            )));
                        }
                    }
                    // No criteria: use a sensible ordinal default backwards-compatible
                    // with the previous behavior.
                    None => levels = vec!["Low".into(), "Medium".into(), "High".into()],
                    _ => {
                        return Err(ApiError::bad_request(format!(
                            "question '{}': type 'score' requires 'criteria' as an array of strings",
                            q_id
                        )));
                    }
                }
                for lvl in &levels {
                    premise_hypo_pairs.push((state_text.to_string(), format!("{}: {}", instr, lvl)));
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

    // Guard: an empty questions map (or questions that produced no evaluation
    // pairs) would otherwise panic on `encodings[0]` below.
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

    // 1. Flatten all questions into a single batch of hypothesis pairs
    let (mut premise_hypo_pairs, tasks) =
        build_tasks(&state_text, &payload.questions)?;

    // 2. Vectorized batch tokenization. CPU-bound — offloaded to the blocking
    // pool so async worker threads stay responsive for /health, SSE, etc.
    let encodings = {
        let pairs = std::mem::take(&mut premise_hypo_pairs);
        let app_ref = Arc::clone(&app);
        match tokio::task::spawn_blocking(move || {
            let token_inputs: Vec<(&str, &str)> =
                pairs.iter().map(|(p, h)| (p.as_str(), h.as_str())).collect();
            app_ref.tokenizer.encode_batch(token_inputs, true)
        })
        .await
        {
            Ok(res) => res.map_err(|e| ApiError::internal(e.to_string()))?,
            Err(e) => {
                return Err(ApiError::internal(format!(
                    "Tokenization task failed: {}",
                    e
                )))
            }
        }
    };

    let batch_size = encodings.len();
    let seq_len = encodings[0].get_ids().len();

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

    // Real (unpadded) token count across the whole batch — for usage accounting
    // and telemetry instead of the padded `batch_size * seq_len`.
    let real_tokens = flat_attention_mask.iter().filter(|&&m| m == 1).count() as u64;

    // 3. Tensor creation
    let shape = [batch_size, seq_len];
    let input_ids_tensor = Tensor::from_array((shape, flat_input_ids))
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let attention_mask_tensor = Tensor::from_array((shape, flat_attention_mask))
        .map_err(|e| ApiError::internal(e.to_string()))?;

    // 4. Single forward pass (GPU / DirectML). Offloaded to the blocking pool
    // so the async runtime stays responsive; bounded by a concurrency permit
    // (overload -> 503) and a per-request timeout (-> 504).
    // Logits extraction & shape validation happen inside the closure because
    // the run output borrows the session guard.
    let _permit = app
        .inference_permits
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::service_unavailable("Server busy: too many concurrent inferences")
        })?;

    let entail_idx = app.entailment_idx;
    let logits_name = app.logits_name.clone();
    let session_mutex = Arc::clone(&app.session);
    let mut session_guard = session_mutex.lock_owned().await;

    let run_result = tokio::time::timeout(
        Duration::from_secs(app.inference_timeout),
        tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ];
            let outputs = session_guard.run(inputs).map_err(|e| e.to_string())?;
            let (out_shape, logits_data) = outputs[logits_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;

            // Guard: the output must be a 2D classification result
            // [batch, num_classes]; return owned entailment logits.
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
            if num_rows != batch_size {
                return Err(format!(
                    "Model output batch mismatch: got {} rows for a batch of {}",
                    num_rows, batch_size
                ));
            }

            let mut entail_logits = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                entail_logits.push(logits_data[i * num_classes + entail_idx]);
            }
            Ok(entail_logits)
        }),
    )
    .await;

    let entail_logits = match run_result {
        Err(_) => {
            return Err(ApiError::gateway_timeout(format!(
                "Inference timed out after {}s",
                app.inference_timeout
            )))
        }
        Ok(Err(join_err)) => {
            return Err(ApiError::internal(format!(
                "Inference task failed: {}",
                join_err
            )))
        }
        Ok(Ok(Err(e))) => return Err(ApiError::internal(e)),
        Ok(Ok(Ok(entail))) => entail,
    };

    // 5. Probability calibration & answer compilation
    let num_questions = tasks.len();
    let mut answers = BTreeMap::new();

    for task in tasks {
        let task_logits = &entail_logits[task.start..task.end];
        let probs = softmax(task_logits, app.temperature);

        match task.kind {
            TaskKind::Noul => {
                answers.insert(
                    task.id,
                    AnswerValue::Noul {
                        q_type: "noul".into(),
                        noul: probs[0],
                    },
                );
            }
            TaskKind::Choice { keys } => {
                let mut prob_map = BTreeMap::new();
                let mut best_idx = 0;
                let mut max_p = -1.0;
                for (i, k) in keys.iter().enumerate() {
                    prob_map.insert(k.clone(), probs[i]);
                    if probs[i] > max_p {
                        max_p = probs[i];
                        best_idx = i;
                    }
                }
                answers.insert(
                    task.id,
                    AnswerValue::Choice {
                        q_type: "choice".into(),
                        choice: keys[best_idx].clone(),
                        confidence: compute_confidence(&probs),
                        probabilities: prob_map,
                    },
                );
            }
            TaskKind::Score { levels } => {
                let mut prob_map = BTreeMap::new();
                let mut exp_score = 0.0;
                for (i, lvl) in levels.iter().enumerate() {
                    prob_map.insert(lvl.clone(), probs[i]);
                    exp_score += probs[i] * ((i + 1) as f32);
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

    // 6. Record telemetry entry, push to dashboard subscribers
    let answers_json = serde_json::to_value(&answers).unwrap_or_default();
    let tokens_total = real_tokens;

    let (total_requests, total_tokens, avg_latency_ms) = {
        let mut stats = app.stats
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.total += 1;
        stats.total_tokens += tokens_total;
        stats.total_latency_ms += elapsed;
        (stats.total, stats.total_tokens, stats.total_latency_ms / stats.total as f64)
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
        state_preview,
        answers: answers_json,
        total_requests,
        total_tokens,
        avg_latency_ms: (avg_latency_ms * 100.0).round() / 100.0,
    };

    {
        let mut hist = app.history
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        hist.insert(0, entry.clone());
        hist.truncate(50);
    }
    let _ = app.tx.send(entry);

    let mut headers = HeaderMap::new();
    headers.insert("x-typesafe-request-id", HeaderValue::from_str(&req_id).unwrap());
    headers.insert("x-response-time-ms", HeaderValue::from_str(&format!("{:.2}", elapsed)).unwrap());

    let resp = SystemOneResponse {
        id: req_id,
        model: payload.model.unwrap_or_else(|| "local-jev-rust".into()),
        answers,
        usage: serde_json::json!({
            "input_tokens": real_tokens,
            "output_tokens": 0
        }),
    };

    Ok((headers, Json(resp)))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8000".into());
    let model_path = std::env::var("JEV_MODEL_PATH").unwrap_or_else(|_| "model_onnx/model.onnx".into());
    let tokenizer_path = std::env::var("JEV_TOKENIZER_PATH").unwrap_or_else(|_| "model_onnx/tokenizer.json".into());
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

    println!("============================================================");
    println!(" ⚡ Jev-RS: High-Performance System One Decision Engine");
    println!("============================================================");
    println!("[*] Tokenizer path:  {}", tokenizer_path);
    println!("[*] Model ONNX path: {}", model_path);
    println!("[*] Calibration T:   {}", temperature);
    println!("[*] Limits: body <= {} B, concurrency = {}, timeout = {} s",
        max_body, max_concurrency, inference_timeout);

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|_| panic!("Failed to load tokenizer from '{}'. Run 'python scripts/setup_model.py' first!", tokenizer_path));
    tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    println!("[*] Initializing ONNX session (Threads: {}, DirectML GPU enabled)...", threads);

    let environment = ort::init().build()?;

    #[cfg(feature = "dml")]
    let session = {
        println!("[*] Execution provider: DirectML (GPU)");
        Session::builder(&environment)?
            .with_execution_providers([DirectML::default().build()])?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(&model_path)
            .unwrap_or_else(|_| panic!("Failed to load model from '{}'. Run 'python scripts/setup_model.py' first!", model_path))
    };
    #[cfg(not(feature = "dml"))]
    let session = {
        println!("[*] Execution provider: CPU");
        Session::builder(&environment)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(&model_path)
            .unwrap_or_else(|_| panic!("Failed to load model from '{}'. Run 'python scripts/setup_model.py' first!", model_path))
    };

    // Resolve the logits output name from the model signature instead of
    // hardcoding it: prefer "logits", otherwise accept the sole output.
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
                names, model_path
            )
            .into());
        }
    };
    println!("[*] Logits output:    {}", logits_name);

    // Resolve the entailment class index from the model's config.json (H4):
    // hardcoded guesses silently break NLI ordering for other exports.
    let entailment_idx = resolve_entailment_index(&model_path);

    // Warm-up inference: validate output shape & entailment indexing before
    // serving any traffic, so a bad ONNX export fails fast with a clear error.
    let mut session = session;
    {
        let shape = [1usize, 8usize];
        let warmup_ids = Tensor::from_array((shape, vec![0i64; 8]))?;
        let warmup_mask = Tensor::from_array((shape, vec![0i64; 8]))?;
        let outputs = session.run(ort::inputs![
            "input_ids" => warmup_ids,
            "attention_mask" => warmup_mask,
        ])?;
        let (out_shape, _) = outputs[logits_name.as_str()]
            .try_extract_tensor::<f32>()?;
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
        inference_timeout,
        inference_permits: Arc::new(Semaphore::new(max_concurrency)),
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
        .route("/v1/systemone", post(handle_systemone))
        .with_state(state)
        .layer(DefaultBodyLimit::max(max_body));

    let bind_addr = format!("{}:{}", host, port);
    let loopback = host == "localhost"
        || host == "::1"
        || host.starts_with("127.");
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
    // in-flight handlers finish before returning.
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

    fn question(q_type: &str, instructions: Option<&str>, criteria: Option<serde_json::Value>) -> QuestionSchema {
        QuestionSchema {
            q_type: q_type.into(),
            instructions: instructions.map(Into::into),
            criteria,
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
        assert!(matches!(&tasks[0].kind, TaskKind::Choice { keys } if keys == &vec!["billing".to_string(), "general".to_string()]));
    }

    #[test]
    fn choice_array_uses_descriptions() {
        let qs = questions(&[(
            "cat",
            question("choice", None, Some(serde_json::json!(["A", "B"]))),
        )]);
        let (_, tasks) = build_tasks("S", &qs).unwrap();
        assert!(matches!(&tasks[0].kind, TaskKind::Choice { keys } if keys == &vec!["A".to_string(), "B".to_string()]));
    }

    #[test]
    fn score_without_criteria_uses_default_levels() {
        let qs = questions(&[("f", question("score", Some("Frustration"), None))]);
        let (pairs, tasks) = build_tasks("S", &qs).unwrap();
        assert_eq!(pairs.len(), 3);
        assert!(matches!(&tasks[0].kind, TaskKind::Score { levels } if levels.len() == 3));
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
        assert_eq!(build_tasks("S", &qs).unwrap_err().status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn choice_with_non_string_values_is_rejected() {
        let qs = questions(&[(
            "a",
            question("choice", None, Some(serde_json::json!({"x": 5}))),
        )]);
        assert!(build_tasks("S", &qs).unwrap_err().message.contains("criteria values must be strings"));
    }

    #[test]
    fn score_with_non_string_items_is_rejected() {
        let qs = questions(&[(
            "a",
            question("score", None, Some(serde_json::json!([1, 2, 3]))),
        )]);
        assert!(build_tasks("S", &qs).unwrap_err().message.contains("array items must be strings"));
    }

    #[test]
    fn unknown_type_is_rejected() {
        let qs = questions(&[("a", question("weird", None, None))]);
        assert!(build_tasks("S", &qs).unwrap_err().message.contains("Unsupported question type"));
    }

    #[test]
    fn empty_questions_map_is_rejected() {
        let qs = questions(&[]);
        assert!(build_tasks("S", &qs).unwrap_err().message.contains("at least one valid question"));
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

    #[test]
    fn entailment_index_resolved_from_id2label() {
        let dir = temp_dir_with_config(Some(
            r#"{"id2label": {"0": "contradiction", "1": "entailment"}}"#,
        ));
        let model_path = dir.join("model.onnx").to_string_lossy().to_string();
        assert_eq!(resolve_entailment_index(&model_path), 1);
    }

    #[test]
    fn entailment_index_resolved_from_true_false() {
        let dir = temp_dir_with_config(Some(r#"{"id2label": {"0": "false", "1": "true"}}"#));
        let model_path = dir.join("model.onnx").to_string_lossy().to_string();
        assert_eq!(resolve_entailment_index(&model_path), 1);
    }

    #[test]
    fn entailment_index_falls_back_without_config() {
        let dir = temp_dir_with_config(None);
        let model_path = dir.join("model.onnx").to_string_lossy().to_string();
        assert_eq!(resolve_entailment_index(&model_path), ENTAILMENT_IDX);
    }
}
