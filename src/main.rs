mod dashboard;

use axum::{
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use dashboard::{LogEntry, Stats};
use ort::ep::DirectML;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, Mutex};
use uuid::Uuid;

// Entailment class index for BGE-M3 / NLI architectures
const ENTAILMENT_IDX: usize = 0;

// -----------------------------------------------------------------------------
// TypeSafe Wire Protocol Data Structures
// -----------------------------------------------------------------------------
#[derive(Debug, Deserialize)]
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
    questions: HashMap<String, QuestionSchema>,
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
        probabilities: HashMap<String, f32>,
        confidence: f32,
    },
    Score {
        #[serde(rename = "type")]
        q_type: String,
        score: f32,
        probabilities: HashMap<String, f32>,
        confidence: f32,
    },
}

#[derive(Debug, Serialize)]
struct SystemOneResponse {
    id: String,
    model: String,
    answers: HashMap<String, AnswerValue>,
    usage: serde_json::Value,
}

struct AppState {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    temperature: f32,
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

enum TaskKind {
    Noul,
    Choice { keys: Vec<String> },
    Score { levels: Vec<String> },
}

struct TaskMeta {
    id: String,
    start: usize,
    end: usize,
    kind: TaskKind,
}

// -----------------------------------------------------------------------------
// Request Handler (Single Forward Pass)
// -----------------------------------------------------------------------------
async fn handle_systemone(
    State(app): State<Arc<AppState>>,
    Json(payload): Json<SystemOneRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let start_time = Instant::now();
    let req_id = format!("so_{}", &Uuid::new_v4().to_string().replace("-", "")[..12]);

    let state_text = match &payload.state {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    let mut premise_hypo_pairs: Vec<(String, String)> = Vec::new();
    let mut tasks: Vec<TaskMeta> = Vec::new();

    // 1. Flatten all questions into a single batch of hypothesis pairs
    for (q_id, q) in &payload.questions {
        let instr = q.instructions.as_deref().unwrap_or("");
        let start_idx = premise_hypo_pairs.len();

        match q.q_type.to_lowercase().as_str() {
            "noul" => {
                premise_hypo_pairs.push((state_text.clone(), format!("{}: Yes, true.", instr)));
                premise_hypo_pairs.push((state_text.clone(), format!("{}: No, false.", instr)));
                tasks.push(TaskMeta {
                    id: q_id.clone(),
                    start: start_idx,
                    end: premise_hypo_pairs.len(),
                    kind: TaskKind::Noul,
                });
            }
            "choice" => {
                let mut keys = Vec::new();
                if let Some(serde_json::Value::Object(map)) = &q.criteria {
                    for (k, v) in map {
                        keys.push(k.clone());
                        let desc = v.as_str().unwrap_or("");
                        premise_hypo_pairs.push((state_text.clone(), format!("{}: {}", instr, desc)));
                    }
                } else if let Some(serde_json::Value::Array(arr)) = &q.criteria {
                    for v in arr {
                        let desc = v.as_str().unwrap_or("");
                        keys.push(desc.to_string());
                        premise_hypo_pairs.push((state_text.clone(), format!("{}: {}", instr, desc)));
                    }
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
                if let Some(serde_json::Value::Array(arr)) = &q.criteria {
                    for v in arr {
                        levels.push(v.as_str().unwrap_or("").to_string());
                    }
                } else {
                    levels = vec!["Low".into(), "Medium".into(), "High".into()];
                }
                for lvl in &levels {
                    premise_hypo_pairs.push((state_text.clone(), format!("{}: {}", instr, lvl)));
                }
                tasks.push(TaskMeta {
                    id: q_id.clone(),
                    start: start_idx,
                    end: premise_hypo_pairs.len(),
                    kind: TaskKind::Score { levels },
                });
            }
            _ => return Err((StatusCode::BAD_REQUEST, format!("Unsupported question type: {}", q.q_type))),
        }
    }

    // 2. Vectorized batch tokenization
    let token_inputs: Vec<(&str, &str)> = premise_hypo_pairs.iter().map(|(p, h)| (p.as_str(), h.as_str())).collect();
    let encodings = app
        .tokenizer
        .encode_batch(token_inputs, true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

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

    // 3. Tensor creation
    let shape = [batch_size, seq_len];
    let input_ids_tensor = Tensor::from_array((shape, flat_input_ids))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let attention_mask_tensor = Tensor::from_array((shape, flat_attention_mask))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let inputs = ort::inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
    ];

    // 4. Single forward pass (GPU / DirectML)
    let mut session = app.session.lock().await;
    let outputs = session
        .run(inputs)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let (out_shape, logits_data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let num_rows = out_shape[0] as usize;
    let num_classes = out_shape[1] as usize;

    let mut entail_logits: Vec<f32> = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        entail_logits.push(logits_data[i * num_classes + ENTAILMENT_IDX]);
    }

    // 5. Probability calibration & answer compilation
    let num_questions = tasks.len();
    let mut answers = HashMap::new();

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
                let mut prob_map = HashMap::new();
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
                let mut prob_map = HashMap::new();
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
    let tokens_total = (batch_size * seq_len) as u64;

    let (total_requests, total_tokens, avg_latency_ms) = {
        let mut stats = app.stats.write().unwrap();
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
        let mut hist = app.history.write().unwrap();
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
            "input_tokens": seq_len,
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
    let temperature = std::env::var("JEV_TEMPERATURE")
        .ok()
        .and_then(|t| t.parse::<f32>().ok())
        .unwrap_or(1.65);

    println!("============================================================");
    println!(" ⚡ Jev-RS: High-Performance System One Decision Engine");
    println!("============================================================");
    println!("[*] Tokenizer path:  {}", tokenizer_path);
    println!("[*] Model ONNX path: {}", model_path);
    println!("[*] Calibration T:   {}", temperature);

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|_| panic!("Failed to load tokenizer from '{}'. Run 'python scripts/setup_model.py' first!", tokenizer_path));
    tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    println!("[*] Initializing ONNX session (Threads: {}, DirectML GPU enabled)...", threads);

    let environment = ort::init().build()?;

    let session = Session::builder(&environment)?
        .with_execution_providers([DirectML::default().build()])?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(threads)?
        .commit_from_file(&model_path)
        .unwrap_or_else(|_| panic!("Failed to load model from '{}'. Run 'python scripts/setup_model.py' first!", model_path));

    println!("[+] Engine loaded successfully. Ready for inference.");

    let (tx, _rx) = broadcast::channel::<LogEntry>(64);
    let state = Arc::new(AppState {
        tokenizer,
        session: Mutex::new(session),
        temperature,
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
        .with_state(state);

    let bind_addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    println!("[+] Server listening on http://{}", bind_addr);
    println!("============================================================");

    axum::serve(listener, app).await?;
    Ok(())
}
