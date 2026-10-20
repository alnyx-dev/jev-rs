<div align="center">

# ⚡ Jev-RS

**A local, non-autoregressive decision engine.**
*Wire-compatible with the TypeSafe AI / Jev protocol.*

[![Rust](https://img.shields.io/badge/rust-1.80%2B-20242a?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-34d399?logo=opensourceinitiative)](https://opensource.org/licenses/MIT)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-DirectML-0078d4?logo=onnx)](https://onnxruntime.ai/)
[![Release](https://img.shields.io/github/v/release/alnyx-dev/jev-rs?label=release&color=38bdf8)](https://github.com/alnyx-dev/jev-rs/releases/latest)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-38bdf8)](https://github.com/alnyx-dev/jev-rs)

Traditional LLMs are slow, non-deterministic and generate tokens one by one.
**Jev-RS** answers every question in a **single forward pass** — no token
generation, no autoregressive loop, one batched hypothesis evaluation.

</div>

---

## 📑 Table of contents

- [Why](#-why)
- [Features](#-features)
- [Quick start](#-quick-start)
- [How it works](#-how-it-works)
- [HTTP API](#-http-api)
  - [Question types](#question-types)
  - [Errors](#errors)
- [Configuration](#-configuration)
- [Live dashboard](#-live-dashboard)
- [Performance](#-performance)
- [Project layout](#-project-layout)
- [Development](#-development)
- [Releases](#-releases)
- [License](#-license)

---

## 💡 Why

A support ticket arrives. You need to know **three** things at once: is it
urgent, which category, and how frustrated the customer is.

An LLM answers with a token stream — slow, stochastic, and it re-reads the
context for every question. A classic classifier answers one label at a time.

Jev-RS treats the whole request as **one batch of hypothesis pairs** and asks
the NLI model to entail all of them in a single pass. Every question is
evaluated simultaneously, on the same premise, with calibrated probabilities
and an explicit confidence value.

| | Traditional LLM | Classic classifier | **Jev-RS** |
| :--- | :--- | :--- | :--- |
| Questions per request | N sequential calls | 1 label | **N labels, 1 pass** |
| Determinism | sampling-dependent | deterministic | **deterministic** |
| Explainability | free text | label only | **label + full distribution + confidence** |
| Output | JSON-ish tokens | class index | **TypeSafe wire protocol** |

---

## 🚀 Features

- **True System One architecture** — zero autoregressive token generation; all
  decisions are evaluated in parallel.
- **Wire-compatible** — drop-in local replacement for the TypeSafe AI API
  (`POST /v1/systemone`).
- **Typed primitives** — `noul` (binary), `choice` (categorical) and `score`
  (ordered rubric) in one request.
- **Calibrated probabilities** — RLCD-style temperature scaling removes neural
  overconfidence.
- **Honest confidence** — normalised Shannon entropy
  `1 - H(P) / log₂(K)`, reported for every non-binary answer.
- **Hardware accelerated** — DirectML (DirectX 12) on Windows, CPU everywhere
  else, behind a cargo feature.
- **Self-contained telemetry** — a live dashboard with an SSE request feed,
  latency chart and rolling statistics.
- **Hardened** — input validation, request limits, timeouts, graceful
  shutdown, deterministic JSON, and a unit-tested core.

---

## ⚡ Quick start

### Windows

```bat
install.bat        :: one-time setup: venv, model export, Rust, release build
run.bat            :: start the engine (Ctrl+C to stop)
```

`install.bat` creates a `.venv`, installs `optimum`/`transformers`/`torch`,
exports the ONNX model into `model_onnx\` (~2.3 GB), installs Rust if needed
and builds the release binary. Skip parts with `JEV_SKIP_PIP=1`,
`JEV_SKIP_MODEL=1` or `JEV_SKIP_RUST=1`.

### Any platform

```bash
# 1. Export the model once (Python)
pip install optimum[onnxruntime] transformers torch
python scripts/setup_model.py

# 2. Build
cargo build --release                        # Windows: DirectML GPU
# cargo build --release --no-default-features # Linux / macOS: CPU

# 3. Run
./target/release/jev-rs
```

The server listens on `http://127.0.0.1:8000`.

> ⚠️ **Windows:** always build with `--release`. The ONNX Runtime binaries are
> built against the release CRT (`/MD`), so debug builds do not link.
>
> **Model placement:** the engine looks for `model_onnx/` in the working
> directory and then next to the executable; `JEV_MODEL_PATH` /
> `JEV_TOKENIZER_PATH` override both.

---

## 🧠 How it works

```
POST /v1/systemone
        │
        ├─ 1. flatten ────── every question becomes premise/hypothesis pairs
        │                    noul    → "…: Yes, true."  /  "…: No, false."
        │                    choice  → one pair per criterion
        │                    score   → one pair per level
        │
        ├─ 2. tokenize ───── one batched pass over the HuggingFace tokenizer
        │
        ├─ 3. forward ────── a single ONNX forward pass over the whole batch
        │                    (DirectML GPU or CPU) → entailment logits
        │
        ├─ 4. calibrate ──── temperature scaling → probabilities per question
        │                    entropy → confidence
        │
        └─ 5. answer ─────── { type, value, probabilities, confidence }
```

Every question shares one premise, so the model only has to judge the
hypothesis — the decision work is batched, not repeated.

---

## 📡 HTTP API

### `POST /v1/systemone`

```bash
curl -X POST "http://127.0.0.1:8000/v1/systemone" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Payment was charged twice, but the order is still pending. Fix this immediately!",
    "questions": {
      "is_urgent": {
        "type": "noul",
        "instructions": "Does this require urgent human intervention?"
      },
      "category": {
        "type": "choice",
        "instructions": "Select issue category",
        "criteria": {
          "billing": "Payment, refund, or invoice issues",
          "tech_bug": "App crash or visual error",
          "general": "General questions"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "Customer frustration level",
        "criteria": ["Calm", "Annoyed", "Extremely Furious"]
      }
    }
  }'
```

```json
{
  "id": "so_3b95af994533",
  "model": "local-jev-rust",
  "answers": {
    "is_urgent": {
      "type": "noul",
      "noul": 0.7179
    },
    "category": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.8282, "tech_bug": 0.1148, "general": 0.0570 },
      "confidence": 0.4831
    },
    "frustration": {
      "type": "score",
      "score": 2.24,
      "probabilities": { "Calm": 0.0719, "Annoyed": 0.6128, "Extremely Furious": 0.3153 },
      "confidence": 0.2233
    }
  },
  "usage": { "input_tokens": 66, "output_tokens": 0 }
}
```

Response headers carry `x-typesafe-request-id` and `x-response-time-ms`.

### Question types

| Type | Shape | Answer |
| :--- | :--- | :--- |
| `noul` | `instructions` | probability that the state is true |
| `choice` | `instructions` + `criteria` (object or array of strings) | winning option, full distribution, confidence |
| `score` | `instructions` + `criteria` (array of strings, ordered) | expected value `1..N`, full distribution, confidence |

`criteria` is mandatory for `choice` and `score`; malformed criteria are
rejected with `400` instead of being silently coerced.

### Routes

| Route | Purpose |
| :--- | :--- |
| `POST /v1/systemone` | inference |
| `GET /` | live dashboard |
| `GET /dashboard` | live dashboard (alias) |
| `GET /events` | Server-Sent Events, one `req` event per inference |
| `GET /history` | last 50 requests (dashboard backfill) |
| `GET /health`, `GET /ready` | liveness probes |

### Errors

Every error is a JSON envelope:

```json
{ "error": { "code": 400, "message": "question 'category': type 'choice' requires a 'criteria' object or array of strings" } }
```

| Code | Meaning |
| :--- | :--- |
| `400` | malformed request (unknown type, missing / empty / non-string `criteria`) |
| `413` | request body over `JEV_MAX_BODY_BYTES` |
| `503` | too many concurrent inferences (`JEV_MAX_CONCURRENCY`) |
| `504` | inference timed out (`JEV_TIMEOUT`) |

---

## ⚙️ Configuration

Everything is environment-based; all defaults work out of the box.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Bind address. A non-loopback host prints a security warning (`JEV_ALLOW_REMOTE=1` acknowledges it). |
| `PORT` | `8000` | Listen port. |
| `JEV_MODEL_PATH` | `model_onnx/model.onnx` | ONNX model file. |
| `JEV_TOKENIZER_PATH` | `model_onnx/tokenizer.json` | Tokenizer file. |
| `JEV_TEMPERATURE` | `1.65` | RLCD calibration temperature; must be finite and `0 < t ≤ 10`, otherwise the server refuses to start. |
| `JEV_MAX_BODY_BYTES` | `131072` | Request-body limit (`413`). |
| `JEV_MAX_CONCURRENCY` | `2` | Concurrent forward passes (`503`). |
| `JEV_TIMEOUT` | `30` | Per-request inference timeout in seconds (`504`). |

---

## 📊 Live dashboard

Open **http://127.0.0.1:8000/** — a built-in telemetry page with:

- a live request feed (state preview, per-question answers, latency badge),
- a rolling latency chart of the last 40 requests,
- tiles for requests, uptime, average latency and processed tokens,
- automatic backfill of the last 50 requests from `/history`.

The feed is driven by Server-Sent Events; slow clients receive a `lagged`
notice instead of stalling the engine.

---

## 📈 Performance

| Backend | Hardware | Latency (8 parallel evaluations) | RAM / VRAM |
| :--- | :--- | :--- | :--- |
| **Jev-RS (DirectML)** | Consumer GPU (DirectX 12) | **~200 ms** | **~450 MB** |
| **Jev-RS (CPU)** | 16-thread CPU (AVX-512) | ~850 ms | ~400 MB |
| Traditional LLM | GPU (token-by-token JSON) | ~1,500 – 3,500 ms | 4 – 8 GB |

> ℹ️ Illustrative back-of-envelope estimates — measure on your own hardware.
> Inference runs off the async worker threads, so `/health` and the dashboard
> stay responsive while a batch is being evaluated.

---

## 🗂️ Project layout

```
jev-rs/
├── src/
│   ├── main.rs          # HTTP server, batched inference, calibration
│   ├── dashboard.rs     # SSE + history endpoints
│   └── dashboard.html   # self-contained telemetry page
├── scripts/
│   ├── setup_model.py   # ONNX export + IO validation
│   ├── test_request.ps1 # smoke test (PowerShell)
│   └── test_request.sh  # smoke test (bash)
├── .cargo/config.toml   # MSVC linker flags, macOS availability symbols
├── install.bat          # Windows one-time setup
├── run.bat              # Windows launcher
└── Cargo.toml
```

---

## 🛠️ Development

```bash
cargo test --release          # 19 unit tests: validation, calibration, entropy
cargo build --release         # Windows / DirectML
cargo build --release --no-default-features   # CPU-only build
```

The core is covered by unit tests: question flattening and its rejection
cases, softmax calibration values, confidence bounds and entailment-index
resolution from the model config.

---

## 📦 Releases

Releases are automated: bumping `version` in `Cargo.toml` and pushing to
`main` creates a `v*` tag and publishes a GitHub Release with the portable
Windows build plus GitHub's source archives.

```bash
git tag v0.x.y && git push     # or just change the version in Cargo.toml
```

---

## 📜 License

MIT — free for commercial and private use. See [LICENSE](LICENSE).
