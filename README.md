# ⚡ Jev-RS

**Blazing-fast, local implementation of System One AI Decision Models.**  
*Wire-compatible with the TypeSafe AI / Jev protocol.*

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-DirectML-blue.svg)](https://onnxruntime.ai/)
[![GitHub](https://img.shields.io/badge/github-alnyx--dev%2Fjev--rs-181717.svg?logo=github)](https://github.com/alnyx-dev/jev-rs)

Traditional LLMs are slow, non-deterministic, and generate tokens sequentially. **Jev-RS** is a non-autoregressive decision engine that performs batch hypothesis evaluation in a **single forward pass** with zero token generation overhead.

---

## 🚀 Features

- **True System One Architecture:** Zero autoregressive token generation. All decisions evaluated simultaneously.
- **Wire-Compatible:** Drop-in local replacement for TypeSafe AI API (`POST /v1/systemone`).
- **Hardware Acceleration:** Native DirectML (DirectX 12 GPU acceleration for Windows) and CPU fallback.
- **RLCD-Calibrated Probabilities:** Uses temperature scaling to eliminate neural overconfidence.
- **Entropy-Based Confidence:** Calculates statistical certainty via normalized Shannon entropy $1 - \frac{H(P)}{\log_2(K)}$.
- **Typed Primitives:** Native support for `noul` (binary), `choice` (categorical), and `score` (ordered rubric).

---

## ⚡ Performance

| Backend | Hardware | Latency (8 parallel evaluations) | RAM / VRAM |
| :--- | :--- | :--- | :--- |
| **Jev-RS (DirectML)** | **Consumer GPU (DirectX 12)** | **~200 ms** | **~450 MB** |
| **Jev-RS (CPU)** | 16-thread CPU (AVX-512) | ~850 ms | ~400 MB |
| **Traditional LLM** | GPU (Token-by-token JSON) | ~1,500 - 3,500 ms | 4 - 8 GB |

> ℹ️ Latency numbers are illustrative back-of-envelope estimates — measure on your own hardware.

---

## 📦 Quickstart

### 1. Prerequisites
- [Rust](https://rustup.rs/) (1.80+)
- Python (only for the one-time ONNX export)

### 2. Export Model
```bash
pip install optimum[onnxruntime] transformers
python scripts/setup_model.py
```

### 3. Build & Run
```bash
# Set C++ compiler flags for Windows MSVC (if needed)
$env:CXXFLAGS = "/MD"
$env:CFLAGS = "/MD"

cargo run --release
```

Server starts on `http://127.0.0.1:8000`.

> ⚠️ Debug builds won't link on Windows: the downloaded ONNX Runtime binaries use `/MD` (release CRT). Always use `--release`.

### 4. Live Dashboard
Open **http://127.0.0.1:8000/** — a built-in telemetry page with real-time request feed.

| Route | Description |
| :--- | :--- |
| `GET /` | Live dashboard: request feed, latency chart, uptime & stats (SSE-driven) |
| `GET /events` | Server-Sent Events stream, one `req` event per inference |
| `GET /history` | Last 50 requests (backfills the dashboard on page load) |
| `GET /health`, `GET /ready` | Liveness probes |

## 📦 Releases

Releases are **source-only**: bumping `version` in `Cargo.toml` and pushing to
`main` creates a `v*` tag and a GitHub Release (with the automatic source
archives and generated release notes). No prebuilt binaries are attached —
build locally:

```bash
pip install optimum[onnxruntime] transformers torch
python scripts/setup_model.py          # one-time ONNX export

cargo build --release                  # Windows: DirectML GPU
# cargo build --release --no-default-features   # Linux/macOS: CPU

./target/release/jev-rs
```

### 🪟 Windows quick start

**Portable build.** Download `jev-rs-windows-x86_64.zip` from the latest
release, extract it, place the `model_onnx` folder next to `jev-rs.exe` (or
point `JEV_MODEL_PATH` / `JEV_TOKENIZER_PATH` at an existing model) and run
the binary.

**From source.** Two batch scripts do everything for you:

| Script | What it does |
| :--- | :--- |
| **`install.bat`** | one-time setup: creates a `.venv`, installs `optimum`/`transformers`/`torch`, exports the ONNX model (`model_onnx\`, ~2.3 GB), installs Rust if missing, builds the release binary |
| **`run.bat`** | starts the server and opens the live dashboard in the browser |

```bat
install.bat        :: run once
run.bat            :: start the engine (Ctrl+C to stop)
```

Skip parts of the installation with `JEV_SKIP_PIP=1`, `JEV_SKIP_MODEL=1` or
`JEV_SKIP_RUST=1`. `run.bat` honours `PORT`, `HOST`, `JEV_MODEL_PATH`,
`JEV_TOKENIZER_PATH`, `JEV_TEMPERATURE` and the other `JEV_*` settings.

> The model is not shipped in releases: it is the exported
> `MoritzLaurer/bge-m3-zeroshot-v2.0` ONNX checkpoint, pulled from
> Hugging Face by `scripts/setup_model.py`.

### Model files

The model is ~2.3 GB, so it is never shipped with a release. Export it once
with `scripts/setup_model.py` and the server will find it automatically:

| Location | Priority |
| :--- | :--- |
| `JEV_MODEL_PATH` / `JEV_TOKENIZER_PATH` env vars | 1 (explicit override) |
| `model_onnx/` in the working directory | 2 |
| `model_onnx/` next to the executable | 3 |

## ⚙️ Configuration & limits

Everything works with defaults; all knobs are environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address. Binding a non-loopback host prints a security warning (`JEV_ALLOW_REMOTE=1` acknowledges it). |
| `JEV_MODEL_PATH` | `model_onnx/model.onnx` | ONNX model file. |
| `JEV_TOKENIZER_PATH` | `model_onnx/tokenizer.json` | Tokenizer file. |
| `JEV_TEMPERATURE` | `1.65` | RLCD calibration temperature; must be finite with `0 < t <= 10`, otherwise the server refuses to start. |
| `JEV_MAX_BODY_BYTES` | `131072` (128 KiB) | Request-body limit (`413` on exceed). |
| `JEV_MAX_CONCURRENCY` | `2` | Max concurrent forward passes (`503` when exhausted). |
| `JEV_TIMEOUT` | `30` (seconds) | Per-request inference timeout (`504`). |

Error status codes:

| Code | Meaning |
| :--- | :--- |
| `400` | Malformed `questions` (unknown type, missing/empty/non-string `criteria`) |
| `413` | Request body over the size limit |
| `503` | Too many concurrent inferences |
| `504` | Inference timed out (`JEV_TIMEOUT`) |

> ℹ️ **Platforms:** DirectML GPU acceleration is Windows-only. Build with
> `cargo build --release --no-default-features` on Linux / macOS for CPU
> inference (the feature-gated `dml` flag routes to plain ONNX Runtime EPs).

---

## 📡 API Usage

### Example Request (`POST /v1/systemone`)

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

### Example Response

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
      "probabilities": {
        "billing": 0.8282,
        "tech_bug": 0.1148,
        "general": 0.0570
      },
      "confidence": 0.4831
    },
    "frustration": {
      "type": "score",
      "score": 2.24,
      "probabilities": {
        "Calm": 0.0719,
        "Annoyed": 0.6128,
        "Extremely Furious": 0.3153
      },
      "confidence": 0.2233
    }
  },
  "usage": {
    "input_tokens": 66,
    "output_tokens": 0
  }
}
```

---

## 📜 License
MIT License. Free for commercial and private use.