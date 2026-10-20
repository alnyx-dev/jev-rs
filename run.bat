@echo off
REM ============================================================================
REM  Jev-RS :: Server launcher
REM
REM  Starts the decision engine and opens the live telemetry dashboard.
REM
REM  Environment variables (optional):
REM    PORT=8000                 listen port
REM    HOST=127.0.0.1            bind address
REM    JEV_MODEL_PATH=...        full path to model.onnx
REM    JEV_TOKENIZER_PATH=...    full path to tokenizer.json
REM    JEV_TEMPERATURE=1.65      calibration temperature
REM    JEV_MAX_BODY_BYTES=131072 request body limit
REM    JEV_MAX_CONCURRENCY=2     concurrent inference limit
REM    JEV_TIMEOUT=30            per-request inference timeout, seconds
REM
REM  Any argument is passed through to the server, e.g.:  run.bat --help
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Jev-RS :: Server

if not defined PORT set "PORT=8000"

if not exist "model_onnx\model.onnx" (
    if not defined JEV_MODEL_PATH (
        echo [X] Model not found: model_onnx\model.onnx
        echo     Run install.bat once to export the model,
        echo     or set JEV_MODEL_PATH to an existing model.onnx.
        echo.
        pause
        exit /b 1
    )
)

if not exist "target\release\jev-rs.exe" (
    echo [*] Server binary not found - building it now ...
    set "CXXFLAGS=/MD"
    set "CFLAGS=/MD"
    cargo build --release --locked
    if errorlevel 1 (
        echo [X] Build failed. Run install.bat for a full setup.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [+] Jev-RS starting on http://127.0.0.1:%PORT%/
echo [+] Dashboard: http://127.0.0.1:%PORT%/
echo [+] Press Ctrl+C to stop the server.
echo.

REM Open the dashboard shortly after the server comes up.
start "" /b powershell -NoProfile -Command ^
    "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:%PORT%/'"

"target\release\jev-rs.exe" %*

echo.
if errorlevel 1 (
    echo [X] The server exited with an error. Scroll up for the message.
) else (
    echo [i] The server has stopped.
)
echo.
pause
