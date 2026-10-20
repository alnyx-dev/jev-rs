@echo off
REM ============================================================================
REM  Jev-RS :: launcher (installed copy)
REM  Starts the local decision engine and opens the live telemetry dashboard.
REM ============================================================================
setlocal
cd /d "%~dp0"

if not defined PORT set "PORT=8000"
set "JEV_MODEL_PATH=%~dp0model_onnx\model.onnx"
set "JEV_TOKENIZER_PATH=%~dp0model_onnx\tokenizer.json"

if not exist "%JEV_MODEL_PATH%" (
    echo [X] The decision model is missing: "%JEV_MODEL_PATH%"
    echo     Downloading it now ^(~2.3 GB, one time^)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-model.ps1" -DestDir "%~dp0model_onnx"
    if errorlevel 1 (
        echo [X] Model download failed. Check your connection and try again.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [+] Jev-RS starting on http://127.0.0.1:%PORT%/
echo [+] Dashboard: http://127.0.0.1:%PORT%/
echo [+] Press Ctrl+C to stop.
echo.

start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 4; Start-Process 'http://127.0.0.1:%PORT%/'"

"%~dp0jev-rs.exe"
echo.
pause
