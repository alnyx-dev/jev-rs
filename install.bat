@echo off
REM ============================================================================
REM  Jev-RS :: Installer
REM
REM  One-time setup:
REM    1. Python virtual environment (.venv) + dependencies
REM    2. ONNX model export (model_onnx\, ~2.3 GB download)
REM    3. Rust toolchain (rustup, if missing) + release build
REM
REM  Skip individual steps with environment variables:
REM    set JEV_SKIP_PIP=1    skip Python dependency installation
REM    set JEV_SKIP_MODEL=1  skip the ONNX model export
REM    set JEV_SKIP_RUST=1   skip toolchain install and cargo build
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Jev-RS :: Installer

echo.
echo ============================================================
echo  Jev-RS :: Installer
echo ============================================================
echo.

REM --- 1. Python -------------------------------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"

if not defined PY (
    echo [X] Python 3 was not found in PATH.
    echo     Install Python 3.10+ from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH" during installation.
    goto :fail
)
echo [+] Python: %PY%

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment .venv ...
    %PY% -m venv .venv || goto :fail
) else (
    echo [1/4] Virtual environment .venv already exists.
)
set "VPY=%CD%\.venv\Scripts\python.exe"

if not defined JEV_SKIP_PIP (
    echo [2/4] Installing Python packages ^(optimum, transformers, torch^) ...
    echo      This downloads a few GB the first time.
    "%VPY%" -m pip install --upgrade pip || goto :fail
    "%VPY%" -m pip install "optimum[onnxruntime]" transformers torch || goto :fail
) else (
    echo [2/4] Skipping Python packages ^(JEV_SKIP_PIP=1^).
)

REM --- 2. Model --------------------------------------------------------------
if not defined JEV_SKIP_MODEL (
    REM The export is split: model.onnx holds the graph, model.onnx_data holds
    REM the ~2.3 GB of external weights. A partial download must not be taken
    REM for a finished export.
    if exist "model_onnx\model.onnx" if exist "model_onnx\model.onnx_data" (
        echo [3/4] Model already exported: model_onnx\model.onnx
    ) else (
        echo [3/4] Exporting the ONNX model ^(one-time ~2.3 GB download^) ...
        "%VPY%" scripts\setup_model.py --out model_onnx || goto :fail
    )
) else (
    echo [3/4] Skipping model export ^(JEV_SKIP_MODEL=1^).
)

REM --- 3. Rust + build -------------------------------------------------------
if not defined JEV_SKIP_RUST (
    where cargo >nul 2>&1
    if errorlevel 1 (
        echo [4/4] Rust toolchain not found - installing rustup ^(stable, minimal^) ...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "Invoke-WebRequest -UseBasicParsing https://win.rustup.rs/x86_64 -OutFile '%CD%\rustup-init.exe'" || goto :fail
        "%CD%\rustup-init.exe" -y --profile minimal --default-toolchain stable || goto :fail
        del "%CD%\rustup-init.exe" >nul 2>&1
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
    echo [4/4] Building the server ^(release, DirectML^) ...
    echo      If linking fails, install the Visual Studio Build Tools ^(MSVC^).
    set "CXXFLAGS=/MD"
    set "CFLAGS=/MD"
    cargo build --release --locked || goto :fail
) else (
    echo [4/4] Skipping Rust toolchain and build ^(JEV_SKIP_RUST=1^).
)

echo.
echo ============================================================
echo  Installation complete.
echo  Start the server with:  run.bat
echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo [X] Installation failed at the step above. Read the error message,
echo     fix the cause and run install.bat again.
echo.
pause
exit /b 1
