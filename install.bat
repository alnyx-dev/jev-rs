@echo off
REM ============================================================================
REM  Jev-RS :: Installer
REM
REM  One-time setup:
REM    1. Rust toolchain (rustup, if missing) + release build
REM    2. Model download (pre-exported ONNX bundle from the Hugging Face Hub)
REM
REM  No Python, no virtualenv: the engine downloads a ready-to-run ONNX export.
REM
REM  Skip individual steps with environment variables:
REM    set JEV_SKIP_MODEL=1  skip the model download
REM    set JEV_SKIP_RUST=1  skip toolchain install and cargo build
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Jev-RS :: Installer

echo.
echo ============================================================
echo  Jev-RS :: Installer
echo ============================================================
echo.

REM --- 1. Rust + build ------------------------------------------------------
if not defined JEV_SKIP_RUST (
    where cargo >nul 2>&1
    if errorlevel 1 (
        echo [1/2] Rust toolchain not found - installing rustup ^(stable, minimal^) ...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "Invoke-WebRequest -UseBasicParsing https://win.rustup.rs/x86_64 -out '%CD%\rustup-init.exe'" || goto :fail
        "%CD%\rustup-init.exe" -y --profile minimal --default-toolchain stable || goto :fail
        del "%CD%\rustup-init.exe" >nul 2>&1
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
    echo [1/2] Building the server ^(release, DirectML^) ...
    echo      If linking fails, install the Visual Studio Build Tools ^(MSVC^).
    set "CXXFLAGS=/MD"
    set "CFLAGS=/MD"
    cargo build --release --locked || goto :fail
) else (
    echo [1/2] Skipping toolchain install and build ^(JEV_SKIP_RUST=1^).
)

REM --- 2. Model --------------------------------------------------------------
if not defined JEV_SKIP_MODEL (
    if exist "model_onnx\model.onnx" if exist "model_onnx\model.onnx_data" (
        echo [2/2] Model already present: model_onnx\model.onnx
    ) else (
        echo [2/2] Downloading the model ^(one-time ~2.3 GB^) ...
        echo      The download is resumable - run it again if the connection drops.
        "target\release\jev-rs.exe" --download-model || goto :fail
    )
) else (
    echo [2/2] Skipping model download ^(JEV_SKIP_MODEL=1^).
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
