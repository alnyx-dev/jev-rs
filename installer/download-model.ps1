<#
.SYNOPSIS
    Downloads the Jev-RS decision model (ONNX) from Hugging Face.

.DESCRIPTION
    The model is the exported BGE-M3 zeroshot NLI checkpoint used by Jev-RS
    for single-pass decision evaluation. Files are resumed when a partial
    download is present, and already-complete files are skipped.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File download-model.ps1
    powershell -ExecutionPolicy Bypass -File download-model.ps1 -DestDir D:\models\jev-rs
#>
[CmdletBinding()]
param(
    [string] $DestDir = (Join-Path $PSScriptRoot 'model_onnx'),
    [string] $BaseUrl = 'https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0/resolve/main/onnx',
    [string[]] $Files,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (-not $Files -or $Files.Count -eq 0) {
    $Files = @(
        'model.onnx',
        'model.onnx_data',
        'tokenizer.json',
        'config.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'sentencepiece.bpe.model'
    )
}

function Get-RemoteSize {
    param([string] $Url)
    try {
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.Method = 'HEAD'
        $request.AllowAutoRedirect = $true
        $request.Timeout = 30000
        $response = $request.GetResponse()
        $length = [int64] $response.ContentLength
        $response.Close()
        return $length
    } catch {
        return -1
    }
}

function Invoke-Download {
    param([string] $Url, [string] $Destination)

    $curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
    if ($curl) {
        & curl.exe -L --fail --retry 5 --retry-delay 3 --continue-at - --output $Destination $Url
        if ($LASTEXITCODE -ne 0) { throw "curl failed with exit code $LASTEXITCODE" }
        return
    }

    Add-Type -AssemblyName System.Net.Http | Out-Null
    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromHours(2)
    try {
        $response = $client.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        $response.EnsureSuccessStatusCode() | Out-Null
        $total = $response.Content.Headers.ContentLength
        if (-not $total) { $total = 0L }
        $input = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $output = [System.IO.File]::Create($Destination)
        try {
            $buffer = New-Object byte[] (1MB)
            $sum = 0L
            $last = 0L
            while (($read = $input.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $output.Write($buffer, 0, $read)
                $sum += $read
                if (($sum - $last) -ge 25MB) {
                    Write-Host ("      {0:N0} MB / {1:N0} MB" -f ($sum / 1MB), ($total / 1MB))
                    $last = $sum
                }
            }
        } finally {
            $output.Dispose()
            $input.Dispose()
        }
    } finally {
        $client.Dispose()
    }
}

Write-Host ''
Write-Host '=== Jev-RS :: model download ===' -ForegroundColor Cyan
Write-Host "Destination : $DestDir"
Write-Host "Source      : $BaseUrl"
Write-Host ''

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
$started = Get-Date

foreach ($name in $Files) {
    $url = "$BaseUrl/$name"
    $target = Join-Path $DestDir $name
    $remoteSize = Get-RemoteSize -Url $url

    if (-not $Force -and (Test-Path $target)) {
        $localSize = (Get-Item $target).Length
        if ($remoteSize -gt 0 -and $localSize -eq $remoteSize) {
            Write-Host ("  [=] {0,-26} already complete ({1:N1} MB)" -f $name, ($localSize / 1MB)) -ForegroundColor DarkGray
            continue
        }
    }

    Write-Host ("  [>] {0,-26} downloading..." -f $name) -ForegroundColor Yellow
    Invoke-Download -Url $url -Destination $target

    if ($remoteSize -gt 0) {
        $localSize = (Get-Item $target).Length
        if ($localSize -ne $remoteSize) {
            throw ("{0}: size mismatch (expected {1} bytes, got {2})" -f $name, $remoteSize, $localSize)
        }
    }
    Write-Host ("  [+] {0,-26} done ({1:N1} MB)" -f $name, ((Get-Item $target).Length / 1MB)) -ForegroundColor Green
}

$elapsed = (Get-Date) - $started
Write-Host ''
Write-Host ("Model ready in {0:N0}s -> {1}" -f $elapsed.TotalSeconds, $DestDir) -ForegroundColor Green
Write-Host 'Start the engine with: run-jev-rs.bat'
Write-Host ''
