param(
    [string]$Model = "qwen2.5:7b"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
py -3.10 -m venv .venv

Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Checking Ollama..." -ForegroundColor Cyan
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama was not found on PATH. Install it from https://ollama.com/download, restart PowerShell, then rerun this script." -ForegroundColor Yellow
    exit 1
}

Write-Host "Pulling local model: $Model" -ForegroundColor Cyan
ollama pull $Model

Write-Host "Running setup check..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe scripts\check_setup.py

Write-Host "Setup complete. Try: .\.venv\Scripts\python.exe run_review.py --config config.quick.json" -ForegroundColor Green
