@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist "env\Scripts\python.exe" set "PYTHON_EXE=env\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

if "%PYTHON_EXE%"=="" (
  echo [ERROR] Virtual environment belum ditemukan (env atau .venv).
  echo Silakan ikuti langkah instalasi di README.md terlebih dahulu.
  pause
  exit /b 1
)

echo Menjalankan Notulen Rapat...
"%PYTHON_EXE%" app_gradio.py
pause
