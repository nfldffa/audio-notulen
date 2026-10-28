@echo off
title Notulen Rapat Otomatis
echo ==========================================
echo Memulai Aplikasi Notulen Rapat Otomatis...
echo ==========================================
echo.

REM Mengaktifkan virtual environment
call env\Scripts\activate

REM Menjalankan aplikasi
python app_gradio.py

echo.
pause