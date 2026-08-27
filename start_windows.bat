@echo off
setlocal
cd /d "%~dp0"

echo Starte Privacy-First Local Anonymizer im Hintergrund (Silent Mode)...

:: Check if uv is available
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FEHLER] 'uv' wurde nicht gefunden. Bitte installieren Sie 'uv' (https://astral.sh/uv).
    pause
    exit /b 1
)

:: Sync dependencies if needed and launch silently with pythonw
start "" uv run --extra gui pythonw app.py

exit /b 0
