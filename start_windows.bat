@echo off
setlocal
cd /d "%~dp0"

echo Starte Privacy-First Local Anonymizer im Hintergrund (Silent Mode)...

REM Check if uv is available
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FEHLER] 'uv' wurde nicht gefunden. Bitte installieren Sie 'uv' (https://astral.sh/uv).
    pause
    exit /b 1
)

REM Run via silent VBScript launcher
wscript.exe "%~dp0start_windows.vbs"

exit /b 0
