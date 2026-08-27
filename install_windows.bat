@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo   Privacy-First Local Anonymizer - Windows Installer
echo =======================================================
echo.

:: Check uv
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FEHLER] 'uv' wurde nicht im PATH gefunden.
    echo Bitte installieren Sie uv ueber: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

echo 1. Installiere/Aktualisiere Python-Abhaengigkeiten...
uv sync --extra gui --extra dev
if %ERRORLEVEL% NEQ 0 (
    echo [FEHLER] 'uv sync' ist fehlgeschlagen.
    pause
    exit /b 1
)

echo.
echo 2. Erstelle Desktop- und Startmenue-Verknuepfungen...

set "TARGET_DIR=%~dp0"
set "BAT_FILE=%TARGET_DIR%start_windows.bat"
set "DESKTOP_DIR=%USERPROFILE%\Desktop"
set "STARTMENU_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

powershell -ExecutionPolicy Bypass -Command ^
  "$WshShell = New-Object -comObject WScript.Shell; " ^
  "$Shortcut = $WshShell.CreateShortcut('%DESKTOP_DIR%\Local Anonymizer.lnk'); " ^
  "$Shortcut.TargetPath = '%BAT_FILE%'; " ^
  "$Shortcut.WorkingDirectory = '%TARGET_DIR%'; " ^
  "$Shortcut.WindowStyle = 7; " ^
  "$Shortcut.Description = 'Privacy-First Local Document Anonymizer'; " ^
  "$Shortcut.Save(); " ^
  "$StartShortcut = $WshShell.CreateShortcut('%STARTMENU_DIR%\Local Anonymizer.lnk'); " ^
  "$StartShortcut.TargetPath = '%BAT_FILE%'; " ^
  "$StartShortcut.WorkingDirectory = '%TARGET_DIR%'; " ^
  "$StartShortcut.WindowStyle = 7; " ^
  "$StartShortcut.Description = 'Privacy-First Local Document Anonymizer'; " ^
  "$StartShortcut.Save();"

echo.
echo [ERFOLGREICH] Verknuepfung 'Local Anonymizer' wurde auf dem Desktop und im Startmenue erstellt!
echo Sie koennen die App ab jetzt direkt ueber den Desktop oder das Startmenue ohne Terminal starten.
echo.
pause
