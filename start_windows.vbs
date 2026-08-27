' Privacy-First Local Anonymizer - Silent Windows Launcher
Set FSO = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

' Explicitly set CurrentDirectory to the script's folder so uv run finds pyproject.toml
scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

' Run silently without command window (0 = hide window, False = don't wait)
WshShell.Run "uv run --extra gui python app.py", 0, False
