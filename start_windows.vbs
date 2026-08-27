' Privacy-First Local Anonymizer - Silent Windows Launcher with Instant Splash
Set FSO = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

' Explicitly set CurrentDirectory to the script folder
scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

' Launch lightweight splash screen instantly (< 100ms) for immediate visual user feedback
WshShell.Run "uv run python splash.py", 0, False

' Run main application silently without console window
WshShell.Run "uv run --extra gui python app.py", 0, False
