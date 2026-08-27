' Privacy-First Local Anonymizer - Silent Windows Launcher
Set FSO = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

pyExe = scriptDir & "\.venv\Scripts\pythonw.exe"
If FSO.FileExists(pyExe) Then
    ' Launch splash screen using pythonw directly (instant < 50ms, 0 console, 0 uv lock collision)
    WshShell.Run """" & pyExe & """ splash.py", 0, False
    ' Launch main application
    WshShell.Run """" & pyExe & """ app.py", 0, False
Else
    ' Fallback to uv
    WshShell.Run "uv run python app.py", 0, False
End If
