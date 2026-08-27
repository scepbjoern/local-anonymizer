' Privacy-First Local Anonymizer - Silent Direct Windows Launcher
Set FSO = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

pyExe = scriptDir & "\.venv\Scripts\pythonw.exe"
If FSO.FileExists(pyExe) Then
    ' Launch main application directly (< 200ms instant UI, zero splash delay)
    WshShell.Run """" & pyExe & """ app.py", 0, False
Else
    ' Fallback to uv
    WshShell.Run "uv run python app.py", 0, False
End If
