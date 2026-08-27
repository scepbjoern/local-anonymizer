' Privacy-First Local Anonymizer - Silent Direct Windows Launcher
Set FSO = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

' Terminate any orphaned pythonw.exe instances to ensure clean port binding
On Error Resume Next
WshShell.Run "taskkill /f /im pythonw.exe", 0, True
On Error GoTo 0

pyExe = scriptDir & "\.venv\Scripts\pythonw.exe"
If FSO.FileExists(pyExe) Then
    ' Launch main application directly (< 200ms instant UI, zero splash delay)
    WshShell.Run """" & pyExe & """ app.py", 0, False
Else
    ' Fallback to uv
    WshShell.Run "uv run python app.py", 0, False
End If

