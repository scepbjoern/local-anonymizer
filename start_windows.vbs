' Privacy-First Local Anonymizer - Silent Direct Windows Launcher
Set FSO = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = scriptDir

' Terminate any orphaned previous instances of this app to ensure clean port binding
On Error Resume Next
WshShell.Run "powershell -NoProfile -NonInteractive -Command ""Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'pythonw.exe' -or $_.Name -eq 'python.exe') -and $_.CommandLine -like '*local-anonymizer*' -and $_.CommandLine -like '*app.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }""", 0, True
On Error GoTo 0

pyExe = scriptDir & "\.venv\Scripts\pythonw.exe"
appPy = scriptDir & "\app.py"
If FSO.FileExists(pyExe) Then
    ' Launch main application directly (< 200ms instant UI, zero splash delay)
    WshShell.Run """" & pyExe & """ """ & appPy & """", 0, False
Else
    ' Fallback to uv
    WshShell.Run "uv run python app.py", 0, False
End If

