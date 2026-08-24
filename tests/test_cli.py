import subprocess
import sys
from pathlib import Path

def test_cli_clean_error_on_missing_file():
    cli_path = Path(__file__).parent.parent / "cli.py"
    result = subprocess.run([sys.executable, str(cli_path), "anonymize", "does_not_exist.txt"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "File not found" in result.stdout or "Error:" in result.stdout or "Error:" in result.stderr
