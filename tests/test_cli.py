import subprocess
import sys
from pathlib import Path

def test_cli_clean_error_on_missing_file():
    cli_path = Path(__file__).parent.parent / "cli.py"
    result = subprocess.run([sys.executable, str(cli_path), "anonymize", "does_not_exist.txt"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "File not found" in result.stdout or "Error:" in result.stdout or "Error:" in result.stderr


def test_cli_eupii_argument_propagation_and_config_precedence(tmp_path, monkeypatch):
    import json
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import cli

    test_doc = tmp_path / "test.txt"
    test_doc.write_text("Test Dokument", encoding="utf-8")

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "enable_eupii": False,
        "eupii_threshold": 0.45,
        "eupii_model_name": "custom/eu-pii",
    }), encoding="utf-8")

    captured_kwargs = {}

    class DummyPipeline:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def process_file(self, **kwargs):
            return type("Res", (), {
                "anonymization_result": type("AnonRes", (), {
                    "entities": [],
                    "anonymized_text": "Anonymisiert",
                })(),
                "anonymized_file": "anon.txt",
                "mapping_file": None,
                "report_file": "report.json",
            })()

    monkeypatch.setattr(cli, "AnonymizationPipeline", DummyPipeline)

    # 1. Config fallback when no CLI flag given
    cli.main_args = ["anonymize", str(test_doc), "--config", str(cfg_file)]
    monkeypatch.setattr("sys.argv", ["cli.py"] + cli.main_args)
    cli.main()
    assert captured_kwargs.get("enable_eupii") is False
    assert captured_kwargs.get("eupii_threshold") == 0.45
    assert captured_kwargs.get("eupii_model") == "custom/eu-pii"

    # 2. CLI flag overrides config
    captured_kwargs.clear()
    cli.main_args = ["anonymize", str(test_doc), "--config", str(cfg_file), "--enable-eupii", "--eupii-threshold", "0.65"]
    monkeypatch.setattr("sys.argv", ["cli.py"] + cli.main_args)
    cli.main()
    assert captured_kwargs.get("enable_eupii") is True
    assert captured_kwargs.get("eupii_threshold") == 0.65

