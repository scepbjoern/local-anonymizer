import json
import uuid
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from app import (
    AppState,
    validate_and_resolve_upload_paths,
    cleanup_upload_paths,
    extract_upload_payload,
)


def test_validate_and_resolve_upload_paths_valid(tmp_path: Path):
    valid_id = str(uuid.uuid4())
    paths = validate_and_resolve_upload_paths(valid_id, upload_dir=tmp_path)
    assert paths is not None
    bin_path, meta_path = paths

    assert bin_path.parent.resolve() == tmp_path.resolve()
    assert meta_path.parent.resolve() == tmp_path.resolve()
    assert bin_path.name == f"{valid_id}.bin"
    assert meta_path.name == f"{valid_id}.json"


def test_validate_and_resolve_upload_paths_rejects_traversal_and_malformed(tmp_path: Path):
    invalid_ids = [
        "",
        None,
        "   ",
        12345,
        "../test",
        "..\\test",
        "../../etc/passwd",
        "..\\..\\Windows\\System32",
        "valid-uuid-with-extra/../path",
        "not-a-valid-uuid-string",
        "12345678-1234-1234-1234-12345678901z",  # invalid hex char
        "/root/file",
        "C:\\autoexec.bat",
    ]
    for bad_id in invalid_ids:
        res = validate_and_resolve_upload_paths(bad_id, upload_dir=tmp_path)
        assert res is None, f"Expected None for invalid id: {bad_id}"


def test_cleanup_upload_paths(tmp_path: Path):
    bin_f = tmp_path / "test.bin"
    meta_f = tmp_path / "test.json"
    bin_f.write_bytes(b"sample binary content")
    meta_f.write_text("{}", encoding="utf-8")

    assert bin_f.exists()
    assert meta_f.exists()

    cleanup_upload_paths(bin_f, meta_f)

    assert not bin_f.exists()
    assert not meta_f.exists()

    # Idempotent: cleaning up non-existent paths should not raise
    cleanup_upload_paths(bin_f, meta_f)
    cleanup_upload_paths(None, None)


def test_extract_upload_payload_success(tmp_path: Path):
    file_id = str(uuid.uuid4())
    bin_f = tmp_path / f"{file_id}.bin"
    meta_f = tmp_path / f"{file_id}.json"
    content = b"Sample Document Data"
    bin_f.write_bytes(content)
    meta_f.write_text(json.dumps({"filename": "bericht.txt"}), encoding="utf-8")

    event_data = {"file_id": file_id, "name": "fallback.txt"}
    raw_bytes, filename, temp_paths = extract_upload_payload(event_data, upload_dir=tmp_path)

    assert raw_bytes == content
    assert filename == "bericht.txt"
    assert temp_paths == (bin_f.resolve(), meta_f.resolve())


def test_extract_upload_payload_rejects_malformed_file_id(tmp_path: Path):
    event_data = {"file_id": "../evil_path", "name": "evil.txt"}
    with pytest.raises(ValueError, match="Ungültige oder unzulässige file_id"):
        extract_upload_payload(event_data, upload_dir=tmp_path)


def test_extract_upload_payload_missing_binary(tmp_path: Path):
    file_id = str(uuid.uuid4())
    # Do not create the .bin file
    event_data = {"file_id": file_id, "name": "missing.txt"}
    with pytest.raises(ValueError, match="existiert nicht"):
        extract_upload_payload(event_data, upload_dir=tmp_path)


@pytest.mark.parametrize("channel", ["main", "restore", "mapping"])
def test_drop_cleanup_on_blocked_mutation(tmp_path: Path, channel: str):
    """
    Test that when is_llm_running is True (mutation blocked), temp files are
    immediately deleted, and the AppState remains completely unaffected.
    """
    state = AppState()
    state.is_llm_running = True  # LLM is active -> mutations blocked
    state.raw_text = "Initial State Text"
    state.restore_anon_text = "Initial Anon Text"
    state.restore_mapping = {"[TEST]": "Initial"}

    file_id = str(uuid.uuid4())
    bin_f = tmp_path / f"{file_id}.bin"
    meta_f = tmp_path / f"{file_id}.json"

    if channel == "mapping":
        payload = json.dumps({"[PERSON_1]": "Max Mustermann"}).encode("utf-8")
        filename = "mapping.json"
    else:
        payload = b"Neuer Dropped Text Inhalt"
        filename = "neues_dokument.txt"

    bin_f.write_bytes(payload)
    meta_f.write_text(json.dumps({"filename": filename}), encoding="utf-8")

    assert bin_f.exists()
    assert meta_f.exists()

    event_data = {"file_id": file_id, "name": filename}

    # Simulate execution of drop handler logic with check_mutation_allowed guard
    temp_paths = None
    try:
        raw_bytes, fn, temp_paths = extract_upload_payload(event_data, upload_dir=tmp_path)
        if not (not state.is_llm_running):  # check_mutation_allowed returns False
            pass
        else:
            if channel == "main":
                state.raw_text = raw_bytes.decode("utf-8")
            elif channel == "restore":
                state.restore_anon_text = raw_bytes.decode("utf-8")
            elif channel == "mapping":
                state.restore_mapping = json.loads(raw_bytes.decode("utf-8"))
    finally:
        if temp_paths:
            cleanup_upload_paths(*temp_paths)

    # 1. Verify files are immediately deleted from disk
    assert not bin_f.exists()
    assert not meta_f.exists()

    # 2. Verify state was NOT modified
    assert state.raw_text == "Initial State Text"
    assert state.restore_anon_text == "Initial Anon Text"
    assert state.restore_mapping == {"[TEST]": "Initial"}


@pytest.mark.parametrize("channel", ["main", "restore", "mapping"])
def test_drop_cleanup_on_success_and_exception(tmp_path: Path, channel: str):
    """
    Test that temp files are cleaned up on both successful processing and exceptions.
    """
    file_id = str(uuid.uuid4())
    bin_f = tmp_path / f"{file_id}.bin"
    meta_f = tmp_path / f"{file_id}.json"

    bin_f.write_bytes(b"valid or invalid bytes")
    meta_f.write_text(json.dumps({"filename": "file.txt"}), encoding="utf-8")

    event_data = {"file_id": file_id, "name": "file.txt"}

    # Success case
    temp_paths = None
    try:
        raw_bytes, fn, temp_paths = extract_upload_payload(event_data, upload_dir=tmp_path)
        assert raw_bytes == b"valid or invalid bytes"
    finally:
        if temp_paths:
            cleanup_upload_paths(*temp_paths)

    assert not bin_f.exists()
    assert not meta_f.exists()

    # Exception during processing case
    bin_f.write_bytes(b"corrupt")
    meta_f.write_text(json.dumps({"filename": "file.txt"}), encoding="utf-8")
    temp_paths = None
    try:
        raw_bytes, fn, temp_paths = extract_upload_payload(event_data, upload_dir=tmp_path)
        raise RuntimeError("Simulated extraction crash")
    except RuntimeError:
        pass
    finally:
        if temp_paths:
            cleanup_upload_paths(*temp_paths)

    assert not bin_f.exists()
    assert not meta_f.exists()
