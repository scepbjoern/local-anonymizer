"""Tests for the static LLM model catalog and validation (Phase 6A.1)."""

import json
import pytest
from pathlib import Path

from local_anonymizer.llm.schema import (
    CatalogSchema,
    CatalogModelEntry,
    CatalogPhaseEvaluation,
    validate_model_name,
)
from local_anonymizer.llm.catalog import (
    load_catalog,
    find_catalog_entry,
    get_model_suitability_badge,
    CatalogError,
)


def test_validate_model_name_valid():
    assert validate_model_name("qwen3:8b") == "qwen3:8b"
    assert validate_model_name("ministral-3:8b") == "ministral-3:8b"
    assert validate_model_name("deepseek-r1:7b") == "deepseek-r1:7b"
    assert validate_model_name("my_model.v1:latest") == "my_model.v1:latest"
    assert validate_model_name("hf.co/user/repo:tag") == "hf.co/user/repo:tag"


def test_validate_model_name_cloud_blocked():
    with pytest.raises(ValueError, match="Cloud-Modelle"):
        validate_model_name("qwen3:8b:cloud")

    with pytest.raises(ValueError, match="Cloud-Modelle"):
        validate_model_name("model:CLOUD")

    with pytest.raises(ValueError, match="Cloud-Modelle"):
        validate_model_name("llama3:cloud")


def test_validate_model_name_invalid_chars_or_length():
    with pytest.raises(ValueError, match="nicht leer"):
        validate_model_name("")

    with pytest.raises(ValueError, match="nicht leer"):
        validate_model_name("   ")

    with pytest.raises(ValueError, match="128 Zeichen"):
        validate_model_name("a" * 129)

    with pytest.raises(ValueError, match="unzulässige Zeichen"):
        validate_model_name("model name with spaces")

    with pytest.raises(ValueError, match="unzulässige Zeichen"):
        validate_model_name("model;rm -rf /")

    with pytest.raises(ValueError, match="unzulässige Zeichen"):
        validate_model_name("model\nname")


def test_bundled_catalog_loads_and_validates():
    catalog = load_catalog(force_reload=True)
    assert catalog.schema_version == "1.0.0"
    assert len(catalog.models) >= 3

    # Check that qwen3:8b is recommended for 6A and untested for 6B
    qwen = find_catalog_entry("qwen3:8b", catalog=catalog)
    assert qwen is not None
    assert qwen.canonical_name == "qwen3:8b"
    assert qwen.phase_6a_triage.status == "recommended"
    assert qwen.phase_6b_smart_linking.status == "untested"
    assert "UAT" in qwen.phase_6a_triage.reason


def test_find_catalog_entry_exact_and_case_insensitive():
    catalog = load_catalog()
    assert find_catalog_entry("QWEN3:8B", catalog=catalog) is not None
    assert find_catalog_entry("ministral-3:8b", catalog=catalog) is not None
    assert find_catalog_entry("non_existent_model:latest", catalog=catalog) is None
    assert find_catalog_entry("", catalog=catalog) is None


def test_get_model_suitability_badge():
    catalog = load_catalog()

    label, color, tooltip = get_model_suitability_badge("qwen3:8b", phase="phase_6a_triage", catalog=catalog)
    assert label == "Empfohlen"
    assert color == "positive"
    assert "Referenzmodell" in tooltip

    label_6b, color_6b, _ = get_model_suitability_badge("qwen3:8b", phase="phase_6b_smart_linking", catalog=catalog)
    assert label_6b == "Nicht evaluiert"

    label_unknown, color_unknown, _ = get_model_suitability_badge("custom_unknown:7b", catalog=catalog)
    assert label_unknown == "Nicht evaluiert"
    assert color_unknown == "grey-7"


def test_catalog_extra_fields_forbidden(tmp_path):
    bad_data = {
        "schema_version": "1.0.0",
        "unknown_root_field": 123,
        "models": [],
    }
    bad_file = tmp_path / "bad_catalog.json"
    bad_file.write_text(json.dumps(bad_data), encoding="utf-8")

    with pytest.raises(CatalogError):
        load_catalog(bad_file)


def test_catalog_missing_file():
    with pytest.raises(CatalogError, match="nicht gefunden"):
        load_catalog(Path("non_existent_dir/missing_catalog.json"))
