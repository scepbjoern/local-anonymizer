import os
import re
import pytest
from local_anonymizer.recognizers import (
    AHVNumberRecognizer,
    AddressPatternRecognizer,
    FuzzyGlossaryRecognizer,
    GLiNERRecognizer,
    UIDNumberRecognizer,
    is_sentence_boundary,
    is_valid_ahv_number,
    is_valid_uid_number,
)

def test_abbreviation_aware_chunking():
    text = "Prof. Dr. Max wohnt in der Bahnhofstr. 12 am 14. Juli. Das ist super."
    # Find dots
    matches = list(re.finditer(r"\.", text))
    # Dot after Prof: should be False
    assert is_sentence_boundary(text, matches[0]) == False
    # Dot after Dr: should be False
    assert is_sentence_boundary(text, matches[1]) == False
    # Dot after Bahnhofstr: should be False
    assert is_sentence_boundary(text, matches[2]) == False
    # Dot after 14 (14. Juli): should be False
    assert is_sentence_boundary(text, matches[3]) == False
    # Dot after super: should be True
    assert is_sentence_boundary(text, matches[4]) == True

def test_fuzzy_glossary_typo_matching():
    recognizer = FuzzyGlossaryRecognizer(glossary={"ZHAW": "ORGANIZATION"}, high_confidence_threshold=80.0, review_threshold=70.0)
    results = recognizer.analyze("Die Studenten der ZHW sind hier.", entities=["ORGANIZATION"])
    # Should find ZHW as ORGANIZATION
    typo_result = next(r for r in results if r.entity_type == "ORGANIZATION" and r.start == 18 and r.end == 21)
    assert typo_result.recognition_metadata["glossary_match"] == "fuzzy"

def test_fuzzy_glossary_exact_match_priority():
    recognizer = FuzzyGlossaryRecognizer(glossary={"ZHAW": "ORGANIZATION"}, high_confidence_threshold=80.0, review_threshold=70.0)
    results = recognizer.analyze("Die ZHAW und die ZHW", entities=["ORGANIZATION"])
    # Both should be found, but ZHAW should have score 1.0
    zhaw_match = next(r for r in results if r.start == 4)
    assert zhaw_match.score == 1.0
    assert zhaw_match.recognition_metadata["glossary_match"] == "direct"


def test_glossary_exact_match_supports_punctuation():
    recognizer = FuzzyGlossaryRecognizer(glossary={"eClaims+": "IT_SYSTEM"})
    results = recognizer.analyze("Das System eClaims+ wurde aktualisiert.", entities=["IT_SYSTEM"])

    match = next(r for r in results if r.entity_type == "IT_SYSTEM")
    assert match.start == 11
    assert match.end == 19
    assert match.score == 1.0
    assert match.recognition_metadata["glossary_match"] == "direct"


def test_get_optimal_device():
    from local_anonymizer.recognizers import get_optimal_device
    device = get_optimal_device()
    assert device in {"cuda", "mps", "cpu"}


def test_gliner_batched_multi_chunk_analysis():
    recognizer = GLiNERRecognizer()
    long_text = (
        "Dr. Andreas Schönenberger leitet das Team in Zürich. " * 10 +
        "Frau Julia Meier arbeitet an der ETH Zürich in Basel. " * 10
    )
    results = recognizer.analyze(long_text, entities=["PERSON", "LOCATION", "ORGANIZATION"])
    assert len(results) > 0
    assert any(r.entity_type == "PERSON" for r in results)
    assert any(r.entity_type == "LOCATION" for r in results)


def test_address_recognizer_supports_swiss_and_german_formats():
    recognizer = AddressPatternRecognizer()

    swiss = recognizer.analyze("Besuch: Bahnhofstrasse 12a, 8001 Zürich", entities=["ADDRESS"])
    german = recognizer.analyze("Büro: Hauptstraße 7, 10115 Berlin", entities=["ADDRESS"])

    assert any(r.entity_type == "ADDRESS" and r.start == 8 and r.end == 39 for r in swiss)
    assert any(r.entity_type == "ADDRESS" and r.start == 6 and r.end == 33 for r in german)


def test_address_recognizer_rejects_year_plus_town_collision():
    recognizer = AddressPatternRecognizer()
    results = recognizer.analyze("Stand: 2026 Zürich", entities=["ADDRESS"])

    assert results == []


def test_ahv_checksum_validation_accepts_valid_and_rejects_invalid_numbers():
    valid = "756.3047.5009.62"
    invalid = "756.3047.5009.63"

    assert is_valid_ahv_number(valid) is True
    assert is_valid_ahv_number(invalid) is False
    recognizer = AHVNumberRecognizer()
    assert len(recognizer.analyze(valid, entities=["AHV_NUMBER"])) == 1
    assert recognizer.analyze(invalid, entities=["AHV_NUMBER"]) == []


def test_uid_checksum_validation_accepts_valid_and_rejects_invalid_numbers():
    valid = "CHE-105.816.788"
    invalid = "CHE-105.816.789"

    assert is_valid_uid_number(valid) is True
    assert is_valid_uid_number(invalid) is False
    recognizer = UIDNumberRecognizer()
    assert len(recognizer.analyze(valid, entities=["UID_NUMBER"])) == 1
    assert recognizer.analyze(invalid, entities=["UID_NUMBER"]) == []


def test_glossary_supported_entities_follow_configured_types():
    recognizer = FuzzyGlossaryRecognizer(glossary={"SAP": "IT_SYSTEM"})
    assert recognizer.supported_entities == ["IT_SYSTEM"]
    assert recognizer.analyze("SAP", entities=["IT_SYSTEM"])[0].entity_type == "IT_SYSTEM"

    recognizer.set_glossary({"ZHAW": "ORGANIZATION", "SAP": "IT_SYSTEM"})
    assert recognizer.supported_entities == ["IT_SYSTEM", "ORGANIZATION"]


def test_person_prompts_are_specific_and_role_prompts_are_available():
    recognizer = GLiNERRecognizer()

    assert recognizer.label_mapping["person"] == "PERSON"
    assert recognizer.label_mapping["name"] == "PERSON"
    assert recognizer.label_mapping["person name"] == "PERSON"
    assert recognizer.label_mapping["first name"] == "PERSON"
    assert recognizer.label_mapping["last name"] == "PERSON"
    assert recognizer.label_mapping["person's proper name"] == "PERSON"
    assert recognizer.label_mapping["named person"] == "PERSON"
    assert recognizer.label_mapping["proper name"] == "PERSON"
    assert recognizer.label_mapping["job title"] == "ROLE"
    assert recognizer.label_mapping["professional role"] == "ROLE"
    assert recognizer.label_mapping["position"] == "ROLE"


def test_gliner_does_not_merge_a_whitespace_separated_name_list():
    class DummyModel:
        def inference(self, chunk_texts, labels, threshold, batch_size):
            text = chunk_texts[0]
            tokens = list(re.finditer(r"\S+", text))
            return [[
                {"label": "person", "start": match.start(), "end": match.end(), "score": 0.9}
                for match in tokens
            ]]

    recognizer = GLiNERRecognizer(label_mapping={"person": "PERSON"}, threshold=0.5)
    recognizer.model = DummyModel()
    text = "Benjamin Kägi Egzona Musliu Christoph Jampen"

    results = recognizer.analyze(text, entities=["PERSON"])

    assert [text[result.start:result.end] for result in results] == [
        "Benjamin Kägi",
        "Egzona Musliu",
        "Christoph Jampen",
    ]


def test_set_huggingface_offline_mode_matrix_and_snapshot_restoration(monkeypatch):
    """
    Test all 6 combinations of initial state (None, '1') and loading path
    (Cache Hit, Miss Success, Miss Error) to ensure exact snapshot restoration.
    """
    import os
    import huggingface_hub.constants as hf_constants
    from local_anonymizer.recognizers import set_huggingface_offline_mode

    # Matrix: initial states to test
    for init_env, init_const in [(None, False), ("1", True)]:
        # 1. Cache Hit (local load in offline mode)
        if init_env is None:
            monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        else:
            monkeypatch.setenv("HF_HUB_OFFLINE", init_env)
        monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", init_const)

        with set_huggingface_offline_mode(True):
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert hf_constants.HF_HUB_OFFLINE is True

        assert os.environ.get("HF_HUB_OFFLINE") == init_env
        assert hf_constants.HF_HUB_OFFLINE is init_const

        # 2. Miss Success (online fallback in online mode)
        if init_env is None:
            monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        else:
            monkeypatch.setenv("HF_HUB_OFFLINE", init_env)
        monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", init_const)

        with set_huggingface_offline_mode(False):
            assert os.environ.get("HF_HUB_OFFLINE") is None
            assert hf_constants.HF_HUB_OFFLINE is False

        assert os.environ.get("HF_HUB_OFFLINE") == init_env
        assert hf_constants.HF_HUB_OFFLINE is init_const

        # 3. Miss Error (exception raised during online load)
        if init_env is None:
            monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        else:
            monkeypatch.setenv("HF_HUB_OFFLINE", init_env)
        monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", init_const)

        with pytest.raises(RuntimeError):
            with set_huggingface_offline_mode(False):
                assert os.environ.get("HF_HUB_OFFLINE") is None
                assert hf_constants.HF_HUB_OFFLINE is False
                raise RuntimeError("Simulated network timeout")

        assert os.environ.get("HF_HUB_OFFLINE") == init_env
        assert hf_constants.HF_HUB_OFFLINE is init_const


def test_eupii_cli_subprocess_import_isolation():
    """
    Subprocess test: Verifies that when HF_HUB_OFFLINE=1 is present in the environment
    prior to any python library imports, set_huggingface_offline_mode correctly synchronizes
    both os.environ and huggingface_hub.constants at runtime without network calls.
    """
    import subprocess
    import sys

    code = """
import os
import sys

# Ensure env var is set before import
assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be 1"

import huggingface_hub.constants as hf_constants
from local_anonymizer.recognizers import set_huggingface_offline_mode

# Before context manager: initial state
assert os.environ.get("HF_HUB_OFFLINE") == "1"
assert hf_constants.HF_HUB_OFFLINE is True

# Online fallback simulation
with set_huggingface_offline_mode(False):
    assert os.environ.get("HF_HUB_OFFLINE") is None, "env must be cleared"
    assert hf_constants.HF_HUB_OFFLINE is False, "constant must be False"

# After context manager: restored state
assert os.environ.get("HF_HUB_OFFLINE") == "1", "env must be restored to 1"
assert hf_constants.HF_HUB_OFFLINE is True, "constant must be restored to True"
print("SUBPROCESS_TEST_OK")
"""
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = "src"

    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Subprocess failed with stderr: {res.stderr}\nstdout: {res.stdout}"
    assert "SUBPROCESS_TEST_OK" in res.stdout


def test_eupii_deterministic_suppression_full_and_partial_spans():
    """
    Test that EUPiiRecognizer pre-computes full-text protected deterministic spans (AHV, UID,
    IBAN, Phone, Email, Address, URL, Date) and strictly rejects ANY overlapping ML candidate spans,
    whether the candidate covers the full span or only a sub-span.
    """
    import torch
    from local_anonymizer.recognizers import EUPiiRecognizer

    # Synthetic text containing various deterministic entities + valid PII (Person, Location, Health)
    text = (
        "Patientin Dr. Beatrix Meier wohnt an der Musterstrasse 12, 8001 Zürich. "
        "AHV: 756.9217.0769.85, UID: CHE-105.816.788, IBAN: CH9300762011623852957. "
        "Tel: +41 79 123 45 67, Mail: test.user@beispiel.ch, Web: https://example.ch/privacy am 15.03.2024. "
        "Diagnose: Diabetes mellitus Typ 2 in Bern."
    )

    rec = EUPiiRecognizer()

    # Pre-computed protected spans must cover all deterministic items
    prot_spans = rec._get_protected_deterministic_spans(text)
    assert len(prot_spans) >= 8

    # Create dummy tokenizer and model double returning candidates
    class DummyTokenizer:
        def __call__(self, t, **kwargs):
            class DummyBatch(dict):
                def to(self, device):
                    return self
            # Return dummy offset mapping covering the tokens we want to test
            offsets = [
                (0, 0),  # CLS
                (text.find("Beatrix Meier"), text.find("Beatrix Meier") + len("Beatrix Meier")),  # PERSON -> KEEP
                (text.find("8001 Zürich"), text.find("8001 Zürich") + len("8001 Zürich")),  # Sub-span of ADDRESS -> SUPPRESS
                (text.find("756.9217.0769.85"), text.find("756.9217.0769.85") + 16),  # Full AHV -> SUPPRESS
                (text.find("105.816.788"), text.find("105.816.788") + 11),  # Partial UID -> SUPPRESS
                (text.find("7620116238"), text.find("7620116238") + 10),  # Partial IBAN -> SUPPRESS
                (text.find("123 45 67"), text.find("123 45 67") + 9),  # Partial Phone -> SUPPRESS
                (text.find("beispiel.ch"), text.find("beispiel.ch") + 11),  # Partial Email -> SUPPRESS
                (text.find("https://example.ch"), text.find("https://example.ch") + 18),  # Partial URL -> SUPPRESS
                (text.find("15.03.2024"), text.find("15.03.2024") + 10),  # Date -> SUPPRESS
                (text.find("Diabetes mellitus Typ 2"), text.find("Diabetes mellitus Typ 2") + len("Diabetes mellitus Typ 2")),  # HEALTH -> KEEP
                (text.find("Bern"), text.find("Bern") + 4),  # LOCATION -> KEEP
                (0, 0),  # SEP
            ]
            batch = DummyBatch({
                "input_ids": torch.tensor([[1] * len(offsets)]),
                "attention_mask": torch.tensor([[1] * len(offsets)]),
                "offset_mapping": [offsets],
            })
            return batch

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {
                "id2label": {
                    0: "O",
                    1: "B-PERSON_NAME",
                    2: "B-LOCATION",
                    3: "B-DOCUMENT_IDENTIFIER",
                    4: "B-HEALTH_DATA",
                }
            })()

        def forward(self, **kwargs):
            # Class logits for each token
            num_tokens = 13
            # logits: (1, num_tokens, 5 classes)
            logits = torch.full((1, num_tokens, 5), -10.0)
            logits[0, 1, 1] = 10.0  # Beatrix Meier -> PERSON
            logits[0, 2, 2] = 10.0  # 8001 Zürich -> LOCATION (partial address)
            logits[0, 3, 3] = 10.0  # AHV -> ID_NUMBER
            logits[0, 4, 3] = 10.0  # Partial UID -> ID_NUMBER
            logits[0, 5, 3] = 10.0  # Partial IBAN -> ID_NUMBER
            logits[0, 6, 3] = 10.0  # Partial Phone -> ID_NUMBER
            logits[0, 7, 3] = 10.0  # Partial Email -> ID_NUMBER
            logits[0, 8, 3] = 10.0  # Partial URL -> ID_NUMBER
            logits[0, 9, 3] = 10.0  # Date -> ID_NUMBER
            logits[0, 10, 4] = 10.0  # Diabetes mellitus Typ 2 -> HEALTH_DATA
            logits[0, 11, 2] = 10.0  # Bern -> LOCATION

            return type("Outputs", (), {"logits": logits})()

    rec.tokenizer = DummyTokenizer()
    rec.model = DummyModel()
    rec.id2label = rec.model.config.id2label
    rec.device = "cpu"

    results = rec.analyze(text)

    # Verify: All overlapping deterministic spans were suppressed!
    detected_texts = [text[r.start:r.end] for r in results]
    assert "Beatrix Meier" in detected_texts
    assert "Diabetes mellitus Typ 2" in detected_texts
    assert "Bern" in detected_texts

    assert "8001 Zürich" not in detected_texts
    assert "756.9217.0769.85" not in detected_texts
    assert "105.816.788" not in detected_texts
    assert "7620116238" not in detected_texts
    assert "123 45 67" not in detected_texts
    assert "beispiel.ch" not in detected_texts
    assert "https://example.ch" not in detected_texts
    assert "15.03.2024" not in detected_texts

    # Check method metadata
    for r in results:
        assert r.recognition_metadata["detection_method"] == "ai"
        assert r.recognition_metadata["recognizer_name"] == "EUPiiRecognizer"


def test_eupii_cache_isolation():
    """Verify _EUPII_MODEL_CACHE retains loaded models and serves them without reload."""
    from local_anonymizer.recognizers import EUPiiRecognizer, _EUPII_MODEL_CACHE, get_optimal_device

    device = get_optimal_device()
    dummy_tok = "dummy_tok"
    dummy_model = type("DummyModel", (), {"config": type("Cfg", (), {"id2label": {1: "O"}})()})()
    _EUPII_MODEL_CACHE[f"dummy/test-model_{device}"] = (dummy_tok, dummy_model)

    rec = EUPiiRecognizer(model_name="dummy/test-model")
    rec.load()
    assert rec.tokenizer == "dummy_tok"
    assert rec.model is dummy_model


def test_eupii_subtoken_span_score_aggregation():
    """Verify that multi-subtoken tokens (e.g. 13 + 890 + 39) are aggregated as a whole span even if token 1 is low."""
    import torch
    from local_anonymizer.recognizers import EUPiiRecognizer

    rec = EUPiiRecognizer(threshold=0.50)
    text = "Versicherten-Nr. 1389039 abgeschlossen."

    class DummyBatch(dict):
        def pop(self, key, default=None):
            return super().pop(key, default)

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            # Token offsets:
            # 0: [0, 16] "Versicherten-Nr." -> O
            # 1: [16, 17] " " -> O
            # 2: [17, 19] "13" -> B-DOCUMENT_IDENTIFIER (score 0.45)
            # 3: [19, 22] "890" -> I-DOCUMENT_IDENTIFIER (score 0.95)
            # 4: [22, 24] "39" -> I-DOCUMENT_IDENTIFIER (score 0.95)
            # 5: [24, 25] " " -> O
            # 6: [25, 38] "abgeschlossen" -> O
            # 7: [38, 39] "." -> O
            offsets = [
                (0, 16),
                (16, 17),
                (17, 19),
                (19, 22),
                (22, 24),
                (24, 25),
                (25, 38),
                (38, 39),
            ]
            batch = DummyBatch({
                "input_ids": torch.tensor([[1] * len(offsets)]),
                "attention_mask": torch.tensor([[1] * len(offsets)]),
                "offset_mapping": [offsets],
            })
            return batch

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {
                "id2label": {
                    0: "O",
                    1: "B-DOCUMENT_IDENTIFIER",
                    2: "I-DOCUMENT_IDENTIFIER",
                }
            })()

        def forward(self, **kwargs):
            num_tokens = 8
            logits = torch.full((1, num_tokens, 3), -10.0)
            logits[0, 0, 0] = 10.0
            logits[0, 1, 0] = 10.0
            # Token 2: 13 with argmax B-DOCUMENT_IDENTIFIER (class 1) but score ~0.40 (< 0.50 threshold)
            logits[0, 2, 0] = -0.5
            logits[0, 2, 1] = 0.0
            logits[0, 2, 2] = -0.1
            # Token 3: 890 with score ~0.95
            logits[0, 3, 2] = 10.0
            # Token 4: 39 with score ~0.95
            logits[0, 4, 2] = 10.0
            logits[0, 5, 0] = 10.0
            logits[0, 6, 0] = 10.0
            logits[0, 7, 0] = 10.0
            return type("Outputs", (), {"logits": logits})()

    rec.tokenizer = DummyTokenizer()
    rec.model = DummyModel()
    rec.id2label = rec.model.config.id2label
    rec.device = "cpu"

    results = rec.analyze(text)
    assert len(results) == 1
    res = results[0]
    assert res.entity_type == "ID_NUMBER"
    assert text[res.start:res.end] == "1389039"
    assert res.score >= 0.50


def test_is_model_cached_offline_check(monkeypatch):
    """Verify is_model_cached correctly checks local cache without triggering downloads."""
    import transformers
    from local_anonymizer.recognizers import is_model_cached

    # Case 1: When local_files_only succeeds
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: "dummy_tok")
    monkeypatch.setattr(transformers.AutoModelForTokenClassification, "from_pretrained", lambda *args, **kwargs: "dummy_model")
    assert is_model_cached("dummy/model", "transformers") is True

    # Case 2: When local_files_only raises (cache miss)
    def fail_load(*args, **kwargs):
        raise OSError("Model not found in cache")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fail_load)
    assert is_model_cached("dummy/model", "transformers") is False


def test_eupii_load_lifecycle_cache_download_and_error(monkeypatch):
    """Verify EUPiiRecognizer.load() handles Cache-Hit, Online-Download, and Download-Error."""
    import pytest
    import transformers
    from local_anonymizer.recognizers import EUPiiRecognizer, _EUPII_MODEL_CACHE

    test_model_id = "test/eupii-lifecycle-model"
    for k in list(_EUPII_MODEL_CACHE.keys()):
        if test_model_id in k:
            _EUPII_MODEL_CACHE.pop(k, None)

    dummy_model = type("DummyModel", (), {
        "config": type("Cfg", (), {"id2label": {1: "O"}})(),
        "eval": lambda self: None,
    })()

    # 1. Test Cache-Hit (offline load succeeds directly)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: "tok_offline")
    monkeypatch.setattr(transformers.AutoModelForTokenClassification, "from_pretrained", lambda *args, **kwargs: dummy_model)
    rec1 = EUPiiRecognizer(model_name=test_model_id)
    rec1.load()
    assert rec1.tokenizer == "tok_offline"
    assert rec1.model is dummy_model

    for k in list(_EUPII_MODEL_CACHE.keys()):
        if test_model_id in k:
            _EUPII_MODEL_CACHE.pop(k, None)

    # 2. Test Online-Download Fallback (offline fails, online succeeds)
    call_counts = {"offline": 0, "online": 0}

    def mock_tok_from_pretrained(*args, **kwargs):
        if kwargs.get("local_files_only"):
            call_counts["offline"] += 1
            raise OSError("Not in cache")
        call_counts["online"] += 1
        return "tok_online"

    def mock_model_from_pretrained(*args, **kwargs):
        if kwargs.get("local_files_only"):
            raise OSError("Not in cache")
        return dummy_model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", mock_tok_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForTokenClassification, "from_pretrained", mock_model_from_pretrained)

    rec2 = EUPiiRecognizer(model_name=test_model_id)
    rec2.load()
    assert rec2.tokenizer == "tok_online"
    assert call_counts["offline"] >= 1
    assert call_counts["online"] >= 1

    for k in list(_EUPII_MODEL_CACHE.keys()):
        if test_model_id in k:
            _EUPII_MODEL_CACHE.pop(k, None)

    # 3. Test Download Failure (both offline and online fail)
    def mock_fail_all(*args, **kwargs):
        raise ConnectionError("No internet connection")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", mock_fail_all)
    rec3 = EUPiiRecognizer(model_name=test_model_id)
    with pytest.raises(RuntimeError) as exc_info:
        rec3.load()
    assert "1.07 GB" in str(exc_info.value)
    assert "konnte weder aus dem lokalen Cache noch online heruntergeladen werden" in str(exc_info.value)
