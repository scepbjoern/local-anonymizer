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
