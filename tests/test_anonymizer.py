import pytest
from local_anonymizer.anonymizer import LocalAnonymizer

def test_enabled_entities_empty_list():
    anonymizer = LocalAnonymizer(enabled_entities=[])
    res = anonymizer.anonymize("Max Mustermann arbeitet bei ZHAW.")
    # Nothing should be redacted
    assert "Max Mustermann" in res.anonymized_text
    assert "ZHAW" in res.anonymized_text
    assert len(res.entities) == 0

def test_enabled_entities_none_allows_all():
    anonymizer = LocalAnonymizer(enabled_entities=None)
    # Should redact both
    res = anonymizer.anonymize("Max Mustermann arbeitet bei ZHAW.")
    assert "Max Mustermann" not in res.anonymized_text
    assert "ZHAW" not in res.anonymized_text
    assert len(res.entities) > 0

def test_gliner_threshold_propagation():
    anonymizer = LocalAnonymizer(gliner_threshold=0.99)
    assert anonymizer.gliner_threshold == 0.99
    assert anonymizer.gliner_recognizer.threshold == 0.99

def test_unknown_entity_warning():
    with pytest.warns(UserWarning, match=r"Unknown entity type\(s\)"):
        LocalAnonymizer(enabled_entities=["PERSON", "UNKNOWN_TYPE"])

def test_de_anonymize_cascading_prevention():
    anonymizer = LocalAnonymizer()
    text = "Max Mustermann and [PERSON_1]"
    # Mapping where a replacement value contains a placeholder
    mapping = {"[PERSON_1]": "Anna [PERSON_2]", "[PERSON_2]": "Bob"}
    restored = anonymizer.de_anonymize(text, mapping)
    # Single-pass regex replaces [PERSON_1] with "Anna [PERSON_2]" and does not cascade to replace [PERSON_2] inside it
    assert restored == "Max Mustermann and Anna [PERSON_2]"

def test_ignore_terms_filtering():
    anonymizer = LocalAnonymizer(ignore_terms=["ZHAW", "Studierende"])
    res = anonymizer.anonymize("Max Mustermann ist bei der ZHAW. Die Studierende Anna.")
    # ZHAW and Studierende should not be redacted
    assert "ZHAW" in res.anonymized_text
    assert "Studierende" in res.anonymized_text
    assert "Max Mustermann" not in res.anonymized_text
