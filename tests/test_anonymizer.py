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


def test_format_modes_numbering_and_roles():
    anonymizer = LocalAnonymizer()
    text = "Julia Meier arbeitet an der ETH Zürich."
    roles = {"julia meier": "STUDENT", "eth zürich": "HOCHSCHULE"}

    # Mode 1: Numbered only
    res1 = anonymizer.anonymize(text, format_mode="numbered", roles=roles)
    assert "[PERSON_1]" in res1.anonymized_text
    assert "[ORGANIZATION_1]" in res1.anonymized_text

    # Mode 2: Numbered + Role
    res2 = anonymizer.anonymize(text, format_mode="numbered_role", roles=roles)
    assert "[PERSON_1_STUDENT]" in res2.anonymized_text
    assert "[ORGANIZATION_1_HOCHSCHULE]" in res2.anonymized_text

    # Mode 3: Role only (unique roles)
    res3 = anonymizer.anonymize(text, format_mode="role_only", roles=roles)
    assert "[PERSON_STUDENT]" in res3.anonymized_text
    assert "[ORGANIZATION_HOCHSCHULE]" in res3.anonymized_text


def test_mode_3_collision_detection_and_fallback():
    anonymizer = LocalAnonymizer()
    text = "Julia Meier und Erika Musterfrau studieren gemeinsam."
    roles = {"julia meier": "STUDENT", "erika musterfrau": "STUDENT"}

    # Both persons have the same role "STUDENT" -> collision!
    with pytest.warns(UserWarning, match=r"Rollenkollision bei Entitätstyp 'PERSON' mit Rolle 'STUDENT'"):
        res = anonymizer.anonymize(text, format_mode="role_only", roles=roles)

    # Should fall back to numbered mode for the colliding role
    assert "[PERSON_1_STUDENT]" in res.anonymized_text
    assert "[PERSON_2_STUDENT]" in res.anonymized_text
    assert "[PERSON_STUDENT]" not in res.anonymized_text

    # Reversibility test
    restored = LocalAnonymizer.de_anonymize(res.anonymized_text, res.mapping)
    assert restored == text


def test_entity_linking_reversibility():
    anonymizer = LocalAnonymizer()
    text = "Das ist Julia Meier. Das ist Julia. Das ist Frau Meier."
    roles = {"julia meier": "STUDENT"}
    entity_links = {
        "julia meier": ("", "VOLLNAME"),
        "julia": ("julia meier", "VORNAME"),
        "frau meier": ("julia meier", "ANREDE"),
    }

    res = anonymizer.anonymize(
        text,
        format_mode="numbered_role",
        roles=roles,
        entity_links=entity_links,
    )

    assert "[PERSON_1_STUDENT_VOLLNAME]" in res.anonymized_text
    assert "[PERSON_1_STUDENT_VORNAME]" in res.anonymized_text
    assert "[PERSON_1_STUDENT_ANREDE]" in res.anonymized_text

    # Verify all map back exactly to their surface forms
    restored = LocalAnonymizer.de_anonymize(res.anonymized_text, res.mapping)
    assert restored == text


def test_entity_linking_homonym_separation():
    anonymizer = LocalAnonymizer()
    text = "Das ist Julia Meier und das ist Julia Suter. Das ist Julia."

    # Two separate entities named Julia Meier and Julia Suter, with Julia kept standalone (unlinked)
    res = anonymizer.anonymize(text, format_mode="numbered")

    # Verify 3 distinct placeholders are created
    assert "[PERSON_1]" in res.anonymized_text
    assert "[PERSON_2]" in res.anonymized_text
    assert "[PERSON_3]" in res.anonymized_text

    restored = LocalAnonymizer.de_anonymize(res.anonymized_text, res.mapping)
    assert restored == text
