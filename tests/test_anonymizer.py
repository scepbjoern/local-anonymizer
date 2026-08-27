import pytest
from typing import Optional
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


def test_genitive_name_recognition_and_anonymization():
    anonymizer = LocalAnonymizer()
    text = "Das ist Julia Meier. Das ist Julias Beitrag und Meiers Vorschlag."
    res = anonymizer.anonymize(text, format_mode="numbered")

    # "Julias" and "Meiers" should be masked
    assert "Julias" not in res.anonymized_text
    assert "Meiers" not in res.anonymized_text
    assert "[PERSON_1]" in res.anonymized_text

    # Reversibility
    restored = LocalAnonymizer.de_anonymize(res.anonymized_text, res.mapping)
    assert restored == text


def test_genitive_apostrophe_variation():
    anonymizer = LocalAnonymizer()
    text = "Das ist Julia Meier. Hier ist Julia's Projekt."
    res = anonymizer.anonymize(text, format_mode="numbered")

    assert "Julia's" not in res.anonymized_text
    restored = LocalAnonymizer.de_anonymize(res.anonymized_text, res.mapping)
    assert restored == text


def test_genitive_false_positive_prevention():
    anonymizer = LocalAnonymizer(enabled_entities=["PERSON"])
    # "Markus" is a distinct name from "Mark", and "Markt" is a common noun.
    # "Studiums" contains genitive 's' of "Studium" (an ignore term).
    text = "Markus geht zum Markt. Das ist Mark. Marks Tasche liegt beim Markt. Das Ziel des Studiums ist klar."
    res = anonymizer.anonymize(text, format_mode="numbered")

    # "Markt" and "Studiums" must NOT be recognized or replaced as a PERSON
    assert "Markt" in res.anonymized_text
    assert "Studiums" in res.anonymized_text
    # "Marks" should be masked because "Mark" is recognized as a person
    assert "Marks" not in res.anonymized_text
    # "Markus" and "Mark" are both persons, but "Markt" and "Studiums" are not
    person_texts = [e.original_text for e in res.entities]
    assert "Markt" not in person_texts
    assert "Studiums" not in person_texts

    restored = LocalAnonymizer.de_anonymize(res.anonymized_text, res.mapping)
    assert restored == text


def test_build_entity_tree_structure():
    from local_anonymizer.anonymizer import build_entity_tree

    class DummyGroup:
        def __init__(self, text: str, parent: Optional[str] = None):
            self.original_text = text
            self.parent_group_text = parent

        @property
        def key(self) -> str:
            return self.original_text.lower()

    items = [
        DummyGroup("Julia Meier"),
        DummyGroup("Julia", parent="Julia Meier"),
        DummyGroup("Frau Meier", parent="Julia Meier"),
        DummyGroup("Julia Suter"),
        DummyGroup("Remo"),
    ]

    tree = build_entity_tree(items)
    # Should have 3 root nodes: Julia Meier, Julia Suter, Remo
    assert len(tree) == 3
    assert tree[0].item.original_text == "Julia Meier"
    assert len(tree[0].children) == 2
    assert [c.item.original_text for c in tree[0].children] == ["Julia", "Frau Meier"]

    assert tree[1].item.original_text == "Julia Suter"
    assert len(tree[1].children) == 0

    assert tree[2].item.original_text == "Remo"
    assert len(tree[2].children) == 0


def test_config_persistence(tmp_path, monkeypatch):
    import local_anonymizer.config as cfg_mod

    # Patch config directory to tmp_path
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg_mod, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg_mod, "LOG_FILE", tmp_path / "app.log")

    # Load defaults
    cfg = cfg_mod.AppConfig.load()
    assert cfg.format_mode == "numbered_role"

    # Modify and save
    cfg.format_mode = "role_only"
    cfg.export_format = "md"
    cfg.save()

    # Verify reload
    cfg2 = cfg_mod.AppConfig.load()
    assert cfg2.format_mode == "role_only"
    assert cfg2.export_format == "md"

