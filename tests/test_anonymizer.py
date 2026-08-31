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


def test_disabled_general_detection_keeps_glossary_terms(monkeypatch):
    """An empty general-detection selection must still apply explicit glossary entries."""
    anonymizer = LocalAnonymizer(glossary={"SAP": "IT_SYSTEM"}, enabled_entities=[])

    def unexpected_general_detection(**kwargs):
        pytest.fail("General recognizers must not run when enabled_entities is empty")

    monkeypatch.setattr(anonymizer.analyzer, "analyze", unexpected_general_detection)
    results = anonymizer.analyze("SAP")

    assert len(results) == 1
    assert results[0].entity_type == "IT_SYSTEM"
    assert results[0].start == 0
    assert results[0].end == 3


def test_glossary_can_be_disabled_independently_of_general_detection(monkeypatch):
    """The category master switch must suppress glossary hits even if general detection is on."""
    from presidio_analyzer import RecognizerResult

    anonymizer = LocalAnonymizer(
        glossary={"SAP": "IT_SYSTEM"},
        enabled_entities=["IT_SYSTEM"],
        enabled_glossary_entities=[],
    )

    def fake_general_detection(**kwargs):
        return [
            RecognizerResult(
                entity_type="IT_SYSTEM",
                start=0,
                end=3,
                score=1.0,
                recognition_metadata={"recognizer_name": "FuzzyGlossaryRecognizer"},
            )
        ]

    monkeypatch.setattr(anonymizer.analyzer, "analyze", fake_general_detection)
    assert anonymizer.analyze("SAP") == []


def test_role_glossary_is_supported_when_role_is_opted_in():
    anonymizer = LocalAnonymizer(
        glossary={"CEO": "ROLE"},
        enabled_entities=[],
        enabled_glossary_entities=["ROLE"],
    )

    results = anonymizer.analyze("Der CEO genehmigt den Antrag.")

    assert len(results) == 1
    assert results[0].entity_type == "ROLE"
    assert results[0].start == 4
    assert results[0].end == 7


def test_role_glossary_is_suppressed_when_role_is_off():
    anonymizer = LocalAnonymizer(
        glossary={"CEO": "ROLE"},
        enabled_entities=[],
        enabled_glossary_entities=[],
    )

    assert anonymizer.analyze("Der CEO genehmigt den Antrag.") == []


def test_detection_method_is_annotated_from_recognizer(monkeypatch):
    """Review metadata distinguishes a deterministic regex result from generic automation."""
    from presidio_analyzer import RecognizerResult

    anonymizer = LocalAnonymizer(enabled_entities=["ADDRESS"])

    def fake_general_detection(**kwargs):
        return [
            RecognizerResult(
                entity_type="ADDRESS",
                start=0,
                end=10,
                score=1.0,
                recognition_metadata={"recognizer_name": "AddressPatternRecognizer"},
            )
        ]

    monkeypatch.setattr(anonymizer.analyzer, "analyze", fake_general_detection)
    results = anonymizer.analyze("Hauptstrasse")

    assert results[0].recognition_metadata["detection_method"] == "regex"


def test_builtin_generic_terms_are_ignored_but_explicit_glossary_wins(monkeypatch):
    """Generic labels such as E-Mail and App are suppressed unless explicitly configured."""
    from presidio_analyzer import RecognizerResult

    anonymizer = LocalAnonymizer(enabled_entities=["EMAIL_ADDRESS", "IT_SYSTEM"])

    def fake_general_detection(**kwargs):
        return [
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=0, end=6, score=1.0),
            RecognizerResult(entity_type="IT_SYSTEM", start=7, end=10, score=1.0),
        ]

    monkeypatch.setattr(anonymizer.analyzer, "analyze", fake_general_detection)
    assert anonymizer.analyze("E-Mail App") == []

    glossary_anonymizer = LocalAnonymizer(
        glossary={"App": "IT_SYSTEM"},
        enabled_entities=[],
    )

    def unexpected_glossary_general_detection(**kwargs):
        pytest.fail("General recognizers must not run when enabled_entities is empty")

    monkeypatch.setattr(glossary_anonymizer.analyzer, "analyze", unexpected_glossary_general_detection)
    glossary_results = glossary_anonymizer.analyze("App")

    assert len(glossary_results) == 1
    assert glossary_results[0].entity_type == "IT_SYSTEM"


def test_user_ignore_overrides_explicit_glossary(monkeypatch):
    """A deliberate user ignore must remain authoritative over a glossary entry."""
    anonymizer = LocalAnonymizer(
        glossary={"Claims": "IT_SYSTEM"},
        ignore_terms=["Claims"],
        enabled_entities=[],
    )

    results = anonymizer.analyze("Claims")

    assert results == []


def test_low_score_detection_requires_review():
    anonymizer = LocalAnonymizer()

    from presidio_analyzer import RecognizerResult

    def fake_analyze(text, on_progress=None):
        return [RecognizerResult(entity_type="ORGANIZATION", start=0, end=5, score=0.66)]

    anonymizer.analyze = fake_analyze
    result = anonymizer.anonymize("Alpha")

    assert result.entities[0].needs_review is True
    assert result.entities[0] in result.review_needed

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
    cfg.entity_modes = {"IT_SYSTEM": "off", "PERSON": "all"}
    cfg.save()

    # Verify reload
    cfg2 = cfg_mod.AppConfig.load()
    assert cfg2.format_mode == "role_only"
    assert cfg2.export_format == "md"
    assert cfg2.entity_modes == {"IT_SYSTEM": "off", "PERSON": "all"}


def test_smart_linking_proposal_logic():
    from local_anonymizer.anonymizer import compute_smart_link_proposals

    class DummyGroup:
        def __init__(self, text: str, entity_type: str = "PERSON", parent: Optional[str] = None):
            self.original_text = text
            self.entity_type = entity_type
            self.parent_group_text = parent
            self.suggested_parent: Optional[str] = None
            self.suggested_tag: Optional[str] = None
            self.suggested_candidates: List[str] = []

    # Test single candidate proposal (German honorific, English honorific, genitive)
    g1 = DummyGroup("Julia Meier", "PERSON")
    g2 = DummyGroup("Frau Meier", "PERSON")
    g3 = DummyGroup("Julias", "PERSON")
    g4 = DummyGroup("John Smith", "PERSON")
    g5 = DummyGroup("Mr. Smith", "PERSON")

    groups = [g1, g2, g3, g4, g5]
    compute_smart_link_proposals(groups)

    # Verify no auto-commit: parent_group_text must still be None
    assert g2.parent_group_text is None
    assert g3.parent_group_text is None
    assert g5.parent_group_text is None

    # Verify proposal generation
    assert g2.suggested_parent == "Julia Meier"
    assert g2.suggested_tag == "ANREDE"

    assert g3.suggested_parent == "Julia Meier"
    assert g3.suggested_tag == "GENITIV"

    assert g5.suggested_parent == "John Smith"
    assert g5.suggested_tag == "ANREDE"

    # Test multiple candidates ambiguity (e.g. two Meiers)
    g6 = DummyGroup("Hans Meier", "PERSON")
    multi_groups = [g1, g6, g2]
    compute_smart_link_proposals(multi_groups)

    # When multiple Meiers exist, single auto-link must be suppressed in favor of candidate list
    assert g2.suggested_parent is None
    assert set(g2.suggested_candidates) == {"Julia Meier", "Hans Meier"}


def test_full_name_precedence_over_subword_glossary():
    from local_anonymizer.anonymizer import LocalAnonymizer

    text = (
        "Für den Herbst ist vorgesehen, dass Remo Weiersmüller übernimmt. "
        "Remo wird im Frühjahr wieder da sein."
    )
    # Even if single first name 'Remo' is in the glossary:
    anonymizer = LocalAnonymizer(glossary={"Remo": "PERSON"})
    result = anonymizer.anonymize(text, format_mode="numbered_role", roles={"remo weiersmüller": "LEHRPERSON", "remo": "LEHRPERSON"})

    # Full name must be replaced completely; 'Weiersmüller' must NOT remain unmasked!
    assert "Weiersmüller" not in result.anonymized_text
    assert "[PERSON_1_LEHRPERSON]" in result.anonymized_text or "[PERSON_2_LEHRPERSON]" in result.anonymized_text


def test_local_anonymizer_analyze_with_progress():
    from local_anonymizer.anonymizer import LocalAnonymizer

    anonymizer = LocalAnonymizer()
    progress_updates = []
    def on_progress(val, msg):
        progress_updates.append((val, msg))

    results = anonymizer.analyze("Julia Meier arbeitet an der ETH Zürich.", on_progress=on_progress)
    assert len(progress_updates) >= 4
    assert progress_updates[0][0] == 0.10
    assert progress_updates[-1][0] == 0.90
    assert any(r.entity_type == "PERSON" for r in results)


def test_trim_entity_span():
    from local_anonymizer.anonymizer import trim_entity_span

    text = "Dies ist <br>Dr. Andreas Schönenberger<br> und **Julia Meier**."
    # Case 1: Span includes <br> at start and end
    s1 = text.find("<br>")
    e1 = text.find("<br> und") + 4 # slice covering '<br>Dr. Andreas Schönenberger<br>'
    clean_s1, clean_e1 = trim_entity_span(text, s1, e1)
    assert text[clean_s1:clean_e1] == "Dr. Andreas Schönenberger"

    # Case 2: Span includes markdown bold ** at start and end
    s2 = text.find("**Julia")
    e2 = text.find("Meier**.") + 7 # slice covering '**Julia Meier**'
    clean_s2, clean_e2 = trim_entity_span(text, s2, e2)
    assert text[clean_s2:clean_e2] == "Julia Meier"


def test_gender_suffix_extension():
    from local_anonymizer.anonymizer import LocalAnonymizer
    anon = LocalAnonymizer()
    text = "Veranlasser:in und Leistungserbringer:in sowie Sachbearbeiter:innen und Kund*innen."
    results = anon.analyze(text)
    detected_texts = [text[r.start:r.end] for r in results]

    assert "Veranlasser:in" in detected_texts or any("Veranlasser" in t for t in detected_texts)
    # Ensure no isolated "in" or ":in" artifacts exist as standalone entities
    assert "in" not in detected_texts
    assert ":in" not in detected_texts


def test_spacy_recognizer_removed_from_registry():
    """SpacyRecognizer runs on a blank (NER-less) spaCy engine and would otherwise silently
    detect nothing while still advertising support for PERSON/ORGANIZATION/LOCATION/... --
    corrupting entity-source transparency and the enabled_entities validation warning."""
    anon = LocalAnonymizer()
    names = [r.name for r in anon.analyzer.registry.get_recognizers(language="de", all_fields=True)]
    assert "SpacyRecognizer" not in names


def test_glossary_short_form_absorbed_into_model_span():
    """A short glossary entry ("Remo") must not truncate a longer name the model already found
    ("Remo Weiersmueller") -- the glossary hit is absorbed into the model's span (boundaries from
    the model, type from the glossary), so the surname is never left leaking in the output."""
    anon = LocalAnonymizer(glossary={"Remo": "PERSON"})
    text = (
        "Remo Weiersmueller leitet das Projekt. Remo war schon frueher hier. "
        "Remo Weiersmueller ist zufrieden."
    )
    res = anon.anonymize(text)

    assert "Weiersmueller" not in res.anonymized_text
    # Both full-name occurrences resolve to the same placeholder
    assert res.anonymized_text.count("Remo Weiersmueller") == 0
    full_name_placeholder = next(
        (ph for ph, val in res.mapping.items() if val == "Remo Weiersmueller"), None
    )
    assert full_name_placeholder is not None
    assert res.anonymized_text.count(full_name_placeholder) == 2

    # The standalone short mention is still detected as its own entity, not silently dropped
    assert any(v == "Remo" for v in res.mapping.values())


def test_glossary_type_wins_on_identical_span():
    """When the glossary and the model agree on the exact same span but disagree on the type,
    the glossary's explicit type wins deterministically -- not just incidentally because its
    score happens to be higher."""
    anon = LocalAnonymizer(glossary={"Julia": "ORGANIZATION"}, enabled_entities=["PERSON", "ORGANIZATION"])
    res = anon.anonymize("Julia hat heute frei.")

    assert len(res.entities) == 1
    assert res.entities[0].entity_type == "ORGANIZATION"
    assert res.entities[0].original_text == "Julia"


def test_residue_warning_flags_unreplaced_adjacent_capitalized_token():
    """If a detected entity is immediately followed by an unreplaced, capitalized word, that is
    flagged for review (e.g. a short glossary match whose surname the model missed entirely, so
    there was no longer span for it to be absorbed into). A lowercase follower must not trigger
    the warning."""
    anon = LocalAnonymizer(glossary={"Remo": "PERSON"})

    from presidio_analyzer import RecognizerResult

    def fake_analyze(text, on_progress=None):
        start = text.index("Remo")
        return [RecognizerResult(entity_type="PERSON", start=start, end=start + 4, score=1.0)]

    anon.analyze = fake_analyze
    res = anon.anonymize("Remo Weiersmueller kam vorbei.")

    assert res.entities[0].needs_review is True
    assert res.entities[0].residue_note is not None
    assert "Weiersmueller" in res.entities[0].residue_note
    assert res.entities[0] in res.review_needed

    # Negative case: a lowercase follower must not trigger the residue warning
    def fake_analyze_lowercase(text, on_progress=None):
        start = text.index("Remo")
        return [RecognizerResult(entity_type="PERSON", start=start, end=start + 4, score=1.0)]

    anon2 = LocalAnonymizer(glossary={"Remo": "PERSON"})
    anon2.analyze = fake_analyze_lowercase
    res2 = anon2.anonymize("Remo kam vorbei.")
    assert res2.entities[0].needs_review is False
    assert res2.entities[0].residue_note is None


def test_entity_source_overview_reflects_actual_recognizers():
    """The transparency overview must reflect what actually detects each category: GLiNER
    prompts, glossary entries, regex patterns, or library-backed recognizers -- with no phantom
    categories left over from the removed SpacyRecognizer."""
    anon = LocalAnonymizer(
        glossary={"ZHAW": "ORGANIZATION"},
        enabled_entities=["PERSON", "ORGANIZATION", "IBAN_CODE"],
    )
    overview = anon.get_entity_source_overview()
    by_category = {row["category"]: row for row in overview}

    # Active flag reflects enabled_entities
    assert by_category["PERSON"]["active"] is True
    assert by_category["EMAIL_ADDRESS"]["active"] is False

    # ORGANIZATION is fed by both GLiNER prompts and the glossary
    org_kinds = {s["kind"] for s in by_category["ORGANIZATION"]["sources"]}
    assert org_kinds == {"prompt", "glossary"}
    glossary_source = next(s for s in by_category["ORGANIZATION"]["sources"] if s["kind"] == "glossary")
    assert glossary_source["entry_count"] == 1

    # IBAN_CODE is fed by both a GLiNER prompt and a real regex pattern
    iban_kinds = {s["kind"] for s in by_category["IBAN_CODE"]["sources"]}
    assert "regex" in iban_kinds
    regex_source = next(s for s in by_category["IBAN_CODE"]["sources"] if s["kind"] == "regex")
    assert regex_source["patterns"], "regex source must expose the actual pattern(s)"

    # PHONE_NUMBER is library-backed (phonenumbers), not an empty regex field
    phone_kinds = {s["kind"] for s in by_category["PHONE_NUMBER"]["sources"]}
    assert "library" in phone_kinds

    # No dead SpacyRecognizer contributing phantom sources
    for row in overview:
        assert all(s["recognizer"] != "SpacyRecognizer" for s in row["sources"])


def test_entity_source_overview_reflects_glossary_master_switch():
    anonymizer = LocalAnonymizer(
        glossary={"SAP": "IT_SYSTEM"},
        enabled_entities=[],
        enabled_glossary_entities=[],
    )
    overview = {row["category"]: row for row in anonymizer.get_entity_source_overview()}

    assert overview["IT_SYSTEM"]["active"] is False
    assert overview["IT_SYSTEM"]["mode"] == "off"


def test_overlap_priority_hierarchy_glossary_over_deterministic_over_ai(monkeypatch):
    """
    Verify the 3-tier overlap hierarchy:
    Tier 3 (Glossar) > Tier 2 (Deterministisch: AHV/UID/IBAN/etc.) > Tier 1 (Lokale KI: GLiNER/EU-PII)
    """
    from presidio_analyzer import RecognizerResult

    anon = LocalAnonymizer(
        glossary={"Spezialprojekt": "ORGANIZATION"},
        enabled_entities=["ORGANIZATION", "AHV_NUMBER", "ID_NUMBER"],
    )

    # 1. Deterministic (AHV) vs AI (GLiNER / EU-PII with higher score or longer subspan)
    def fake_analyze_det_vs_ai(text, **kwargs):
        return [
            RecognizerResult(
                entity_type="ID_NUMBER",
                start=0,
                end=16,
                score=0.99,
                recognition_metadata={"recognizer_name": "EUPiiRecognizer", "detection_method": "ai"},
            ),
            RecognizerResult(
                entity_type="AHV_NUMBER",
                start=0,
                end=16,
                score=0.80,
                recognition_metadata={"recognizer_name": "AHVNumberRecognizer", "detection_method": "regex"},
            ),
        ]

    monkeypatch.setattr(anon.analyzer, "analyze", fake_analyze_det_vs_ai)
    results = anon.analyze("756.9217.0769.85")
    assert len(results) == 1
    assert results[0].entity_type == "AHV_NUMBER"
    assert results[0].recognition_metadata["detection_method"] == "regex"

    # 2. Glossary vs Deterministic vs AI
    def fake_analyze_glossary_vs_det(text, **kwargs):
        return [
            RecognizerResult(
                entity_type="ORGANIZATION",
                start=0,
                end=14,
                score=1.0,
                recognition_metadata={"recognizer_name": "FuzzyGlossaryRecognizer", "detection_method": "glossary"},
            ),
            RecognizerResult(
                entity_type="ID_NUMBER",
                start=0,
                end=14,
                score=1.0,
                recognition_metadata={"recognizer_name": "UIDNumberRecognizer", "detection_method": "regex"},
            ),
        ]

    monkeypatch.setattr(anon.analyzer, "analyze", fake_analyze_glossary_vs_det)
    results2 = anon.analyze("Spezialprojekt")
    assert len(results2) == 1
    assert results2[0].entity_type == "ORGANIZATION"
    assert results2[0].recognition_metadata["recognizer_name"] == "FuzzyGlossaryRecognizer"


def test_eupii_transparency_overview_and_metadata():
    """Verify get_entity_source_overview includes EUPiiRecognizer when enabled."""
    anon = LocalAnonymizer(enable_eupii=True, eupii_threshold=0.52)
    overview = anon.get_entity_source_overview()
    by_category = {row["category"]: row for row in overview}

    # Categories supported by EUPii
    for cat in ["PERSON", "LOCATION", "ID_NUMBER", "HEALTH_DATA"]:
        assert cat in by_category
        sources = by_category[cat]["sources"]
        eupii_src = next((s for s in sources if s.get("recognizer") == "EUPiiRecognizer"), None)
        assert eupii_src is not None
        assert eupii_src["kind"] == "model"
        assert eupii_src["threshold"] == 0.52


def test_local_anonymizer_set_eupii_enabled_toggle():
    """Verify dynamic enable and disable of EUPiiRecognizer on LocalAnonymizer."""
    anon = LocalAnonymizer(enable_eupii=False)
    reg_names = [r.name for r in anon.analyzer.registry.recognizers]
    assert "EUPiiRecognizer" not in reg_names

    anon.set_eupii_enabled(True, threshold=0.60)
    reg_names = [r.name for r in anon.analyzer.registry.recognizers]
    assert "EUPiiRecognizer" in reg_names
    assert anon.eupii_recognizer.threshold == 0.60

    anon.set_eupii_enabled(False)
    reg_names = [r.name for r in anon.analyzer.registry.recognizers]
    assert "EUPiiRecognizer" not in reg_names
