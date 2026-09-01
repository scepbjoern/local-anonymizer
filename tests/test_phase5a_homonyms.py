"""
Unit and regression tests for Phase 5a: Homonym and Occurrence Disambiguation Handling.
Directly tests productive lifecycle functions:
- rebind_overrides_after_analysis
- reset_app_state
- load_document_into_state
- format_link_dropdown_label
- split_occurrence_to_new_group
- revert_occurrence_to_base
- group_tree_nodes_by_homonym
- compute_smart_link_proposals
- compute_reactive_preview
- LocalAnonymizer.de_anonymize
"""

import uuid
from dataclasses import dataclass
from typing import List, Optional
import pytest

from app import (
    AppState,
    EntityGroup,
    EntityOccurrence,
    OccurrenceOverride,
    compute_context_fingerprint,
    compute_reactive_preview,
    split_occurrence_to_new_group,
    revert_occurrence_to_base,
    sync_group_overrides,
    rebind_overrides_after_analysis,
    reset_app_state,
    load_document_into_state,
    format_link_dropdown_label,
    group_tree_nodes_by_homonym,
    is_category_enabled,
    ENTITY_MODE_ALL,
    ENTITY_MODE_OFF,
)
from local_anonymizer.anonymizer import LocalAnonymizer, build_entity_tree, compute_smart_link_proposals


@dataclass
class MockRecognizerResult:
    start: int
    end: int
    entity_type: str
    score: float
    analysis_explanation: Optional[str] = None
    recognition_metadata: Optional[dict] = None


def test_occurrence_fingerprint_deterministic():
    """Verify context fingerprint is deterministic and unaffected by remote text changes outside the window."""
    prefix = "A" * 60 + "Frau "
    suffix = " ist Projektleiterin bei der Firma." + "B" * 60
    text1 = prefix + "Müller" + suffix
    start1 = text1.index("Müller")
    end1 = start1 + len("Müller")
    fp1 = compute_context_fingerprint(text1, start1, end1)

    text2 = "Sehr langer Text am Anfang des Dokuments... " + text1
    start2 = text2.index("Müller")
    end2 = start2 + len("Müller")
    fp2 = compute_context_fingerprint(text2, start2, end2)

    assert fp1 == fp2


def test_exact_occurrence_text_split_and_case_sensitive_restore():
    """
    Auflage 1: Split and restore with exact occurrence text evidence.
    Text contains 'Müller' and uppercase 'MÜLLER'.
    Splitting 'MÜLLER' must keep 'MÜLLER' as original_text and expected_original_text.
    De-anonymization must restore exact byte-for-byte casing.
    """
    st = AppState()
    st.raw_text = "Frau Müller sprach mit Herrn MÜLLER über das Projekt."
    st.format_mode = "numbered_role"

    idx1 = st.raw_text.find("Müller")
    idx2 = st.raw_text.find("MÜLLER")

    occ1 = EntityOccurrence(
        start=idx1,
        end=idx1 + len("Müller"),
        score=0.95,
        context_html="",
        needs_review=False,
        occ_id="occ_1",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx1, idx1 + len("Müller")),
    )
    occ2 = EntityOccurrence(
        start=idx2,
        end=idx2 + len("MÜLLER"),
        score=0.92,
        context_html="",
        needs_review=False,
        occ_id="occ_2",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx2, idx2 + len("MÜLLER")),
    )

    base_grp = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller")
    base_grp.role = "Chef"
    base_grp.occurrences = [occ1, occ2]
    st.entity_groups = [base_grp]

    # Split occurrence 2 (MÜLLER)
    split_grp = split_occurrence_to_new_group(st, base_grp, occ2)
    split_grp.role = "Praktikant"
    sync_group_overrides(st, split_grp)

    # Verify exact casing in split group and override
    assert split_grp.original_text == "MÜLLER"
    assert split_grp.text_key == "müller"
    assert st.occurrence_overrides["occ_2"].expected_original_text == "MÜLLER"
    assert st.occurrence_overrides["occ_2"].role == "Praktikant"

    # Compute preview
    anon_text, mapping, report = compute_reactive_preview(st)

    assert "[PERSON_1_CHEF]" in mapping
    assert "[PERSON_2_PRAKTIKANT]" in mapping
    assert mapping["[PERSON_1_CHEF]"] == "Müller"
    assert mapping["[PERSON_2_PRAKTIKANT]"] == "MÜLLER"

    # Verify byte-exact de-anonymization
    restored = LocalAnonymizer.de_anonymize(anon_text, mapping)
    assert restored == st.raw_text

    # Re-analysis must rebind MÜLLER because actual_orig == "MÜLLER" == expected_original_text
    results = [
        MockRecognizerResult(start=idx1, end=idx1 + len("Müller"), entity_type="PERSON", score=0.95),
        MockRecognizerResult(start=idx2, end=idx2 + len("MÜLLER"), entity_type="PERSON", score=0.92),
    ]
    rebound_groups, rebound_overrides = rebind_overrides_after_analysis(st.raw_text, results, st.occurrence_overrides)
    assert len(rebound_overrides) == 1
    new_occ2_id = list(rebound_overrides.keys())[0]
    assert rebound_overrides[new_occ2_id].expected_original_text == "MÜLLER"
    rebound_split = next((g for g in rebound_groups if g.group_id == split_grp.group_id), None)
    assert rebound_split is not None
    assert rebound_split.original_text == "MÜLLER"


def test_smart_linking_with_duplicate_named_parents_exact_id():
    """
    Auflage 2: Smart linking suggestions convey exact group_id even if multiple parents
    have identical names.
    """
    p1 = EntityGroup(original_text="Julia Meier", entity_type="PERSON", group_id="p1_id")
    p2 = EntityGroup(original_text="Julia Meier", entity_type="PERSON", group_id="p2_id")
    child = EntityGroup(original_text="Frau Meier", entity_type="PERSON", group_id="child_id")

    items = [p1, p2, child]
    compute_smart_link_proposals(items)

    # When multiple candidates match, candidate list must hold the exact candidate IDs
    assert child.suggested_parent is None
    assert set(child.suggested_candidates) == {"p1_id", "p2_id"}

    # When exactly one parent exists:
    single_items = [p1, child]
    compute_smart_link_proposals(single_items)
    assert child.suggested_parent == "p1_id"
    assert child.suggested_parent_text == "Julia Meier"


def test_format_link_dropdown_label_unambiguous_between_splits():
    """
    Restauflage 3: Link dropdown labels must be unambiguous even between two split groups
    with identical text, entity_type, role, and hit count.
    """
    base_grp = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller")
    base_grp.role = "Chef"
    base_grp.occurrences = [EntityOccurrence(start=0, end=6, score=0.9, context_html="", needs_review=False)]

    split1 = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="split_a1b2c3d4")
    split1.role = "Chef"
    split1.occurrences = [EntityOccurrence(start=10, end=16, score=0.9, context_html="", needs_review=False)]

    split2 = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="split_e5f6g7h8")
    split2.role = "Chef"
    split2.occurrences = [EntityOccurrence(start=20, end=26, score=0.9, context_html="", needs_review=False)]

    label_base = format_link_dropdown_label(base_grp)
    label_s1 = format_link_dropdown_label(split1)
    label_s2 = format_link_dropdown_label(split2)

    assert label_base == "Müller (PERSON, Chef, Basis, 1x)"
    assert label_s1 == "Müller (PERSON, Chef, Ausgliederung · a1b2c3, 1x)"
    assert label_s2 == "Müller (PERSON, Chef, Ausgliederung · e5f6g7, 1x)"

    # All three labels must be strictly distinct
    assert len({label_base, label_s1, label_s2}) == 3


def test_visual_homonym_clustering_independent_roots_and_sorting():
    """
    Auflage 3: Visual homonym clustering groups independent tree roots by text_key
    without modifying parent_group_id.
    """
    g1 = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller_base")
    g1.occurrences = [EntityOccurrence(start=10, end=16, score=0.9, context_html="", needs_review=False)]

    g2 = EntityGroup(original_text="Schmidt", entity_type="PERSON", group_id="schmidt_base")
    g2.occurrences = [
        EntityOccurrence(start=0, end=7, score=0.9, context_html="", needs_review=False),
        EntityOccurrence(start=100, end=107, score=0.9, context_html="", needs_review=False),
    ]

    g3 = EntityGroup(original_text="MÜLLER", entity_type="PERSON", group_id="split_muller_2")
    g3.occurrences = [EntityOccurrence(start=50, end=56, score=0.9, context_html="", needs_review=False)]

    # Unsorted list
    tree = build_entity_tree([g1, g2, g3])
    clusters = group_tree_nodes_by_homonym(tree)

    assert len(clusters) == 2
    m_cluster = next((c for c in clusters if c.text_key == "müller"), None)
    s_cluster = next((c for c in clusters if c.text_key == "schmidt"), None)

    assert m_cluster is not None
    assert len(m_cluster.nodes) == 2
    assert {n.item.group_id for n in m_cluster.nodes} == {"müller_base", "split_muller_2"}

    assert s_cluster is not None
    assert len(s_cluster.nodes) == 1

    # Invariance: Semantic parent links must strictly remain None!
    assert g1.parent_group_id is None
    assert g3.parent_group_id is None


def test_rebind_overrides_1_to_1_success():
    """
    Auflage 4: Productive rebind_overrides_after_analysis handles 1:1 match successfully.
    """
    raw_text = "Herr Weber besuchte Herrn Weber."
    idx1 = raw_text.find("Weber")
    idx2 = raw_text.find("Weber", idx1 + 1)
    fp1 = compute_context_fingerprint(raw_text, idx1, idx1 + len("Weber"))
    fp2 = compute_context_fingerprint(raw_text, idx2, idx2 + len("Weber"))

    old_overrides = {
        "old_occ_2": OccurrenceOverride(
            target_group_id="split_weber_special",
            context_fingerprint=fp2,
            expected_original_text="Weber",
            entity_type="PERSON",
            role="Vorstand",
            enabled=True,
        )
    }

    results = [
        MockRecognizerResult(start=idx1, end=idx1 + len("Weber"), entity_type="PERSON", score=0.9),
        MockRecognizerResult(start=idx2, end=idx2 + len("Weber"), entity_type="PERSON", score=0.9),
    ]

    rebound_groups, new_overrides = rebind_overrides_after_analysis(raw_text, results, old_overrides)

    assert len(new_overrides) == 1
    new_occ_id = list(new_overrides.keys())[0]
    assert new_occ_id != "old_occ_2"  # Re-keyed to new occ_id
    assert new_overrides[new_occ_id].target_group_id == "split_weber_special"
    assert new_overrides[new_occ_id].role == "Vorstand"

    assert len(rebound_groups) == 2
    split_g = next((g for g in rebound_groups if g.group_id == "split_weber_special"), None)
    assert split_g is not None
    assert split_g.role == "Vorstand"
    assert len(split_g.occurrences) == 1


def test_rebind_overrides_invalid_empty_target_id():
    """
    Restauflage 2: Overrides with empty or whitespace target_group_id are fail-safely discarded.
    """
    raw_text = "Herr Weber besuchte Herrn Weber."
    idx1 = raw_text.find("Weber")
    fp1 = compute_context_fingerprint(raw_text, idx1, idx1 + len("Weber"))

    old_overrides = {
        "old_1": OccurrenceOverride(
            target_group_id="   ",
            context_fingerprint=fp1,
            expected_original_text="Weber",
            entity_type="PERSON",
        )
    }

    results = [MockRecognizerResult(start=idx1, end=idx1 + len("Weber"), entity_type="PERSON", score=0.9)]
    groups, new_overrides = rebind_overrides_after_analysis(raw_text, results, old_overrides)

    assert len(new_overrides) == 0
    assert len(groups) == 1
    assert groups[0].group_id == "weber"


def test_rebind_overrides_base_group_id_collision():
    """
    Restauflage 2: Overrides whose target_group_id collides with a canonical base group id
    are fail-safely discarded.
    """
    raw_text = "Herr Weber besuchte Herrn Weber."
    idx1 = raw_text.find("Weber")
    fp1 = compute_context_fingerprint(raw_text, idx1, idx1 + len("Weber"))

    old_overrides = {
        "old_1": OccurrenceOverride(
            target_group_id="weber",  # Collides with canonical base group key
            context_fingerprint=fp1,
            expected_original_text="Weber",
            entity_type="PERSON",
            role="Special",
        )
    }

    results = [MockRecognizerResult(start=idx1, end=idx1 + len("Weber"), entity_type="PERSON", score=0.9)]
    groups, new_overrides = rebind_overrides_after_analysis(raw_text, results, old_overrides)

    assert len(new_overrides) == 0
    assert len(groups) == 1
    assert groups[0].group_id == "weber"
    assert groups[0].role == ""


def test_rebind_overrides_conflicting_target_group_id_dropped():
    """
    Restauflage 2: If multiple overrides claim the same target_group_id with conflicting
    text, entity_type, or role, all are fail-safely dropped.
    """
    raw_text = "Alpha und Beta trafen sich."
    idx1 = raw_text.find("Alpha")
    idx2 = raw_text.find("Beta")
    fp1 = compute_context_fingerprint(raw_text, idx1, idx1 + len("Alpha"))
    fp2 = compute_context_fingerprint(raw_text, idx2, idx2 + len("Beta"))

    old_overrides = {
        "old_1": OccurrenceOverride(
            target_group_id="split_common",
            context_fingerprint=fp1,
            expected_original_text="Alpha",
            entity_type="PERSON",
            role="RoleA",
        ),
        "old_2": OccurrenceOverride(
            target_group_id="split_common",
            context_fingerprint=fp2,
            expected_original_text="Beta",
            entity_type="PERSON",
            role="RoleB",
        ),
    }

    results = [
        MockRecognizerResult(start=idx1, end=idx1 + len("Alpha"), entity_type="PERSON", score=0.9),
        MockRecognizerResult(start=idx2, end=idx2 + len("Beta"), entity_type="PERSON", score=0.9),
    ]

    groups, new_overrides = rebind_overrides_after_analysis(raw_text, results, old_overrides)

    assert len(new_overrides) == 0
    assert len(groups) == 2
    assert {g.group_id for g in groups} == {"alpha", "beta"}


def test_rebind_overrides_ambiguous_multiple_old_dropped():
    """
    Auflage 4: When multiple old overrides have the same fingerprint, all are fail-safe discarded.
    """
    raw_text = "Test Text"
    fp = "same_fp"
    old_overrides = {
        "old_1": OccurrenceOverride(target_group_id="s1", context_fingerprint=fp, expected_original_text="Test"),
        "old_2": OccurrenceOverride(target_group_id="s2", context_fingerprint=fp, expected_original_text="Test"),
    }
    results = [MockRecognizerResult(start=0, end=4, entity_type="PERSON", score=0.9)]

    groups, new_overrides = rebind_overrides_after_analysis(raw_text, results, old_overrides)
    assert len(new_overrides) == 0
    assert len(groups) == 1
    assert groups[0].group_id == "test"


def test_rebind_overrides_ambiguous_multiple_new_dropped():
    """
    Auflage 4: When multiple new occurrences share the same fingerprint, overrides are fail-safe discarded.
    """
    pad_pre = "X" * 60 + " "
    pad_post = " " + "Y" * 60
    seg = pad_pre + "Kunde" + pad_post
    raw_text = seg + " --- " + seg

    idx1 = raw_text.find("Kunde")
    idx2 = raw_text.find("Kunde", idx1 + 1)
    fp = compute_context_fingerprint(raw_text, idx1, idx1 + len("Kunde"))

    old_overrides = {
        "old_1": OccurrenceOverride(target_group_id="s1", context_fingerprint=fp, expected_original_text="Kunde", role="VIP")
    }

    results = [
        MockRecognizerResult(start=idx1, end=idx1 + len("Kunde"), entity_type="PERSON", score=0.9),
        MockRecognizerResult(start=idx2, end=idx2 + len("Kunde"), entity_type="PERSON", score=0.9),
    ]

    groups, new_overrides = rebind_overrides_after_analysis(raw_text, results, old_overrides)
    assert len(new_overrides) == 0
    assert len(groups) == 1  # Kept in single base group


def test_rebind_overrides_text_mismatch_dropped():
    """
    Auflage 4: If context fingerprint matches but raw text changed, override is fail-safe discarded.
    """
    raw_text = "Firma Schmidt liefert."
    fp = compute_context_fingerprint(raw_text, 6, 13)

    old_overrides = {
        "old_1": OccurrenceOverride(target_group_id="s1", context_fingerprint=fp, expected_original_text="Meier", role="Chef")
    }

    results = [MockRecognizerResult(start=6, end=13, entity_type="ORGANIZATION", score=0.9)]

    groups, new_overrides = rebind_overrides_after_analysis(raw_text, results, old_overrides)
    assert len(new_overrides) == 0


def test_rebind_overrides_optional_fallback():
    """
    Auflage 4: Rebinding with None values for entity_type/role/enabled retains base recognizer values.
    """
    raw_text = "Kontakt mit Dr. Frank."
    start = raw_text.find("Dr. Frank")
    end = start + len("Dr. Frank")
    fp = compute_context_fingerprint(raw_text, start, end)

    old_overrides = {
        "old_1": OccurrenceOverride(
            target_group_id="split_frank",
            context_fingerprint=fp,
            expected_original_text="Dr. Frank",
            entity_type=None,
            role=None,
            enabled=None,
        )
    }

    results = [MockRecognizerResult(start=start, end=end, entity_type="PERSON", score=0.9)]

    groups, new_overrides = rebind_overrides_after_analysis(raw_text, results, old_overrides)
    assert len(new_overrides) == 1
    assert len(groups) == 1
    assert groups[0].group_id == "split_frank"
    assert groups[0].entity_type == "PERSON"
    assert groups[0].role == ""
    assert groups[0].enabled is True


def test_reset_app_state_and_unified_loading_lifecycle():
    """
    Restauflage 1 & Testklarstellung 2:
    - Productive load_document_into_state clears previous state and preserves raw_bytes.
    - reset_app_state completely empties all state including raw_bytes.
    """
    st = AppState()
    fake_docx_bytes = b"PKfake_docx_stream"

    # Step 1: Load document with raw_bytes
    load_document_into_state(st, text="Extrahierter Inhalt", filename="dokument.docx", raw_bytes=fake_docx_bytes)

    assert st.filename == "dokument.docx"
    assert st.raw_text == "Extrahierter Inhalt"
    assert st.last_raw_bytes == fake_docx_bytes
    assert st.entity_groups == []
    assert st.occurrence_overrides == {}

    # Step 2: Simulate adding analysis results and overrides
    st.entity_groups = [EntityGroup(original_text="A", entity_type="PERSON")]
    st.occurrence_overrides = {"occ_1": OccurrenceOverride("s1", "fp", "A")}
    st.current_mapping = {"[PERSON_1]": "A"}
    st.current_anon_text = "[PERSON_1]"

    # Step 3: Load new document -> state is reset but new raw_bytes are preserved
    fake_pdf_bytes = b"%PDF-1.4-fake"
    load_document_into_state(st, text="Neuer PDF Text", filename="neu.pdf", raw_bytes=fake_pdf_bytes)

    assert st.filename == "neu.pdf"
    assert st.raw_text == "Neuer PDF Text"
    assert st.last_raw_bytes == fake_pdf_bytes
    assert st.entity_groups == []
    assert st.occurrence_overrides == {}
    assert st.current_mapping == {}
    assert st.current_anon_text == ""

    # Step 4: Workspace reset clears everything
    reset_app_state(st)

    assert st.filename == ""
    assert st.raw_text == ""
    assert st.last_raw_bytes is None
    assert st.entity_groups == []
    assert st.occurrence_overrides == {}
    assert st.current_mapping == {}
    assert st.current_anon_text == ""


def test_category_modes_with_split_groups():
    """
    Testklarstellung 1: Category mode toggle (e.g. ENTITY_MODE_OFF vs ENTITY_MODE_ALL)
    correctly governs inclusion of base and split groups.
    """
    st = AppState()
    st.raw_text = "Firma A traf Person B."

    # Set explicit category modes in state.entity_modes
    st.entity_modes["ORGANIZATION"] = ENTITY_MODE_OFF
    st.entity_modes["PERSON"] = ENTITY_MODE_ALL

    assert is_category_enabled(st, "ORGANIZATION") is False
    assert is_category_enabled(st, "PERSON") is True

    g_org = EntityGroup(original_text="Firma A", entity_type="ORGANIZATION", group_id="split_org_1")
    g_org.occurrences = [EntityOccurrence(start=0, end=7, score=0.9, context_html="", needs_review=False)]

    g_per = EntityGroup(original_text="Person B", entity_type="PERSON", group_id="person_b")
    g_per.occurrences = [EntityOccurrence(start=13, end=21, score=0.9, context_html="", needs_review=False)]

    st.entity_groups = [g_org, g_per]

    anon_text, mapping, report = compute_reactive_preview(st)

    # PERSON is active, ORGANIZATION is OFF
    assert "[PERSON_1]" in mapping
    assert "[ORGANIZATION" not in anon_text
    assert "Firma A" in anon_text
    assert "[PERSON_1]" in anon_text


def test_revert_occurrence_to_base():
    """
    Reverting a split occurrence returns it to the base group and removes the override.
    """
    st = AppState()
    base_grp = EntityGroup(original_text="Schmidt", entity_type="PERSON", group_id="schmidt")
    occ1 = EntityOccurrence(start=0, end=7, score=0.9, context_html="", needs_review=False, occ_id="occ_1", context_fingerprint="fp1")
    occ2 = EntityOccurrence(start=20, end=27, score=0.9, context_html="", needs_review=False, occ_id="occ_2", context_fingerprint="fp2")
    base_grp.occurrences = [occ1, occ2]
    st.entity_groups = [base_grp]

    split_grp = split_occurrence_to_new_group(st, base_grp, occ2)
    assert len(st.entity_groups) == 2
    assert "occ_2" in st.occurrence_overrides

    revert_occurrence_to_base(st, split_grp, occ2)

    assert "occ_2" not in st.occurrence_overrides
    assert len(st.entity_groups) == 1
    assert st.entity_groups[0].group_id == "schmidt"
    assert len(st.entity_groups[0].occurrences) == 2


def test_placeholder_modes_with_homonyms_and_collision():
    """
    Verify all 3 placeholder modes with homonym groups, including collision fallback in Mode 3 (role_only).
    """
    st = AppState()
    st.raw_text = "A und B trafen C."

    g1 = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller_1")
    g1.role = "Chef"
    g1.occurrences = [EntityOccurrence(start=0, end=1, score=0.9, context_html="", needs_review=False)]

    g2 = EntityGroup(original_text="Müller", entity_type="PERSON", group_id="müller_2")
    g2.role = "Chef"
    g2.occurrences = [EntityOccurrence(start=6, end=7, score=0.9, context_html="", needs_review=False)]

    st.entity_groups = [g1, g2]

    # Mode 1: Numbered
    st.format_mode = "numbered"
    _, map1, _ = compute_reactive_preview(st)
    assert "[PERSON_1]" in map1
    assert "[PERSON_2]" in map1

    # Mode 2: Numbered Role
    st.format_mode = "numbered_role"
    _, map2, _ = compute_reactive_preview(st)
    assert "[PERSON_1_CHEF]" in map2
    assert "[PERSON_2_CHEF]" in map2

    # Mode 3: Role Only -> Colliding role falls back to numbered_role
    st.format_mode = "role_only"
    _, map3, _ = compute_reactive_preview(st)
    assert "[PERSON_1_CHEF]" in map3
    assert "[PERSON_2_CHEF]" in map3


def test_rebind_overrides_preserves_base_group_role_and_split_role():
    """
    Verify that upon re-analysis, user-assigned roles on base groups (e.g. 'Vater')
    and split groups (e.g. 'Sohn') are both preserved.
    """
    st = AppState()
    st.raw_text = "Dr. Thomas Müller forscht. Sein Sohn Thomas Müller hilft. Thomas Müller genehmigt."
    st.format_mode = "numbered_role"

    idx1 = st.raw_text.find("Thomas Müller")
    idx2 = st.raw_text.find("Thomas Müller", idx1 + 1)
    idx3 = st.raw_text.find("Thomas Müller", idx2 + 1)

    occ1 = EntityOccurrence(
        start=idx1,
        end=idx1 + len("Thomas Müller"),
        score=0.95,
        context_html="",
        needs_review=False,
        occ_id="occ_1",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx1, idx1 + len("Thomas Müller")),
    )
    occ2 = EntityOccurrence(
        start=idx2,
        end=idx2 + len("Thomas Müller"),
        score=0.95,
        context_html="",
        needs_review=False,
        occ_id="occ_2",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx2, idx2 + len("Thomas Müller")),
    )
    occ3 = EntityOccurrence(
        start=idx3,
        end=idx3 + len("Thomas Müller"),
        score=0.95,
        context_html="",
        needs_review=False,
        occ_id="occ_3",
        context_fingerprint=compute_context_fingerprint(st.raw_text, idx3, idx3 + len("Thomas Müller")),
    )

    base_grp = EntityGroup(original_text="Thomas Müller", entity_type="PERSON", group_id="thomas müller")
    base_grp.role = "Vater"
    base_grp.occurrences = [occ1, occ2, occ3]
    st.entity_groups = [base_grp]

    # Split occurrence 2 (Sohn)
    split_grp = split_occurrence_to_new_group(st, base_grp, occ2)
    split_grp.role = "Sohn"
    sync_group_overrides(st, split_grp)

    # Pre-check: 2 groups exist (Vater with 2 occs, Sohn with 1 occ)
    assert base_grp.role == "Vater"
    assert split_grp.role == "Sohn"

    # Simulate re-analysis
    results = [
        MockRecognizerResult(start=idx1, end=idx1 + len("Thomas Müller"), entity_type="PERSON", score=0.95),
        MockRecognizerResult(start=idx2, end=idx2 + len("Thomas Müller"), entity_type="PERSON", score=0.95),
        MockRecognizerResult(start=idx3, end=idx3 + len("Thomas Müller"), entity_type="PERSON", score=0.95),
    ]

    rebound_groups, rebound_overrides = rebind_overrides_after_analysis(
        st.raw_text,
        results,
        st.occurrence_overrides,
        existing_groups=st.entity_groups,
    )

    st.entity_groups = rebound_groups
    st.occurrence_overrides = rebound_overrides

    # Verify both roles are preserved after re-analysis
    base_after = next((g for g in st.entity_groups if g.group_id == "thomas müller"), None)
    split_after = next((g for g in st.entity_groups if g.group_id == split_grp.group_id), None)

    assert base_after is not None
    assert base_after.role == "Vater"
    assert len(base_after.occurrences) == 2

    assert split_after is not None
    assert split_after.role == "Sohn"
    assert len(split_after.occurrences) == 1

    anon_text, mapping, report = compute_reactive_preview(st)
    assert "[PERSON_1_VATER]" in mapping
    assert "[PERSON_2_SOHN]" in mapping
    assert mapping["[PERSON_1_VATER]"] == "Thomas Müller"
    assert mapping["[PERSON_2_SOHN]"] == "Thomas Müller"
