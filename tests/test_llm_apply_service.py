import copy
import pytest
from app import EntityGroup, EntityOccurrence
from local_anonymizer.llm.apply_service import (
    ApplyCommand,
    ApplyService,
    compute_triage_snapshot,
    CANONICAL_APP_ENTITY_TYPES,
    normalize_entity_type,
    check_mutation_allowed,
)


class MockState:
    def __init__(self, raw_text: str, groups: list):
        self.raw_text = raw_text
        self.entity_groups = groups
        self.occurrence_overrides = {}
        self.current_mapping = {"Dr. Müller": "[PERSON_1]"}
        self.current_anon_text = "[PERSON_1] ist da."
        self.current_report = {"summary": "ok"}
        self.document_revision = 1
        self.analysis_revision = 1
        self.preview_revision = 1
        self.llm_triage_results = {"occ-1": "res1", "occ-2": "res2"}
        self.llm_triage_snapshot = ""
        self.llm_staged_selections = {"occ-1", "occ-2"}
        self.is_llm_running = False


def test_normalize_entity_type():
    assert normalize_entity_type("ORGANIZATION") == "ORGANIZATION"
    assert normalize_entity_type("org") == "ORGANIZATION"
    assert normalize_entity_type("PER") == "PERSON"
    assert normalize_entity_type("person") == "PERSON"
    assert normalize_entity_type("LOC") == "LOCATION"
    assert normalize_entity_type("GPE") == "LOCATION"
    assert normalize_entity_type("DATE") == "DATE_TIME"
    assert normalize_entity_type("TIME") == "DATE_TIME"
    assert normalize_entity_type("UNKNOWN_XYZ") is None
    assert normalize_entity_type("MISC") is None


def test_check_mutation_allowed():
    st = MockState("text", [])
    assert check_mutation_allowed(st) is True
    st.is_llm_running = True
    assert check_mutation_allowed(st) is False


def test_compute_triage_snapshot_deterministic():
    g1 = EntityGroup("Dr. Müller", "PERSON")
    g1.occurrences = [
        EntityOccurrence(start=0, end=10, score=0.9, context_html="<b>Dr. Müller</b>", needs_review=False, method="spacy", occ_id="occ-1"),
    ]
    s1 = compute_triage_snapshot("Dr. Müller war hier.", 1, [g1])
    s2 = compute_triage_snapshot("Dr. Müller war hier.", 1, [g1])
    assert s1 == s2
    assert len(s1) == 64  # sha256 hex


def test_prevalidate_and_preview_impact_valid_and_canonical_types():
    g1 = EntityGroup("Dr. Müller", "PERSON")
    g1.occurrences = [
        EntityOccurrence(start=0, end=10, score=0.9, context_html="<b>Dr. Müller</b>", needs_review=False, method="spacy", occ_id="occ-1"),
        EntityOccurrence(start=20, end=30, score=0.9, context_html="<b>Dr. Müller</b>", needs_review=False, method="spacy", occ_id="occ-2"),
    ]
    st = MockState("Dr. Müller und Dr. Müller.", [g1])
    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    st.llm_triage_snapshot = snap

    cmds = [
        ApplyCommand(occ_id="occ-1", action="recategorize", new_entity_type="ORG", descriptor_suggestion="Firma"),
        ApplyCommand(occ_id="occ-2", action="discard"),
    ]

    is_valid, err, impacts = ApplyService.prevalidate_and_preview_impact(st, snap, cmds)
    assert is_valid is True
    assert err == ""
    assert len(impacts) == 2
    assert impacts[0]["will_split"] is True
    assert impacts[0]["new_type"] == "ORGANIZATION"
    assert impacts[0]["new_role"] == "Firma"
    assert impacts[1]["action"] == "discard"


def test_prevalidate_rejects_invalid_entity_type():
    g1 = EntityGroup("Dr. Müller", "PERSON")
    g1.occurrences = [
        EntityOccurrence(start=0, end=10, score=0.9, context_html="<b>Dr. Müller</b>", needs_review=False, method="spacy", occ_id="occ-1"),
    ]
    st = MockState("Dr. Müller.", [g1])
    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    st.llm_triage_snapshot = snap

    cmds = [
        ApplyCommand(occ_id="occ-1", action="recategorize", new_entity_type="INVALID_TYPE_STRING"),
    ]

    is_valid, err, _ = ApplyService.prevalidate_and_preview_impact(st, snap, cmds)
    assert is_valid is False
    assert "Ungültiger Entitätstyp" in err


def test_apply_mutations_atomic_rollback_on_split_failure():
    g1 = EntityGroup("Dr. Müller", "PERSON")
    g1.occurrences = [
        EntityOccurrence(start=0, end=10, score=0.9, context_html="<b>Dr. Müller</b>", needs_review=False, method="spacy", occ_id="occ-1"),
        EntityOccurrence(start=20, end=30, score=0.9, context_html="<b>Dr. Müller</b>", needs_review=False, method="spacy", occ_id="occ-2"),
    ]
    st = MockState("Dr. Müller und Dr. Müller.", [g1])
    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    st.llm_triage_snapshot = snap

    initial_groups = copy.deepcopy(st.entity_groups)
    initial_overrides = copy.deepcopy(st.occurrence_overrides)
    initial_mapping = copy.deepcopy(st.current_mapping)

    def failing_split(state, grp, occ):
        raise RuntimeError("Simulated failure during occurrence split")

    def mock_sync(state, grp):
        pass

    def mock_preview(state):
        pass

    cmds = [
        ApplyCommand(occ_id="occ-1", action="recategorize", new_entity_type="ORG"),
    ]

    success, msg = ApplyService.apply_mutations(
        st,
        snap,
        cmds,
        split_fn=failing_split,
        sync_fn=mock_sync,
        preview_fn=mock_preview,
    )

    assert success is False
    assert "Simulated failure during occurrence split" in msg
    # Verify state rolled back completely
    assert len(st.entity_groups) == len(initial_groups)
    assert st.entity_groups[0].count == initial_groups[0].count
    assert st.occurrence_overrides == initial_overrides
    assert st.current_mapping == initial_mapping
    assert st.llm_staged_selections == {"occ-1", "occ-2"}


def test_apply_mutations_atomic_rollback_on_preview_failure():
    g1 = EntityGroup("Dr. Müller", "PERSON")
    g1.occurrences = [
        EntityOccurrence(start=0, end=10, score=0.9, context_html="<b>Dr. Müller</b>", needs_review=False, method="spacy", occ_id="occ-1"),
    ]
    st = MockState("Dr. Müller.", [g1])
    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    st.llm_triage_snapshot = snap

    initial_groups = copy.deepcopy(st.entity_groups)

    def mock_split(state, grp, occ):
        pass

    def mock_sync(state, grp):
        pass

    def failing_preview(state):
        raise ValueError("Simulated preview calculation crash")

    cmds = [
        ApplyCommand(occ_id="occ-1", action="recategorize", new_entity_type="ORG"),
    ]

    success, msg = ApplyService.apply_mutations(
        st,
        snap,
        cmds,
        split_fn=mock_split,
        sync_fn=mock_sync,
        preview_fn=failing_preview,
    )

    assert success is False
    assert "Simulated preview calculation crash" in msg
    # Entity type must be rolled back to PERSON
    assert st.entity_groups[0].entity_type == "PERSON"
    assert st.entity_groups[0].occurrences[0].occ_id == "occ-1"


def test_apply_mutations_successful_execution():
    g1 = EntityGroup("Dr. Müller", "PERSON")
    g1.occurrences = [
        EntityOccurrence(start=0, end=10, score=0.9, context_html="<b>Dr. Müller</b>", needs_review=False, method="spacy", occ_id="occ-1"),
    ]
    st = MockState("Dr. Müller.", [g1])
    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    st.llm_triage_snapshot = snap

    def mock_split(state, grp, occ):
        pass

    def mock_sync(state, grp):
        pass

    def mock_preview(state):
        state.preview_revision += 1

    cmds = [
        ApplyCommand(occ_id="occ-1", action="recategorize", new_entity_type="ORG", descriptor_suggestion="Firma"),
    ]

    success, msg = ApplyService.apply_mutations(
        st,
        snap,
        cmds,
        split_fn=mock_split,
        sync_fn=mock_sync,
        preview_fn=mock_preview,
    )

    assert success is True
    assert "1 Änderungen erfolgreich übernommen" in msg
    assert st.entity_groups[0].entity_type == "ORGANIZATION"
    assert st.entity_groups[0].role == "Firma"
    assert st.preview_revision == 2
    # occ-1 was removed from staged selections and triage results
    assert "occ-1" not in st.llm_staged_selections
    assert "occ-1" not in st.llm_triage_results
