import pytest
from local_anonymizer.llm.apply_service import (
    ApplyCommand,
    compute_triage_snapshot,
    ApplyService,
)
from app import (
    AppState,
    EntityGroup,
    EntityOccurrence,
    split_occurrence_to_new_group,
    sync_group_overrides,
    compute_reactive_preview,
)


def create_test_state() -> AppState:
    st = AppState()
    st.raw_text = "Dr. Meier arbeitet bei der Siemens AG in Berlin. Frau Meier ist auch dort."
    st.filename = "test.txt"
    st.document_revision = 1
    st.analysis_revision = 1

    # Group 1: 'Meier' (2 occurrences)
    g_meier = EntityGroup(
        original_text="Meier",
        entity_type="PERSON",
        group_id="meier",
    )
    g_meier.occurrences = [
        EntityOccurrence(
            start=4,
            end=9,
            score=0.95,
            context_html="Dr. <b>Meier</b> arbeitet",
            needs_review=False,
            occ_id="occ-meier-1",
        ),
        EntityOccurrence(
            start=55,
            end=60,
            score=0.90,
            context_html="Frau <b>Meier</b> ist",
            needs_review=False,
            occ_id="occ-meier-2",
        ),
    ]

    # Group 2: 'Siemens AG' (1 occurrence)
    g_siemens = EntityGroup(
        original_text="Siemens AG",
        entity_type="ORG",
        group_id="siemens ag",
    )
    g_siemens.occurrences = [
        EntityOccurrence(
            start=27,
            end=37,
            score=0.99,
            context_html="bei der <b>Siemens AG</b> in",
            needs_review=False,
            occ_id="occ-siemens-1",
        ),
    ]

    st.entity_groups = [g_meier, g_siemens]
    compute_reactive_preview(st)
    return st


def test_compute_triage_snapshot_deterministic():
    st = create_test_state()
    snap1 = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    snap2 = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    assert snap1 == snap2
    assert len(snap1) == 64  # SHA-256 hex string

    # Text change changes snapshot
    snap3 = compute_triage_snapshot("Modified text", st.analysis_revision, st.entity_groups)
    assert snap3 != snap1

    # Analysis revision change changes snapshot
    snap4 = compute_triage_snapshot(st.raw_text, st.analysis_revision + 1, st.entity_groups)
    assert snap4 != snap1


def test_prevalidate_and_preview_impact_success():
    st = create_test_state()
    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)

    commands = [
        ApplyCommand(
            occ_id="occ-meier-1",
            action="keep",
            descriptor_suggestion="Arzt",
        ),
        ApplyCommand(
            occ_id="occ-meier-2",
            action="recategorize",
            new_entity_type="PERSON",
            descriptor_suggestion="Patientin",
        ),
        ApplyCommand(
            occ_id="occ-siemens-1",
            action="discard",
        ),
    ]

    is_valid, err, impacts = ApplyService.prevalidate_and_preview_impact(st, snap, commands)
    assert is_valid is True
    assert err == ""
    assert len(impacts) == 3

    # occ-meier-1 is in a 2-item group, so it will split
    assert impacts[0]["occ_id"] == "occ-meier-1"
    assert impacts[0]["will_split"] is True
    assert impacts[0]["new_role"] == "Arzt"

    # occ-siemens-1 is in a 1-item group, so will_split is False
    assert impacts[2]["occ_id"] == "occ-siemens-1"
    assert impacts[2]["will_split"] is False
    assert impacts[2]["action"] == "discard"


def test_prevalidate_stale_snapshot_rejected():
    st = create_test_state()
    commands = [ApplyCommand(occ_id="occ-meier-1", action="keep")]
    is_valid, err, impacts = ApplyService.prevalidate_and_preview_impact(st, "stale_snapshot_hash", commands)
    assert is_valid is False
    assert "veraltet" in err


def test_apply_mutations_atomic_execution():
    st = create_test_state()
    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)

    commands = [
        ApplyCommand(
            occ_id="occ-meier-1",
            action="keep",
            descriptor_suggestion="Arzt",
        ),
        ApplyCommand(
            occ_id="occ-meier-2",
            action="keep",
            descriptor_suggestion="Patientin",
        ),
        ApplyCommand(
            occ_id="occ-siemens-1",
            action="discard",
        ),
    ]

    success, msg = ApplyService.apply_mutations(
        st,
        snap,
        commands,
        split_fn=split_occurrence_to_new_group,
        sync_fn=sync_group_overrides,
        preview_fn=compute_reactive_preview,
    )

    assert success is True
    assert "3 Änderungen erfolgreich übernommen" in msg

    # Verify state after mutation
    # occ-meier-1 was split into its own group with role 'Arzt'
    g_m1, occ1 = ApplyService._find_occurrence_and_group(st.entity_groups, "occ-meier-1")
    assert g_m1 is not None
    assert g_m1.role == "Arzt"

    # occ-meier-2 was split into its own group with role 'Patientin'
    g_m2, occ2 = ApplyService._find_occurrence_and_group(st.entity_groups, "occ-meier-2")
    assert g_m2 is not None
    assert g_m2.role == "Patientin"
    assert g_m1.group_id != g_m2.group_id  # isolated into separate groups!

    # occ-siemens-1 was discarded (enabled=False)
    g_s, occ_s = ApplyService._find_occurrence_and_group(st.entity_groups, "occ-siemens-1")
    assert g_s is not None
    assert g_s.enabled is False
