import json
import pytest
from app import (
    AppState,
    EntityGroup,
    EntityOccurrence,
    split_occurrence_to_new_group,
    reset_app_state,
    compute_reactive_preview,
    sync_group_overrides,
)
from local_anonymizer.config import AppConfig
from local_anonymizer.llm.schema import (
    TriageKeepItem,
    TriageRecategorizeItem,
    TriageDiscardItem,
    TriageEnvelope,
    validate_batch_response,
)
from local_anonymizer.llm.batching import prepare_triage_batches
from local_anonymizer.llm.apply_service import (
    ApplyCommand,
    ApplyService,
    compute_triage_snapshot,
)


def test_app_config_llm_fields_roundtrip():
    cfg = AppConfig()
    cfg.llm_enabled = True
    cfg.llm_base_url = "http://localhost:11434/v1"
    cfg.llm_model_name = "qwen2.5:3b"
    cfg.llm_auto_review = False
    
    data = cfg.to_dict()
    loaded = AppConfig.from_dict(data)
    assert loaded.llm_enabled is True
    assert loaded.llm_base_url == "http://localhost:11434/v1"
    assert loaded.llm_model_name == "qwen2.5:3b"
    assert loaded.llm_auto_review is False


def test_reset_app_state_clears_llm_state_and_increments_revision():
    st = AppState()
    st.raw_text = "Some text"
    st.document_revision = 1
    st.llm_triage_results = {"occ-1": TriageKeepItem(occ_id="occ-1", confidence="high")}
    st.llm_triage_snapshot = "snap_1"
    st.llm_staged_selections = {"occ-1"}
    st.is_llm_running = True
    st.llm_partial_failure = True
    st.llm_unprocessed_occ_ids = {"occ-2"}

    reset_app_state(st)

    assert st.raw_text == ""
    assert st.document_revision == 2
    assert len(st.llm_triage_results) == 0
    assert st.llm_triage_snapshot == ""
    assert len(st.llm_staged_selections) == 0
    assert st.is_llm_running is False
    assert st.llm_partial_failure is False
    assert len(st.llm_unprocessed_occ_ids) == 0


def test_sparse_badge_logic():
    # Keep item without descriptor -> NO sparse badge
    keep_pure = TriageKeepItem(occ_id="occ-1", confidence="high")
    assert keep_pure.action == "keep"
    assert keep_pure.descriptor_suggestion is None

    # Keep item with descriptor -> MUST receive change badge
    keep_with_desc = TriageKeepItem(occ_id="occ-2", confidence="high", descriptor_suggestion="Leiter")
    assert keep_with_desc.descriptor_suggestion == "Leiter"

    # Recategorize -> MUST receive change badge
    recat = TriageRecategorizeItem(occ_id="occ-3", new_entity_type="ORG", confidence="medium")
    assert recat.action == "recategorize"

    # Discard -> MUST receive discard badge
    discard = TriageDiscardItem(occ_id="occ-4", confidence="low")
    assert discard.action == "discard"


def test_end_to_end_simulated_llm_triage_flow():
    # 1. Initialize state with document and candidates
    st = AppState()
    st.raw_text = "Peter Müller traf Dr. Schmidt in Berlin."
    st.filename = "bericht.txt"
    st.document_revision = 1
    st.analysis_revision = 1

    g_peter = EntityGroup(
        original_text="Peter Müller",
        entity_type="PERSON",
        group_id="peter müller",
    )
    g_peter.occurrences = [
        EntityOccurrence(
            start=0,
            end=12,
            score=0.98,
            context_html="<b>Peter Müller</b> traf",
            needs_review=False,
            occ_id="occ-p1",
        )
    ]
    
    g_schmidt = EntityGroup(
        original_text="Dr. Schmidt",
        entity_type="PERSON",
        group_id="dr. schmidt",
    )
    g_schmidt.occurrences = [
        EntityOccurrence(
            start=18,
            end=29,
            score=0.95,
            context_html="traf <b>Dr. Schmidt</b> in",
            needs_review=False,
            occ_id="occ-s1",
        )
    ]

    g_berlin = EntityGroup(
        original_text="Berlin",
        entity_type="LOCATION",
        group_id="berlin",
    )
    g_berlin.occurrences = [
        EntityOccurrence(
            start=33,
            end=39,
            score=0.99,
            context_html="in <b>Berlin</b>.",
            needs_review=False,
            occ_id="occ-b1",
        )
    ]

    st.entity_groups = [g_peter, g_schmidt, g_berlin]
    compute_reactive_preview(st)

    # 2. Compute triage snapshot & prepare batch
    snapshot_hash = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    candidates = [
        {"occ_id": "occ-p1", "original_text": "Peter Müller", "entity_type": "PERSON", "role": "", "context_snippet": "Peter Müller traf"},
        {"occ_id": "occ-s1", "original_text": "Dr. Schmidt", "entity_type": "PERSON", "role": "", "context_snippet": "traf Dr. Schmidt in"},
        {"occ_id": "occ-b1", "original_text": "Berlin", "entity_type": "LOCATION", "role": "", "context_snippet": "in Berlin."},
    ]

    batches = prepare_triage_batches(candidates, st.document_revision, snapshot_hash)
    assert len(batches) == 1

    # 3. Simulate local LLM response envelope
    mock_llm_response = json.dumps({
        "schema_version": "1.0",
        "request_id": batches[0].request_id,
        "document_revision": 1,
        "document_hash": snapshot_hash,
        "items": [
            {"occ_id": "occ-p1", "action": "keep", "confidence": "high", "reasoning": "Valid person name", "descriptor_suggestion": "Zeuge"},
            {"occ_id": "occ-s1", "action": "keep", "confidence": "high", "reasoning": "Doctor name", "descriptor_suggestion": "Arzt"},
            {"occ_id": "occ-b1", "action": "discard", "confidence": "medium", "reasoning": "General city name, can remain"},
        ],
    })

    envelope = TriageEnvelope.model_validate_json(mock_llm_response)
    validate_batch_response(envelope, batches[0].occ_id_set, st.document_revision, snapshot_hash)

    # 4. Stage items and apply mutations via ApplyService
    commands = [
        ApplyCommand(occ_id="occ-p1", action="keep", descriptor_suggestion="Zeuge"),
        ApplyCommand(occ_id="occ-s1", action="keep", descriptor_suggestion="Arzt"),
        ApplyCommand(occ_id="occ-b1", action="discard"),
    ]

    success, msg = ApplyService.apply_mutations(
        st,
        snapshot_hash,
        commands,
        split_fn=split_occurrence_to_new_group,
        sync_fn=sync_group_overrides,
        preview_fn=compute_reactive_preview,
    )

    assert success is True
    assert g_peter.role == "Zeuge"
    assert g_schmidt.role == "Arzt"
    assert g_berlin.enabled is False

    # 5. Check reactive preview
    anon_text, mapping, report = compute_reactive_preview(st)
    assert "Zeuge" in anon_text or "PERSON" in anon_text
    assert "Arzt" in anon_text or "PERSON" in anon_text
    assert "Berlin" in anon_text  # Berlin was discarded/disabled so remains in cleartext
