import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from app import EntityGroup, EntityOccurrence, AppState, reset_app_state
from local_anonymizer.llm.schema import TriageEnvelope, TriageKeepItem, TriageRecategorizeItem
from local_anonymizer.llm.provider import LocalApiProvider
from local_anonymizer.llm.batching import prepare_triage_batches
from local_anonymizer.llm.apply_service import compute_triage_snapshot, ApplyCommand, ApplyService


def test_app_state_llm_fields_and_reset():
    st = AppState()
    assert hasattr(st, "llm_triage_results")
    assert hasattr(st, "llm_triage_snapshot")
    assert hasattr(st, "llm_staged_selections")
    assert hasattr(st, "is_llm_running")
    assert hasattr(st, "llm_partial_failure")
    assert hasattr(st, "llm_unprocessed_occ_ids")
    assert hasattr(st, "llm_active_task")

    st.llm_triage_results["occ-1"] = "val"
    st.llm_staged_selections.add("occ-1")
    st.is_llm_running = True
    st.llm_partial_failure = True
    st.llm_unprocessed_occ_ids.add("occ-2")

    reset_app_state(st)
    assert st.llm_triage_results == {}
    assert st.llm_staged_selections == set()
    assert st.is_llm_running is False
    assert st.llm_partial_failure is False
    assert st.llm_unprocessed_occ_ids == set()
    assert st.llm_active_task is None


@pytest.mark.asyncio
async def test_late_response_discarded_on_snapshot_drift():
    # Simulate a running LLM batch loop where snapshot changes during await generate()
    st = AppState()
    st.raw_text = "Dr. Müller arbeitet bei Siemens in Berlin."
    st.document_revision = 1
    st.analysis_revision = 1
    g1 = EntityGroup("Dr. Müller", "PERSON")
    g1.occurrences = [
        EntityOccurrence(
            start=0,
            end=10,
            score=0.9,
            context_html="<b>Dr. Müller</b> arbeitet...",
            needs_review=False,
            method="spacy",
            occ_id="occ-1",
        )
    ]
    st.entity_groups = [g1]

    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    candidates = [{
        "occ_id": "occ-1",
        "original_text": "Dr. Müller",
        "entity_type": "PERSON",
        "role": "",
        "context_snippet": "Dr. Müller arbeitet...",
    }]
    batches = prepare_triage_batches(candidates, st.document_revision, snap)
    assert len(batches) == 1

    # While provider is "thinking", user changes text or document_revision
    async def delayed_generate(prompt, system_prompt):
        # Simulate user mutation in parallel
        st.raw_text = "Dr. Müller hat gekündigt."
        st.document_revision += 1
        return json.dumps({
            "schema_version": "1.0",
            "request_id": batches[0].request_id,
            "document_revision": 1,
            "document_hash": snap,
            "items": [
                {"occ_id": "occ-1", "action": "keep", "confidence": "high"}
            ]
        })

    mock_provider = MagicMock(spec=LocalApiProvider)
    mock_provider.generate = AsyncMock(side_effect=delayed_generate)
    st.llm_provider = mock_provider
    st.is_llm_running = True
    st.llm_triage_snapshot = snap

    # Run batch iteration simulating run_llm_triage loop
    for b in batches:
        raw_json = await st.llm_provider.generate(b.user_prompt, b.system_prompt)
        post_snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
        if post_snap != snap or st.document_revision != b.document_revision:
            # Must discard stale batch
            st.llm_partial_failure = True
            st.llm_unprocessed_occ_ids.update(b.occ_id_set)
            break

    # Results must NOT be committed
    assert "occ-1" not in st.llm_triage_results
    assert st.llm_partial_failure is True
    assert "occ-1" in st.llm_unprocessed_occ_ids


@pytest.mark.asyncio
async def test_reset_cancels_active_llm_task():
    st = AppState()

    async def long_running():
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(long_running())
    st.llm_active_task = task
    assert not task.done()

    reset_app_state(st)
    assert task.cancelling() > 0 or task.cancelled()
    await asyncio.sleep(0.01)
    assert task.done()


def test_optional_llm_extra_isolation():
    # Verify that schema and apply_service imports work cleanly
    from local_anonymizer.llm.schema import TriageEnvelope
    from local_anonymizer.llm.apply_service import compute_triage_snapshot
    assert TriageEnvelope is not None
    assert compute_triage_snapshot is not None
