import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from app import EntityGroup, EntityOccurrence, AppState, reset_app_state, reset_app_state_async
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
async def test_late_response_discarded_on_snapshot_drift_multi_batch():
    # 3 batches: batch 1 succeeds, batch 2 experiences drift -> batch 2 and 3 marked unprocessed
    st = AppState()
    st.raw_text = "Dr. Müller und Dr. Schmidt und Dr. Weber."
    st.document_revision = 1
    st.analysis_revision = 1
    g1 = EntityGroup("Dr. Müller", "PERSON")
    g1.occurrences = [
        EntityOccurrence(start=0, end=10, score=0.9, context_html="Dr. Müller", needs_review=False, method="spacy", occ_id="occ-1"),
        EntityOccurrence(start=15, end=26, score=0.9, context_html="Dr. Schmidt", needs_review=False, method="spacy", occ_id="occ-2"),
        EntityOccurrence(start=31, end=40, score=0.9, context_html="Dr. Weber", needs_review=False, method="spacy", occ_id="occ-3"),
    ]
    st.entity_groups = [g1]

    snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
    candidates = [
        {"occ_id": "occ-1", "original_text": "Dr. Müller", "entity_type": "PERSON", "role": "", "context_snippet": "Dr. Müller"},
        {"occ_id": "occ-2", "original_text": "Dr. Schmidt", "entity_type": "PERSON", "role": "", "context_snippet": "Dr. Schmidt"},
        {"occ_id": "occ-3", "original_text": "Dr. Weber", "entity_type": "PERSON", "role": "", "context_snippet": "Dr. Weber"},
    ]
    batches = prepare_triage_batches(candidates, st.document_revision, snap, max_items_per_batch=1)
    assert len(batches) == 3

    async def generate_mock(prompt, system_prompt):
        if "occ-2" in prompt:
            # Simulate drift during batch 2
            st.raw_text = "Dr. Müller und jemand anderes."
            st.document_revision += 1
            return json.dumps({
                "schema_version": "1.0",
                "request_id": batches[1].request_id,
                "document_revision": 1,
                "document_hash": snap,
                "items": [{"occ_id": "occ-2", "action": "keep", "confidence": "high"}]
            })
        elif "occ-1" in prompt:
            return json.dumps({
                "schema_version": "1.0",
                "request_id": batches[0].request_id,
                "document_revision": 1,
                "document_hash": snap,
                "items": [{"occ_id": "occ-1", "action": "keep", "confidence": "high"}]
            })
        else:
            return json.dumps({
                "schema_version": "1.0",
                "request_id": batches[2].request_id,
                "document_revision": 1,
                "document_hash": snap,
                "items": [{"occ_id": "occ-3", "action": "keep", "confidence": "high"}]
            })

    mock_provider = MagicMock(spec=LocalApiProvider)
    mock_provider.generate = AsyncMock(side_effect=generate_mock)
    st.llm_provider = mock_provider
    st.is_llm_running = True
    st.llm_triage_snapshot = snap

    # Execute batch loop matching production logic in launch_llm_triage
    for batch_idx, b in enumerate(batches):
        pre_snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
        if pre_snap != snap or st.document_revision != b.document_revision:
            st.llm_partial_failure = True
            for rem in batches[batch_idx:]:
                st.llm_unprocessed_occ_ids.update(rem.occ_id_set)
            break

        raw_json = await st.llm_provider.generate(b.user_prompt, b.system_prompt)

        post_snap = compute_triage_snapshot(st.raw_text, st.analysis_revision, st.entity_groups)
        if post_snap != snap or st.document_revision != b.document_revision:
            st.llm_partial_failure = True
            for rem in batches[batch_idx:]:
                st.llm_unprocessed_occ_ids.update(rem.occ_id_set)
            break

        data = json.loads(raw_json)
        for item in data.get("items", []):
            st.llm_triage_results[item["occ_id"]] = item

    # Batch 1 was processed and saved
    assert "occ-1" in st.llm_triage_results
    # Batches 2 and 3 were marked unprocessed
    assert "occ-2" not in st.llm_triage_results
    assert "occ-3" not in st.llm_triage_results
    assert st.llm_partial_failure is True
    assert st.llm_unprocessed_occ_ids == {"occ-2", "occ-3"}


@pytest.mark.asyncio
async def test_reset_app_state_async_lifecycle():
    st = AppState()

    async def long_running():
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(long_running())
    st.llm_active_task = task

    mock_provider = MagicMock(spec=LocalApiProvider)
    mock_provider.close = AsyncMock()
    st.llm_provider = mock_provider

    await reset_app_state_async(st)
    assert task.done()
    assert mock_provider.close.called
    assert st.llm_provider is None
    assert st.llm_active_task is None


def test_optional_llm_extra_isolation():
    from local_anonymizer.llm.schema import TriageEnvelope
    from local_anonymizer.llm.apply_service import compute_triage_snapshot
    assert TriageEnvelope is not None
    assert compute_triage_snapshot is not None
