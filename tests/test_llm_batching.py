import pytest
from local_anonymizer.llm.batching import (
    estimate_tokens,
    build_candidate_prompt,
    prepare_triage_batches,
    SYSTEM_TRIAGE_PROMPT,
)


def test_estimate_tokens():
    text = "Hello world! This is a test."
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert estimate_tokens("") == 0


def test_build_candidate_prompt():
    candidates = [
        {
            "occ_id": "occ-123",
            "original_text": "Dr. Meier",
            "entity_type": "PERSON",
            "role": "Chefarzt",
            "context_snippet": "Befund von Dr. Meier erstellt.",
        }
    ]
    prompt = build_candidate_prompt(candidates, "req-1", 2, "hash_123")
    assert 'occ_id: "occ-123"' in prompt
    assert 'Erkannter Begriff: "Dr. Meier"' in prompt
    assert "Aktueller Typ: PERSON" in prompt
    assert "Aktuelle Rolle: 'Chefarzt'" in prompt
    assert 'Kontext: "Befund von Dr. Meier erstellt."' in prompt
    assert '"document_revision": 2' in prompt
    assert '"document_hash": "hash_123"' in prompt


def test_system_prompt_contains_contract():
    assert "1.0" in SYSTEM_TRIAGE_PROMPT
    assert "schema_version" in SYSTEM_TRIAGE_PROMPT
    assert "Triage" in SYSTEM_TRIAGE_PROMPT


def test_prepare_triage_batches_single_batch():
    candidates = [
        {
            "occ_id": f"occ-{i}",
            "original_text": f"Name_{i}",
            "entity_type": "PERSON",
            "role": "",
            "context_snippet": f"Context for occurrence {i}.",
        }
        for i in range(5)
    ]
    batches = prepare_triage_batches(candidates, document_revision=1, document_hash="snap_abc")
    assert len(batches) == 1
    assert batches[0].batch_index == 1
    assert batches[0].total_batches == 1
    assert len(batches[0].candidates) == 5
    assert len(batches[0].occ_id_set) == 5
    assert batches[0].document_hash == "snap_abc"
    assert batches[0].document_revision == 1


def test_prepare_triage_batches_multiple_batches():
    candidates = [
        {
            "occ_id": f"occ-{i}",
            "original_text": f"Name_{i} von der Abteilung {i}",
            "entity_type": "PERSON",
            "role": "Mitarbeiter",
            "context_snippet": f"Hier steht ein langer Kontexttext für die Fundstelle Nummer {i} im Dokument.",
        }
        for i in range(50)
    ]
    # Set a small max_items_per_batch or max_tokens_per_batch to force slicing into multiple batches
    batches = prepare_triage_batches(
        candidates,
        document_revision=3,
        document_hash="snap_multi",
        max_tokens_per_batch=400,
        max_items_per_batch=10,
    )
    assert len(batches) > 1
    all_occ_ids = set()
    for idx, batch in enumerate(batches, start=1):
        assert batch.batch_index == idx
        assert batch.total_batches == len(batches)
        assert batch.document_revision == 3
        assert batch.document_hash == "snap_multi"
        all_occ_ids.update(batch.occ_id_set)

    # Every candidate must be present in exactly one batch
    assert all_occ_ids == {f"occ-{i}" for i in range(50)}
    assert sum(len(b.candidates) for b in batches) == 50


def test_prepare_triage_batches_empty():
    batches = prepare_triage_batches([], document_revision=1, document_hash="snap_empty")
    assert len(batches) == 0
