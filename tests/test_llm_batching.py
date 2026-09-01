import json
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


def test_build_candidate_prompt_adversarial_escaping():
    candidates = [
        {
            "occ_id": 'occ-"123"\\evil',
            "original_text": 'Dr. "Evil" \n\r\t \x00',
            "entity_type": "PERSON",
            "role": 'Chefarzt "Klinik"',
            "context_snippet": 'Befund von "Dr. Evil", status: {"injected": true}.',
        }
    ]
    prompt = build_candidate_prompt(candidates, "req-1", 2, "hash_123")
    assert "UNTRUSTED DOCUMENT PAYLOAD" in prompt
    assert "END OF UNTRUSTED PAYLOAD" in prompt
    # Check that candidate fields are escaped properly via json.dumps
    assert json.dumps('occ-"123"\\evil') in prompt
    assert json.dumps('Dr. "Evil" \n\r\t \x00') in prompt
    assert json.dumps('Chefarzt "Klinik"') in prompt

    # Extract JSON example from prompt and verify that it is 100% valid JSON
    json_start = prompt.find("{\n  \"schema_version\"")
    if json_start != -1:
        json_str = prompt[json_start:]
        parsed_example = json.loads(json_str)
        assert parsed_example["schema_version"] == "1.0"
        assert parsed_example["request_id"] == "req-1"


def test_prepare_triage_batches_oversized_candidate_truncation():
    # An extraordinarily large context snippet exceeding batch budget
    huge_context = "A" * 20000
    candidates = [
        {
            "occ_id": "occ-huge",
            "original_text": "Huge Candidate",
            "entity_type": "PERSON",
            "role": "",
            "context_snippet": huge_context,
        }
    ]
    batches = prepare_triage_batches(
        candidates,
        document_revision=1,
        document_hash="snap_huge",
        max_tokens_per_batch=1000,
    )
    assert len(batches) == 1
    # Check that context was bounded / truncated with '...'
    truncated_snippet = batches[0].candidates[0]["context_snippet"]
    assert len(truncated_snippet) < len(huge_context)
    assert truncated_snippet.endswith("...")


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
    batches = prepare_triage_batches(
        candidates,
        document_revision=3,
        document_hash="snap_multi",
        max_tokens_per_batch=1200,
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


def test_prepare_triage_batches_min_budget_value_error():
    candidates = [
        {
            "occ_id": "occ-1",
            "original_text": "Alice",
            "entity_type": "PERSON",
            "role": "CEO",
            "context_snippet": "Alice is the CEO of Acme Corp.",
        }
    ]
    with pytest.raises(ValueError) as excinfo:
        prepare_triage_batches(
            candidates,
            document_revision=1,
            document_hash="snap_val_err",
            max_tokens_per_batch=100,  # Far below baseline
        )
    assert "kleiner als das minimale Basisbudget" in str(excinfo.value)


def test_prepare_triage_batches_hard_token_budget_invariant_long_fields():
    # Test candidate with long text, long role, long occ_id, and long context snippet
    long_name = "Prof. Dr. med. Maximilian-Alexander von Hohenzollern-Sigmaringen der Dritte " * 10
    long_role = "Leitender Oberarzt für Experimentelle Nuklearmedizin und Molekulare Bildgebung " * 10
    long_ctx = "Der Patient wurde von " + long_name + " in der Klinik untersucht. " * 30

    candidates = [
        {
            "occ_id": f"occ-super-long-identifier-with-extra-suffix-{i}",
            "original_text": long_name,
            "entity_type": "PERSON",
            "role": long_role,
            "context_snippet": long_ctx,
        }
        for i in range(10)
    ]

    limit = 1000
    batches = prepare_triage_batches(
        candidates,
        document_revision=2,
        document_hash="snap_hard_invariant",
        max_tokens_per_batch=limit,
    )

    assert len(batches) > 0
    all_occ_ids = set()
    sys_tokens = estimate_tokens(SYSTEM_TRIAGE_PROMPT)

    for b in batches:
        total_tokens = sys_tokens + estimate_tokens(b.user_prompt)
        assert total_tokens <= limit, f"Batch {b.batch_index} exceeded limit: {total_tokens} > {limit}"
        all_occ_ids.update(b.occ_id_set)

    # 100% of candidates must be preserved
    assert all_occ_ids == {f"occ-super-long-identifier-with-extra-suffix-{i}" for i in range(10)}
    assert sum(len(b.candidates) for b in batches) == 10
