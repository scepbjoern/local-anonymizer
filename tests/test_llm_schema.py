import pytest
from pydantic import ValidationError
from local_anonymizer.llm.schema import (
    TriageKeepItem,
    TriageRecategorizeItem,
    TriageDiscardItem,
    TriageEnvelope,
    validate_batch_response,
)


def test_triage_keep_item_valid():
    item = TriageKeepItem(
        occ_id="occ-1",
        confidence="high",
        reasoning="Correct person name in signature.",
        descriptor_suggestion="Chef",
    )
    assert item.action == "keep"
    assert item.occ_id == "occ-1"
    assert item.confidence == "high"
    assert item.descriptor_suggestion == "Chef"


def test_triage_recategorize_item_valid():
    item = TriageRecategorizeItem(
        occ_id="occ-2",
        new_entity_type="ORG",
        confidence="medium",
        reasoning="This is a company not a person.",
        descriptor_suggestion="Arbeitgeber",
    )
    assert item.action == "recategorize"
    assert item.new_entity_type == "ORG"
    assert item.descriptor_suggestion == "Arbeitgeber"


def test_triage_discard_item_valid():
    item = TriageDiscardItem(
        occ_id="occ-3",
        confidence="low",
        reasoning="Common German verb, not a named entity.",
    )
    assert item.action == "discard"
    assert item.occ_id == "occ-3"


def test_schema_version_strict_binding():
    # schema_version must be exactly "1.0"
    with pytest.raises(ValidationError):
        TriageEnvelope(
            schema_version="2.0",
            request_id="req-1",
            document_revision=1,
            document_hash="abc",
            items=[],
        )


def test_missing_confidence_rejected():
    # confidence is required, cannot be omitted
    with pytest.raises(ValidationError):
        TriageKeepItem.model_validate({"occ_id": "occ-1", "action": "keep"})


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        TriageKeepItem(
            occ_id="occ-1",
            confidence="high",
            unknown_property="unexpected",  # forbidden
        )


def test_invalid_confidence_rejected():
    with pytest.raises(ValidationError):
        TriageDiscardItem(
            occ_id="occ-1",
            confidence="very_high",  # invalid literal
        )


def test_envelope_deserialization_and_polymorphism():
    raw_json = """{
        "schema_version": "1.0",
        "request_id": "req-123",
        "document_revision": 3,
        "document_hash": "sha256_mock_hash",
        "items": [
            {
                "occ_id": "id-1",
                "action": "keep",
                "confidence": "high",
                "reasoning": "Valid context"
            },
            {
                "occ_id": "id-2",
                "action": "recategorize",
                "new_entity_type": "LOCATION",
                "confidence": "medium",
                "reasoning": "City name"
            },
            {
                "occ_id": "id-3",
                "action": "discard",
                "confidence": "low",
                "reasoning": "Dictionary word"
            }
        ]
    }"""
    envelope = TriageEnvelope.model_validate_json(raw_json)
    assert envelope.schema_version == "1.0"
    assert envelope.request_id == "req-123"
    assert envelope.document_revision == 3
    assert envelope.document_hash == "sha256_mock_hash"
    assert len(envelope.items) == 3
    assert isinstance(envelope.items[0], TriageKeepItem)
    assert isinstance(envelope.items[1], TriageRecategorizeItem)
    assert isinstance(envelope.items[2], TriageDiscardItem)


def test_validate_batch_response_success():
    expected_ids = {"id-1", "id-2"}
    envelope = TriageEnvelope(
        schema_version="1.0",
        request_id="req-1",
        document_revision=2,
        document_hash="snap_123",
        items=[
            TriageKeepItem(occ_id="id-1", confidence="high"),
            TriageDiscardItem(occ_id="id-2", confidence="medium"),
        ],
    )
    validate_batch_response(envelope, expected_ids, 2, "snap_123", expected_request_id="req-1")


def test_validate_batch_response_request_id_mismatch():
    expected_ids = {"id-1"}
    envelope = TriageEnvelope(
        schema_version="1.0",
        request_id="req-wrong",
        document_revision=2,
        document_hash="snap_123",
        items=[TriageKeepItem(occ_id="id-1", confidence="high")],
    )
    with pytest.raises(ValueError, match="Request ID mismatch"):
        validate_batch_response(envelope, expected_ids, 2, "snap_123", expected_request_id="req-correct")


def test_validate_batch_response_missing_id():
    expected_ids = {"id-1", "id-2"}
    envelope = TriageEnvelope(
        schema_version="1.0",
        request_id="req-1",
        document_revision=2,
        document_hash="snap_123",
        items=[
            TriageKeepItem(occ_id="id-1", confidence="high"),
        ],
    )
    with pytest.raises(ValueError, match="missing"):
        validate_batch_response(envelope, expected_ids, 2, "snap_123")


def test_validate_batch_response_hallucinated_id():
    expected_ids = {"id-1"}
    envelope = TriageEnvelope(
        schema_version="1.0",
        request_id="req-1",
        document_revision=2,
        document_hash="snap_123",
        items=[
            TriageKeepItem(occ_id="id-1", confidence="high"),
            TriageKeepItem(occ_id="hallucinated-id", confidence="low"),
        ],
    )
    with pytest.raises(ValueError, match="unexpected"):
        validate_batch_response(envelope, expected_ids, 2, "snap_123")


def test_validate_batch_response_duplicate_id():
    expected_ids = {"id-1"}
    envelope = TriageEnvelope(
        schema_version="1.0",
        request_id="req-1",
        document_revision=2,
        document_hash="snap_123",
        items=[
            TriageKeepItem(occ_id="id-1", confidence="high"),
            TriageDiscardItem(occ_id="id-1", confidence="low"),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_batch_response(envelope, expected_ids, 2, "snap_123")


def test_validate_batch_response_revision_mismatch():
    expected_ids = {"id-1"}
    envelope = TriageEnvelope(
        schema_version="1.0",
        request_id="req-1",
        document_revision=1,  # expected 2
        document_hash="snap_123",
        items=[TriageKeepItem(occ_id="id-1", confidence="high")],
    )
    with pytest.raises(ValueError, match="Document revision mismatch"):
        validate_batch_response(envelope, expected_ids, 2, "snap_123")


def test_validate_batch_response_snapshot_mismatch():
    expected_ids = {"id-1"}
    envelope = TriageEnvelope(
        schema_version="1.0",
        request_id="req-1",
        document_revision=2,
        document_hash="snap_old",  # expected snap_new
        items=[TriageKeepItem(occ_id="id-1", confidence="high")],
    )
    with pytest.raises(ValueError, match="Snapshot hash mismatch"):
        validate_batch_response(envelope, expected_ids, 2, "snap_new")
