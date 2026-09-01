"""Pydantic V2 schema definitions for LLM Triage Layer (Phase 6A)."""

from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Union
from pydantic import BaseModel, ConfigDict, Field


TriageAction = Literal["keep", "recategorize", "discard"]
TriageConfidence = Literal["high", "medium", "low"]


class BaseTriageItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    occ_id: str = Field(..., min_length=1, max_length=64, description="Stable occurrence identifier")
    reasoning: Optional[str] = Field(None, max_length=500, description="Brief justification from LLM")
    confidence: TriageConfidence = Field(..., description="Model confidence level")


class TriageKeepItem(BaseTriageItem):
    action: Literal["keep"] = Field("keep", description="Confirm existing entity classification")
    descriptor_suggestion: Optional[str] = Field(
        None,
        max_length=100,
        description="Suggested role / descriptor (e.g., 'Kläger', 'Projektleiter')",
    )


class TriageRecategorizeItem(BaseTriageItem):
    action: Literal["recategorize"] = Field("recategorize", description="Change entity category")
    new_entity_type: str = Field(..., min_length=1, max_length=64, description="Corrected entity type (e.g. PERSON, ORGANIZATION)")
    descriptor_suggestion: Optional[str] = Field(
        None,
        max_length=100,
        description="Suggested role / descriptor",
    )


class TriageDiscardItem(BaseTriageItem):
    action: Literal["discard"] = Field("discard", description="Mark occurrence as false positive (ignore)")


TriageItem = Annotated[
    Union[TriageKeepItem, TriageRecategorizeItem, TriageDiscardItem],
    Field(discriminator="action"),
]


class TriageEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = Field(..., description="Strict schema version requirement")
    request_id: str = Field(..., min_length=1, max_length=64, description="Batch request tracking ID")
    document_revision: int = Field(..., ge=0, description="Document revision at request time")
    document_hash: str = Field(..., min_length=1, max_length=128, description="Snapshot hash binding")
    items: List[TriageItem] = Field(..., description="List of triage verdicts for this batch")


def validate_batch_response(
    envelope: TriageEnvelope,
    expected_occ_ids: Set[str],
    expected_doc_rev: int,
    expected_doc_hash: str,
    expected_request_id: Optional[str] = None,
) -> None:
    """
    Perform strict atomic verification of a batch response envelope against expectations.
    Raises ValueError with a descriptive message if any check fails.
    """
    if envelope.schema_version != "1.0":
        raise ValueError(f"Schema version mismatch: expected '1.0', got '{envelope.schema_version}'")

    if expected_request_id and envelope.request_id != expected_request_id:
        raise ValueError(f"Request ID mismatch: expected '{expected_request_id}', got '{envelope.request_id}'")

    if envelope.document_revision != expected_doc_rev:
        raise ValueError(
            f"Document revision mismatch: expected {expected_doc_rev}, got {envelope.document_revision}"
        )

    if envelope.document_hash != expected_doc_hash:
        raise ValueError("Snapshot hash mismatch: response is bound to a stale document state")

    received_ids = [item.occ_id for item in envelope.items]
    received_set = set(received_ids)

    if len(received_ids) != len(received_set):
        raise ValueError("Duplicate occurrence IDs detected in LLM response")

    missing = expected_occ_ids - received_set
    if missing:
        raise ValueError(f"Incomplete batch: missing {len(missing)} expected occurrence IDs")

    foreign = received_set - expected_occ_ids
    if foreign:
        raise ValueError(f"Invalid batch: received {len(foreign)} unexpected occurrence IDs")
