import json
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


def validate_model_name(name: str) -> str:
    """
    Validate that a model name is non-empty, within length bounds (1-128 chars),
    contains no control or invalid characters, and does NOT contain the forbidden ':cloud' tag.
    Returns normalized stripped model name or raises ValueError.
    """
    if not isinstance(name, str):
        raise ValueError("Modellname muss ein String sein.")
    stripped = name.strip()
    if not stripped:
        raise ValueError("Modellname darf nicht leer sein.")
    if len(stripped) > 128:
        raise ValueError("Modellname darf maximal 128 Zeichen lang sein.")
    if not re.fullmatch(r"^[a-zA-Z0-9_\.\-:\/]+$", stripped):
        raise ValueError(
            "Modellname enthält unzulässige Zeichen (nur Buchstaben, Ziffern, '_', '.', '-', ':', '/' erlaubt)."
        )
    # Defense-in-depth: Block any Cloud tags case-insensitively
    lower = stripped.lower()
    if ":cloud" in lower or lower.endswith("cloud") and ":" in lower:
        raise ValueError(
            "Cloud-Modelle (':cloud') sind aus Datenschutzgründen unzulässig. Bitte ausschliesslich lokale Modelle verwenden."
        )
    return stripped


CatalogSuitability = Literal["recommended", "suitable", "limited", "not_recommended", "untested"]
DiscoveryStatus = Literal["success", "empty", "unreachable", "timeout", "invalid_response", "unsupported"]
SetupState = Literal["idle", "discovering", "preloading", "testing"]


class CatalogPhaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: CatalogSuitability = Field(..., description="Eignung für diese Phase")
    reason: str = Field(..., min_length=1, max_length=500, description="Evidenzbasierte Begründung")


class CatalogModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    canonical_name: str = Field(..., min_length=1, max_length=128, description="Stabiler kanonischer Modellname")
    aliases: List[str] = Field(default_factory=list, description="Verifizierte Tags / Aliasse")
    tested_tag: str = Field(..., min_length=1, max_length=128, description="Tatsächlich getesteter Tag")
    provider: str = Field(..., min_length=1, max_length=64, description="Getesteter Provider (z. B. Ollama)")
    hardware_class: str = Field(..., min_length=1, max_length=128, description="Dokumentierte Referenzhardware")
    test_date: str = Field(..., min_length=10, max_length=10, description="Testdatum YYYY-MM-DD")
    phase_6a_triage: CatalogPhaseEvaluation = Field(..., description="Bewertung Phase 6A")
    phase_6b_smart_linking: CatalogPhaseEvaluation = Field(..., description="Bewertung Phase 6B")

    @field_validator("canonical_name", "tested_tag")
    @classmethod
    def check_valid_model_name(cls, v: str) -> str:
        return validate_model_name(v)

    @field_validator("aliases")
    @classmethod
    def check_aliases(cls, v: List[str]) -> List[str]:
        return [validate_model_name(a) for a in v]

    @field_validator("test_date")
    @classmethod
    def check_test_date(cls, v: str) -> str:
        if not re.fullmatch(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"Ungültiges Datumsformat '{v}': Erwartet wird YYYY-MM-DD.")
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Ungültiges Kalenderdatum '{v}': {e}") from e
        return v


class CatalogSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0.0"] = Field("1.0.0", description="Strikte Katalog-Schemaversion")
    models: List[CatalogModelEntry] = Field(..., description="Liste geprüfter Modelle")

    @field_validator("models")
    @classmethod
    def check_unique_models_and_tags(cls, models: List[CatalogModelEntry]) -> List[CatalogModelEntry]:
        seen_names: Set[str] = set()
        seen_tags: Set[str] = set()

        for m in models:
            c_name = m.canonical_name.lower()
            if c_name in seen_names:
                raise ValueError(f"Doppelter kanonischer Modellname im Katalog: '{m.canonical_name}'")
            seen_names.add(c_name)

            all_tags = [m.tested_tag.lower()] + [a.lower() for a in m.aliases]
            for tag in all_tags:
                if tag in seen_tags:
                    raise ValueError(f"Kollidierender Tag/Alias im Katalog: '{tag}' bei Modell '{m.canonical_name}'")
                seen_tags.add(tag)

        return models


class DiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: DiscoveryStatus = Field(..., description="Ergebnis der Discovery")
    models: List[str] = Field(default_factory=list, description="Gefundene lokale Modellnamen")
    message: str = Field("", max_length=500, description="Statusmeldung / Fehlerhinweis")


class PsModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=128)
    model: str = Field(..., min_length=1, max_length=128)
    size: Optional[int] = Field(None, ge=0)
    size_vram: Optional[int] = Field(None, ge=0)
    expires_at: Optional[str] = Field(None, max_length=64)


TriageAction = Literal["keep", "recategorize", "discard"]
TriageConfidence = Literal["high", "medium", "low"]


class BaseTriageItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    occ_id: str = Field(..., min_length=1, max_length=64, description="Stable occurrence identifier")
    reasoning: Optional[str] = Field(None, max_length=500, description="Brief justification from LLM")
    confidence: TriageConfidence = Field(..., description="Model confidence level")


class TriageKeepItem(BaseTriageItem):
    action: Literal["keep"] = Field("keep", description="Confirm existing entity classification")
    new_entity_type: Optional[Literal[None]] = Field(None, description="Must be null if present")
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
    new_entity_type: Optional[Literal[None]] = Field(None, description="Must be null if present")
    descriptor_suggestion: Optional[Literal[None]] = Field(None, description="Must be null if present")


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


def extract_json_from_llm_response(text: str) -> str:
    """
    Robustly extract JSON payload from raw LLM output, stripping reasoning tags (<think>...</think>),
    markdown code blocks (```json ... ```), and extraneous leading/trailing text.
    """
    if not text:
        return ""
    # Strip <think>...</think> tags from reasoning models (e.g. DeepSeek R1)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Match markdown code block
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Extract outermost JSON object { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1].strip()

    return cleaned


def validate_batch_response(
    envelope: TriageEnvelope,
    expected_occ_ids: Set[str],
    expected_doc_rev: int,
    expected_doc_hash: str,
    expected_request_id: str,
    strict_count: bool = True,
) -> None:
    """
    Perform verification of a batch response envelope against expectations.
    Raises ValueError with a descriptive message if any check fails.
    """
    if envelope.schema_version != "1.0":
        raise ValueError(f"Schema version mismatch: expected '1.0', got '{envelope.schema_version}'")

    if envelope.request_id != expected_request_id:
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

    foreign = received_set - expected_occ_ids
    if foreign:
        raise ValueError(f"Invalid batch: received {len(foreign)} unexpected occurrence IDs")

    if strict_count:
        missing = expected_occ_ids - received_set
        if missing:
            raise ValueError(f"Incomplete batch: missing {len(missing)} expected occurrence IDs")
